"""Local Viser workbench: live MuJoCo, Marble splats and bounded language commands."""
import argparse
import json
import math
import queue
import time
import traceback
import webbrowser

import mujoco
import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation
from .scene import ROOT, build_scene, available_environments
from .control import Controller, solve_arm_ik
from .motion import SomaMotion
from .objects import OBJECTS
from .commands import parse_command
from .navigation import Planner, Walker
from .splats import load_world_splats
from .labels import text_card
from .object_tasks import ObjectTaskRunner
from .realsense import RealSensePanel, DEFAULT_SOCKET


class Workbench:
    def __init__(self, port=8765, environment='warm_living_room', realsense_socket=DEFAULT_SOCKET):
        self.server=viser.ViserServer(host='127.0.0.1',port=port,label='Robot World')
        self.server.scene.set_up_direction('+z')
        self.server.gui.configure_theme(control_layout='floating',control_width='medium',dark_mode=True,
            show_logo=False,show_share_button=False,brand_color=(32,185,165))
        self.commands=queue.Queue()
        self.handles=[];self.geometry=[];self.labels=[];self.axis_labels=[];self.path_handle=None
        self.followup=None;self.last_reply='Ready. Give the robot a command.'
        self.paused=False
        self.grasp_result_notified=False
        gui=self.server.gui
        gui.add_markdown('## ROBOT WORLD\nG1 · Five-finger hands · Your Marble scene')
        options=available_environments()
        if environment not in options.values():
            raise ValueError('Scene assets are missing. Run scripts/bootstrap_assets.py or use --environment lab.')
        self.environments=options
        self.environment=gui.add_dropdown('3D environment',tuple(options),initial_value=next(k for k,v in options.items() if v==environment))
        self.environment.on_update(lambda _:self.commands.put(('environment',options[self.environment.value])))
        with gui.add_form(None) as prompt_form:
            self.prompt=gui.add_text('Ask the robot',initial_value='grab the red mug',multiline=False,
                                     hint='Press Enter or click Run command.')
            run_button=gui.add_button('Run command',color='teal')
        async def submit_prompt(_):
            # Snapshot in websocket event order, before a later edit can arrive.
            self.commands.put(('prompt',self.prompt.value))
        run_button.on_click(submit_prompt)
        prompt_form.on_submit(submit_prompt)
        self.grasp_button=gui.add_button('Grasp blue bottle',color='teal')
        self.grasp_button.on_click(lambda _:self.commands.put(('prompt','pick up the blue bottle')))
        self.reply=gui.add_markdown(self.last_reply)
        self.realsense=RealSensePanel(self.server, realsense_socket)
        self.alignment_info=gui.add_markdown('',visible=False)
        with gui.add_folder('Try a command',expand_by_default=False):
            gui.add_markdown('`walk backward 1 meter`\n\n`walk to x 1.5 y 0.8`\n\n`go to the red mug`\n\n`reach for the blue bottle`\n\n`pick up the blue bottle`\n\n`grab the green apple`\n\n`open hands` · `turn left` · `stop`\n\nLocal commands understand these actions and object names.')
            for title,command in [('Walk around table','walk around the table'),('Reach for bottle','reach for the blue bottle'),('Open hands','open hands')]:
                gui.add_button(title).on_click(lambda _,c=command:self.commands.put(('prompt',c)))
        with gui.add_folder('View & control',expand_by_default=True):
            gui.add_button('Stop robot',color='orange').on_click(lambda _:self.commands.put(('prompt','stop')))
            gui.add_button('Reset scene').on_click(lambda _:self.commands.put(('prompt','reset')))
            gui.add_button('Focus robot').on_click(lambda _:self.commands.put(('camera','robot')))
            gui.add_button('Table close-up').on_click(lambda _:self.commands.put(('camera','table')))
            gui.add_button('Room overview').on_click(lambda _:self.commands.put(('camera','room')))
            self.show_labels=gui.add_checkbox('Object labels',True)
            self.show_axes=gui.add_checkbox('XYZ axes',True)
            self.show_room=gui.add_checkbox('World Labs room',True)
            self.show_grid=gui.add_checkbox('Floor grid',True)
            self.speed=gui.add_slider('Walking speed (m/s)',min=.1,max=.5,step=.05,initial_value=.3)
        self.telemetry=gui.add_markdown('Loading physics…')
        gui.add_markdown('**Prototype mode**\n\nWalking uses base support. Room collisions are approximate. Bottle, mug, and apple grasps are tested locally. Other object grasps remain experimental.\n\nDrag to orbit · right-drag to pan · scroll to zoom.')
        self.load(environment)
        @self.server.on_client_connect
        def on_connect(client):
            self.camera('robot',client)
            @client.camera.on_update
            def orient_labels(camera):
                for _,label in self.labels+self.axis_labels:
                    label.wxyz=camera.wxyz

    def add_handle(self,handle):
        self.handles.append(handle)
        return handle

    def camera(self,mode,client=None):
        clients=[client] if client else list(self.server.get_clients().values())
        p=self.data.qpos[:3]
        if mode=='table': position=(1.6,-1.65,1.7);target=(.55,-.1,.74)
        elif mode=='room': position=(2.,-3.,2.8);target=(0,0,1.)
        else: position=p+np.array([1.65,-2.1,1.05]);target=p+np.array([.3,0,-.02])
        for c in clients:
            c.camera.position=position;c.camera.look_at=target;c.camera.up_direction=(0,0,1)

    def load(self,environment):
        for handle in self.handles: handle.remove()
        if self.path_handle: self.path_handle.remove();self.path_handle=None
        self.handles=[];self.geometry=[];self.labels=[];self.axis_labels=[];self.followup=None
        self.model,self.data,self.config,_=build_scene(environment)
        self.grasp_button.disabled=bool(self.config.get('camera_layout'))
        self.prompt.value='pick up the Observed black mug' if self.config.get('camera_layout') else 'grab the red mug'
        self.alignment_info.visible=bool(self.config.get('camera_layout'))
        if self.config.get('camera_layout'):
            self.alignment_info.content='**Paper-aligned snapshot · approximate**\n\n210 × 147 mm paper anchored to the table corner. Black mug position mapped from its base; mug size is illustrative. Flat card and cable are visual proxies. Cropped items omitted.\n\nObjects are not tracked live. Camera movement invalidates this alignment. You can pick up the simulated mug using its displayed name. This does not control a physical robot.'
        self.control=Controller(self.model,self.data)
        self.planner=Planner(self.config)
        gait=SomaMotion(ROOT/'vendor/soma-retargeter/assets/motions/csv/Neutral_walk_forward_002__A057.csv',self.model,self.data.qpos)
        self.walker=Walker(self.control,gait)
        self.object_tasks=ObjectTaskRunner(self.control,self.walker,self.planner,self.start_path,self.say,self.config.get('object_labels'))
        self.room=None
        if self.config.get('world_manifest'):
            manifest=json.loads((ROOT/self.config['world_manifest']).read_text())
            if manifest.get('splat_file'):
                self.room=self.add_handle(self.server.scene.add_gaussian_splats('/room',**load_world_splats(manifest,ROOT)))
        self.draw_geometry()
        self.axes=self.add_handle(self.server.scene.add_frame('/axes',axes_length=1.,axes_radius=.012,position=(-.55,-.65,.015)))
        for name,pos in [('X · 1 m',(.48,-.65,.045)),('Y · 1 m',(-.55,.4,.045)),('Z · 1 m',(-.55,-.65,1.07))]:
            handle=self.make_label('/axis_labels/'+name,name,pos,height=.07)
            self.axis_labels.append((name,handle))
        self.robot_axes=self.add_handle(self.server.scene.add_frame('/robot_axes',axes_length=.3,axes_radius=.006))
        self.grid=self.add_handle(self.server.scene.add_grid('/grid',width=12,height=12,cell_size=.5,section_size=1,
            cell_color=(135,148,151),section_color=(186,200,200),plane_opacity=0,position=(0,0,.015)))
        for key,info in OBJECTS.items():
            if mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,key)<0: continue
            title=self.config.get('object_labels',{}).get(key,info['label'])
            handle=self.make_label('/labels/'+key,title,self.data.body(key).xpos,height=.085)
            self.labels.append((key,handle))
        self.leaders=self.add_handle(self.server.scene.add_line_segments('/label_leaders',points=np.zeros((10,2,3),dtype=np.float32),colors=(130,213,199),line_width=1))
        self.paused=False
        self.say('Ready in '+self.config['name']+('. Approximate camera snapshot loaded; black mug, paper, card and visible cable.' if self.config.get('camera_layout') else '. Ten movable objects are on the table.'))
        self.camera('robot')
        for client in self.server.get_clients().values():
            for _,label in self.labels+self.axis_labels: label.wxyz=client.camera.wxyz
        self.sync()

    def make_label(self,name,text,position,height):
        pixels,aspect=text_card(text)
        return self.add_handle(self.server.scene.add_image(name,pixels,render_width=height*aspect,render_height=height,
            position=position,cast_shadow=False,receive_shadow=False))

    def draw_geometry(self):
        m=self.model
        for gid in range(m.ngeom):
            kind=int(m.geom_type[gid]);group=int(m.geom_group[gid]);rgba=m.geom_rgba[gid]
            if kind==int(mujoco.mjtGeom.mjGEOM_PLANE) or group==3 or rgba[3]<.01: continue
            # Mesh-only environments are also supported for existing imported rooms.
            name='/physics/'+str(gid)
            size=m.geom_size[gid]
            if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
                mid=m.geom_dataid[gid];va=m.mesh_vertadr[mid];vn=m.mesh_vertnum[mid];fa=m.mesh_faceadr[mid];fn=m.mesh_facenum[mid]
                vertices=m.mesh_vert[va:va+vn];faces=m.mesh_face[fa:fa+fn]
            else:
                if kind==int(mujoco.mjtGeom.mjGEOM_BOX): mesh=trimesh.creation.box(extents=2*size)
                elif kind==int(mujoco.mjtGeom.mjGEOM_SPHERE): mesh=trimesh.creation.icosphere(subdivisions=2,radius=size[0])
                elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER): mesh=trimesh.creation.cylinder(radius=size[0],height=2*size[1],sections=20)
                elif kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE): mesh=trimesh.creation.capsule(radius=size[0],height=2*size[1],count=[8,12])
                elif kind==int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                    mesh=trimesh.creation.icosphere(subdivisions=2);mesh.vertices*=size
                else: continue
                vertices,faces=mesh.vertices,mesh.faces
            matid=m.geom_matid[gid]
            if matid>=0: rgba=m.mat_rgba[matid]
            handle=self.add_handle(self.server.scene.add_mesh_simple(name,np.asarray(vertices,dtype=np.float32),np.asarray(faces,dtype=np.uint32),
              color=tuple((rgba[:3]*255).astype(int)),opacity=float(rgba[3]),side='double'))
            self.geometry.append((gid,handle))
        self.update_geometry()

    def update_geometry(self):
        quat=np.zeros(4)
        for gid,handle in self.geometry:
            handle.position=self.data.geom_xpos[gid]
            mujoco.mju_mat2Quat(quat,self.data.geom_xmat[gid]);handle.wxyz=quat.copy()

    def sync(self):
        with self.server.atomic():
            self.update_geometry()
            for gid,handle in self.geometry:
                if self.model.geom_group[gid]==1: handle.visible=self.show_room.value
            clients=list(self.server.get_clients().values())
            camera=clients[0].camera if clients else None
            occupied=[];leaders=[]
            if camera:
                camera_position=np.asarray(camera.position)
                r=Rotation.from_quat(np.asarray(camera.wxyz)[[1,2,3,0]]).as_matrix()
            for i,(key,label) in enumerate(self.labels):
                obj=self.data.body(key).xpos.copy()
                position=obj+np.array([0,0,.16])
                if camera:
                    local=r.T@(position-camera_position)
                    depth=max(.1,local[2])
                    x,y=local[:2]/depth
                    width=label.render_width/depth+.008
                    height=label.render_height/depth+.008
                    # Stack intersecting callouts upward in camera space.
                    for _ in range(30):
                        overlap=[b for b in occupied if abs(x-b[0])<(width+b[2])/2 and abs(y-b[1])<(height+b[3])/2]
                        if not overlap: break
                        y=min(b[1]-(height+b[3])/2-.003 for b in overlap)
                    occupied.append((x,y,width,height))
                    position=camera_position+r@np.array([x*depth,y*depth,depth])
                    label.wxyz=camera.wxyz
                label.position=position;label.visible=self.show_labels.value
                leaders.append([obj+np.array([0,0,.05]),position])
            self.leaders.points=np.asarray(leaders,dtype=np.float32)
            self.leaders.visible=self.show_labels.value
            self.axes.visible=self.show_axes.value;self.robot_axes.visible=self.show_axes.value
            for _,label in self.axis_labels: label.visible=self.show_axes.value
            self.robot_axes.position=self.data.qpos[:3];self.robot_axes.wxyz=self.data.qpos[3:7]
            self.grid.visible=self.show_grid.value
            if self.room: self.room.visible=self.show_room.value
        p=self.data.qpos[:3]
        self.telemetry.content=f'**{self.control.status}**\n\nPosition: X {p[0]:+.2f} · Y {p[1]:+.2f} · Z {p[2]:.2f} m\n\nHeading: {math.degrees(self.walker.yaw)%360:.0f}° · Simulation: {self.data.time:.1f} s'
        if self.control.grasp_monitor:
            evidence=self.control.grasp_monitor.evidence
            self.telemetry.content+=f'\n\nGrasp: {100*evidence.get("lift_m",0):.1f} cm lift · {evidence.get("hand_contacts",0)} contacts · {evidence.get("held_seconds",0):.1f} s held'

    def object_label(self,name):
        return self.config.get('object_labels',{}).get(name,OBJECTS[name]['label'])

    def say(self,text):
        self.last_reply=text;self.reply.content=text
        print(text,flush=True)

    def start_path(self,path):
        if self.path_handle: self.path_handle.remove()
        points=np.column_stack([path,np.full(len(path),.035)])
        # Linear segments avoid a smoothed display suggesting unsafe corner cuts.
        self.path_handle=self.server.scene.add_line_segments('/planned_path',points=np.stack([points[:-1],points[1:]],axis=1),colors=(25,230,190),line_width=3)
        self.walker.begin(path)

    def execute(self,text):
        command=parse_command(text,self.config.get('object_labels'));action,values=command.action,command.values
        if action in ('pick','reach','approach'):
            if mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,values[0])<0:
                raise ValueError('That object is not present in this scene.')
        self.object_tasks.cancel()
        self.say('Received: '+text.strip())
        if action=='stop':
            self.walker.stop();self.control.motion=None;self.control.demo_start=None;self.followup=None
            self.control.target_qpos=self.data.qpos.copy();self.say('Stopped. Holding the current pose.');return
        if action=='reset':
            self.control.reset();self.walker.stop();self.walker.yaw=0;self.walker.target_yaw=0;self.followup=None
            if self.path_handle: self.path_handle.remove();self.path_handle=None
            self.say('Scene reset. Robot and table objects are back at their starting positions.');return
        if action=='hands':
            self.control.hand_closure=values[0];self.control.demo_start=None;self.control.grasp_hand_targets=None
            self.control.status='Hands open' if values[0]==0 else 'Hands closed'
            self.say('Opening both five-finger hands.' if values[0]==0 else 'Closing both five-finger hands.');return
        if action=='turn':
            self.followup=None;self.control.demo_start=None
            self.walker.stop();self.walker.target_yaw=self.walker.yaw+math.radians(values[0]);self.say(f'Turning {abs(values[0]):g} degrees.');return
        if action=='pick':
            self.followup=None
            if self.object_tasks.request_grasp(values[0]):
                self.grasp_result_notified=False
            return
        if action=='reach':
            self.followup=None
            self.walker.stop();self.control.demo_start=None
            target=self.data.body(values[0]).xpos.copy()+np.array([0,0,.06])
            pose,error=solve_arm_ik(self.model,self.data.qpos,target,iterations=200)
            if error>.045: raise ValueError(f'{self.object_label(values[0])} is out of reach from here. Try: grab the {self.object_label(values[0]).lower()}.')
            self.control.target_qpos=pose;self.control.last_ik_error=error;self.control.status='Reaching with right five-finger hand'
            self.say(f'Reaching for {self.object_label(values[0])}.')
            return
        p=self.data.mocap_pos[0,:2].copy()
        if action=='tour':
            self.followup=None
            # Go around all four sides in free space, then return to the start.
            stops=[[-.18,-.85],[1.35,-.85],[1.35,.7],[-.18,.7],p]
            path=[p]
            for goal in stops:
                segment=self.planner.plan(path[-1],goal);path.extend(segment[1:])
            self.start_path(np.asarray(path));self.say('Walking around the table along the green path.');return
        if action=='position': goal=np.array(values)
        elif action=='walk':
            direction,distance=values
            angle={'forward':0,'backward':math.pi,'backwards':math.pi,'back':math.pi,'left':math.pi/2,'right':-math.pi/2}[direction]+self.walker.yaw
            goal=p+distance*np.array([math.cos(angle),math.sin(angle)])
        elif action=='approach':
            obj=self.data.body(values[0]).xpos[:2].copy()
            # Evaluate reachable stances on each edge of the table.
            candidates=[np.asarray(goal) for offset in (-.2,-.1,0,.1,.2) for goal in
                ([0,obj[1]+offset],[1.14,obj[1]+offset],[obj[0]+offset,-.74],[obj[0]+offset,.54])]
            paths=[]
            for goal in candidates:
                try:
                    path=self.planner.plan(p,goal)
                    heading=math.atan2(obj[1]-goal[1],obj[0]-goal[0])
                    stance=self.data.qpos.copy()
                    stance[:2]=goal;stance[3:7]=[math.cos(heading/2),0,0,math.sin(heading/2)]
                    target=self.data.body(values[0]).xpos.copy()+np.array([0,0,.06])
                    _,error=solve_arm_ik(self.model,stance,target,iterations=120)
                    if error<.045:
                        paths.append((np.linalg.norm(np.diff(path,axis=0),axis=1).sum(),path))
                except ValueError: pass
            if not paths: raise ValueError('No clear stance near that object. Try resetting the scene.')
            _,path=min(paths,key=lambda v:v[0]);goal=path[-1]
            self.start_path(path);self.followup=('face',obj)
            self.say('Walking to '+self.object_label(values[0])+'.');return
        else: raise ValueError('That action is not supported yet.')
        self.followup=None
        self.start_path(self.planner.plan(p,goal));self.say(f'Walking to X {goal[0]:.2f}, Y {goal[1]:.2f} metres.')

    def run(self):
        frame=0
        while True:
            started=time.monotonic()
            while not self.commands.empty():
                kind,value=self.commands.get()
                try:
                    if kind=='environment': self.load(value)
                    elif kind=='camera': self.camera(value)
                    else: self.execute(value)
                except ValueError as error: self.say(str(error))
                except Exception:
                    traceback.print_exc();self.say('The command could not finish. Reset the scene and try again.')
            self.walker.speed=round(self.speed.value,2)
            was_walking=bool(self.walker.path)
            for _ in range(16):
                self.walker.step();self.control.step()
            if was_walking and not self.walker.path and self.object_tasks.pending is None:
                self.say('Walking complete. Ready for another command.')
            if self.followup and not self.walker.path:
                _,obj=self.followup;delta=obj-self.data.mocap_pos[0,:2]
                self.walker.target_yaw=self.walker.yaw+(math.atan2(delta[1],delta[0])-self.walker.yaw+math.pi)%(2*math.pi)-math.pi;self.followup=None
            self.object_tasks.tick()
            monitor=self.control.grasp_monitor
            if monitor is not None and self.control.demo_start is not None and not self.grasp_result_notified:
                if monitor.held_seconds>=1:
                    self.say(f'Holding {self.object_label(monitor.object_name)}: {100*monitor.evidence["lift_m"]:.1f} cm lift confirmed by finger contacts.')
                    self.grasp_result_notified=True
                elif self.control.status.startswith(('Grasp missed','Lift was not secured','Object slipped')):
                    self.say(self.control.status);self.grasp_result_notified=True
            if self.control.status.startswith(('Grasp missed','Lift was not secured','Object slipped')) and self.last_reply != self.control.status:
                self.say(self.control.status)
            frame+=1
            self.sync()
            if frame%60==0:
                (ROOT/'build/live_status.json').write_text(json.dumps({**self.control.metrics(),'reply':self.last_reply,'world':self.config['name'],'path_remaining':len(self.walker.path)},indent=2))
            time.sleep(max(0,.032-(time.monotonic()-started)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--environment',default='warm_living_room')
    parser.add_argument('--open-browser',action='store_true')
    parser.add_argument('--realsense-socket',default=DEFAULT_SOCKET,help='Socket of the existing RealSense camera helper')
    args=parser.parse_args()
    app=Workbench(args.port,args.environment,args.realsense_socket)
    if args.open_browser: webbrowser.open(f'http://127.0.0.1:{args.port}')
    try:
        app.run()
    finally:
        app.realsense.close()
        app.server.stop()

if __name__=='__main__': main()
