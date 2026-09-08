"""Conservative 2D A* and assisted SOMA locomotion for the desktop prototype."""
import heapq
import math
import json
import numpy as np
from .scene import ROOT

class Planner:
    def __init__(self, config, radius=.22, resolution=.10):
        self.radius,self.resolution=radius,resolution
        self.bounds=np.array([[-4.,-4.],[4.,4.]])
        boxes=list(config.get('furniture',[]))
        if config.get('world_manifest'):
            manifest=json.loads((ROOT/config['world_manifest']).read_text())
            boxes+=manifest.get('collision_boxes',[])
            # Stay within the reconstructed footprint's conservative bounding box.
            self.bounds=np.array(manifest['bounds'])[:,:2]
        boxes.append({'pos':'0.57 -0.1 0.7','size':'0.34 0.4 .025'})
        self.boxes=[]
        for box in boxes:
            pos=np.fromstring(box['pos'],sep=' ');size=np.fromstring(box['size'],sep=' ')
            if pos[2]-size[2]<1.5 and pos[2]+size[2]>.12:
                self.boxes.append((pos[:2],size[:2]))
        self.centers=np.array([b[0] for b in self.boxes]).reshape(-1,2)
        self.halves=np.array([b[1] for b in self.boxes]).reshape(-1,2)

    def free(self,p):
        p=np.asarray(p)
        if np.any(p<self.bounds[0]+self.radius) or np.any(p>self.bounds[1]-self.radius): return False
        delta=np.maximum(abs(self.centers-p)-self.halves,0)
        return not np.any(np.linalg.norm(delta,axis=1)<self.radius)

    def segment_free(self,a,b):
        return all(self.free(p) for p in np.linspace(a,b,max(2,int(np.linalg.norm(np.asarray(a)-b)/.04)+1)))

    def plan(self,start,goal):
        start,goal=np.asarray(start),np.asarray(goal)
        if not self.free(goal): raise ValueError('That destination is occupied or outside the room. Choose a clear floor position.')
        if not self.free(start): raise ValueError('The robot is too close to an obstacle; reset before walking.')
        # Anchor the grid at the actual start, avoiding rounding into an obstacle.
        source=(0,0)
        target=tuple(np.rint((goal-start)/self.resolution).astype(int))
        def pos(node): return start+np.array(node)*self.resolution
        queue=[(0.,source)];cost={source:0.};parent={}
        neighbors=[(i,j) for i in (-1,0,1) for j in (-1,0,1) if i or j]
        found=None
        while queue and len(cost)<60000:
            _,node=heapq.heappop(queue)
            p=pos(node)
            if np.linalg.norm(p-goal)<self.resolution*1.5 and self.segment_free(p,goal): found=node;break
            for dx,dy in neighbors:
                nxt=(node[0]+dx,node[1]+dy);q=pos(nxt)
                if not self.free(q): continue
                if dx and dy and (not self.free(p+[dx*self.resolution,0]) or not self.free(p+[0,dy*self.resolution])): continue
                value=cost[node]+math.hypot(dx,dy)
                if value>=cost.get(nxt,float('inf')): continue
                cost[nxt]=value;parent[nxt]=node
                heapq.heappush(queue,(value+np.linalg.norm(np.array(nxt)-target),nxt))
        if found is None: raise ValueError('No clear walking path was found through the room.')
        path=[goal,pos(found)]
        while found!=source:
            found=parent[found];path.append(pos(found))
        path=path[::-1]
        # Line-of-sight smoothing retains clearance along every shortcut.
        output=[path[0]];index=0
        while index<len(path)-1:
            nxt=len(path)-1
            while nxt>index+1 and not self.segment_free(path[index],path[nxt]): nxt-=1
            output.append(path[nxt]);index=nxt
        return np.array(output)

class Walker:
    def __init__(self,controller,motion):
        self.control,self.motion=controller,motion
        self.path=[];self.yaw=0.;self.target_yaw=0.;self.speed=.32
        self.leg_ids=[controller.model.joint(n).qposadr[0] for n in motion.joint_names if any(x in n for x in ('hip','knee','ankle'))]
        self.start_time=0.

    def begin(self,path):
        self.path=list(np.asarray(path)[1:])
        self.start_time=self.control.data.time
        self.control.motion=None;self.control.demo_start=None
        self.control.status='Walking · SOMA gait with base support'

    def stop(self):
        self.path=[]
        self.target_yaw=self.yaw
        self.control.target_qpos[self.leg_ids]=self.control.home_qpos[self.leg_ids]
        self.control.status='Stopped · supported standing'

    def step(self):
        c=self.control;dt=c.model.opt.timestep
        p=c.data.mocap_pos[0,:2].copy()
        if self.path:
            delta=self.path[0]-p;distance=np.linalg.norm(delta)
            if distance<.003:
                self.path.pop(0)
                if not self.path: self.stop()
            else:
                heading=math.atan2(delta[1],delta[0])
                error=(heading-self.yaw+math.pi)%(2*math.pi)-math.pi
                self.target_yaw=self.yaw+error
                # Turn before advancing, which keeps the swept body close to the path.
                if abs(error)<.3:
                    c.data.mocap_pos[0,:2]+=delta/distance*min(distance,self.speed*dt)
                gait=self.motion.sample(.8+(c.data.time-self.start_time)%.9)
                c.target_qpos[self.leg_ids]=gait[self.leg_ids]
        error=self.target_yaw-self.yaw
        self.yaw+=float(np.clip(error,-1.2*dt,1.2*dt))
        c.data.mocap_quat[0]=[math.cos(self.yaw/2),0,0,math.sin(self.yaw/2)]
