"""One request carries an object action through approach, alignment and grasp."""
from dataclasses import dataclass
import math
import mujoco
import numpy as np
from .control import solve_arm_ik
from .grasp import SideGraspPlan
from .objects import OBJECTS


@dataclass
class PendingObjectTask:
    object_name: str
    yaw: float
    stance_offset: np.ndarray
    phase: str = 'walking'
    aligned_since: float | None = None
    phase_started: float = 0.
    adjustments: int = 0


def grasp_stance_candidates(object_name,position):
    """Prefer task-tested stances, then consider other table edges."""
    if object_name in ('mug','apple'):
        yield position+[-.06,-.45],math.radians(110)
    else:
        yield position+[-.28,.27],0.
    for offset in (0,-.1,.1,-.2,.2):
        for goal in ([0,position[1]+offset],[1.14,position[1]+offset],
                     [position[0]+offset,-.74],[position[0]+offset,.54]):
            goal=np.asarray(goal)
            heading=math.atan2(position[1]-goal[1],position[0]-goal[0])
            yield goal,heading


class ObjectTaskRunner:
    def __init__(self,control,walker,planner,start_path,notify):
        self.control,self.walker,self.planner=control,walker,planner
        self.start_path,self.notify=start_path,notify
        self.pending=None

    def cancel(self):
        self.pending=None

    def request_grasp(self,name):
        c=self.control;m,d=c.model,c.data
        monitor=c.grasp_monitor
        if monitor is not None and monitor.evidence.get('held_seconds',0)>0:
            label=OBJECTS[monitor.object_name]['label']
            if monitor.object_name==name:
                self.notify(f'Already holding {label}.')
            else:
                self.notify(f'Still holding {label}. Use Reset scene before starting another pick.')
            return False
        self.cancel()
        self.walker.stop();c.motion=None;c.demo_start=None
        if d.body(name).xpos[2]<.70:
            raise ValueError(f'{OBJECTS[name]["label"]} is off the table. Reset the scene before retrying.')
        try:
            c.start_pick(name)
        except ValueError:
            pass
        else:
            self.notify(f'Grasping {OBJECTS[name]["label"]}: positioning, closing, then lifting.')
            return True
        position=d.body(name).xpos[:2].copy()
        # A feasibility check uses a separate MuJoCo data object. The live robot
        # reaches the chosen stance using Walker; its pose is never teleported.
        planned=None
        for goal,yaw in grasp_stance_candidates(name,position):
            if not self.planner.free(goal):continue
            hypothetical=mujoco.MjData(m)
            hypothetical.qpos[:]=d.qpos
            for ji in range(1,m.njnt):
                if m.jnt_type[ji]==mujoco.mjtJoint.mjJNT_HINGE:
                    hypothetical.qpos[m.jnt_qposadr[ji]]=c.home_qpos[m.jnt_qposadr[ji]]
            hypothetical.qpos[:2]=goal
            hypothetical.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
            mujoco.mj_forward(m,hypothetical)
            try:
                SideGraspPlan(m,hypothetical,name,solve_arm_ik)
                path=self.planner.plan(d.mocap_pos[0,:2],goal)
            except ValueError:
                continue
            planned=(path,yaw,np.asarray(goal)-position);break
        if planned is None:
            raise ValueError(f'I could not find a clear, reachable grasp for {OBJECTS[name]["label"]} in its current position.')
        c.grasp_monitor=None;c.grasp_hand_targets=None;c.hand_closure=0.
        c.target_qpos=c.home_qpos.copy()
        path,yaw,offset=planned
        self.pending=PendingObjectTask(name,yaw,offset)
        self.start_path(path)
        self.notify(f'Walking to {OBJECTS[name]["label"]}, then I will grasp it.')
        return True

    def tick(self):
        task=self.pending
        if task is None:return
        c=self.control
        if task.phase=='walking':
            if self.walker.path:return
            error=(task.yaw-self.walker.yaw+math.pi)%(2*math.pi)-math.pi
            self.walker.target_yaw=self.walker.yaw+error
            task.phase='aligning'
            task.phase_started=c.data.time
            c.target_qpos=c.home_qpos.copy()
            c.gravity_compensation=True
            self.notify(f'Aligning the hand with {OBJECTS[task.object_name]["label"]}.')
        if task.phase=='aligning':
            if c.data.time-task.phase_started>10:
                self.pending=None;self.walker.stop()
                self.notify('The robot could not settle into a grasping stance. Reset the scene and retry.')
                return
            error=abs((task.yaw-self.walker.yaw+math.pi)%(2*math.pi)-math.pi)
            if error>.015 or np.linalg.norm(c.data.qvel[:6])>.02:
                task.aligned_since=None;return
            if task.aligned_since is None:task.aligned_since=c.data.time
            if c.data.time-task.aligned_since<.7:return
            # Reobserve after walking: an object may settle or be displaced.
            # Reach the revised stance physically before computing arm IK.
            goal=c.data.body(task.object_name).xpos[:2]+task.stance_offset
            if np.linalg.norm(goal-c.data.mocap_pos[0,:2])>.006 and task.adjustments<2:
                try:path=self.planner.plan(c.data.mocap_pos[0,:2],goal)
                except ValueError as error:
                    self.pending=None;self.notify(str(error));return
                task.phase='walking';task.aligned_since=None;task.adjustments+=1
                self.start_path(path)
                self.notify(f'Adjusting the stance to {OBJECTS[task.object_name]["label"]}’s current position.')
                return
            self.pending=None
            try:
                c.start_pick(task.object_name)
            except ValueError as error:
                self.notify(str(error));return
            self.notify(f'Grasping {OBJECTS[task.object_name]["label"]}: closing fingers, then lifting.')
