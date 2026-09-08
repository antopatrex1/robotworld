"""Contact-based grasp measurements for the local five-finger controller."""
from dataclasses import dataclass, field
import mujoco
import numpy as np


def contact_summary(model, data, object_name):
    object_id=model.body(object_name).id
    parts=set();hand_contacts=0;normal_force=0.;environment_contacts=0
    for index,contact in enumerate(data.contact):
        bodies=[int(model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2)]
        if object_id not in bodies or contact.efc_address<0: continue
        other=bodies[1] if bodies[0]==object_id else bodies[0]
        name=model.body(other).name
        force=np.zeros(6)
        mujoco.mj_contactForce(model,data,index,force)
        if force[0]<.001: continue
        if name.startswith('right_shadow_'):
            parts.add(name.split('_rh_')[-1][:2])
            hand_contacts+=1;normal_force+=float(force[0])
        else:
            environment_contacts+=1
    return {'hand_contacts':hand_contacts,'hand_parts':sorted(parts),
            'normal_force_n':normal_force,'environment_contacts':environment_contacts}


@dataclass
class GraspMonitor:
    object_name: str
    origin: np.ndarray
    held_seconds: float = 0.
    longest_hold_seconds: float = 0.
    peak_lift: float = 0.
    evidence: dict = field(default_factory=dict)

    def observe(self,model,data):
        self.evidence=contact_summary(model,data,self.object_name)
        lift=float(data.body(self.object_name).xpos[2]-self.origin[2])
        self.peak_lift=max(self.peak_lift,lift)
        held=(lift>=.10 and self.evidence['hand_contacts']>=2
              and len(self.evidence['hand_parts'])>=2 and self.evidence['environment_contacts']==0)
        self.held_seconds=self.held_seconds+model.opt.timestep if held else 0.
        self.longest_hold_seconds=max(self.longest_hold_seconds,self.held_seconds)
        self.evidence.update(lift_m=lift,held_seconds=self.held_seconds,longest_hold_seconds=self.longest_hold_seconds)
        return self.evidence


class SideGraspPlan:
    """Clear the table, oppose the thumb, close the fingers, then lift and hold.

    The plan supplies actuator targets only. No free-body state is assigned.
    """
    def __init__(self,model,data,object_name,solve_ik):
        self.model=model
        self.object_name=object_name
        low_object=object_name in ('mug','apple')
        self.clearance_duration=3. if low_object else 8.
        self.close_time=self.clearance_duration+3.
        self.lift_time=self.clearance_duration+5.
        self.origin=data.body(object_name).xpos.copy()
        if self.origin[2]<.70:
            raise ValueError('The object is off the table. Reset the scene before retrying the grasp.')
        from scipy.spatial.transform import Rotation
        yaw=Rotation.from_quat(data.qpos[3:7][[1,2,3,0]]).as_euler('xyz')[2]
        heading=Rotation.from_euler('z',yaw).as_matrix()
        rotation=heading@np.array([[0,0,1],[0,-1,0],[1,0,0.]])
        if low_object:
            # Spread the fingers across a low object. The upright bottle uses
            # a vertical finger row instead.
            rotation=rotation@Rotation.from_euler('z',-np.pi/2).as_matrix()
        # The curved finger pads and opposing thumb enclose this palm-local point.
        palm_depth={'mug':-.035,'apple':-.04}.get(object_name,-.047)
        object_in_palm=np.array([0,palm_depth,.11])
        target=self.origin-rotation@(object_in_palm-np.array([0,-.035,.09]))
        pre=target+rotation@np.array([0,.10,0])
        base=data.qpos[:3].copy()
        self.arm_ids=np.array([model.joint('right_'+name+'_joint').qposadr[0] for name in
            ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')])
        self.home=data.qpos.copy()
        # Match the verified hand approach. The first waypoint raises the wrist
        # outside the table; orientation is applied only after the fingers clear it.
        high1,e=solve_ik(model,self.home,base+heading@np.array([.08,-.36,.33]),iterations=300)
        # Solve the final approach from the initial pose, as in the validated
        # experiment, to avoid selecting a different redundant-arm solution.
        prepose,e4=solve_ik(model,self.home,pre,rotation)
        grasp,e5=solve_ik(model,prepose,target,rotation)
        lift,e6=solve_ik(model,grasp,target+[0,0,.18],rotation)
        if low_object:
            high2,e2=high1,e
            high3,e3=lift,e6
        else:
            high2,e2=solve_ik(model,high1,base+heading@np.array([.12,-.38,.29]),rotation,iterations=300)
            high3,e3=solve_ik(model,high2,pre+[0,0,.18],rotation,iterations=300)
        if max(e,e2,e3,e5,e6)>.015 or e4>.05:
            raise ValueError('The grasp pose is out of reach from this stance.')
        self.clearance=[high1,high2,high3,prepose]
        self.prepose,self.grasp,self.lift=prepose,grasp,lift
        self.lift_poses=[grasp]
        for height in np.linspace(0,.18,61)[1:]:
            pose,error=solve_ik(model,self.lift_poses[-1],target+[0,0,height],rotation,iterations=100)
            if error>.015: raise ValueError('The arm cannot follow a clear lift from this position.')
            self.lift_poses.append(pose)
        self.lift=self.lift_poses[-1]
        self.clearance_source=None;self.clearance_index=-1
        self.stage='Preparing grasp'
        self.closure=0.
        self.thumb_ready=0.

    def sample(self,elapsed,data):
        # A shorter clearance keeps the low mug in place while the open hand
        # approaches. Grasp closure and lifting keep their original timing.
        elapsed=(elapsed*8/self.clearance_duration if elapsed<self.clearance_duration
                 else elapsed+8-self.clearance_duration)
        if elapsed<8:
            index=min(3,int(elapsed/2))
            if index!=self.clearance_index:
                self.clearance_source=data.qpos.copy();self.clearance_index=index
            alpha=min(1,(elapsed-index*2)/1.7)
            alpha=alpha*alpha*(3-2*alpha)
            pose=self.clearance_source*(1-alpha)+self.clearance[index]*alpha
            self.closure=0.;self.thumb_ready=0.
            self.stage='Raising and positioning the hand'
        else:
            t=elapsed-8
            self.thumb_ready=min(1,t/.5)
            if t<1.5:
                a=t/1.5;pose=self.prepose*(1-a)+self.grasp*a;self.closure=0.
                self.stage='Placing thumb opposite the fingers'
            elif t<3:
                pose=self.grasp;self.closure=min(1,(t-1.5))
                self.stage='Closing fingers around the object'
            elif t<5:
                a=(t-3)/2;index=min(59,int(a*60));fraction=a*60-index
                pose=self.lift_poses[index]*(1-fraction)+self.lift_poses[index+1]*fraction
                self.closure=1.;self.stage='Lifting with finger contacts'
            else:
                pose=self.lift;self.closure=1.;self.stage='Holding the object'
        targets={}
        for aid in range(self.model.nu):
            name=self.model.actuator(aid).name
            if 'right_shadow' not in name: continue
            joint=name.split('_A_')[-1]
            value=0.
            if joint[:2] in ('FF','MF','RF','LF'):
                if joint.endswith('J3'): value=.65*self.closure
                elif joint.endswith('J0'): value=2.2*self.closure
            elif joint.startswith('TH'):
                thumb_base=.7 if self.object_name=='apple' else 1.047
                value={'THJ5':thumb_base,'THJ4':1.059,'THJ3':0.,'THJ2':.65,'THJ1':0.}[joint]*self.thumb_ready
                if joint=='THJ4' and elapsed>=8: value=.4+(1.059-.4)*self.closure
            targets[aid]=value
        return pose,targets
