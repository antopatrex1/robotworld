"""Explicitly assisted control for local experiments; no trained balance policy."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .grasp import SideGraspPlan, GraspMonitor, contact_summary


def solve_arm_ik(model, initial, target, orientation=None, side="right", iterations=160):
    data = mujoco.MjData(model)
    data.qpos[:] = initial
    joints = [model.joint(f"{side}_{name}_joint") for name in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
    qids = np.array([j.qposadr[0] for j in joints])
    vids = np.array([j.dofadr[0] for j in joints])
    ranges = np.array([j.range for j in joints])
    sid = model.site(f"{side}_grasp").id
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(iterations):
        # IK needs site poses and joint axes, not contact/dynamics solves.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        error = np.asarray(target) - data.site_xpos[sid]
        mujoco.mj_jacSite(model, data, jp, jr, sid)
        jac = jp[:, vids]
        if orientation is not None:
            rotation_error = Rotation.from_matrix(orientation @ data.site_xmat[sid].reshape(3, 3).T).as_rotvec()
            error = np.concatenate([error, 0.2 * rotation_error])
            jac = np.vstack([jac, 0.2 * jr[:, vids]])
        if np.linalg.norm(error) < 0.001:
            break
        delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.002 * np.eye(len(error)), error)
        data.qpos[qids] = np.clip(data.qpos[qids] + np.clip(delta, -0.12, 0.12), ranges[:, 0], ranges[:, 1])
    mujoco.mj_kinematics(model, data)
    return data.qpos.copy(), float(np.linalg.norm(np.asarray(target) - data.site_xpos[sid]))


class Controller:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.home_qpos, self.home_ctrl = data.qpos.copy(), data.ctrl.copy()
        self.target_qpos = data.qpos.copy()
        self.hand_closure = 0.0
        self.motion = None
        self.motion_start = 0.0
        self.demo_start = None
        self.status = "ASSISTED PHYSICS · base support active"
        self.last_ik_error = None
        self._last_ik_time = -1.0
        self._grasp_origin = None
        self.reference_hands = False
        self._reference_data = mujoco.MjData(model)
        self.peak_lift = 0.0
        self.pick_object = "bottle"
        self.grasp_plan = None
        self.grasp_monitor = None
        self.grasp_hand_targets = None
        self.gravity_compensation = False
        self.original_gain = model.actuator_gainprm.copy()
        self.original_bias = model.actuator_biasprm.copy()
        self.original_noslip = model.opt.noslip_iterations

    def reset(self):
        self.model.actuator_gainprm[:] = self.original_gain
        self.model.actuator_biasprm[:] = self.original_bias
        self.model.opt.noslip_iterations = self.original_noslip
        self.grasp_plan = self.grasp_monitor = self.grasp_hand_targets = None
        self.gravity_compensation = False
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.home_qpos
        self.data.ctrl[:] = self.home_ctrl
        self.target_qpos = self.home_qpos.copy()
        self.data.mocap_pos[0] = self.home_qpos[:3]
        self.data.mocap_quat[0] = self.home_qpos[3:7]
        self.hand_closure = 0
        self.motion = None
        self.demo_start = None
        self.reference_hands = False
        self.peak_lift = 0.0
        self._grasp_origin = None
        self.last_ik_error = None
        self.status = "ASSISTED PHYSICS · base support active"
        mujoco.mj_forward(self.model, self.data)

    def start_motion(self, motion, include_hands=False):
        self.reset()
        self.reference_hands = include_hands
        self.motion, self.motion_start = motion, self.data.time
        self.status = "Imported reference tracking" if include_hands else "SOMA reference tracking"

    def start_pick(self, object_name="bottle"):
        plan = SideGraspPlan(self.model,self.data,object_name,solve_arm_ik)
        self.motion = None
        self.demo_start = self.data.time
        self.pick_object = object_name
        self.target_qpos = self.data.qpos.copy()
        self._grasp_origin = self.data.body(object_name).xpos.copy()
        self.grasp_plan = plan
        self.grasp_monitor = GraspMonitor(object_name,self._grasp_origin.copy())
        self.grasp_hand_targets = None
        self.gravity_compensation = True
        # Static-friction postprocessing prevents gradual soft-contact drift.
        self.model.opt.noslip_iterations = 3
        for aid in range(self.model.nu):
            if 'right_shadow' in self.model.actuator(aid).name:
                self.model.actuator_gainprm[aid,0] = 4*self.original_gain[aid,0]
                self.model.actuator_biasprm[aid,1] = 4*self.original_bias[aid,1]
        self.peak_lift = 0.
        self.reference_hands = False
        self.status = 'Preparing a side grasp'

    def set_body_targets(self, pose):
        for actuator in range(self.model.nu):
            name = self.model.actuator(actuator).name
            if "shadow" not in name:
                joint = self.model.joint(name)
                value = pose[joint.qposadr[0]]
                self.data.ctrl[actuator] = np.clip(value, *self.model.actuator_ctrlrange[actuator])

    def set_hands(self, closure):
        for aid in range(self.model.nu):
            name = self.model.actuator(aid).name
            if "shadow" not in name:
                continue
            joint = name.split("_A_")[-1]
            target = 0.0
            if joint[:2] in ("FF", "MF", "RF", "LF"):
                if joint.endswith("J4"):
                    target = 0.2
                elif joint.endswith("J3"):
                    target = 0.8
                elif joint.endswith("J0"):
                    target = 1.4
            else:
                target = {"THJ5": 1.047, "THJ4": 1.059, "THJ3": 0.0, "THJ2": 0.65, "THJ1": 0.0}.get(joint, 0)
            self.data.ctrl[aid] = np.clip(target * closure, *self.model.actuator_ctrlrange[aid])

    def step(self):
        if self.motion is not None:
            t = self.data.time - self.motion_start
            pose = self.motion.sample(t)
            self.target_qpos = pose
            self.data.mocap_pos[0] = pose[:3]
            self.data.mocap_quat[0] = pose[3:7]
            if t >= self.motion.duration:
                self.motion = None
                self.status = "SOMA clip complete · assisted pose hold"
        if self.demo_start is not None and self.grasp_plan is not None:
            elapsed = self.data.time-self.demo_start
            self.target_qpos,self.grasp_hand_targets = self.grasp_plan.sample(elapsed,self.data)
            self.status = self.grasp_plan.stage
            # A missed closure must not trigger a successful-looking empty lift.
            if self.grasp_plan.close_time-.01 <= elapsed < self.grasp_plan.close_time+.05:
                contact = contact_summary(self.model,self.data,self.pick_object)
                if contact['hand_contacts'] < 2 or len(contact['hand_parts']) < 2:
                    self.demo_start = None
                    self.status = 'Grasp missed: opposing contacts were not established. Reset and retry.'
        self.set_body_targets(self.target_qpos)
        if self.reference_hands:
            self._reference_data.qpos[:] = self.target_qpos
            mujoco.mj_forward(self.model, self._reference_data)
            # This evaluates tendon sums through MuJoCo's own transmission model.
            self.data.ctrl[:] = np.clip(self._reference_data.actuator_length, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        else:
            self.set_hands(self.hand_closure)
        if self.grasp_hand_targets is not None:
            for aid,value in self.grasp_hand_targets.items():
                self.data.ctrl[aid] = np.clip(value,*self.model.actuator_ctrlrange[aid])
        if self.gravity_compensation:
            # Feedforward goes through existing position actuators, preserving
            # G1 joint force limits. Never apply forces to the object/free joints.
            for aid in range(self.model.nu):
                name = self.model.actuator(aid).name
                if 'shadow' in name: continue
                joint = self.model.joint(name)
                kp = self.model.actuator_gainprm[aid,0]
                self.data.ctrl[aid] = np.clip(self.data.ctrl[aid]+self.data.qfrc_bias[joint.dofadr[0]]/kp,
                                             *self.model.actuator_ctrlrange[aid])
        mujoco.mj_step(self.model, self.data)
        if self.grasp_monitor is not None:
            evidence = self.grasp_monitor.observe(self.model,self.data)
            if self.demo_start is not None and self.data.time-self.demo_start >= self.grasp_plan.lift_time:
                if evidence['held_seconds'] >= 1:
                    self.status = f"Holding {self.pick_object}: {100*evidence['lift_m']:.1f} cm lift, finger contacts verified"
                elif self.grasp_monitor.longest_hold_seconds >= 1 and evidence['hand_contacts'] == 0:
                    self.demo_start = None
                    self.status = 'Object slipped from the hand. Reset and retry.'
                elif self.data.time-self.demo_start > self.grasp_plan.lift_time+2 and evidence['held_seconds'] < .1:
                    self.demo_start = None
                    self.status = 'Lift was not secured. Reset and retry.'
        if self._grasp_origin is not None:
            self.peak_lift = max(self.peak_lift, float(self.data.body(self.pick_object).xpos[2] - self._grasp_origin[2]))

    def metrics(self):
        evidence = contact_summary(self.model,self.data,self.pick_object)
        if self.grasp_monitor is not None: evidence.update(self.grasp_monitor.evidence)
        return {"time": round(self.data.time,3), "bottle_position": self.data.body("bottle").xpos.tolist(),
                "object_name": self.pick_object,"object_position":self.data.body(self.pick_object).xpos.tolist(),
                "hand_object_contacts":evidence['hand_contacts'],"grasp_evidence":evidence,
                "ik_position_error_m": self.last_ik_error,
                "base_support_active":bool(self.data.eq_active[self.model.equality("base_support").id]),
                "peak_bottle_lift_m":self.peak_lift,"warnings":self.data.warning.number.tolist(),"status":self.status}
