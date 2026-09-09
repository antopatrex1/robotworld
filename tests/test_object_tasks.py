"""A named object request must finish the complete approach and physical grasp."""
import json
import unittest
import mujoco
import numpy as np
from robot_world.scene import build_scene,ROOT
from robot_world.control import Controller
from robot_world.motion import SomaMotion
from robot_world.navigation import Planner,Walker
from robot_world.object_tasks import ObjectTaskRunner
from robot_world.commands import parse_command
from robot_world.objects import OBJECTS


class ObjectTaskTests(unittest.TestCase):
    def test_untouched_mug_stays_in_place_while_waiting_for_a_prompt(self):
        model,data,_,_=build_scene('warm_living_room')
        control=Controller(model,data)
        for _ in range(1000):control.step()
        settled=data.body('mug').xpos.copy()
        for _ in range(5000):control.step()
        # The original single base contact produced ~18mm drift in this time.
        self.assertLess(np.linalg.norm(data.body('mug').xpos-settled),.001)
        self.assertFalse(data.warning.number.any())

    def run_named_grasp(self,name,phrase,environment='warm_living_room'):
        model,data,config,_=build_scene(environment)
        control=Controller(model,data)
        gait=SomaMotion(ROOT/'vendor/soma-retargeter/assets/motions/csv/Neutral_walk_forward_002__A057.csv',model,data.qpos)
        walker=Walker(control,gait);walker.speed=.3
        messages=[]
        labels=config.get('object_labels',{})
        label=labels.get(name,OBJECTS[name]['label'])
        task=ObjectTaskRunner(control,walker,Planner(config),walker.begin,messages.append,labels)
        for _ in range(1000):control.step()
        object_start=data.body(name).xpos.copy()
        robot_start=data.qpos[:3].copy()
        command=parse_command(phrase,labels)
        self.assertEqual(command.action,'pick')
        task.request_grasp(command.values[0])
        # Planning must not teleport either participant.
        np.testing.assert_array_equal(data.qpos[:3],robot_start)
        np.testing.assert_array_equal(data.body(name).xpos,object_start)
        needs_walk=task.pending is not None
        if environment!='camera_table':self.assertTrue(needs_walk)
        max_base_step=0.
        limited=np.flatnonzero(model.jnt_actfrclimited)
        for frame in range(1100):
            for _ in range(16):
                before=data.mocap_pos[0,:2].copy()
                walker.step();control.step()
                max_base_step=max(max_base_step,np.linalg.norm(data.mocap_pos[0,:2]-before))
            task.tick()
            if frame%30==0:
                for ji in limited:
                    force=data.qfrc_actuator[model.jnt_dofadr[ji]]
                    low,high=model.jnt_actfrcrange[ji]
                    self.assertLessEqual(force,high+1e-6)
                    self.assertGreaterEqual(force,low-1e-6)
            if control.grasp_monitor and control.grasp_monitor.held_seconds>=5:break
        self.assertLessEqual(max_base_step,walker.speed*model.opt.timestep+1e-10)
        if needs_walk:
            self.assertGreater(np.linalg.norm(data.qpos[:2]-robot_start[:2]),.5)
            self.assertTrue(any('Walking to '+label in msg for msg in messages))
        self.assertTrue(any('Grasping '+label in msg for msg in messages))
        self.assertIsNone(task.pending)
        self.assertIsNotNone(control.grasp_monitor)
        evidence=control.grasp_monitor.evidence
        self.assertGreater(evidence['lift_m'],.10)
        self.assertGreaterEqual(evidence['held_seconds'],5)
        self.assertIn('th',evidence['hand_parts'])
        self.assertGreaterEqual(len(evidence['hand_parts']),2)
        self.assertEqual(evidence['environment_contacts'],0)
        self.assertFalse(data.warning.number.any())
        np.testing.assert_array_equal(data.qfrc_applied,0.)
        np.testing.assert_array_equal(data.xfrc_applied,0.)
        self.assertEqual(model.neq,1)
        self.assertEqual(model.equality(0).name,'base_support')
        (ROOT/f'build/{name}_command_test.json').write_text(json.dumps(control.metrics(),indent=2))
        # New requests must preserve an existing physical hold.
        active_start=control.demo_start
        finger_targets=control.grasp_hand_targets.copy()
        self.assertFalse(task.request_grasp(name))
        self.assertEqual(messages[-1],'Already holding '+label+'.')
        self.assertFalse(task.request_grasp('bottle'))
        self.assertIn('Still holding '+label,messages[-1])
        self.assertEqual(control.demo_start,active_start)
        self.assertEqual(control.grasp_hand_targets,finger_targets)

    def test_red_mug_command_walks_then_grasps_with_opposing_contacts(self):
        self.run_named_grasp('mug','grab the red mug')

    def test_observed_black_mug_grasp_from_camera_layout(self):
        self.run_named_grasp('mug','pick up the Observed black mug','camera_table')

    def test_green_apple_command_walks_then_grasps_with_opposing_contacts(self):
        self.run_named_grasp('apple','pick up the green apple')



if __name__=='__main__':unittest.main()
