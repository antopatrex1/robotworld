"""Regression: the original hand closure pushed the bottle away and lifted <1cm."""
import json
import unittest
import mujoco
import numpy as np
from robot_world.scene import build_scene, ROOT
from robot_world.control import Controller


class GraspTests(unittest.TestCase):
    def run_grasp(self,offset=(0,0),seconds=20):
        model,data,_,_=build_scene('warm_living_room')
        address=model.joint('bottle_free').qposadr[0]
        # Only vary the initial condition. Execution never writes the object pose.
        data.qpos[address:address+2]+=offset
        mujoco.mj_forward(model,data)
        controller=Controller(model,data)
        original_gains=model.actuator_gainprm.copy()
        for _ in range(500):controller.step()
        controller.start_pick()
        max_effort_violation=0.
        limited_joints=np.flatnonzero(model.jnt_actfrclimited)
        for step in range(int(seconds/model.opt.timestep)):
            controller.step()
            if step%50==0:
                for ji in limited_joints:
                    force=data.qfrc_actuator[model.jnt_dofadr[ji]]
                    low,high=model.jnt_actfrcrange[ji]
                    max_effort_violation=max(max_effort_violation,force-high,low-force)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertFalse(data.warning.number.any())
        self.assertLess(max_effort_violation,1e-6)
        np.testing.assert_array_equal(data.qfrc_applied,0.)
        np.testing.assert_array_equal(data.xfrc_applied,0.)
        self.assertEqual(model.neq,1)
        self.assertEqual(model.equality(0).name,'base_support')
        evidence=controller.grasp_monitor.evidence
        self.assertGreater(evidence['lift_m'],.15)
        self.assertGreater(evidence['held_seconds'],3.)
        self.assertIn('th',evidence['hand_parts'])
        self.assertGreaterEqual(len(evidence['hand_parts']),2)
        self.assertEqual(evidence['environment_contacts'],0)
        limited=model.actuator_forcelimited.astype(bool)
        self.assertTrue(np.all(data.actuator_force[limited]<=model.actuator_forcerange[limited,1]+1e-6))
        self.assertTrue(np.all(data.actuator_force[limited]>=model.actuator_forcerange[limited,0]-1e-6))
        result=controller.metrics()
        controller.reset()
        self.assertIsNone(controller.grasp_plan)
        self.assertIsNone(controller.grasp_hand_targets)
        self.assertEqual(model.opt.noslip_iterations,0)
        np.testing.assert_array_equal(model.actuator_gainprm,original_gains)
        return result

    def test_bottle_lifts_and_holds_for_thirty_seconds_with_real_contacts(self):
        result=self.run_grasp(seconds=45)
        self.assertGreater(result['grasp_evidence']['held_seconds'],30)
        (ROOT/'build/grasp_validation.json').write_text(json.dumps(result,indent=2))

    def test_grasp_handles_small_initial_position_changes(self):
        for offset in ((.01,0),(0,-.01)):
            with self.subTest(offset=offset): self.run_grasp(offset)

    def test_off_table_object_is_rejected_without_starting_motion(self):
        model,data,_,_=build_scene('lab')
        data.qpos[model.joint('bottle_free').qposadr[0]+2]=.08
        mujoco.mj_forward(model,data)
        controller=Controller(model,data)
        with self.assertRaisesRegex(ValueError,'off the table'):controller.start_pick()
        self.assertIsNone(controller.demo_start)
        self.assertFalse(controller.gravity_compensation)

if __name__=='__main__':unittest.main()
