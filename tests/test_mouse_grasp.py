"""Mouse pickups must hold through contacts, including legacy camera detections."""
import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from robot_world.commands import parse_command
from robot_world.control import Controller
from robot_world.motion import SomaMotion
from robot_world.navigation import Planner, Walker
from robot_world.object_tasks import ObjectTaskRunner
from robot_world.scene import ROOT, build_scene
from robot_world.table_alignment import homography, project


class MouseGraspTests(unittest.TestCase):
    def layout(self, legacy=False):
        layout=json.loads((ROOT/'configs/calibration/paper_table.json').read_text())
        inverse=np.linalg.inv(homography(layout['paper_pixels'],layout['paper_size_m']))
        name='proxy_2' if legacy else 'mouse'
        layout['objects']=[dict(id=name,kind='proxy' if legacy else 'mouse',category='mouse',
            label='Observed blue mouse',color_rgb=[72,143,173],
            base_pixel=project(inverse,[[.122934,.401467]])[0].tolist())]
        layout['source_capture']='mouse regression fixture'
        layout['limitations']=['Approximate mouse dimensions.']
        return layout,name

    def test_new_and_legacy_mouse_are_pickable_but_phone_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            for legacy in (False,True):
                layout,name=self.layout(legacy)
                layout['objects'].append(dict(id='phone',kind='proxy',category='cell phone',
                    label='Observed blue cell phone',color_rgb=[66,102,112],base_pixel=[400,225]))
                path=Path(directory)/'layout.json';path.write_text(json.dumps(layout))
                model,_,config,_=build_scene('camera_table',camera_layout=path,output_path=Path(directory)/'scene.xml')
                self.assertIn(name,config['graspable_objects'])
                self.assertNotIn('phone',config['graspable_objects'])
                self.assertEqual(parse_command('pick up the Observed blue mouse',config['object_labels']).values[0],name)
                self.assertGreaterEqual(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,name+'_mouse_grasp'),0)

    def test_command_walks_then_lifts_mouse_with_opposing_contacts(self):
        with tempfile.TemporaryDirectory() as directory:
            layout,name=self.layout(legacy=True)
            path=Path(directory)/'layout.json';path.write_text(json.dumps(layout))
            model,data,config,_=build_scene('camera_table',camera_layout=path,output_path=Path(directory)/'scene.xml')
            control=Controller(model,data)
            gait=SomaMotion(ROOT/'vendor/soma-retargeter/assets/motions/csv/Neutral_walk_forward_002__A057.csv',model,data.qpos)
            walker=Walker(control,gait);walker.speed=.3
            messages=[]
            tasks=ObjectTaskRunner(control,walker,Planner(config),walker.begin,messages.append,config['object_labels'])
            for _ in range(500):control.step()
            before=data.qpos.copy()
            tasks.request_grasp(name)
            np.testing.assert_array_equal(data.qpos,before)
            max_step=0.
            for frame in range(1400):
                for _ in range(16):
                    before_base=data.mocap_pos[0,:2].copy()
                    walker.step();control.step()
                    max_step=max(max_step,np.linalg.norm(data.mocap_pos[0,:2]-before_base))
                tasks.tick()
                if control.grasp_monitor and control.grasp_monitor.held_seconds>=3:break
                if control.status.startswith(('Grasp missed','Lift was','Object slipped')):break
            self.assertIsNotNone(control.grasp_monitor, messages)
            evidence=control.grasp_monitor.evidence
            self.assertGreaterEqual(evidence['held_seconds'],3, evidence)
            self.assertGreater(evidence['lift_m'],.15)
            self.assertIn('th',evidence['hand_parts'])
            self.assertGreaterEqual(len(evidence['hand_parts']),2)
            self.assertEqual(evidence['environment_contacts'],0)
            self.assertTrue(any('Walking to' in message for message in messages))
            self.assertLessEqual(max_step,walker.speed*model.opt.timestep+1e-10)
            self.assertFalse(data.warning.number.any())
            np.testing.assert_array_equal(data.xfrc_applied,0)
            np.testing.assert_array_equal(data.qfrc_applied,0)
            self.assertEqual(model.neq,1)
            control.reset()
            self.assertIsNone(control.grasp_plan)
