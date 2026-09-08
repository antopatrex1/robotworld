import json
import unittest

import mujoco
import numpy as np

from robot_world.scene import ROOT, build_scene
from robot_world.control import Controller
from robot_world.table_alignment import homography, project, table_to_world


class TableAlignmentTests(unittest.TestCase):
    def test_recovers_known_projective_table_coordinates(self):
        size = [.210, .147]
        corners = np.array([[0,0], [.210,0], [.210,.147], [0,.147]])
        forward = np.array([[1000,100,80], [10,250,210], [.4,1.2,1]])
        pixels = project(forward, corners)
        inverse = homography(pixels, size)
        # Check points not used to fit the transform, including extrapolation.
        samples = np.array([[.05,.07], [.32,.1], [.4,.3]])
        np.testing.assert_allclose(project(inverse, project(forward, samples)), samples, atol=1e-10)
        with self.assertRaises(ValueError):
            homography(np.zeros((4,2)), size)
        with self.assertRaises(ValueError):
            project(np.diag([1,1,0]), [[0,0]])

    def test_metric_paper_corner_and_axes(self):
        calibration = json.loads((ROOT/'configs/calibration/paper_table.json').read_text())
        corners = project(homography(calibration['paper_pixels'], calibration['paper_size_m']), calibration['paper_pixels'])
        world = table_to_world(calibration, corners)
        np.testing.assert_allclose(world[0], [.91,-.5,.725])
        self.assertAlmostEqual(np.linalg.norm(world[1]-world[0]), .210)
        self.assertAlmostEqual(np.linalg.norm(world[3]-world[0]), .147)
        np.testing.assert_allclose(world[:,2], .725)

    def test_camera_scene_has_no_hidden_demo_props_and_mug_rests(self):
        model, data, _, _ = build_scene('camera_table')
        self.assertEqual(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'bottle'), -1)
        self.assertEqual(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'apple'), -1)
        initial = data.body('mug').xpos.copy()
        controller = Controller(model, data)
        for _ in range(1000):
            controller.step()
        np.testing.assert_allclose(data.body('mug').xpos[:2], initial[:2], atol=.005)
        self.assertGreater(data.body('mug').xpos[2], .75)
        self.assertFalse(data.warning.number.any())
        self.assertIsNone(controller.metrics()['bottle_position'])
        controller.reset()
        mujoco.mj_forward(model,data)
        np.testing.assert_allclose(data.body('mug').xpos, initial)


if __name__ == '__main__':
    unittest.main()
