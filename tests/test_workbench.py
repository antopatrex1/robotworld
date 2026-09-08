import tempfile
from pathlib import Path
import unittest

import mujoco
import numpy as np

from robot_world.scene import ROOT, build_scene, available_environments
from robot_world.control import Controller, solve_arm_ik
from robot_world.motion import SomaMotion, load_spider, ReferenceMotion


class WorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.initial, _, cls.xml = build_scene()

    def fresh(self):
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.initial.qpos
        data.ctrl[:] = self.initial.ctrl
        mujoco.mj_forward(self.model, data)
        return data

    def test_model_has_two_five_finger_hands_and_no_duplicate_wrists(self):
        model = self.model
        names = [model.joint(i).name for i in range(model.njnt)]
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(sum("shadow" in n for n in names), 44)
        self.assertEqual(model.nu, 65)
        self.assertEqual(model.ntendon, 8)
        self.assertFalse(any("WRJ" in n for n in names))
        for side, short in [("left", "lh"), ("right", "rh")]:
            for finger in ["ff", "mf", "rf", "lf", "th"]:
                self.assertGreater(model.body(f"{side}_shadow_{short}_{finger}distal").id, 0)

    def test_supported_stand_and_free_bottle_settle(self):
        data = self.fresh()
        bottle = self.model.joint("bottle_free").qposadr[0]
        data.qpos[bottle + 2] += 0.1
        controller = Controller(self.model, data)
        for _ in range(1500):
            controller.step()
        self.assertTrue(np.isfinite(data.qpos).all())
        np.testing.assert_allclose(data.qpos[:3], [0, 0, 0.79], atol=0.005)
        self.assertAlmostEqual(data.body("bottle").xpos[2], 0.8, delta=0.005)
        self.assertFalse(data.warning.number.any())

    def test_arm_reaches_task_pose(self):
        target = np.array([0.28, -0.27, 0.805])
        pose, error = solve_arm_ik(self.model, self.initial.qpos, target, np.diag([1., -1., -1.]))
        self.assertLess(error, 0.002)
        for i in range(self.model.njnt):
            if self.model.jnt_limited[i]:
                q = pose[self.model.jnt_qposadr[i]]
                low, high = self.model.jnt_range[i]
                self.assertTrue(low - 1e-6 <= q <= high + 1e-6)

    def test_real_soma_sample_units_names_and_quaternion(self):
        path = ROOT / "vendor/soma-retargeter/assets/motions/csv/Neutral_walk_forward_002__A057.csv"
        motion = SomaMotion(path, self.model, self.initial.qpos)
        original = np.loadtxt(path, delimiter=",", skiprows=1)
        self.assertAlmostEqual(motion.qpos[0, 2], original[0, 3] / 100)
        header = path.read_text().splitlines()[0].split(",")
        name = "right_elbow_joint"
        col = header.index(name + "_dof")
        self.assertAlmostEqual(motion.qpos[0, self.model.joint(name).qposadr[0]], np.deg2rad(original[0, col]))
        self.assertAlmostEqual(np.linalg.norm(motion.sample(0.145)[3:7]), 1.0)
        for i in range(self.model.njnt):
            if "shadow" in self.model.joint(i).name:
                self.assertEqual(motion.sample(0.145)[self.model.jnt_qposadr[i]], self.initial.qpos[self.model.jnt_qposadr[i]])

    def test_spider_mapping_uses_names_across_inserted_hand_joints(self):
        source = ROOT / "vendor/mujoco_menagerie/unitree_g1/g1.xml"
        source_model = mujoco.MjModel.from_xml_path(str(source))
        qpos = np.repeat(source_model.qpos0[None, :], 3, axis=0)
        qpos[:, source_model.joint("right_elbow_joint").qposadr[0]] = [0.4, 0.6, 0.8]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spider.npz"
            np.savez(path, qpos=qpos[:, None, :], time=np.array([0, 0.02, 0.04])[:, None, None])
            result = load_spider(path, source, self.model, self.initial.qpos)
        np.testing.assert_allclose(result["qpos"][:, self.model.joint("right_elbow_joint").qposadr[0]], [0.4, 0.6, 0.8])
        self.assertEqual(result["unmapped"], [])
        self.assertEqual(result["qpos"].shape, (3, self.model.nq))

    def test_reject_spider_with_wrong_source_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.npz"
            np.savez(path, qpos=np.zeros((2, 12)))
            with self.assertRaisesRegex(ValueError, "source MJCF"):
                load_spider(path, self.xml, self.model, self.initial.qpos)

    def test_scene_switches_keep_robot_schema_and_physics_finite(self):
        expected = [self.model.joint(i).name for i in range(self.model.njnt)]
        for name in ("kitchen", "living_room", "world_kitchen"):
            with self.subTest(environment=name):
                if name not in available_environments().values():
                    continue
                model, data, _, _ = build_scene(name)
                self.assertEqual([model.joint(i).name for i in range(model.njnt)], expected)
                controller = Controller(model, data)
                for _ in range(500):
                    controller.step()
                self.assertTrue(np.isfinite(data.qpos).all())
                self.assertFalse(data.warning.number.any())

    def test_no_hidden_object_attachment(self):
        self.assertEqual(self.model.neq, 1)
        self.assertEqual(self.model.equality(0).name, "base_support")
        self.assertEqual(int(self.model.joint("bottle_free").type[0]), int(mujoco.mjtJoint.mjJNT_FREE))


if __name__ == "__main__":
    unittest.main()
