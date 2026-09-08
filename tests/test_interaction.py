import json
import tempfile
import unittest
from pathlib import Path
import gzip
import struct
import numpy as np
from robot_world.commands import parse_command
from robot_world.navigation import Planner, Walker
from robot_world.splats import decode_spz, load_world_splats
from robot_world.scene import ROOT, build_scene
from robot_world.control import Controller
from robot_world.motion import SomaMotion
from robot_world.objects import OBJECTS

class InteractionTests(unittest.TestCase):
    def test_prompt_understands_labeled_objects_and_rejects_unbounded_actions(self):
        self.assertEqual(parse_command('walk backward 1 meter').values,('backward',1.))
        self.assertEqual(parse_command('walk to x -1.5 y 0.8').values,(-1.5,.8))
        for key,info in OBJECTS.items():
            self.assertEqual(parse_command('go to the '+info['label']).values,(key,))
        for text in ('walk forward 500 meters','run shell commands','pick up a car','open hands and delete all files','walk forward nan meters'):
            with self.assertRaises(ValueError): parse_command(text)

    def test_navigation_routes_around_obstacle_and_rejects_collision(self):
        planner=Planner({})
        path=planner.plan([0,0],[1.4,0])
        self.assertGreater(len(path),2)
        self.assertTrue(all(planner.segment_free(a,b) for a,b in zip(path,path[1:])))
        with self.assertRaises(ValueError): planner.plan([0,0],[.57,-.1])
        with self.assertRaises(ValueError): planner.plan([0,0],[100,100])

    def test_spz_signed_coordinates_opacity_and_rgb(self):
        # One known v2 Gaussian: fixed point (-1, .5, 2), identity rotation.
        coordinates=b''.join((n & 0xffffff).to_bytes(3,'little') for n in (-4096,2048,8192))
        payload=struct.pack('<III4B',0x5053474e,2,1,0,12,0,0)+coordinates+bytes([255])+bytes([128]*3)+bytes([80]*3)+bytes([128]*3)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'one.spz';p.write_bytes(gzip.compress(payload));s=decode_spz(p)
            np.testing.assert_allclose(s['centers'],[[-1,.5,2]])
            self.assertEqual(float(s['opacities'][0,0]),1)
            self.assertTrue(np.all(np.linalg.eigvalsh(s['covariances'])>0))
            self.assertTrue(np.all(abs(s['rgbs'].astype(int)-128)<3))
            p.write_bytes(gzip.compress(payload[:-1]))
            with self.assertRaisesRegex(ValueError,'payload length'): decode_spz(p)

    def test_requested_world_and_ten_physical_objects(self):
        model,data,config,_=build_scene('warm_living_room')
        manifest=json.loads((ROOT/config['world_manifest']).read_text())
        self.assertEqual(manifest['world_id'],'7f964e6d-dbdd-4cf7-a8fb-9292001f69ba')
        splats=load_world_splats(manifest,ROOT)
        self.assertEqual(len(splats['centers']),150000)
        self.assertTrue(np.isfinite(splats['covariances']).all())
        planner=Planner(config)
        self.assertTrue(planner.free([0,0]))
        start=np.array([0.,0.])
        for goal in [[-.18,-.85],[1.35,-.85],[1.35,.7],[-.18,.7],[0,0]]:
            path=planner.plan(start,goal)
            self.assertTrue(all(planner.segment_free(a,b) for a,b in zip(path,path[1:])))
            start=path[-1]
        control=Controller(model,data)
        for _ in range(500): control.step()
        for name in OBJECTS:
            self.assertGreater(data.body(name).xpos[2],.72)
            self.assertEqual(int(model.joint(name+'_free').type[0]),0)
        self.assertFalse(data.warning.number.any())

    def test_full_turn_and_stop_cancel_rotation(self):
        model,data,_,_=build_scene('lab')
        control=Controller(model,data)
        motion=SomaMotion(ROOT/'vendor/soma-retargeter/assets/motions/csv/Neutral_walk_forward_002__A057.csv',model,data.qpos)
        walker=Walker(control,motion)
        walker.target_yaw=-1.5*np.pi
        for _ in range(2100): walker.step();control.step()
        self.assertAlmostEqual(walker.yaw,-1.5*np.pi,places=3)
        walker.target_yaw=0
        walker.stop()
        yaw=walker.yaw
        for _ in range(100): walker.step()
        self.assertEqual(walker.yaw,yaw)

    def test_assisted_walker_reaches_destination_without_moving_objects(self):
        model,data,config,_=build_scene('warm_living_room')
        control=Controller(model,data)
        motion=SomaMotion(ROOT/'vendor/soma-retargeter/assets/motions/csv/Neutral_walk_forward_002__A057.csv',model,data.qpos)
        walker=Walker(control,motion)
        walker.begin(Planner(config).plan([0,0],[0,-.6]))
        for _ in range(2500): walker.step();control.step()
        self.assertEqual(len(walker.path),0)
        np.testing.assert_allclose(data.qpos[:2],[0,-.6],atol=.035)
        np.testing.assert_allclose(data.body('bottle').xpos[:2],[.28,-.27],atol=.005)
        self.assertFalse(data.warning.number.any())

if __name__=='__main__': unittest.main()
