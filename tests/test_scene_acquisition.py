import copy
import json
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

try:
    import cv2
except ImportError:
    cv2 = None
import mujoco
import numpy as np
import viser

from robot_world.scene import ROOT, build_scene
from robot_world.control import Controller
from robot_world.commands import parse_command
from robot_world.realsense import RealSensePanel
from robot_world.scene_acquisition import find_paper, validate_layout, track_paper
from robot_world.table_alignment import homography, project
from robot_world.web_app import Workbench


class SceneAcquisitionTests(unittest.TestCase):
    def reference(self):
        return json.loads((ROOT/'configs/calibration/paper_table.json').read_text())

    def layout(self):
        ref=self.reference()
        ref['objects']=[{'id':'mug','kind':'mug','label':'Observed black mug',
                         'base_pixel':[397,224],'color_rgb':[15,15,15]}]
        for item in ref['objects']:
            item['table_xy_m']=project(homography(ref['paper_pixels'],ref['paper_size_m']),[item['base_pixel']])[0].tolist()
        return ref

    @unittest.skipIf(cv2 is None, "Install requirements-scene.txt for paper detection")
    def test_paper_is_redetected_and_missing_paper_is_rejected(self):
        image=np.full((480,640,3),[125,105,85],dtype=np.uint8)
        moved=np.array([[65,216],[258,215],[290,238],[59,242]],np.int32)
        cv2.fillConvexPoly(image,moved,(245,245,245))
        found=find_paper(image,self.reference())
        np.testing.assert_allclose(found,moved,atol=3)
        with self.assertRaisesRegex(ValueError,'reference paper'):
            find_paper(np.full_like(image,100),self.reference())

    @unittest.skipIf(cv2 is None, "Install requirements-scene.txt for paper detection")
    def test_corner_tracking_handles_shift_and_rejects_missing_sheet(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = self.reference()
            old = np.clip(np.full((480,640,3), [125,105,85]) + np.random.default_rng(7).integers(-15,16,(480,640,1)),0,255).astype(np.uint8)
            corners = np.asarray(reference['paper_pixels'], np.int32)
            cv2.fillConvexPoly(old, corners, (235,240,245))
            cv2.imwrite(str(Path(directory)/'color.png'), cv2.cvtColor(old, cv2.COLOR_RGB2BGR))
            reference['source_capture'] = directory
            shifted = cv2.warpAffine(old, np.float32([[1,0,24],[0,1,-5]]), (640,480))
            found = track_paper(shifted, reference)
            self.assertIsNotNone(found)
            np.testing.assert_allclose(found, corners+[24,-5], atol=1)
            self.assertIsNone(track_paper(np.full_like(old,100), reference))

    def test_duplicate_click_is_ignored_and_retry_is_possible(self):
        server=viser.ViserServer(host='127.0.0.1',port=0)
        callback=Mock()
        panel=RealSensePanel(server,Path('/tmp/missing-test-rs.sock'),callback)
        try:
            panel.request_acquisition();panel.request_acquisition()
            callback.assert_called_once()
            self.assertTrue(panel.acquire_button.disabled)
            panel.finish_acquisition('Camera disconnected')
            self.assertFalse(panel.acquire_button.disabled)
            panel.request_acquisition()
            self.assertEqual(callback.call_count,2)
        finally:
            panel.close();server.stop()

    def test_overlap_and_off_table_results_are_rejected(self):
        layout=self.layout()
        validate_layout(layout)
        other=copy.deepcopy(layout['objects'][0]);other['id']='mug_2'
        layout['objects'].append(other)
        with self.assertRaisesRegex(ValueError,'too close'):validate_layout(layout)
        layout=self.layout();layout['objects'][0]['base_pixel']=[0,0]
        with self.assertRaisesRegex(ValueError,'within the virtual table'):validate_layout(layout)

    def test_acquisition_failure_keeps_previous_scene(self):
        app=Workbench.__new__(Workbench)
        app.scene_generation=3;app.acquired_layout=None;app.commands=queue.Queue()
        app.model=object();original=app.model
        app.acquirer=SimpleNamespace(acquire=Mock(side_effect=ValueError('paper missing')))
        app.realsense=SimpleNamespace(path='/tmp/test.sock',acquire_status=SimpleNamespace(content=''))
        app.start_acquisition()
        kind,message=app.commands.get(timeout=5)
        self.assertEqual(kind,'acquisition_error')
        self.assertIn('Scene unchanged',message)
        self.assertIs(app.model,original)

    def test_scene_change_discards_inflight_acquisition(self):
        app=Workbench.__new__(Workbench)
        app.scene_generation=4;app.load=Mock()
        app.realsense=SimpleNamespace(finish_acquisition=Mock())
        app.apply_acquisition((3,(Path('/tmp/uncommitted.json'),{},None)))
        app.load.assert_not_called()
        app.realsense.finish_acquisition.assert_called_once()

    def test_dynamic_and_empty_layouts_build_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'layout.json'
            layout=self.layout()
            second=copy.deepcopy(layout['objects'][0])
            second.update(id='mug_2',label='Observed white mug',base_pixel=[530,265],color_rgb=[230,230,230])
            second['table_xy_m']=project(homography(layout['paper_pixels'],layout['paper_size_m']),[second['base_pixel']])[0].tolist()
            layout['objects'].append(second)
            validate_layout(layout)
            path.write_text(json.dumps(layout))
            m,d,c,_=build_scene('camera_table',camera_layout=path,output_path=Path(directory)/'scene.xml')
            self.assertGreaterEqual(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'mug_2'),0)
            self.assertEqual(parse_command('pick up the observed white mug',c['object_labels']).values,('mug_2',))
            controller=Controller(m,d)
            for _ in range(100):controller.step()
            self.assertFalse(d.warning.number.any())
            controller.reset()
            layout['objects']=[];path.write_text(json.dumps(layout))
            m,d,c,_=build_scene('camera_table',camera_layout=path,output_path=Path(directory)/'empty.xml')
            self.assertEqual(c['object_labels'],{})
            self.assertIsNone(Controller(m,d).metrics()['object_position'])
            self.assertEqual(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'mug'),-1)


if __name__=='__main__':unittest.main()
