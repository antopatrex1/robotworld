import struct
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import viser

from robot_world.realsense import DEFAULT_SOCKET, RealSensePanel, decode_response, depth_preview, latest_frame


class RealSenseTests(unittest.TestCase):
    def test_offline_response_and_invalid_messages(self):
        message = struct.pack('<Q', 7) + b'offline' + b'\0'
        self.assertEqual(decode_response(message), ('offline', None))
        for invalid in (b'', message[:-1], message + b'extra', message[:-1] + b'\2'):
            with self.assertRaises(ValueError):
                decode_response(invalid)

    def test_native_helper_interoperability(self):
        binary = DEFAULT_SOCKET.parents[1] / 'target/debug/realsense-studio'
        if not binary.exists():
            self.skipTest('Build realsense first')
        with tempfile.TemporaryDirectory(prefix='rw-rs-', dir='/tmp') as directory:
            path = Path(directory) / 'camera.sock'
            helper = subprocess.Popen([str(binary), '--socket', str(path), 'camera', '--demo'], stderr=subprocess.PIPE)
            server = None
            panel = None
            def wait_for(condition):
                deadline = time.monotonic() + 5
                while not condition() and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue(condition())
            try:
                deadline = time.monotonic() + 5
                frame = None
                while time.monotonic() < deadline:
                    if helper.poll() is not None:
                        self.fail(helper.stderr.read().decode())
                    if path.exists():
                        _, frame = latest_frame(path)
                        if frame is not None:
                            break
                    time.sleep(.05)
                self.assertIsNotNone(frame)
                self.assertTrue(frame.demo)
                self.assertEqual(frame.serial, 'SYNTHETIC-DEMO')
                self.assertEqual(frame.color.shape, (480, 640, 3))
                self.assertEqual(frame.depth.shape, (480, 640))
                self.assertAlmostEqual(frame.depth_scale_m, .001)
                np.testing.assert_array_equal(frame.color[0, 0], [0, 0, 125])
                self.assertEqual(frame.depth[0, 0], 600)
                frame.depth = frame.depth.copy()
                frame.depth[0, :3] = [0, 200, 4000]
                preview = depth_preview(frame)
                np.testing.assert_array_equal(preview[0, :3], [[0, 0, 0], [255, 85, 55], [45, 90, 210]])
                time.sleep(.1)
                _, next_frame = latest_frame(path)
                self.assertGreater(next_frame.sequence, frame.sequence)
                server = viser.ViserServer(host='127.0.0.1', port=0)
                panel = RealSensePanel(server, path)
                wait_for(lambda: panel.color.visible)
                self.assertIn('DEMO', panel.status.content)
                panel.enabled.value = False
                wait_for(lambda: not panel.color.visible and 'paused' in panel.status.content)
                panel.enabled.value = True
                wait_for(lambda: panel.color.visible)
                helper.terminate()
                helper.wait(timeout=5)
                helper.stderr.close()
                wait_for(lambda: not panel.color.visible and 'disconnected' in panel.status.content)
                helper = subprocess.Popen([str(binary), '--socket', str(path), 'camera', '--demo'], stderr=subprocess.PIPE)
                wait_for(lambda: panel.color.visible and 'DEMO' in panel.status.content)
            finally:
                if panel is not None:
                    panel.close()
                if server is not None:
                    server.stop()
                helper.terminate()
                helper.wait(timeout=5)
                helper.stderr.close()
            with self.assertRaises(OSError):
                latest_frame(path)


if __name__ == '__main__':
    unittest.main()
