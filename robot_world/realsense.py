"""Read-only client for realsense/src/ipc.rs (bincode 1 fixed-int encoding)."""
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_MESSAGE = 8 * 1024 * 1024
DEFAULT_SOCKET = Path(__file__).resolve().parents[1] / 'realsense/.runtime/camera.sock'


@dataclass
class Frame:
    sequence: int
    captured_unix_ms: int
    color: np.ndarray
    depth: np.ndarray
    depth_scale_m: float
    serial: str
    demo: bool


def decode_response(payload):
    """Decode bounded native frames without opening the USB camera."""
    if len(payload) > MAX_MESSAGE:
        raise ValueError('Camera message too large')
    offset = 0

    def take(size):
        nonlocal offset
        if size < 0 or offset + size > len(payload):
            raise ValueError('Truncated camera message')
        value = payload[offset:offset + size]
        offset += size
        return value

    def number(fmt):
        return struct.unpack('<' + fmt, take(struct.calcsize('<' + fmt)))[0]

    def string():
        return take(number('Q')).decode('utf-8')

    status = string()
    present = number('B')
    frame = None
    if present == 1:
        sequence, captured = number('Q'), number('Q')
        width, height = number('I'), number('I')
        if not (0 < width <= 1920 and 0 < height <= 1080):
            raise ValueError('Invalid camera dimensions')
        count = width * height
        if number('Q') != count * 3:
            raise ValueError('Invalid RGB buffer length')
        color = np.frombuffer(take(count * 3), dtype=np.uint8).reshape(height, width, 3)
        if number('Q') != count:
            raise ValueError('Invalid depth buffer length')
        depth = np.frombuffer(take(count * 2), dtype='<u2').reshape(height, width)
        scale = number('f')
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid depth scale')
        take(32)  # Two f64 device timestamps and two u64 frame numbers.
        serial = string()
        demo = number('B')
        if demo not in (0, 1):
            raise ValueError('Invalid demo flag')
        frame = Frame(sequence, captured, color, depth, scale, serial, bool(demo))
    elif present != 0:
        raise ValueError('Invalid frame option')
    if offset != len(payload):
        raise ValueError('Unexpected camera message trailing bytes')
    return status, frame


def latest_frame(path=DEFAULT_SOCKET):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(1.0)
        connection.connect(str(path))
        connection.sendall(struct.pack('<II', 4, 1))  # Request::Latest

        def receive(size):
            parts = bytearray()
            while len(parts) < size:
                chunk = connection.recv(size - len(parts))
                if not chunk:
                    raise ValueError('Camera connection closed mid-frame')
                parts.extend(chunk)
            return bytes(parts)

        size = struct.unpack('<I', receive(4))[0]
        if size > MAX_MESSAGE:
            raise ValueError('Camera message too large')
        return decode_response(receive(size))


def depth_preview(frame):
    """Match the native viewer's fixed 0.2–4 m palette; invalid pixels black."""
    stops = np.array([[255, 85, 55], [255, 208, 80], [45, 205, 180], [45, 90, 210]], dtype=np.float32)
    position = np.clip((frame.depth.astype(np.float32) * frame.depth_scale_m - .2) / 3.8, 0, 1) * 3
    index = np.minimum(position.astype(int), 2)
    fraction = (position - index)[..., None]
    pixels = (stops[index] * (1 - fraction) + stops[index + 1] * fraction).astype(np.uint8)
    pixels[frame.depth == 0] = 0
    return pixels


class RealSensePanel:
    def __init__(self, server, path=DEFAULT_SOCKET, on_acquire=None):
        self.server = server
        self.path = path
        self.stop = threading.Event()
        self.acquire_lock = threading.Lock()
        self.on_acquire = on_acquire
        with server.gui.add_folder('RealSense · live camera', expand_by_default=True):
            self.acquire_button = server.gui.add_button('Acquire Scene', color='teal', disabled=on_acquire is None)
            self.acquire_status = server.gui.add_markdown('Keep the 210 × 147 mm paper flat at the table corner. Acquire Scene takes a new photo, replaces the table objects, and resets the simulated robot.')
            self.acquire_button.on_click(lambda _: self.request_acquisition())
            self.enabled = server.gui.add_checkbox('Show live camera', True)
            self.status = server.gui.add_markdown('Connecting to RealSense…')
            empty = np.zeros((480, 640, 3), dtype=np.uint8)
            self.color = server.gui.add_image(empty, label='Color', format='jpeg', jpeg_quality=80, visible=False)
            self.depth = server.gui.add_image(empty, label='Depth · 0.2–4 m', format='jpeg', jpeg_quality=80, visible=False)
            self.detail = server.gui.add_markdown('Start **realsense/Start RealSense.command** to connect.')
            server.gui.add_markdown('Live physical camera · depth is not aligned to color.\n\nRobot World remains a separate simulated scene.')
        self.thread = threading.Thread(target=self._run, name='realsense-preview', daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)

    def request_acquisition(self):
        if self.on_acquire is not None and self.acquire_lock.acquire(blocking=False):
            self.acquire_button.disabled = True
            self.acquire_status.content = 'Capturing a new scene…'
            try:
                self.on_acquire()
            except Exception:
                self.finish_acquisition('Scene acquisition could not start. Please retry.')
                raise

    def finish_acquisition(self, message):
        self.acquire_status.content = message
        self.acquire_button.disabled = False
        if self.acquire_lock.locked():
            self.acquire_lock.release()

    def _run(self):
        previous = None
        last_new = 0
        while not self.stop.is_set():
            delay = .1  # Limit previews to 10 Hz; helper still streams at 30 Hz.
            try:
                if not self.enabled.value:
                    self.color.visible = self.depth.visible = False
                    self.status.content = 'Camera preview paused.'
                    previous = None
                    self.stop.wait(.2)
                    continue
                status, frame = latest_frame(self.path)
                identity = (frame.captured_unix_ms, frame.sequence) if frame else None
                if frame is not None and identity != previous:
                    last_new = time.monotonic()
                if frame is None or time.monotonic() - last_new > 2:
                    self.color.visible = self.depth.visible = False
                    self.status.content = status if frame is None else 'Camera stalled; waiting for new frames.'
                    self.detail.content = 'Waiting for live frames. Check the RealSense window and camera connection.'
                    if frame is None:
                        previous = None
                elif identity != previous:
                    with self.server.atomic():
                        self.color.image = frame.color
                        self.depth.image = depth_preview(frame)
                        self.color.visible = self.depth.visible = True
                        self.status.content = ('**DEMO · synthetic**' if frame.demo else '**LIVE · RealSense D435**') + f' · frame {frame.sequence}'
                        center = frame.depth[frame.depth.shape[0] // 2, frame.depth.shape[1] // 2]
                        distance = f'{center * frame.depth_scale_m:.3f} m' if center else 'no return'
                        self.detail.content = f'{frame.color.shape[1]} × {frame.color.shape[0]} · center depth: {distance}'
                    previous = identity
            except (OSError, ValueError) as error:
                self.color.visible = self.depth.visible = False
                self.status.content = 'RealSense disconnected · reconnecting…'
                self.detail.content = 'Start **realsense/Start RealSense.command**. ' + str(error)
                previous = None
                delay = 1
            self.stop.wait(delay)
