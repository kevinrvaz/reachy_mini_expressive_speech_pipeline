"""Camera colors and continuous frame logging without robot hardware."""
import io
from threading import Event
from types import SimpleNamespace
import unittest
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
from expressive_ai.rerun.reachy_camera import log_camera_frame, stream_camera


class Recording:
    def __init__(self):
        self.frames = []
        self.times = []

    def set_time(self, name, **kwargs):
        self.times.append((name, kwargs))

    def log(self, path, value):
        self.frames.append((path, value))


class CameraTest(unittest.TestCase):
    def test_bgr_colors_and_capture_timestamp(self):
        rec = Recording()
        frame = np.zeros((16, 24, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # SDK red in BGR order
        self.assertTrue(log_camera_frame(rec, frame, timestamp=123.5, timeline='capture'))
        self.assertEqual(rec.times, [('capture', {'timestamp': 123.5})])
        encoded = bytes(rec.frames[0][1].blob.as_arrow_array().to_pylist()[0])
        decoded = np.array(Image.open(io.BytesIO(encoded)))
        self.assertEqual(decoded.shape, frame.shape)
        self.assertGreater(decoded[0, 0, 0], 240)
        self.assertLess(decoded[0, 0, 2], 10)
        self.assertFalse(log_camera_frame(rec, None))
        self.assertEqual(len(rec.frames), 1)

    def test_stream_continues_past_old_encoder_flush_boundary(self):
        rec, stop = Recording(), Event()
        count = 0
        def get_frame():
            nonlocal count
            count += 1
            if count == 1:
                return None
            return np.zeros((8, 12, 3), dtype=np.uint8)
        mini = SimpleNamespace(media=SimpleNamespace(get_frame=get_frame))
        original_log = rec.log
        def log(path, value):
            original_log(path, value)
            if len(rec.frames) == 65:
                stop.set()
        rec.log = log
        stream_camera(mini, rec, stop=stop, fps=10000)
        self.assertEqual(len(rec.frames), 65)


    def test_slow_logger_skips_old_frames(self):
        stop, produced, logged = Event(), [], []
        def get_frame():
            value = len(produced)
            produced.append(value)
            return np.full((4, 4, 3), value, dtype=np.int32)
        def slow_log(rec, frame, **kwargs):
            logged.append(int(frame[0, 0, 0]))
            if len(logged) == 2:
                stop.set()
            else:
                time.sleep(0.05)
        mini = SimpleNamespace(media=SimpleNamespace(get_frame=get_frame))
        with patch('expressive_ai.rerun.reachy_camera.log_camera_frame', side_effect=slow_log):
            stream_camera(mini, Recording(), stop=stop, fps=1000)
        self.assertEqual(len(logged), 2)
        self.assertGreater(logged[1] - logged[0], 1)
        self.assertLessEqual(produced[-1] - logged[-1], 2)

    def test_preview_resizes_without_upscaling(self):
        rec = Recording()
        log_camera_frame(rec, np.zeros((108, 192, 3), dtype=np.uint8), max_width=96)
        encoded = bytes(rec.frames[0][1].blob.as_arrow_array().to_pylist()[0])
        self.assertEqual(Image.open(io.BytesIO(encoded)).size, (96, 54))

    def test_reader_failure_propagates(self):
        def get_frame():
            raise OSError('Disconnected')
        mini = SimpleNamespace(media=SimpleNamespace(get_frame=get_frame))
        with self.assertRaisesRegex(RuntimeError, 'Camera capture failed'):
            stream_camera(mini, Recording())
