"""Integration contracts for embedding the robot in a shared dashboard."""
import io
import json
import logging
import threading
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np
import rerun as rr
from expressive_ai.rerun import ReachyMiniView


STATE = {'head_joints': [0] * 7, 'antennas_position': [0, 0],
         'head_pose': {'m': np.eye(4).ravel().tolist()}}


class Recording:
    def __init__(self):
        self.events = []
        self.times = []
        self.links = set()
        self.received = threading.Event()

    def log_file_from_path(self, path, **kwargs):
        self.links.update(n.get('name') for n in ET.parse(path).getroot().findall('link'))

    def log(self, path, *values, **kwargs):
        self.events.append((path, values))
        if path.endswith('transforms/rod_6'):
            self.received.set()

    def set_time(self, name, **kwargs):
        self.times.append((name, kwargs))

    def flush(self):
        pass


class ReachyViewTest(unittest.TestCase):
    def test_scopes_frames_and_shared_timestamps(self):
        rec = Recording()
        with ReachyMiniView(rec, prefix='robots/left', timeline='capture') as left, ReachyMiniView(rec, prefix='robots/right') as right:
            left.log_state(STATE, timestamp=123.5)
            self.assertEqual(rec.times[-1], ('capture', {'timestamp': 123.5}))
            right.log_state(STATE, timestamp=124.5)
            for path, values in rec.events:
                prefix = 'robots/left/' if path.startswith('robots/left/') else 'robots/right/'
                self.assertTrue(path.startswith(prefix))
                for value in values:
                    if isinstance(value, rr.Transform3D):
                        parent = value.parent_frame.as_arrow_array().to_pylist()[0]
                        child = value.child_frame.as_arrow_array().to_pylist()[0]
                        self.assertTrue(parent.startswith(prefix))
                        self.assertTrue(child.startswith(prefix))
                        self.assertIn(parent, rec.links)
                        self.assertTrue(child in rec.links or child.endswith('/environment/floor'))
            handler = left.logging_handler()
            handler.emit(logging.LogRecord('speech', logging.INFO, '', 0, 'Speaking', (), None))
            self.assertEqual(rec.events[-1][0], 'robots/left/logs/speech')
            handler.close()
        with self.assertRaises(RuntimeError):
            left.log_state(STATE)
        left.close()  # idempotent cleanup

    def test_background_polling_stops_and_can_restart(self):
        rec = Recording()
        with patch('expressive_ai.rerun.reachy_view.urlopen', side_effect=lambda *a, **k: io.BytesIO(json.dumps(STATE).encode())) as request:
            with ReachyMiniView(rec) as view:
                view.start(hz=100)
                self.assertTrue(rec.received.wait(2))
                view.stop()
                self.assertIsNone(view._thread)
                self.assertIn('use_pose_matrix=true', request.call_args.args[0])
                rec.received.clear()
                view.start(hz=100)
                self.assertTrue(rec.received.wait(2))
            self.assertIsNone(view._thread)
