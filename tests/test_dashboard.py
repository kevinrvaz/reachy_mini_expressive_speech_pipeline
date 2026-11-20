from pathlib import Path
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import rerun as rr

from expressive_ai.rerun.dashboard import live_dashboard
from expressive_ai.rerun.reachy_vad import create_vad, process_vad


class Capture:
    def __init__(self):
        self.probabilities = []
        self.assets = []
    def set_time(self, *args, **kwargs): pass
    def send_columns(self, *args, **kwargs): pass
    def log(self, path, value):
        if path == 'vad/probability':
            self.probabilities.append(value)
        if isinstance(value, rr.AssetAudio):
            self.assets.append(value)


class DashboardTest(unittest.TestCase):
    def test_notebook_thresholds_and_silence_capture(self):
        scores = iter([.1, .2, .7, .4] + [.1] * 26)
        utterances = []
        state = create_vad(score=lambda frame: next(scores), silence_seconds=.8, on_utterance=lambda audio, at: utterances.append((audio, at)))
        rec = Capture()
        # Exercise arbitrary SDK block boundaries: every 512-sample window must
        # be consumed, including multiple windows within a single input block.
        samples = np.zeros(30 * 512, np.float32)
        process_vad(state, rec, samples[:1000], 16000, 10)
        process_vad(state, rec, samples[1000:], 16000, 10 + 1000 / 16000)
        self.assertEqual(len(rec.probabilities), 30)
        self.assertEqual(len(rec.assets), 1)
        self.assertEqual(len(utterances), 1)
        self.assertEqual(utterances[0][0].shape, (30 * 512,))
        self.assertAlmostEqual(utterances[0][1], 10)
        self.assertFalse(state['active'])
        import io, wave
        with wave.open(io.BytesIO(bytes(rec.assets[0].blob.as_arrow_array().to_pylist()[0]))) as audio:
            self.assertEqual(audio.getnframes(), 30 * 512)  # no duplicated trigger frame

    def test_resample_and_gap_reset(self):
        seen, resets = [], []
        state = create_vad(score=lambda frame: seen.append(frame.shape) or .1,
                           reset=lambda: resets.append(True))
        rec = Capture()
        process_vad(state, rec, np.zeros((4800, 2), np.float32), 48000, 20)
        self.assertEqual(seen, [(512,)] * 3)
        process_vad(state, rec, np.zeros((4800, 2), np.float32), 48000, 25)
        self.assertEqual(len(resets), 2)

    def test_one_camera_and_microphone_worker_and_cleanup(self):
        ready, camera_done, audio_done, motor_done = Event(), Event(), Event(), Event()
        frame = np.zeros((8, 8, 3), np.uint8)
        def camera(mini, rec, *, stop, on_frame):
            on_frame(frame, 123)
            ready.set()
            stop.wait(2)
            camera_done.set()
        def audio(mini, rec, *, stop, on_samples):
            on_samples(np.zeros((512, 2), np.float32), 16000, 123)
            stop.wait(2)
            audio_done.set()
        def motor(mini, robot, stop):
            stop.wait(2)
            motor_done.set()
        with tempfile.TemporaryDirectory() as d, patch('expressive_ai.rerun.dashboard.stream_camera', side_effect=camera) as c, patch('expressive_ai.rerun.dashboard.stream_audio', side_effect=audio) as a, patch('expressive_ai.rerun.dashboard.capture_motors', side_effect=motor):
            with live_dashboard(SimpleNamespace(), spawn=False, save=Path(d)/'dashboard.rrd', vad=create_vad(score=lambda frame: .1)) as get_frame:
                self.assertTrue(ready.wait(1))
                np.testing.assert_array_equal(get_frame(), frame)
                returned = get_frame()
                returned[:] = 255
                np.testing.assert_array_equal(get_frame(), frame)
            self.assertTrue(camera_done.is_set() and audio_done.is_set() and motor_done.is_set())
            self.assertEqual(c.call_count, 1)
            self.assertEqual(a.call_count, 1)

    def test_paused_microphone_does_not_requeue_robot_speech(self):
        listening, processed = Event(), Event()
        utterances = []
        state = create_vad(score=lambda frame: .9)
        # A partially collected utterance must be discarded on pause.
        state['captured'] = [np.ones(512, np.float32)]
        state['active'] = True
        def audio(mini, rec, *, stop, on_samples):
            on_samples(np.ones((512, 2), np.float32), 16000, 100)
            processed.set()
            stop.wait(2)
        def idle(*args, stop, **kwargs):
            stop.wait(2)
        def motor(mini, robot, stop):
            stop.wait(2)
        with patch('expressive_ai.rerun.dashboard.stream_camera', side_effect=idle), patch('expressive_ai.rerun.dashboard.stream_audio', side_effect=audio), patch('expressive_ai.rerun.dashboard.capture_motors', side_effect=motor):
            with live_dashboard(SimpleNamespace(), spawn=False, vad=state, listening=listening,
                                on_utterance=lambda samples, at: utterances.append(samples)):
                self.assertTrue(processed.wait(1))
                self.assertFalse(state['active'])
                self.assertEqual(state['captured'], [])
        self.assertEqual(utterances, [])

    def test_short_pause_does_not_end_utterance_and_speech_restarts_timer(self):
        utterances = []
        confidence = [.9]
        state = create_vad(score=lambda frame: confidence[0],
                           on_utterance=lambda samples, at: utterances.append(samples))
        rec, cursor = Capture(), [0]
        def feed(frames, score):
            confidence[0] = score
            process_vad(state, rec, np.zeros(frames * 512, np.float32), 16000, cursor[0] / 16000)
            cursor[0] += frames * 512
        feed(1, .9)
        feed(31, .1)  # ~1 second pause used to close the capture
        self.assertTrue(state['active'])
        self.assertEqual(utterances, [])
        feed(1, .9)
        self.assertEqual(state['silent'], 0)
        feed(46, .1)
        self.assertEqual(utterances, [])
        feed(1, .1)  # more than 1.5 seconds of actual low-confidence audio
        self.assertEqual(len(utterances), 1)
        self.assertEqual(len(utterances[0]), 80 * 512)

    def test_timestamp_correction_does_not_submit_partial_speech(self):
        utterances = []
        state = create_vad(score=lambda frame: .9,
                           on_utterance=lambda samples, at: utterances.append(samples))
        rec = Capture()
        process_vad(state, rec, np.ones(512, np.float32), 16000, 100)
        process_vad(state, rec, np.ones(512, np.float32), 16000, 100.5)
        self.assertTrue(state['active'])
        self.assertEqual(len(state['captured']), 2)
        self.assertEqual(utterances, [])
        self.assertEqual(rec.assets, [])
