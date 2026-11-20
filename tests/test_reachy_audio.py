"""Validate audio format, sample continuity, timestamps and SDK cleanup."""
import io
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np
import rerun as rr
from expressive_ai.rerun.reachy_audio import log_audio_segment, stream_audio


class Recording:
    def __init__(self):
        self.assets = []
        self.amplitudes = []
        self.timestamp = None

    def send_columns(self, path, *, indexes, columns):
        self.amplitudes.append((path, indexes[0].timeline,
                                indexes[0].as_arrow_array().cast('int64').to_numpy(),
                                np.array(list(columns)[0].as_arrow_array().to_pylist()).reshape(-1)))

    def set_time(self, timeline, *, timestamp):
        self.timestamp = timestamp

    def log(self, path, value):
        if isinstance(value, rr.AssetAudio):
            content = bytes(value.blob.as_arrow_array().to_pylist()[0])
            with wave.open(io.BytesIO(content)) as wav:
                self.assets.append((self.timestamp, wav.getframerate(), wav.getnchannels(),
                                    np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').copy()))


class AudioTest(unittest.TestCase):
    def test_wav_channels_clipping_and_timestamp(self):
        rec = Recording()
        log_audio_segment(rec, [[-2, 2], [-.5, .5]], 48000, timestamp=123.5)
        timestamp, rate, channels, pcm = rec.assets[0]
        self.assertEqual((timestamp, rate, channels), (123.5, 48000, 2))
        np.testing.assert_array_equal(pcm, [-32767, 32767, -16384, 16384])
        with self.assertRaises(ValueError):
            log_audio_segment(rec, [float('nan')], 48000, timestamp=0)

    def test_variable_blocks_and_partial_tail_preserve_samples(self):
        rec, stop = Recording(), Event()
        data = np.linspace(-.8, .8, 58, dtype=np.float32).reshape(29, 2)
        blocks = iter([data[:7], data[7:25], data[25:]])
        calls = []
        def read():
            block = next(blocks, None)
            if block is None:
                stop.set()
            return block
        media = SimpleNamespace(get_input_audio_samplerate=lambda: 1000,
                                get_input_channels=lambda: 2,
                                start_recording=lambda: calls.append('start'),
                                stop_recording=lambda: calls.append('stop'),
                                get_audio_sample=read)
        with patch('expressive_ai.rerun.reachy_audio.time.time', return_value=100):
            stream_audio(SimpleNamespace(media=media), rec, stop=stop, segment_seconds=.02)
        self.assertEqual(calls, ['start', 'stop'])
        self.assertEqual([len(a[3]) // 2 for a in rec.assets], [20, 9])
        self.assertAlmostEqual(rec.assets[0][0], 99.993)
        self.assertAlmostEqual(rec.assets[1][0] - rec.assets[0][0], .02)
        np.testing.assert_array_equal(np.concatenate([a[3] for a in rec.assets]),
                                      np.rint(data.flatten() * 32767).astype('<i2'))

    def test_capture_error_still_stops_microphone(self):
        stopped = []
        def read():
            raise OSError('Disconnected')
        media = SimpleNamespace(get_input_audio_samplerate=lambda: 48000,
                                get_input_channels=lambda: 1,
                                start_recording=lambda: None,
                                stop_recording=lambda: stopped.append(True),
                                get_audio_sample=read)
        with self.assertRaises(OSError):
            stream_audio(SimpleNamespace(media=media), Recording())
        self.assertEqual(stopped, [True])


    def test_amplitude_samples_have_individual_timestamps(self):
        rec = Recording()
        samples = np.array([[0, .75], [-.25, -.5], [.5, 0]], dtype=np.float32)
        log_audio_segment(rec, samples, 1000, timestamp=123.5,
                          entity_path='robot/mic', timeline='capture')
        self.assertEqual(len(rec.amplitudes), 2)
        for channel, (path, timeline, times, values) in enumerate(rec.amplitudes):
            self.assertEqual(path, f'robot/mic/amplitude/channel_{channel}')
            self.assertEqual(timeline, 'capture')
            np.testing.assert_array_equal(times, [123500000000, 123501000000, 123502000000])
            np.testing.assert_array_equal(values, samples[:, channel])

    def test_amplitude_precedes_vad_and_survives_callback_failure(self):
        rec, stop = Recording(), Event()
        blocks = iter([np.ones((10, 1), np.float32), np.zeros((10, 1), np.float32)])
        observed = []
        def read():
            block = next(blocks, None)
            if block is None:
                stop.set()
            return block
        def vad(*args):
            observed.append(len(rec.amplitudes))
            self.assertEqual(rec.assets, [])  # no complete WAV segment yet
            raise RuntimeError('VAD failed')
        media = SimpleNamespace(get_input_audio_samplerate=lambda: 1000,
                                get_input_channels=lambda: 1,
                                start_recording=lambda: None, stop_recording=lambda: None,
                                get_audio_sample=read)
        with self.assertLogs('expressive_ai.rerun.reachy_audio', level='ERROR'):
            stream_audio(SimpleNamespace(media=media), rec, stop=stop, on_samples=vad)
        self.assertEqual(observed, [1, 2])
        # Tail WAV flush must not duplicate already logged amplitude samples.
        self.assertEqual(sum(len(row[3]) for row in rec.amplitudes), 20)
        self.assertEqual(len(rec.assets), 1)

    def test_dropped_samples_do_not_drift_out_of_live_window(self):
        rec, stop = Recording(), Event()
        received = [100.0]
        count = [0]
        def read():
            count[0] += 1
            if count[0] > 70:
                stop.set()
                return None
            received[0] += .1  # only 10 ms of samples per 100 ms: upstream loss
            return np.zeros((10, 1), np.float32)
        media = SimpleNamespace(get_input_audio_samplerate=lambda: 1000,
                                get_input_channels=lambda: 1,
                                start_recording=lambda: None, stop_recording=lambda: None,
                                get_audio_sample=read)
        with patch('expressive_ai.rerun.reachy_audio.time.time', side_effect=lambda: received[0]), patch('expressive_ai.rerun.reachy_audio.LOGGER.warning'):
            stream_audio(SimpleNamespace(media=media), rec, stop=stop)
        latest = rec.amplitudes[-1][2][-1] / 1e9
        self.assertLess(received[0] - latest, .3)
        self.assertEqual(sum(len(row[3]) for row in rec.amplitudes), 700)
