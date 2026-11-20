import base64
import copy
from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import expressive_ai as app


class ConversationTest(unittest.TestCase):
    def test_think_classifier_uses_current_text_and_only_false_or_low(self):
        from unittest.mock import Mock
        for content, expected in [('{"reasoning":true}', 'low'), ('{"reasoning":false}', False), ('{"reasoning":"true"}', False), ('{}', False), ('[]', False), ('invalid', False)]:
            chat = Mock(return_value=SimpleNamespace(message=SimpleNamespace(content=content)))
            with patch.dict('sys.modules', {'ollama': SimpleNamespace(chat=chat)}):
                self.assertEqual(app.classify_think_mode('Solve this puzzle'), expected)
            request = chat.call_args.kwargs
            self.assertEqual(len(request['messages']), 2)
            self.assertEqual(request['messages'][1], {'role': 'user', 'content': 'Solve this puzzle'})
            self.assertIs(request['think'], False)
            self.assertFalse(request['stream'])

    def test_think_classifier_failure_defaults_to_false(self):
        from unittest.mock import Mock
        with patch.dict('sys.modules', {'ollama': SimpleNamespace(chat=Mock(side_effect=RuntimeError('Unavailable')))}), self.assertLogs(level='WARNING'):
            self.assertIs(app.classify_think_mode('Hello'), False)

    def test_classifier_uses_only_current_text_and_strict_boolean(self):
        from unittest.mock import Mock
        for content, expected in [('{"addressed":true}', True), ('{"addressed":false}', False), ('{"addressed":"true"}', False), ('[]', False), ('invalid', False)]:
            chat = Mock(return_value=SimpleNamespace(message=SimpleNamespace(content=content)))
            with patch.dict('sys.modules', {'ollama': SimpleNamespace(chat=chat)}):
                self.assertEqual(app.is_addressed_to_robot('Hey Reachy, look here'), expected)
            request = chat.call_args.kwargs
            self.assertEqual(len(request['messages']), 2)
            self.assertEqual(request['messages'][1], {'role': 'user', 'content': 'Hey Reachy, look here'})
            self.assertIn("'Hey' followed by any name", request['messages'][0]['content'])
            self.assertFalse(request['think'])
            self.assertFalse(request['stream'])

    def test_fragmented_and_multiple_tag_blocks(self):
        speech, moves = Queue(), Queue()
        with patch('builtins.print'):
            tail = app.dispatch_buffer('<sentence>Hi', speech, moves)
            self.assertTrue(speech.empty())
            tail = app.dispatch_buffer(tail + '!</sentence><emotion>yes1</emotion><sentence>Next.</sentence><emo', speech, moves)
        self.assertEqual([speech.get_nowait(), speech.get_nowait()], ['Hi!', 'Next.'])
        self.assertEqual(moves.get_nowait(), 'yes1')
        self.assertEqual(tail, '<emo')

    def test_fresh_frame_is_attached_as_jpeg(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        frame[:, :, 2] = 255
        mini = SimpleNamespace(media=SimpleNamespace(get_frame=lambda: frame))
        message = app.camera_message(mini, 'What do you see?')
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(message['images'][0]), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(message['content'], 'What do you see?')
        self.assertGreater(decoded[0, 0, 2], 240)
        self.assertLess(decoded[0, 0, 0], 10)

    def test_unavailable_camera_sends_text_only(self):
        from unittest.mock import Mock
        for getter in [Mock(return_value=None), Mock(side_effect=RuntimeError('Camera offline'))]:
            with patch.object(app.time, 'monotonic', side_effect=[0, 3]), self.assertLogs(level='WARNING'):
                message = app.camera_message(None, 'Hello', getter)
            self.assertEqual(message, {'role': 'user', 'content': 'Hello'})

    def test_failed_camera_encoding_sends_text_only(self):
        for outcome in [{'return_value': (False, None)}, {'side_effect': cv2.error('Bad frame')}]:
            with patch.object(cv2, 'imencode', **outcome), self.assertLogs(level='WARNING'):
                message = app.camera_message(None, 'Hello', lambda: np.zeros((8, 8, 3), dtype=np.uint8))
            self.assertEqual(message, {'role': 'user', 'content': 'Hello'})

    def test_main_streams_vision_into_both_queues_and_shuts_down(self):
        speech, moves, requests = [], [], []
        thinking_events = []
        class SpyQueue(Queue):
            def put(self, item, *args, **kwargs):
                if isinstance(item, tuple) and item[0] == 'thoughtful2':
                    thinking_events.append(item[1])
                return super().put(item, *args, **kwargs)
        class Mini:
            media = SimpleNamespace(get_frame=lambda: np.zeros((8, 8, 3), dtype=np.uint8))
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def cancel_move(self): pass
        def chat(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            yield SimpleNamespace(message=SimpleNamespace(thinking='Thinking', content=''))
            self.assertEqual(len(thinking_events), 1)
            for thinking, content in [('', ''), (None, None), ('', ' '), ('Still thinking', '')]:
                yield SimpleNamespace(message=SimpleNamespace(thinking=thinking, content=content))
                self.assertFalse(thinking_events[0].is_set())
            yield SimpleNamespace(message=SimpleNamespace(thinking='', content='<sentence>Hello</sentence><emotion>yes1</emotion>'))
            self.assertTrue(thinking_events[0].is_set())
        def drain(queue, mini, resource, output):
            while True:
                item = queue.get()
                queue.task_done()
                if item is None: return
                output.append(item[0] if isinstance(item, tuple) else item)
        modules = {
            'kokoro': SimpleNamespace(KPipeline=lambda **kwargs: object()),
            'ollama': SimpleNamespace(chat=chat),
            'reachy_mini': SimpleNamespace(ReachyMini=lambda **kwargs: Mini()),
            'reachy_mini.motion.recorded_move': SimpleNamespace(RecordedMoves=lambda name: object()),
        }
        with patch.object(app, 'classify_think_mode', return_value='low'), patch.object(app, 'Queue', SpyQueue), patch.object(app, 'live_dashboard', side_effect=lambda mini, **kwargs: nullcontext(mini.media.get_frame)), patch.dict('sys.modules', modules), patch('builtins.input', side_effect=['Look around', EOFError]), patch('builtins.print'), patch.object(app, 'speech_generator', side_effect=lambda *args: drain(*args, speech)), patch.object(app, 'movement_generator', side_effect=lambda *args: drain(*args, moves)):
            app.main(["--text"])
        self.assertEqual(speech, ['Hello'])
        self.assertEqual(moves, ['thoughtful2', 'yes1', 'reset_pos'])
        self.assertEqual(requests[0]['model'], 'gemma4:e4b')
        self.assertTrue(requests[0]['stream'])
        self.assertEqual(requests[0]['think'], 'low')
        self.assertEqual(requests[0]['messages'][-1]['content'], 'Look around')
        self.assertTrue(requests[0]['messages'][-1]['images'])
        self.assertTrue(requests[0]['messages'][1]['images'])

    def test_voice_utterance_is_transcribed_and_sent_with_camera(self):
        self.run_voice_utterance(True)

    def test_unaddressed_voice_skips_camera_response_and_resumes_listening(self):
        self.run_voice_utterance(False)

    def test_voice_utterance_reaches_model_when_camera_fails(self):
        self.run_voice_utterance(True, camera_failed=True)

    def run_voice_utterance(self, accepted, camera_failed=False):
        from contextlib import contextmanager
        speech, moves, requests, asr_calls = [], [], [], []
        real_queue = Queue
        class UtteranceQueue(Queue):
            consumed = False
            def get(self, *args, **kwargs):
                if self.consumed:
                    assert gate[0].is_set(), 'Listening did not resume'
                    raise KeyboardInterrupt
                self.consumed = True
                return super().get(*args, **kwargs)
        class Mini:
            media = SimpleNamespace(get_frame=lambda: np.zeros((8, 8, 3), dtype=np.uint8))
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def cancel_move(self): pass
        gate = []
        @contextmanager
        def dashboard(mini, *, on_utterance, listening, **kwargs):
            gate.append(listening)
            on_utterance(np.ones(16000, dtype=np.float32), 123.5)
            self.assertFalse(listening.is_set())
            def get_frame():
                if camera_failed:
                    raise RuntimeError('Camera offline')
                return mini.media.get_frame()
            yield get_frame
        def transcribe(audio, **kwargs):
            self.assertFalse(gate[0].is_set())
            asr_calls.append((audio, kwargs))
            return SimpleNamespace(text=' What do you see? ')
        def chat(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            yield SimpleNamespace(message=SimpleNamespace(thinking='', content='<sentence>Hello</sentence><emotion>yes1</emotion>'))
        def drain(queue, mini, resource, output):
            while True:
                item = queue.get()
                queue.task_done()
                if item is None: return
                self.assertFalse(gate[0].is_set())
                output.append(item[0] if isinstance(item, tuple) else item)
        modules = {
            'kokoro': SimpleNamespace(KPipeline=lambda **kwargs: object()),
            'ollama': SimpleNamespace(chat=chat),
            'reachy_mini': SimpleNamespace(ReachyMini=lambda **kwargs: Mini()),
            'reachy_mini.motion.recorded_move': SimpleNamespace(RecordedMoves=lambda name: object()),
            'mlx_qwen3_asr': SimpleNamespace(transcribe=transcribe),
        }
        with patch.object(app, 'classify_think_mode', return_value=False) as think_classifier, patch.object(app, 'is_addressed_to_robot', return_value=accepted) as classify, patch.object(app, 'camera_message', wraps=app.camera_message) as camera, patch.dict('sys.modules', modules), patch.object(app, 'live_dashboard', dashboard), patch.object(app, 'Queue', side_effect=lambda maxsize=0: UtteranceQueue(maxsize) if maxsize else real_queue()), patch('builtins.print'), patch.object(app, 'speech_generator', side_effect=lambda *args: drain(*args, speech)), patch.object(app, 'movement_generator', side_effect=lambda *args: drain(*args, moves)):
            app.main([])
        classify.assert_called_once_with('What do you see?')
        if not accepted:
            think_classifier.assert_not_called()
            camera.assert_not_called()
            self.assertEqual(requests, [])
            self.assertEqual(speech, [])
            self.assertEqual(moves, [])
            return
        think_classifier.assert_called_once_with('What do you see?')
        self.assertIs(requests[0]['think'], False)
        self.assertEqual(asr_calls[0][1]['model'], 'Qwen/Qwen3-ASR-1.7B')
        self.assertEqual(asr_calls[0][1]['language'], 'English')
        self.assertEqual(asr_calls[0][0].shape, (16000,))
        self.assertEqual(requests[0]['messages'][-1]['content'], 'What do you see?')
        if camera_failed:
            self.assertNotIn('images', requests[0]['messages'][-1])
        else:
            self.assertTrue(requests[0]['messages'][-1]['images'])
        self.assertEqual(speech, ['Hello'])
        self.assertEqual(moves, ['yes1', 'reset_pos'])


    def test_thinking_motion_repeats_and_stops_on_event(self):
        from threading import Event
        stop = Event()
        evaluated, targets = [], []
        def evaluate(at):
            evaluated.append(at)
            return 'head', 'antennas', 'body'
        def target(**kwargs):
            targets.append(kwargs)
            if len(targets) == 3:
                stop.set()
        mini = SimpleNamespace(goto_target=lambda **kwargs: None, set_target=target)
        move = SimpleNamespace(duration=.02, evaluate=evaluate)
        with patch.object(app.time, 'monotonic', side_effect=[10, 10.005, 10.025, 10.045]):
            app.play_thinking_move(mini, move, stop)
        self.assertEqual(len(targets), 3)
        for at in evaluated[1:]:
            self.assertAlmostEqual(at, .005)

    def test_cancelled_queued_thinking_does_not_start(self):
        from threading import Event
        from unittest.mock import Mock
        stop = Event()
        stop.set()
        mini, move = Mock(), Mock()
        app.play_thinking_move(mini, move, stop)
        mini.goto_target.assert_not_called()
        mini.set_target.assert_not_called()
        move.evaluate.assert_not_called()
