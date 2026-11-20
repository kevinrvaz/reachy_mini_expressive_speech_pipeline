"""Functions composing the existing robot, camera, audio and Silero VAD flows."""
import logging
import time
from contextlib import contextmanager
from threading import Event, Lock, Thread

import rerun as rr
import rerun.blueprint as rrb

from .reachy_audio import audio_blueprint, stream_audio
from .reachy_camera import stream_camera
from .reachy_vad import create_vad, finish_vad, process_vad
from .reachy_view import ReachyMiniView, forward_logs

LOGGER = logging.getLogger(__name__)


def dashboard_blueprint(robot, timeline="time"):
    return rrb.Vertical(
        rrb.Horizontal(
            robot.blueprint(logs=False),
            rrb.Spatial2DView(name="Live camera", origin="/camera"),
        ),
        rrb.Horizontal(
            audio_blueprint(volume=0, timeline=timeline),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Silero VAD — speech probability", origin="/vad",
                                  contents=["/vad/probability", "/vad/active", "/vad/start_threshold", "/vad/stop_threshold", "/vad/listening"]),
                rrb.AudioView(name="VAD captured speech", origin="/vad/capture", contents=["/vad/capture"],
                              playback=rrb.AudioPlayback(volume=0)),
            ),
        ),
        rrb.TextLogView(name="Live logs", origin="/logs"),
        row_shares=[4, 4, 1],
    )


def capture_motors(mini, robot, stop):
    last_warning = -float('inf')
    while not stop.is_set():
        try:
            head, antennas = mini.get_current_joint_positions()
            pose = mini.get_current_head_pose()
            robot.log_state(dict(head_joints=head, antennas_position=antennas,
                                 head_pose={"m": pose.reshape(-1).tolist()}))
        except Exception:
            if time.monotonic() - last_warning > 5:
                LOGGER.exception("Motor telemetry unavailable")
                last_warning = time.monotonic()
        stop.wait(1 / 30)


@contextmanager
def live_dashboard(mini, *, spawn=True, save=None, vad=None, on_utterance=None, listening=None, vad_silence_seconds=1.5, max_utterance_seconds=60):
    """Share one robot connection and expose a recent frame getter for Gemma.

    Exiting stops all capture threads before the caller closes the robot. SDK
    reads must return for shutdown to complete. No transcription is performed.
    """
    vad = create_vad(silence_seconds=vad_silence_seconds, max_utterance_seconds=max_utterance_seconds) if vad is None else vad
    if on_utterance is not None:
        vad["on_utterance"] = on_utterance
    stop, lock = Event(), Lock()
    latest = [None, 0.0]
    threads = []

    def microphone_samples(samples, rate, timestamp):
        rec.set_time("time", timestamp=timestamp)
        rec.log("vad/listening", rr.Scalars(int(listening is None or listening.is_set())))
        if listening is not None and not listening.is_set():
            # Keep raw microphone logging, but discard speech captured during
            # ASR/response playback so the robot cannot enqueue its own reply.
            vad['captured'] = []
            vad['active'] = False
            vad['silent'] = 0
            vad['prefix'].clear()
            vad['rate'] = None
            rec.log("vad/active", rr.Scalars(0))
            return
        process_vad(vad, rec, samples, rate, timestamp)

    def remember_frame(frame, timestamp):
        with lock:
            latest[:] = [frame, time.monotonic()]

    def get_frame():
        with lock:
            if latest[0] is None or time.monotonic() - latest[1] > 1:
                return None
            return latest[0].copy()

    def run(name, target, *args, **kwargs):
        def worker():
            try:
                target(*args, **kwargs)
            except Exception:
                LOGGER.exception("%s capture stopped", name)
        thread = Thread(target=worker, name=name, daemon=True)
        thread.start()
        threads.append(thread)

    with rr.RecordingStream("expressive_ai_dashboard") as rec:
        sinks = []
        if spawn:
            rec.spawn(connect=False, memory_limit="1GB")
            sinks.append(rr.GrpcSink())
        if save:
            sinks.append(rr.FileSink(save))
        if sinks:
            rec.set_sinks(*sinks)
        with ReachyMiniView(rec) as robot, forward_logs(rec):
            rec.send_blueprint(dashboard_blueprint(robot))
            try:
                run("camera", stream_camera, mini, rec, stop=stop, on_frame=remember_frame)
                run("microphone", stream_audio, mini, rec, stop=stop,
                    on_samples=microphone_samples)
                run("motors", capture_motors, mini, robot, stop)
                yield get_frame
            finally:
                stop.set()
                for thread in threads:
                    thread.join()
                vad["on_utterance"] = None
                finish_vad(vad, rec)
                rec.flush()
