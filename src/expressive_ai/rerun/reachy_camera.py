"""Show Reachy Mini's live camera in Rerun, or embed it in another recording. See https://rerun.io/docs/concepts/logging-and-ingestion/video"""

import argparse
import logging
import math
import time
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread

import cv2
import rerun as rr
import rerun.blueprint as rrb

from expressive_ai.rerun.reachy_view import forward_logs

LOGGER = logging.getLogger(__name__)


def log_camera_frame(
    recording,
    frame,
    *,
    timestamp=None,
    entity_path="camera/image",
    timeline="time",
    jpeg_quality=75,
    max_width=960,
):
    """Log one SDK BGR frame at its received time (Unix seconds)."""
    if frame is None:
        return False
    if max_width < 0:
        raise ValueError("max_width must be nonnegative (0 means original resolution)")
    if max_width and frame.shape[1] > max_width:
        height = max(1, round(frame.shape[0] * max_width / frame.shape[1]))
        frame = cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("Camera JPEG encoding failed")
    image = rr.EncodedImage(contents=encoded.tobytes(), media_type="image/jpeg")
    recording.set_time(
        timeline, timestamp=time.time() if timestamp is None else timestamp
    )
    recording.log(entity_path, image)
    return True


def stream_camera(
    mini,
    recording,
    *,
    stop=None,
    fps=20,
    jpeg_quality=75,
    max_width=960,
    entity_path="camera/image",
    timeline="time",
    on_frame=None,
):
    """Drain capture independently and log only the newest waiting frame.

    The caller owns the SDK connection and recording. A single-slot queue bounds
    application backlog; older unlogged frames are intentionally discarded.
    SDK reads must return before the capture thread can finish during shutdown.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    if max_width < 0:
        raise ValueError("max_width must be nonnegative")
    stop = stop if stop is not None else Event()
    finished = Event()
    pending = Queue(maxsize=1)
    errors = []

    def capture():
        try:
            while not stop.is_set() and not finished.is_set():
                frame = mini.media.get_frame()
                if frame is None:
                    finished.wait(0.005)
                    continue
                # Own the buffer in case an SDK backend reuses its storage.
                sample = (frame.copy(), time.time())
                if on_frame is not None:
                    on_frame(*sample)
                try:
                    pending.put_nowait(sample)
                except Full:
                    try:
                        pending.get_nowait()
                    except Empty:
                        pass
                    pending.put_nowait(sample)
                # Avoid spinning on backends that return cached frames instantly.
                finished.wait(0.001)
        except Exception as error:
            errors.append(error)

    reader = Thread(target=capture, name="reachy-camera-capture", daemon=True)
    reader.start()
    last_warning = time.monotonic()
    receiving = False
    LOGGER.info("Waiting for Reachy Mini camera frames...")
    try:
        while not stop.is_set():
            if errors:
                raise RuntimeError("Camera capture failed") from errors[0]
            started = time.monotonic()
            try:
                frame, received_at = pending.get(timeout=0.1)
            except Empty:
                if started - last_warning >= 5:
                    LOGGER.warning(
                        "No camera frames received; check the daemon's media connection."
                    )
                    last_warning = started
                continue
            last_warning = started
            if not receiving:
                LOGGER.info(
                    "Live camera connected (%s × %s), preview capped at %s px / %s FPS.",
                    frame.shape[1],
                    frame.shape[0],
                    max_width or "native",
                    fps,
                )
                receiving = True
            log_camera_frame(
                recording,
                frame,
                timestamp=received_at,
                entity_path=entity_path,
                timeline=timeline,
                jpeg_quality=jpeg_quality,
                max_width=max_width,
            )
            stop.wait(max(0, 1 / fps - (time.monotonic() - started)))
    finally:
        finished.set()
        reader.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connection-mode",
        choices=["auto", "localhost_only", "network"],
        default="network",
    )
    parser.add_argument(
        "--media-backend",
        default="webrtc",
        help="Reachy SDK media backend (default: webrtc)",
    )
    parser.add_argument("--robot-name", default="reachy_mini")
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument(
        "--max-width",
        type=int,
        default=960,
        help="Preview width cap; 0 preserves native resolution",
    )
    parser.add_argument(
        "--save", type=Path, help="Also save camera frames to an .rrd recording"
    )
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.max_width < 0:
        parser.error("--max-width must be nonnegative")
    if args.no_viewer and not args.save:
        parser.error("--no-viewer requires --save")

    from reachy_mini import ReachyMini

    logging.basicConfig(level=logging.INFO)
    with rr.RecordingStream("reachy_camera_capture") as recording:
        if not args.no_viewer:
            recording.spawn(memory_limit="1GB")
        if args.save:
            recording.save(args.save)
        recording.send_blueprint(
            rrb.Vertical(
                rrb.Spatial2DView(name="Live Reachy camera", origin="/camera"),
                rrb.TextLogView(name="Camera logs", origin="/logs"),
                row_shares=[4, 1],
            )
        )
        with forward_logs(recording):
            try:
                with ReachyMini(
                    robot_name=args.robot_name,
                    media_backend=args.media_backend,
                    connection_mode=args.connection_mode,
                ) as mini:
                    stream_camera(
                        mini,
                        recording,
                        fps=args.fps,
                        jpeg_quality=args.jpeg_quality,
                        max_width=args.max_width,
                    )
            except KeyboardInterrupt:
                LOGGER.info("Camera viewer stopped.")
            finally:
                recording.flush()


if __name__ == "__main__":
    main()
