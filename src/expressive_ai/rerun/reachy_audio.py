"""Capture Reachy Mini's microphone as short WAV assets in a Rerun AudioView. see https://rerun.io/docs/concepts/logging-and-ingestion/audio"""

import argparse
import io
import logging
import math
import time
import wave
from pathlib import Path
from threading import Event

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from expressive_ai.rerun.reachy_view import forward_logs

LOGGER = logging.getLogger(__name__)


def _samples(samples):
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[1] == 0 or not np.isfinite(data).all():
        raise ValueError("Audio must be finite float samples shaped (frames, channels)")
    return data


def log_audio_segment(
    recording,
    samples,
    sample_rate,
    *,
    timestamp,
    entity_path="audio/microphone",
    timeline="time",
    log_amplitude=True,
):
    """Log float PCM as a PCM16 WAV at the segment's start in Unix seconds."""
    if sample_rate <= 0 or int(sample_rate) != sample_rate:
        raise ValueError("sample_rate must be a positive integer")
    if not math.isfinite(timestamp):
        raise ValueError("timestamp must be finite Unix seconds")
    data = _samples(samples)
    if not len(data):
        return False
    pcm = np.rint(np.clip(data, -1, 1) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(data.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    recording.set_time(timeline, timestamp=timestamp)
    recording.log(
        entity_path, rr.AssetAudio(contents=buffer.getvalue(), media_type="audio/wav")
    )
    if log_amplitude:
        log_audio_amplitude(recording, data, sample_rate, timestamp=timestamp,
                            entity_path=entity_path, timeline=timeline)
    for channel in range(data.shape[1]):
        level = float(np.sqrt(np.mean(data[:, channel].astype(np.float64) ** 2)))
        recording.log(f"{entity_path}/levels/channel_{channel}", rr.Scalars(level))
    return True


def log_audio_amplitude(recording, samples, sample_rate, *, timestamp,
                        entity_path="audio/microphone", timeline="time"):
    """Log each received block independently of WAV batching and VAD."""
    data = _samples(samples)
    sample_times = (
        round(timestamp * 1_000_000_000)
        + np.rint(np.arange(len(data)) * (1_000_000_000 / sample_rate)).astype(np.int64)
    ).astype("datetime64[ns]")
    index = rr.TimeColumn(timeline, timestamp=sample_times)
    for channel in range(data.shape[1]):
        recording.send_columns(
            f"{entity_path}/amplitude/channel_{channel}",
            indexes=[index],
            columns=rr.Scalars.columns(scalars=data[:, channel]),
        )

def audio_blueprint(entity_path="audio/microphone", *, volume=0.5, timeline="time"):
    """Return a composable waveform/playback and level-meter layout."""
    path = "/" + entity_path.strip("/")
    return rrb.Vertical(
        rrb.AudioView(
            name="Reachy microphone",
            origin=path,
            contents=[path],
            playback=rrb.AudioPlayback(volume=volume),
        ),
        rrb.TimeSeriesView(
            name="Microphone amplitude over time",
            origin=f"{path}/amplitude",
            time_ranges=rrb.VisibleTimeRange(
                timeline,
                start=rrb.TimeRangeBoundary.cursor_relative(seconds=-5),
                end=rrb.TimeRangeBoundary.cursor_relative(seconds=0),
            ),
        ),
        rrb.TimeSeriesView(name="Microphone RMS levels", origin=f"{path}/levels"),
        row_shares=[2, 3, 1],
    )


def stream_audio(
    mini,
    recording,
    *,
    stop=None,
    segment_seconds=0.25,
    entity_path="audio/microphone",
    timeline="time",
    manage_recording=True,
    on_samples=None,
):
    """Read sequential PCM blocks, preserving every sample and partial segments.

    Run in a caller-owned thread when composing with video. Uses SDK sample rate
    and channel metadata. The first received block anchors an approximate clock;
    subsequent segments advance by sample count. Long input gaps reset that clock.
    Set manage_recording=False when your application already owns mic capture.
    """
    if not math.isfinite(segment_seconds) or not 0.02 <= segment_seconds <= 5:
        raise ValueError("segment_seconds must be between 0.02 and 5")
    stop = stop if stop is not None else Event()
    media = mini.media
    rate = media.get_input_audio_samplerate()
    channels = media.get_input_channels()
    if rate <= 0 or channels <= 0:
        raise RuntimeError("Reachy microphone is unavailable; check the media backend")
    target = max(1, round(rate * segment_seconds))
    pending = []
    count = 0
    segment_start = None
    last_received = None
    last_warning = time.monotonic()
    callback_warning = -math.inf

    def flush():
        nonlocal pending, count, segment_start
        if count:
            log_audio_segment(
                recording,
                np.concatenate(pending),
                rate,
                timestamp=segment_start,
                entity_path=entity_path,
                timeline=timeline,
                log_amplitude=False,
            )
            segment_start += count / rate
            pending, count = [], 0

    if manage_recording:
        media.start_recording()
    LOGGER.info(
        "Listening to Reachy microphone: %s Hz, %s channels, %.0f ms segments.",
        rate,
        channels,
        segment_seconds * 1000,
    )
    try:
        while not stop.is_set():
            block = media.get_audio_sample()
            now = time.monotonic()
            if block is None or len(block) == 0:
                if now - last_warning >= 5:
                    LOGGER.warning(
                        "No microphone samples received; check the media connection."
                    )
                    last_warning = now
                stop.wait(0.005)
                continue
            data = _samples(block)
            if data.shape[1] != channels:
                raise ValueError(
                    f"Expected {channels} audio channels, got {data.shape[1]}"
                )
            received_at = time.time()
            clock_lag = (0 if segment_start is None else
                         received_at - (segment_start + (count + len(data)) / rate))
            if segment_start is None or clock_lag > max(.25, len(data) / rate * 2) or (
                last_received is not None
                and now - last_received > max(0.5, len(data) / rate + 0.25)
            ):
                flush()
                segment_start = received_at - len(data) / rate
                if last_received is not None:
                    LOGGER.warning(
                        "Microphone gap or timestamp drift; restarting the audio timeline anchor."
                    )
            block_start = segment_start + count / rate
            log_audio_amplitude(recording, data, rate, timestamp=block_start,
                                entity_path=entity_path, timeline=timeline)
            if on_samples is not None:
                try:
                    on_samples(data, rate, block_start)
                except Exception:
                    if now - callback_warning >= 5:
                        LOGGER.exception("Audio processing callback failed; microphone logging continues")
                        callback_warning = now
            last_received = last_warning = now
            offset = 0
            while offset < len(data):
                size = min(target - count, len(data) - offset)
                pending.append(data[offset : offset + size].copy())
                count += size
                offset += size
                if count == target:
                    flush()
    finally:
        try:
            flush()
        finally:
            if manage_recording:
                media.stop_recording()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connection-mode",
        choices=["auto", "localhost_only", "network"],
        default="network",
    )
    parser.add_argument("--media-backend", default="webrtc")
    parser.add_argument("--robot-name", default="reachy_mini")
    parser.add_argument("--segment-ms", type=float, default=250)
    parser.add_argument("--volume", type=float, default=0.5)
    parser.add_argument(
        "--save", type=Path, help="Save microphone audio and levels to an .rrd file"
    )
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.segment_ms) or not 20 <= args.segment_ms <= 5000:
        parser.error("--segment-ms must be between 20 and 5000")
    if not math.isfinite(args.volume) or not 0 <= args.volume <= 1:
        parser.error("--volume must be between 0 and 1")
    if args.no_viewer and not args.save:
        parser.error("--no-viewer requires --save")

    from reachy_mini import ReachyMini

    logging.basicConfig(level=logging.INFO)
    with rr.RecordingStream("reachy_audio_capture") as rec:
        sinks = []
        if not args.no_viewer:
            rec.spawn(connect=False, memory_limit="1GB")
            sinks.append(rr.GrpcSink())
        if args.save:
            sinks.append(rr.FileSink(args.save))
        rec.set_sinks(*sinks)
        rec.send_blueprint(
            rrb.Vertical(
                audio_blueprint(volume=args.volume),
                rrb.TextLogView(name="Audio logs", origin="/logs"),
                row_shares=[4, 1],
            )
        )
        with forward_logs(rec):
            try:
                with ReachyMini(
                    robot_name=args.robot_name,
                    media_backend=args.media_backend,
                    connection_mode=args.connection_mode,
                ) as mini:
                    stream_audio(mini, rec, segment_seconds=args.segment_ms / 1000)
            except KeyboardInterrupt:
                LOGGER.info("Audio capture stopped.")
            finally:
                rec.flush()


if __name__ == "__main__":
    main()
