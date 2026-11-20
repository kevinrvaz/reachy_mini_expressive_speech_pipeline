"""Notebook-style Silero VAD and utterance capture, displayed in Rerun."""

import math
from collections import deque

import numpy as np
import rerun as rr

from .reachy_audio import log_audio_segment


def create_vad(
    score=None,
    reset=None,
    on_utterance=None,
    *,
    silence_seconds=1.5,
    max_utterance_seconds=60,
):
    if not math.isfinite(silence_seconds) or silence_seconds <= 0:
        raise ValueError("silence_seconds must be finite and positive")
    if (
        not math.isfinite(max_utterance_seconds)
        or max_utterance_seconds <= silence_seconds
    ):
        raise ValueError("max_utterance_seconds must exceed silence_seconds")
    if score is None:
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )

        def score(frame):
            with torch.inference_mode():
                return model(torch.from_numpy(frame), 16000).item()

        reset = model.reset_states
    return {
        "silence_seconds": silence_seconds,
        "max_utterance_seconds": max_utterance_seconds,
        "score": score,
        "reset": reset,
        "on_utterance": on_utterance,
        "buffer": np.empty(0, np.float32),
        "prefix": deque(maxlen=3),
        "captured": [],
        "active": False,
        "silent": 0,
        "timestamp": None,
        "expected": None,
        "start": None,
        "rate": None,
        "resampler": None,
    }


def finish_vad(state, rec, *, timeline="time", reason="shutdown"):
    if state["captured"]:
        samples = np.concatenate(state["captured"])
        log_audio_segment(
            rec,
            samples,
            16000,
            timestamp=state["start"],
            entity_path="vad/capture",
            timeline=timeline,
        )
        rec.set_time(timeline, timestamp=state["start"] + len(samples) / 16000)
        rec.log(
            "logs/vad",
            rr.TextLog(
                f"VAD capture ended: {reason} ({len(samples) / 16000:.2f}s)",
                level="INFO",
            ),
        )
        rec.log("vad/active", rr.Scalars(0))
        if state.get("on_utterance") is not None:
            state["on_utterance"](samples, state["start"])
    state["captured"] = []
    state["active"] = False
    state["silent"] = 0
    state["prefix"].clear()


def process_vad(state, rec, samples, rate, timestamp, *, timeline="time"):
    # A missing block or timestamp re-anchor is not evidence of end-of-speech.
    # Reset inference timing, but retain the utterance and wait for actual silence.
    if state["rate"] != rate or (
        state["expected"] is not None and abs(timestamp - state["expected"]) > 0.05
    ):
        state["silent"] = 0
        state["prefix"].clear()
        state["buffer"] = np.empty(0, np.float32)
        state["timestamp"] = timestamp
        state["rate"] = rate
        if rate != 16000:
            import av

            state["resampler"] = av.AudioResampler(
                format="fltp", layout="mono", rate=16000
            )
        if state["reset"] is not None:
            state["reset"]()
    state["expected"] = timestamp + len(samples) / rate
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    if rate != 16000:
        import av

        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(mono[None, :], dtype=np.float32),
            format="fltp",
            layout="mono",
        )
        frame.sample_rate = rate
        converted = state["resampler"].resample(frame)
        if not converted:
            return
        mono = np.concatenate([f.to_ndarray().reshape(-1) for f in converted])
    state["buffer"] = np.concatenate((state["buffer"], mono)).astype(np.float32)
    while len(state["buffer"]) >= 512:
        frame = state["buffer"][:512].copy()
        state["buffer"] = state["buffer"][512:]
        at = state["timestamp"]
        state["timestamp"] += 512 / 16000
        confidence = float(state["score"](frame))
        state["prefix"].append(frame)
        if not state["active"]:
            if confidence > 0.5:
                state["active"] = True
                state["silent"] = 0
                state["captured"] = list(state["prefix"])
                state["start"] = at - (len(state["prefix"]) - 1) * 512 / 16000
        else:
            state["captured"].append(frame)
            state["silent"] = 0 if confidence > 0.35 else state["silent"] + 512
        rec.set_time(timeline, timestamp=at)
        rec.log("vad/probability", rr.Scalars(confidence))
        rec.log("vad/active", rr.Scalars(int(state["active"])))
        rec.log("vad/start_threshold", rr.Scalars(0.5))
        rec.log("vad/stop_threshold", rr.Scalars(0.35))
        if state["active"]:
            if state["silent"] / 16000 > state["silence_seconds"]:
                finish_vad(state, rec, timeline=timeline, reason="silence")
            elif len(state["captured"]) * 512 >= state["max_utterance_seconds"] * 16000:
                finish_vad(state, rec, timeline=timeline, reason="maximum duration")
