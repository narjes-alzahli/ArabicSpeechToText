#!/usr/bin/env python3
"""
Speaker diarization via pyannote.audio 3.1.

Requirements:
  1. pip install pyannote.audio>=3.1
  2. Accept model terms at huggingface.co/pyannote/speaker-diarization-3.1
     and huggingface.co/pyannote/segmentation-3.0
  3. Provide your HuggingFace token via the UI field or HF_TOKEN env var
"""

from __future__ import annotations

import os
from pathlib import Path

_pipeline = None

_SPEAKER_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def speaker_label(pyannote_label: str) -> str:
    try:
        n = int(pyannote_label.split("_")[-1])
        letter = _SPEAKER_LABELS[n] if n < len(_SPEAKER_LABELS) else str(n + 1)
    except (ValueError, IndexError):
        letter = pyannote_label
    return f"[{letter} متحدث]"


def load_pipeline(token: str = ""):
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import sys
    import types

    if "__main__" not in sys.modules:
        sys.modules["__main__"] = types.ModuleType("__main__")

    from pyannote.audio import Pipeline
    import torch

    token = token.strip() or os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "A HuggingFace token is required for speaker diarization.\n"
            "Paste your token in the HF token field, or set the HF_TOKEN env var.\n"
            "Get a free token at huggingface.co/settings/tokens and accept terms at\n"
            "huggingface.co/pyannote/speaker-diarization-3.1"
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )

    cluster_threshold = _env_float("DIARIZATION_CLUSTERING_THRESHOLD", 0.7)
    try:
        pipeline.instantiate({"clustering": {"threshold": cluster_threshold}})
    except Exception:
        pass

    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    else:
        pipeline.to(torch.device("cpu"))

    pipeline.segmentation_batch_size = 8
    pipeline.embedding_batch_size = 8

    _pipeline = pipeline
    return _pipeline


def _load_audio_for_pyannote(audio_path: str) -> dict:
    """
    Load WAV in-memory for pyannote. On Windows, torchcodec often fails to load
    (missing FFmpeg DLLs), so file paths break diarization unless we pass waveform.
    """
    import soundfile as sf
    import torch

    path = Path(audio_path)
    if not path.is_file():
        raise RuntimeError(f"Diarization audio not found: {path}")

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    waveform = torch.from_numpy(data.T.copy())
    if waveform.ndim != 2:
        raise RuntimeError("Unexpected audio shape from soundfile.")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return {
        "waveform": waveform,
        "sample_rate": int(sr),
        "uri": path.stem,
    }


def smooth_turns(
    turns: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Merge tiny gaps between same speaker; drop very short blips."""
    if not turns:
        return []

    # Wider default match “diarize-first then Whisper per turn” (merge adjacent same-speaker gaps).
    merge_gap = _env_float("DIARIZATION_MERGE_GAP", 1.5)
    min_duration = _env_float("DIARIZATION_MIN_TURN", 0.5)

    ordered = sorted(turns, key=lambda x: x[0])
    merged: list[tuple[float, float, str]] = []
    for start, end, spk in ordered:
        if end <= start:
            continue
        if merged and spk == merged[-1][2] and start - merged[-1][1] <= merge_gap:
            prev_s, _, prev_spk = merged[-1]
            merged[-1] = (prev_s, max(merged[-1][1], end), prev_spk)
        else:
            merged.append((start, end, spk))

    return [(s, e, spk) for s, e, spk in merged if e - s >= min_duration]


def run_diarization(
    audio_path: str,
    token: str = "",
    num_speakers: int = 0,
    min_speakers: int = 0,
    max_speakers: int = 0,
) -> list[tuple[float, float, str]]:
    """Returns (start_sec, end_sec, pyannote_speaker_id) sorted by start time."""
    pipeline = load_pipeline(token)
    kwargs: dict = {}
    if num_speakers > 0:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers > 0:
            kwargs["min_speakers"] = min_speakers
        if max_speakers > 0:
            kwargs["max_speakers"] = max_speakers

    audio_input = _load_audio_for_pyannote(audio_path)
    result = pipeline(audio_input, **kwargs)
    annotation = (
        result.speaker_diarization
        if hasattr(result, "speaker_diarization")
        else result.diarization
        if hasattr(result, "diarization")
        else result
    )
    turns = [
        (turn.start, turn.end, spk)
        for turn, _, spk in annotation.itertracks(yield_label=True)
    ]
    return smooth_turns(turns)


def _overlap(start: float, end: float, ts: float, te: float) -> float:
    return max(0.0, min(end, te) - max(start, ts))


def _best_pyannote_speaker(start: float, end: float, turns: list[tuple[float, float, str]]) -> str:
    if not turns:
        return "SPEAKER_00"

    scores: dict[str, float] = {}
    for ts, te, spk in turns:
        amount = _overlap(start, end, ts, te)
        if amount > 0:
            scores[spk] = scores.get(spk, 0.0) + amount

    if scores:
        return max(scores, key=scores.get)

    mid = (start + end) / 2
    return min(turns, key=lambda x: min(abs(x[0] - mid), abs(x[1] - mid)))[2]


def assign_speakers(
    segments: list[dict],
    turns: list[tuple[float, float, str]],
) -> list[dict]:
    """Tag each Whisper segment with the speaker who talked most during that time."""
    if not turns:
        return segments

    out: list[dict] = []
    for seg in segments:
        item = dict(seg)
        pyannote_spk = _best_pyannote_speaker(float(seg["start"]), float(seg["end"]), turns)
        item["speaker"] = speaker_label(pyannote_spk)
        out.append(item)
    return out
