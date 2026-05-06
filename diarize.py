#!/usr/bin/env python3
"""
Speaker diarization via pyannote.audio 3.1.

Requirements:
  1. pip install pyannote.audio>=3.1
  2. Accept model terms at huggingface.co/pyannote/speaker-diarization-3.1
     and huggingface.co/pyannote/segmentation-3.0
  3. Provide your HuggingFace token via the UI field or HF_TOKEN env var
"""

import os

_pipeline = None

_SPEAKER_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def speaker_label(pyannote_label: str) -> str:
    try:
        n = int(pyannote_label.split("_")[-1])
        letter = _SPEAKER_LABELS[n] if n < len(_SPEAKER_LABELS) else str(n + 1)
    except (ValueError, IndexError):
        letter = pyannote_label
    return f"[Speaker {letter}]"


def load_pipeline(token: str = ""):
    global _pipeline
    if _pipeline is not None:
        return _pipeline

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

    _pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )

    # Lower segmentation onset/offset so shorter/quieter speaker turns are detected.
    # Default is ~0.5 — 0.3 catches more speech at the cost of slightly more noise.
    try:
        _pipeline._segmentation.onset = 0.3
        _pipeline._segmentation.offset = 0.3
    except Exception:
        pass

    if torch.backends.mps.is_available():
        _pipeline.to(torch.device("mps"))
    else:
        _pipeline.to(torch.device("cpu"))

    _pipeline.segmentation_batch_size = 64
    _pipeline.embedding_batch_size = 64

    return _pipeline


def run_diarization(audio_path: str, token: str = "", num_speakers: int = 0) -> list[tuple[float, float, str]]:
    """Returns (start_sec, end_sec, speaker_label) tuples sorted by start time."""
    pipeline = load_pipeline(token)
    kwargs = {}
    if num_speakers > 0:
        kwargs["num_speakers"] = num_speakers
    result = pipeline(audio_path, **kwargs)
    annotation = (
        result.speaker_diarization if hasattr(result, "speaker_diarization")
        else result.diarization if hasattr(result, "diarization")
        else result
    )
    turns = [
        (turn.start, turn.end, spk)
        for turn, _, spk in annotation.itertracks(yield_label=True)
    ]
    return sorted(turns, key=lambda x: x[0])


def _speaker_at(t: float, turns: list[tuple[float, float, str]]) -> str:
    # collect all turns that contain t
    matches = [(end - start, spk) for start, end, spk in turns if start <= t <= end]
    if matches:
        # prefer the shortest (most specific) matching turn — handles overlapping speech
        return min(matches, key=lambda x: x[0])[1]
    if not turns:
        return "SPEAKER_00"
    return min(turns, key=lambda x: min(abs(x[0] - t), abs(x[1] - t)))[2]


def assign_speakers(
    segments: list[dict],
    turns: list[tuple[float, float, str]],
) -> list[dict]:
    if not turns:
        return segments

    # Flatten all Whisper words into one time-sorted list
    all_words: list[dict] = []
    for seg in segments:
        words = seg.get("words") or []
        if words:
            all_words.extend(words)
        else:
            all_words.append({"start": seg["start"], "end": seg["end"],
                               "word": seg["text"].strip()})
    all_words.sort(key=lambda w: w["start"])

    # Assign each word to its best (shortest) overlapping raw turn.
    # Fall back to nearest turn for words that land in a gap.
    labeled: list[tuple[str, dict]] = []
    for w in all_words:
        mid = (w["start"] + w["end"]) / 2
        best_spk, best_dur = None, float("inf")
        for ts, te, spk in turns:
            if ts <= mid <= te:
                dur = te - ts
                if dur < best_dur:
                    best_dur, best_spk = dur, spk
        if best_spk is None:
            best_spk = min(turns, key=lambda x: min(abs(x[0] - mid), abs(x[1] - mid)))[2]
        labeled.append((speaker_label(best_spk), w))

    # Group consecutive same-speaker words into one segment per run.
    # Also split when a different-speaker turn falls in the gap between two
    # consecutive same-speaker words — this handles both overlapping and
    # zero-duration interjections that capture no word midpoints.
    groups: list[tuple[str, list[dict]]] = []
    for spk, w in labeled:
        if not groups or groups[-1][0] != spk:
            groups.append((spk, [w]))
        else:
            prev_w = groups[-1][1][-1]
            gap_start, gap_end = prev_w["end"], w["start"]
            # Only look for a speaker-change split if the gap between the two
            # same-speaker words is at least 200 ms.  A shorter gap means the
            # words are essentially adjacent and any overlapping turn from the
            # other speaker is just simultaneous speech, not a real change.
            split = False
            if gap_end - gap_start >= 0.2:
                split = any(
                    speaker_label(ts_spk) != spk
                    and min(te, gap_end) - max(ts, gap_start) > 0.05
                    for ts, te, ts_spk in turns
                )
            if split:
                groups.append((spk, [w]))
            else:
                groups[-1][1].append(w)

    result: list[dict] = []
    for spk, words in groups:
        text = "".join(w.get("word", w.get("text", "")) for w in words).strip()
        if not text:
            continue
        result.append({
            "start":   words[0]["start"],
            "end":     words[-1]["end"],
            "text":    " " + text,
            "words":   words,
            "speaker": spk,
        })

    return result
