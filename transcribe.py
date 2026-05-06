#!/usr/bin/env python3
"""
Arabic STT - post-processing transcription using mlx-whisper (Apple Silicon optimized).

Usage:
    python transcribe.py meeting.mp4
    python transcribe.py meeting.mp4 --format srt
    python transcribe.py meeting.mp4 --format json
    python transcribe.py meeting.mp4 --model large-v3 --format txt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx_whisper


MODEL_MAP = {
    "large-v3-4bit": "mlx-community/whisper-large-v3-mlx-4bit",   # recommended: fast + accurate
    "turbo-4bit":    "mlx-community/whisper-large-v3-turbo-q4",    # recommended for live
    "large-v3":      "mlx-community/whisper-large-v3-mlx",         # max accuracy, slow
    "turbo":         "mlx-community/whisper-large-v3-turbo",
    "medium":        "mlx-community/whisper-medium-mlx",
    "small":         "mlx-community/whisper-small-mlx",
}

DEFAULT_MODEL = "large-v3"

# Shared Whisper kwargs — used by both CLI and UI
WHISPER_FULL_KWARGS = dict(
    language="ar",
    word_timestamps=True,
    verbose=False,
    no_speech_threshold=0.6,
    logprob_threshold=-3.0,
    compression_ratio_threshold=3.0,
    condition_on_previous_text=True,
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    initial_prompt="هذا تسجيل لاجتماع باللغة العربية.",
)

WHISPER_LIVE_KWARGS = dict(
    language="ar",
    word_timestamps=False,
    verbose=False,
    no_speech_threshold=0.9,
    logprob_threshold=-2.0,
    condition_on_previous_text=True,
    temperature=(0.0, 0.4, 0.8),
    initial_prompt="هذا كلام باللغة العربية.",
)


def fmt_ts(seconds: float) -> str:
    """HH:MM:SS — used for display and live timestamps."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_ts_ms(seconds: float) -> str:
    """HH:MM:SS.mmm — millisecond precision for saved files."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def format_timestamp(seconds: float) -> str:
    """HH:MM:SS,mmm — used for SRT/TXT file output."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_txt(segments) -> str:
    lines = []
    for seg in segments:
        ts = f"[{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}]"
        lines.append(f"{ts}\n{seg['text'].strip()}\n")
    return "\n".join(lines)


def to_srt(segments) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"])
        end   = format_timestamp(seg["end"])
        text  = seg["text"].strip()
        blocks.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_json(segments, audio_path: str, model: str) -> str:
    out = {
        "file": str(audio_path),
        "model": model,
        "language": "ar",
        "segments": [
            {
                "id": i,
                "start": round(seg["start"], 3),
                "end":   round(seg["end"],   3),
                "text":  seg["text"].strip(),
            }
            for i, seg in enumerate(segments)
        ],
    }
    return json.dumps(out, ensure_ascii=False, indent=2)


def transcribe(audio_path: Path, model_key: str, fmt: str) -> str:
    repo = MODEL_MAP[model_key]

    print(f"Model  : {model_key} ({repo})")
    print(f"File   : {audio_path}")
    print(f"Output : {fmt.upper()}")
    print()
    print("Loading model (first run downloads weights ~3 GB, cached after)...")

    t0 = time.time()

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=repo,
        **WHISPER_FULL_KWARGS,
    )

    elapsed = time.time() - t0
    segments = result.get("segments", [])

    if not segments:
        print("Warning: no speech detected.", file=sys.stderr)
        return ""

    duration = segments[-1]["end"] if segments else 0
    speed    = duration / elapsed if elapsed > 0 else 0

    print(f"Done in {elapsed:.1f}s  |  audio: {duration/60:.1f} min  |  {speed:.1f}x real-time")
    print(f"Segments: {len(segments)}")
    print()

    if fmt == "srt":
        return to_srt(segments)
    elif fmt == "json":
        return to_json(segments, audio_path, model_key)
    else:
        return to_txt(segments)


def main():
    parser = argparse.ArgumentParser(description="Transcribe Arabic audio/video to text.")
    parser.add_argument("audio", help="Path to audio or video file")
    parser.add_argument(
        "--model",
        choices=list(MODEL_MAP.keys()),
        default=DEFAULT_MODEL,
        help=f"Whisper model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "srt", "json"],
        default="txt",
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: same name as input with new extension)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    transcript = transcribe(audio_path, args.model, args.format)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = audio_path.with_suffix(f".{args.format}")

    out_path.write_text(transcript, encoding="utf-8")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
