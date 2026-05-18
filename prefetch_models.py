#!/usr/bin/env python3
"""
One-time online download of all models used at transcription time.
After this, set HF_HUB_OFFLINE=1 in .env for air-gapped use.

  python prefetch_models.py
  python prefetch_models.py --diarize
  python prefetch_models.py --whisper large-v3,turbo-4bit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
_env = _ROOT / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _prefetch_whisper(model_keys: list[str]) -> None:
    from transcribe import MODEL_MAP_FASTER, get_backend

    if get_backend() != "faster":
        print("Whisper backend is not faster-whisper; skipping Whisper prefetch.", file=sys.stderr)
        return

    from faster_whisper import WhisperModel

    device = os.environ.get("FASTER_WHISPER_DEVICE", "cpu")
    for key in model_keys:
        model_id = MODEL_MAP_FASTER.get(key, key)
        compute = os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "int8")
        print(f"Downloading faster-whisper: {model_id} ({device}, {compute})")
        WhisperModel(model_id, device=device, compute_type=compute)


def _prefetch_text_fix() -> None:
    from text_fix import default_model, prefetch

    print(f"Downloading typo-fix model: {default_model()}")
    prefetch()


def _prefetch_diarize() -> None:
    from diarize import load_pipeline

    print("Downloading pyannote speaker-diarization-3.1 (needs HF_TOKEN + accepted terms)")
    load_pipeline()


def main() -> None:
    p = argparse.ArgumentParser(description="Download models for offline Arabic STT")
    p.add_argument(
        "--whisper",
        default="large-v3-4bit",
        help="Comma-separated UI model keys (default: large-v3-4bit)",
    )
    p.add_argument("--diarize", action="store_true", help="Also download pyannote diarization")
    p.add_argument("--no-text-fix", action="store_true", help="Skip local typo-fix LLM")
    args = p.parse_args()

    keys = [k.strip() for k in args.whisper.split(",") if k.strip()]
    _prefetch_whisper(keys)
    if not args.no_text_fix:
        _prefetch_text_fix()
    if args.diarize:
        _prefetch_diarize()

    print("\nDone. Copy project + HF cache (or HF_HOME folder) to the offline PC.")
    print("Then set HF_HUB_OFFLINE=1 in .env and run: python app.py")


if __name__ == "__main__":
    main()
