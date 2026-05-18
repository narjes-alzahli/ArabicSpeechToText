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
import os
import sys
import time
from pathlib import Path

_BACKEND = os.environ.get("WHISPER_BACKEND", "").strip().lower()  # "mlx" | "faster" | "openai" | ""


def _have_mlx() -> bool:
    try:
        import mlx_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _have_faster_whisper() -> bool:
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        return True
    except Exception:
        return False


def _faster_whisper_import_error() -> str | None:
    """If faster-whisper cannot be imported, return a short error string for diagnostics."""
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def _torch_import_error() -> str | None:
    try:
        import torch  # noqa: F401
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def _have_openai_whisper() -> bool:
    try:
        import whisper  # noqa: F401
        return True
    except Exception:
        return False


def describe_backend_failures() -> str:
    """Human-readable Markdown when no Whisper backend can start (for UI banner)."""
    lines = [
        "## Transcription is not ready on this PC\n",
        "Whisper needs native libraries that are missing or cannot load.\n\n",
    ]
    te = _torch_import_error()
    fe = _faster_whisper_import_error()
    if te:
        lines.append("**PyTorch** (used by the Whisper fallback here):\n")
        lines.append(f"```\n{te}\n```\n\n")
    if fe:
        lines.append("**faster-whisper / CTranslate2** (optional faster path):\n")
        lines.append(f"```\n{fe}\n```\n\n")
    lines.append(
        "**Fix (Windows):** install Microsoft **Visual C++ Redistributable for x64** from "
        "[this Microsoft download](https://aka.ms/vs/17/release/vc_redist.x64.exe), "
        "then **restart the machine** (or at least sign out and back in). "
        "Reopen the terminal, activate `.venv`, and run `python app.py` again.\n\n"
        "That same runtime is required for **PyTorch** and **CTranslate2**; without it, "
        "upload → Transcribe will keep failing with a generic error."
    )
    return "".join(lines)


def get_backend() -> str:
    """
    Select backend:
    - macOS/Apple Silicon: mlx-whisper (if installed)
    - otherwise: faster-whisper (if installed), else openai-whisper (CPU/GPU via PyTorch)
    Override with env: WHISPER_BACKEND=mlx|faster|openai
    """
    if _BACKEND == "mlx" and _have_mlx():
        return "mlx"
    if _BACKEND == "faster" and _have_faster_whisper():
        return "faster"
    if _BACKEND == "openai" and _have_openai_whisper():
        return "openai"
    if _BACKEND in ("mlx", "faster", "openai"):
        pass  # forced backend missing — fall through to auto
    if _have_mlx():
        return "mlx"
    if _have_faster_whisper():
        return "faster"
    if _have_openai_whisper():
        return "openai"
    return "none"


MODEL_MAP_MLX = {
    "large-v3-4bit": "mlx-community/whisper-large-v3-mlx-4bit",  # recommended: fast + accurate
    "turbo-4bit": "mlx-community/whisper-large-v3-turbo-q4",  # recommended for live
    "large-v3": "mlx-community/whisper-large-v3-mlx",  # max accuracy, slow
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
}

# faster-whisper model identifiers. "4bit" here maps to lower-precision compute,
# not an actual 4-bit weight format.
MODEL_MAP_FASTER = {
    "large-v3-4bit": "large-v3",
    "turbo-4bit": "large-v3-turbo",
    "large-v3": "large-v3",
    "turbo": "large-v3-turbo",
    "medium": "medium",
    "small": "small",
}


def get_model_map() -> dict:
    b = get_backend()
    return MODEL_MAP_MLX if b == "mlx" else MODEL_MAP_FASTER


MODEL_MAP = get_model_map()

DEFAULT_MODEL = "large-v3"

# Default text priming Whisper (initial_prompt) — UI hints are appended to these.
DEFAULT_INITIAL_PROMPT_FULL = "هذا تسجيل باللغة العربية"
DEFAULT_INITIAL_PROMPT_LIVE = "هذا كلام باللغة العربية."


def build_initial_prompt(user_hint: str | None = None, *, live: bool = False) -> str:
    """Merge optional user hints with the built-in Arabic priming sentence."""
    base = DEFAULT_INITIAL_PROMPT_LIVE if live else DEFAULT_INITIAL_PROMPT_FULL
    hint = (user_hint or "").strip()
    if not hint:
        return base
    return f"{base} {hint}"


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
    initial_prompt=DEFAULT_INITIAL_PROMPT_FULL,
)

WHISPER_LIVE_KWARGS = dict(
    language="ar",
    word_timestamps=False,
    verbose=False,
    no_speech_threshold=0.9,
    logprob_threshold=-2.0,
    condition_on_previous_text=True,
    temperature=(0.0, 0.4, 0.8),
    initial_prompt=DEFAULT_INITIAL_PROMPT_LIVE,
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
    model_map = get_model_map()
    backend = get_backend()
    if backend == "none":
        raise RuntimeError(
            "No Whisper backend installed. Install mlx-whisper (macOS Apple Silicon), "
            "faster-whisper (Windows/Linux; needs a working CTranslate2), or openai-whisper: "
            "pip install openai-whisper"
        )

    repo = model_map[model_key]

    print(f"Backend: {backend}")
    print(f"Model  : {model_key} ({repo})")
    print(f"File   : {audio_path}")
    print(f"Output : {fmt.upper()}")
    print()
    print("Loading model (first run downloads weights ~3 GB, cached after)...")

    t0 = time.time()

    result = transcribe_any(
        str(audio_path),
        model_key=model_key,
        live=False,
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


_fw_model = None
_fw_model_key = None


def _fw_compute_type_for_key(model_key: str) -> str:
    # good default on CPU Windows; if user has CUDA they can override via env.
    if model_key.endswith("-4bit"):
        return os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "int8")
    return os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "float16")


def _fw_device() -> str:
    return os.environ.get("FASTER_WHISPER_DEVICE", "cpu")


def _fw_model_for_key(model_key: str):
    global _fw_model, _fw_model_key
    model_id = MODEL_MAP_FASTER[model_key]
    compute_type = _fw_compute_type_for_key(model_key)
    device = _fw_device()

    # recreate model if key changes (compute type can differ)
    if _fw_model is not None and _fw_model_key == (model_id, device, compute_type):
        return _fw_model

    from faster_whisper import WhisperModel

    _fw_model = WhisperModel(
        model_id,
        device=device,
        compute_type=compute_type,
    )
    _fw_model_key = (model_id, device, compute_type)
    return _fw_model


_openai_model = None
_openai_model_id: str | None = None


def reset_whisper_caches() -> None:
    """Drop loaded faster-whisper / openai-whisper models (e.g. after UI model change)."""
    global _fw_model, _fw_model_key, _openai_model, _openai_model_id
    _fw_model = None
    _fw_model_key = None
    _openai_model = None
    _openai_model_id = None


def _openai_device() -> str:
    d = os.environ.get("OPENAI_WHISPER_DEVICE", "").strip()
    if d:
        return d
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _openai_model_for_key(model_key: str):
    """openai-whisper: same HF-style keys as faster-whisper map to load_model names."""
    global _openai_model, _openai_model_id
    model_id = MODEL_MAP_FASTER[model_key]
    device = _openai_device()
    if _openai_model is not None and _openai_model_id == f"{model_id}:{device}":
        return _openai_model

    import whisper

    _openai_model = whisper.load_model(model_id, device=device)
    _openai_model_id = f"{model_id}:{device}"
    return _openai_model


def transcribe_any(
    audio_path: str,
    model_key: str,
    live: bool,
    user_hint: str | None = None,
) -> dict:
    """
    Returns a dict compatible with mlx_whisper.transcribe output:
    { "segments": [ {start, end, text, words?}, ... ] }

    user_hint: optional extra text appended to the default Arabic initial_prompt.
    """
    backend = get_backend()
    kwargs = dict(WHISPER_LIVE_KWARGS if live else WHISPER_FULL_KWARGS)
    kwargs["initial_prompt"] = build_initial_prompt(user_hint, live=live)
    if backend == "none":
        te = _torch_import_error()
        fe = _faster_whisper_import_error()
        parts = ["No working Whisper backend is available.\n"]
        if te:
            parts.append(f"\nPyTorch: {te}")
        if fe:
            parts.append(f"\nfaster-whisper/CTranslate2: {fe}")
        parts.append(
            "\n\nOn Windows, install Visual C++ Redistributable (x64) from:\n"
            "  https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
            "then restart the PC and try again. (PyTorch and CTranslate2 both need it.)"
        )
        raise RuntimeError("".join(parts))
    if backend == "mlx":
        import mlx_whisper

        repo = MODEL_MAP_MLX[model_key]
        return mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=repo,
            **kwargs,
        )

    if backend == "faster":
        model = _fw_model_for_key(model_key)

        # Map common params; ignore thresholds that are mlx-specific.
        segments_iter, info = model.transcribe(
            audio_path,
            language=kwargs.get("language", "ar"),
            word_timestamps=bool(kwargs.get("word_timestamps", False)),
            condition_on_previous_text=bool(kwargs.get("condition_on_previous_text", True)),
            initial_prompt=kwargs.get("initial_prompt", None),
            temperature=kwargs.get("temperature", 0.0),
        )

        segments_out = []
        for seg in segments_iter:
            seg_dict = {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": (seg.text or ""),
            }
            words = getattr(seg, "words", None)
            if words:
                seg_dict["words"] = [
                    {
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": (w.word or ""),
                    }
                    for w in words
                    if getattr(w, "start", None) is not None and getattr(w, "end", None) is not None
                ]
            segments_out.append(seg_dict)

        return {"segments": segments_out, "info": getattr(info, "__dict__", {})}

    if backend == "openai":
        model = _openai_model_for_key(model_key)
        temp = kwargs.get("temperature", 0.0)
        result = model.transcribe(
            audio_path,
            language=kwargs.get("language"),
            verbose=bool(kwargs.get("verbose", False)),
            word_timestamps=bool(kwargs.get("word_timestamps", False)),
            condition_on_previous_text=bool(kwargs.get("condition_on_previous_text", True)),
            initial_prompt=kwargs.get("initial_prompt"),
            temperature=temp,
            no_speech_threshold=kwargs.get("no_speech_threshold"),
            logprob_threshold=kwargs.get("logprob_threshold"),
            compression_ratio_threshold=kwargs.get("compression_ratio_threshold"),
        )
        segments_raw = result.get("segments") or []
        segments_out: list[dict] = []
        for seg in segments_raw:
            seg_dict = {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": (seg.get("text") or ""),
            }
            words = seg.get("words")
            if words:
                seg_dict["words"] = [
                    {
                        "start": float(w.get("start", seg_dict["start"])),
                        "end": float(w.get("end", seg_dict["end"])),
                        "word": (w.get("word", "") or ""),
                    }
                    for w in words
                    if w.get("start") is not None and w.get("end") is not None
                ]
            segments_out.append(seg_dict)
        return {"segments": segments_out}

    raise RuntimeError("Unsupported backend selection.")


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
