#!/usr/bin/env python3
"""
Arabic STT — Gradio UI
Run: python app.py
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Load .env if present
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# Gradio reads this at import time — use a project-local dir to reduce Windows temp/permission issues.
_app_gradio_tmp = Path(__file__).parent / ".gradio_temp"
_app_gradio_tmp.mkdir(exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(_app_gradio_tmp.resolve()))
_ui_upload_dir = _app_gradio_tmp / "ui_uploads"
_ui_upload_dir.mkdir(exist_ok=True)
_certs_dir = Path(__file__).parent / ".certs"

import numpy as np
import soundfile as sf
import gradio as gr

from transcribe import (
    DEFAULT_INITIAL_PROMPT_FULL,
    MODEL_MAP,
    describe_backend_failures,
    fmt_ts,
    fmt_ts_ms,
    get_backend,
    reset_whisper_caches,
    transcribe_any,
)
from diarize import run_diarization, assign_speakers

_ffmpeg_exe_cache: str | None = None


def _ffmpeg_executable() -> str:
    global _ffmpeg_exe_cache
    if _ffmpeg_exe_cache:
        return _ffmpeg_exe_cache
    for key in ("FFMPEG_BIN", "FFMPEG_PATH"):
        raw = os.environ.get(key, "").strip().strip('"')
        if raw:
            p = Path(raw)
            if p.is_file():
                _ffmpeg_exe_cache = str(p.resolve())
                return _ffmpeg_exe_cache
    found = shutil.which("ffmpeg")
    if found:
        _ffmpeg_exe_cache = found
        return _ffmpeg_exe_cache
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    for guess in (
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(program_files) / "ffmpeg" / "bin" / "ffmpeg.exe",
    ):
        if guess.is_file():
            _ffmpeg_exe_cache = str(guess.resolve())
            return _ffmpeg_exe_cache
    raise RuntimeError(
        "ffmpeg was not found. Install ffmpeg and add it to your PATH, or set FFMPEG_BIN in .env "
        "to the full path of ffmpeg.exe (for example C:\\\\ffmpeg\\\\bin\\\\ffmpeg.exe)."
    )


_AUDIO_SUFFIXES = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".mpeg", ".mpga", ".aac",
}


def _audio_path_from_upload(audio) -> str | None:
    if audio is None:
        return None
    if isinstance(audio, str):
        return audio
    if isinstance(audio, dict):
        return audio.get("path") or audio.get("name")
    return getattr(audio, "path", None) or getattr(audio, "name", None)


def stage_upload_audio(audio) -> str | None:
    """
    Copy uploads to ASCII-only paths under .gradio_temp so the browser can play them.
    Arabic/special characters in filenames often break Gradio file serving on Windows.
    """
    path = _audio_path_from_upload(audio)
    if not path:
        return None
    src = Path(path)
    if not src.is_file():
        return path
    resolved = str(src.resolve())
    # Already safe — return unchanged so the player is not asked to reload (avoids AbortError).
    if resolved.isascii():
        return resolved
    try:
        if src.parent.resolve() == _ui_upload_dir.resolve():
            return resolved
    except OSError:
        pass
    suffix = src.suffix.lower() if src.suffix else ".bin"
    if suffix not in _AUDIO_SUFFIXES:
        suffix = ".bin"
    dest = _ui_upload_dir / f"audio_{int(time.time() * 1000)}{suffix}"
    try:
        shutil.copy2(src, dest)
    except (PermissionError, OSError) as e:
        raise gr.Error(
            "Could not read the uploaded file. Try again or use a simpler file name."
        ) from e
    return str(dest.resolve())


def _copy_upload_for_processing(upload_path: str) -> str:
    """Copy Gradio’s upload to a short path so ffmpeg/subprocess can open it reliably on Windows."""
    staged = stage_upload_audio(upload_path)
    upload_path = staged or upload_path
    src = Path(upload_path)
    if not src.is_file():
        raise gr.Error("Uploaded file is missing or not readable.")
    suffix = src.suffix.lower() or ".bin"
    fd, dest = tempfile.mkstemp(prefix="upload_", suffix=suffix)
    os.close(fd)
    try:
        shutil.copyfile(str(src), dest)
    except (PermissionError, OSError) as e:
        Path(dest).unlink(missing_ok=True)
        raise gr.Error(
            "Could not read the uploaded file. Try a simpler file name (letters and numbers only), "
            "or copy the file to your Desktop and upload again."
        ) from e
    return dest


def _ssl_enabled() -> bool:
    return os.environ.get("GRADIO_ENABLE_SSL", "").strip().lower() in ("1", "true", "yes", "y")


def _find_openssl() -> str | None:
    found = shutil.which("openssl")
    if found:
        return found
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    for guess in (
        Path(program_files) / "Git" / "usr" / "bin" / "openssl.exe",
        Path(program_files) / "OpenSSL-Win64" / "bin" / "openssl.exe",
        Path(r"C:\Program Files\Git\usr\bin\openssl.exe"),
    ):
        if guess.is_file():
            return str(guess)
    return None


def _looks_like_ip(host: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))


def _ensure_ssl_certs() -> tuple[str, str] | None:
    """Return (cert, key) paths; generate self-signed material when GRADIO_ENABLE_SSL is set."""
    explicit_cert = os.environ.get("SSL_CERTFILE", "").strip().strip('"')
    explicit_key = os.environ.get("SSL_KEYFILE", "").strip().strip('"')
    if explicit_cert and explicit_key:
        if Path(explicit_cert).is_file() and Path(explicit_key).is_file():
            return explicit_cert, explicit_key
        print("SSL_CERTFILE / SSL_KEYFILE are set but files are missing.", file=sys.stderr)
        return None

    if not _ssl_enabled():
        return None

    _certs_dir.mkdir(exist_ok=True)
    cert_file = _certs_dir / "cert.pem"
    key_file = _certs_dir / "key.pem"
    if cert_file.is_file() and key_file.is_file():
        return str(cert_file), str(key_file)

    openssl = _find_openssl()
    if not openssl:
        print(
            "GRADIO_ENABLE_SSL is set but openssl was not found. "
            "Install OpenSSL (e.g. via Git for Windows) or set SSL_CERTFILE and SSL_KEYFILE.",
            file=sys.stderr,
        )
        return None

    cn = os.environ.get("SSL_CN", os.environ.get("GRADIO_SERVER_NAME", "localhost")).strip() or "localhost"
    if _looks_like_ip(cn):
        san = f"IP:{cn},DNS:localhost"
    else:
        san = f"DNS:{cn},DNS:localhost"

    cmd = [
        openssl,
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(key_file),
        "-out",
        str(cert_file),
        "-days",
        "825",
        "-subj",
        f"/CN={cn}",
        "-addext",
        f"subjectAltName={san}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        cmd = [c for c in cmd if c != "-addext" and c != f"subjectAltName={san}"]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode(errors="replace")
            print(f"Could not generate SSL certificate:\n{err}", file=sys.stderr)
            return None

    if cert_file.is_file() and key_file.is_file():
        print(f"Generated self-signed certificate in {_certs_dir}", file=sys.stderr)
        return str(cert_file), str(key_file)
    return None


def _launch_kwargs(*, inbrowser: bool) -> dict:
    server_name = os.environ.get("GRADIO_SERVER_NAME", "172.24.4.204").strip() or "0.0.0.0"
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    kwargs: dict = {
        "server_name": server_name,
        "server_port": port,
        "inbrowser": inbrowser,
        "theme": THEME,
        "css": CSS,
        "allowed_paths": [str(_app_gradio_tmp.resolve())],
    }
    ssl = _ensure_ssl_certs()
    if ssl:
        kwargs["ssl_certfile"], kwargs["ssl_keyfile"] = ssl
        kwargs["ssl_verify"] = False
    return kwargs


# ── Theme & CSS ───────────────────────────────────────────────────────────────

IS_WINDOWS = os.name == "nt"

THEME = (
    gr.themes.Base(primary_hue="stone", neutral_hue="stone")
    if IS_WINDOWS
    else gr.themes.Base(
        primary_hue="stone",
        neutral_hue="stone",
        # GoogleFont can cause long stalls in offline/server environments.
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    )
).set(
    body_background_fill="#faf8f4",
    body_text_color="#2c2825",
    background_fill_primary="#faf8f4",
    background_fill_secondary="#f4f0e8",
    border_color_primary="#e8e2d6",
    color_accent_soft="#ede8df",
    button_primary_background_fill="#2c2825",
    button_primary_background_fill_hover="#1a1714",
    button_primary_text_color="#faf8f4",
    button_secondary_background_fill="#f4f0e8",
    button_secondary_background_fill_hover="#ede8df",
    button_secondary_text_color="#2c2825",
    button_secondary_border_color="#e8e2d6",
    input_background_fill="#ffffff",
    input_border_color="#e8e2d6",
    input_placeholder_color="#c4bab0",
    block_background_fill="#faf8f4",
    block_border_color="#e8e2d6",
    block_label_background_fill="#faf8f4",
    block_label_text_color="#9e9188",
    block_title_text_color="#9e9188",
    panel_background_fill="#f4f0e8",
    checkbox_background_color="#ffffff",
    shadow_drop="none",
    shadow_drop_lg="none",
    shadow_inset="none",
)

_FONT_IMPORT = "" if IS_WINDOWS else "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');"

CSS = _FONT_IMPORT + """

/* ── base ── */
html, body, .gradio-container { background: #faf8f4 !important; }
.gradio-container { max-width: 100% !important; width: 100% !important; margin: 0 auto !important; padding: 0 !important; }
.main { padding: 0 !important; }

/* ── topbar ── */
#topbar {
    display: flex;
    align-items: center;
    padding: 16px 36px;
    border-bottom: 1px solid #e8e2d6;
    background: #faf8f4;
}
#topbar .title {
    font-family: 'Inter', sans-serif;
    font-size: 1.65rem;
    font-weight: 600;
    color: #2c2825;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

/* ── tab bar ── */
.tab-nav {
    background: #faf8f4 !important;
    border-bottom: 1px solid #e8e2d6 !important;
    padding: 0 36px !important;
    gap: 0 !important;
}
.tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #c4bab0 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 12px 20px !important;
    margin: 0 !important;
    letter-spacing: 0.01em !important;
    transition: color 0.15s, border-color 0.15s !important;
}
.tab-nav button.selected {
    color: #2c2825 !important;
    border-bottom-color: #2c2825 !important;
}

/* ── layout ── */
.tab-content > .flex { padding: 28px 36px !important; gap: 28px !important; }
.upload-layout { width: 100% !important; gap: 12px !important; }

/* ── section title above cards ── */
.section-heading,
.section-heading p {
    margin: 0 0 8px 0 !important;
    color: #9e9188 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
.section-heading h3 {
    margin: 0 !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    color: inherit !important;
}

/* ── upload: audio card (heading inside beige, like Model / Transcript) ── */
.upload-audio-card {
    background: #f4f0e8 !important;
    border: 1px solid #e8e2d6 !important;
    border-radius: 8px !important;
    padding: 12px 14px 14px !important;
    width: 100% !important;
    min-width: 0 !important;
    gap: 8px !important;
}
#upload-audio {
    width: 100% !important;
    min-width: 0 !important;
}
#upload-audio > .form,
#upload-audio > .block {
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
}
#upload-audio,
#upload-audio > .form,
#upload-audio > .block,
#upload-audio .audio-container,
#upload-audio [class*="waveform"],
#upload-audio wave {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
}
/* empty upload strip only — stay short until a file is loaded */
#upload-audio:not(:has(wave)) > .form,
#upload-audio:not(:has(wave)) > .block,
#upload-audio:not(:has(wave)) .wrap {
    min-height: 0 !important;
    height: auto !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    margin: 0 !important;
}
#upload-audio .empty,
#upload-audio .upload-container {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.3em !important;
    text-align: center !important;
    white-space: nowrap !important;
    min-height: 0 !important;
    height: 26px !important;
    max-height: 26px !important;
    padding: 0 8px !important;
    overflow: hidden !important;
    font-size: 0.75rem !important;
    line-height: 1 !important;
    border-radius: 6px !important;
}
#upload-audio .empty svg,
#upload-audio .upload-container svg,
#upload-audio .icon-wrap svg,
#upload-audio .icon-wrap img {
    width: 14px !important;
    height: 14px !important;
    max-width: 14px !important;
    max-height: 14px !important;
    flex-shrink: 0 !important;
}
#upload-audio .icon-wrap {
    width: auto !important;
    height: auto !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}
#upload-audio .empty > *,
#upload-audio .upload-container > *,
#upload-audio .icon-wrap,
#upload-audio .or,
#upload-audio .upload-text,
#upload-audio .empty p,
#upload-audio .empty span,
#upload-audio .empty label,
#upload-audio .empty div {
    display: inline !important;
    flex: 0 0 auto !important;
    width: auto !important;
    max-width: none !important;
    white-space: nowrap !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.4 !important;
}
#upload-audio wave,
#upload-audio .wrapper,
#upload-audio [class*="scroll"] {
    overflow: hidden !important;
    overflow-x: hidden !important;
    width: 100% !important;
}
#upload-audio wave {
    height: 72px !important;
    min-height: 72px !important;
}
#upload-audio canvas {
    width: 100% !important;
    max-width: 100% !important;
    height: 72px !important;
}
#upload-audio .audio-container,
#upload-audio .waveform-container {
    min-height: 72px !important;
    height: auto !important;
    max-height: none !important;
    margin: 0 !important;
    padding: 0 !important;
}
#upload-audio .controls,
#upload-audio .control-wrapper,
#upload-audio .timeline,
#upload-audio [class*="time"],
#upload-audio [class*="duration"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 6px 10px !important;
    margin-top: 4px !important;
    padding: 0 !important;
    overflow: visible !important;
    height: auto !important;
    min-height: 24px !important;
    visibility: visible !important;
}

/* ── upload: three matching settings tiles in a row ── */
.upload-settings-row {
    flex-wrap: nowrap !important;
    gap: 12px !important;
    align-items: stretch !important;
    width: 100% !important;
}
.settings-tile {
    background: #f4f0e8 !important;
    border: 1px solid #e8e2d6 !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
    flex: 1 1 0 !important;
    min-width: 0 !important;
}
.settings-tile-heading,
.settings-tile-heading p {
    margin: 0 0 10px 0 !important;
    color: #9e9188 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
.settings-tile-heading h3 {
    margin: 0 !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    color: inherit !important;
}
/* normal labels inside tiles (global label rule is uppercase) */
.settings-tile label,
.settings-tile .label-wrap span {
    text-transform: none !important;
    letter-spacing: 0 !important;
}
/* no extra inner box on diarization checkbox */
#diarize-check > .block,
#diarize-check > .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.upload-transcribe-row {
    margin-top: 2px !important;
    width: 100% !important;
}
.upload-transcribe-row > .column,
.upload-transcribe-row > .form,
.upload-transcribe-row #transcribe-btn,
.upload-transcribe-row .block {
    width: 100% !important;
    max-width: 100% !important;
    flex-grow: 1 !important;
}
@media (max-width: 960px) {
    .upload-settings-row { flex-wrap: wrap !important; }
    .settings-tile { flex: 1 1 calc(50% - 8px) !important; min-width: 220px !important; }
}
@media (max-width: 560px) {
    .settings-tile { flex: 1 1 100% !important; }
}

/* ── transcript card (matches settings-tile / audio card) ── */
.transcript-card {
    background: #f4f0e8 !important;
    border: 1px solid #e8e2d6 !important;
    border-radius: 8px !important;
    padding: 12px 14px 14px !important;
    width: 100% !important;
    min-width: 0 !important;
    gap: 8px !important;
}
.transcript-card #stats,
.transcript-card #live-stats {
    border-top: 1px solid #e8e2d6 !important;
    margin-top: 8px !important;
    padding-top: 8px !important;
    font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif !important;
    font-size: 0.78rem !important;
    color: #9e9188 !important;
    text-align: right !important;
    direction: rtl !important;
}

/* ── controls panel (live tab) ── */
.controls-col {
    background: #f4f0e8 !important;
    border: 1px solid #e8e2d6 !important;
    border-radius: 8px !important;
    padding: 24px !important;
}
.live-audio-card {
    width: 100% !important;
}
#live-mic-audio wave,
#live-mic-audio .wrapper {
    height: 52px !important;
    min-height: 52px !important;
    max-height: 52px !important;
}
#live-mic-audio,
#live-mic-audio > .form,
#live-mic-audio > .block,
#live-mic-audio .audio-container,
#live-mic-audio wave {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

/* ── live transcript ── */
#live-transcript { height: 100% !important; }
#live-transcript textarea {
    font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.8 !important;
    color: #2c2825 !important;
    background: #ffffff !important;
    border: 1px solid #e8e2d6 !important;
    border-radius: 8px !important;
    min-height: 380px !important;
    padding: 16px !important;
    resize: none !important;
    direction: rtl !important;
    text-align: right !important;
}
#live-transcript textarea::placeholder { color: #d4cec8 !important; }

/* ── upload transcript heading (Markdown — avoids orphan <label for>) ── */
.transcript-block-heading,
.transcript-block-heading p {
    margin: 0 0 8px 0 !important;
    color: #9e9188 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}
.transcript-block-heading h3 {
    margin: 0 !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    color: inherit !important;
}

/* ── clickable transcript (upload tab) ── */
/* Gradio wraps gr.HTML in extra divs — do not style those (avoids a second scroll strip). */
#transcript-box,
#transcript-box > .form,
#transcript-box > .block,
#transcript-box > div {
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    overflow: visible !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}
#transcript-box textarea,
#transcript-box pre,
#transcript-box code {
    display: none !important;
}
#transcript-box .transcript-body {
    font-family: 'Segoe UI', 'Tahoma', 'Arial', sans-serif;
    font-size: 0.95rem;
    line-height: 1.8;
    background: #ffffff;
    border: 1px solid #e8e2d6;
    border-radius: 8px;
    min-height: 380px;
    max-height: min(70vh, 720px);
    padding: 16px;
    overflow-y: auto;
    color: #2c2825;
    direction: rtl;
    text-align: right;
    unicode-bidi: isolate;
}
#transcript-box .transcript-body.transcript-empty {
    min-height: 200px;
}
#transcript-box .seg {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.35em 0.75em;
    margin-bottom: 1.4em;
    direction: rtl;
    text-align: right;
}
#transcript-box .seg-text {
    flex: 1 1 auto;
    min-width: 10em;
}
#transcript-box .ts {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72em;
    color: #c4bab0;
    cursor: pointer;
    margin-inline-end: 8px;
    user-select: none;
    unicode-bidi: embed;
    direction: ltr;
    display: inline-block;
}
#transcript-box .ts:hover { color: #9e9188; }
#transcript-box .spk {
    font-size: 0.78em;
    font-weight: 600;
    color: #a08060;
    margin-inline-end: 6px;
    unicode-bidi: embed;
    direction: ltr;
    display: inline-block;
}
#transcript-box .word {
    cursor: pointer;
    border-radius: 2px;
    padding: 1px 2px;
    transition: background 0.1s;
}
#transcript-box .word:hover { background: #ede8df; }
/* ── labels ── */
label, .label-wrap span, fieldset legend {
    font-family: 'Inter', sans-serif !important;
    color: #9e9188 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}
.info {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.73rem !important;
    color: #c4bab0 !important;
}
input, select {
    font-family: 'Inter', sans-serif !important;
    color: #2c2825 !important;
}

/* ── buttons ── */
button {
    font-family: 'Inter', sans-serif !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
}
#transcribe-btn,
#transcribe-btn button {
    width: 100% !important;
    max-width: 100% !important;
    height: 40px !important;
}
#clear-btn button, #save-live-btn button { width: 100% !important; height: 36px !important; }
.stt-download-btn button {
    width: 100% !important;
    height: 36px !important;
    background: #ffffff !important;
    color: #9e9188 !important;
    border: 1px solid #e8e2d6 !important;
}

/* ── stats ── */
#stats, #live-stats {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #c4bab0 !important;
    padding: 10px 2px 0 !important;
    border-top: 1px solid #e8e2d6 !important;
    margin-top: 14px !important;
}

/* ── slider ── */
.gradio-slider input[type=range] { accent-color: #2c2825; }

/* ── scrollbar ── */
textarea::-webkit-scrollbar { width: 4px; }
textarea::-webkit-scrollbar-track { background: transparent; }
textarea::-webkit-scrollbar-thumb { background: #e8e2d6; border-radius: 2px; }

/* ── backend missing (PyTorch / VC++ runtime) ── */
.backend-setup-warning {
    border: 1px solid #c4a574 !important;
    background: #fff8ec !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
    margin: 0 0 16px 0 !important;
}
.backend-setup-warning p { margin: 0.5em 0 !important; }
"""


_loaded_model_repo: str | None = None


def evict_model_if_changed(new_model_key: str):
    global _loaded_model_repo
    new_repo = MODEL_MAP[new_model_key]
    if _loaded_model_repo is not None and _loaded_model_repo != new_repo:
        try:
            import mlx.core as mx  # type: ignore

            mx.clear_cache()
        except Exception:
            pass
        reset_whisper_caches()
    _loaded_model_repo = new_repo


# ── Helpers ───────────────────────────────────────────────────────────────────

_EMPTY_TRANSCRIPT_HTML = (
    '<div class="transcript-body transcript-empty" dir="rtl" lang="ar"></div>'
)


def fmt_duration_ar(seconds: float) -> str:
    """Human-readable Arabic duration: seconds only below 60, else minutes/hours."""
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s} ثانية"
    m, sec = divmod(s, 60)
    if s < 3600:
        parts = [f"{m} دقيقة"] if m else []
        if sec:
            parts.append(f"{sec} ثانية")
        return " و ".join(parts)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    parts = [f"{h} ساعة"] if h else []
    if m:
        parts.append(f"{m} دقيقة")
    if sec:
        parts.append(f"{sec} ثانية")
    return " و ".join(parts)


def reset_transcript_ui():
    """Clear transcript area when starting a new run."""
    return (
        _EMPTY_TRANSCRIPT_HTML,
        gr.DownloadButton(visible=False, elem_classes=["stt-download-btn"]),
        "",
    )


def segments_to_html(segments) -> str:
    """Renders transcript as clickable word spans. Clicking a word seeks the audio player."""
    if not segments:
        return _EMPTY_TRANSCRIPT_HTML + "\n"

    parts = []
    for seg in segments:
        if not seg.get("text", "").strip():
            continue

        speaker = seg.get("speaker", "")
        spk_html = f'<span class="spk">{speaker}</span>' if speaker else ""
        seg_start = seg["start"]
        seg_end   = seg["end"]
        ts_html = (
            f'<span class="ts" onclick="seekAudio({seg_start:.3f})">'
            f'[{fmt_ts(seg_start)} ← {fmt_ts(seg_end)}]</span>'
        )

        words = seg.get("words") or []
        if words:
            word_spans = []
            for w in words:
                t = w.get("start", seg["start"])
                word_text = w.get("word", "").strip()
                if word_text:
                    word_spans.append(
                        f'<span class="word" onclick="seekAudio({t:.3f})">{word_text}</span>'
                    )
            words_html = " ".join(word_spans)
        else:
            t = seg["start"]
            words_html = (
                f'<span class="word" onclick="seekAudio({t:.3f})">'
                f'{seg["text"].strip()}</span>'
            )

        parts.append(
            f'<div class="seg">{ts_html}{spk_html}'
            f'<span class="seg-text">{words_html}</span></div>'
        )

    return (
        f'<div class="transcript-body" dir="rtl" lang="ar">\n'
        + "\n".join(parts)
        + "\n</div>"
    )


def segments_to_file(segments) -> str:
    lines = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        speaker = seg.get("speaker", "")
        prefix = f"{speaker}  " if speaker else ""
        lines.append(f"{prefix}[{fmt_ts_ms(seg['start'])} → {fmt_ts_ms(seg['end'])}] {text}")
    return "\n".join(lines)


def _save_diarization(turns: list, source_path: str) -> None:
    stem = Path(source_path).stem
    out = Path(__file__).parent / f"{stem}_diarization.txt"
    lines = [f"[{fmt_ts_ms(s)} → {fmt_ts_ms(e)}]  {spk}" for s, e, spk in turns]
    out.write_text("\n".join(lines), encoding="utf-8")


def _save_transcript(segments: list, source_path: str) -> None:
    stem = Path(source_path).stem
    out = Path(__file__).parent / f"{stem}_transcript.txt"
    out.write_text(segments_to_file(segments), encoding="utf-8")


def _save_word_diarization(segments: list, source_path: str) -> None:
    stem = Path(source_path).stem
    out = Path(__file__).parent / f"{stem}_words.txt"
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "")
        for w in seg.get("words") or []:
            word = w.get("word", "").strip()
            if not word:
                continue
            start = fmt_ts_ms(w.get("start", seg["start"]))
            end   = fmt_ts_ms(w.get("end",   seg["end"]))
            lines.append(f"{speaker}  [{start} → {end}]  {word}")
    out.write_text("\n".join(lines), encoding="utf-8")


# ── Audio preprocessing ───────────────────────────────────────────────────────

def extract_audio_for_diarization(input_path: str, start: float = 0.0, end: float = 0.0) -> str:
    """Minimal conversion for pyannote: audio-only WAV, no filtering that strips speaker characteristics."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    ff = _ffmpeg_executable()
    cmd = [ff, "-y"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", input_path]
    if end > 0 and end > start:
        cmd += ["-t", str(end - start)]
    cmd += [
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        tmp.name,
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        Path(tmp.name).unlink(missing_ok=True)
        err = (result.stderr or b"").decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed:\n{err}")

    return tmp.name


def preprocess_audio(input_path: str, start: float = 0.0, end: float = 0.0) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    ff = _ffmpeg_executable()
    cmd = [ff, "-y"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", input_path]
    if end > 0 and end > start:
        cmd += ["-t", str(end - start)]
    cmd += [
        "-vn",
        "-af", (
            "highpass=f=100,"
            "lowpass=f=8000,"
            "afftdn=nf=-25,"
            "dynaudnorm=f=150:g=15"
        ),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        tmp.name,
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        Path(tmp.name).unlink(missing_ok=True)
        err = (result.stderr or b"").decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed:\n{err}")

    return tmp.name


# ── Upload tab ────────────────────────────────────────────────────────────────

def run_transcription(
    file,
    model_key,
    do_diarize,
    min_spk,
    max_spk,
    start_time,
    end_time,
    user_hint,
    progress=gr.Progress(),
):
    if file is None:
        raise gr.Error("Please upload an audio file first.")

    progress(0.05, desc="Preprocessing audio")
    evict_model_if_changed(model_key)
    t0   = time.time()

    raw_stem = Path(str(file)).stem.strip() or "audio"
    _bad = '<>:"/\\|?*'
    out_stem = ("".join(c if c not in _bad else "_" for c in raw_stem))[:120] or "audio"
    saves_ref = f"{out_stem}.wav"

    work_path = _copy_upload_for_processing(str(file))
    cleaned = None
    try:
        try:
            cleaned = preprocess_audio(
                work_path,
                start=float(start_time or 0),
                end=float(end_time or 0),
            )
        except RuntimeError as e:
            raise gr.Error(str(e)) from e

        try:
            progress(0.15, desc="Transcribing")

            _result: list = [None]
            _err: list = [None]

            def _whisper():
                try:
                    _result[0] = transcribe_any(
                        cleaned,
                        model_key=model_key,
                        live=False,
                        user_hint=user_hint,
                    )
                except Exception as e:
                    _err[0] = e

            t = threading.Thread(target=_whisper, daemon=True)
            t.start()

            p = 0.15
            while t.is_alive():
                time.sleep(0.4)
                p = min(p + 0.008, 0.68)
                progress(p, desc="Transcribing")
            t.join()

            if _err[0]:
                raise gr.Error(str(_err[0])) from _err[0]
            result = _result[0]

            segments = result.get("segments", [])
            if not segments:
                raise gr.Error("No speech detected in the file.")

            if do_diarize:
                progress(0.72, desc="Identifying speakers")
                diarize_wav = None
                try:
                    try:
                        import mlx.core as mx  # type: ignore

                        mx.clear_cache()
                    except Exception:
                        pass
                    diarize_wav = extract_audio_for_diarization(
                        work_path,
                        start=float(start_time or 0),
                        end=float(end_time or 0),
                    )
                    turns = run_diarization(
                        diarize_wav,
                        min_speakers=int(min_spk or 0),
                        max_speakers=int(max_spk or 0),
                    )
                    segments = assign_speakers(segments, turns)
                    _save_diarization(turns, saves_ref)
                    _save_word_diarization(segments, saves_ref)
                except RuntimeError as e:
                    raise gr.Error(str(e))
                finally:
                    if diarize_wav:
                        Path(diarize_wav).unlink(missing_ok=True)
        finally:
            if cleaned:
                Path(cleaned).unlink(missing_ok=True)
    finally:
        Path(work_path).unlink(missing_ok=True)

    elapsed  = time.time() - t0
    duration = segments[-1]["end"]

    out_fd, out_path_str = tempfile.mkstemp(suffix=".txt")
    os.close(out_fd)
    out_path = Path(out_path_str)
    out_path.write_text(segments_to_file(segments), encoding="utf-8")

    _save_transcript(segments, saves_ref)

    stats = (
        f"مدة التسجيل: {fmt_duration_ar(duration)} · "
        f"وقت المعالجة: {fmt_duration_ar(elapsed)}"
    )

    progress(1.0, desc="Done")
    return (
        segments_to_html(segments),
        gr.DownloadButton(
            value=str(out_path),
            visible=True,
            label="Download transcript ⬇",
            elem_classes=["stt-download-btn"],
        ),
        stats,
    )


# ── Live tab ──────────────────────────────────────────────────────────────────

def _transcribe_buffer(buffer, sr, secs_done, chunk_count, transcript, model_key):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, buffer, sr, subtype="PCM_16")
    tmp.close()
    try:
        result = transcribe_any(
            tmp.name,
            model_key=model_key,
            live=True,
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    segments = result.get("segments", [])
    new_text = " ".join(s["text"].strip() for s in segments if s["text"].strip())

    chunk_dur = len(buffer) / sr
    if new_text:
        start = fmt_ts(secs_done)
        end   = fmt_ts(secs_done + chunk_dur)
        line  = f"[{start} <- {end}]  {new_text}"
        transcript = (transcript + "\n\n" + line) if transcript else line

    secs_done  += chunk_dur
    chunk_count += 1
    word_count = len(transcript.split()) if transcript else 0
    stats = f"words: {word_count}  ·  chunks: {chunk_count}"
    return transcript, None, secs_done, chunk_count, stats


def process_live_chunk(chunk, buffer, transcript, secs_done, chunk_count, model_key, chunk_seconds):
    if chunk is None:
        yield transcript, buffer, secs_done, chunk_count, ""
        return

    sr, audio = chunk
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = float(np.abs(audio).max() or 0.0)
    if peak > 1.5:
        audio = audio / 32768.0

    buffer = audio if buffer is None else np.concatenate([buffer, audio])

    if len(buffer) / sr < chunk_seconds:
        yield transcript, buffer, secs_done, chunk_count, ""
        return

    yield transcript, buffer, secs_done, chunk_count, "⏳ processing…"
    yield _transcribe_buffer(buffer, sr, secs_done, chunk_count, transcript, model_key)


def flush_live_buffer(buffer, transcript, secs_done, chunk_count, model_key):
    if buffer is None or len(buffer) == 0:
        return transcript, None, secs_done, chunk_count, ""
    sr = 16000
    return _transcribe_buffer(buffer, sr, secs_done, chunk_count, transcript, model_key)


def save_live_transcript(transcript):
    if not transcript:
        raise gr.Error("No transcript to download.")
    out_fd, out_path_str = tempfile.mkstemp(suffix=".txt")
    os.close(out_fd)
    Path(out_path_str).write_text(transcript, encoding="utf-8")
    return gr.DownloadButton(
        value=out_path_str,
        visible=True,
        label="Download transcript ⬇",
        elem_classes=["stt-download-btn"],
    )


def clear_live():
    return "", None, 0.0, 0, "", gr.DownloadButton(visible=False, elem_classes=["stt-download-btn"])


# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Arabic Speech to Text") as demo:

    gr.HTML("""
        <div id="topbar">
            <span class="title">Arabic Speech to Text</span>
        </div>
        <script>
        // ── word-click seek ──────────────────────────────────────────────
        function seekAudio(t) {
            const el = document.querySelector('#upload-audio');
            if (!el) return;
            // try WaveSurfer instance first (Gradio 6 stores it on the element)
            const ws = el._wavesurfer || el.__wavesurfer;
            if (ws && ws.getDuration) {
                ws.seekTo(t / ws.getDuration());
                return;
            }
            // fallback: native <audio>
            const audio = el.querySelector('audio');
            if (audio) { audio.currentTime = t; if (audio.paused) audio.play(); }
        }

        // ── start/end crop highlight on waveform ─────────────────────────
        function highlightCrop(start, end) {
            const el = document.querySelector('#upload-audio');
            if (!el) return;
            const ws = el._wavesurfer || el.__wavesurfer;
            if (!ws) return;
            const dur = ws.getDuration();
            if (!dur) return;

            // draw an orange overlay on the waveform canvas
            let overlay = el.querySelector('#region-overlay');
            if (end <= 0 || start >= end) {
                if (overlay) overlay.remove();
                return;
            }
            const wrapper = el.querySelector('wave') || el.querySelector('.wrapper');
            if (!wrapper) return;

            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = 'region-overlay';
                overlay.style.cssText = [
                    'position:absolute', 'top:0', 'bottom:0',
                    'background:rgba(255,165,0,0.25)',
                    'border-left:2px solid orange',
                    'border-right:2px solid orange',
                    'pointer-events:none', 'z-index:5'
                ].join(';');
                wrapper.style.position = 'relative';
                wrapper.appendChild(overlay);
            }
            overlay.style.left  = (start / dur * 100) + '%';
            overlay.style.width = ((end - start) / dur * 100) + '%';
        }

        // watch crop start/end inputs and update the highlight
        function watchCropInputs() {
            const inputs = document.querySelectorAll(
                '#start-end-crop-inputs input[type=number]'
            );
            if (inputs.length < 2) { setTimeout(watchCropInputs, 500); return; }
            const update = () => {
                const s = parseFloat(inputs[0].value) || 0;
                const e = parseFloat(inputs[1].value) || 0;
                highlightCrop(s, e);
                fitUploadWaveform();
            };
            inputs.forEach(inp => inp.addEventListener('input', update));
        }
        document.addEventListener('DOMContentLoaded', watchCropInputs);
        setTimeout(watchCropInputs, 1000);

        function fitWaveform(containerId) {
            const el = document.querySelector(containerId);
            if (!el) return;
            const ws = el._wavesurfer || el.__wavesurfer;
            if (!ws) return;
            const w = el.clientWidth;
            if (w <= 0) return;
            const dur = (typeof ws.getDuration === 'function' && ws.getDuration()) || 0;
            const opts = { width: w };
            if (dur > 0) {
                opts.minPxPerSec = w / dur;
            }
            try {
                if (ws.setOptions) ws.setOptions(opts);
            } catch (e) {}
            el.querySelectorAll('wave, .wrapper, [class*="scroll"]').forEach((node) => {
                node.style.overflowX = 'hidden';
                node.style.overflow = 'hidden';
            });
            if (ws.drawBuffer) ws.drawBuffer();
            else if (typeof ws.render === 'function') ws.render();
        }

        function fitUploadWaveform() { fitWaveform('#upload-audio'); }
        function fitLiveWaveform() { fitWaveform('#live-mic-audio'); }

        window.addEventListener('resize', () => {
            fitUploadWaveform();
            fitLiveWaveform();
        });

        function watchWaveformContainers() {
            ['#upload-audio', '#live-mic-audio'].forEach((sel) => {
                const el = document.querySelector(sel);
                if (!el) return;
                const obs = new MutationObserver(() => {
                    fitWaveform(sel);
                    [100, 300, 800, 1500, 3000].forEach((ms) => {
                        setTimeout(() => fitWaveform(sel), ms);
                    });
                });
                obs.observe(el, { childList: true, subtree: true, attributes: true });
            });
        }
        document.addEventListener('DOMContentLoaded', watchWaveformContainers);
        setTimeout(watchWaveformContainers, 500);

        function compactUploadDropZone() {
            const root = document.querySelector('#upload-audio');
            if (!root || root.querySelector('wave')) return;
            root.querySelectorAll('.form, .block, .wrap').forEach((el) => {
                el.style.minHeight = '0';
                el.style.paddingTop = '0';
                el.style.paddingBottom = '0';
            });
            const zone = root.querySelector('.empty, [class*="upload"]');
            if (!zone) return;
            zone.style.height = '26px';
            zone.style.maxHeight = '26px';
            zone.style.minHeight = '0';
            zone.style.padding = '0 8px';
            zone.style.lineHeight = '1';
            if (zone.dataset.sttFlat === '1') return;
            const parts = zone.innerText.split(/\n+/).map((s) => s.trim()).filter(Boolean);
            if (parts.length <= 1) { zone.dataset.sttFlat = '1'; return; }
            zone.dataset.sttFlat = '1';
            zone.style.display = 'flex';
            zone.style.flexDirection = 'row';
            zone.style.flexWrap = 'nowrap';
            zone.style.alignItems = 'center';
            zone.style.justifyContent = 'center';
            zone.style.whiteSpace = 'nowrap';
            zone.style.gap = '0.3em';
            const icon = zone.querySelector('.icon-wrap, svg, img');
            zone.textContent = '';
            if (icon) zone.appendChild(icon);
            const line = document.createElement('span');
            line.style.whiteSpace = 'nowrap';
            line.style.fontSize = '0.75rem';
            line.style.lineHeight = '1';
            line.textContent = parts.join(' ').replace(/\s+/g, ' ').trim();
            zone.appendChild(line);
        }
        document.addEventListener('DOMContentLoaded', compactUploadDropZone);
        [200, 600, 1200, 2500].forEach((ms) => setTimeout(compactUploadDropZone, ms));
        const uploadObs = new MutationObserver(() => compactUploadDropZone());
        document.addEventListener('DOMContentLoaded', () => {
            const root = document.querySelector('#upload-audio');
            if (root) uploadObs.observe(root, { childList: true, subtree: true });
        });
        </script>
    """)

    if get_backend() == "none":
        gr.Markdown(describe_backend_failures(), elem_classes=["backend-setup-warning"])

    with gr.Tabs():

        # ── Tab 1: Upload ─────────────────────────────────────────────────
        with gr.Tab("Upload"):
            with gr.Column(elem_classes=["upload-layout"]):
                with gr.Column(elem_classes=["upload-audio-card"]):
                    gr.Markdown("### Audio", elem_classes=["settings-tile-heading"])
                    with gr.Column(elem_id="upload-audio"):
                        file_input = gr.Audio(
                            type="filepath",
                            label=None,
                            show_label=False,
                        )

                with gr.Row(elem_classes=["upload-settings-row"]):
                    with gr.Column(elem_classes=["settings-tile"], scale=1):
                        gr.Markdown("### Model", elem_classes=["settings-tile-heading"])
                        model_picker = gr.Dropdown(
                            choices=list(MODEL_MAP.keys()),
                            value="large-v3-4bit",
                            label=None,
                            show_label=False,
                            info="4bit = fast  ·  large-v3 = max accuracy",
                        )
                    with gr.Column(elem_classes=["settings-tile"], scale=1):
                        gr.Markdown("### Crop", elem_classes=["settings-tile-heading"])
                        with gr.Row(elem_id="start-end-crop-inputs"):
                            start_time = gr.Number(
                                label="Start (s)",
                                value=0,
                                minimum=0,
                            )
                            end_time = gr.Number(
                                label="End (s)",
                                value=0,
                                minimum=0,
                            )
                    with gr.Column(elem_classes=["settings-tile"], scale=1):
                        gr.Markdown("### Speakers", elem_classes=["settings-tile-heading"])
                        diarize_check = gr.Checkbox(
                            label="Diarization",
                            value=False,
                            elem_id="diarize-check",
                            container=False,
                        )
                        with gr.Row(elem_id="speaker-range-inputs"):
                            min_speakers = gr.Number(
                                label="Min speakers",
                                value=0,
                                minimum=0,
                                precision=0,
                            )
                            max_speakers = gr.Number(
                                label="Max speakers",
                                value=0,
                                minimum=0,
                                precision=0,
                            )
                    with gr.Column(elem_classes=["settings-tile"], scale=2):
                        gr.Markdown("### Hints", elem_classes=["settings-tile-heading"])
                        hint_input = gr.Textbox(
                            lines=2,
                            max_lines=4,
                            label=None,
                            show_label=False,
                            placeholder=(
                                "Optional: names, topic, jargon — e.g. board meeting, "
                                "أحمد، سارة، مشروع النور"
                            ),
                            info=f"Always includes: {DEFAULT_INITIAL_PROMPT_FULL}",
                        )
                with gr.Row(elem_classes=["upload-transcribe-row"]):
                    run_btn = gr.Button(
                        "Transcribe",
                        variant="primary",
                        elem_id="transcribe-btn",
                    )

                with gr.Column(elem_classes=["transcript-card"]):
                    gr.Markdown("### Transcript", elem_classes=["settings-tile-heading"])
                    transcript_box = gr.HTML(
                        value='<div class="transcript-body transcript-empty" dir="rtl" lang="ar"></div>',
                        elem_id="transcript-box",
                        label=None,
                        show_label=False,
                        container=False,
                        padding=False,
                        apply_default_css=False,
                    )
                    with gr.Row():
                        download_btn = gr.DownloadButton(
                            label="Download transcript ⬇",
                            visible=False,
                            elem_classes=["stt-download-btn"],
                            scale=1,
                        )
                    stats_md = gr.Markdown(elem_id="stats")

        # ── Tab 2: Live ───────────────────────────────────────────────────
        with gr.Tab("Live"):
            with gr.Column(elem_classes=["upload-layout"]):
                with gr.Column(elem_classes=["live-audio-card upload-audio-card"]):
                    gr.Markdown("### Microphone", elem_classes=["settings-tile-heading"])
                    with gr.Column(elem_id="live-mic-audio"):
                        mic_input = gr.Audio(
                            sources=["microphone"],
                            streaming=True,
                            type="numpy",
                            label=None,
                            show_label=False,
                        )

                with gr.Row(elem_classes=["upload-settings-row"]):
                    with gr.Column(elem_classes=["settings-tile"], scale=1):
                        gr.Markdown("### Model", elem_classes=["settings-tile-heading"])
                        model_picker_live = gr.Dropdown(
                            choices=list(MODEL_MAP.keys()),
                            value="turbo-4bit",
                            label=None,
                            show_label=False,
                            info="turbo-4bit recommended for live mode",
                        )
                    with gr.Column(elem_classes=["settings-tile"], scale=1):
                        gr.Markdown("### Chunk size", elem_classes=["settings-tile-heading"])
                        chunk_slider = gr.Slider(
                            minimum=3,
                            maximum=15,
                            value=6,
                            step=1,
                            label=None,
                            show_label=False,
                            info="Lower = faster · Higher = more accurate",
                        )
                    with gr.Column(elem_classes=["settings-tile"], scale=1):
                        gr.Markdown("### Actions", elem_classes=["settings-tile-heading"])
                        save_live_btn = gr.Button(
                            "Save transcript",
                            variant="secondary",
                            elem_id="save-live-btn",
                        )
                        clear_btn = gr.Button(
                            "Clear",
                            variant="secondary",
                            elem_id="clear-btn",
                        )

                with gr.Column(elem_classes=["transcript-card"]):
                    gr.Markdown("### Transcript", elem_classes=["settings-tile-heading"])
                    live_transcript = gr.Textbox(
                        lines=22,
                        elem_id="live-transcript",
                        placeholder="",
                        interactive=False,
                        label=None,
                        show_label=False,
                        rtl=True,
                    )
                    with gr.Row():
                        live_download_btn = gr.DownloadButton(
                            label="Download transcript ⬇",
                            visible=False,
                            elem_classes=["stt-download-btn"],
                            scale=1,
                        )
                    live_stats = gr.Markdown(elem_id="live-stats")

    # ── State ─────────────────────────────────────────────────────────────

    audio_buffer = gr.State(None)
    secs_done    = gr.State(0.0)
    chunk_count  = gr.State(0)

    # ── Events ────────────────────────────────────────────────────────────

    file_input.upload(
        fn=stage_upload_audio,
        inputs=[file_input],
        outputs=[file_input],
    )

    run_btn.click(
        fn=reset_transcript_ui,
        outputs=[transcript_box, download_btn, stats_md],
    ).then(
        fn=run_transcription,
        inputs=[
            file_input,
            model_picker,
            diarize_check,
            min_speakers,
            max_speakers,
            start_time,
            end_time,
            hint_input,
        ],
        outputs=[transcript_box, download_btn, stats_md],
        show_progress="minimal",
    )

    mic_input.stream(
        fn=process_live_chunk,
        inputs=[mic_input, audio_buffer, live_transcript, secs_done, chunk_count, model_picker_live, chunk_slider],
        outputs=[live_transcript, audio_buffer, secs_done, chunk_count, live_stats],
    )

    mic_input.stop_recording(
        fn=flush_live_buffer,
        inputs=[audio_buffer, live_transcript, secs_done, chunk_count, model_picker_live],
        outputs=[live_transcript, audio_buffer, secs_done, chunk_count, live_stats],
    )

    save_live_btn.click(
        fn=save_live_transcript,
        inputs=[live_transcript],
        outputs=[live_download_btn],
    )

    clear_btn.click(
        fn=clear_live,
        inputs=[],
        outputs=[live_transcript, audio_buffer, secs_done, chunk_count, live_stats, live_download_btn],
    )


if __name__ == "__main__":
    inbrowser = os.environ.get("INBROWSER", "").strip().lower() in ("1", "true", "yes", "y")
    launch_kw = _launch_kwargs(inbrowser=inbrowser)
    ssl = launch_kw.get("ssl_certfile")
    host = launch_kw["server_name"]
    port = launch_kw["server_port"]
    scheme = "https" if ssl else "http"
    print(f"Open: {scheme}://{host}:{port}", file=sys.stderr)
    demo.launch(**launch_kw)
