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

gr.set_static_paths([str(_ui_upload_dir.resolve())])

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
from diarize import run_diarization, speaker_label
from text_fix import fix_segments, is_available as text_fix_available

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
_PLAYBACK_EMPTY_HTML = ""
_PLAYBACK_NATIVE_SUFFIXES = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".webm", ".mpeg", ".mpga"}


def _audio_path_from_upload(audio) -> str | None:
    if audio is None:
        return None
    if isinstance(audio, str):
        return audio
    if isinstance(audio, dict):
        return audio.get("path") or audio.get("name")
    return getattr(audio, "path", None) or getattr(audio, "name", None)


def _file_serve_url(path: str) -> str:
    """URL for the HTML5 player (files under ui_uploads are static / inline)."""
    from urllib.parse import quote

    p = Path(path).resolve().as_posix()
    return f"/gradio_api/file={quote(p, safe='/:@')}"


def _playback_html(path: str | None) -> str:
    if not path or not Path(path).is_file():
        return _PLAYBACK_EMPTY_HTML
    src = _file_serve_url(path)
    return (
        f'<audio id="stt-playback-audio" controls preload="metadata" '
        f'style="width:100%;min-height:42px" src="{src}"></audio>'
    )


def _to_playback_mp3(src: Path) -> str:
    """Always serve MP3 from ui_uploads so the browser player gets a known-good file."""
    if not src.is_file():
        raise gr.Error("Uploaded file is missing or not readable.")
    if src.suffix.lower() == ".mp3" and src.parent.resolve() == _ui_upload_dir.resolve():
        return str(src.resolve())
    dest = _ui_upload_dir / f"play_{int(time.time() * 1000)}.mp3"
    ff = _ffmpeg_executable()
    cmd = [ff, "-y", "-i", str(src), "-vn", "-acodec", "libmp3lame", "-q:a", "2", str(dest)]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = (result.stderr or b"").decode(errors="replace")
        raise gr.Error(f"Could not prepare audio for playback.\n{err[:500]}")
    return str(dest.resolve())


def stage_upload_audio(audio) -> str | None:
    """
    Copy uploads into .gradio_temp/ui_uploads (an allowed_paths dir).

    Gradio serves files outside allowed_paths with Content-Disposition: attachment for
    many audio MIME types, which breaks the in-browser <audio> player. Paths here are
    served inline. Also avoids Arabic/special characters in Windows temp paths.
    """
    path = _audio_path_from_upload(audio)
    if not path:
        return None
    src = Path(path)
    if not src.is_file():
        return path
    try:
        if src.parent.resolve() == _ui_upload_dir.resolve():
            return str(src.resolve())
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


def stage_playback_path(file) -> str | None:
    """Path for the HTML5 player — copy only; transcode only when the browser cannot play the format."""
    staged = stage_upload_audio(file)
    if not staged:
        return None
    src = Path(staged)
    if src.suffix.lower() in _PLAYBACK_NATIVE_SUFFIXES:
        return staged
    return _to_playback_mp3(src)


def handle_audio_upload(file):
    """Refresh the HTML5 player only (do not rewrite the file input — avoids lag / double work)."""
    if file is None:
        return _PLAYBACK_EMPTY_HTML
    return _playback_html(stage_playback_path(file))


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
    justify-content: center;
    padding: 16px 24px;
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
    gap: 8px !important;
}
#stt-playback,
#stt-playback > .form,
#stt-playback > .block,
#stt-playback > div {
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#stt-playback audio,
#stt-playback-audio {
    width: 100% !important;
    min-height: 42px !important;
    display: block !important;
}
.stt-audio-hint {
    margin: 0;
    font-size: 0.78rem;
    color: #9e9188;
    text-align: center;
}
#upload-audio .wrap {
    min-height: 0 !important;
}
/* crop timeline under upload audio */
.stt-crop-timeline {
    margin-top: 6px;
    width: 100%;
    user-select: none;
}
.stt-crop-timeline.inactive { display: none; }
#transcript-box .word,
#transcript-box .ts {
    cursor: pointer;
}
.stt-crop-readout {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #9e9188;
    margin-bottom: 4px;
    direction: ltr;
    text-align: left;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.stt-crop-track {
    position: relative;
    height: 22px;
    background: #e8e2d6;
    border-radius: 4px;
    cursor: pointer;
    overflow: hidden;
}
.stt-crop-track::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        90deg,
        transparent,
        transparent 9.5%,
        rgba(0,0,0,0.04) 9.5%,
        rgba(0,0,0,0.04) 10%
    );
    pointer-events: none;
}
.stt-crop-region {
    position: absolute;
    top: 0;
    bottom: 0;
    background: rgba(192, 120, 64, 0.35);
    border-left: 2px solid #a08060;
    border-right: 2px solid #a08060;
    pointer-events: none;
    z-index: 2;
}
.stt-crop-playhead {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 2px;
    background: #2c2825;
    transform: translateX(-1px);
    pointer-events: none;
    z-index: 3;
}

/* ── upload: settings grid (each option in its own beige tile) ── */
.upload-settings-card {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    margin-top: 12px !important;
    gap: 12px !important;
}
.upload-settings-card > .block,
.upload-settings-card > .form,
.upload-settings-card > .settings-tile-heading,
.upload-settings-card .settings-tile .settings-tile-heading,
.upload-settings-card .prose {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
.upload-settings-card .settings-tile > .block,
.upload-settings-card .settings-tile > .form {
    box-shadow: none !important;
    background: transparent !important;
}
.upload-settings-card .prose hr {
    display: none !important;
}
.upload-settings-cols {
    flex-wrap: wrap !important;
    gap: 8px 16px !important;
    align-items: flex-start !important;
    width: 100% !important;
}
.upload-settings-col {
    flex: 1 1 280px !important;
    min-width: 0 !important;
    gap: 8px !important;
    display: flex !important;
    flex-direction: column !important;
}
.upload-settings-card .settings-tile {
    width: 100% !important;
    margin: 0 !important;
}
/* tighten Gradio's default inner block padding inside tiles */
.upload-settings-card .settings-tile > .block,
.upload-settings-card .settings-tile > .form {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
.upload-settings-card .hints-tile {
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
}
.upload-settings-card .hints-tile > .block,
.upload-settings-card .hints-tile > div {
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
}
.upload-settings-card .hints-tile textarea {
    flex: 1 1 auto !important;
    resize: none !important;
}
.settings-section-heading,
.settings-section-heading p,
.settings-section-heading h3 {
    margin: 0 0 10px 0 !important;
    color: #9e9188 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
#diarization-group,
#crop-group {
    background: #faf8f4 !important;
    border: 1px solid #e8e2d6 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    padding: 10px 12px !important;
    gap: 0 !important;
}
/* strip inner wrappers — checkbox and row sit flat inside the group */
#diarization-group > .block,
#diarization-group > .form,
#diarization-group > .row,
#diarization-group > div,
#crop-group > .block,
#crop-group > .form,
#crop-group > div {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}
/* space between checkbox and the speakers row */
#diarization-group #speaker-range-inputs {
    margin-top: 14px !important;
}
#diarization-group #speaker-range-inputs,
#diarization-group #speaker-range-inputs > .block,
#diarization-group #speaker-range-inputs > .form,
#diarization-group #speaker-range-inputs .block,
#diarization-group #speaker-range-inputs .form,
#diarization-group #speaker-range-inputs .wrap {
    gap: 8px !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
}
/* white background on speaker dropdowns to match other inputs */
#diarization-group #speaker-range-inputs .wrap > ul,
#diarization-group #speaker-range-inputs input,
#diarization-group #speaker-range-inputs .dropdown-arrow,
#diarization-group #speaker-range-inputs .secondary-wrap {
    background: #ffffff !important;
}
/* align dropdown input flush under its label */
#diarization-group #speaker-range-inputs .block,
#diarization-group #speaker-range-inputs .form {
    padding: 0 !important;
    margin: 0 !important;
}
#diarization-group #speaker-range-inputs .label-wrap {
    padding: 0 !important;
    margin: 0 0 4px 0 !important;
}
#diarization-group #speaker-range-inputs .wrap {
    margin: 0 !important;
    padding: 0 !important;
}
.settings-tile {
    background: #f4f0e8 !important;
    border: 1px solid #e8e2d6 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    flex: 1 1 0 !important;
    min-width: 0 !important;
}
.upload-settings-row {
    flex-wrap: nowrap !important;
    gap: 12px !important;
    align-items: stretch !important;
    width: 100% !important;
}
.settings-tile-heading,
.settings-tile-heading p {
    margin: 0 0 4px 0 !important;
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
/* From / To labels inside Crop tile — match Gradio label style */
.crop-hms-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #9e9188;
    letter-spacing: 0;
    text-transform: none;
    margin: 8px 0 3px;
    line-height: 1.2;
}
#start-end-crop-inputs > div:first-child .crop-hms-label,
#start-end-crop-inputs .crop-hms-label:first-of-type {
    margin-top: 0;
}
.crop-hms-row {
    gap: 6px !important;
    flex-wrap: nowrap !important;
}
#start-end-crop-inputs .crop-hms-row label,
#start-end-crop-inputs .crop-hms-row .label-wrap span {
    font-size: 0.68rem !important;
}
@keyframes reveal-flash {
    0%   { box-shadow: 0 0 0 2px rgba(160,128,96,0.55); border-radius: 6px; }
    60%  { box-shadow: 0 0 0 2px rgba(160,128,96,0.55); border-radius: 6px; }
    100% { box-shadow: 0 0 0 0   rgba(160,128,96,0);    border-radius: 6px; }
}
.reveal-flash {
    animation: reveal-flash 1.4s ease-out forwards !important;
}
#start-end-crop-inputs {
    padding: 0 !important;
    gap: 6px !important;
    border: none !important;
    box-shadow: none !important;
}
#start-end-crop-inputs > .block,
#start-end-crop-inputs > .form {
    padding: 0 !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
#start-end-crop-inputs .block {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}
.crop-hms-hint,
.crop-hms-hint p {
    font-size: 0.68rem !important;
    color: #b0a69c !important;
    margin: 4px 0 0 !important;
    line-height: 1.2 !important;
}
.settings-tile label,
.settings-tile .label-wrap span,
.upload-settings-card label,
.upload-settings-card .label-wrap span {
    text-transform: none !important;
    letter-spacing: 0 !important;
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
    .upload-settings-col { flex: 1 1 100% !important; }
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
    text-align: left !important;
    direction: ltr !important;
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

/* White inner panel: progress bars + transcript body (inside beige .transcript-card) */
.transcript-panel {
    background: #ffffff !important;
    border: 1px solid #e8e2d6 !important;
    border-radius: 8px !important;
    min-height: 380px;
    max-height: min(70vh, 720px);
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    width: 100% !important;
    min-width: 0 !important;
}
/* ── clickable transcript (upload tab) ── */
/* Gradio wraps gr.HTML in extra divs — do not style those (avoids a second scroll strip). */
.transcript-panel #transcript-box,
.transcript-panel #transcript-box > .form,
.transcript-panel #transcript-box > .block,
.transcript-panel #transcript-box > div {
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    flex: 1 1 auto !important;
    overflow: visible !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}
/* make the transcript body itself scroll within the panel */
#transcript-box .transcript-body {
    overflow-y: auto !important;
    max-height: calc(min(70vh, 720px) - 60px) !important;
    padding-right: 4px !important;
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
    background: transparent;
    border: none;
    border-radius: 0;
    min-height: 200px;
    max-height: none;
    padding: 16px;
    overflow-y: auto;
    color: #2c2825;
    direction: rtl;
    text-align: right;
    unicode-bidi: isolate;
}
#transcript-box .transcript-body.transcript-empty {
    min-height: 200px;
    cursor: text;
}
#transcript-box .stt-transcript-caret {
    display: inline-block;
    width: 2px;
    height: 1.15em;
    background: #a08060;
    vertical-align: text-top;
    animation: stt-caret-blink 1.05s step-end infinite;
}
@keyframes stt-caret-blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
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
/* ── stacked pipeline progress (above transcript card) ── */
#progress-steps,
#progress-steps > .form,
#progress-steps > .block,
#progress-steps > div {
    margin: 0 0 10px 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    min-height: 0 !important;
}
#progress-steps:not(:has(.stt-step-row)) {
    display: none !important;
    margin: 0 !important;
}
#progress-steps:has(.stt-step-row) {
    margin-top: 16px !important;
    margin-bottom: 10px !important;
}
#progress-steps { min-height: 0 !important; }
#progress-steps .stt-steps {
    display: flex;
    flex-direction: column;
    gap: 6px;
    width: 100%;
}
#progress-steps .stt-step-row {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.78rem;
    color: #9e9188;
}
#progress-steps .stt-step-row.active { color: #2c2825; font-weight: 600; }
#progress-steps .stt-step-row.done { color: #6b8f71; }
#progress-steps .stt-step-label { flex: 0 0 auto; min-width: 7.5em; text-align: start; }
#progress-steps .stt-step-time {
    flex: 0 0 auto;
    min-width: 5.5em;
    text-align: end;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #9e9188;
    white-space: nowrap;
}
#progress-steps .stt-step-row.active .stt-step-time { color: #2c2825; }
#progress-steps .stt-step-row.done .stt-step-time { color: #6b8f71; }
#progress-steps .stt-step-track {
    flex: 1 1 auto;
    height: 6px;
    background: #e8e2d6;
    border-radius: 3px;
    overflow: hidden;
}
#progress-steps .stt-step-fill {
    height: 100%;
    background: #a08060;
    border-radius: 3px;
    transition: width 0.25s ease;
}
#progress-steps .stt-step-row.done .stt-step-fill { background: #6b8f71; }

#transcript-box .ts {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72em;
    color: #c4bab0;
    cursor: pointer;
    margin-inline-end: 8px;
    user-select: none;
    unicode-bidi: isolate;
    direction: rtl;
    display: inline-flex;
    flex-direction: row;
    align-items: center;
    gap: 0.35em;
}
#transcript-box .ts-from,
#transcript-box .ts-to {
    direction: ltr;
    unicode-bidi: isolate;
}
#transcript-box .ts-sep {
    direction: ltr;
    opacity: 0.85;
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
.hints-tile .info {
    font-size: 0.875rem !important;
    color: #b0a69c !important;
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
    '<div class="transcript-body transcript-empty" dir="rtl" lang="ar">'
    '<span class="stt-transcript-caret" aria-hidden="true"></span></div>'
)

_EMPTY_PROGRESS_HTML = '<div class="stt-steps"></div>'


def _pipeline_steps(do_diarize: bool, do_text_fix: bool) -> list[str]:
    steps: list[str] = []
    if do_diarize:
        steps.append("Speakers")
    steps.append("Transcribing")
    if do_text_fix and text_fix_available():
        steps.append("Cleaning up")
    return steps


def hms_to_seconds(h, m, s) -> float:
    """Convert hour / minute / second fields to total seconds."""
    hours = max(0, int(h or 0))
    minutes = max(0, int(m or 0))
    secs = max(0.0, float(s or 0))
    if minutes > 59:
        raise gr.Error("Crop minutes must be between 0 and 59.")
    if secs >= 60:
        raise gr.Error("Crop seconds must be between 0 and 59.")
    return hours * 3600 + minutes * 60 + secs


def fmt_duration_en(seconds: float) -> str:
    """Human-readable English duration (seconds / minutes / hours)."""
    s = max(0, int(round(seconds)))
    if s == 0:
        return "0 seconds"
    if s < 60:
        return f"{s} second" if s == 1 else f"{s} seconds"
    if s < 3600:
        m, sec = divmod(s, 60)
        parts = []
        if m:
            parts.append(f"{m} minute" if m == 1 else f"{m} minutes")
        if sec:
            parts.append(f"{sec} second" if sec == 1 else f"{sec} seconds")
        return " and ".join(parts)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h} hour" if h == 1 else f"{h} hours")
    if m:
        parts.append(f"{m} minute" if m == 1 else f"{m} minutes")
    if sec:
        parts.append(f"{sec} second" if sec == 1 else f"{sec} seconds")
    return " and ".join(parts)


def _render_steps_progress(
    steps: list[str],
    active: int,
    within: float = 0.0,
    step_times: dict[int, float] | None = None,
    step_started: dict[int, float] | None = None,
) -> str:
    if not steps:
        return _EMPTY_PROGRESS_HTML
    within = max(0.0, min(1.0, within))
    step_times = step_times or {}
    step_started = step_started or {}
    now = time.time()
    rows = []
    for i, label in enumerate(steps):
        if i < active:
            state, pct = "done", 100
        elif i == active:
            state, pct = "active", int(within * 100)
        else:
            state, pct = "pending", 0
        if i in step_times:
            time_label = fmt_duration_en(step_times[i])
        elif i == active and i in step_started:
            time_label = fmt_duration_en(now - step_started[i])
        else:
            time_label = ""
        time_html = (
            f'<span class="stt-step-time">{time_label}</span>' if time_label else ""
        )
        rows.append(
            f'<div class="stt-step-row {state}">'
            f'<span class="stt-step-label">{label}</span>'
            f'<div class="stt-step-track"><div class="stt-step-fill" '
            f'style="width:{pct}%"></div></div>'
            f"{time_html}"
            f"</div>"
        )
    return '<div class="stt-steps">' + "".join(rows) + "</div>"


class _PipelineProgress:
    def __init__(self, gr_progress, steps: list[str]):
        self._p = gr_progress
        self.steps = steps
        self._n = max(len(steps), 1)
        self._step_times: dict[int, float] = {}
        self._step_started: dict[int, float] = {}
        self._active = -1

    def _close_step(self, idx: int) -> None:
        if idx < 0 or idx in self._step_times:
            return
        started = self._step_started.get(idx)
        if started is not None:
            self._step_times[idx] = time.time() - started

    def tick(self, step: int, within: float = 0.0, desc: str | None = None) -> str:
        within = max(0.0, min(1.0, within))
        if self._active >= 0 and step != self._active:
            self._close_step(self._active)
        if step not in self._step_started:
            self._step_started[step] = time.time()
        self._active = step
        if within >= 1.0:
            self._close_step(step)
        total = min(0.99, (step + within) / self._n)
        label = desc or self.steps[step]
        self._p(total, desc=label)
        return _render_steps_progress(
            self.steps, step, within, self._step_times, self._step_started
        )


def _pending_transcription_outputs(progress_html: str):
    return (
        _EMPTY_TRANSCRIPT_HTML,
        gr.DownloadButton(visible=False, elem_classes=["stt-download-btn"]),
        "",
        progress_html,
    )




def reset_transcript_ui():
    """Clear transcript area when starting a new run."""
    return (
        _EMPTY_TRANSCRIPT_HTML,
        gr.DownloadButton(visible=False, elem_classes=["stt-download-btn"]),
        "",
        _EMPTY_PROGRESS_HTML,
    )


def segments_to_html(segments, time_offset: float = 0.0) -> str:
    """Renders transcript as clickable word spans. Clicking a word seeks the audio player."""
    if not segments:
        return _EMPTY_TRANSCRIPT_HTML + "\n"

    parts = []
    for seg in segments:
        if not seg.get("text", "").strip():
            continue

        speaker = seg.get("speaker", "")
        spk_html = f'<span class="spk">{speaker}</span>' if speaker else ""
        off = float(time_offset or 0)
        seg_start = float(seg["start"]) + off
        seg_end = float(seg["end"]) + off
        ts_html = (
            f'<span class="ts" data-seek="{seg_start:.3f}" title="Seek to start">'
            f'<span class="ts-from">{fmt_ts(seg_start)}</span>'
            f'<span class="ts-sep" aria-hidden="true">←</span>'
            f'<span class="ts-to">{fmt_ts(seg_end)}</span>'
            f"</span>"
        )

        words = seg.get("words") or []
        if words:
            word_spans = []
            for w in words:
                t = float(w.get("start", seg["start"])) + off
                word_text = w.get("word", "").strip()
                if word_text:
                    word_spans.append(
                        f'<span class="word" data-seek="{t:.3f}">{word_text}</span>'
                    )
            words_html = " ".join(word_spans)
        else:
            t = seg_start
            words_html = (
                f'<span class="word" data-seek="{t:.3f}">'
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


def segments_to_file(segments, time_offset: float = 0.0) -> str:
    off = float(time_offset or 0)
    lines = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        speaker = seg.get("speaker", "")
        prefix = f"{speaker}  " if speaker else ""
        start = float(seg["start"]) + off
        end = float(seg["end"]) + off
        lines.append(f"{prefix}[{fmt_ts_ms(end)} ← {fmt_ts_ms(start)}] {text}")
    return "\n".join(lines)


def _save_diarization(turns: list, source_path: str, time_offset: float = 0.0) -> None:
    off = float(time_offset or 0)
    stem = Path(source_path).stem
    out = Path(__file__).parent / f"{stem}_diarization.txt"
    lines = [
        f"[{fmt_ts_ms(e + off)} ← {fmt_ts_ms(s + off)}]  {spk}" for s, e, spk in turns
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def _save_transcript(segments: list, source_path: str, time_offset: float = 0.0) -> None:
    stem = Path(source_path).stem
    out = Path(__file__).parent / f"{stem}_transcript.txt"
    out.write_text(segments_to_file(segments, time_offset), encoding="utf-8")


def _save_word_diarization(segments: list, source_path: str, time_offset: float = 0.0) -> None:
    off = float(time_offset or 0)
    stem = Path(source_path).stem
    out = Path(__file__).parent / f"{stem}_words.txt"
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "")
        for w in seg.get("words") or []:
            word = w.get("word", "").strip()
            if not word:
                continue
            start = fmt_ts_ms(float(w.get("start", seg["start"])) + off)
            end = fmt_ts_ms(float(w.get("end", seg["end"])) + off)
            lines.append(f"{speaker}  [{end} ← {start}]  {word}")
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


def _extract_mp3_clip(input_path: str, start: float, end: float) -> str:
    """Extract [start, end] seconds to a temp MP3 (16 kHz mono), matching Z_ArabicSTT extract_clip."""
    if end <= start:
        raise ValueError("clip end must be greater than start")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    ff = _ffmpeg_executable()
    cmd = [
        ff,
        "-y",
        "-ss",
        str(start),
        "-to",
        str(end),
        "-i",
        input_path,
        "-f",
        "mp3",
        "-ac",
        "1",
        "-ar",
        "16000",
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        Path(tmp.name).unlink(missing_ok=True)
        err = (result.stderr or b"").decode(errors="replace")
        raise RuntimeError(f"ffmpeg clip extract failed:\n{err}")
    return tmp.name


def _diarization_speaker_bounds(min_spk, max_spk) -> tuple[int, int]:
    """Default min=2, max=5 (meeting_transcriber). 'auto' leaves bounds to pyannote."""
    min_v = 2 if not min_spk or min_spk == "auto" else int(min_spk)
    max_v = 5 if not max_spk or max_spk == "auto" else int(max_spk)
    if min_spk == "auto":
        min_v = 0
    if max_spk == "auto":
        max_v = 0
    return min_v, max_v


# ── Upload tab ────────────────────────────────────────────────────────────────

def run_transcription(
    file,
    model_key,
    do_diarize,
    min_spk,
    max_spk,
    start_h,
    start_m,
    start_s,
    end_h,
    end_m,
    end_s,
    user_hint,
    do_text_fix,
    progress=gr.Progress(),
):
    if file is None:
        raise gr.Error("Please upload an audio file first.")

    crop_start = hms_to_seconds(start_h, start_m, start_s)
    crop_end = hms_to_seconds(end_h, end_m, end_s)
    steps = _pipeline_steps(do_diarize, do_text_fix)
    pipe = _PipelineProgress(progress, steps)
    transcribe_step_idx = steps.index("Transcribing")
    speakers_step_idx = steps.index("Speakers") if do_diarize else -1
    fix_step = steps.index("Cleaning up") if "Cleaning up" in steps else -1

    yield _pending_transcription_outputs(pipe.tick(0, 0.0))

    evict_model_if_changed(model_key)
    t0 = time.time()

    raw_stem = Path(str(file)).stem.strip() or "audio"
    _bad = '<>:"/\\|?*'
    out_stem = ("".join(c if c not in _bad else "_" for c in raw_stem))[:120] or "audio"
    saves_ref = f"{out_stem}.wav"

    work_path = _copy_upload_for_processing(str(file))
    cleaned = None
    dia_audio = None
    segments: list[dict] = []
    try:
        try:
            if do_diarize:
                dia_audio = extract_audio_for_diarization(
                    work_path,
                    start=crop_start,
                    end=crop_end,
                )
            else:
                cleaned = preprocess_audio(
                    work_path,
                    start=crop_start,
                    end=crop_end,
                )
        except RuntimeError as e:
            raise gr.Error(str(e)) from e

        yield _pending_transcription_outputs(pipe.tick(0, 0.08))

        try:
            if do_diarize:
                dia_min, dia_max = _diarization_speaker_bounds(min_spk, max_spk)
                turns_box: list = [None]
                dia_err: list = [None]

                def _dia_job():
                    try:
                        try:
                            import mlx.core as mx  # type: ignore

                            mx.clear_cache()
                        except Exception:
                            pass
                        turns_box[0] = run_diarization(
                            dia_audio,
                            min_speakers=dia_min,
                            max_speakers=dia_max,
                        )
                    except Exception as e:
                        dia_err[0] = e

                t_d = threading.Thread(target=_dia_job, daemon=True)
                t_d.start()
                dia_within = 0.05
                last_yield = time.time()
                while t_d.is_alive():
                    time.sleep(0.35)
                    dia_within = min(0.92, dia_within + 0.03)
                    if time.time() - last_yield >= 0.45:
                        yield _pending_transcription_outputs(
                            pipe.tick(
                                speakers_step_idx,
                                dia_within,
                                desc="Speakers (pyannote)",
                            )
                        )
                        last_yield = time.time()
                t_d.join()
                if dia_err[0]:
                    raise gr.Error(str(dia_err[0])) from dia_err[0]
                turns = turns_box[0] or []
                if not turns:
                    raise gr.Error(
                        "No speech detected by speaker diarization — try adjusting min/max speakers "
                        "or the audio crop."
                    )
                _save_diarization(turns, saves_ref, time_offset=crop_start)
                yield _pending_transcription_outputs(
                    pipe.tick(speakers_step_idx, 1.0, desc="Speakers")
                )

                n_turns = len(turns)
                segments = []
                for i, (t_start, t_end, spk) in enumerate(turns):
                    yield _pending_transcription_outputs(
                        pipe.tick(
                            transcribe_step_idx,
                            i / max(n_turns, 1),
                            desc=f"Transcribing turn {i + 1} of {n_turns}",
                        )
                    )
                    clip_path = _extract_mp3_clip(dia_audio, t_start, t_end)
                    try:
                        res = transcribe_any(
                            clip_path,
                            model_key=model_key,
                            live=False,
                            user_hint=None,
                            diarize_turn=True,
                        )
                    finally:
                        Path(clip_path).unlink(missing_ok=True)

                    spk_label = speaker_label(spk)
                    for s in res.get("segments", []):
                        txt = (s.get("text") or "").strip()
                        if not txt:
                            continue
                        abs_start = t_start + float(s["start"])
                        abs_end = t_start + float(s["end"])
                        if segments and txt == segments[-1]["text"].strip():
                            segments[-1]["end"] = abs_end
                            continue
                        seg_dict: dict = {
                            "start": abs_start,
                            "end": abs_end,
                            "text": s["text"],
                            "speaker": spk_label,
                        }
                        words = s.get("words")
                        if words:
                            seg_dict["words"] = [
                                {
                                    "start": t_start + float(w.get("start", 0)),
                                    "end": t_start + float(w.get("end", 0)),
                                    "word": w.get("word", ""),
                                }
                                for w in words
                            ]
                        segments.append(seg_dict)

                if not segments:
                    raise gr.Error("No speech detected in the file.")

                yield _pending_transcription_outputs(
                    pipe.tick(transcribe_step_idx, 1.0, desc="Transcribing")
                )
                _save_word_diarization(segments, saves_ref, time_offset=crop_start)

            else:
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

                within = 0.1
                last_yield = time.time()
                while t.is_alive():
                    time.sleep(0.35)
                    within = min(0.92, within + 0.04)
                    if time.time() - last_yield >= 0.45:
                        yield _pending_transcription_outputs(
                            pipe.tick(transcribe_step_idx, within)
                        )
                        last_yield = time.time()
                t.join()

                if _err[0]:
                    raise gr.Error(str(_err[0])) from _err[0]
                result = _result[0]

                segments = result.get("segments", [])
                if not segments:
                    raise gr.Error("No speech detected in the file.")

                yield _pending_transcription_outputs(pipe.tick(transcribe_step_idx, 1.0))

        finally:
            if cleaned:
                Path(cleaned).unlink(missing_ok=True)
            if dia_audio:
                Path(dia_audio).unlink(missing_ok=True)
    finally:
        Path(work_path).unlink(missing_ok=True)

    if do_text_fix:
        if not text_fix_available():
            gr.Warning(
                "Local typo fix needs transformers installed. "
                "Run once online: pip install -r requirements.txt && python prefetch_models.py"
            )
        else:
            yield _pending_transcription_outputs(pipe.tick(fix_step, 0.0))

            def _fix_progress(frac: float, desc: str):
                pipe.tick(fix_step, frac, desc or "Cleaning up")

            try:
                segments = fix_segments(
                    segments,
                    user_hint=user_hint or "",
                    on_progress=_fix_progress,
                )
            except Exception as e:
                gr.Warning(f"Typo fix skipped: {e}")

            yield _pending_transcription_outputs(pipe.tick(fix_step, 1.0))

    elapsed = time.time() - t0
    duration = segments[-1]["end"]

    out_fd, out_path_str = tempfile.mkstemp(suffix=".txt")
    os.close(out_fd)
    out_path = Path(out_path_str)
    out_path.write_text(segments_to_file(segments, crop_start), encoding="utf-8")

    _save_transcript(segments, saves_ref, time_offset=crop_start)

    stats = (
        f"Recording length: {fmt_duration_en(duration)} · "
        f"Processing time: {fmt_duration_en(elapsed)}"
    )

    for i in range(len(steps)):
        pipe._close_step(i)
    done_html = _render_steps_progress(
        steps, len(steps), 0.0, pipe._step_times, pipe._step_started
    )
    progress(1.0, desc="Done")
    yield (
        segments_to_html(segments, time_offset=crop_start),
        gr.DownloadButton(
            value=str(out_path),
            visible=True,
            label="Download transcript ⬇",
            elem_classes=["stt-download-btn"],
        ),
        stats,
        done_html,
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
        function getUploadAudio() {
            const direct = document.getElementById('stt-playback-audio');
            if (direct) return direct;
            const root = document.querySelector('#upload-audio');
            return root ? root.querySelector('audio') : null;
        }

        function seekAudio(t) {
            const audio = getUploadAudio();
            if (!audio) return;
            const dur = audio.duration;
            if (!isFinite(dur) || dur <= 0) {
                const onMeta = () => {
                    audio.removeEventListener('loadedmetadata', onMeta);
                    seekAudio(t);
                };
                audio.addEventListener('loadedmetadata', onMeta);
                return;
            }
            audio.currentTime = Math.max(0, Math.min(t, dur));
            if (audio.paused) audio.play().catch(() => {});
            updateCropTimeline();
        }

        function bindTranscriptSeek() {
            const box = document.getElementById('transcript-box');
            if (!box || box.dataset.sttSeekBound === '1') return;
            box.dataset.sttSeekBound = '1';
            box.addEventListener('click', (e) => {
                const el = e.target.closest('.word, .ts');
                if (!el || !box.contains(el)) return;
                const t = parseFloat(el.dataset.seek);
                if (Number.isFinite(t)) seekAudio(t);
            });
        }

        function formatTime(s) {
            s = Math.max(0, Math.floor(s || 0));
            const h = Math.floor(s / 3600);
            const m = Math.floor((s % 3600) / 60);
            const sec = s % 60;
            const pad = (n) => String(n).padStart(2, '0');
            return pad(h) + ':' + pad(m) + ':' + pad(sec);
        }

        function getCropInputs() {
            return document.querySelectorAll('#start-end-crop-inputs input[type=number]');
        }

        function hmsFromInputs(inputs, base) {
            const h = parseFloat(inputs[base]?.value) || 0;
            const m = parseFloat(inputs[base + 1]?.value) || 0;
            const s = parseFloat(inputs[base + 2]?.value) || 0;
            return Math.max(0, h * 3600 + m * 60 + s);
        }

        function readCropSeconds(dur) {
            const inputs = getCropInputs();
            const start = inputs.length >= 3 ? hmsFromInputs(inputs, 0) : 0;
            let end = inputs.length >= 6 ? hmsFromInputs(inputs, 3) : 0;
            if (!end || end <= start) end = dur;
            end = Math.min(end, dur);
            return { start: Math.max(0, Math.min(start, dur)), end };
        }

        function updateCropTimeline() {
            const box = document.getElementById('stt-crop-timeline');
            if (!box) return;
            const audio = getUploadAudio();
            const region = box.querySelector('.stt-crop-region');
            const playhead = box.querySelector('.stt-crop-playhead');
            const readout = box.querySelector('.stt-crop-readout');
            if (!region || !playhead || !readout) return;
            const dur = (audio && audio.duration && isFinite(audio.duration)) ? audio.duration : 0;
            if (!dur) {
                box.classList.add('inactive');
                readout.textContent = '';
                region.style.display = 'none';
                return;
            }
            box.classList.remove('inactive');
            const crop = readCropSeconds(dur);
            const pos = audio ? (audio.currentTime || 0) : 0;
            const cropLabel = crop.end < dur
                ? (formatTime(crop.start) + ' → ' + formatTime(crop.end))
                : (formatTime(crop.start) + ' → end');
            readout.textContent =
                'Position: ' + formatTime(pos) +
                ' · Crop: ' + cropLabel +
                ' · Total: ' + formatTime(dur);
            if (crop.end > crop.start) {
                region.style.display = 'block';
                region.style.left = (crop.start / dur * 100) + '%';
                region.style.width = ((crop.end - crop.start) / dur * 100) + '%';
            } else {
                region.style.display = 'none';
            }
            playhead.style.left = (Math.min(pos, dur) / dur * 100) + '%';
        }

        function seekFromTimeline(clientX) {
            const audio = getUploadAudio();
            const box = document.getElementById('stt-crop-timeline');
            const track = box && box.querySelector('.stt-crop-track');
            if (!audio || !track || !audio.duration) return;
            const rect = track.getBoundingClientRect();
            const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
            audio.currentTime = ratio * audio.duration;
            updateCropTimeline();
        }

        function bindCropTimeline() {
            const box = document.getElementById('stt-crop-timeline');
            if (!box || box.dataset.sttBound === '1') return;
            const track = box.querySelector('.stt-crop-track');
            if (!track) { setTimeout(bindCropTimeline, 400); return; }
            box.dataset.sttBound = '1';
            let dragging = false;
            const onDown = (ev) => {
                dragging = true;
                seekFromTimeline(ev.clientX);
                ev.preventDefault();
            };
            const onMove = (ev) => {
                if (!dragging) return;
                seekFromTimeline(ev.clientX);
            };
            const onUp = () => { dragging = false; };
            track.addEventListener('mousedown', onDown);
            window.addEventListener('mousemove', onMove);
            window.addEventListener('mouseup', onUp);
        }

        function watchCropInputs() {
            const inputs = getCropInputs();
            if (inputs.length < 2) { setTimeout(watchCropInputs, 500); return; }
            inputs.forEach((inp) => {
                if (inp.dataset.sttCropWatch) return;
                inp.dataset.sttCropWatch = '1';
                inp.addEventListener('input', updateCropTimeline);
                inp.addEventListener('change', updateCropTimeline);
            });
            updateCropTimeline();
        }

        let _sttBoundUploadAudio = null;
        function watchUploadAudio() {
            const audio = getUploadAudio();
            if (!audio) { setTimeout(watchUploadAudio, 400); return; }
            if (audio === _sttBoundUploadAudio) return;
            _sttBoundUploadAudio = audio;
            ['loadedmetadata', 'durationchange', 'timeupdate', 'seeked', 'error'].forEach((ev) => {
                audio.addEventListener(ev, updateCropTimeline);
            });
            updateCropTimeline();
        }

        function initUploadAudio() {
            bindCropTimeline();
            watchCropInputs();
            watchUploadAudio();
            bindTranscriptSeek();
        }
        document.addEventListener('DOMContentLoaded', initUploadAudio);
        setTimeout(initUploadAudio, 600);
        const transcriptSeekObs = new MutationObserver(() => bindTranscriptSeek());
        document.addEventListener('DOMContentLoaded', () => {
            const tb = document.getElementById('transcript-box');
            if (tb) transcriptSeekObs.observe(tb, { childList: true, subtree: true });
        });
        let _cropTimelineTimer = null;
        const uploadPreviewObs = new MutationObserver(() => {
            watchUploadAudio();
            if (_cropTimelineTimer) clearTimeout(_cropTimelineTimer);
            _cropTimelineTimer = setTimeout(updateCropTimeline, 80);
        });
        document.addEventListener('DOMContentLoaded', () => {
            const root = document.getElementById('stt-playback')
                || document.querySelector('#upload-audio');
            if (root) uploadPreviewObs.observe(root, { childList: true, subtree: true });
        });

        function fitWaveform(containerId) {
            if (containerId === '#upload-audio') return;
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

        function fitLiveWaveform() { fitWaveform('#live-mic-audio'); }
        window.addEventListener('resize', fitLiveWaveform);

        // ── reveal-flash on section show ──────────────────────────────────
        function watchReveal(id) {
            function flash(el) {
                el.classList.remove('reveal-flash');
                void el.offsetWidth;
                el.classList.add('reveal-flash');
                el.addEventListener('animationend', function handler() {
                    el.classList.remove('reveal-flash');
                    el.removeEventListener('animationend', handler);
                });
            }
            function attach() {
                const el = document.getElementById(id);
                if (!el) { setTimeout(attach, 400); return; }
                let wasHidden = (el.style.display === 'none' || el.closest('[style*="display: none"]'));
                new MutationObserver(function() {
                    const hidden = (el.style.display === 'none');
                    if (wasHidden && !hidden) flash(el);
                    wasHidden = hidden;
                }).observe(el, { attributes: true, attributeFilter: ['style'] });
            }
            attach();
        }
        watchReveal('start-end-crop-inputs');
        watchReveal('speaker-range-inputs');
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
                        file_input = gr.File(
                            label=None,
                            show_label=False,
                            file_count="single",
                            file_types=[
                                ".mp3", ".wav", ".m4a", ".ogg", ".flac",
                                ".webm", ".mp4", ".mpeg", ".mpga", ".aac",
                            ],
                            type="filepath",
                        )
                        playback_html = gr.HTML(
                            value=_PLAYBACK_EMPTY_HTML,
                            elem_id="stt-playback",
                            show_label=False,
                            container=False,
                            padding=False,
                        )
                    gr.HTML(
                            """
                            <div id="stt-crop-timeline" class="stt-crop-timeline inactive">
                              <div class="stt-crop-readout"></div>
                              <div class="stt-crop-track" title="Click or drag to seek">
                                <div class="stt-crop-region" style="display:none"></div>
                                <div class="stt-crop-playhead" style="left:0%"></div>
                              </div>
                            </div>
                            """,
                            show_label=False,
                            container=False,
                            padding=False,
                        )

                with gr.Column(elem_classes=["upload-settings-card"]):
                    with gr.Row(elem_classes=["upload-settings-cols"]):
                        with gr.Column(elem_classes=["upload-settings-col"]):
                            with gr.Column(elem_classes=["settings-tile"]):
                                gr.Markdown("### Model", elem_classes=["settings-tile-heading"])
                                model_picker = gr.Dropdown(
                                    choices=[
                                        ("Fast + Accurate [large-v3-4bit]", "large-v3-4bit"),
                                        ("Turbo Fast [turbo-4bit]",         "turbo-4bit"),
                                        ("Max Accuracy [large-v3]",         "large-v3"),
                                        ("Turbo [turbo]",                   "turbo"),
                                        ("Medium [medium]",                 "medium"),
                                        ("Small [small]",                   "small"),
                                    ],
                                    value="large-v3",
                                    label=None,
                                    show_label=False,
                                    filterable=False,
                                    allow_custom_value=False,
                                )
                            with gr.Column(elem_classes=["settings-tile"]):
                                gr.Markdown("### Cleaning", elem_classes=["settings-tile-heading"])
                                text_fix_check = gr.Checkbox(
                                    label="Fix Typos",
                                    value=text_fix_available(),
                                    interactive=text_fix_available(),
                                    elem_id="text-fix-check",
                                    info=(
                                        None
                                        if text_fix_available()
                                        else "Requires transformers (pip install -r requirements.txt)"
                                    ),
                                )
                            with gr.Column(elem_classes=["settings-tile", "hints-tile"]):
                                gr.Markdown("### Hints", elem_classes=["settings-tile-heading"])
                                hint_input = gr.Textbox(
                                    lines=8,
                                    max_lines=6,
                                    label=None,
                                    show_label=False,
                                    placeholder=(
                                        "Optional: names, topic, jargon"
                                    ),
                                    info=f"Always includes: {DEFAULT_INITIAL_PROMPT_FULL}",
                                )
                        with gr.Column(elem_classes=["upload-settings-col"]):
                            with gr.Column(elem_classes=["settings-tile"]):
                                gr.Markdown("### Diarization", elem_classes=["settings-tile-heading"])
                                with gr.Column(elem_id="diarization-group"):
                                    diarize_check = gr.Checkbox(
                                        label="Enable diarization",
                                        value=False,
                                        elem_id="diarize-check",
                                        container=False,
                                    )
                                    with gr.Row(elem_id="speaker-range-inputs", visible=False) as speaker_range_row:
                                        _spk_choices = ["auto", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
                                        min_speakers = gr.Dropdown(
                                            label="Min Speakers",
                                            choices=_spk_choices,
                                            value="2",
                                            allow_custom_value=False,
                                        )
                                        max_speakers = gr.Dropdown(
                                            label="Max Speakers",
                                            choices=_spk_choices,
                                            value="5",
                                            allow_custom_value=False,
                                        )
                            with gr.Column(elem_classes=["settings-tile"]):
                                gr.Markdown("### Crop", elem_classes=["settings-tile-heading"])
                                with gr.Column(elem_id="crop-group"):
                                    crop_enable = gr.Checkbox(
                                        label="Choose Segment",
                                        value=False,
                                        container=False,
                                    )
                                with gr.Column(elem_id="start-end-crop-inputs", visible=False) as crop_inputs_col:
                                    gr.HTML('<p class="crop-hms-label">From</p>')
                                    with gr.Row(elem_classes=["crop-hms-row"]):
                                        start_h = gr.Number(
                                            label="Hour",
                                            value=0,
                                            minimum=0,
                                            precision=0,
                                            scale=1,
                                        )
                                        start_m = gr.Number(
                                            label="Minute",
                                            value=0,
                                            minimum=0,
                                            maximum=59,
                                            precision=0,
                                            scale=1,
                                        )
                                        start_s = gr.Number(
                                            label="Second",
                                            value=0,
                                            minimum=0,
                                            maximum=59,
                                            precision=0,
                                            scale=1,
                                        )
                                    gr.HTML('<p class="crop-hms-label">To</p>')
                                    with gr.Row(elem_classes=["crop-hms-row"]):
                                        end_h = gr.Number(
                                            label="Hour",
                                            value=0,
                                            minimum=0,
                                            precision=0,
                                            scale=1,
                                        )
                                        end_m = gr.Number(
                                            label="Minute",
                                            value=0,
                                            minimum=0,
                                            maximum=59,
                                            precision=0,
                                            scale=1,
                                        )
                                        end_s = gr.Number(
                                            label="Second",
                                            value=0,
                                            minimum=0,
                                            maximum=59,
                                            precision=0,
                                            scale=1,
                                        )
                                    gr.Markdown(
                                        "0 hour, 0 minute, 0 second = through end of file",
                                        elem_classes=["crop-hms-hint"],
                                    )
                with gr.Row(elem_classes=["upload-transcribe-row"]):
                    run_btn = gr.Button(
                        "Transcribe",
                        variant="primary",
                        elem_id="transcribe-btn",
                    )

                progress_steps = gr.HTML(
                    value=_EMPTY_PROGRESS_HTML,
                    elem_id="progress-steps",
                    show_label=False,
                    container=False,
                    padding=False,
                )

                with gr.Column(elem_classes=["transcript-card"]):
                    gr.Markdown("### Transcript", elem_classes=["settings-tile-heading"])
                    with gr.Column(elem_classes=["transcript-panel"]):
                        transcript_box = gr.HTML(
                            value=_EMPTY_TRANSCRIPT_HTML,
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
                            choices=[
                                ("Turbo Fast [turbo-4bit]",         "turbo-4bit"),
                                ("Fast + Accurate [large-v3-4bit]", "large-v3-4bit"),
                                ("Max Accuracy [large-v3]",         "large-v3"),
                                ("Turbo [turbo]",                   "turbo"),
                                ("Medium [medium]",                 "medium"),
                                ("Small [small]",                   "small"),
                            ],
                            value="turbo-4bit",
                            label=None,
                            show_label=False,
                            filterable=False,
                            allow_custom_value=False,
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
        fn=handle_audio_upload,
        inputs=[file_input],
        outputs=[playback_html],
    )

    crop_enable.change(
        fn=lambda checked: gr.update(visible=checked),
        inputs=[crop_enable],
        outputs=[crop_inputs_col],
    )

    diarize_check.change(
        fn=lambda checked: gr.update(visible=checked),
        inputs=[diarize_check],
        outputs=[speaker_range_row],
    )

    run_btn.click(
        fn=reset_transcript_ui,
        outputs=[transcript_box, download_btn, stats_md, progress_steps],
    ).then(
        fn=run_transcription,
        inputs=[
            file_input,
            model_picker,
            diarize_check,
            min_speakers,
            max_speakers,
            start_h,
            start_m,
            start_s,
            end_h,
            end_m,
            end_s,
            hint_input,
            text_fix_check,
        ],
        outputs=[transcript_box, download_btn, stats_md, progress_steps],
        show_progress="hidden",
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
