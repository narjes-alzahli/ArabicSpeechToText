#!/usr/bin/env python3
"""
Arabic STT — Gradio UI
Run: python app.py
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

# Load .env if present
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import mlx_whisper
import numpy as np
import soundfile as sf
import gradio as gr

from transcribe import MODEL_MAP, fmt_ts, fmt_ts_ms, WHISPER_FULL_KWARGS, WHISPER_LIVE_KWARGS
from diarize import run_diarization, assign_speakers


# ── Theme & CSS ───────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue="stone",
    neutral_hue="stone",
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
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

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── base ── */
html, body, .gradio-container { background: #faf8f4 !important; }
.gradio-container { max-width: 1600px !important; margin: 0 auto !important; padding: 0 !important; }
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
    font-size: 0.9rem;
    font-weight: 500;
    color: #9e9188;
    letter-spacing: 0.01em;
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

/* ── controls panel ── */
.controls-col {
    background: #f4f0e8 !important;
    border: 1px solid #e8e2d6 !important;
    border-radius: 8px !important;
    padding: 24px !important;
}

/* ── live transcript ── */
#live-transcript { height: 100% !important; }
#live-transcript textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.8 !important;
    color: #2c2825 !important;
    background: #ffffff !important;
    border: 1px solid #e8e2d6 !important;
    border-radius: 8px !important;
    min-height: 520px !important;
    padding: 24px !important;
    resize: none !important;
}
#live-transcript textarea::placeholder { color: #d4cec8 !important; }

/* ── clickable transcript (upload tab) ── */
#transcript-box { height: 100% !important; }
#transcript-box > div {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
    line-height: 1.8;
    background: #ffffff;
    border: 1px solid #e8e2d6;
    border-radius: 8px;
    min-height: 520px;
    padding: 24px;
    overflow-y: auto;
    color: #2c2825;
}
#transcript-box .placeholder { color: #d4cec8; }
#transcript-box .seg { margin-bottom: 1.4em; }
#transcript-box .ts {
    font-size: 0.72em;
    color: #c4bab0;
    cursor: pointer;
    margin-right: 8px;
    user-select: none;
}
#transcript-box .ts:hover { color: #9e9188; }
#transcript-box .spk {
    font-size: 0.78em;
    font-weight: 600;
    color: #a08060;
    margin-right: 6px;
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
#transcribe-btn button { width: 100% !important; height: 40px !important; }
#clear-btn button, #save-live-btn button { width: 100% !important; height: 36px !important; }
#download-btn button, #live-download-btn button {
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

/* ── prevent column stacking before upload ── */
.gradio-row { flex-wrap: nowrap !important; }
"""


_loaded_model_repo: str | None = None


def evict_model_if_changed(new_model_key: str):
    global _loaded_model_repo
    import mlx.core as mx
    new_repo = MODEL_MAP[new_model_key]
    if _loaded_model_repo is not None and _loaded_model_repo != new_repo:
        mx.clear_cache()
    _loaded_model_repo = new_repo


# ── Helpers ───────────────────────────────────────────────────────────────────

def segments_to_html(segments) -> str:
    """Renders transcript as clickable word spans. Clicking a word seeks the audio player."""
    if not segments:
        return '<div class="placeholder">$ _</div>'

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
            f'[{fmt_ts(seg_start)} → {fmt_ts(seg_end)}]</span>'
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

        parts.append(f'<div class="seg">{spk_html}{ts_html} {words_html}</div>')

    return "\n".join(parts)


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


# ── Audio preprocessing ───────────────────────────────────────────────────────

def extract_audio_for_diarization(input_path: str, start: float = 0.0, end: float = 0.0) -> str:
    """Minimal conversion for pyannote: audio-only WAV, no filtering that strips speaker characteristics."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    cmd = ["ffmpeg", "-y"]
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
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode()}")

    return tmp.name


def preprocess_audio(input_path: str, start: float = 0.0, end: float = 0.0) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    cmd = ["ffmpeg", "-y"]
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
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode()}")

    return tmp.name


# ── Upload tab ────────────────────────────────────────────────────────────────

def run_transcription(file, model_key, do_diarize, num_speakers, start_time, end_time, progress=gr.Progress()):
    if file is None:
        raise gr.Error("Please upload an audio file first.")

    progress(0.05, desc="Preprocessing audio")
    evict_model_if_changed(model_key)
    repo = MODEL_MAP[model_key]
    t0   = time.time()

    cleaned = preprocess_audio(str(file), start=float(start_time or 0), end=float(end_time or 0))
    try:
        progress(0.2, desc="Transcribing")
        result = mlx_whisper.transcribe(
            cleaned,
            path_or_hf_repo=repo,
            **WHISPER_FULL_KWARGS,
        )

        segments = result.get("segments", [])
        if not segments:
            raise gr.Error("No speech detected in the file.")

        if do_diarize:
            progress(0.75, desc="Identifying speakers")
            diarize_wav = None
            try:
                import mlx.core as mx
                mx.clear_cache()
                diarize_wav = extract_audio_for_diarization(
                    file,
                    start=float(start_time or 0),
                    end=float(end_time or 0),
                )
                turns = run_diarization(diarize_wav, num_speakers=int(num_speakers or 0))
                segments = assign_speakers(segments, turns)
                _save_diarization(turns, file)
            except RuntimeError as e:
                raise gr.Error(str(e))
            finally:
                if diarize_wav:
                    Path(diarize_wav).unlink(missing_ok=True)
    finally:
        Path(cleaned).unlink(missing_ok=True)

    elapsed  = time.time() - t0
    duration = segments[-1]["end"]
    speed    = duration / elapsed if elapsed > 0 else 0

    out_fd, out_path_str = tempfile.mkstemp(suffix=".txt")
    os.close(out_fd)
    out_path = Path(out_path_str)
    out_path.write_text(segments_to_file(segments), encoding="utf-8")

    _save_transcript(segments, file)

    stats = (
        f"duration: {int(duration//60)}:{int(duration%60):02d}  ·  "
        f"processing: {elapsed:.0f}s  ·  "
        f"speed: {speed:.1f}x real-time"
    )

    progress(1.0, desc="Done")
    return (
        segments_to_html(segments),
        gr.DownloadButton(value=str(out_path), visible=True, label="Download transcript ⬇"),
        stats,
    )


# ── Live tab ──────────────────────────────────────────────────────────────────

def _transcribe_buffer(buffer, sr, secs_done, chunk_count, transcript, model_key):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, buffer, sr, subtype="PCM_16")
    tmp.close()
    try:
        result = mlx_whisper.transcribe(
            tmp.name,
            path_or_hf_repo=MODEL_MAP[model_key],
            **WHISPER_LIVE_KWARGS,
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
    return gr.DownloadButton(value=out_path_str, visible=True, label="Download transcript ⬇")


def clear_live():
    return "", None, 0.0, 0, "", gr.DownloadButton(visible=False)


# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Arabic Speech to Text", css=CSS) as demo:

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

        // ── start/end region highlight ───────────────────────────────────
        function highlightRegion(start, end) {
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

        // watch the start/end number inputs and update the highlight
        function watchRegionInputs() {
            const inputs = document.querySelectorAll(
                '#start-time input[type=number], #end-time input[type=number]'
            );
            if (inputs.length < 2) { setTimeout(watchRegionInputs, 500); return; }
            const update = () => {
                const s = parseFloat(inputs[0].value) || 0;
                const e = parseFloat(inputs[1].value) || 0;
                highlightRegion(s, e);
            };
            inputs.forEach(inp => inp.addEventListener('input', update));
        }
        document.addEventListener('DOMContentLoaded', watchRegionInputs);
        setTimeout(watchRegionInputs, 1000);
        </script>
    """)

    with gr.Tabs():

        # ── Tab 1: Upload ─────────────────────────────────────────────────
        with gr.Tab("Upload"):
            with gr.Row():
                with gr.Column(scale=1, min_width=300, elem_classes="controls-col"):
                    file_input = gr.Audio(
                        label="Audio / video file",
                        type="filepath",
                        elem_id="upload-audio",
                    )
                    model_picker = gr.Dropdown(
                        choices=list(MODEL_MAP.keys()),
                        value="large-v3-4bit",
                        label="Model",
                        info="4bit = fast  ·  large-v3 = max accuracy",
                    )
                    with gr.Row():
                        start_time = gr.Number(label="Start (s)", value=0, minimum=0, scale=1, elem_id="start-time")
                        end_time   = gr.Number(label="End (s)",   value=0, minimum=0, scale=1, elem_id="end-time",
                                               info="0 = until end")
                    diarize_check = gr.Checkbox(
                        label="Speaker diarization",
                        value=False,
                    )
                    num_speakers = gr.Number(
                        label="Number of speakers",
                        value=0,
                        minimum=0,
                        precision=0,
                        info="0 = auto-detect",
                        visible=False,
                    )
                    run_btn = gr.Button(
                        "Transcribe",
                        variant="primary",
                        elem_id="transcribe-btn",
                    )

                with gr.Column(scale=2):
                    transcript_box = gr.HTML(
                        value='<div class="placeholder">$ _</div>',
                        elem_id="transcript-box",
                        label="Transcript",
                    )
                    download_btn = gr.DownloadButton(
                        label="Download transcript ⬇",
                        visible=False,
                        elem_id="download-btn",
                    )
                    stats_md = gr.Markdown(elem_id="stats")

        # ── Tab 2: Live ───────────────────────────────────────────────────
        with gr.Tab("Live"):
            with gr.Row():
                with gr.Column(scale=1, min_width=300, elem_classes="controls-col"):
                    model_picker_live = gr.Dropdown(
                        choices=list(MODEL_MAP.keys()),
                        value="turbo-4bit",
                        label="Model",
                        info="turbo-4bit recommended for live mode",
                    )
                    chunk_slider = gr.Slider(
                        minimum=3,
                        maximum=15,
                        value=6,
                        step=1,
                        label="Chunk size (sec)",
                        info="Lower = faster response · Higher = better accuracy",
                    )
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        label="Microphone",
                        type="numpy",
                    )
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

                with gr.Column(scale=2):
                    live_transcript = gr.Textbox(
                        label="Live transcript",
                        lines=22,
                        elem_id="live-transcript",
                        placeholder="$ _",
                        interactive=False,
                    )
                    live_download_btn = gr.DownloadButton(
                        label="Download transcript ⬇",
                        visible=False,
                        elem_id="live-download-btn",
                    )
                    live_stats = gr.Markdown(elem_id="live-stats")

    # ── State ─────────────────────────────────────────────────────────────

    audio_buffer = gr.State(None)
    secs_done    = gr.State(0.0)
    chunk_count  = gr.State(0)

    # ── Events ────────────────────────────────────────────────────────────

    diarize_check.change(
        fn=lambda v: gr.update(visible=v),
        inputs=diarize_check,
        outputs=num_speakers,
    )

    run_btn.click(
        fn=run_transcription,
        inputs=[file_input, model_picker, diarize_check, num_speakers, start_time, end_time],
        outputs=[transcript_box, download_btn, stats_md],
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
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=THEME,
    )
