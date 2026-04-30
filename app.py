#!/usr/bin/env python3
"""
Arabic STT — Gradio UI
Run: python app.py
"""

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import gradio as gr

from transcribe import MODEL_MAP

# how many seconds to buffer before each live transcription pass
LIVE_CHUNK_SECONDS = 6


# ── Theme & CSS ───────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Cairo"),
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700&display=swap');

* { font-family: 'Cairo', sans-serif !important; }

html, body, .gradio-container, .main { direction: rtl !important; }
.gradio-row { flex-direction: row-reverse !important; }

#header { text-align: center; padding: 28px 0 16px; border-bottom: 1px solid #e2e8f0; margin-bottom: 24px; }
#header h1 { font-size: 1.65rem; font-weight: 700; color: #0f172a; margin: 0 0 4px; }
#header p  { color: #94a3b8; font-size: 0.88rem; margin: 0; }

label, .label-wrap, fieldset legend { text-align: right !important; direction: rtl !important; }

#transcript-box textarea, #live-transcript textarea {
    direction: rtl !important;
    text-align: right !important;
    font-size: 1.05rem !important;
    line-height: 2 !important;
    color: #1e293b;
    background: #f8fafc;
    min-height: 380px;
    padding: 20px !important;
}

#transcribe-btn, #clear-btn { margin-top: 10px; }
#transcribe-btn button, #clear-btn button {
    width: 100%;
    height: 50px;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border-radius: 12px;
}

#download-btn button { width: 100%; border-radius: 10px; margin-top: 8px; font-weight: 600 !important; }

#stats, #live-stats { text-align: right; font-size: 0.82rem; color: #94a3b8; margin-top: 6px; padding-right: 4px; }

.info { text-align: right !important; }

/* live indicator dot */
#live-dot { color: #ef4444; font-size: 0.85rem; text-align: right; margin-top: 6px; }
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def segments_to_display(segments) -> str:
    lines = []
    for seg in segments:
        text = seg["text"].strip()
        if text:
            lines.append(f"[{fmt_ts(seg['start'])} ← {fmt_ts(seg['end'])}]  {text}")
    return "\n\n\n".join(lines)


def segments_to_file(segments) -> str:
    lines = []
    for seg in segments:
        text = seg["text"].strip()
        if text:
            lines.append(f"[{fmt_ts(seg['start'])} → {fmt_ts(seg['end'])}] {text}")
    return "\n".join(lines)


# ── Audio preprocessing ───────────────────────────────────────────────────────

def preprocess_audio(input_path: str) -> str:
    """
    Works on any audio OR video file.
    Extracts audio, normalizes volume, reduces noise, outputs 16kHz mono WAV.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",                          # drop video stream if present
        "-af", (
            "highpass=f=100,"
            "lowpass=f=8000,"
            "afftdn=nf=-25,"
            "dynaudnorm=f=150:g=15"     # single-pass loudness norm, much faster than loudnorm
        ),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        tmp.name,
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode()}")

    return tmp.name


# ── Upload tab: transcription ─────────────────────────────────────────────────

def run_transcription(file, model_key, progress=gr.Progress(track_tqdm=True)):
    if file is None:
        raise gr.Error("من فضلك ارفع ملفاً صوتياً أو مرئياً أولاً")

    import mlx_whisper

    progress(0.05, desc="معالجة الملف…")
    repo = MODEL_MAP[model_key]
    t0   = time.time()

    cleaned = preprocess_audio(file)

    progress(0.2, desc="جارٍ تحويل الكلام إلى نص…")
    result = mlx_whisper.transcribe(
        cleaned,
        path_or_hf_repo=repo,
        language="ar",
        word_timestamps=False,
        verbose=False,
        no_speech_threshold=0.95,
        logprob_threshold=-3.0,
        compression_ratio_threshold=3.0,
        condition_on_previous_text=True,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        initial_prompt="هذا تسجيل لاجتماع باللغة العربية.",
    )

    elapsed  = time.time() - t0
    Path(cleaned).unlink(missing_ok=True)
    segments = result.get("segments", [])

    if not segments:
        raise gr.Error("لم يُكتشف أي كلام في هذا الملف")

    duration = segments[-1]["end"]
    speed    = duration / elapsed if elapsed > 0 else 0

    out_path = Path(file).with_suffix(".txt")
    out_path.write_text(segments_to_file(segments), encoding="utf-8")

    stats = (
        f"المدة: {int(duration//60)}:{int(duration%60):02d} دقيقة  ·  "
        f"وقت المعالجة: {elapsed:.0f} ثانية  ·  "
        f"السرعة: {speed:.1f}× الوقت الحقيقي"
    )

    progress(1.0, desc="اكتمل ✓")
    return (
        segments_to_display(segments),
        gr.DownloadButton(value=str(out_path), visible=True, label="⬇  تحميل النص"),
        stats,
    )


# ── Live tab: chunk processing ────────────────────────────────────────────────

def process_live_chunk(chunk, buffer, transcript, secs_done, model_key):
    """
    Called on every mic audio chunk from Gradio streaming.
    Accumulates audio until LIVE_CHUNK_SECONDS, then transcribes and appends.
    """
    if chunk is None:
        return transcript, buffer, secs_done

    import mlx_whisper

    sr, audio = chunk

    # normalize to float32 mono
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio /= 32768.0 if audio.max() > 1.0 else 1.0

    # accumulate
    buffer = audio if buffer is None else np.concatenate([buffer, audio])

    # wait until we have enough audio
    if len(buffer) / sr < LIVE_CHUNK_SECONDS:
        return transcript, buffer, secs_done

    # write buffer to temp WAV
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, buffer, sr, subtype="PCM_16")
    tmp.close()

    result = mlx_whisper.transcribe(
        tmp.name,
        path_or_hf_repo=MODEL_MAP[model_key],
        language="ar",
        word_timestamps=False,
        verbose=False,
        no_speech_threshold=0.95,
        logprob_threshold=-3.0,
        compression_ratio_threshold=3.0,
        condition_on_previous_text=True,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        initial_prompt="هذا تسجيل لاجتماع باللغة العربية.",
    )

    Path(tmp.name).unlink(missing_ok=True)

    segments = result.get("segments", [])
    new_text = " ".join(seg["text"].strip() for seg in segments if seg["text"].strip())

    chunk_duration = len(buffer) / sr
    if new_text:
        start = fmt_ts(secs_done)
        end   = fmt_ts(secs_done + chunk_duration)
        line  = f"[{start} ← {end}]  {new_text}"
        transcript = (transcript + "\n\n\n" + line).lstrip()

    secs_done += chunk_duration
    buffer = None  # reset for next chunk

    return transcript, buffer, secs_done


def clear_live(transcript, buffer, secs):
    return "", None, 0.0


# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="تحويل الكلام إلى نص") as demo:

    gr.HTML("""
        <div id="header">
            <h1>🎙 تحويل الكلام العربي إلى نص</h1>
            <p>معالجة محلية بالكامل — لا يغادر أي ملف جهازك</p>
        </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Upload (audio or video) ────────────────────────────────
        with gr.Tab("📁 رفع ملف"):
            with gr.Row():
                with gr.Column(scale=1, min_width=280):

                    file_input = gr.Audio(
                        label="ارفع الملف الصوتي",
                        type="filepath",
                    )

                    model_picker = gr.Dropdown(
                        choices=list(MODEL_MAP.keys()),
                        value="large-v3",
                        label="النموذج",
                        info="large-v3 = أعلى دقة  |  turbo = أسرع بمرتين",
                    )

                    run_btn = gr.Button("تحويل إلى نص", variant="primary", elem_id="transcribe-btn")

                with gr.Column(scale=2):
                    transcript_box = gr.Textbox(
                        label="النص المستخرج",
                        lines=20,
                        elem_id="transcript-box",
                        placeholder="سيظهر النص هنا بعد اكتمال التحويل…",
                        interactive=False,
                    )
                    download_btn = gr.DownloadButton(
                        label="⬇  تحميل النص",
                        visible=False,
                        elem_id="download-btn",
                    )
                    stats_md = gr.Markdown(elem_id="stats")

        # ── Tab 2: Live mic ───────────────────────────────────────────────
        with gr.Tab("🔴 مباشر"):
            with gr.Row():
                with gr.Column(scale=1, min_width=280):

                    model_picker_live = gr.Dropdown(
                        choices=list(MODEL_MAP.keys()),
                        value="turbo",
                        label="النموذج",
                        info="turbo مُوصى به للوضع المباشر — أسرع استجابةً",
                    )

                    mic_input = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        label="المايكروفون",
                        type="numpy",
                    )

                    clear_btn = gr.Button("🗑  مسح النص", variant="secondary", elem_id="clear-btn")

                    gr.HTML('<div id="live-dot">● يبدأ التحويل كل 6 ثوانٍ من الكلام</div>')

                with gr.Column(scale=2):
                    live_transcript = gr.Textbox(
                        label="النص المباشر",
                        lines=20,
                        elem_id="live-transcript",
                        placeholder="ابدأ الكلام وسيظهر النص هنا تلقائياً…",
                        interactive=False,
                    )
                    live_stats = gr.Markdown(elem_id="live-stats")

    # ── State ─────────────────────────────────────────────────────────────

    audio_buffer  = gr.State(None)
    live_text     = gr.State("")
    secs_done     = gr.State(0.0)

    # ── Events ────────────────────────────────────────────────────────────

    run_btn.click(
        fn=run_transcription,
        inputs=[file_input, model_picker],
        outputs=[transcript_box, download_btn, stats_md],
    )

    mic_input.stream(
        fn=process_live_chunk,
        inputs=[mic_input, audio_buffer, live_text, secs_done, model_picker_live],
        outputs=[live_transcript, audio_buffer, secs_done],
    )

    clear_btn.click(
        fn=clear_live,
        inputs=[live_text, audio_buffer, secs_done],
        outputs=[live_transcript, audio_buffer, secs_done],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=THEME,
        css=CSS,
    )
