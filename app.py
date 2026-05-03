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

LIVE_CHUNK_SECONDS = 6


# ── Theme & CSS ───────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    radius_size="lg",
    font=gr.themes.GoogleFont("Cairo"),
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700&display=swap');

* { font-family: 'Cairo', sans-serif !important; }

html, body, .gradio-container, .main { direction: rtl !important; }
.gradio-row { flex-direction: row-reverse !important; }

/* container: more breathing room */
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; padding: 24px !important; }

/* header — softer, no border */
#header { text-align: center; padding: 16px 0 24px; }
#header h1 {
    font-size: 1.45rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0;
    letter-spacing: 0.01em;
}
#header p {
    color: #94a3b8;
    font-size: 0.85rem;
    margin: 6px 0 0;
    font-weight: 400;
}

/* tabs — cleaner */
.tab-nav button {
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 10px 20px !important;
}

/* labels right-aligned */
label, .label-wrap, fieldset legend, .info {
    text-align: right !important;
    direction: rtl !important;
    font-weight: 500 !important;
}

/* inputs */
input, select, textarea, .input-text {
    direction: rtl !important;
    text-align: right !important;
}

/* transcript boxes */
#transcript-box textarea, #live-transcript textarea {
    direction: rtl !important;
    text-align: right !important;
    font-size: 1rem !important;
    line-height: 2.1 !important;
    color: #1e293b;
    background: #fafbfc;
    min-height: 420px;
    padding: 22px !important;
    border: 1px solid #e5e7eb !important;
}

/* primary button */
#transcribe-btn button, #clear-btn button {
    width: 100%;
    height: 46px;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: 10px;
    margin-top: 8px;
}

/* download */
#download-btn button {
    width: 100%;
    border-radius: 10px;
    margin-top: 10px;
    font-weight: 500 !important;
    background: #f1f5f9 !important;
    color: #1e293b !important;
    border: 1px solid #e2e8f0 !important;
}

/* stats — subtle */
#stats, #live-stats {
    text-align: right;
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 10px;
    padding: 6px 4px;
}

/* hint text */
.hint {
    text-align: right;
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 8px;
    padding: 4px;
}

/* dropdown info text */
.gradio-dropdown .help { text-align: right !important; font-size: 0.78rem !important; color: #94a3b8 !important; }
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
    return "\n\n".join(lines)


def segments_to_file(segments) -> str:
    lines = []
    for seg in segments:
        text = seg["text"].strip()
        if text:
            lines.append(f"[{fmt_ts(seg['start'])} → {fmt_ts(seg['end'])}] {text}")
    return "\n".join(lines)


# ── Audio preprocessing ───────────────────────────────────────────────────────

def preprocess_audio(input_path: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
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
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode()}")

    return tmp.name


# ── Upload tab ────────────────────────────────────────────────────────────────

def run_transcription(file, model_key, progress=gr.Progress()):
    if file is None:
        raise gr.Error("الرجاء رفع ملف صوتي أولاً")

    import mlx_whisper

    progress(0.05, desc="معالجة الصوت")
    repo = MODEL_MAP[model_key]
    t0   = time.time()

    cleaned = preprocess_audio(file)

    progress(0.2, desc="جارٍ التحويل")
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

    elapsed = time.time() - t0
    Path(cleaned).unlink(missing_ok=True)
    segments = result.get("segments", [])

    if not segments:
        raise gr.Error("لم يُكتشف أي كلام في الملف")

    duration = segments[-1]["end"]
    speed    = duration / elapsed if elapsed > 0 else 0

    out_path = Path(file).with_suffix(".txt")
    out_path.write_text(segments_to_file(segments), encoding="utf-8")

    stats = (
        f"المدة: {int(duration//60)}:{int(duration%60):02d}  ·  "
        f"وقت المعالجة: {elapsed:.0f} ثانية  ·  "
        f"السرعة: {speed:.1f}× الوقت الحقيقي"
    )

    progress(1.0, desc="اكتمل")
    return (
        segments_to_display(segments),
        gr.DownloadButton(value=str(out_path), visible=True, label="تحميل النص ⬇"),
        stats,
    )


# ── Live tab ──────────────────────────────────────────────────────────────────

def process_live_chunk(chunk, buffer, transcript, secs_done, model_key):
    if chunk is None:
        return transcript, buffer, secs_done

    import mlx_whisper

    sr, audio = chunk

    # to float32 mono in [-1, 1]
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = float(np.abs(audio).max() or 0.0)
    if peak > 1.5:
        audio = audio / 32768.0

    # accumulate
    buffer = audio if buffer is None else np.concatenate([buffer, audio])

    if len(buffer) / sr < LIVE_CHUNK_SECONDS:
        return transcript, buffer, secs_done

    # write to temp WAV
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, buffer, sr, subtype="PCM_16")
    tmp.close()

    result = mlx_whisper.transcribe(
        tmp.name,
        path_or_hf_repo=MODEL_MAP[model_key],
        language="ar",
        word_timestamps=False,
        verbose=False,
        no_speech_threshold=0.9,
        logprob_threshold=-2.0,
        condition_on_previous_text=True,
        temperature=(0.0, 0.4, 0.8),
        initial_prompt="هذا كلام باللغة العربية.",
    )

    Path(tmp.name).unlink(missing_ok=True)

    segments = result.get("segments", [])
    new_text = " ".join(s["text"].strip() for s in segments if s["text"].strip())

    chunk_dur = len(buffer) / sr
    if new_text:
        start = fmt_ts(secs_done)
        end   = fmt_ts(secs_done + chunk_dur)
        line  = f"[{start} ← {end}]  {new_text}"
        transcript = (transcript + "\n\n" + line) if transcript else line

    secs_done += chunk_dur
    buffer = None
    return transcript, buffer, secs_done


def clear_live(transcript, buffer, secs):
    return "", None, 0.0


# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="تحويل الكلام إلى نص") as demo:

    gr.HTML("""
        <div id="header">
            <h1>تحويل الكلام العربي إلى نص</h1>
            <p>معالجة محلية بالكامل — لا يغادر أي ملف جهازك</p>
        </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Upload ─────────────────────────────────────────────────
        with gr.Tab("رفع ملف"):
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    file_input = gr.Audio(
                        label="الملف الصوتي",
                        type="filepath",
                    )
                    model_picker = gr.Dropdown(
                        choices=list(MODEL_MAP.keys()),
                        value="large-v3",
                        label="النموذج",
                        info="large-v3 للدقة العالية · turbo للسرعة",
                    )
                    run_btn = gr.Button(
                        "تحويل إلى نص",
                        variant="primary",
                        elem_id="transcribe-btn",
                    )

                with gr.Column(scale=2):
                    transcript_box = gr.Textbox(
                        label="النص المستخرج",
                        lines=18,
                        elem_id="transcript-box",
                        placeholder="سيظهر النص هنا بعد التحويل",
                        interactive=False,
                    )
                    download_btn = gr.DownloadButton(
                        label="تحميل النص ⬇",
                        visible=False,
                        elem_id="download-btn",
                    )
                    stats_md = gr.Markdown(elem_id="stats")

        # ── Tab 2: Live ───────────────────────────────────────────────────
        with gr.Tab("مباشر"):
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    model_picker_live = gr.Dropdown(
                        choices=list(MODEL_MAP.keys()),
                        value="turbo",
                        label="النموذج",
                        info="turbo مُوصى به للوضع المباشر",
                    )
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        label="المايكروفون",
                        type="numpy",
                    )
                    clear_btn = gr.Button(
                        "مسح النص",
                        variant="secondary",
                        elem_id="clear-btn",
                    )
                    gr.HTML(
                        '<div class="hint">يتم تحديث النص كل 6 ثوانٍ من الكلام</div>'
                    )

                with gr.Column(scale=2):
                    live_transcript = gr.Textbox(
                        label="النص المباشر",
                        lines=18,
                        elem_id="live-transcript",
                        placeholder="ابدأ التسجيل وسيظهر النص هنا تلقائياً",
                        interactive=False,
                    )
                    live_stats = gr.Markdown(elem_id="live-stats")

    # ── State ─────────────────────────────────────────────────────────────

    audio_buffer = gr.State(None)
    secs_done    = gr.State(0.0)

    # ── Events ────────────────────────────────────────────────────────────

    run_btn.click(
        fn=run_transcription,
        inputs=[file_input, model_picker],
        outputs=[transcript_box, download_btn, stats_md],
    )

    mic_input.stream(
        fn=process_live_chunk,
        inputs=[mic_input, audio_buffer, live_transcript, secs_done, model_picker_live],
        outputs=[live_transcript, audio_buffer, secs_done],
    )

    clear_btn.click(
        fn=clear_live,
        inputs=[live_transcript, audio_buffer, secs_done],
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
