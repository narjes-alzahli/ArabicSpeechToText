# Arabic Speech-to-Text

Local, on-device Arabic speech transcription. No cloud, no API keys, no audio leaves your machine.

Built on [mlx-whisper](https://github.com/ml-explore/mlx-examples) — a port of OpenAI Whisper optimized for Apple Silicon. Runs `whisper-large-v3` (1.5B parameters) on the M-series GPU via Apple's MLX framework.

---

## Features

### Two transcription modes

- **File upload** — drag-and-drop any audio file, get a timestamped Arabic transcript
- **Live mode** — stream from your microphone, transcript grows line-by-line every 6 seconds as you speak

### Pipeline

- **Audio preprocessing** with ffmpeg before transcription:
  - High-pass / low-pass filters (cuts low-frequency rumble and high-frequency hiss)
  - FFT-based noise reduction
  - Dynamic loudness normalization (levels out quiet speakers)
  - Resamples to 16 kHz mono — Whisper's native format
- **Tuned Whisper settings** for higher coverage on Arabic speech:
  - Loosened `no_speech_threshold` and `logprob_threshold` so quiet/uncertain segments aren't dropped
  - Temperature fallback decoding — retries difficult segments with increasing randomness instead of skipping them
  - Initial Arabic prompt to prime the language detection
  - Context conditioning across segments for better continuity

### UI

- Fully Arabic interface, RTL throughout (Cairo font)
- Two-tab layout: file upload + live mic
- Built-in audio playback before transcribing
- Timestamped transcript with `[hh:mm:ss ← hh:mm:ss]` ranges
- Download button for `.txt` export
- Stats: audio duration, processing time, real-time speed multiplier

### CLI

A standalone `transcribe.py` script for headless / scripted use:

```bash
python transcribe.py meeting.mp3                  # outputs meeting.txt
python transcribe.py meeting.mp3 --format srt     # subtitle file
python transcribe.py meeting.mp3 --format json    # timestamped JSON
python transcribe.py meeting.mp3 --model turbo    # faster model
```

---

## Requirements

- macOS with **Apple Silicon** (M1/M2/M3/M4) — required for MLX runtime
- Python 3.11
- ~3.5 GB disk space for model weights
- ffmpeg (installed via Homebrew)

---

## Installation

```bash
git clone https://github.com/narjes-alzahli/ArabicSpeechToText.git
cd ArabicSpeechToText

# system dep
brew install ffmpeg

# Python env
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Web UI

```bash
source venv/bin/activate
python app.py
```

Opens automatically at [http://127.0.0.1:7860](http://127.0.0.1:7860).

First run downloads the `large-v3` model (~3 GB) from HuggingFace, then caches it permanently.

### CLI

```bash
source venv/bin/activate
python transcribe.py path/to/audio.mp3
```

---

## Models

| Model | Size | Best for |
|---|---|---|
| `large-v3` | ~3 GB | Highest accuracy — recommended for files |
| `turbo` | ~1.5 GB | 2× faster — recommended for live mode |
| `medium` | ~1.5 GB | Lighter alternative |
| `small` | ~500 MB | Fastest, lower accuracy |

Switch via the dropdown in the UI or `--model` flag in CLI.

---

## Architecture

```
Audio file / mic chunk
        │
        ▼
   ffmpeg preprocessing
   (denoise, normalize, 16kHz mono)
        │
        ▼
   mlx-whisper transcribe
   (Apple GPU via Metal)
        │
        ▼
   Segments with timestamps
        │
        ▼
   Format → display + download
```

**Why MLX instead of faster-whisper?**
mlx-whisper is built specifically for Apple Silicon GPUs. On an M3 Mac it runs ~15× real-time on `large-v3`. The standard `faster-whisper` runtime falls back to CPU on Mac, which is significantly slower.

---

## Customization

Common knobs live at the top of `app.py`:

- `LIVE_CHUNK_SECONDS` — how long to buffer mic audio before each transcription pass (default 6)
- `MODEL_MAP` (in `transcribe.py`) — add or swap Whisper model variants
- `THEME` and `CSS` — visual styling
- `initial_prompt` inside `run_transcription` — domain-specific priming (replace with your meeting context for better vocabulary recognition)

---

## Project structure

```
ArabicSpeechToText/
├── app.py              # Gradio UI (upload + live tabs)
├── transcribe.py       # CLI entry point + shared model map
├── requirements.txt    # pinned Python deps
├── .gitignore
└── README.md
```

---

## Known limitations

- **Apple Silicon only.** mlx-whisper does not run on Intel Macs, Linux, or Windows. For other platforms swap to [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) (different runtime, same model weights).
- **No beam search.** mlx-whisper hasn't implemented beam search yet — falls back to greedy/temperature decoding.
- **Live mode latency.** ~1–2 second processing lag per chunk on M3. Lower on M4 / M-series Pro/Max chips.
- **Dialect coverage.** Whisper is trained heavily on Modern Standard Arabic. Heavy dialect content (Maghrebi, Gulf, Levantine) may benefit from LoRA fine-tuning on dialect-labeled data.

---

## License

MIT
