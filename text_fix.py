#!/usr/bin/env python3
"""Arabic ASR typo correction via a small local instruct model (runs on CPU/GPU)."""

from __future__ import annotations

import os
import re
import threading
from typing import Callable

_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_BATCH_SIZE = 6
_MAX_LINE_CHARS = 400

_model = None
_tokenizer = None
_model_id: str | None = None
_load_lock = threading.Lock()


def default_model() -> str:
    return os.environ.get("TEXT_FIX_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL


def is_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _hf_hub_kwargs() -> dict:
    kw: dict = {}
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        kw["token"] = token
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
        kw["local_files_only"] = True
    return kw


def _device() -> str:
    forced = os.environ.get("TEXT_FIX_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_model():
    global _model, _tokenizer, _model_id
    model_id = default_model()
    if _model is not None and _tokenizer is not None and _model_id == model_id:
        return _model, _tokenizer

    with _load_lock:
        if _model is not None and _tokenizer is not None and _model_id == model_id:
            return _model, _tokenizer

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hub_kw = _hf_hub_kwargs()
        device = _device()
        dtype = torch.float32 if device == "cpu" else torch.float16

        tok = AutoTokenizer.from_pretrained(model_id, **hub_kw)
        load_kw = dict(hub_kw)
        try:
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                **load_kw,
            )
        except TypeError:
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                **load_kw,
            )
        mdl.to(device)
        mdl.eval()

        _tokenizer = tok
        _model = mdl
        _model_id = model_id
        return _model, _tokenizer


def prefetch() -> None:
    """Download weights once while online (no-op if already cached)."""
    _load_model()


def _parse_numbered_lines(response: str, expected_n: int) -> list[str] | None:
    numbered: dict[int, str] = {}
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\.\s*(.+)$", line)
        if m:
            numbered[int(m.group(1))] = m.group(2).strip()
    if len(numbered) == expected_n:
        return [numbered[i] for i in range(1, expected_n + 1)]

    plain = [re.sub(r"^\d+\.\s*", "", ln.strip()) for ln in response.strip().splitlines() if ln.strip()]
    if len(plain) == expected_n:
        return plain
    return None


def _build_prompt(numbered_block: str, user_hint: str) -> str:
    hint_line = ""
    if user_hint and user_hint.strip():
        hint_line = f"\nContext (names/terms to preserve): {user_hint.strip()[:300]}\n"
    return (
        "You correct Arabic speech-to-text output. Fix typos, wrong letters, and common ASR "
        "homophone mistakes only. Do not rephrase, summarize, translate, or add content."
        f"{hint_line}\n"
        "Return exactly the same number of numbered lines, Arabic only:\n\n"
        f"{numbered_block}"
    )


def _generate(prompt: str, max_new_tokens: int) -> str:
    import torch

    model, tokenizer = _load_model()
    device = _device()
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _fix_batch(texts: list[str], user_hint: str = "") -> list[str]:
    if not texts:
        return []
    if len(texts) == 1 and len(texts[0]) <= 12:
        return texts

    numbered = "\n".join(f"{i + 1}. {t[:_MAX_LINE_CHARS]}" for i, t in enumerate(texts))
    prompt = _build_prompt(numbered, user_hint)
    max_new = min(512, 72 * len(texts) + 48)
    raw = _generate(prompt, max_new_tokens=max_new)
    parsed = _parse_numbered_lines(raw, len(texts))
    if parsed:
        return parsed
    return texts


def _apply_text_to_segment(seg: dict, new_text: str) -> None:
    seg["text"] = new_text
    words = seg.get("words")
    if not words:
        return
    old_tokens = [w.get("word", "").strip() for w in words if w.get("word", "").strip()]
    new_tokens = new_text.split()
    if len(new_tokens) == len(old_tokens):
        for w, tok in zip(words, new_tokens):
            w["word"] = tok
    else:
        seg["words"] = [
            {"word": new_text, "start": seg["start"], "end": seg["end"]},
        ]


def fix_segments(
    segments: list[dict],
    user_hint: str = "",
    on_progress: Callable[[float, str], None] | None = None,
) -> list[dict]:
    """Return a copy of segments with corrected Arabic text (segment + word level when possible)."""
    if not segments or not is_available():
        return segments

    indices: list[int] = []
    texts: list[str] = []
    for i, seg in enumerate(segments):
        t = (seg.get("text") or "").strip()
        if t:
            indices.append(i)
            texts.append(t)

    if not texts:
        return segments

    fixed_texts: list[str] = []
    total_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE
    for b, start in enumerate(range(0, len(texts), _BATCH_SIZE)):
        batch = texts[start : start + _BATCH_SIZE]
        try:
            fixed_texts.extend(_fix_batch(batch, user_hint=user_hint))
        except Exception:
            fixed_texts.extend(batch)
        if on_progress:
            on_progress((b + 1) / total_batches, "Fixing typos")

    out = [dict(seg) for seg in segments]
    for idx, new_t in zip(indices, fixed_texts):
        old_t = (out[idx].get("text") or "").strip()
        if new_t and new_t != old_t:
            _apply_text_to_segment(out[idx], new_t)
    return out
