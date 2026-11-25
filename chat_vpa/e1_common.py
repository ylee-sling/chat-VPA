# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_common.py -- what E1's Step 0 measurements share (docs/e1.md): loading the released Gemma 4 E2B-it for text,
the chat prompt, the prompt file, timing, and the record of the environment a number was measured in.

Nothing here is specific to a drafter. The target is always the released model, frozen, as loaded from its
directory; the prompt of a response is the model's own chat template around one user turn, with thinking off
(Gemma 4 switches thinking on by a ``<|think|>`` token at the start of the system prompt; no system prompt is
written here).

torch is imported inside the functions that need it: ``e1_litert.py`` runs in LiteRT-LM's environment, which has
no torch, and takes the prompt file, the median and the JSON writer from here.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

__all__ = ["DTYPES", "chat_ids", "environment", "file_sha256", "load_target", "median", "read_prompts",
           "synchronizer", "torch_dtype", "write_json"]

DTYPES: dict[str, str] = {"bf16": "bfloat16", "fp32": "float32"}          # a precision's name -> torch's


def torch_dtype(name: str) -> torch.dtype:
    import torch

    if name not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    return getattr(torch, DTYPES[name])


def load_target(model_dir: str | Path, dtype: str, device: torch.device, threads: int | None = None
                ) -> tuple[Any, Any]:
    """The released checkpoint (``Gemma4ForConditionalGeneration``, text only, eval mode) on ``device`` in
    ``dtype``, and its Hugging Face tokenizer. ``threads`` fixes the CPU threads of this process."""
    import torch

    from chat_vpa.backbone import load_gemma4

    precision = torch_dtype(dtype)
    if threads:
        torch.set_num_threads(int(threads))
    model, tokenizer = load_gemma4(str(model_dir), dtype=precision, token=os.environ.get("HF_TOKEN") or None)
    for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(model.model, tower):
            setattr(model.model, tower, None)                       # text only
    return model.to(device).eval(), tokenizer


def chat_ids(tokenizer: Any, prompt: str) -> list[int]:
    """One user turn in the model's own chat template, with the generation prompt, as token ids."""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("this tokenizer has no chat template: E1 decodes the instruction-tuned checkpoint")
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                         tokenize=False)
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def read_prompts(path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    """The prompt file: one JSON object per line with ``id`` and ``prompt`` (``e1_prompts.py`` writes it)."""
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({"id": str(row["id"]), "prompt": str(row["prompt"])})
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"{path} holds no prompt")
    return rows


def synchronizer(device: torch.device) -> Callable[[], None]:
    """A function that waits for the device's queued work, so that a wall-clock reading covers it."""
    if device.type == "cuda":
        import torch

        return lambda: torch.cuda.synchronize(device)
    return lambda: None


def median(values: Sequence[float]) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("median of nothing")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment(device: torch.device, dtype: str, threads: int | None = None) -> dict[str, Any]:
    """Where a measurement ran: every timing in a document names these (agreement 6)."""
    import torch
    import transformers

    info: dict[str, Any] = {"device": str(device), "dtype": dtype, "torch": torch.__version__,
                            "transformers": transformers.__version__, "runtime": "PyTorch, eager",
                            "cpu_threads": torch.get_num_threads() if threads is None else int(threads),
                            "machine": platform.machine(), "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)
