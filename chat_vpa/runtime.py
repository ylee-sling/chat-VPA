# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
runtime.py -- checkpoints and metric logging shared by the trainers (from text-VPA's ``engine/runtime.py``).

Checkpoints are written atomically (temporary file + rename) and load with ``weights_only=True``. New
checkpoints carry the format ``chat-vpa/1``; the predecessor's ``text-vpa/1`` is read as well, so that its runs
(``~/text-VPA/runs/stage0-a/last.pt``) load unchanged.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

__all__ = ["CHECKPOINT_FORMAT", "READABLE_FORMATS", "MetricLogger", "load_checkpoint", "save_checkpoint"]

CHECKPOINT_FORMAT = "chat-vpa/1"
READABLE_FORMATS: tuple[str, ...] = (CHECKPOINT_FORMAT, "text-vpa/1")


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomic ``torch.save`` of a payload of tensors and plain Python containers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format": CHECKPOINT_FORMAT, **payload}, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("format") not in READABLE_FORMATS:
        raise ValueError(f"{path} is not a chat-VPA or text-VPA checkpoint (format {payload.get('format')!r})")
    return payload


# =============================================================================================
# Logging
# =============================================================================================
class MetricLogger:
    """JSON-lines metrics (always), console lines, and Weights & Biases when a project is given (rank 0)."""

    def __init__(self, output_dir: str | Path, enabled: bool = True, wandb_project: str | None = None,
                 run_name: str | None = None, config: dict[str, Any] | None = None) -> None:
        self.enabled = enabled
        self.path = Path(output_dir) / "metrics.jsonl"
        self._wandb = None
        self._start = time.time()
        if not enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if wandb_project:
            try:
                import wandb  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError("--wandb-project needs the wandb package (pip install wandb)") from exc
            self._wandb = wandb.init(project=wandb_project, name=run_name, config=config)

    def log(self, step: int, metrics: dict[str, float], prefix: str = "") -> None:
        if not self.enabled:
            return
        row: dict[str, Any] = {"step": step, "time_s": round(time.time() - self._start, 3)}
        for key, value in metrics.items():
            if isinstance(value, int | float | torch.Tensor):
                value = float(value)
                if not math.isfinite(value):                # JSON has no NaN / Infinity literals
                    value = "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
            row[f"{prefix}{key}"] = value
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        shown = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
                         if k not in ("time_s",))
        print(shown, file=sys.stdout, flush=True)
        if self._wandb is not None:
            self._wandb.log(row, step=step)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
