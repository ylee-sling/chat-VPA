# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
tasks.py -- HumanEval and MBPP problems as (model input, test harness) pairs.

``context`` is the benchmark's own text (HumanEval: the prompt; MBPP: the task text and its first test); the
prompt formats built from it are in ``chat_vpa.data``. A sample is a whole program and is executed as

    HumanEval   prompt + sample + test + "check(entry_point)"
                (the prompt keeps its imports and helper functions; the sample's definition replaces the
                prompt's docstring-only one)
    MBPP        test_setup_code + sample + the asserts of test_list

and passes iff the program exits with status 0. ``reference`` is the canonical solution in the same form, so
running it checks the harness itself (``python -m chat_vpa.evaluate --decoder reference``).

MBPP splits (Austin et al. 2021): prompt 1-10, test 11-510, validation 511-600, train 601-974.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["MBPP_SPLITS", "EvalTask", "load_humaneval_tasks", "load_mbpp_tasks"]

MBPP_SPLITS: dict[str, tuple[int, int]] = {
    "prompt": (1, 10), "test": (11, 510), "validation": (511, 600), "train": (601, 974), "all": (1, 10**9),
}


def _read_jsonl(path: str | Path) -> list[dict]:
    """The rows of a JSON-lines file, gzip-compressed if its name ends in ``.gz`` (blank lines skipped)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
    return rows


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    context: str                # model input c
    prefix: str                 # code run before the sample
    suffix: str                 # tests run after the sample
    reference: str              # canonical solution, in the form the model writes

    def program(self, sample: str) -> str:
        return self.prefix + sample.rstrip() + "\n" + self.suffix


def load_humaneval_tasks(path: str | Path) -> list[EvalTask]:
    tasks = []
    for row in _read_jsonl(path):
        try:
            prompt, entry = row["prompt"], row["entry_point"]
            tasks.append(EvalTask(
                task_id=row["task_id"], context=prompt, prefix=prompt.rstrip() + "\n\n\n",
                suffix="\n\n" + row["test"].strip() + f"\n\n\ncheck({entry})\n",
                reference=prompt + row["canonical_solution"],
            ))
        except KeyError as exc:
            raise ValueError(f"{path}: HumanEval record without field {exc}") from exc
    return tasks


def load_mbpp_tasks(path: str | Path, split: str = "test") -> list[EvalTask]:
    if split not in MBPP_SPLITS:
        raise ValueError(f"unknown MBPP split {split!r}; choose one of {sorted(MBPP_SPLITS)}")
    low, high = MBPP_SPLITS[split]
    tasks = []
    for row in _read_jsonl(path):
        try:
            task_id = int(row["task_id"])
            if not low <= task_id <= high:
                continue
            tests = row.get("test_list") or []
            setup = (row.get("test_setup_code") or "").strip()
            tasks.append(EvalTask(
                task_id=f"MBPP/{task_id}",
                context=row["text"] + ("\n" + tests[0] if tests else ""),
                prefix=(setup + "\n\n") if setup else "",
                suffix="\n\n" + "\n".join(tests) + "\n",
                reference=row["code"].replace("\r\n", "\n"),
            ))
        except KeyError as exc:
            raise ValueError(f"{path}: MBPP record without field {exc}") from exc
    return tasks
