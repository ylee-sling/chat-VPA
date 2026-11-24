# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e0_report.py -- E0's report (docs/e0.md): the table of Section 6, read by the rules of Section 5, with what
Section 7 reports alongside.

    python -m chat_vpa.e0_report --runs runs --baseline ~/text-VPA/runs/stage0-a/sweep/mbpp/ar \\
        --predecessor ~/text-VPA/runs/stage0-a/sweep/mbpp --data runs/e0-data \\
        --mbpp ~/text-VPA/data2/eval/mbpp.jsonl --out runs/e0-report

(``bash scripts/ncp/e0_eval.sh report`` runs it with the server's paths.)

Inputs. For each run NAME of T16, R16, T32, R32 that exists, ``RUNS/e0-NAME/eval/mbpp/<tag>/{summary.json,
samples.jsonl}`` (``block_evaluate.py`` through ``scripts/ncp/e0_eval.sh``; tags ``block-B16-K4``,
``block-B16-K4-prefix``, ``block-B16-K4-confidence``, ...) and the training run's own files (``config.json``,
``metrics.jsonl``, ``timing.json``, ``memory.json``). The AR baseline's samples: the released model's
``generate``, measured on the predecessor and reproduced 450 of 450 by the port check. Optionally the
predecessor's sweep (the whole-canvas rows and their intervals), the rollout writer's directory (the targets)
and MBPP's tests (the def-line check's ``copied``). A run or a configuration that is missing is printed as
``--`` and every rule that needs it as pending: H1 and the B = 16 half of H2 can be read after T16 and R16 (C12).

Statistics (Section 5). Pass@1 is the mean over problems of the share of a problem's samples that pass. Every
interval is the 95% percentile interval of a bootstrap over problems, 10,000 resamples with seed 0, and a
quantity that involves several configurations is computed on the same resampled problems for all of them (the
index matrix of ``report.bootstrap_ratio``, so that a ratio here equals its value there).

The rules, as docs/e0.md fixes them:

- H1 (5.1): rho = Pass@1(T16, (16, 4), exact kernel) / Pass@1(AR). Holds at rho >= 0.5, fails below; clear when
  the interval lies on one side of 0.5 (lower end >= 0.5, or upper end < 0.5, as in ``report.py``).
- H2 (C10): D(B, K) = Pass@1(arm T) - Pass@1(arm R), exact kernel. (i) exceeds at a cell if D > 0, clear if the
  interval's lower end is above 0; (ii) grows if D(16, 4) <= D_8 <= D(32, 2), with D_8 the mean of D(16, 2) and
  D(32, 4), clear if the interval of D(32, 2) - D(16, 4) lies above 0; (iii) holds if (i) holds at all four cells
  and (ii) holds. A failed (ii) with arm R at the floor at every cell is "not testable at this floor". And (C6)
  H2 is read with the held-out loss curves: where arm R's rises while arm T's does not, the repetition favours
  arm T, and the reading says so.
- H3 (5.3): Pass@1(prefix-growing) - Pass@1(exact kernel), arm T, at each cell; at least as good if >= 0;
  decided at (16, 4). An interval that includes 0 is read as the block structure and the target having removed
  the order effect.
- If H1 fails (5.4, C11): arm R near the floor is Pass@1(R16, (16, 4), exact) <= 2.0%; a clear gain of arm T is
  a lower end of D(16, 4)'s interval above 0. A clear gain selects the second row of Section 5.4's table, arm R
  near the floor without one the first, and neither the third.

Two readings where the text needs a number that docs/e0.md does not give (proposed on 23 November 2026,
docs/e0.md Section 8):

- "arm R at the floor at every cell" (C10's caveat): arm R's exact-kernel Pass@1 is at most 2.0%, the floor of
  C11, at all four cells;
- "arm R's held-out loss rises" (C6): a run's pooled held-out L_DFM on its own target is higher at its last
  evaluation than at its lowest one. The rise is printed with it.

The report also checks every configuration against what Section 5 and C8 fix (problems, samples, split, token
draw, no early exit, fused commit, no registers, the run's final checkpoint, the pass audit) and lists what does
not match under "Deviations found by the report". Writes ``OUT/report.md`` and ``OUT/report.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from chat_vpa.analyze import classify, def_line_summary, error_line, load_mbpp_tests, summarize_samples

__all__ = ["CELLS", "FLOOR", "RUN_NAMES", "THRESHOLD", "curve_reading", "failed_h1_reading", "h1_verdict",
           "h2_verdict", "h3_verdict", "paired_bootstrap", "wrong_argument_count"]

RUN_NAMES: tuple[str, ...] = ("T16", "R16", "T32", "R32")
ARMS = {"T": "rollouts", "R": "references"}
CELLS: tuple[tuple[int, int], ...] = ((16, 4), (16, 2), (32, 4), (32, 2))
ORDERS: tuple[str, ...] = ("exact", "prefix", "confidence")
H1_CELL = (16, 4)
THRESHOLD = 0.5            # H1 (Section 5.1)
FLOOR = 0.02               # "near the floor" (Section 5.4, C11)
RESAMPLES, SEED = 10_000, 0
EPS = 1e-9                 # Pass@1 moves in steps of 1/450; below this, two values are equal
AR_DOCUMENTED = 0.238      # the released generate on MBPP validation (measured on the predecessor)
# The two readings of the module docstring that docs/e0.md (Section 8) proposes; printed with every report.
READINGS = ("proposed on 23 November 2026 and not yet confirmed by the author (docs/e0.md, Section 8): arm R is "
            "\"at the floor everywhere\" when its exact-kernel Pass@1 is at most 2.0% at all four cells, and a "
            "held-out curve \"rises\" when its last value is above its lowest")

# What Section 5 and C8 fix for every evaluated configuration of a block run.
FIXED: dict[str, Any] = {"benchmark": "mbpp", "split": "validation", "problems": 90, "samples_per_problem": 5,
                         "temperature": 0.2, "top_p": 0.95, "top_k": 1024, "exit_budget": 0.0, "schedule": "cosine",
                         "fused_commit": True, "registers": 0, "adapters": True, "causal_window": False,
                         "canvas_len": 256, "seed": 0, "checkpoint_stage": "e0", "pass_audit": "passed"}

# The whole-canvas rows of Section 6: label, tokens per pass, and per column the tag in the predecessor's sweep
# with the Pass@1 that docs/e0.md documents (all measured on the predecessor, runs/stage0-a's sweep).
PREDECESSOR: tuple[tuple[str, str, dict[str, tuple[str, float]]], ...] = (
    ("Whole-canvas realizer, left to right", "1", {"prefix": ("realizer-K256-left_to_right", 0.171)}),
    ("Whole-canvas realizer, K = 32", "4 to 6", {"exact": ("realizer-K32", 0.011),
                                               "prefix": ("realizer-K32-left_to_right", 0.016),
                                               "confidence": ("realizer-K32-confidence", 0.020)}),
    ("Whole-canvas realizer, K = 8", "32 positions", {"exact": ("realizer-K8", 0.004),
                                                     "prefix": ("realizer-K8-left_to_right", 0.0),
                                                     "confidence": ("realizer-K8-confidence", 0.0)}),
)

SECTION_5_4 = {
    "floor": "against block-causal few-step decoding of this backbone at this adaptation budget; the student is not "
             "trained before the cause is found",
    "gain": "for more distillation or fewer tokens per pass",
    "neither": "no claim; reported with the loss curves and the outcome tables, and the author decides the next run",
}

Interval = tuple[float, float, float]          # point, low, high


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------

def paired_bootstrap(vectors: Sequence[Sequence[float]], statistic: Callable[..., Any],
                     resamples: int = RESAMPLES, seed: int = SEED) -> Interval:
    """``statistic`` of the vectors' means, and its 95% percentile interval over ``resamples`` bootstrap draws of
    problems, the same draw for every vector. Draws on which the statistic is not finite (a ratio over 0) are
    left out. With one vector and the identity this is the interval of a Pass@1; with (a, b) and a / b it is
    ``report.bootstrap_ratio``."""
    arrays = [np.asarray(v, dtype=np.float64) for v in vectors]
    size = arrays[0].size if arrays else 0
    if size == 0 or any(a.ndim != 1 or a.size != size for a in arrays):
        raise ValueError("need equally long, non-empty vectors of per-problem pass rates")
    with np.errstate(divide="ignore", invalid="ignore"):
        point = float(statistic(*[a.mean() for a in arrays]))
        index = np.random.default_rng(seed).integers(0, size, size=(resamples, size))
        values = np.asarray(statistic(*[a[index].mean(axis=1) for a in arrays]), dtype=np.float64)
    values = values[np.isfinite(values)]
    point = point if math.isfinite(point) else math.nan
    if values.size == 0:
        return point, math.nan, math.nan
    low, high = np.percentile(values, [2.5, 97.5])
    return point, float(low), float(high)


def _ratio(a: Any, b: Any) -> Any:
    return a / b


def _difference(a: Any, b: Any) -> Any:
    return a - b


def _difference_of_differences(a: Any, b: Any, c: Any, d: Any) -> Any:
    return (a - b) - (c - d)


# ---------------------------------------------------------------------------------------------
# The rules of Section 5
# ---------------------------------------------------------------------------------------------

def h1_verdict(ratio: Interval | None) -> dict[str, Any]:
    """H1 on rho = Pass@1(T16, (16, 4), exact) / Pass@1(AR) and its interval (None: T16 not evaluated)."""
    if ratio is None:
        return {"status": "pending", "clear": False, "text": "pending: T16 at (16, 4) with the exact kernel"}
    rho, low, high = ratio
    if math.isnan(rho):
        return {"status": "undecided", "clear": False, "text": "undecided: the AR baseline solves nothing"}
    if rho >= THRESHOLD - EPS:
        status, clear = "holds", low >= THRESHOLD - EPS
    else:
        status, clear = "fails", high < THRESHOLD - EPS
    return {"status": status, "clear": clear, "rho": rho, "low": low, "high": high,
            "text": ("clearly " if clear else "") + status}


def h2_verdict(d: Mapping[tuple[int, int], Interval], growth: Interval | None,
               arm_r: Mapping[tuple[int, int], float]) -> dict[str, Any]:
    """C10 on the differences D(B, K) (T - R, exact kernel) found, the interval of D(32, 2) - D(16, 4) (None
    unless all four cells exist) and arm R's exact-kernel Pass@1 per cell."""
    exceeds = {cell: {"holds": d[cell][0] > EPS, "clear": d[cell][1] > EPS} for cell in CELLS if cell in d}
    out: dict[str, Any] = {"exceeds": exceeds, "missing": [cell for cell in CELLS if cell not in d]}
    if out["missing"] or growth is None:
        partial = ", ".join(f"({b}, {k}): {'holds' if e['holds'] else 'does not hold'}"
                            + (" clearly" if e["clear"] else "") for (b, k), e in exceeds.items())
        out.update(status="pending", grows=None, text="pending: (ii) and (iii) need all four cells"
                   + (f"; (i) so far: {partial}" if partial else ""))
        return out
    d8 = (d[(16, 2)][0] + d[(32, 4)][0]) / 2
    grows = d[(16, 4)][0] <= d8 + EPS and d8 <= d[(32, 2)][0] + EPS
    floor_everywhere = all(arm_r[cell] <= FLOOR + EPS for cell in CELLS)
    if grows:
        grows_text = ("clearly " if growth[1] > EPS else "") + "holds"
    elif floor_everywhere:
        grows_text = "not testable at this floor (arm R at most 2.0% at every cell)"
    else:
        grows_text = "does not hold"
    all_exceed = all(e["holds"] for e in exceeds.values())
    holds = all_exceed and grows
    text = "holds" if holds else "does not hold"
    if not holds and not grows and floor_everywhere:
        text += ": (ii) is not testable at this floor, so this is not evidence against Proposition 4"
    out.update(status="holds" if holds else "does not hold", d8=d8, grows=grows, grows_clear=grows and growth[1] > EPS,
               grows_text=grows_text, floor_everywhere=floor_everywhere, all_exceed=all_exceed, text=text)
    return out


def h3_verdict(d: Mapping[tuple[int, int], Interval]) -> dict[str, Any]:
    """H3 on prefix-growing minus exact kernel, arm T, per cell found; decided at (16, 4)."""
    cells = {}
    for cell in CELLS:
        if cell not in d:
            continue
        point, low, high = d[cell]
        if low <= EPS and high >= -EPS:
            reading = "the interval includes 0: read as the block structure and the target having removed the order effect"
        elif low > EPS:
            reading = "prefix-growing is clearly better: an order effect remains, in the direction of Finding 1"
        else:
            reading = "the exact kernel is clearly better"
        cells[cell] = {"at_least_as_good": point >= -EPS, "reading": reading}
    if H1_CELL not in cells:
        return {"status": "pending", "cells": cells, "text": "pending: T16 at (16, 4), both orders"}
    decided = cells[H1_CELL]
    status = "holds" if decided["at_least_as_good"] else "fails"
    return {"status": status, "cells": cells, "text": f"{status}; {decided['reading']}"}


def failed_h1_reading(h1: Mapping[str, Any], arm_r_16_4: float | None, d_16_4: Interval | None) -> dict[str, Any]:
    """Section 5.4 with C11's numbers."""
    if h1["status"] != "fails":
        why = {"pending": "is pending", "undecided": "is undecided"}.get(h1["status"], "does not fail")
        return {"applies": False, "text": f"does not apply: H1 {why}"}
    if arm_r_16_4 is None or d_16_4 is None:
        return {"applies": True, "row": None, "text": "pending: R16 at (16, 4) with the exact kernel"}
    near_floor = arm_r_16_4 <= FLOOR + EPS
    clear_gain = d_16_4[1] > EPS
    row = "gain" if clear_gain else ("floor" if near_floor else "neither")
    return {"applies": True, "row": row, "near_floor": near_floor, "clear_gain": clear_gain,
            "text": SECTION_5_4[row]}


def curve_reading(curve: Sequence[tuple[int, float]], final_step: int) -> dict[str, Any] | None:
    """Whether a held-out loss curve rises: its last value against its lowest (None until the final step)."""
    if not curve or curve[-1][0] < final_step:
        return None
    low_step, low = min(curve, key=lambda point: (point[1], point[0]))
    final = curve[-1][1]
    return {"final": final, "lowest": low, "lowest_step": low_step, "rise": final - low, "rises": final > low + EPS}


# ---------------------------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------------------------

_ARGUMENT_COUNT = re.compile(r"TypeError: .*\(\) (?:takes .* positional arguments? but \d+ (?:was|were) given|"
                             r"missing \d+ required (?:positional|keyword-only) arguments?)")


def wrong_argument_count(row: Mapping[str, Any]) -> bool:
    """A sample that fails at run time because a function was called with the wrong number of arguments (the
    symptom of a ``def`` line that does not fit the test's call)."""
    return classify(dict(row)) == "runtime" and bool(_ARGUMENT_COUNT.search(error_line(str(row.get("detail", "")))))


def outcomes(rows: Sequence[Mapping[str, Any]], tests: Mapping[str, str] | None) -> dict[str, Any]:
    """Section 7's outcomes of one configuration: ``analyze.py``'s six outcomes and def-line check, and the share
    of samples that fail on a wrong argument count."""
    total = max(len(rows), 1)
    return {**summarize_samples([dict(r) for r in rows]), "def_line": def_line_summary([dict(r) for r in rows], tests),
            "wrong_argument_count": sum(wrong_argument_count(r) for r in rows) / total}


# ---------------------------------------------------------------------------------------------
# Reading the files
# ---------------------------------------------------------------------------------------------

def _read_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def per_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[bool]]:
    tasks: dict[str, list[bool]] = {}
    for row in rows:
        tasks.setdefault(str(row["task_id"]), []).append(bool(row["passed"]))
    return tasks


class Configuration:
    """One evaluated configuration: its summary, its samples and its per-problem pass rates."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.summary: dict[str, Any] = (json.loads((directory / "summary.json").read_text())
                                        if (directory / "summary.json").exists() else {})
        self.rows = _read_rows(directory / "samples.jsonl")
        self.tasks = per_task(self.rows)

    @staticmethod
    def find(directory: Path) -> Configuration | None:
        return Configuration(directory) if (directory / "samples.jsonl").exists() else None

    def vector(self, tasks: Sequence[str]) -> np.ndarray:
        return np.array([sum(self.tasks[t]) / len(self.tasks[t]) for t in tasks], dtype=np.float64)


def block_tag(block: int, steps: int, order: str) -> str:
    return f"block-B{block}-K{steps}" + ("" if order == "exact" else f"-{order}")


def held_out_curve(metrics: Sequence[Mapping[str, Any]], target: str) -> list[tuple[int, float]]:
    """(step, pooled held-out L_DFM on ``target``) from a run's metrics; a repeated step keeps its last value."""
    key = f"eval/val_{target}_loss"
    points = {int(r["step"]): float(r[key]) for r in metrics if key in r}
    return sorted(points.items())


def _json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------------

def _pct(iv: Interval | None) -> str:
    if iv is None:
        return "--"
    point, low, high = iv
    if math.isnan(low):
        return f"{100 * point:.1f}"
    return f"{100 * point:.1f} [{100 * low:.1f}, {100 * high:.1f}]"


def _ratio_text(iv: Interval | None) -> str:
    if iv is None:
        return "--"
    point, low, high = iv
    if math.isnan(point):
        return "undefined"
    return f"{point:.3f}" if math.isnan(low) else f"{point:.3f} [{low:.3f}, {high:.3f}]"


def _points(iv: Interval | None) -> str:
    if iv is None:
        return "--"
    point, low, high = iv
    return f"{100 * point:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]"


def build(runs_dir: Path, baseline_dir: Path, predecessor_dir: Path | None = None, data_dir: Path | None = None,
          mbpp_path: Path | None = None, resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, Any]:
    """Everything the report says, as one dictionary (``render`` turns it into Markdown)."""
    baseline = Configuration.find(baseline_dir)
    if baseline is None:
        raise SystemExit(f"{baseline_dir}/samples.jsonl is missing: the AR baseline's samples are needed")
    tasks = sorted(baseline.tasks)
    deviations: list[str] = []
    counts = Counter(len(flags) for flags in baseline.tasks.values())
    if len(tasks) != FIXED["problems"] or set(counts) != {FIXED["samples_per_problem"]}:
        deviations.append(f"the AR baseline has {len(tasks)} problems with {dict(counts)} samples each, not "
                          f"{FIXED['problems']} x {FIXED['samples_per_problem']}")
    tests = load_mbpp_tests(mbpp_path) if mbpp_path else None

    def aligned(config: Configuration, name: str) -> np.ndarray:
        if sorted(config.tasks) != tasks:
            raise SystemExit(f"{name} ({config.directory}) was scored on other problems than the AR baseline")
        return config.vector(tasks)

    def boot(vectors: Sequence[np.ndarray], statistic: Callable[..., Any]) -> Interval:
        return paired_bootstrap(vectors, statistic, resamples, seed)

    ar = aligned(baseline, "the AR baseline")
    ar_pass = boot([ar], lambda a: a)
    if abs(ar_pass[0] - AR_DOCUMENTED) > 0.0005:
        deviations.append(f"the AR baseline's samples give {100 * ar_pass[0]:.2f}%, not the predecessor's "
                          f"{100 * AR_DOCUMENTED:.1f}%")
    rows: list[dict[str, Any]] = [{"model": "Gemma 4 E2B, AR, released `generate`", "target": "n/a", "cell": None,
                                   "g": "1", "source": "measured, predecessor", "exact": ar_pass,
                                   "prefix": None, "confidence": None, "ratio": (1.0, math.nan, math.nan)}]
    for label, g, columns in PREDECESSOR:
        row: dict[str, Any] = {"model": label, "target": "human references", "cell": None, "g": g,
                               "source": "measured, predecessor", "documented": {}}
        vectors: dict[str, np.ndarray] = {}
        for order in ORDERS:
            row[order] = None
            if order not in columns:
                continue
            tag, documented = columns[order]
            config = Configuration.find(predecessor_dir / tag) if predecessor_dir else None
            if config is None:
                row["documented"][order] = documented
                row[order] = (documented, math.nan, math.nan)
                continue
            vectors[order] = aligned(config, f"the predecessor's {tag}")
            row[order] = boot([vectors[order]], lambda a: a)
        ratio_order = "exact" if "exact" in columns else "prefix"
        row["ratio_order"] = ratio_order
        if ratio_order in vectors:
            row["ratio"] = boot([vectors[ratio_order], ar], _ratio)
        else:                                       # documented values: the point, without an interval
            row["ratio"] = (columns[ratio_order][1] / ar_pass[0] if ar_pass[0] > 0 else math.nan, math.nan, math.nan)
        rows.append(row)

    # The block runs.
    runs: dict[str, dict[str, Any]] = {}
    vectors_of: dict[tuple[str, int, int, str], np.ndarray] = {}
    configs_of: dict[tuple[str, int, int, str], Configuration] = {}
    for name in RUN_NAMES:
        run_dir = runs_dir / f"e0-{name}"
        arm, block = ARMS[name[0]], int(name[1:])
        config_json = _json(run_dir / "config.json")
        metrics = _read_rows(run_dir / "metrics.jsonl") if (run_dir / "metrics.jsonl").exists() else []
        info: dict[str, Any] = {"exists": run_dir.is_dir(), "arm": arm, "block": block, "config": config_json,
                                "timing": _json(run_dir / "timing.json"), "memory": _json(run_dir / "memory.json"),
                                "curves": {t: held_out_curve(metrics, t) for t in ("references", "rollouts")},
                                "last_train": next((r for r in reversed(metrics) if "train/loss" in r), None),
                                "peak_gib": max((float(r["train/peak_gib"]) for r in metrics if "train/peak_gib" in r),
                                                default=None),
                                "configurations": {}}
        final_step = int(config_json["steps"]) if config_json else None
        info["final_step"] = final_step
        if config_json and (config_json.get("arm") != arm or int(config_json.get("block_size", 0)) != block):
            deviations.append(f"{name}: the run's configuration says arm {config_json.get('arm')}, block size "
                              f"{config_json.get('block_size')}")
        for cell in CELLS:
            if cell[0] != block:
                continue
            for order in ORDERS:
                tag = block_tag(*cell, order)
                config = Configuration.find(run_dir / "eval" / "mbpp" / tag)
                if config is None:
                    continue
                key = (name, *cell, order)
                vectors_of[key] = aligned(config, f"{name} {tag}")
                configs_of[key] = config
                s = config.summary
                expected = {**FIXED, "arm": arm, "train_block_size": block, "block_size": cell[0],
                            "num_steps": cell[1], "order": order}
                if final_step is not None:
                    expected["step"] = final_step
                for field_name, value in expected.items():
                    if s.get(field_name) != value:
                        deviations.append(f"{name} {tag}: {field_name} is {s.get(field_name)!r}, not {value!r}")
                info["configurations"][tag] = {
                    "pass@1": boot([vectors_of[key]], lambda a: a),
                    "outcomes": outcomes(config.rows, tests),
                    **{k: s.get(k) for k in ("parse_rate", "mean_tokens", "mean_blocks", "mean_passes_per_block",
                                             "stops", "seconds_per_block", "seconds_per_pass", "rows_per_batch",
                                             "pass_audit", "step")}}
        runs[name] = info

    def pass_of(name: str, cell: tuple[int, int], order: str) -> np.ndarray | None:
        return vectors_of.get((name, *cell, order))

    for arm_letter in ("R", "T"):
        for cell in CELLS:
            name = f"{arm_letter}{cell[0]}"
            label = f"Block-DFM, arm {arm_letter}" + (" (H1, H3)" if arm_letter == "T" and cell == H1_CELL else "")
            row = {"model": label, "target": "AR rollouts" if arm_letter == "T" else "human references",
                   "cell": cell, "g": str(cell[0] // cell[1]), "source": "measured, E0"}
            for order in ORDERS:
                vec = pass_of(name, cell, order)
                row[order] = boot([vec], lambda a: a) if vec is not None else None
            exact = pass_of(name, cell, "exact")
            row["ratio"] = boot([exact, ar], _ratio) if exact is not None else None
            rows.append(row)

    # H1.
    t16 = pass_of("T16", H1_CELL, "exact")
    h1 = h1_verdict(boot([t16, ar], _ratio) if t16 is not None else None)

    # H2 (C10) and the loss curves (C6).
    d: dict[tuple[int, int], Interval] = {}
    arm_r: dict[tuple[int, int], float] = {}
    for cell in CELLS:
        t, r = pass_of(f"T{cell[0]}", cell, "exact"), pass_of(f"R{cell[0]}", cell, "exact")
        if r is not None:
            arm_r[cell] = float(r.mean())
        if t is not None and r is not None:
            d[cell] = boot([t, r], _difference)
    growth = None
    if all(cell in d for cell in CELLS):
        growth = boot([pass_of("T32", (32, 2), "exact"), pass_of("R32", (32, 2), "exact"),
                       pass_of("T16", (16, 4), "exact"), pass_of("R16", (16, 4), "exact")], _difference_of_differences)
    h2 = h2_verdict(d, growth, arm_r)
    h2["growth"] = growth
    h2["differences"] = d
    repetition: dict[int, dict[str, Any]] = {}
    for block in (16, 32):
        r_info, t_info = runs[f"R{block}"], runs[f"T{block}"]
        r_curve = (curve_reading(r_info["curves"]["references"], r_info["final_step"])
                   if r_info["final_step"] else None)
        t_curve = (curve_reading(t_info["curves"]["rollouts"], t_info["final_step"])
                   if t_info["final_step"] else None)
        if r_curve is None or t_curve is None:
            repetition[block] = {"status": "pending", "text": f"pending: both B = {block} runs to their last step"}
            continue
        favours = r_curve["rises"] and not t_curve["rises"]
        text = (f"arm R's held-out loss on its references {'rises' if r_curve['rises'] else 'does not rise'} "
                f"(last {r_curve['final']:.4f}, lowest {r_curve['lowest']:.4f} at step {r_curve['lowest_step']}); "
                f"arm T's on its rollouts {'rises' if t_curve['rises'] else 'does not rise'} "
                f"(last {t_curve['final']:.4f}, lowest {t_curve['lowest']:.4f} at step {t_curve['lowest_step']})")
        if favours:
            text += ": the repetition favours arm T at this block size, and H2's reading says so"
        repetition[block] = {"status": "favours arm T" if favours else "no sign", "references": r_curve,
                             "rollouts": t_curve, "text": text}
    h2["repetition"] = repetition

    # H3 and Section 5.4.
    order_d: dict[tuple[int, int], Interval] = {}
    for cell in CELLS:
        prefix, exact = pass_of(f"T{cell[0]}", cell, "prefix"), pass_of(f"T{cell[0]}", cell, "exact")
        if prefix is not None and exact is not None:
            order_d[cell] = boot([prefix, exact], _difference)
    h3 = h3_verdict(order_d)
    h3["differences"] = order_d
    section_5_4 = failed_h1_reading(h1, arm_r.get(H1_CELL), d.get(H1_CELL))

    data_summary = _json(data_dir / "summary.json") if data_dir else None
    return {"tasks": len(tasks), "resamples": resamples, "seed": seed, "deviations": deviations, "rows": rows,
            "h1": h1, "h2": h2, "h3": h3, "section_5_4": section_5_4, "runs": runs, "data": data_summary,
            "copied_checked": tests is not None}


def _cell(cell: tuple[int, int] | None) -> str:
    return "" if cell is None else f"({cell[0]}, {cell[1]})"


def render(report: Mapping[str, Any]) -> str:
    """Markdown of ``build``'s dictionary."""
    runs = report["runs"]
    found = []
    for name in RUN_NAMES:
        info = runs[name]
        if not info["exists"]:
            found.append(f"{name}: not found")
            continue
        step = int(info["last_train"]["step"]) if info["last_train"] else 0
        final = f"{info['final_step']:,}" if info["final_step"] else "?"
        found.append(f"{name}: step {step:,} of {final}, {len(info['configurations'])} of 6 configurations evaluated")
    lines = ["# E0: Block-DFM on the frozen Gemma 4 E2B against its autoregressive decoding", "",
             "Read by the rules of `docs/e0.md`, Section 5 (frozen on 23 November 2026). MBPP validation, "
             f"{report['tasks']} problems; 95% intervals from a bootstrap over problems, paired between "
             f"configurations, {report['resamples']:,} resamples, seed {report['seed']}. Block rows are *measured* "
             "by E0; the AR and whole-canvas rows are *measured on the predecessor*.", "",
             "Runs: " + "; ".join(found) + ".", "", "## Deviations found by the report", ""]
    lines += [f"- {d}" for d in report["deviations"]] or ["- none"]

    lines += ["", "## The table of Section 6 (Pass@1 in %, 95% interval)", "",
              "| Model | Target | (B, K) | g | exact kernel | prefix-growing | confidence | ratio to AR (exact) |",
              "|---|---|---|---|---|---|---|---|"]
    for row in report["rows"]:
        cells = []
        for order in ORDERS:
            notes = []
            if row["model"].startswith("Whole-canvas") and order == "prefix" and row[order] is not None:
                notes.append("left to right")
            if order in row.get("documented", {}):
                notes.append("documented")
            cells.append(_pct(row[order]) + (f" ({'; '.join(notes)})" if notes else ""))
        lines.append(f"| {row['model']} | {row['target']} | {_cell(row['cell'])} | {row['g']} | {' | '.join(cells)} | "
                     f"{_ratio_text(row['ratio'])} |")
    lines += ["", "The whole-canvas rows' ratio uses their exact-kernel column, and the left-to-right row its own. "
              "\"(documented)\": the predecessor's sweep was not given or lacks the configuration, so the value is "
              "`docs/e0.md`'s, without an interval (and so is its ratio)."]

    h1 = report["h1"]
    lines += ["", f"## H1, the bridge: {h1['text']}", "",
              "Pass@1 of T16 at (16, 4) with the exact kernel over Pass@1 of the AR baseline; holds at >= 0.5, "
              "clear when the interval lies on one side of 0.5 (Section 5.1)."]
    if "rho" in h1:
        lines += ["", f"- rho = {_ratio_text((h1['rho'], h1['low'], h1['high']))}"]

    h2 = report["h2"]
    lines += ["", f"## H2, the target: {h2['text']}", "",
              "D(B, K) = Pass@1(arm T) - Pass@1(arm R), exact kernel, in points, with paired intervals (C10).", "",
              "| (B, K) | g | D | (i) exceeds |", "|---|---|---|---|"]
    for cell in CELLS:
        if cell in h2["differences"]:
            e = h2["exceeds"][cell]
            verdict = ("clearly " if e["clear"] else "") + ("holds" if e["holds"] else "does not hold")
            lines.append(f"| {_cell(cell)} | {cell[0] // cell[1]} | {_points(h2['differences'][cell])} | {verdict} |")
        else:
            lines.append(f"| {_cell(cell)} | {cell[0] // cell[1]} | -- | pending |")
    if h2.get("grows") is not None:
        lines += ["", f"- (ii) grows: {h2['grows_text']}. D(16, 4) = {100 * h2['differences'][(16, 4)][0]:+.1f}, "
                  f"D_8 = {100 * h2['d8']:+.1f}, D(32, 2) = {100 * h2['differences'][(32, 2)][0]:+.1f}; "
                  f"D(32, 2) - D(16, 4) = {_points(h2['growth'])}.",
                  f"- (iii) {'(i) holds at all four cells' if h2['all_exceed'] else '(i) does not hold at every cell'}"
                  f" and (ii) {'holds' if h2['grows'] else 'does not hold'}: H2 {h2['status']}."]
    lines += ["", f"Readings {READINGS}.", "", "Held-out loss curves (C6):"]
    lines += [f"- B = {block}: {info['text']}" for block, info in h2["repetition"].items()]

    h3 = report["h3"]
    lines += ["", f"## H3, the order: {h3['text']}", "",
              "Pass@1(prefix-growing) - Pass@1(exact kernel), arm T, in points; decided at (16, 4) (Section 5.3).", "",
              "| (B, K) | difference | at least as good | reading |", "|---|---|---|---|"]
    for cell in CELLS:
        if cell in h3["cells"]:
            c = h3["cells"][cell]
            lines.append(f"| {_cell(cell)} | {_points(h3['differences'][cell])} | "
                         f"{'yes' if c['at_least_as_good'] else 'no'} | {c['reading']} |")
        else:
            lines.append(f"| {_cell(cell)} | -- | pending | |")

    s54 = report["section_5_4"]
    lines += ["", f"## If H1 fails (Section 5.4): {s54['text']}"]
    if s54.get("row"):
        lines += [""]
        lines.append(f"Arm R near the floor (at most 2.0% at (16, 4)): {'yes' if s54['near_floor'] else 'no'}; "
                     f"a clear gain of arm T at (16, 4): {'yes' if s54['clear_gain'] else 'no'}.")

    lines += ["", "## Reported alongside (Section 7)", "", "### The targets", ""]
    data = report["data"]
    if data is None:
        lines.append("The rollout writer's summary was not given.")
    else:
        for name in ("validation", "train"):
            s = data[name]
            lines.append(f"- {name}: {s['records']} kept of quota {s['quota']}; kept rate {s['kept_rate']}; "
                         f"candidates {s.get('candidates', '--')} of {s['source_examples_read']} examples read; "
                         f"{s['generation_hours']} h")
            lines.append(f"  stops: {s['stops']}")
        lines.append(f"- kept `write` rollouts that define the named function: {data['named_function']}")
        for f, stats in data.get("lengths", {}).items():
            lines.append(f"- program tokens, {f}: {stats}")
        for f, stats in data.get("entropy", {}).items():
            lines.append(f"- entropy and NLL, {f}: {stats}")

    def share(value: Any) -> str:
        return "--" if value is None else f"{100 * value:.1f}"

    configurations = [(name, tag, c) for name in RUN_NAMES for tag, c in runs[name]["configurations"].items()]
    lines += ["", "### Outcomes, per configuration (% of samples)", "",
              "`copied` " + ("is checked against MBPP's asserts." if report["copied_checked"] else
                             "needs MBPP's tests (--mbpp)."), "",
              "| run | configuration | Pass@1 | parses | passed | empty | syntax | wrong | runtime | timeout | no def | "
              "bad def | literal | copied | wrong arg. count |", "|" + "---|" * 15]
    for name, tag, c in configurations:
        o, dl = c["outcomes"], c["outcomes"]["def_line"]
        lines.append(f"| {name} | {tag} | {_pct(c['pass@1'])} | {share(c['parse_rate'])} | "
                     + " | ".join(share(o[k]) for k in ("passed", "empty", "syntax", "wrong", "runtime", "timeout"))
                     + " | " + " | ".join(share(dl[k]) for k in ("no_def", "bad_def", "literal", "copied"))
                     + f" | {share(o['wrong_argument_count'])} |")
    lines += ["", "### Decoding, per configuration", "",
              "Tokens and blocks per sample; K-bar, the passes a block needed; seconds per block and per pass on the "
              "evaluation's device, with the rows of one batch (the samples of one problem).", "",
              "| run | configuration | tokens | blocks | K-bar | stops | s/block | s/pass | rows | pass audit |",
              "|" + "---|" * 10]
    for name, tag, c in configurations:
        stops = ", ".join(f"{k} {100 * v:.0f}%" for k, v in (c["stops"] or {}).items())
        lines.append(f"| {name} | {tag} | {c['mean_tokens']} | {c['mean_blocks']} | {c['mean_passes_per_block']} | "
                     f"{stops} | {c['seconds_per_block']} | {c['seconds_per_pass']} | {c['rows_per_batch'] or '--'} | "
                     f"{c['pass_audit']} |")

    lines += ["", "### Training", ""]
    for name in RUN_NAMES:
        info = runs[name]
        if not info["exists"]:
            lines.append(f"- {name}: not found")
            continue
        last = info["last_train"] or {}
        timing = info["timing"] or {}
        batch = info["memory"] or info["config"] or {}          # the memory check's choice, else the configured one
        loss = last.get("train/loss")
        lines.append(f"- {name}: step {int(last.get('step', 0)):,} of {info['final_step'] or '?'}; last training loss "
                     f"{'--' if loss is None else f'{float(loss):.4f}'}; {timing.get('steps_per_s', '--')} steps/s over "
                     f"the first {timing.get('steps', '--')} steps; peak {info['peak_gib']} GiB; micro-batch "
                     f"{batch.get('micro_batch', '--')} x {batch.get('grad_accum', '--')}")
        steps = sorted({s for curve in info["curves"].values() for s, _ in curve})
        if steps:
            refs, rolls = dict(info["curves"]["references"]), dict(info["curves"]["rollouts"])
            lines.append("  held-out L_DFM (references / rollouts): " + "; ".join(
                f"{s}: {refs.get(s, float('nan')):.4f} / {rolls.get(s, float('nan')):.4f}" for s in steps))

    lines += ["", "## What E0 cannot show (Section 5.5)", "",
              "One seed per run: the intervals cover the choice of problems, not the variance between runs. The "
              "rollout arm's teacher is the backbone itself, so its ceiling is the AR baseline. The result is about "
              "Python on 450 samples and about adapters on a frozen backbone, not about the student or about "
              "dialogue."]
    return "\n".join(lines) + "\n"


def _plain(value: Any) -> Any:
    """JSON-safe copy: tuple keys as "(B, K)" strings, NaN as None."""
    if isinstance(value, Mapping):
        return {(_cell(k) if isinstance(k, tuple) else str(k)): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return _plain(value.item())
    return value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E0's report (docs/e0.md, Sections 5 to 7)", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("runs"), help="the directory that holds e0-T16, e0-R16, ...")
    p.add_argument("--baseline", type=Path, required=True, help="the AR baseline's evaluation (samples.jsonl)")
    p.add_argument("--predecessor", type=Path, default=None, help="the predecessor's sweep (its mbpp directory)")
    p.add_argument("--data", type=Path, default=None, help="the rollout writer's directory")
    p.add_argument("--mbpp", type=Path, default=None, help="mbpp.jsonl, for the def-line check's `copied`")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    report = build(args.runs, args.baseline, args.predecessor, args.data, args.mbpp)
    text = render(report)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(text)
    (args.out / "report.json").write_text(json.dumps(_plain(report), indent=2))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
