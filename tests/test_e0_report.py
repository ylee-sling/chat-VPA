# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""E0's report (chat_vpa/e0_report.py): its bootstrap against report.py and explicit loops, each rule of
docs/e0.md Section 5 at and around its thresholds, and a synthetic set of runs read end to end."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from chat_vpa import e0_report as er
from chat_vpa.report import bootstrap_ratio


def _rates(rng: np.random.Generator, size: int = 90) -> np.ndarray:
    return rng.integers(0, 6, size=size) / 5.0                      # c / 5 per problem


def _loop_interval(vectors, statistic, resamples, seed):
    """The bootstrap with Python loops over the same index matrix."""
    index = np.random.default_rng(seed).integers(0, len(vectors[0]), size=(resamples, len(vectors[0])))
    values = []
    for row in index:
        means = [sum(float(v[i]) for i in row) / len(row) for v in vectors]
        value = statistic(*means)
        if math.isfinite(value):
            values.append(value)
    point = statistic(*[sum(float(x) for x in v) / len(v) for v in vectors])
    return point, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def test_the_bootstrap_is_report_py_for_ratios_and_the_explicit_loop_for_the_rest():
    rng = np.random.default_rng(7)
    a, b, c, d = (_rates(rng) for _ in range(4))
    assert er.paired_bootstrap([a, b], lambda x, y: x / y) == bootstrap_ratio(a, b)   # 10,000 resamples, seed 0
    for vectors, statistic in (([a], lambda x: x), ([a, b], lambda x, y: x - y),
                               ([a, b, c, d], lambda w, x, y, z: (w - x) - (y - z))):
        got = er.paired_bootstrap(vectors, statistic, resamples=500, seed=3)
        assert got == pytest.approx(_loop_interval(vectors, statistic, 500, 3), rel=1e-12, abs=1e-12)
    zero = np.zeros(90)
    point, low, high = er.paired_bootstrap([a, zero], lambda x, y: x / y)
    assert math.isnan(point) and math.isnan(low) and math.isnan(high)
    with pytest.raises(ValueError):
        er.paired_bootstrap([a, b[:10]], lambda x, y: x - y)


def test_h1_at_and_around_one_half():
    assert er.h1_verdict(None)["status"] == "pending"
    assert er.h1_verdict((0.6, 0.52, 0.7))["text"] == "clearly holds"
    assert er.h1_verdict((0.6, 0.45, 0.7))["text"] == "holds"
    assert er.h1_verdict((0.5, 0.40, 0.6))["text"] == "holds"                  # at least half
    assert er.h1_verdict((0.55, 0.5, 0.6))["text"] == "clearly holds"          # report.py: lower end >= 0.5
    assert er.h1_verdict((0.4, 0.3, 0.49))["text"] == "clearly fails"
    assert er.h1_verdict((0.4, 0.3, 0.55))["text"] == "fails"
    assert er.h1_verdict((0.4, 0.3, 0.5))["text"] == "fails"                   # touches 0.5: not clear
    assert er.h1_verdict((math.nan, math.nan, math.nan))["status"] == "undecided"


def _d(points, width=0.01):
    return {cell: (p, p - width, p + width) for cell, p in points.items()}


def test_h2_follows_c10():
    cells = dict(zip(er.CELLS, (0.02, 0.03, 0.05, 0.06), strict=True))      # (16, 4), (16, 2), (32, 4), (32, 2)
    arm_r = dict.fromkeys(er.CELLS, 0.10)
    out = er.h2_verdict(_d(cells), (0.04, 0.01, 0.07), arm_r)
    assert out["status"] == "holds" and out["grows_text"] == "clearly holds" and out["d8"] == pytest.approx(0.04)
    assert all(e["holds"] and e["clear"] for e in out["exceeds"].values())
    assert er.h2_verdict(_d(cells), (0.04, -0.01, 0.07), arm_r)["grows_text"] == "holds"
    # D(16, 4) = D_8 = D(32, 2): "grows" holds with equality (<=).
    flat = dict.fromkeys(er.CELLS, 0.03)
    assert er.h2_verdict(_d(flat), (0.0, -0.02, 0.02), arm_r)["grows"] is True
    # A cell at D = 0 does not exceed, so H2 does not hold although the differences grow.
    zero_first = {**cells, (16, 4): 0.0}
    out = er.h2_verdict(_d(zero_first), (0.06, 0.03, 0.09), arm_r)
    assert out["exceeds"][(16, 4)]["holds"] is False and out["grows"] is True and out["status"] == "does not hold"
    # Not growing: D_8 = 0.01 < D(16, 4) = 0.05.
    shrinking = dict(zip(er.CELLS, (0.05, 0.01, 0.01, 0.06), strict=True))
    out = er.h2_verdict(_d(shrinking), (0.01, -0.02, 0.04), arm_r)
    assert out["grows"] is False and out["grows_text"] == "does not hold" and out["text"] == "does not hold"
    floor = dict.fromkeys(er.CELLS, 0.02)                                       # arm R at most 2.0% everywhere
    out = er.h2_verdict(_d(shrinking), (0.01, -0.02, 0.04), floor)
    assert out["grows_text"].startswith("not testable at this floor") and "not evidence against" in out["text"]
    assert er.h2_verdict(_d(shrinking), (0.01, -0.02, 0.04), {**floor, (32, 2): 0.0223})["floor_everywhere"] is False
    # After two runs: (i) on the B = 16 cells only.
    out = er.h2_verdict(_d({(16, 4): 0.02, (16, 2): -0.01}), None, {(16, 4): 0.1, (16, 2): 0.1})
    assert out["status"] == "pending" and out["missing"] == [(32, 4), (32, 2)]
    assert out["exceeds"][(16, 2)]["holds"] is False and "(16, 4): holds clearly" in out["text"]


def test_h3_is_decided_at_16_4_and_reads_the_interval():
    assert er.h3_verdict({(16, 2): (0.0, -0.01, 0.01)})["status"] == "pending"
    out = er.h3_verdict({(16, 4): (0.0, -0.02, 0.02), (32, 2): (-0.03, -0.05, -0.01)})
    assert out["status"] == "holds" and "removed the order effect" in out["text"]       # >= 0, interval holds 0
    assert out["cells"][(32, 2)]["at_least_as_good"] is False
    assert out["cells"][(32, 2)]["reading"] == "the exact kernel is clearly better"
    out = er.h3_verdict({(16, 4): (0.04, 0.01, 0.07)})
    assert out["status"] == "holds" and "an order effect remains" in out["text"]
    out = er.h3_verdict({(16, 4): (-0.01, -0.03, 0.01)})
    assert out["status"] == "fails" and "removed the order effect" in out["text"]
    # The floor caveat (24 November 2026): both orders at most 2.0% at (16, 4).
    zero = {(16, 4): (0.0, 0.0, 0.0), (32, 2): (0.0222, 0.0, 0.05)}
    out = er.h3_verdict(zero, {(16, 4): (0.0, 0.02), (32, 2): (0.02, 0.0422)})
    assert out["status"] == "not testable at this floor" and out["cells"][(16, 4)]["at_floor"] is True
    assert out["cells"][(32, 2)]["at_floor"] is False and "removed the order effect" in out["cells"][(32, 2)]["reading"]
    out = er.h3_verdict(zero, {(16, 4): (0.0, 0.0223)})                      # one order above the floor
    assert out["status"] == "holds" and "removed the order effect" in out["text"]


def test_section_5_4_selects_its_row():
    holds, fails = er.h1_verdict((0.6, 0.52, 0.7)), er.h1_verdict((0.1, 0.05, 0.2))
    assert er.failed_h1_reading(holds, 0.0, (0.1, 0.05, 0.2))["applies"] is False
    assert er.failed_h1_reading(er.h1_verdict(None), None, None)["text"] == "does not apply: H1 is pending"
    undecided = er.h1_verdict((math.nan, math.nan, math.nan))
    assert er.failed_h1_reading(undecided, 0.0, None)["text"] == "does not apply: H1 is undecided"
    assert er.failed_h1_reading(fails, None, None)["text"].startswith("pending")
    assert er.failed_h1_reading(fails, 0.02, (0.01, 0.002, 0.02))["row"] == "gain"    # a clear gain comes first
    out = er.failed_h1_reading(fails, 0.02, (0.01, 0.0, 0.02))                        # lower end 0: not clear
    assert out["row"] == "floor" and out["text"] == er.SECTION_5_4["floor"]
    assert er.failed_h1_reading(fails, 0.0223, (0.01, -0.01, 0.02))["row"] == "neither"


def test_a_held_out_curve_rises_when_its_last_value_is_above_its_lowest():
    curve = [(0, 5.0), (2500, 3.0), (5000, 2.9), (7500, 2.95), (10000, 3.1)]
    out = er.curve_reading(curve, 10000)
    assert out["rises"] is True and out["lowest_step"] == 5000 and out["rise"] == pytest.approx(0.2)
    assert er.curve_reading([(0, 5.0), (5000, 3.0), (10000, 2.8)], 10000)["rises"] is False
    assert er.curve_reading(curve[:3], 10000) is None                       # not at its last step yet


def _row(passed, detail="", parses=True, status="error", raw="def f(a):\n    return a\n"):
    return {"task_id": "MBPP/511", "sample": 0, "passed": passed, "status": "ok" if passed else status,
            "parses": parses, "detail": detail, "completion": raw, "raw": raw}


def test_the_wrong_argument_count_share():
    tb = "Traceback (most recent call last):\n  File \"p.py\", line 4, in <module>\n"
    yes = [tb + "TypeError: f() takes 2 positional arguments but 3 were given",
           tb + "TypeError: f() takes 1 positional argument but 2 were given",
           tb + "TypeError: Solution.f() takes from 1 to 2 positional arguments but 3 were given",
           tb + "TypeError: f() missing 1 required positional argument: 'b'",
           tb + "TypeError: f() missing 2 required positional arguments: 'a' and 'b'"]
    no = [tb + "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
          tb + "NameError: name 'g' is not defined", tb + "AssertionError"]
    assert all(er.wrong_argument_count(_row(False, d)) for d in yes)
    assert not any(er.wrong_argument_count(_row(False, d)) for d in no)
    assert not er.wrong_argument_count(_row(True, yes[0]))
    rows = [_row(False, yes[0]), _row(False, no[0]), _row(True), _row(False, no[2])]
    out = er.outcomes(rows, None)
    assert out["wrong_argument_count"] == 0.25 and out["passed"] == 0.25 and out["runtime"] == 0.5
    assert out["def_line"]["copied"] is None


# ---------------------------------------------------------------------------------------------
# A synthetic set of runs, read end to end
# ---------------------------------------------------------------------------------------------

TASKS = [f"MBPP/{i}" for i in range(511, 601)]                       # MBPP's validation split: 90 problems


def _write_config(directory, passing, summary=None):
    """samples.jsonl with ``passing`` passed samples (the first ones, in task order) and its summary."""
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "samples.jsonl", "w") as handle:
        for i, task in enumerate(TASKS):
            for j in range(5):
                passed = 5 * i + j < passing
                handle.write(json.dumps({"task_id": task, "sample": j, "passed": passed, "status": "ok" if passed
                                         else "error", "parses": True, "detail": "" if passed else "AssertionError",
                                         "completion": "def f(x):\n    return x\n", "raw": "def f(x):\n    return x\n"})
                             + "\n")
    if summary is not None:
        (directory / "summary.json").write_text(json.dumps(summary))


def _summary(name, block, steps, order, **changes):
    return {**er.FIXED, "arm": er.ARMS[name[0]], "train_block_size": block, "block_size": block, "num_steps": steps,
            "order": order, "step": 10000, "parse_rate": 1.0, "mean_tokens": 40.0, "mean_blocks": 3.0,
            "mean_passes_per_block": float(steps), "stops": {"eos": 0.9, "done": 0.1, "budget": 0.0},
            "seconds_per_block": 0.37, "seconds_per_pass": 0.09, "rows_per_batch": 5, **changes}


# Passing samples of 450 per run and configuration (exact, prefix, confidence) at each of the run's two cells.
PASSING = {"T16": {(16, 4): (60, 70, 50), (16, 2): (45, 40, 30)}, "R16": {(16, 4): (50, 40, 35), (16, 2): (25, 20, 15)},
           "T32": {(32, 4): (55, 50, 45), (32, 2): (50, 50, 35)}, "R32": {(32, 4): (35, 30, 20), (32, 2): (20, 15, 10)}}


def _write_run(runs, name, curves):
    run = runs / f"e0-{name}"
    for cell, counts in PASSING[name].items():
        for order, passing in zip(er.ORDERS, counts, strict=True):
            changes = {"exit_budget": 0.05} if (name, cell, order) == ("R32", (32, 2), "confidence") else {}
            _write_config(run / "eval" / "mbpp" / er.block_tag(*cell, order), passing,
                          _summary(name, cell[0], cell[1], order, **changes))
    (run / "config.json").write_text(json.dumps({"arm": er.ARMS[name[0]], "block_size": int(name[1:]),
                                                  "steps": 10000}))
    (run / "timing.json").write_text(json.dumps({"steps": 100, "steps_per_s": 0.18}))
    (run / "memory.json").write_text(json.dumps({"micro_batch": 8, "grad_accum": 4}))
    rows = [{"step": s, "train/loss": 1.0, "train/peak_gib": 20.5} for s in (100, 10000)]
    for target, values in curves.items():
        for step, value in zip((0, 2500, 5000, 7500, 10000), values, strict=True):
            rows.append({"step": step, f"eval/val_{target}_loss": value})
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


FALLING, RISING = (5.0, 3.0, 2.8, 2.7, 2.6), (5.0, 3.0, 2.8, 2.9, 3.0)


def test_the_report_fills_the_table_and_reads_the_rules(tmp_path, capsys):
    baseline, sweep, runs, out = tmp_path / "sweep" / "ar", tmp_path / "sweep", tmp_path / "runs", tmp_path / "report"
    _write_config(baseline, 107, {"pass@1": 107 / 450})                            # 23.8%
    _write_config(sweep / "realizer-K8", 2, {})
    data = tmp_path / "data"
    data.mkdir()
    set_summary = {"records": {"complete": 221, "write": 35}, "quota": {"complete": 221, "write": 35},
                   "kept_rate": {"complete": 0.65, "write": 0.12}, "source_examples_read": 1000,
                   "generation_hours": 0.2, "stops": {"complete/eos": 10}}
    (data / "summary.json").write_text(json.dumps({"validation": set_summary, "train": set_summary,
                                                   "named_function": {"share": 0.7}, "lengths": {}, "entropy": {}}))
    # After the first two runs (C12): H1, the B = 16 half of H2 and H3 can be read.
    _write_run(runs, "T16", {"rollouts": FALLING, "references": FALLING})
    _write_run(runs, "R16", {"references": RISING, "rollouts": RISING})
    args = ["--runs", str(runs), "--baseline", str(baseline), "--predecessor", str(sweep), "--data", str(data),
            "--out", str(out)]
    assert er.main(args) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["h1"]["status"] == "holds" and report["h1"]["rho"] == pytest.approx(60 / 107)
    assert report["h2"]["status"] == "pending" and report["h3"]["status"] == "holds"
    assert report["h2"]["repetition"]["16"]["status"] == "favours arm T"
    assert report["h2"]["repetition"]["32"]["status"] == "pending"
    assert report["section_5_4"]["applies"] is False and report["deviations"] == []
    # All four runs.
    _write_run(runs, "T32", {"rollouts": FALLING, "references": FALLING})
    _write_run(runs, "R32", {"references": FALLING, "rollouts": FALLING})
    capsys.readouterr()
    assert er.main(args) == 0
    text = capsys.readouterr().out
    report = json.loads((out / "report.json").read_text())
    rows = {(r["model"], r["cell"] and tuple(r["cell"])): r for r in report["rows"]}
    ar = rows[("Gemma 4 E2B, AR, released `generate`", None)]
    assert ar["exact"][0] == pytest.approx(107 / 450)
    t16 = rows[("Block-DFM, arm T (H1, H3)", (16, 4))]
    assert [t16[o][0] for o in er.ORDERS] == pytest.approx([60 / 450, 70 / 450, 50 / 450])
    assert t16["ratio"][0] == pytest.approx(60 / 107)
    k8 = rows[("Whole-canvas realizer, K = 8", None)]                     # from the sweep's samples
    assert k8["exact"][0] == pytest.approx(2 / 450) and k8["ratio"][0] == pytest.approx(2 / 107)
    assert k8["documented"] == {"prefix": 0.0, "confidence": 0.0}
    l2r = rows[("Whole-canvas realizer, left to right", None)]               # documented only
    assert l2r["prefix"][0] == 0.171 and l2r["ratio"][0] == pytest.approx(0.171 / (107 / 450)) and l2r["ratio"][1] is None
    h2 = report["h2"]                                                         # D = 10, 20, 20, 30 samples of 450
    assert [h2["differences"][c][0] for c in ("(16, 4)", "(16, 2)", "(32, 4)", "(32, 2)")] == pytest.approx(
        [10 / 450, 20 / 450, 20 / 450, 30 / 450])
    assert h2["d8"] == pytest.approx(20 / 450) and h2["grows"] is True and h2["status"] == "holds"
    assert h2["growth"][0] == pytest.approx(20 / 450)
    assert h2["repetition"]["32"]["status"] == "no sign"
    assert report["h3"]["cells"]["(32, 4)"]["at_least_as_good"] is False
    assert not any(c["at_floor"] for c in report["h3"]["cells"].values())
    assert report["deviations"] == ["R32 block-B32-K2-confidence: exit_budget is 0.05, not 0.0"]
    tag = report["runs"]["T16"]["configurations"]["block-B16-K4"]
    assert tag["outcomes"]["wrong"] == pytest.approx(1 - 60 / 450) and tag["rows_per_batch"] == 5
    assert "## H1, the bridge: " in text and "| Block-DFM, arm T (H1, H3) | AR rollouts | (16, 4) | 4 |" in text
    assert "the repetition favours arm T" in text and "kept rate" in text
    # The arms of a block size must start from the same held-out losses (the same untrained model).
    _write_run(runs, "R32", {"references": (5.1, 3.0, 2.8, 2.7, 2.6), "rollouts": FALLING})
    assert er.main(args) == 0
    deviations = json.loads((out / "report.json").read_text())["deviations"]
    assert "T32 and R32 start from different held-out losses on the references (5.0000 and 5.1000 at step 0): " \
           "the two arms should share the untrained model, the validation set and its states" in deviations
    # A configuration scored on other problems is refused.
    _write_config(runs / "e0-T16" / "eval" / "mbpp" / "block-B16-K2", 5, _summary("T16", 16, 2, "exact"))
    with open(runs / "e0-T16" / "eval" / "mbpp" / "block-B16-K2" / "samples.jsonl", "a") as handle:
        handle.write(json.dumps({"task_id": "MBPP/601", "sample": 0, "passed": False}) + "\n")
    with pytest.raises(SystemExit, match="other problems"):
        er.main(args)
