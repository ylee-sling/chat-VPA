# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_readings.py -- Step 1's readings of A2 against A1 (docs/e1.md, Section 11.4), applied as registered to the decode
records of one measurement (S9): R1 and R2 for every drafter, R6 (acceptance) and R7 (speed) for A2 against A1,
and R9 (the keys Transformers hands its drafter against the accepted prefix's, Section 11.9), in every PyTorch
table; R1, R2 and R7 in the vLLM table behind S11's drift rule. It reads records and applies rules; it chooses
nothing.

    python -m chat_vpa.e1_readings best --table runs/e1-step1/measure/l4-fp32 --drafter a1
    python -m chat_vpa.e1_readings breakeven --table runs/e1-step1/measure/l4-fp32 \\
        --ceiling runs/e1-step0/ceiling/gpu-fp32 --against a1 a2 --out runs/e1-step1/breakeven
    python -m chat_vpa.e1_readings read --tables runs/e1-step1/measure/l4-fp32 runs/e1-step1/measure/l4-bf16 \\
        runs/e1-step1/measure/cpu-fp32 --vllm runs/e1-step1/measure/vllm --out runs/e1-step1/measure

A PyTorch table is a directory of ``e1_decode`` (``samples.jsonl``, ``summary.json``) whose arms are ``ar``,
``D-G`` (drafter D at draft length G through Transformers' own call) and ``D-G-clean`` (the accepted prefix's
keys). A vLLM table is a directory of ``e1_vllm`` whose arms are tagged ``off-1``, ``off-2`` and ``D-G``.

    best G      a drafter's arm with the highest decode speed-up over ``ar`` (point estimate) among its arms
                through Transformers' own call: the "best setting" of R2 and R7 (``best`` prints G)
    R1          the arm's tokens against ``ar``'s, prompt by prompt: exact if every prompt is identical
    R2          the drafter at its best G against ``ar``: pays clearly (interval above 1), costs clearly (below
                1), undecided
    R6          A2's tokens per cycle minus A1's at the same G, PyTorch tables only: A2 accepts more (interval
                above 0), fewer (below 0), undecided; for both kinds of keys at every G that both drafters have
    R7          A2's decode rate at its best G over A1's at its best G: A2 pays over A1 clearly, costs clearly,
                undecided
    R9          per drafter and G, tokens per cycle with the accepted prefix's keys minus with Transformers' own:
                more, fewer, undecided; with the ratio of the decode rates beside it (described, not read)
    S11         vLLM: R2 and R7 are read only if the interval of ``off-2``'s decode rate over ``off-1``'s lies
                inside [0.98, 1.02]

``breakeven`` fills in A3's break-even of Section 6.1 from one PyTorch table and the ceiling measured in the same
runtime and precision (Section 11.5: Step 1's own float32 rows): a cycle's seconds are its tokens per cycle over the
decode rate; the drafter's seconds per call are the least-squares slope of a cycle's seconds beyond the target's
pass over G + 1 positions against G, as Section 10.3, item 1, computed them in bfloat16; A3's drafting is 1.25
calls (Section 6.1's estimate until A3 exists); and for blocks of B,

    tau* = tau_r (T_r - D_r + D_3 + t(B + 1) - t(G_r + 1)) / T_r

against a reference drafter r at its best G_r (R2's rule): the first named is the registered gate (A1), the
others are bars beside it. Where tau* > B + 1, no drafter of A3's kind can pay at B (the gate of 6.1). Everything
there is *predicted* from *measured* costs.

Tokens per cycle is ``e1_decode``'s: the tokens after the first pass over the target passes after the first, as a
ratio of sums over prompts. Intervals are 95% paired bootstraps over prompts (``e0_report.paired_bootstrap``,
10,000 resamples, seed 0) of statistics of per-prompt sums. Everything here is *measured* in the table's runtime,
device and precision, and says which.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["best_length", "break_even", "line_fit", "pass_seconds", "read_pytorch", "read_vllm", "render",
           "render_break_even", "verdict"]

DRIFT = (0.98, 1.02)
_DRAFT = re.compile(r"^([a-z][a-z0-9]*)-([1-9]\d*)(-clean)?$")


def verdict(low: float, high: float, at: float, above: str, below: str) -> str:
    """'clearly' as in docs/e1.md, Section 4: the interval lies above (or below) ``at``."""
    if low > at:
        return above
    if high < at:
        return below
    return "undecided"


def _arms(names: Sequence[str]) -> dict[tuple[str, int, bool], str]:
    out: dict[tuple[str, int, bool], str] = {}
    for name in names:
        match = _DRAFT.match(name)
        if match and match.group(1) not in ("ar", "lookup", "off"):
            out[(match.group(1), int(match.group(2)), bool(match.group(3)))] = name
    return out


def _columns(records: dict[str, dict[str, Any]], ids: Sequence[str]) -> dict[str, list[float]]:
    """Per prompt, as ``e1_decode.summarize`` counts them."""
    rows = [records[i] for i in ids]
    return {"decode_tokens": [float(len(r["tokens"]) - r["first_tokens"]) for r in rows],
            "decode_seconds": [float(r["seconds"] - r["first_seconds"]) for r in rows],
            "cycles": [float(r["target_passes"] - 1) for r in rows]}


def _load_pytorch(table: Path) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]], list[str]]:
    summary = json.loads((table / "summary.json").read_text(encoding="utf-8"))
    by_arm: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in summary["arms"]}
    with open(table / "samples.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                by_arm[record["arm"]][record["id"]] = record
    ids = list(by_arm["ar"])
    for name, records in by_arm.items():
        if list(records) != ids:
            raise ValueError(f"{table}: arm {name} does not cover ar's prompts in the same order")
    return summary, by_arm, ids


def best_length(arms: dict[str, dict[str, Any]], label: str, speedup: str = "decode_speedup") -> int | None:
    """The G of ``label``'s arm with the highest point estimate of ``speedup`` among its arms through
    Transformers' own call (vLLM: its arms), or None if it has none."""
    candidates = [(s[speedup], g) for (name, g, clean), arm in _arms(list(arms)).items()
                  if name == label and not clean and (s := arms[arm]).get(speedup) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c[0], -c[1]))[1]


def _interval(vectors: Sequence[Sequence[float]], statistic: Any) -> dict[str, float]:
    from chat_vpa.e0_report import paired_bootstrap

    point, low, high = paired_bootstrap(vectors, statistic)
    return {"point": point, "low": low, "high": high}


def _difference_of_ratios(a: Any, b: Any, c: Any, d: Any) -> Any:
    return a / b - c / d


def _ratio_of_ratios(a: Any, b: Any, c: Any, d: Any) -> Any:
    return (a / b) / (c / d)


def read_pytorch(table: Path, first: str = "a1", second: str = "a2") -> dict[str, Any]:
    """R1, R2, R6, R7 and R9 for one PyTorch table."""
    summary, by_arm, ids = _load_pytorch(table)
    arms = summary["arms"]
    drafts = _arms(list(arms))
    columns = {name: _columns(by_arm[name], ids) for name in drafts.values()}
    out: dict[str, Any] = {"table": str(table), "environment": summary["environment"], "prompts": len(ids),
                           "max_new_tokens": summary["max_new_tokens"], "r1": {}, "r2": {}, "r6": [], "r9": []}
    for _, name in sorted(drafts.items()):
        out["r1"][name] = {"exact": arms[name]["exact"], "of": len(ids),
                           "reading": "exact" if arms[name]["exact"] == len(ids) else "not token-identical"}
    best: dict[str, int | None] = {}
    for label in sorted({k[0] for k in drafts}):
        g = best_length(arms, label)
        best[label] = g
        if g is None:
            continue
        s = arms[drafts[(label, g, False)]]
        low, high = s["decode_speedup_interval"]
        out["r2"][label] = {"best_g": g, "speedup": s["decode_speedup"], "low": low, "high": high,
                            "reading": verdict(low, high, 1.0, "pays clearly", "costs clearly")}
    for clean in (False, True):
        for g in sorted({k[1] for k in drafts if k[0] == first and k[2] == clean}):
            a, b = drafts.get((second, g, clean)), drafts.get((first, g, clean))
            if a is None or b is None:
                continue
            ci = _interval([columns[a]["decode_tokens"], columns[a]["cycles"], columns[b]["decode_tokens"],
                            columns[b]["cycles"]], _difference_of_ratios)
            out["r6"].append({"g": g, "keys": "accepted prefix" if clean else "Transformers'", "second": a,
                              "first": b, "difference": ci, "reading": verdict(
                                  ci["low"], ci["high"], 0.0, f"{second} accepts more", f"{second} accepts fewer")})
    if best.get(first) is not None and best.get(second) is not None:
        a, b = drafts[(second, best[second], False)], drafts[(first, best[first], False)]
        ci = _interval([columns[a]["decode_tokens"], columns[a]["decode_seconds"], columns[b]["decode_tokens"],
                        columns[b]["decode_seconds"]], _ratio_of_ratios)
        out["r7"] = {"second": a, "first": b, "ratio": ci,
                     "reading": verdict(ci["low"], ci["high"], 1.0, f"{second} pays over {first} clearly",
                                        f"{second} costs against {first} clearly")}
    for (label, g, clean), name in sorted(drafts.items()):
        plain = drafts.get((label, g, False))
        if not clean or plain is None:
            continue
        tpc = _interval([columns[name]["decode_tokens"], columns[name]["cycles"], columns[plain]["decode_tokens"],
                         columns[plain]["cycles"]], _difference_of_ratios)
        rate = _interval([columns[name]["decode_tokens"], columns[name]["decode_seconds"],
                          columns[plain]["decode_tokens"], columns[plain]["decode_seconds"]], _ratio_of_ratios)
        out["r9"].append({"drafter": label, "g": g, "clean": name, "plain": plain, "tokens_per_cycle": tpc,
                          "rate_ratio": rate, "reading": verdict(tpc["low"], tpc["high"], 0.0,
                                                                 "the accepted prefix's keys accept more",
                                                                 "the accepted prefix's keys accept fewer")})
    out["tokens_per_cycle"] = {name: arms[name]["tokens_per_cycle"] for name in drafts.values()}
    return out


def read_vllm(table: Path, first: str = "a1", second: str = "a2") -> dict[str, Any]:
    """S11's drift rule, then R1, R2 and R7, for one vLLM table."""
    summary = json.loads((table / "summary.json").read_text(encoding="utf-8"))
    arms = summary["arms"]
    out: dict[str, Any] = {"table": str(table), "environment": summary["environment"], "r1": {}, "r2": {}}
    if "off-2" not in arms:
        out["drift"] = {"reading": "no off-2: R2 and R7 are not read"}
        return out
    low, high = arms["off-2"]["decode_speedup_interval"]
    inside = DRIFT[0] <= low and high <= DRIFT[1]
    out["drift"] = {"ratio": arms["off-2"]["decode_speedup"], "low": low, "high": high, "read": inside,
                    "reading": "inside [0.98, 1.02]: R2 and R7 are read" if inside
                    else "not inside [0.98, 1.02]: R2 and R7 are not read (S11)"}
    drafts = _arms([t for t in arms if not t.startswith("off-")])
    for name in sorted(drafts.values()):
        out["r1"][name] = {"same_tokens": arms[name]["same_tokens"], "of": arms[name]["prompts"]}
    best = {label: best_length(arms, label) for label in sorted({k[0] for k in drafts})}
    for label, g in best.items():
        if g is None:
            continue
        s = arms[drafts[(label, g, False)]]
        lo, hi = s["decode_speedup_interval"]
        out["r2"][label] = {"best_g": g, "speedup": s["decode_speedup"], "low": lo, "high": hi,
                            "reading": verdict(lo, hi, 1.0, "pays clearly", "costs clearly") if inside
                            else "not read (S11)"}
    if inside and best.get(first) is not None and best.get(second) is not None:
        records = {}
        for tag in (drafts[(second, best[second], False)], drafts[(first, best[first], False)]):
            with open(table / "arms" / tag / "samples.jsonl", encoding="utf-8") as handle:
                records[tag] = [json.loads(line) for line in handle if line.strip()]
        a, b = drafts[(second, best[second], False)], drafts[(first, best[first], False)]
        if [r["id"] for r in records[a]] != [r["id"] for r in records[b]]:
            raise ValueError(f"{table}: {a} and {b} cover different prompts")

        def cols(tag: str) -> tuple[list[float], list[float]]:       # as e1_vllm.summarize counts them
            return ([float(max(len(r["tokens"]) - 1, 0)) for r in records[tag]],
                    [float(r["decode_seconds"]) for r in records[tag]])

        ci = _interval([*cols(a), *cols(b)], _ratio_of_ratios)
        out["r7"] = {"second": a, "first": b, "ratio": ci,
                     "reading": verdict(ci["low"], ci["high"], 1.0, f"{second} pays over {first} clearly",
                                        f"{second} costs against {first} clearly")}
    out["tokens_per_cycle"] = {name: arms[name]["tokens_per_cycle"] for name in drafts.values()}
    return out


# ---------------------------------------------------------------------------------------------
# A3's break-even (docs/e1.md, Sections 6.1, 10.3 and 11.5)
# ---------------------------------------------------------------------------------------------

def pass_seconds(rows: Sequence[dict[str, Any]], n: int) -> tuple[float, bool]:
    """t(n) from the ceiling's rows (``e1_ceiling.ceiling_table``): measured, or on the straight line between the
    nearest measured lengths below and above n, as Section 10.3 interpolated t(7) and t(9); the flag says it is
    interpolated. Nothing is extrapolated."""
    measured = {int(r["n"]): float(r["seconds"]) for r in rows}
    if n in measured:
        return measured[n], False
    below, above = [m for m in measured if m < n], [m for m in measured if m > n]
    if not below or not above:
        raise ValueError(f"t({n}) is neither measured nor between two measured lengths")
    lo, hi = max(below), min(above)
    return measured[lo] + (measured[hi] - measured[lo]) * (n - lo) / (hi - lo), True


def line_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, list[float]]:
    """The least-squares line y = c + d x: (c, d, the residuals y - c - d x)."""
    if len(xs) != len(ys) or len(set(xs)) < 2:
        raise ValueError("a line needs at least two distinct x")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    d = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum((x - mx) ** 2 for x in xs)
    c = my - d * mx
    return c, d, [y - c - d * x for x, y in zip(xs, ys, strict=True)]


def break_even(table: Path, ceiling: Path, references: Sequence[str] = ("a1", "a2"),
               blocks: Sequence[int] = (2, 4, 6, 8), calls: float = 1.25) -> dict[str, Any]:
    """A3's break-even against each reference drafter of a PyTorch table (the module docstring)."""
    summary = json.loads((Path(table) / "summary.json").read_text(encoding="utf-8"))
    measured = json.loads((Path(ceiling) / "summary.json").read_text(encoding="utf-8"))
    env, cenv = summary["environment"], measured["environment"]
    for key in ("device", "dtype", "gpu", "cpu_threads"):
        if env.get(key) != cenv.get(key) and not (key == "cpu_threads" and env.get("device") != "cpu"):
            raise ValueError(f"the table and the ceiling were measured with different {key}: {env.get(key)} and "
                             f"{cenv.get(key)} (a cost is never carried from one runtime or precision to another)")
    arms, rows = summary["arms"], measured["rows"]
    drafts = _arms(list(arms))
    out: dict[str, Any] = {
        "what": "A3's break-even (docs/e1.md, Sections 6.1 and 11.5), predicted from measured costs",
        "table": str(table), "ceiling": str(ceiling), "environment": env, "ceiling_environment": cenv,
        "prompts": arms["ar"]["prompts"], "max_new_tokens": summary.get("max_new_tokens"), "calls": calls,
        "ar": {"decode_tokens_per_second": arms["ar"]["decode_tokens_per_second"],
               "ms_per_token": 1e3 / arms["ar"]["decode_tokens_per_second"]},
        "t1_ms": pass_seconds(rows, 1)[0] * 1e3, "arms": [], "fits": {}, "bars": []}
    for label in references:
        plain = sorted((g, name) for (lab, g, clean), name in drafts.items() if lab == label and not clean)
        if len({g for g, _ in plain}) < 2:
            raise ValueError(f"{table}: {label} needs arms at two draft lengths or more for its call time")
        xs, ys = [], []
        for g, name in plain:
            s = arms[name]
            cycle = s["tokens_per_cycle"] / s["decode_tokens_per_second"]
            t, interpolated = pass_seconds(rows, g + 1)
            out["arms"].append({"arm": name, "drafter": label, "g": g, "decode_tokens_per_second":
                                s["decode_tokens_per_second"], "decode_speedup": s.get("decode_speedup"),
                                "tokens_per_cycle": s["tokens_per_cycle"], "cycle_ms": cycle * 1e3,
                                "pass_ms": t * 1e3, "pass_interpolated": interpolated, "rest_ms": (cycle - t) * 1e3})
            xs.append(float(g))
            ys.append((cycle - t) * 1e3)
        c, d, residuals = line_fit(xs, ys)
        out["fits"][label] = {"call_ms": d, "intercept_ms": c, "residuals_ms": residuals,
                              "call_over_t1": d / out["t1_ms"]}
        best = best_length(arms, label)
        if best is None:
            raise ValueError(f"{table}: no speed-up for {label}")
        s = arms[drafts[(label, best, False)]]
        tau, cycle = s["tokens_per_cycle"], s["tokens_per_cycle"] / s["decode_tokens_per_second"] * 1e3
        own, a3 = best * d, calls * d
        t_ref = pass_seconds(rows, best + 1)[0] * 1e3
        by_block = []
        for b in blocks:
            t_b, interpolated = pass_seconds(rows, b + 1)
            bar = tau * (cycle - own + a3 + t_b * 1e3 - t_ref) / cycle
            by_block.append({"b": b, "tau_star": bar, "pass_ms": t_b * 1e3, "pass_interpolated": interpolated,
                             "share_of_tau": bar / tau, "pays_at_all": bar <= b + 1})
        out["bars"].append({"against": label, "registered": label == references[0], "g": best, "tau": tau,
                            "cycle_ms": cycle, "drafter_ms": own, "a3_drafting_ms": a3, "pass_ms": t_ref,
                            "by_block": by_block})
    gate = out["bars"][0]
    blocked = [x["b"] for x in gate["by_block"] if not x["pays_at_all"]]
    out["gate"] = {"against": gate["against"], "blocked": blocked,
                   "reading": "tau* <= B + 1 at every B: the gate does not stop A3" if not blocked else
                   f"tau* > B + 1 at B = {', '.join(map(str, blocked))}: no drafter of A3's kind can pay there"}
    return out


def render_break_even(result: dict[str, Any]) -> str:
    """The break-even as markdown; every number says what it is."""
    env = result["environment"]
    lines = ["# A3's break-even (docs/e1.md, Sections 6.1 and 11.5)", "",
             f"Decode table `{result['table']}` ({_where(env)}; {result['prompts']} prompts x "
             f"{result['max_new_tokens']} tokens; *measured*) and ceiling `{result['ceiling']}` "
             f"({_where(result['ceiling_environment'])}; *measured*). A cycle's ms are its tokens per cycle over the "
             "decode rate (*computed*); t(n) is the ceiling's pass over n positions (*measured*, or interpolated "
             "between measured lengths where marked *i*); a drafter call's ms is the least-squares slope of the rest "
             "of a cycle against G, as Section 10.3, item 1, computed it; A3's drafting is "
             f"{result['calls']:g} calls (Section 6.1's estimate). The bars are *predicted*.", "",
             "| arm | decode tok/s | speed-up | tokens per cycle | ms per cycle | t(G + 1), ms | rest, ms |",
             "|---|---|---|---|---|---|---|",
             f"| ar | {result['ar']['decode_tokens_per_second']:.2f} | 1 | 1.00 | {result['ar']['ms_per_token']:.2f} "
             f"(per token) | t(1) = {result['t1_ms']:.2f} | |"]
    for a in result["arms"]:
        speed = "--" if a["decode_speedup"] is None else f"{a['decode_speedup']:.3f}"
        mark = " *i*" if a["pass_interpolated"] else ""
        lines.append(f"| {a['arm']} | {a['decode_tokens_per_second']:.2f} | {speed} | {a['tokens_per_cycle']:.3f} | "
                     f"{a['cycle_ms']:.2f} | {a['pass_ms']:.2f}{mark} | {a['rest_ms']:.2f} |")
    lines.append("")
    for label, f in result["fits"].items():
        residuals = ", ".join(f"{r:+.2f}" for r in f["residuals_ms"])
        lines.append(f"- {label}: the rest of a cycle is {f['intercept_ms']:.2f} ms + {f['call_ms']:.3f} ms x G "
                     f"(residuals {residuals} ms): a call costs {f['call_over_t1']:.3f} of a target pass t(1)")
    lines += ["", "| against | G | tau | ms per cycle | its drafter, ms | A3's drafting, ms | "
              + " | ".join(f"tau* at B = {x['b']}" for x in result["bars"][0]["by_block"]) + " |",
              "|---|---|---|---|---|---|" + "---|" * len(result["bars"][0]["by_block"])]
    for bar in result["bars"]:
        name = f"{bar['against']} (the registered gate)" if bar["registered"] else bar["against"]
        cells = " | ".join(f"{x['tau_star']:.3f} ({100 * x['share_of_tau']:.1f}%)" for x in bar["by_block"])
        lines.append(f"| {name} | {bar['g']} | {bar['tau']:.3f} | {bar['cycle_ms']:.2f} | {bar['drafter_ms']:.2f} | "
                     f"{bar['a3_drafting_ms']:.2f} | {cells} |")
    lines += ["", "A3 with blocks of B is faster than the reference if its tokens per cycle exceed tau* (in brackets: "
              "tau* as a share of the reference's tokens per cycle).", "",
              f"Gate (Section 6.1), against {result['gate']['against']}: {result['gate']['reading']}."]
    return "\n".join(lines)


def _where(environment: dict[str, Any]) -> str:
    device = environment.get("gpu") or environment.get("device", "?")
    threads = environment.get("cpu_threads") if environment.get("device") == "cpu" else None
    return (f"{environment.get('runtime', 'PyTorch')}, {environment.get('dtype', '')}, {device}"
            + (f", {threads} threads" if threads else ""))


def _ci(ci: dict[str, float], digits: int = 3) -> str:
    return f"{ci['point']:.{digits}f} [{ci['low']:.{digits}f}, {ci['high']:.{digits}f}]"


def render(readings: Sequence[dict[str, Any]]) -> str:
    """The readings as markdown, every number *measured* in the table it names."""
    lines = ["# E1 Step 1: A2 against A1, read by R1, R2, R6, R7, R9 and S11 (docs/e1.md, Section 11.4)", "",
             "Every number is *measured* by the run of the table it stands under; intervals are 95% paired "
             "bootstraps over prompts.", ""]
    for r in readings:
        lines += [f"## {r['table']}", "", f"Where: {_where(r['environment'])}"
                  + (f"; {r['prompts']} prompts x {r['max_new_tokens']} tokens" if "prompts" in r else ""), ""]
        if "drift" in r:
            d = r["drift"]
            lines.append(f"- S11, the drift: off-2 / off-1 = {d.get('ratio', float('nan')):.3f} "
                         f"[{d.get('low', float('nan')):.3f}, {d.get('high', float('nan')):.3f}]: {d['reading']}")
        for name, x in r["r1"].items():
            if "exact" in x:
                lines.append(f"- R1, {name}: {x['exact']} of {x['of']} prompts token-identical to ar: {x['reading']}")
            else:
                lines.append(f"- R1, {name}: {x['same_tokens']} of {x['of']} prompts token-identical to off-1")
        for label, x in r["r2"].items():
            lines.append(f"- R2, {label} at its best G = {x['best_g']}: decode speed-up {x['speedup']:.3f} "
                         f"[{x['low']:.3f}, {x['high']:.3f}]: {x['reading']}")
        for x in r.get("r6", []):
            lines.append(f"- R6, G = {x['g']}, {x['keys']} keys: tokens per cycle {x['second']} - {x['first']} = "
                         f"{_ci(x['difference'])}: {x['reading']}")
        if "r7" in r:
            x = r["r7"]
            lines.append(f"- R7: decode rate {x['second']} / {x['first']} = {_ci(x['ratio'])}: {x['reading']}")
        for x in r.get("r9", []):
            lines.append(f"- R9, {x['drafter']} at G = {x['g']}: tokens per cycle {x['clean']} - {x['plain']} = "
                         f"{_ci(x['tokens_per_cycle'])}: {x['reading']}; decode rate ratio "
                         f"{_ci(x['rate_ratio'])} (described)")
        cycles = ", ".join(f"{k} {v:.2f}" for k, v in r["tokens_per_cycle"].items() if v is not None)
        lines += [f"- tokens per cycle: {cycles}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E1 Step 1: A2 against A1, read by the registered rules", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("best", help="print a drafter's best G in a PyTorch table")
    b.add_argument("--table", type=Path, required=True)
    b.add_argument("--drafter", required=True)
    e = sub.add_parser("breakeven", help="A3's break-even from a PyTorch table and its ceiling")
    e.add_argument("--table", type=Path, required=True)
    e.add_argument("--ceiling", type=Path, required=True, help="e1_ceiling's run in the same runtime and precision")
    e.add_argument("--against", nargs="+", default=["a1", "a2"], help="reference drafters; the first is the gate's")
    e.add_argument("--blocks", type=int, nargs="+", default=[2, 4, 6, 8])
    e.add_argument("--calls", type=float, default=1.25, help="A3's drafting in calls of the reference's drafter")
    e.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("read", help="read the tables and write readings.json and readings.md")
    r.add_argument("--tables", type=Path, nargs="*", default=[])
    r.add_argument("--vllm", type=Path, default=None)
    r.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == "best":
        summary = json.loads((args.table / "summary.json").read_text(encoding="utf-8"))
        g = best_length(summary["arms"], args.drafter)
        if g is None:
            raise SystemExit(f"{args.table} has no arm of {args.drafter}")
        print(g)
        return 0
    if args.command == "breakeven":
        try:
            result = break_even(args.table, args.ceiling, args.against, args.blocks, args.calls)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "breakeven.json", "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        text = render_break_even(result)
        (args.out / "breakeven.md").write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    readings = [read_pytorch(t) for t in args.tables if (t / "summary.json").exists()]
    if args.vllm is not None and (args.vllm / "summary.json").exists():
        readings.append(read_vllm(args.vllm))
    if not readings:
        raise SystemExit("no finished table to read")
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "readings.json", "w", encoding="utf-8") as handle:
        json.dump(readings, handle, indent=2)
    text = render(readings)
    (args.out / "readings.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
