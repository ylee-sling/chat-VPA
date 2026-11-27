# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_readings.py -- Step 1's readings of A2 against A1 (docs/e1.md, Section 11.4), applied as registered to the decode
records of one measurement (S9): R1 and R2 for every drafter, R6 (acceptance) and R7 (speed) for A2 against A1,
and R9 (the keys Transformers hands its drafter against the accepted prefix's, Section 11.9), in every PyTorch
table; R1, R2 and R7 in the vLLM table behind S11's drift rule. It reads records and applies rules; it chooses
nothing.

    python -m chat_vpa.e1_readings best --table runs/e1-step1/measure/l4-fp32 --drafter a1
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

__all__ = ["best_length", "read_pytorch", "read_vllm", "render", "verdict"]

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
