# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_count.py -- tokens per cycle counted along the target's greedy continuations of the development prompts
(docs/e1.md, Sections 3.3, 6.1 and 11.5): what A3's oracle gate reads (A3-oracle against A3-0, a paired bootstrap
over the prompts), and the same counts for A1, A2 and A3-0 beside it. Nothing is timed.

    python -m chat_vpa.e1_count --model ASSETS/models/gemma-4-E2B-it --table runs/e1-step1/measure/l4-fp32 \\
        --prompts ASSETS/prompts/dev.jsonl --drafter a1=ASSETS/models/gemma-4-E2B-it-assistant \\
        --drafter a2=runs/e1-step1/a2/final --drafter a3-0=runs/e1-step1/a3-0/final --out runs/e1-step1/count

**The continuations** are the ``ar`` responses of a decode table (``e1_decode``): the target's own greedy decoding
in that table's runtime and precision, the reference of every arm there. Each prompt's chat ids are made again from
the prompt file the table names (its SHA-256 must match) and must have the table's length.

**The count.** The target runs once over each prompt and its continuation, in the table's precision unless
``--dtype`` says otherwise, and every drafter drafts at every position q of the continuation with the keys of a
cycle that starts at q (``e1_train.visibility``: the accepted prefix's keys, the kind R9 measures with the arms
``a1-4-clean`` and ``a2-4-clean``): Google's drafter and A2 unrolled on the accepted path over up to 8 depths
(``e1_train.a2_hidden``), an A3 realiser (a ``realiser.pt`` beside its drafter, ``e1_train.load_realiser``) in one
call over its slots. A draft is the drafter's greedy choice, the best token among its top clusters, as it drafts at
inference. A cycle at q accepts the leading drafts among its first G that equal the continuation and adds the next
token; the first token after the prompt comes from the prefill (``e1_train.tokens_per_cycle``). Under greedy
verification this is what decoding with those keys yields, up to where the one teacher-forced pass and the decode
loop's passes round differently.

**Reported** per drafter and G: tokens per cycle (a ratio of sums over prompts), the hit rate by depth, and the
acceptance profile (depth k's hits among the positions where every depth below it hit); per pair of drafters and
each G both have, the later one's tokens per cycle minus the earlier one's with its 95% paired bootstrap over
prompts (``e0_report.paired_bootstrap``, 10,000 resamples, seed 0), and which way it lies (``e1_readings.verdict``);
beside them, where the table has the same drafter's arm with the accepted prefix's keys at the same G, the tokens
per cycle that decoding measured there (described, not read). It applies no rule: the oracle gate (6.1) is read on
these counts once A3-oracle exists.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from chat_vpa.e1_common import chat_ids, environment, file_sha256, read_prompts, torch_dtype, write_json

__all__ = ["acceptance_profile", "continuations", "count_drafter", "compare", "gate", "render"]

LENGTHS = (2, 4, 6, 8)


def continuations(table: Path, prompts: Path, tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The ``ar`` responses of a decode table as rollout records (``e1_train.prepare``'s input), and the table's
    summary."""
    summary = json.loads((Path(table) / "summary.json").read_text(encoding="utf-8"))
    expected = summary.get("prompts_sha256")
    if expected is not None and expected != file_sha256(prompts):
        raise ValueError(f"{prompts} is not the prompt file of {table} (SHA-256 differs)")
    rows = {row["id"]: row for row in read_prompts(prompts)}
    records = []
    with open(Path(table) / "samples.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["arm"] != "ar":
                continue
            ids = chat_ids(tokenizer, rows[record["id"]]["prompt"])
            if len(ids) != int(record["prompt_tokens"]):
                raise ValueError(f"prompt {record['id']}: {len(ids)} chat ids, the table has {record['prompt_tokens']}")
            records.append({"id": record["id"], "prompt_ids": ids, "response_ids": [int(t) for t in record["tokens"]],
                            "stop_id": None})
    if not records:
        raise ValueError(f"{table} has no 'ar' responses")
    return records, summary


def acceptance_profile(hits: Sequence[Sequence[Sequence[bool]]], count: int) -> list[float | None]:
    """Per depth k (1-based), among the drafted positions where depth k has a token of the continuation to match and
    every depth below it hit, the share at which depth k hits too (None where no position qualifies). Along a
    continuation of P drafted positions, depth k at the i-th (from 0) has one if i + k <= P."""
    out: list[float | None] = []
    for k in range(count):
        reached = accepted = 0
        for sequence in hits:
            for i, depths in enumerate(sequence):
                if i + k < len(sequence) and len(depths) > k and all(depths[:k]):
                    reached += 1
                    accepted += bool(depths[k])
        out.append(accepted / reached if reached else None)
    return out


def _is_out_of_memory(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def count_drafter(target: Any, path: Path, dtype: str, device: torch.device, examples: Sequence[dict[str, Any]],
                  lengths: Sequence[int], batch_size: int, pad_id: int, vocab_chunk: int = 1_024) -> dict[str, Any]:
    """One drafter's counts along the examples (``e1_train.prepare``'s output); a batch that runs out of GPU memory
    is counted again in batches of half the size."""
    from transformers import Gemma4AssistantForCausalLM

    from chat_vpa.e1_train import choice_hits, cluster_map, load_realiser, tokens_per_cycle

    drafter = Gemma4AssistantForCausalLM.from_pretrained(str(path), dtype=torch_dtype(dtype)).to(device).eval()
    module = load_realiser(path, drafter).eval() if (Path(path) / "realiser.pt").exists() else None
    count = module.slots if module is not None else max(lengths)
    clusters = cluster_map(drafter)
    size = batch_size
    while True:
        try:
            hits, _, sums = choice_hits(target, drafter, module, count, examples, size, pad_id, device, vocab_chunk,
                                        clusters)
            break
        except Exception as error:                              # noqa: BLE001 - only out-of-memory is retried
            if not _is_out_of_memory(error) or size == 1:
                raise
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        size = max(1, size // 2)
    per_prompt = {g: [tokens_per_cycle(h, g) for h in hits] for g in lengths if g <= count}
    source = getattr(module, "source", "one vector") if module is not None else None
    out = {"path": str(path), "kind": "one pass" if module is not None else "sequential", "depths": count,
           "slot_input": None if module is None else module.slot_input, "code": source, "batch_size": size,
           "per_prompt": {str(g): [list(x) for x in v] for g, v in per_prompt.items()},
           "tokens_per_cycle": {str(g): sum(t for _, t in v) / sum(c for c, _ in v) for g, v in per_prompt.items()},
           "hit": [sums["clustered"][k] / sums["n"][k] if sums["n"][k] else None for k in range(count)],
           "profile": acceptance_profile(hits, count)}
    del drafter, module
    gc.collect()
    return out


def _difference(a: Any, b: Any, c: Any, d: Any) -> Any:
    return a / b - c / d


def compare(counts: dict[str, dict[str, Any]], resamples: int = 10_000, seed: int = 0) -> list[dict[str, Any]]:
    """Every later drafter against every earlier one, at each G both have: tokens per cycle, later minus earlier,
    with its paired bootstrap over prompts."""
    from chat_vpa.e0_report import paired_bootstrap
    from chat_vpa.e1_readings import verdict

    labels = list(counts)
    out = []
    for j, later in enumerate(labels):
        for earlier in labels[:j]:
            for g in sorted(set(counts[later]["per_prompt"]) & set(counts[earlier]["per_prompt"]), key=int):
                a, b = counts[later]["per_prompt"][g], counts[earlier]["per_prompt"][g]
                vectors = [[float(t) for _, t in a], [float(c) for c, _ in a], [float(t) for _, t in b],
                           [float(c) for c, _ in b]]
                point, low, high = paired_bootstrap(vectors, _difference, resamples, seed)
                out.append({"later": later, "earlier": earlier, "g": int(g),
                            "difference": {"point": point, "low": low, "high": high},
                            "reading": verdict(low, high, 0.0, f"{later} accepts more", f"{later} accepts fewer")})
    return out


def _ratio(a: Any, b: Any) -> Any:
    return a / b


def gate(counts: dict[str, dict[str, Any]], comparisons: Sequence[dict[str, Any]], oracle: str, control: str,
         length: int, bar: float | None = None, resamples: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """Section 6.1's oracle gate, read at G = ``length`` (A3's B): A3 is trained only if the oracle's tokens per
    cycle are clearly above the control's (the interval of the difference above 0). Beside it, the oracle's tokens
    per cycle with its interval against ``bar`` (tau* against A1 at B = ``length``): if the whole interval lies below
    it, no predictor can make A3 faster than A1 in that table."""
    from chat_vpa.e0_report import paired_bootstrap

    if oracle not in counts or control not in counts:
        raise ValueError(f"the gate needs counts of {oracle} and {control}")
    key = str(length)
    difference = next((x for x in comparisons if x["g"] == length and {x["later"], x["earlier"]} == {oracle, control}),
                      None)
    if difference is None:
        raise ValueError(f"no comparison of {oracle} and {control} at G = {length}")
    point = difference["difference"]
    if difference["later"] != oracle:                  # the comparison is control - oracle: turn it round
        point = {"point": -point["point"], "low": -point["high"], "high": -point["low"]}
    above = point["low"] > 0
    out: dict[str, Any] = {"oracle": oracle, "control": control, "g": length, "difference": point,
                           "reading": ("the oracle is clearly above the control: A3 may be trained" if above else
                                       "the oracle is not clearly above the control: A3 is not trained (6.1)")}
    pairs = counts[oracle]["per_prompt"][key]
    value, low, high = paired_bootstrap([[float(t) for _, t in pairs], [float(c) for c, _ in pairs]], _ratio,
                                        resamples, seed)
    out["oracle_tokens_per_cycle"] = {"point": value, "low": low, "high": high}
    if bar is not None:
        out["bar"] = bar
        out["bar_reading"] = ("the oracle lies below tau* against A1: no predictor can make A3 faster than A1 here"
                              if high < bar else "the oracle does not lie below tau* against A1")
    return out


def render(result: dict[str, Any]) -> str:
    """The counts as markdown; every number is *counted* along the table's continuations."""
    env = result["environment"]
    counts = result["drafters"]
    lengths = sorted({int(g) for c in counts.values() for g in c["tokens_per_cycle"]})
    lines = ["# Tokens per cycle counted along the development prompts' greedy continuations "
             "(docs/e1.md, Sections 6.1 and 11.5)", "",
             f"Continuations: the `ar` responses of `{result['table']}` ({result['prompts']} prompts, "
             f"{result['tokens']} tokens counted after the first of each). The target's pass in {env['dtype']} on "
             f"{env.get('gpu') or env['device']}; every draft with the keys of a cycle that starts at its position "
             "(the accepted prefix's). Counted, not timed: exact under greedy verification up to rounding.", "",
             "| drafter | kind | " + " | ".join(f"G = {g}" for g in lengths) + " | hit rate by depth |",
             "|---|---|" + "---|" * len(lengths) + "---|"]
    for label, c in counts.items():
        code = {"oracle": "the true block's code: conditions on the reference", "predicted": "the predicted code",
                "one vector": "one learned vector"}.get(c.get("code") or "", "")
        kind = (f"one pass, {c['depths']} slots ({c['slot_input']}; {code})" if c["kind"] == "one pass"
                else f"sequential, {c['depths']} depths")
        cells = " | ".join(f"{c['tokens_per_cycle'][str(g)]:.3f}" if str(g) in c["tokens_per_cycle"] else "--"
                           for g in lengths)
        hits = ", ".join("--" if h is None else f"{h:.3f}" for h in c["hit"])
        lines.append(f"| {label} | {kind} | {cells} | {hits} |")
    lines.append("")
    for label, c in counts.items():
        profile = ", ".join("--" if p is None else f"{p:.3f}" for p in c["profile"])
        lines.append(f"- {label}: acceptance profile (depth k's hits where every depth below it hit): {profile}")
    if result["comparisons"]:
        lines += ["", "| later - earlier | G | tokens per cycle | which way |", "|---|---|---|---|"]
        for x in result["comparisons"]:
            d = x["difference"]
            lines.append(f"| {x['later']} - {x['earlier']} | {x['g']} | {d['point']:+.3f} [{d['low']:+.3f}, "
                         f"{d['high']:+.3f}] | {x['reading']} |")
    if result.get("gate"):
        g = result["gate"]
        d, o = g["difference"], g["oracle_tokens_per_cycle"]
        lines += ["", f"**The oracle gate** (Section 6.1, at G = {g['g']}): {g['oracle']} - {g['control']} = "
                  f"{d['point']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}] tokens per cycle: {g['reading']}. "
                  f"{g['oracle']} alone: {o['point']:.3f} [{o['low']:.3f}, {o['high']:.3f}]"
                  + (f" against tau* = {g['bar']:.3f}: {g['bar_reading']}." if "bar" in g else ".")]
    if result["measured_beside"]:
        lines += ["", "Beside them (described, not read): tokens per cycle that decoding measured in the table with "
                  "the accepted prefix's keys, against the count at the same G:"]
        for x in result["measured_beside"]:
            lines.append(f"- {x['arm']}: measured {x['measured']:.3f}, counted {x['counted']:.3f}")
    return "\n".join(lines)


def _drafters(pairs: Sequence[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for pair in pairs:
        label, sep, path = pair.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"--drafter takes LABEL=PATH, not {pair!r}")
        if label in out:
            raise SystemExit(f"drafter {label} is named twice")
        out[label] = Path(path)
    if not out:
        raise SystemExit("name at least one --drafter")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E1 Step 1: tokens per cycle counted along the development prompts' "
                                "greedy continuations", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="the instruction-tuned target")
    p.add_argument("--table", type=Path, required=True, help="a decode table of e1_decode with an 'ar' arm")
    p.add_argument("--prompts", type=Path, required=True, help="the table's prompt file")
    p.add_argument("--drafter", action="append", default=[], metavar="LABEL=PATH", help="repeatable; in order")
    p.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    p.add_argument("--dtype", default=None, choices=["bf16", "fp32"], help="default: the table's")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gate", default=None, metavar="ORACLE:CONTROL", help="read the oracle gate (Section 6.1)")
    p.add_argument("--gate-length", type=int, default=8, help="the G at which the gate is read (A3's B)")
    p.add_argument("--breakeven", type=Path, default=None, help="breakeven.json: tau* against A1 beside the gate")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    drafters = _drafters(args.drafter)
    if min(args.lengths) < 1:
        raise SystemExit("--lengths must be positive")

    from chat_vpa.device import resolve_device
    from chat_vpa.e1_common import load_target
    from chat_vpa.e1_train import prepare

    device = resolve_device(args.device)
    summary = json.loads((args.table / "summary.json").read_text(encoding="utf-8"))
    dtype = args.dtype or summary["environment"]["dtype"]
    target, tokenizer = load_target(args.model, dtype, device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    try:
        records, summary = continuations(args.table, args.prompts, tokenizer)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    examples, dropped = prepare(records)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    counts = {}
    for label, path in drafters.items():
        counts[label] = count_drafter(target, path, dtype, device, examples, sorted(set(args.lengths)),
                                      args.batch_size, pad_id)
        print(f"  {label}: " + ", ".join(f"G={g} {v:.3f}" for g, v in counts[label]["tokens_per_cycle"].items()),
              file=sys.stderr, flush=True)
    beside = []
    for label, c in counts.items():
        for g, counted in c["tokens_per_cycle"].items():
            arm = summary["arms"].get(f"{label}-{g}-clean")
            if arm is not None and arm.get("tokens_per_cycle") is not None:
                beside.append({"arm": f"{label}-{g}-clean", "measured": arm["tokens_per_cycle"], "counted": counted})
    result = {"what": "tokens per cycle counted along the target's greedy continuations of the development prompts "
                      "(docs/e1.md, Sections 6.1 and 11.5); not timed",
              "table": str(args.table), "table_environment": summary["environment"],
              "prompts_file": str(args.prompts), "prompts_sha256": file_sha256(args.prompts),
              "model": str(args.model), "environment": environment(device, dtype), "prompts": len(examples),
              "dropped": dropped, "tokens": sum(len(e["ids"]) - e["start"] - 1 for e in examples),
              "drafters": counts, "comparisons": compare(counts), "measured_beside": beside}
    if args.gate:
        oracle, _, control = args.gate.partition(":")
        bar = None
        if args.breakeven is not None:
            bars = json.loads(args.breakeven.read_text(encoding="utf-8"))["bars"]
            gate_bar = next(b for b in bars if b["registered"])
            bar = next((x["tau_star"] for x in gate_bar["by_block"] if x["b"] == args.gate_length), None)
        result["gate"] = gate(counts, result["comparisons"], oracle, control, args.gate_length, bar)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "count.json", result)
    text = render(result)
    (args.out / "count.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
