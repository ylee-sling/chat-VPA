# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_bridge.py -- the E0 bridge of E1's Step 0 (docs/e1.md, Section 3.3): an E0 checkpoint as a block drafter whose
blocks the frozen backbone verifies greedily.

    python -m chat_vpa.e1_bridge --model MODEL --checkpoint runs/e0-T16/last.pt --eval-dir DATA/eval \\
        --out runs/e1-step0/bridge/T16

E0 asked whether a Block-DFM realizer on the frozen Gemma 4 E2B can decode alone, and it cannot (docs/e0.md,
Section 10). Orthrus and Uno use the same construction, a trainable diffusion view that reads the frozen
model's KV cache, as a *drafter*: the frozen model checks the block, the tokens up to the first disagreement
are kept, and the output is the frozen model's own. This module measures what E0's checkpoints are worth in
that role, without training anything.

The measurement. For each MBPP validation problem (E0's evaluation prompt):

1. y is the frozen backbone's greedy continuation, by the released model's own ``generate`` (``rollouts.
   generate_rollouts`` at temperature 0): the tokens before its <eos>, or up to the truncation rule's stop, or
   the budget of 256. This is what exact greedy verification would output, whatever the drafter proposes.
2. Speculative decoding along y: at position s the drafter drafts B tokens after the prefix (prompt, y[:s]),
   with K passes of the block sampler at temperature 0. a is the number of leading draft tokens equal to
   y[s:s + a]; the cycle yields a + 1 tokens (the verifying pass also gives the frozen model's own next
   token), capped at the end of y; s moves on by that many. A cycle costs K draft passes and 1 verifying pass
   of the backbone, so

       tokens per backbone pass = tokens / (cycles * (K + 1)),      autoregressive decoding = 1.

   Comparing with y stands for the verifying pass: under greedy verification the accepted text never leaves y,
   so the drafter is only ever asked about prefixes of y.
3. The same drafts at the block boundaries of training, s = 0, B, 2B, ...: accepted tokens per block and, per
   slot of the block, how often the draft equals y (the drafter in the states it was trained in).

``--check N`` first runs the committed stream itself as the drafter on N problems (a causal window, adapters off,
B = K = 1: the backbone's own next token through this repository's code path). In exact arithmetic every such
draft is accepted; a rejection is a numerical difference between ``generate`` and the realizer's path in this
precision. ``--dtype bf16`` (the default) is the precision E0 was trained and evaluated in, bfloat16 weights under
autocast; ``--dtype fp32`` runs both paths in float32, where the check accepts every draft on the tiny model of
the tests while bfloat16 does not.

What this is not: a speed measurement (passes are counted, not timed; whether a pass over a block costs what a
pass over one token costs is the ceiling measurement's question), and not E1's domain (code prompts, the base
checkpoint, 90 problems, one greedy trajectory each). K = 1 uses a step (t = 0, h = 1) that E0 did not train.
Intervals: 95% paired bootstrap over problems, 10,000 resamples, seed 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from chat_vpa.e0_report import paired_bootstrap
from chat_vpa.e1_common import DTYPES, torch_dtype, write_json

__all__ = ["Trace", "accepted_prefix", "aligned", "block_drafter", "simulate", "summarize"]

Drafter = Callable[[list[list[int]]], list[list[int]]]


def accepted_prefix(draft: Sequence[int], target: Sequence[int]) -> int:
    """Leading tokens of ``draft`` that equal ``target``'s."""
    count = 0
    for a, b in zip(draft, target, strict=False):
        if a != b:
            break
        count += 1
    return count


@dataclass
class Trace:
    """One problem's speculative decoding along its greedy continuation."""

    accepted: list[int] = field(default_factory=list)      # per cycle, the draft tokens kept
    gained: list[int] = field(default_factory=list)        # per cycle, the tokens it yielded (kept + 1, capped)

    @property
    def tokens(self) -> int:
        return sum(self.gained)

    @property
    def cycles(self) -> int:
        return len(self.gained)


def _batches(indices: list[int], size: int) -> list[list[int]]:
    return [indices[i:i + size] for i in range(0, len(indices), size)]


def simulate(prompts: Sequence[Sequence[int]], targets: Sequence[Sequence[int]], draft: Drafter, batch_size: int
             ) -> list[Trace]:
    """Speculative decoding of every problem along its target (step 2 of the module docstring).

    Args:
        prompts, targets: per problem the prompt ids and the greedy continuation y.
        draft: prefixes -> one draft (B tokens) per prefix.
    """
    position = [0] * len(prompts)
    traces = [Trace() for _ in prompts]
    while True:
        active = [i for i, y in enumerate(targets) if position[i] < len(y)]
        if not active:
            return traces
        for group in _batches(active, batch_size):
            drafts = draft([[*prompts[i], *targets[i][:position[i]]] for i in group])
            for i, tokens in zip(group, drafts, strict=True):
                rest = targets[i][position[i]:]
                kept = accepted_prefix(tokens, rest)
                gained = min(kept + 1, len(rest))
                traces[i].accepted.append(kept)
                traces[i].gained.append(gained)
                position[i] += gained


def aligned(prompts: Sequence[Sequence[int]], targets: Sequence[Sequence[int]], draft: Drafter, block_size: int,
            batch_size: int) -> tuple[list[list[int]], list[int], list[int]]:
    """Drafts at the block boundaries s = 0, B, 2B, ... of each target (step 3 of the module docstring).

    Returns per problem the accepted tokens of each block, and per slot of the block the number of drafts that
    equal the target there and the number of drafts that have a target token there.
    """
    jobs = [(i, s) for i, y in enumerate(targets) for s in range(0, len(y), block_size)]
    accepted: list[list[int]] = [[] for _ in prompts]
    hits, counts = [0] * block_size, [0] * block_size
    for group in _batches(list(range(len(jobs))), batch_size):
        drafts = draft([[*prompts[jobs[j][0]], *targets[jobs[j][0]][:jobs[j][1]]] for j in group])
        for j, tokens in zip(group, drafts, strict=True):
            i, s = jobs[j]
            rest = targets[i][s:s + block_size]
            accepted[i].append(accepted_prefix(tokens, rest))
            for slot, (a, b) in enumerate(zip(tokens, rest, strict=False)):
                counts[slot] += 1
                hits[slot] += int(a == b)
    return accepted, hits, counts


def summarize(traces: Sequence[Trace], num_steps: int, block_size: int, resamples: int = 10_000, seed: int = 0
              ) -> dict[str, Any]:
    """What a set of traces says (the quantities of the module docstring)."""
    cycles = sum(t.cycles for t in traces)
    tokens = sum(t.tokens for t in traces)
    if not cycles:
        raise ValueError("no cycle: every target is empty")
    accepted = [a for t in traces for a in t.accepted]
    histogram = [0] * (block_size + 1)
    for a in accepted:
        histogram[a] += 1
    passes = [float(t.cycles * (num_steps + 1)) for t in traces if t.cycles]
    gained = [float(t.tokens) for t in traces if t.cycles]
    point, low, high = paired_bootstrap([gained, passes], lambda a, b: a / b, resamples, seed)
    return {"problems": len(traces), "tokens": tokens, "cycles": cycles,
            "passes_per_cycle": num_steps + 1,
            "accepted_per_cycle": sum(accepted) / cycles,
            "tokens_per_cycle": tokens / cycles,
            "tokens_per_backbone_pass": point, "tokens_per_backbone_pass_interval": [low, high],
            "cycles_with_nothing_accepted": histogram[0] / cycles,
            "cycles_with_the_whole_block_accepted": histogram[block_size] / cycles,
            "accepted_histogram": histogram}


def block_drafter(model: Any, cfg: Any, forbidden: Sequence[int], device: torch.device, causal: bool = False,
                  adapters: bool = True, seed: int = 0, precise: bool = False) -> Drafter:
    """The block sampler as a drafter: one block of ``cfg.block_size`` tokens after each prefix. ``precise`` runs
    in the weights' own precision (float32 weights: float32); otherwise under E0's bfloat16 autocast."""
    from chat_vpa.block_sampler import sample_blocks
    from chat_vpa.evaluate import autocast

    generator = torch.Generator(device=device).manual_seed(seed)

    def precision() -> Any:
        return torch.autocast(device_type=device.type, enabled=False) if precise else autocast(device)

    @torch.no_grad()
    def draft(prefixes: list[list[int]]) -> list[list[int]]:
        with precision():
            cache = model.prefill(prefixes, device)
            anchor = torch.tensor([p[-1] for p in prefixes], dtype=torch.long, device=device)
            result = sample_blocks(model, cache, anchor, cfg, 1, forbidden, generator, stop_id=None,
                                   causal=causal, adapters=adapters)
        return [list(row) for row in result.tokens]

    return draft


def _table(rows: Sequence[dict[str, Any]]) -> str:
    lines = ["| drafter | K | order | accepted per cycle | tokens per cycle | tokens per backbone pass | "
             "nothing accepted | whole block | at training's boundaries: accepted per block |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["speculative"]
        low, high = s["tokens_per_backbone_pass_interval"]
        lines.append(f"| {r['drafter']} | {r['num_steps']} | {r['order']} | {s['accepted_per_cycle']:.2f} | "
                     f"{s['tokens_per_cycle']:.2f} | {s['tokens_per_backbone_pass']:.3f} [{low:.3f}, {high:.3f}] | "
                     f"{100 * s['cycles_with_nothing_accepted']:.1f}% | "
                     f"{100 * s['cycles_with_the_whole_block_accepted']:.1f}% | {r['aligned']['accepted_per_block']:.2f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E1 Step 0: an E0 checkpoint as a verified block drafter",
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="the released base checkpoint the run was trained on")
    p.add_argument("--checkpoint", type=Path, required=True, help="an E0 run's last.pt")
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k-steps", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--orders", nargs="+", default=["prefix", "confidence"], choices=["prefix", "confidence", "exact"])
    p.add_argument("--mbpp-split", default="validation")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--check", type=int, default=10, help="problems of the committed-stream check (0: skip)")
    p.add_argument("--dtype", default="bf16", choices=sorted(DTYPES), help="bf16: E0's precision; fp32: float32")
    p.add_argument("--device", default="auto", help="auto: cuda, then mps, then cpu (chat_vpa.device)")
    args = p.parse_args(argv)

    from chat_vpa.backbone import RealizerSpec, Stage0Realizer, load_gemma4
    from chat_vpa.block_dfm import BlockDFM
    from chat_vpa.block_sampler import BlockSamplerConfig
    from chat_vpa.data import HFTokenizer
    from chat_vpa.device import resolve_device
    from chat_vpa.evaluate import load_tasks, prompt_text
    from chat_vpa.rollouts import generate_rollouts

    device = resolve_device(args.device)
    precise = args.dtype == "fp32"
    model, hf_tokenizer = load_gemma4(args.model, dtype=torch_dtype(args.dtype), token=os.environ.get("HF_TOKEN") or None)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("stage") != "e0":
        raise SystemExit(f"{args.checkpoint} is not an E0 checkpoint")
    tokenizer = HFTokenizer(hf_tokenizer)
    spec = RealizerSpec.from_dict(ckpt["spec"])
    if spec.mask_id != tokenizer.mask_id:
        raise SystemExit("the checkpoint's [MASK] id is not this tokenizer's")
    realizer = Stage0Realizer.from_pretrained_model(model, spec)
    realizer.load_trainable_state(ckpt["trainable"])
    for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(model.model, tower):
            setattr(model.model, tower, None)
    registers = int(ckpt["config"].get("num_registers", 0))
    block_model = BlockDFM(realizer, num_registers=registers)
    if registers:
        with torch.no_grad():
            block_model.registers.copy_(ckpt["registers"].to(block_model.registers.dtype))
    block_model.to(device).eval()
    model.to(device)
    block_size = int(ckpt["config"]["block_size"])
    schedule = ckpt["config"]["schedule"]
    name = f"{ckpt['config']['arm']} B{block_size} (step {ckpt['step']})"

    tasks = load_tasks("mbpp", args.eval_dir, args.mbpp_split, args.limit)
    if not tasks:
        raise SystemExit(f"no MBPP problem under {args.eval_dir}")
    prompts = [[tokenizer.bos_id, *tokenizer.encode_batch([prompt_text(task, "mbpp")])[0]] for task in tasks]
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. The frozen backbone's greedy continuations.
    start = time.perf_counter()
    targets: list[list[int]] = []
    ended: list[bool] = []
    for group in _batches(list(range(len(prompts))), args.batch_size):
        generated, done, _ = generate_rollouts(model, hf_tokenizer, [prompts[i] for i in group], ["write"] * len(group),
                                               args.max_new_tokens, 0.0, 1.0, 1, seed=0)
        targets += generated
        ended += done
    with open(args.out / "reference.jsonl", "w", encoding="utf-8") as handle:
        for task, prompt, y, done in zip(tasks, prompts, targets, ended, strict=True):
            handle.write(json.dumps({"task_id": task.task_id, "prompt_ids": prompt, "greedy_ids": y,
                                     "ended_with_eos": done}) + "\n")
    lengths = [len(y) for y in targets]
    print(f"greedy continuations: {len(targets)} problems, {sum(lengths)} tokens, mean {sum(lengths) / len(lengths):.1f}, "
          f"{sum(1 for n in lengths if n == 0)} empty, {time.perf_counter() - start:.0f} s", file=sys.stderr, flush=True)

    forbidden = [tokenizer.pad_id, tokenizer.bos_id]
    summary: dict[str, Any] = {
        "what": "an E0 checkpoint as a block drafter under greedy verification (docs/e1.md, Section 3.3); passes "
                "are counted, not timed", "checkpoint": str(args.checkpoint), "drafter": name, "model": args.model,
        "split": args.mbpp_split, "problems": len(tasks), "greedy_tokens": sum(lengths), "block_size": block_size,
        "max_new_tokens": args.max_new_tokens, "device": str(device), "dtype": args.dtype, "rows": []}

    # The committed stream as its own drafter: every draft must be accepted.
    if args.check:
        n = min(args.check, len(prompts))
        own = block_drafter(block_model, BlockSamplerConfig(1, 1, schedule, "prefix", 0.0, 0.0), forbidden, device,
                            causal=True, adapters=False, precise=precise)
        traces = simulate(prompts[:n], targets[:n], own, args.batch_size)
        drafts = sum(t.cycles for t in traces)
        kept = sum(sum(t.accepted) for t in traces)
        summary["committed_stream_check"] = {"problems": n, "drafts": drafts, "accepted": kept,
                                             "share": kept / max(drafts, 1)}
        print(f"committed-stream check ({args.dtype}): {kept} of {drafts} one-token drafts accepted on {n} problems"
              + ("" if kept == drafts else ": generate and the realizer's path choose different tokens there, a "
                 "numerical difference in this precision"), file=sys.stderr, flush=True)

    for order in args.orders:
        for k in args.k_steps:
            cfg = BlockSamplerConfig(block_size, k, schedule, order, 0.0, 0.0)
            draft = block_drafter(block_model, cfg, forbidden, device, precise=precise)
            start = time.perf_counter()
            traces = simulate(prompts, targets, draft, args.batch_size)
            per_block, hits, counts = aligned(prompts, targets, draft, block_size, args.batch_size)
            blocks = sum(len(b) for b in per_block)
            row = {"drafter": name, "num_steps": k, "order": order,
                   "speculative": summarize(traces, k, block_size),
                   "aligned": {"blocks": blocks, "accepted_per_block": sum(sum(b) for b in per_block) / max(blocks, 1),
                               "slot_agreement": [h / c if c else None for h, c in zip(hits, counts, strict=True)]},
                   "seconds": round(time.perf_counter() - start, 1)}
            summary["rows"].append(row)
            write_json(args.out / "summary.json", summary)
            s = row["speculative"]
            print(f"K = {k}, {order}: {s['accepted_per_cycle']:.2f} accepted per cycle, "
                  f"{s['tokens_per_backbone_pass']:.3f} tokens per backbone pass (AR = 1), {row['seconds']} s",
                  file=sys.stderr, flush=True)
    print(_table(summary["rows"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
