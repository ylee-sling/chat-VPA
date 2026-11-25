# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_ceiling.py -- the ceiling of E1's Step 0 (docs/e1.md, Section 3.2): what any drafter could gain on one device
and runtime, measured on the target model alone.

    python -m chat_vpa.e1_ceiling --model MODEL --prompts ASSETS/prompts/dev.jsonl --out runs/e1-step0/ceiling/l4-bf16

Speculative decoding replaces n passes of the target over one token each by one pass over n tokens. It can only
pay where that one pass is cheaper than the n it replaces. For each prompt this script prefills the prompt,
decodes N tokens greedily one pass at a time (the autoregressive reference), and then times one pass over the
first n of those tokens from the same prompt cache, for each n of ``--lengths``:

    t(n)        median seconds of a pass over n new positions (cache copied outside the timer, device synchronized)
    ceiling(n)  = n * t(1) / t(n): the speed-up of a drafter that costs nothing and is always right, with n - 1
                draft tokens per cycle (a pass over n positions yields at most n tokens)
    predicted   speed-up at a per-token acceptance rate alpha with free drafting:
                E[tokens] * t(1) / t(n), E[tokens] = (1 - alpha^n) / (1 - alpha)

The ceiling and the predictions are *predicted* quantities (a roofline of this device, runtime and precision,
computed from measured pass times); only t(n) is measured. A real drafter adds its own time to every cycle.

The same passes answer a second question, the numerical one behind "lossless": does a pass over n tokens choose
the tokens that n passes over one token choose? For every n the script counts the positions where the two
argmaxes differ. In exact arithmetic the count is zero; in a finite precision it need not be, and where it is
not, token-identical verification cannot be promised in that precision whatever the drafter.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from chat_vpa.e1_common import (
    chat_ids,
    environment,
    file_sha256,
    load_target,
    median,
    read_prompts,
    synchronizer,
    write_json,
)

__all__ = ["ALPHAS", "ceiling_table", "expected_tokens", "measure_prompt"]

ALPHAS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)


def expected_tokens(alpha: float, n: int) -> float:
    """Tokens a cycle yields when each of n - 1 draft tokens is accepted with probability ``alpha``, independently,
    up to the first rejection, plus the target's own token: sum_{j < n} alpha^j."""
    if not 0 <= alpha <= 1 or n < 1:
        raise ValueError("need 0 <= alpha <= 1 and n >= 1")
    return float(n) if alpha == 1 else (1 - alpha ** n) / (1 - alpha)


def ceiling_table(seconds: dict[int, float], alphas: Sequence[float] = ALPHAS) -> list[dict[str, Any]]:
    """Per n: t(n) / t(1), the ceiling n t(1) / t(n) and the predicted speed-ups (see the module docstring)."""
    if 1 not in seconds:
        raise ValueError("the pass over one position is the unit: include n = 1")
    t1 = seconds[1]
    rows = []
    for n in sorted(seconds):
        rows.append({"n": n, "seconds": seconds[n], "relative_cost": seconds[n] / t1, "ceiling": n * t1 / seconds[n],
                     "predicted": {f"{a:g}": expected_tokens(a, n) * t1 / seconds[n] for a in alphas}})
    return rows


@torch.no_grad()
def measure_prompt(model: Any, prompt_ids: Sequence[int], lengths: Sequence[int], repeats: int,
                   device: torch.device) -> dict[str, Any]:
    """One prompt: the times of a pass over n positions, and the disagreements with one-token passes.

    Returns ``seconds`` (n -> list of ``repeats`` readings), ``sequential_seconds`` (one reading per one-token
    pass of the reference decoding), ``mismatches`` (n -> positions of the pass over n whose argmax differs from
    the reference), ``positions`` (n -> n), and the reference decoding itself: ``first_token`` (after the
    prompt) and ``reference_tokens`` (the next ``max(lengths)`` tokens).
    """
    sync = synchronizer(device)
    steps = max(lengths)
    ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    out = model(input_ids=ids, use_cache=True)
    prompt_cache = out.past_key_values
    token = out.logits[:, -1].float().argmax(dim=-1)
    first_token = int(token)
    # The reference: one token per pass. inputs[j] is the token fed at step j, chosen[j] the argmax it produces.
    inputs: list[Tensor] = []
    chosen: list[int] = []
    sequential: list[float] = []
    cache = copy.deepcopy(prompt_cache)
    for _ in range(steps):
        inputs.append(token)
        sync()
        start = time.perf_counter()
        out = model(input_ids=token[:, None], past_key_values=cache, use_cache=True)
        sync()
        sequential.append(time.perf_counter() - start)
        cache = out.past_key_values
        token = out.logits[:, -1].float().argmax(dim=-1)
        chosen.append(int(token))
    fed = torch.stack(inputs, dim=1)                                           # [1, steps]
    seconds: dict[int, list[float]] = {}
    mismatches: dict[int, int] = {}
    for n in lengths:
        readings = []
        for _ in range(repeats):
            cache = copy.deepcopy(prompt_cache)
            sync()
            start = time.perf_counter()
            out = model(input_ids=fed[:, :n], past_key_values=cache, use_cache=True)
            sync()
            readings.append(time.perf_counter() - start)
        seconds[n] = readings
        picked = out.logits[0, -n:].float().argmax(dim=-1).tolist()
        mismatches[n] = sum(int(a != b) for a, b in zip(picked, chosen[:n], strict=True))
    return {"prompt_tokens": len(prompt_ids), "seconds": seconds, "sequential_seconds": sequential,
            "mismatches": mismatches, "positions": {n: n for n in lengths}, "first_token": first_token,
            "reference_tokens": chosen}


def _markdown(rows: Sequence[dict[str, Any]], alphas: Sequence[float]) -> str:
    head = ("| n | t(n), ms | t(n) / t(1) | ceiling | " + " | ".join(f"alpha = {a:g}" for a in alphas)
            + " | argmax differs |")
    lines = [head, "|" + "---|" * (5 + len(alphas))]
    for r in rows:
        lines.append(f"| {r['n']} | {1000 * r['seconds']:.2f} | {r['relative_cost']:.2f} | {r['ceiling']:.2f} | "
                     + " | ".join(f"{r['predicted'][f'{a:g}']:.2f}" for a in alphas)
                     + f" | {r['mismatches']} of {r['positions']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E1 Step 0: the ceiling of one device and runtime", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32])
    p.add_argument("--limit", type=int, default=20, help="prompts (the first of the file)")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2, help="prompts measured first and thrown away")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--threads", type=int, default=None, help="CPU threads (CPU runs)")
    p.add_argument("--device", default="auto", help="auto: cuda, then mps, then cpu (chat_vpa.device)")
    args = p.parse_args(argv)
    if 1 not in args.lengths or min(args.lengths) < 1:
        raise SystemExit("--lengths must be positive and include 1")

    from chat_vpa.device import resolve_device

    device = resolve_device(args.device)
    model, tokenizer = load_target(args.model, args.dtype, device, args.threads)
    prompts = read_prompts(args.prompts, args.limit)
    lengths = sorted(set(args.lengths))
    for row in prompts[: args.warmup]:
        measure_prompt(model, chat_ids(tokenizer, row["prompt"]), lengths, 1, device)
    per_prompt = []
    for index, row in enumerate(prompts):
        result = measure_prompt(model, chat_ids(tokenizer, row["prompt"]), lengths, args.repeats, device)
        per_prompt.append({"id": row["id"], **result})
        if (index + 1) % 5 == 0:
            print(f"  {index + 1}/{len(prompts)} prompts", file=sys.stderr, flush=True)
    pooled = {n: median([s for r in per_prompt for s in r["seconds"][n]]) for n in lengths}
    rows = ceiling_table(pooled)
    for r in rows:
        r["mismatches"] = sum(q["mismatches"][r["n"]] for q in per_prompt)
        r["positions"] = sum(q["positions"][r["n"]] for q in per_prompt)
    summary = {
        "what": "pass times of the target alone (measured) and the speed-ups they allow (predicted): "
                "docs/e1.md, Section 3.2",
        "environment": environment(device, args.dtype, args.threads), "model": args.model,
        "prompts": str(args.prompts), "prompts_sha256": file_sha256(args.prompts), "prompt_count": len(prompts),
        "mean_prompt_tokens": sum(r["prompt_tokens"] for r in per_prompt) / len(per_prompt),
        "repeats": args.repeats, "alphas": list(ALPHAS),
        "sequential_pass_seconds": median([s for r in per_prompt for s in r["sequential_seconds"]]),
        "rows": rows, "per_prompt": per_prompt}
    write_json(args.out / "summary.json", summary)
    print(_markdown(rows, ALPHAS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
