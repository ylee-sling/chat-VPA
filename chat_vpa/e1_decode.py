# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_decode.py -- E1's Step 0 in Hugging Face Transformers (docs/e1.md, Section 3.1): the released Gemma 4 E2B-it
decoding the development prompts greedily, alone and with a drafter that it verifies.

    python -m chat_vpa.e1_decode --model MODEL --assistant DRAFTER --prompts ASSETS/prompts/dev.jsonl \\
        --arms ar mtp:2 mtp:4 mtp:6 mtp:8 lookup:10 --out runs/e1-step0/decode/l4-bf16

Arms (all through the released model's own ``generate``, batch 1, ``do_sample=False``):

    ar          the model alone: one pass per token (the baseline)
    mtp:G       with Google's Multi-Token Prediction drafter (``assistant_model=``): the drafter proposes G tokens
                per cycle, the target checks them in one pass and keeps those that equal its own argmax
    lookup:G    prompt-lookup decoding (``prompt_lookup_num_tokens=G``): drafts copied from the context, no
                drafter model; what repetition alone is worth

Per prompt and arm the script records the new tokens, the wall-clock time of the call (device synchronized), the
time until the first new tokens arrive, and the forward passes of the target and of the drafter (counted with
hooks, as ``PassAudit`` does). From these:

    decode tok/s            tokens after the first arrival / time after the first arrival
    end-to-end tok/s        all new tokens / the whole call (prefill included)
    tokens per cycle        tokens after the first pass / target passes after the first: 1 for ``ar``
    exact                   the arm's tokens equal ``ar``'s for the same prompt, in the same run; with the
                            position of the first difference where they do not
    speed-up                the arm's decode tok/s over ``ar``'s, with a 95% paired bootstrap over prompts
                            (10,000 resamples, seed 0)

The arms of a prompt run back to back, so that a drift of the machine falls on all of them. ``--warmup`` prompts
are decoded first and thrown away. Everything is *measured* in this runtime (PyTorch, eager) on this device in
this precision, and the summary says which: a speed-up here is not a speed-up in vLLM or LiteRT-LM, where the
cost of a pass is a different mix of arithmetic and overhead (docs/e1.md, Section 3.4).

``--profile`` adds synchronized timers around every target and drafter pass and reports where the time went. The
timers themselves cost time, so a profiled run is written beside the plain one and never used for a speed-up.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from chat_vpa.e0_report import paired_bootstrap
from chat_vpa.e1_common import (
    DTYPES,
    chat_ids,
    environment,
    file_sha256,
    load_target,
    median,
    read_prompts,
    synchronizer,
    torch_dtype,
    write_json,
)

__all__ = ["Arm", "PassCounter", "decode_once", "first_difference", "parse_arm", "summarize"]

_ARM = re.compile(r"^(ar)$|^(mtp|lookup):([1-9]\d*)$")


@dataclass(frozen=True)
class Arm:
    kind: str            # "ar" | "mtp" | "lookup"
    length: int = 0      # draft tokens per cycle

    @property
    def name(self) -> str:
        return self.kind if self.kind == "ar" else f"{self.kind}-{self.length}"


def parse_arm(text: str) -> Arm:
    match = _ARM.match(text)
    if not match:
        raise ValueError(f"an arm is 'ar', 'mtp:G' or 'lookup:G' with G >= 1, not {text!r}")
    return Arm("ar") if match.group(1) else Arm(match.group(2), int(match.group(3)))


class PassCounter:
    """Counts a module's forward passes with a hook and, when ``sync`` is given, times them (synchronized)."""

    def __init__(self, module: nn.Module | None, sync: Any = None) -> None:
        self.calls, self.seconds = 0, 0.0
        self._sync, self._start, self._handles = sync, 0.0, []
        if module is not None:
            if sync is not None:
                self._handles.append(module.register_forward_pre_hook(self._before))
            self._handles.append(module.register_forward_hook(self._after))

    def _before(self, module: nn.Module, args: Any) -> None:
        self._sync()
        self._start = time.perf_counter()

    def _after(self, module: nn.Module, args: Any, output: Any) -> None:
        self.calls += 1
        if self._sync is not None:
            self._sync()
            self.seconds += time.perf_counter() - self._start

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


class _Arrival:
    """A streamer that notes when the first new tokens arrive and how many they are."""

    def __init__(self, sync: Any) -> None:
        self._sync, self._puts = sync, 0
        self.first_time: float | None = None
        self.first_tokens = 0

    def put(self, value: Tensor) -> None:
        self._puts += 1
        if self._puts == 1 or self.first_time is not None:        # the first put is the prompt
            return
        self._sync()
        self.first_time = time.perf_counter()
        self.first_tokens = int(value.numel())

    def end(self) -> None:
        return None


@torch.no_grad()
def decode_once(model: Any, assistant: Any, ids: Tensor, arm: Arm, max_new_tokens: int, profile: bool = False
                ) -> dict[str, Any]:
    """One greedy response of one arm. ``ids`` is ``[1, P]`` on the model's device."""
    device = ids.device
    sync = synchronizer(device)
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": False, "temperature": None,
                              "top_p": None, "top_k": None}
    if arm.kind == "mtp":
        if assistant is None:
            raise ValueError("an mtp arm needs the drafter (--assistant)")
        assistant.generation_config.num_assistant_tokens = arm.length
        assistant.generation_config.num_assistant_tokens_schedule = "constant"
        kwargs["assistant_model"] = assistant
    elif arm.kind == "lookup":
        kwargs["prompt_lookup_num_tokens"] = arm.length
    arrival = _Arrival(sync)
    target = PassCounter(model, sync if profile else None)
    drafter = PassCounter(assistant if arm.kind == "mtp" else None, sync if profile else None)
    try:
        sync()
        start = time.perf_counter()
        output = model.generate(ids, attention_mask=torch.ones_like(ids), streamer=arrival, **kwargs)
        sync()
        end = time.perf_counter()
    finally:
        target.close()
        drafter.close()
    tokens = output[0, ids.shape[1]:].tolist()
    first = (arrival.first_time - start) if arrival.first_time is not None else end - start
    record = {"tokens": tokens, "seconds": end - start, "first_seconds": first,
              "first_tokens": arrival.first_tokens if arrival.first_time is not None else len(tokens),
              "target_passes": target.calls, "drafter_passes": drafter.calls}
    if profile:
        record.update(target_seconds=target.seconds, drafter_seconds=drafter.seconds)
    return record


def first_difference(a: Sequence[int], b: Sequence[int]) -> int | None:
    """Index of the first position where the sequences differ (a missing token differs); None if equal."""
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def _rate(tokens: Any, seconds: Any) -> Any:
    return tokens / seconds


def _speedup(tokens: Any, seconds: Any, base_tokens: Any, base_seconds: Any) -> Any:
    return (tokens / seconds) / (base_tokens / base_seconds)


def summarize(records: Sequence[dict[str, Any]], arms: Sequence[str], resamples: int = 10_000, seed: int = 0
              ) -> dict[str, dict[str, Any]]:
    """Per arm, the quantities of the module docstring. ``records`` are ``decode_once``'s with ``id`` and
    ``arm``; every prompt must have every arm, and ``ar`` must be among them."""
    by_arm: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in arms}
    for record in records:
        by_arm[record["arm"]][record["id"]] = record
    if "ar" not in by_arm:
        raise ValueError("the 'ar' arm is the reference of every comparison")
    ids = list(by_arm["ar"])
    for name in arms:
        if list(by_arm[name]) != ids:
            raise ValueError(f"arm {name} does not cover the prompts of 'ar' in the same order")

    def columns(name: str) -> dict[str, list[float]]:
        rows = [by_arm[name][i] for i in ids]
        return {"tokens": [float(len(r["tokens"])) for r in rows], "seconds": [float(r["seconds"]) for r in rows],
                "decode_tokens": [float(len(r["tokens"]) - r["first_tokens"]) for r in rows],
                "decode_seconds": [float(r["seconds"] - r["first_seconds"]) for r in rows],
                "first_seconds": [float(r["first_seconds"]) for r in rows],
                "passes": [float(r["target_passes"]) for r in rows]}

    base = columns("ar")
    out: dict[str, dict[str, Any]] = {}
    for name in arms:
        c = columns(name)
        rows = [by_arm[name][i] for i in ids]
        differences = [first_difference(r["tokens"], by_arm["ar"][i]["tokens"]) for r, i in zip(rows, ids, strict=True)]
        cycles = sum(c["passes"]) - len(ids)
        summary: dict[str, Any] = {
            "prompts": len(ids), "new_tokens": int(sum(c["tokens"])),
            "decode_tokens_per_second": sum(c["decode_tokens"]) / sum(c["decode_seconds"]),
            "end_to_end_tokens_per_second": sum(c["tokens"]) / sum(c["seconds"]),
            "median_seconds_to_first_tokens": median(c["first_seconds"]),
            "target_passes": int(sum(c["passes"])),
            "drafter_passes": int(sum(r["drafter_passes"] for r in rows)),
            "tokens_per_cycle": sum(c["decode_tokens"]) / cycles if cycles > 0 else None,
            "exact": sum(d is None for d in differences), "exact_share": sum(d is None for d in differences) / len(ids),
            "first_differences": {i: d for i, d in zip(ids, differences, strict=True) if d is not None}}
        if name != "ar":
            for key, numer, denom in (("decode_speedup", "decode_tokens", "decode_seconds"),
                                      ("end_to_end_speedup", "tokens", "seconds")):
                point, low, high = paired_bootstrap([c[numer], c[denom], base[numer], base[denom]], _speedup,
                                                    resamples, seed)
                summary[key] = point
                summary[key + "_interval"] = [low, high]
        if all("target_seconds" in r for r in rows):
            total = sum(c["seconds"])
            summary["profile"] = {"target_share": sum(r["target_seconds"] for r in rows) / total,
                                  "drafter_share": sum(r["drafter_seconds"] for r in rows) / total}
        out[name] = summary
    return out


def _markdown(summary: dict[str, dict[str, Any]]) -> str:
    lines = ["| arm | decode tok/s | speed-up (decode) | end-to-end tok/s | speed-up (end to end) | tokens per cycle | "
             "exact | first tokens after, s |", "|---|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        def ratio(key: str, s: dict[str, Any] = s) -> str:
            if key not in s:
                return "1 (reference)"
            low, high = s[key + "_interval"]
            return f"{s[key]:.2f} [{low:.2f}, {high:.2f}]"
        cycle = "--" if s["tokens_per_cycle"] is None else f"{s['tokens_per_cycle']:.2f}"
        lines.append(f"| {name} | {s['decode_tokens_per_second']:.1f} | {ratio('decode_speedup')} | "
                     f"{s['end_to_end_tokens_per_second']:.1f} | {ratio('end_to_end_speedup')} | {cycle} | "
                     f"{s['exact']} of {s['prompts']} | {s['median_seconds_to_first_tokens']:.3f} |")
    return "\n".join(lines)


def load_assistant(path: str | Path, dtype: str, device: torch.device) -> Any:
    """Google's drafter (``Gemma4AssistantForCausalLM``), frozen, on ``device``."""
    from transformers import Gemma4AssistantForCausalLM

    return Gemma4AssistantForCausalLM.from_pretrained(str(path), dtype=torch_dtype(dtype)).to(device).eval()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E1 Step 0: greedy decoding with and without a verified drafter",
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="the released instruction-tuned checkpoint (a directory)")
    p.add_argument("--assistant", default=None, help="Google's drafter for it (needed by mtp arms)")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arms", nargs="+", default=["ar", "mtp:4"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    p.add_argument("--threads", type=int, default=None, help="CPU threads (CPU runs)")
    p.add_argument("--profile", action="store_true", help="time every pass (never used for a speed-up)")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    arms = [parse_arm(a) for a in args.arms]
    if arms[0].kind != "ar" or len({a.name for a in arms}) != len(arms):
        raise SystemExit("--arms must start with 'ar' and name each arm once")

    from chat_vpa.device import resolve_device

    device = resolve_device(args.device)
    model, tokenizer = load_target(args.model, args.dtype, device, args.threads)
    assistant = None
    if any(a.kind == "mtp" for a in arms):
        if not args.assistant:
            raise SystemExit("mtp arms need --assistant")
        assistant = load_assistant(args.assistant, args.dtype, device)
    prompts = read_prompts(args.prompts, args.limit)
    encoded = [torch.tensor([chat_ids(tokenizer, row["prompt"])], dtype=torch.long, device=device) for row in prompts]
    for ids in encoded[: args.warmup]:
        for arm in arms:
            decode_once(model, assistant, ids, arm, min(args.max_new_tokens, 16))

    out = args.out / ("profile" if args.profile else "")
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with open(out / "samples.jsonl", "w", encoding="utf-8") as handle:
        for index, (row, ids) in enumerate(zip(prompts, encoded, strict=True)):
            for arm in arms:
                record = {"id": row["id"], "arm": arm.name, "prompt_tokens": int(ids.shape[1]),
                          **decode_once(model, assistant, ids, arm, args.max_new_tokens, args.profile)}
                records.append(record)
                handle.write(json.dumps({**record, "text": tokenizer.decode(record["tokens"])},
                                        ensure_ascii=False) + "\n")
            handle.flush()
            if (index + 1) % 10 == 0:
                print(f"  {index + 1}/{len(prompts)} prompts", file=sys.stderr, flush=True)
    summary = summarize(records, [a.name for a in arms])
    template = tokenizer.apply_chat_template([{"role": "user", "content": prompts[0]["prompt"][:80]}],
                                             add_generation_prompt=True, tokenize=False)
    write_json(out / "summary.json", {
        "what": "greedy decoding of the development prompts, measured (docs/e1.md, Section 3.1)"
                + ("; profiled: not for speed-ups" if args.profile else ""),
        "environment": environment(device, args.dtype, args.threads), "model": args.model,
        "assistant": args.assistant, "prompts": str(args.prompts), "prompts_sha256": file_sha256(args.prompts),
        "max_new_tokens": args.max_new_tokens, "warmup": args.warmup, "profiled": args.profile,
        "chat_template_of_the_first_prompt": template, "arms": summary})
    print(_markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
