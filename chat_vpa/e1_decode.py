# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_decode.py -- E1's Step 0 in Hugging Face Transformers (docs/e1.md, Section 3.1): the released Gemma 4 E2B-it
decoding the development prompts greedily, alone and with a drafter that it verifies.

    python -m chat_vpa.e1_decode --model MODEL --assistant DRAFTER --prompts ASSETS/prompts/dev.jsonl \\
        --arms ar mtp:2 mtp:4 mtp:6 mtp:8 lookup:10 --out runs/e1-step0/decode/l4-bf16
    python -m chat_vpa.e1_decode --model MODEL --drafter a1=DRAFTER --drafter a2=runs/e1-step1/a2/final \\
        --prompts ASSETS/prompts/dev.jsonl --arms ar a1:4 a2:4 a1:4:clean a2:4:clean --out ...

Arms (all through the released model's own ``generate``, batch 1, ``do_sample=False``):

    ar          the model alone: one pass per token (the baseline)
    mtp:G       with Google's Multi-Token Prediction drafter (``assistant_model=``, the drafter of ``--assistant``):
                the drafter proposes G tokens per cycle, the target checks them in one pass and keeps those that
                equal its own argmax
    D:G         the same with the drafter labelled D by ``--drafter D=PATH`` (Step 1: A1 and A2 in one run, S9)
    D:G:clean   the same drafter through Transformers' own candidate generator, except that the target's keys it
                hands the drafter are cut to the accepted prefix (``AcceptedPrefixKeys``; docs/e1.md, Section
                11.9, flag 1: after a partly accepted cycle Transformers 5.21 also hands it rejected drafts' keys).
                After a cycle that accepted no draft the target's sliding cache no longer holds the oldest of the
                W + 1 keys that a clean cycle shows the drafter's sliding layers, so those see W of them
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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
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

__all__ = ["Arm", "PassCounter", "accepted_prefix", "accepted_prefix_keys", "decode_once", "first_difference",
           "parse_arm", "summarize"]

_ARM = re.compile(r"^(ar)$|^(lookup):([1-9]\d*)$|^([a-z][a-z0-9]*):([1-9]\d*)(:clean)?$")


@dataclass(frozen=True)
class Arm:
    kind: str            # "ar" | "mtp" (a drafter) | "lookup"
    length: int = 0      # draft tokens per cycle
    label: str = "mtp"   # which drafter: "mtp" is --assistant's, any other label one of --drafter's
    clean: bool = False  # the target's keys cut to the accepted prefix (AcceptedPrefixKeys)

    @property
    def name(self) -> str:
        if self.kind == "ar":
            return "ar"
        if self.kind == "lookup":
            return f"lookup-{self.length}"
        return f"{self.label}-{self.length}" + ("-clean" if self.clean else "")


def parse_arm(text: str, labels: Sequence[str] = ("mtp",)) -> Arm:
    """An arm from its command-line form; ``labels`` are the drafters that exist (``mtp`` and ``--drafter``'s)."""
    match = _ARM.match(text)
    if match and match.group(1):
        return Arm("ar")
    if match and match.group(2):
        return Arm("lookup", int(match.group(3)))
    if match and match.group(4) in labels and match.group(4) not in ("ar", "lookup"):
        return Arm("mtp", int(match.group(5)), match.group(4), bool(match.group(6)))
    raise ValueError(f"an arm is 'ar', 'lookup:G', or 'D:G' or 'D:G:clean' for a drafter D among {sorted(labels)} "
                     f"('mtp' is --assistant's; the others come from --drafter D=PATH), with G >= 1; not {text!r}")


def accepted_prefix(states: dict[str, tuple[Tensor, Tensor]], position: int) -> dict[str, tuple[Tensor, Tensor]]:
    """The target's shared keys and values without the entries at ``position`` and after: those of drafts the
    verifying pass saw and the target rejected. Every tensor ends at the verifying pass's last position, and the
    full-attention one starts at position 0, so its length less ``position`` is the number of such entries."""
    full = states.get("full_attention")
    if full is None:
        raise ValueError("the accepted prefix is located through the full-attention keys, which are missing")
    stale = full[0].shape[2] - position
    if stale <= 0:
        return states
    return {name: (k[:, :, : k.shape[2] - stale], v[:, :, : v.shape[2] - stale]) for name, (k, v) in states.items()}


@lru_cache(maxsize=1)
def _accepted_prefix_class() -> Any:
    """``SinglePositionMultiTokenCandidateGenerator`` with the target's keys cut to the accepted prefix before it
    uses them; everything else is Transformers' own."""
    from transformers.generation.candidate_generator import SinglePositionMultiTokenCandidateGenerator

    class AcceptedPrefixKeys(SinglePositionMultiTokenCandidateGenerator):
        def get_candidates(self, input_ids: Tensor, model_kwargs: dict[str, Any], model_outputs: Any,
                           is_first_iteration: bool, n_last_matches: int, **kwargs: Any) -> Any:
            states = getattr(model_outputs, "shared_kv_states", None) if model_outputs is not None else None
            if not is_first_iteration and states:
                # The drafted queries sit at the last token's position (Transformers' own rule).
                model_outputs = SimpleNamespace(hidden_states=model_outputs.hidden_states,
                                                shared_kv_states=accepted_prefix(states, input_ids.shape[1] - 1))
            return super().get_candidates(input_ids, model_kwargs, model_outputs, is_first_iteration,
                                          n_last_matches, **kwargs)

    return AcceptedPrefixKeys


@contextmanager
def accepted_prefix_keys() -> Iterator[None]:
    """Within the block, ``generate`` builds ``AcceptedPrefixKeys`` where it would build Transformers' MTP
    candidate generator (it looks the class up by name in ``transformers.generation.utils`` at each call)."""
    import transformers.generation.utils as generation

    original = generation.SinglePositionMultiTokenCandidateGenerator
    generation.SinglePositionMultiTokenCandidateGenerator = _accepted_prefix_class()
    try:
        yield
    finally:
        generation.SinglePositionMultiTokenCandidateGenerator = original


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
    """One greedy response of one arm. ``ids`` is ``[1, P]`` on the model's device. ``assistant`` is the drafter
    of every drafter arm, or a mapping from the arms' labels to drafters."""
    device = ids.device
    sync = synchronizer(device)
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": False, "temperature": None,
                              "top_p": None, "top_k": None}
    drafter_model = None
    if arm.kind == "mtp":
        drafter_model = assistant.get(arm.label) if isinstance(assistant, dict) else assistant
        if drafter_model is None:
            raise ValueError(f"the arm {arm.name} needs its drafter ({arm.label}: --assistant or --drafter)")
        drafter_model.generation_config.num_assistant_tokens = arm.length
        drafter_model.generation_config.num_assistant_tokens_schedule = "constant"
        kwargs["assistant_model"] = drafter_model
    elif arm.kind == "lookup":
        kwargs["prompt_lookup_num_tokens"] = arm.length
    arrival = _Arrival(sync)
    target = PassCounter(model, sync if profile else None)
    drafter = PassCounter(drafter_model, sync if profile else None)
    keys = accepted_prefix_keys() if arm.clean else _nothing()
    try:
        with keys:
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


@contextmanager
def _nothing() -> Iterator[None]:
    yield


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
    p.add_argument("--drafter", action="append", default=[], metavar="LABEL=PATH",
                   help="a drafter for the arms LABEL:G and LABEL:G:clean (repeatable; Step 1: a1=, a2=)")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arms", nargs="+", default=["ar", "mtp:4"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    p.add_argument("--threads", type=int, default=None, help="CPU threads (CPU runs)")
    p.add_argument("--profile", action="store_true", help="time every pass (never used for a speed-up)")
    p.add_argument("--device", default="auto", help="auto: cuda, then mps, then cpu (chat_vpa.device)")
    args = p.parse_args(argv)
    paths: dict[str, str] = {}
    for item in args.drafter:
        label, _, path = item.partition("=")
        if not path or not re.fullmatch(r"[a-z][a-z0-9]*", label) or label in ("ar", "lookup", "mtp") \
                or label in paths:
            raise SystemExit(f"--drafter takes LABEL=PATH with a new lower-case label other than ar, lookup and mtp, "
                             f"not {item!r}")
        paths[label] = path
    if args.assistant:
        paths["mtp"] = args.assistant
    try:
        arms = [parse_arm(a, ("mtp", *paths)) for a in args.arms]
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if arms[0].kind != "ar" or len({a.name for a in arms}) != len(arms):
        raise SystemExit("--arms must start with 'ar' and name each arm once")

    from chat_vpa.device import resolve_device

    device = resolve_device(args.device)
    if any(a.kind == "mtp" and a.label not in paths for a in arms):
        raise SystemExit("mtp arms need --assistant")
    model, tokenizer = load_target(args.model, args.dtype, device, args.threads)
    used = sorted({a.label for a in arms if a.kind == "mtp"})
    assistant: Any = {label: load_assistant(paths[label], args.dtype, device) for label in used}
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
        "assistant": args.assistant, "drafters": {label: paths[label] for label in used},
        "prompts": str(args.prompts), "prompts_sha256": file_sha256(args.prompts),
        "max_new_tokens": args.max_new_tokens, "warmup": args.warmup, "profiled": args.profile,
        "chat_template_of_the_first_prompt": template, "arms": summary})
    print(_markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
