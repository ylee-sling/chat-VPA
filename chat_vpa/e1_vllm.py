# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_vllm.py -- E1's Step 0 in vLLM (docs/e1.md, Section 3.4): greedy decoding of the development prompts by vLLM's
engine, alone and with Google's drafter, one engine per arm.

    ASSETS/venvs/vllm/bin/python -m chat_vpa.e1_vllm run --arm mtp:4 --tag mtp-4 \\
        --model ASSETS/models/gemma-4-E2B-it --assistant ASSETS/models/gemma-4-E2B-it-assistant \\
        --prompts ASSETS/prompts/dev.jsonl --limit 100 --out runs/e1-step0/vllm/gpu-bf16
    python -m chat_vpa.e1_vllm summarize --out runs/e1-step0/vllm/gpu-bf16
    ASSETS/venvs/vllm/bin/python -m chat_vpa.e1_vllm smoke --model ... --assistant ... --prompts ...

vLLM is the runtime of Google's reported 1.97x for E2B (an H100) and a server engine: on the L4 it bounds what the
discrete GPU can do, and it is not the product's runtime (decision 4). It runs in vLLM's own environment
(runs/assets/venvs/vllm, with its own torch); ``summarize`` needs only numpy and runs anywhere.

Arms. ``off``, the model alone, and ``mtp:G``: speculative_config = {"method": "mtp", "model": <the assistant
checkpoint>, "num_speculative_tokens": G}. vLLM 0.34.0 maps Google's ``gemma4_assistant`` checkpoint to its Gemma 4
MTP drafter, which reads the target's KV cache (``vllm/config/speculative.py``; its documentation,
``docs/features/speculative_decoding/mtp.md``, names the E2B assistant as supported).

One engine per arm, and one process per engine: vLLM holds the GPU for the engine's life, so the arms cannot
alternate prompt by prompt as Section 3.5 asks. Each arm decodes every prompt, one request at a time, and the
reference arm runs first and again last (tags ``off-1`` and ``off-2``): off-2's speed against off-1 is the drift
of the machine over the run, which should be 1.

Engine settings, the same for every arm: bfloat16, seed 0, ``max_model_len`` 4096, prefix caching off (so that a
prompt of the warm-up, or a repeated prefix, is prefilled again), the multimodal encoders off
(``language_model_only``), request statistics on (the engine's timestamps of each request), and for the
speculative arms vLLM's per-request acceptance counts; CUDA graphs and compilation are vLLM's defaults. Sampling:
temperature 0 (vLLM's greedy path), at most ``--max-new-tokens`` tokens, the stop tokens of the model's generation
config. The prompt is ``e1_common.chat_ids`` (one user turn in the model's template, thinking off, as in the
PyTorch arms), passed as token ids.

Recorded per prompt: the generated token ids and text; the wall-clock seconds of the generate call; from the
engine's timestamps, the seconds from scheduling to the first token (prefill) and from the first token to the last
(decode); for the speculative arms, vLLM's acceptance histogram. ``summarize`` computes:

    decode tok/s        tokens after the first / seconds from the first token to the last, summed over prompts
    end-to-end tok/s    tokens / seconds of the generate call, summed over prompts
    tokens per cycle    1 + accepted draft tokens / verification steps (vLLM's counts; the bonus token included)
    same tokens         the arm's token ids equal off-1's for the same prompt; otherwise the first differing position
    speed-up            the arm's tok/s over off-1's, 95% paired bootstrap over prompts (10,000 resamples, seed 0)

Everything is *measured* on this machine with the versions the record names.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chat_vpa.e1_common import chat_ids, file_sha256, median, read_prompts, write_json

__all__ = ["VllmArm", "engine_kwargs", "parse_vllm_arm", "run_arm", "summarize", "tag_of"]

REFERENCE = "off-1"


@dataclass(frozen=True)
class VllmArm:
    kind: str                       # "off" or "mtp"
    length: int = 0                 # num_speculative_tokens of an mtp arm

    @property
    def name(self) -> str:
        return "off" if self.kind == "off" else f"mtp:{self.length}"


def parse_vllm_arm(text: str) -> VllmArm:
    """``off`` or ``mtp:G`` with G >= 1."""
    if text == "off":
        return VllmArm("off")
    kind, _, length = text.partition(":")
    if kind == "mtp" and length.isdigit() and int(length) >= 1:
        return VllmArm("mtp", int(length))
    raise ValueError(f"a vLLM arm is 'off' or 'mtp:G' with G >= 1, not {text!r}")


def tag_of(arms: Sequence[str]) -> list[str]:
    """The directory names of a sequence of arms: off-1, off-2, ... for the repeated reference, mtp-G otherwise."""
    tags, offs = [], 0
    for text in arms:
        arm = parse_vllm_arm(text)
        if arm.kind == "off":
            offs += 1
            tags.append(f"off-{offs}")
        else:
            tags.append(f"mtp-{arm.length}")
    if len(set(tags)) != len(tags):
        raise ValueError("each speculative arm is run once")
    return tags


def engine_kwargs(model: str | Path, assistant: str | Path | None, arm: VllmArm, max_model_len: int = 4096,
                  gpu_memory_utilization: float = 0.85) -> dict[str, Any]:
    """The arguments of ``vllm.LLM`` for an arm (the module docstring says why each is set)."""
    kwargs: dict[str, Any] = {"model": str(model), "dtype": "bfloat16", "seed": 0, "max_model_len": max_model_len,
                              "gpu_memory_utilization": gpu_memory_utilization, "enable_prefix_caching": False,
                              "language_model_only": True, "disable_log_stats": False}
    if arm.kind == "mtp":
        if assistant is None:
            raise ValueError("an mtp arm needs --assistant, the drafter's checkpoint")
        kwargs["speculative_config"] = {"method": "mtp", "model": str(assistant),
                                        "num_speculative_tokens": arm.length}
        kwargs["per_request_spec_decode_metrics"] = "summary"
    return kwargs


def _record(output: Any, seconds: float) -> dict[str, Any]:
    """What one finished request says about itself."""
    completion = output.outputs[0]
    stats = output.metrics
    if stats is None:
        raise RuntimeError("vLLM returned no request statistics: the engine must run with disable_log_stats=False")
    spec = completion.spec_decode_metrics
    return {"tokens": [int(t) for t in completion.token_ids], "text": completion.text,
            "finish_reason": completion.finish_reason, "seconds": seconds,
            "prefill_seconds": float(stats.first_token_ts - stats.scheduled_ts),
            "decode_seconds": float(stats.last_token_ts - stats.first_token_ts),
            "spec": None if spec is None else spec.to_dict()}


def run_arm(llm: Any, sampling_params: Any, tokenizer: Any, prompts: Sequence[dict[str, str]], warmup: int,
            progress: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """Every prompt through one engine, one request at a time; the first ``warmup`` prompts are decoded first and
    thrown away."""
    encoded = [chat_ids(tokenizer, row["prompt"]) for row in prompts]
    for ids in encoded[:warmup]:
        llm.generate([{"prompt_token_ids": ids}], sampling_params, use_tqdm=False)
    records: list[dict[str, Any]] = []
    for number, (row, ids) in enumerate(zip(prompts, encoded, strict=True), start=1):
        start = time.perf_counter()
        outputs = llm.generate([{"prompt_token_ids": ids}], sampling_params, use_tqdm=False)
        seconds = time.perf_counter() - start
        records.append({"id": row["id"], "prompt_tokens": len(ids), **_record(outputs[0], seconds)})
        if progress is not None and (number % 10 == 0 or number == len(prompts)):
            progress(f"  {number}/{len(prompts)} prompts")
    return records


def _ids_sha256(tokenizer: Any, prompts: Sequence[dict[str, str]]) -> str:
    encoded = [chat_ids(tokenizer, row["prompt"]) for row in prompts]
    return hashlib.sha256(json.dumps(encoded).encode()).hexdigest()


def _speedup(tokens: Any, seconds: Any, base_tokens: Any, base_seconds: Any) -> Any:
    return (tokens / seconds) / (base_tokens / base_seconds)


def summarize(arms: dict[str, Sequence[dict[str, Any]]], resamples: int = 10_000, seed: int = 0
              ) -> dict[str, dict[str, Any]]:
    """Per arm (``tag -> records``, in run order), the quantities of the module docstring against ``off-1``, which
    must be among them; every arm must cover off-1's prompts in the same order."""
    from chat_vpa.e0_report import paired_bootstrap
    from chat_vpa.e1_decode import first_difference

    if REFERENCE not in arms:
        raise ValueError(f"the reference arm {REFERENCE} is missing")
    ids = [r["id"] for r in arms[REFERENCE]]
    for tag, records in arms.items():
        if [r["id"] for r in records] != ids:
            raise ValueError(f"arm {tag} does not cover the prompts of {REFERENCE} in the same order")

    def columns(records: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
        return {"tokens": [float(len(r["tokens"])) for r in records],
                "decode_tokens": [float(max(len(r["tokens"]) - 1, 0)) for r in records],
                "decode_seconds": [float(r["decode_seconds"]) for r in records],
                "seconds": [float(r["seconds"]) for r in records]}

    reference = {r["id"]: r for r in arms[REFERENCE]}
    base = columns(arms[REFERENCE])
    if sum(base["decode_seconds"]) <= 0 or sum(base["decode_tokens"]) <= 0:
        raise ValueError(f"{REFERENCE} decoded nothing after its first tokens")
    out: dict[str, dict[str, Any]] = {}
    for tag, records in arms.items():
        c = columns(records)
        differences = {r["id"]: first_difference(r["tokens"], reference[r["id"]]["tokens"]) for r in records}
        summary: dict[str, Any] = {
            "prompts": len(records), "tokens": int(sum(c["tokens"])),
            "decode_tokens_per_second": sum(c["decode_tokens"]) / sum(c["decode_seconds"]),
            "end_to_end_tokens_per_second": sum(c["tokens"]) / sum(c["seconds"]),
            "median_prefill_seconds": median([float(r["prefill_seconds"]) for r in records]),
            "same_tokens": sum(d is None for d in differences.values()),
            "first_differences": {i: d for i, d in differences.items() if d is not None},
            "tokens_per_cycle": None, "draft_acceptance_rate": None, "acceptance_histogram": None}
        specs = [r["spec"] for r in records if r.get("spec")]
        if specs:
            width = max(len(s["acceptance_histogram"]) for s in specs)
            histogram = [sum(s["acceptance_histogram"][j] for s in specs if j < len(s["acceptance_histogram"]))
                         for j in range(width)]
            steps = sum(histogram)
            accepted = sum(j * count for j, count in enumerate(histogram))
            drafted = sum(int(s["num_draft_tokens"]) for s in specs)
            summary.update(tokens_per_cycle=1.0 + accepted / steps if steps else None,
                           draft_acceptance_rate=accepted / drafted if drafted else None,
                           acceptance_histogram=histogram)
        if tag != REFERENCE:
            for key, numerator, denominator in (("decode_speedup", "decode_tokens", "decode_seconds"),
                                                ("end_to_end_speedup", "tokens", "seconds")):
                point, low, high = paired_bootstrap([c[numerator], c[denominator], base[numerator],
                                                     base[denominator]], _speedup, resamples, seed)
                summary[key] = point
                summary[key + "_interval"] = [low, high]
        out[tag] = summary
    return out


def _markdown(summary: dict[str, dict[str, Any]]) -> str:
    lines = ["| arm | decode tok/s | speed-up (decode) | end-to-end tok/s | speed-up (end to end) | tokens per cycle "
             "| draft tokens accepted | same tokens as off-1 | prefill, s |", "|---|---|---|---|---|---|---|---|---|"]
    for tag, s in summary.items():
        def ratio(key: str, s: dict[str, Any] = s) -> str:
            if key not in s:
                return "1 (reference)"
            low, high = s[key + "_interval"]
            return f"{s[key]:.2f} [{low:.2f}, {high:.2f}]"
        cycle = "--" if s["tokens_per_cycle"] is None else f"{s['tokens_per_cycle']:.2f}"
        rate = "--" if s["draft_acceptance_rate"] is None else f"{100 * s['draft_acceptance_rate']:.0f}%"
        lines.append(f"| {tag} | {s['decode_tokens_per_second']:.1f} | {ratio('decode_speedup')} | "
                     f"{s['end_to_end_tokens_per_second']:.1f} | {ratio('end_to_end_speedup')} | {cycle} | {rate} | "
                     f"{s['same_tokens']} of {s['prompts']} | {s['median_prefill_seconds']:.3f} |")
    return "\n".join(lines)


def _import_vllm() -> tuple[Any, Any]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise SystemExit("vllm is not importable: run this with the interpreter of vLLM's environment "
                         "(runs/assets/venvs/vllm/bin/python; bash scripts/ncp/e1_setup.sh runtimes)") from None
    return LLM, SamplingParams


def _vllm_version() -> str:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(sys.modules.get("vllm"), "__version__", "of unknown version"))


def _environment() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"device": "cuda", "dtype": "bf16", "torch": torch.__version__,
                            "runtime": f"vLLM {_vllm_version()} (CUDA graphs and compilation as its defaults)",
                            "machine": platform.machine(), "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _start(args: argparse.Namespace, arm: VllmArm) -> tuple[Any, Any, float, dict[str, Any]]:
    LLM, SamplingParams = _import_vllm()
    kwargs = engine_kwargs(args.model, args.assistant, arm, args.max_model_len, args.gpu_memory_utilization)
    start = time.perf_counter()
    llm = LLM(**kwargs)
    startup = time.perf_counter() - start
    return llm, SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens), startup, kwargs


def _run(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    arm = parse_vllm_arm(args.arm)
    prompts = read_prompts(args.prompts, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    llm, params, startup, kwargs = _start(args, arm)
    print(f"{arm.name}: engine ready in {startup:.1f} s", flush=True)
    records = run_arm(llm, params, tokenizer, prompts, args.warmup, lambda text: print(text, flush=True))
    out = args.out / "arms" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "samples.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(out / "arm.json", {
        "arm": arm.name, "tag": args.tag, "environment": _environment(), "engine": kwargs,
        "startup_seconds": startup, "finished": time.time(), "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "prompts": {"file": str(args.prompts), "sha256": file_sha256(args.prompts), "count": len(prompts),
                    "token_ids_sha256": _ids_sha256(tokenizer, prompts)}})
    tokens = sum(len(r["tokens"]) for r in records)
    decode = sum(max(len(r["tokens"]) - 1, 0) for r in records) / max(sum(r["decode_seconds"] for r in records), 1e-9)
    print(f"{arm.name} ({args.tag}): {len(records)} prompts, {tokens} tokens, decode {decode:.1f} tok/s", flush=True)
    return 0


def _summarize(args: argparse.Namespace) -> int:
    arms_dir = args.out / "arms"
    if not arms_dir.is_dir():
        raise SystemExit(f"no arm has run under {arms_dir}")
    directories = sorted((d for d in arms_dir.iterdir() if (d / "samples.jsonl").is_file()),
                         key=lambda d: json.loads((d / "arm.json").read_text())["finished"])
    arms: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for directory in directories:
        meta[directory.name] = json.loads((directory / "arm.json").read_text())
        arms[directory.name] = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()
                                if line.strip()]
    if REFERENCE not in arms:
        raise SystemExit(f"no finished {REFERENCE} under {args.out / 'arms'}: nothing to compare with")
    hashes = {m["prompts"]["token_ids_sha256"] for m in meta.values()}
    if len(hashes) != 1:
        raise SystemExit("the arms were run on different prompt token ids: they cannot be compared")
    summary = summarize(arms, args.resamples, args.seed)
    reference = meta[REFERENCE]
    write_json(args.out / "summary.json", {
        "what": "greedy decoding of the development prompts in vLLM, one engine per arm, measured "
                "(docs/e1.md, Section 3.4)",
        "environment": reference["environment"], "reference": REFERENCE, "order": list(arms),
        "max_new_tokens": reference["max_new_tokens"], "warmup": reference["warmup"],
        "prompts": reference["prompts"], "engine": reference["engine"],
        "startup_seconds": {tag: m["startup_seconds"] for tag, m in meta.items()},
        "speculative_config": {tag: m["engine"].get("speculative_config") for tag, m in meta.items()},
        "arms": summary})
    print(_markdown(summary))
    return 0


def _smoke(args: argparse.Namespace) -> int:
    """One prompt through one engine: a check that the drafter loads and drafts, not a result."""
    from transformers import AutoTokenizer

    arm = parse_vllm_arm(args.arm)
    row = read_prompts(args.prompts, 1)[0]
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    print(f"  vllm {_vllm_version()}; arm {arm.name}; {args.max_new_tokens} tokens of the first prompt", flush=True)
    try:
        llm, params, startup, _ = _start(args, arm)
        record = run_arm(llm, params, tokenizer, [row], warmup=0)[0]
    except Exception as error:                                   # noqa: BLE001 - the message is the finding
        print(f"UNEXPECTED: vLLM could not decode with {arm.name}: {type(error).__name__}: "
              f"{' '.join(str(error).split())[:600]}")
        return 1
    print(f"  engine ready in {startup:.1f} s; {record['prompt_tokens']} prompt tokens, {len(record['tokens'])} new "
          f"tokens ({record['finish_reason']}); prefill {record['prefill_seconds']:.3f} s, decode "
          f"{record['decode_seconds']:.3f} s", flush=True)
    print(f"  text: {record['text'][:160]!r}")
    unexpected = []
    if not record["tokens"]:
        unexpected.append("the engine generated nothing")
    if arm.kind == "mtp":
        spec = record["spec"]
        if not spec or not spec.get("num_spec_steps"):
            unexpected.append("no acceptance counts: the drafter did not run, or vLLM did not report it")
        else:
            print(f"  drafting: {spec['num_spec_steps']} verification steps, {spec['num_accepted_draft_tokens']} of "
                  f"{spec['num_draft_tokens']} draft tokens accepted, {spec['mean_acceptance_length']:.2f} tokens per "
                  "cycle")
    for line in unexpected:
        print(f"UNEXPECTED: {line}")
    return 1 if unexpected else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E1 Step 0 in vLLM: greedy decoding with and without Google's drafter",
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "smoke"):
        q = sub.add_parser(name)
        q.add_argument("--model", type=Path, required=True, help="the instruction-tuned checkpoint (a directory)")
        q.add_argument("--assistant", type=Path, default=None, help="Google's drafter (mtp arms)")
        q.add_argument("--prompts", type=Path, required=True)
        q.add_argument("--max-model-len", type=int, default=4096)
        q.add_argument("--gpu-memory-utilization", type=float, default=0.85)
        if name == "run":
            q.add_argument("--arm", required=True, help="off or mtp:G")
            q.add_argument("--tag", required=True, help="the arm's directory under OUT/arms (off-1, mtp-4, ...)")
            q.add_argument("--out", type=Path, required=True)
            q.add_argument("--limit", type=int, default=None)
            q.add_argument("--max-new-tokens", type=int, default=256)
            q.add_argument("--warmup", type=int, default=2)
        else:
            q.add_argument("--arm", default="mtp:4")
            q.add_argument("--max-new-tokens", type=int, default=32)
    q = sub.add_parser("summarize")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--resamples", type=int, default=10_000)
    q.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "smoke":
        return _smoke(args)
    return _summarize(args)


if __name__ == "__main__":
    sys.exit(main())
