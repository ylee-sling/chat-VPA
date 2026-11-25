# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_litert.py -- E1's Step 0 in LiteRT-LM (docs/e1.md, Section 3.4): Google's on-device runtime decoding the
development prompts with its own build of Gemma 4 E2B-it, with its speculative decoding off and on.

    ASSETS/venvs/litert/bin/python -m chat_vpa.e1_litert --model ASSETS/litert/gemma-4-E2B-it.litertlm \\
        --prompts ASSETS/prompts/dev.jsonl --backend cpu --threads 8 --out runs/e1-step0/litert/cpu

This is the runtime the product would ship in, and the one in which Google reports the drafter's speed-up, so its
``off`` against ``on`` is the baseline that E1 has to beat on a given machine. It runs in LiteRT-LM's own
environment (``litert-lm`` from PyPI; no torch), through the package's Python interface.

Modes (``SpeculativeDecodingMode``; lenience 0, the lossless setting, is passed explicitly):

    off         the model alone (the baseline)
    on          speculate at every decoding step, with the drafter packaged in the model file
    adaptive    the runtime speculates only while it measures that to be faster

One engine per mode is created up front and stays loaded; a mode whose engine cannot be created (a backend the
machine lacks, a file without a drafter) is recorded as failed with the runtime's message and the others are
measured. Per prompt, the modes run back to back, each in a fresh conversation: one user turn in the model's own
template, thinking off where the model has it, at most ``--max-new-tokens`` tokens, and the sampler at top-k 1
(the argmax, whatever the temperature). Recorded per prompt and mode: the text, the wall-clock time of the call,
and the runtime's own benchmark record of the turn (tokens decoded, decode tokens per second, time to the first
token, prompt tokens). From these:

    decode tok/s            tokens decoded / the runtime's decode time, summed over prompts (the runtime's timer)
    end-to-end tok/s        tokens decoded / the wall-clock time of the call (prompt processing included)
    same text               the mode's response equals ``off``'s for the same prompt, character for character;
                            with the position of the first differing character where it does not
    speed-up                the mode's tok/s over ``off``'s, with a 95% paired bootstrap over prompts
                            (10,000 resamples, seed 0)

What this is not. The model file is Google's build for this runtime (its own quantisation), not the safetensors
checkpoint that ``e1_decode.py`` runs, so a text here need not equal that script's text for the same prompt:
exactness is read inside one runtime, between ``off`` and the other modes. And "same text" is weaker than "same
tokens" only in principle (two token sequences with one text); the runtime's token counts are recorded beside it.
Everything is *measured* on this machine, backend and thread count, and the summary says which.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from chat_vpa.e1_common import file_sha256, median, read_prompts, write_json

__all__ = ["MODES", "first_text_difference", "message_text", "respond", "run", "summarize"]

MODES = ("off", "on", "adaptive")


def message_text(message: Any) -> str:
    """The text of a LiteRT-LM message (a mapping with ``content``: a string or a list of typed parts)."""
    content = message["content"] if "content" in message else ""
    if isinstance(content, str):
        return content
    return "".join(str(part.get("text", "")) for part in content if part.get("type", "text") == "text")


def first_text_difference(a: str, b: str) -> int | None:
    """The first position at which two texts differ; None if they are equal."""
    if a == b:
        return None
    for position, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return position
    return min(len(a), len(b))


def respond(engine: Any, prompt: str, max_new_tokens: int, conversation_args: dict[str, Any]) -> dict[str, Any]:
    """One user turn in a fresh conversation of ``engine``: the response and what the runtime measured."""
    conversation = engine.create_conversation(max_output_tokens=max_new_tokens, **conversation_args)
    try:
        start = time.perf_counter()
        message = conversation.send_message(prompt)
        seconds = time.perf_counter() - start
        info = conversation.get_benchmark_info()
    finally:
        conversation.close()
    return {"text": message_text(message), "seconds": seconds,
            "decode_tokens": int(info.last_decode_token_count),
            "decode_tokens_per_second": float(info.last_decode_tokens_per_second),
            "first_token_seconds": float(info.time_to_first_token_in_second),
            "prompt_tokens": int(info.last_prefill_token_count)}


def run(engines: dict[str, Any], prompts: Sequence[dict[str, str]], max_new_tokens: int, warmup: int,
        conversation_args: dict[str, Any], progress: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """Every prompt through every engine, the modes back to back; the first ``warmup`` prompts are decoded by
    every engine first and thrown away."""
    for row in prompts[:warmup]:
        for engine in engines.values():
            respond(engine, row["prompt"], max_new_tokens, conversation_args)
    records: list[dict[str, Any]] = []
    for number, row in enumerate(prompts, start=1):
        for mode, engine in engines.items():
            records.append({"id": row["id"], "mode": mode,
                            **respond(engine, row["prompt"], max_new_tokens, conversation_args)})
        if progress is not None:
            progress(f"{number} of {len(prompts)} prompts")
    return records


def _speedup(tokens: Any, seconds: Any, base_tokens: Any, base_seconds: Any) -> Any:
    return (tokens / seconds) / (base_tokens / base_seconds)


def summarize(records: Sequence[dict[str, Any]], modes: Sequence[str], resamples: int = 10_000, seed: int = 0
              ) -> dict[str, dict[str, Any]]:
    """Per mode, the quantities of the module docstring; ``off`` must be among the modes and every prompt must
    have every mode."""
    from chat_vpa.e0_report import paired_bootstrap

    by_mode: dict[str, dict[str, dict[str, Any]]] = {mode: {} for mode in modes}
    for record in records:
        by_mode[record["mode"]][record["id"]] = record
    if "off" not in by_mode:
        raise ValueError("the 'off' mode is the reference of every comparison")
    ids = list(by_mode["off"])
    for mode in modes:
        if list(by_mode[mode]) != ids:
            raise ValueError(f"mode {mode} does not cover the prompts of 'off' in the same order")

    def columns(mode: str) -> dict[str, list[float]]:
        rows = [by_mode[mode][i] for i in ids]
        if any(r["decode_tokens_per_second"] <= 0 or r["decode_tokens"] <= 0 for r in rows):
            raise ValueError(f"mode {mode}: a response without decoded tokens or without a decode rate")
        return {"tokens": [float(r["decode_tokens"]) for r in rows],
                "decode_seconds": [r["decode_tokens"] / r["decode_tokens_per_second"] for r in rows],
                "seconds": [float(r["seconds"]) for r in rows],
                "first": [float(r["first_token_seconds"]) for r in rows]}

    base = columns("off")
    out: dict[str, dict[str, Any]] = {}
    for mode in modes:
        c = columns(mode)
        differences = [first_text_difference(by_mode[mode][i]["text"], by_mode["off"][i]["text"]) for i in ids]
        same_count = sum(by_mode[mode][i]["decode_tokens"] == by_mode["off"][i]["decode_tokens"] for i in ids)
        summary: dict[str, Any] = {
            "prompts": len(ids), "decode_tokens": int(sum(c["tokens"])),
            "decode_tokens_per_second": sum(c["tokens"]) / sum(c["decode_seconds"]),
            "end_to_end_tokens_per_second": sum(c["tokens"]) / sum(c["seconds"]),
            "median_seconds_to_first_token": median(c["first"]),
            "same_text": sum(d is None for d in differences),
            "same_text_share": sum(d is None for d in differences) / len(ids),
            "same_token_count": same_count,
            "first_differences": {i: d for i, d in zip(ids, differences, strict=True) if d is not None}}
        if mode != "off":
            for key, denom in (("decode_speedup", "decode_seconds"), ("end_to_end_speedup", "seconds")):
                point, low, high = paired_bootstrap([c["tokens"], c[denom], base["tokens"], base[denom]], _speedup,
                                                    resamples, seed)
                summary[key] = point
                summary[key + "_interval"] = [low, high]
        out[mode] = summary
    return out


def _markdown(summary: dict[str, dict[str, Any]]) -> str:
    lines = ["| mode | decode tok/s (the runtime's timer) | speed-up (decode) | end-to-end tok/s | speed-up (end to end) "
             "| same text as off | first token after, s |", "|---|---|---|---|---|---|---|"]
    for mode, s in summary.items():
        if mode == "off":
            decode = whole = "1 (reference)"
        else:
            decode = (f"{s['decode_speedup']:.2f} [{s['decode_speedup_interval'][0]:.2f}, "
                      f"{s['decode_speedup_interval'][1]:.2f}]")
            whole = (f"{s['end_to_end_speedup']:.2f} [{s['end_to_end_speedup_interval'][0]:.2f}, "
                     f"{s['end_to_end_speedup_interval'][1]:.2f}]")
        lines.append(f"| {mode} | {s['decode_tokens_per_second']:.1f} | {decode} | "
                     f"{s['end_to_end_tokens_per_second']:.1f} | {whole} | {s['same_text']} of {s['prompts']} | "
                     f"{s['median_seconds_to_first_token']:.3f} |")
    return "\n".join(lines)


def _runtime_version() -> str:
    try:
        return importlib.metadata.version("litert-lm")
    except importlib.metadata.PackageNotFoundError:
        return "of unknown version"


def _open_engines(model: Path, modes: Sequence[str], backend: str, threads: int | None, cache_dir: Path
                  ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any]]:
    """The engines that could be created, the modes that could not (with the runtime's message), the arguments
    of every conversation, and what the model file says about itself."""
    try:
        import litert_lm
    except ImportError:
        raise SystemExit("litert_lm is not importable: run this with the interpreter of LiteRT-LM's environment "
                         "(runs/assets/venvs/litert/bin/python; bash scripts/ncp/e1_setup.sh runtimes)") from None
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
    info = litert_lm.ModelInfo(str(model))
    try:
        capability = info.llm
        facts = {"speculative_decoding_support": bool(capability.has_speculative_decoding_support()),
                 "thinking_support": bool(capability.supports_thinking()),
                 "max_context_tokens": info.max_context_tokens,
                 "default_sampler": str(capability.default_sampler_params)}
    finally:
        info.close()
    device = litert_lm.Backend.GPU() if backend == "gpu" else litert_lm.Backend.CPU(thread_count=threads)
    named = {"off": litert_lm.SpeculativeDecodingMode.OFF, "on": litert_lm.SpeculativeDecodingMode.ON,
             "adaptive": litert_lm.SpeculativeDecodingMode.ADAPTIVE}
    engines: dict[str, Any] = {}
    failed: dict[str, str] = {}
    for mode in modes:
        try:
            engines[mode] = litert_lm.Engine(
                str(model), backend=device, cache_dir=str(cache_dir), enable_benchmark=True,
                speculative_decoding_config=litert_lm.SpeculativeDecodingConfig(mode=named[mode], lenience=0.0))
        except Exception as error:                              # noqa: BLE001 - the runtime's message is the finding
            failed[mode] = f"{type(error).__name__}: {' '.join(str(error).split())[:600]}"
    conversation_args: dict[str, Any] = {"sampler_config": litert_lm.SamplerConfig(top_k=1, temperature=1.0, seed=0)}
    if facts["thinking_support"]:
        conversation_args["thinking_config"] = litert_lm.ThinkingConfig(enable_thinking=False)
    return engines, failed, conversation_args, facts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E1 Step 0 in LiteRT-LM: speculative decoding off and on",
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="the .litertlm file")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--backend", default="cpu", choices=("cpu", "gpu"))
    p.add_argument("--threads", type=int, default=None, help="CPU threads (the cpu backend)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--cache-dir", type=Path, default=None, help="compiled artifacts [beside the model, in cache/]")
    args = p.parse_args(argv)
    if "off" not in args.modes:
        p.error("--modes must include off, the reference")
    modes = [m for m in MODES if m in args.modes]
    prompts = read_prompts(args.prompts, args.limit)
    cache_dir = args.cache_dir or args.model.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    engines, failed, conversation_args, facts = _open_engines(args.model, modes, args.backend, args.threads, cache_dir)
    for mode, reason in failed.items():
        print(f"mode {mode} could not be started: {reason}", flush=True)
    if "off" not in engines:
        raise SystemExit("the 'off' engine could not be created on this backend: nothing to measure")
    try:
        conversation = engines["off"].create_conversation(**conversation_args)
        try:
            rendered = conversation.render_message_to_string({"role": "user", "content": prompts[0]["prompt"]})
        finally:
            conversation.close()
        records = run(engines, prompts, args.max_new_tokens, args.warmup, conversation_args,
                      lambda text: print(text, flush=True))
    finally:
        for engine in engines.values():
            engine.close()

    summary = summarize(records, list(engines))
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "samples.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(args.out / "summary.json", {
        "what": "greedy decoding of the development prompts in LiteRT-LM, measured (docs/e1.md, Section 3.4)",
        "environment": {"runtime": f"LiteRT-LM {_runtime_version()}", "backend": args.backend,
                        "cpu_threads": args.threads, "cpus": os.cpu_count(), "machine": platform.machine(),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")},
        "model": {"file": args.model.name, "bytes": args.model.stat().st_size, "sha256": file_sha256(args.model),
                  **facts},
        "prompts": {"file": str(args.prompts), "sha256": file_sha256(args.prompts), "count": len(prompts)},
        "max_new_tokens": args.max_new_tokens, "warmup": args.warmup, "sampler": "top-k 1 (argmax), seed 0",
        "lenience": 0.0, "thinking": "off" if facts["thinking_support"] else "not supported by the file",
        "first_prompt_as_rendered": rendered, "modes": summary, "failed": failed})
    print(_markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
