# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
block_evaluate.py -- Pass@k of the block sampler (Chat-VPA Algorithm 1) in the repository's harness.

    python -m chat_vpa.block_evaluate --model data2/models/gemma-4-E2B --checkpoint runs/stage0-a/last.pt \\
        --block-size 16 --k-steps 4 --orders exact --eval-dir data2/eval --out runs/e0-check/eval

The decoder is ``block_sampler.sample_blocks`` on a ``BlockDFM`` whose window carries the checkpoint's adapters:
a Stage 0 realizer's, or an E0 run's (``train_e0.py``; its registers and, unless ``--block-size`` is given, its
block size).
Everything else is the treatment of ``evaluate.py`` (rule 3 of CLAUDE.md): the prompts (HumanEval ``complete``,
MBPP ``write``), the token draw, the truncation rule, the tests, the seeds (one generator per problem, seeded
``seed * 100003 + index``), and the autoregressive baseline's budget of ``--canvas-len`` new tokens, here
ceil(canvas_len / B) blocks. A row stops at the first block that holds <eos>, or once the truncation rule has
found the program's end in the blocks finished so far.

Results go to ``OUT/<benchmark>/<tag>/{samples.jsonl, summary.json}`` in ``evaluate.py``'s format, so that
``report.py`` and ``analyze.py`` read them; the tag is ``block-B16-K4`` for the exact kernel, with ``-prefix`` or
``-confidence`` for the other orders, ``-exit0.05`` for an exit budget and ``-separate`` for a separate commit
pass. The summary also holds the passes per block that rows needed (K-bar), the seconds per block, and the pass
audit's outcome (``sample_blocks`` raises if a block did not cost exactly its passes).

With B = canvas_len and the exact kernel this is the whole-canvas realizer: a Stage 0 checkpoint must give the
samples of ``evaluate.py --decoder realizer`` at the same K and seed (the port check of Task D).

``--cache-check N``: for the first N problems, replay one sample's blocks through the fused passes and compare
the cache they build with the frozen backbone's causal keys and values of the same tokens (``prefill``), next to
two yardsticks: the same cache built by separate commit passes (rounding of a cache built block by block) and
the keys and values the adapters would give those tokens (a leak). The released weights are bfloat16, where one
unit in the last place of a value near 4 is 0.03; the float32 tests bound the fused commit at 1e-4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from chat_vpa.autoregressive import cut_at_eos
from chat_vpa.backbone import RealizerSpec, Stage0Realizer, _Recorder, left_pad, load_gemma4, prefill_masks
from chat_vpa.block_dfm import BlockDFM
from chat_vpa.block_sampler import BLOCK_ORDERS, BlockSamplerConfig, sample_blocks
from chat_vpa.data import HFTokenizer, Stage0Tokenizer
from chat_vpa.device import resolve_device
from chat_vpa.evaluate import autocast, load_tasks, prompt_text, score, write_results
from chat_vpa.evaluation.tasks import MBPP_SPLITS, EvalTask
from chat_vpa.postprocess import is_complete

__all__ = ["block_samples", "block_tag", "cache_check"]


def block_tag(cfg: BlockSamplerConfig, fused: bool = True) -> str:
    return (f"block-B{cfg.block_size}-K{cfg.num_steps}" + ("" if cfg.order == "exact" else f"-{cfg.order}")
            + (f"-exit{cfg.exit_budget:g}" if cfg.exit_budget > 0 else "") + ("" if fused else "-separate"))


def block_samples(model: BlockDFM, tokenizer: Stage0Tokenizer, tasks: Sequence[EvalTask], benchmark: str,
                  cfg: BlockSamplerConfig, canvas_len: int, num_samples: int, batch_size: int, seed: int,
                  device: torch.device, progress: bool = False, fused: bool = True, adapters: bool = True,
                  causal: bool = False, audit: bool = True) -> tuple[list[list[str]], dict]:
    """``num_samples`` responses per task from the block sampler, decoded.

    The budget is ceil(``canvas_len`` / B) blocks. Seeds and batching are those of ``evaluate.realizer_samples``.
    """
    texts: list[list[str]] = []
    forbidden = [tokenizer.pad_id, tokenizer.bos_id]
    kind = "complete" if benchmark == "humaneval" else "write"
    max_blocks = math.ceil(canvas_len / cfg.block_size)

    def row_done(tokens: list[int]) -> bool:
        return is_complete(tokenizer.decode(tokens), kind)

    seconds, tokens_out = 0.0, 0
    batch_blocks = batch_passes = 0
    row_blocks: list[int] = []
    row_passes: list[int] = []
    stops: dict[str, int] = {"eos": 0, "done": 0, "budget": 0}
    for index, task in enumerate(tasks):
        ids = [tokenizer.bos_id, *tokenizer.encode_batch([prompt_text(task, benchmark)])[0]]
        generator = torch.Generator(device=device).manual_seed(seed * 100_003 + index)
        start = time.perf_counter()
        rows: list[str] = []
        with autocast(device):
            cache = model.prefill([ids], device)
            while len(rows) < num_samples:
                n = min(batch_size, num_samples - len(rows))
                anchor = torch.full((n,), ids[-1], dtype=torch.long, device=device)
                result = sample_blocks(model, cache.expand(n), anchor, cfg, max_blocks, forbidden, generator,
                                       stop_id=tokenizer.eos_id, row_done=row_done, causal=causal,
                                       adapters=adapters, fused=fused, audit=audit)
                batch_blocks += len(result.passes)
                batch_passes += result.network_passes
                row_blocks += result.blocks
                row_passes += [sum(p) for p in result.row_passes]
                for stop in result.stops:
                    stops[stop] += 1
                for row in result.tokens:
                    kept = cut_at_eos(row, [tokenizer.eos_id])
                    tokens_out += len(kept)
                    rows.append(tokenizer.decode(kept))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds += time.perf_counter() - start
        texts.append(rows)
        if progress and (index + 1) % 20 == 0:
            print(f"  generated {index + 1}/{len(tasks)} problems", file=sys.stderr, flush=True)
    total = max(len(tasks) * num_samples, 1)
    info = {"decoder": "block", **cfg.to_dict(), "registers": model.num_registers, "fused_commit": fused,
            "adapters": adapters, "causal_window": causal, "canvas_len": canvas_len, "max_blocks": max_blocks,
            "mean_blocks": round(sum(row_blocks) / total, 3),
            # K-bar: passes a row's block needed until it had no [MASK], averaged over the blocks of all rows
            "mean_passes_per_block": round(sum(row_passes) / max(sum(row_blocks), 1), 3),
            # prefill + the passes of its own blocks, per response
            "mean_sequential_depth": round(1 + sum(row_passes) / total, 2),
            "stops": {k: round(v / total, 4) for k, v in stops.items()},
            "generation_seconds": round(seconds, 3),
            # wall time of a batch of rows per block of that batch (prefill and decoding included)
            "seconds_per_block": round(seconds / max(batch_blocks, 1), 4),
            "seconds_per_pass": round(seconds / max(batch_passes + len(tasks), 1), 4),
            "mean_tokens": round(tokens_out / total, 2),
            "pass_audit": "passed" if audit else "off"}
    return texts, info


def _largest_difference(a_keys: dict[int, torch.Tensor], a_values: dict[int, torch.Tensor],
                        b_keys: dict[int, torch.Tensor], b_values: dict[int, torch.Tensor]) -> float:
    worst = 0.0
    for layer, keys in b_keys.items():
        worst = max(worst, float((a_keys[layer].float() - keys.float()).abs().max()),
                    float((a_values[layer].float() - b_values[layer].float()).abs().max()))
    return worst


@torch.no_grad()
def cache_check(model: BlockDFM, prompt: Sequence[int], response: Sequence[int], block_size: int,
                device: torch.device) -> dict[str, float]:
    """Replay ``response`` block by block and compare the cache of the fused commits with the frozen backbone's
    causal keys and values of the committed tokens, computed in one ``prefill``.

    Returns the largest absolute differences over all KV-storing blocks, next to two yardsticks:

        fused_commit      the cache built by fused passes (adapters on, all slots masked) against the prefill
        separate_commit   the cache built by separate commit passes, in which no adapter exists, against the
                          prefill: what building a cache block by block costs in rounding on this device
        adapters_on       the committed tokens read causally with the adapters and the modulation acting on
                          them, against the prefill: what a leak into the commit rows would look like

    and the root mean square of the reference and the number of committed positions. The fused commit is pure
    if its difference is of the order of ``separate_commit`` and far below ``adapters_on``.
    """
    realizer = model.realizer
    blocks = len(response) // block_size
    if blocks < 2:
        return {"blocks": float(blocks)}
    zeros = torch.zeros(1, device=device)
    with autocast(device):
        fused, separate = model.prefill([list(prompt)], device), model.prefill([list(prompt)], device)
        anchor = torch.tensor([prompt[-1]], dtype=torch.long, device=device)
        slots = torch.full((1, block_size), realizer.spec.mask_id, dtype=torch.long, device=device)
        commit = None
        for m in range(blocks):
            model.fused_pass(fused, commit, anchor, slots, torch.ones_like(slots, dtype=torch.bool), zeros,
                             zeros + 0.25)
            if commit is not None:
                model.commit_pass(separate, commit)
            block = torch.tensor([list(response[m * block_size:(m + 1) * block_size])], dtype=torch.long, device=device)
            commit = torch.cat([anchor[:, None], block[:, : block_size - 1]], dim=1)
            anchor = block[:, -1]
        committed = [*prompt, *response[: (blocks - 1) * block_size - 1]]
        ids, valid, pos = left_pad([committed], realizer.pad_id, device)
        reference = realizer.prefill(ids, valid, pos)
        embeds, per_layer = realizer._embed(ids, None)
        adapted = _Recorder()
        realizer._run(embeds, per_layer, pos, prefill_masks(pos, valid, realizer.window), adapted,
                      realizer.num_kv_layers, checkpointing=False, active=True,
                      modulation=realizer.modulation(zeros, zeros + 0.25))
    if int(fused.lengths) != len(committed) or int(separate.lengths) != len(committed):
        raise RuntimeError("the replayed cache does not hold the committed tokens")
    square = sum(float(k.float().pow(2).sum()) for k in reference.keys.values())
    count = sum(k.numel() for k in reference.keys.values())
    return {"blocks": float(blocks), "committed_positions": float(len(committed)),
            "fused_commit": _largest_difference(fused.keys, fused.values, reference.keys, reference.values),
            "separate_commit": _largest_difference(separate.keys, separate.values, reference.keys, reference.values),
            "adapters_on": _largest_difference(adapted.keys, adapted.values, reference.keys, reference.values),
            "reference_rms": math.sqrt(square / max(count, 1))}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pass@k of the block sampler", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="google/gemma-4-E2B", help="Hub id or local directory of the backbone")
    p.add_argument("--revision", default=None)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="the adapters: a Stage 0 checkpoint or an E0 run's (train_e0.py)")
    p.add_argument("--benchmarks", nargs="+", choices=("humaneval", "mbpp"), default=["mbpp"])
    p.add_argument("--eval-dir", type=Path, required=True, help="directory with HumanEval.jsonl.gz and mbpp.jsonl")
    p.add_argument("--mbpp-split", choices=sorted(MBPP_SPLITS), default="validation")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--block-size", type=int, default=None, help="B (default: an E0 run's own, else 16)")
    p.add_argument("--k-steps", type=int, nargs="+", default=[4], help="passes per block K")
    p.add_argument("--orders", nargs="+", choices=BLOCK_ORDERS, default=["exact"],
                   help="exact: the exact jump kernel (the primary sampler); prefix; confidence")
    p.add_argument("--exit-budget", type=float, default=0.0, help="delta of the early exit; 0 is off")
    p.add_argument("--schedule", default=None, help="kappa schedule (default: the checkpoint's)")
    p.add_argument("--canvas-len", type=int, default=256, help="token budget: ceil(canvas_len / B) blocks")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--k", type=int, nargs="+", default=[1])
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--separate-commit", action="store_true", help="ablation A10: commit in a pass of its own")
    p.add_argument("--no-adapters", action="store_true", help="LoRA and modulation off in the window")
    p.add_argument("--causal-window", action="store_true",
                   help="with --no-adapters, B = 1 and K = 1: the backbone's own autoregressive decoding")
    p.add_argument("--cache-check", type=int, default=0, metavar="N",
                   help="compare the fused commits' cache with the frozen backbone's on the first N problems")
    p.add_argument("--tag-suffix", default="")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--exec-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    model, hf_tokenizer = load_gemma4(args.model, revision=args.revision, token=os.environ.get("HF_TOKEN") or None)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    stage = ckpt.get("stage", "stage0")
    if stage not in ("stage0", "e0"):
        raise SystemExit(f"{args.checkpoint} is a {stage!r} checkpoint; this evaluator reads a Stage 0 realizer or "
                         "an E0 run")
    tokenizer = HFTokenizer(hf_tokenizer)
    spec = RealizerSpec.from_dict(ckpt["spec"])
    if spec.mask_id != tokenizer.mask_id:
        raise SystemExit("the checkpoint's [MASK] id is not this tokenizer's")
    realizer = Stage0Realizer.from_pretrained_model(model, spec)
    realizer.load_trainable_state(ckpt["trainable"])
    for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(model.model, tower):
            setattr(model.model, tower, None)                       # text only
    num_registers = int(ckpt["config"].get("num_registers", 0)) if stage == "e0" else 0
    block_model = BlockDFM(realizer, num_registers=num_registers)
    if num_registers:
        with torch.no_grad():
            block_model.registers.copy_(ckpt["registers"].to(block_model.registers.dtype))
    block_model.to(device).eval()
    block_size = args.block_size or (int(ckpt["config"]["block_size"]) if stage == "e0" else 16)
    schedule = args.schedule or ckpt["config"]["schedule"]
    common: dict[str, Any] = {"model": args.model, "revision": args.revision, "seed": args.seed,
                              "device": str(device), "checkpoint": str(args.checkpoint), "step": ckpt["step"],
                              "checkpoint_stage": stage}
    if stage == "e0":
        common.update(arm=ckpt["config"]["arm"], train_block_size=int(ckpt["config"]["block_size"]),
                      registers=num_registers)
    configs = [BlockSamplerConfig(block_size, k, schedule, order, args.exit_budget, args.temperature, args.top_p,
                                  args.top_k) for order in args.orders for k in args.k_steps]
    fused = not args.separate_commit

    for benchmark in args.benchmarks:
        tasks = load_tasks(benchmark, args.eval_dir, args.mbpp_split, args.limit)
        if not tasks:
            raise SystemExit(f"no {benchmark} problem under {args.eval_dir}")
        for cfg in configs:
            tag = (block_tag(cfg, fused) + ("-no_adapters" if args.no_adapters else "")
                   + ("-causal" if args.causal_window else "") + args.tag_suffix)
            out = args.out / benchmark / tag
            if (out / "summary.json").exists() and not args.force:
                previous = json.loads((out / "summary.json").read_text())
                if previous.get("step") == common["step"] and previous.get("problems") == len(tasks):
                    print(f"{out}: done, skipping (--force to redo)", file=sys.stderr)
                    continue
            print(f"== {benchmark} / {tag}: {len(tasks)} problems x {args.num_samples} samples", file=sys.stderr,
                  flush=True)
            texts, info = block_samples(block_model, tokenizer, tasks, benchmark, cfg, args.canvas_len,
                                        args.num_samples, args.batch_size, args.seed, device, progress=True,
                                        fused=fused, adapters=not args.no_adapters, causal=args.causal_window)
            if args.cache_check:
                checks = []
                for task, group in zip(tasks[: args.cache_check], texts, strict=False):
                    prompt = [tokenizer.bos_id, *tokenizer.encode_batch([prompt_text(task, benchmark)])[0]]
                    response = tokenizer.encode_batch([group[0]])[0]
                    checks.append({"task_id": task.task_id,
                                   **cache_check(block_model, prompt, response, cfg.block_size, device)})
                info["cache_check"] = checks
            summary, rows = score(tasks, benchmark, texts, args.k, False, args.timeout, args.exec_workers)
            summary = {**summary, "split": args.mbpp_split if benchmark == "mbpp" else "test", "tag": tag, **info,
                       **common}
            write_results(out, summary, rows)
            print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
