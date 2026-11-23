# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
rollout_throughput.py -- how many autoregressive rollouts are affordable (the test before E0's data is fixed).

    python -m chat_vpa.rollout_throughput --model data2/models/gemma-4-E2B --corpus data2/corpus \\
        --out runs/e0-throughput

E0's rollout arm needs one rollout of the frozen backbone per training prompt (``rollouts.py``). This measures,
on the first ``--num-prompts`` training examples in ``CorpusStream`` order (both prompt formats as drawn, the
stream of the predecessor's ``runs/stage0-a``; with ``--kind`` the first examples of that format only, and the
number of stream examples read to find them is recorded), for each batch size in turn:

    wall time, rollouts and generated tokens per second, peak memory
    how rows ended: the truncation rule, <eos>, the budget (dropped) or a program too long for the canvas
    (dropped), over all prompts and per prompt format
    the length of the kept programs after truncation

and stops at the first batch size that runs out of memory. Prompts are sorted by length before batching unless
``--no-sort`` (less padding; recorded). For the fastest batch size it projects the hours of 100,000 and 320,000
rollouts, and measures on its kept rollouts what Corollary 1 asks for: the entropy of the decoding distribution
along each rollout (``rollouts.teacher_forced_stats``), next to the backbone's negative log-likelihood at
temperature 1 of the rollouts and of the human references of the same prompts.

Writes ``OUT/throughput.json`` (after every batch size, so that a later failure keeps what was measured),
``OUT/examples.txt`` (the first ``--examples`` prompts with their reference and their rollout) and
``OUT/rollouts.jsonl`` (every rollout of the fastest batch size: prompt, reference, the truncated program and how
it stopped; for ``inspect_rollouts.py``), the last two before the teacher-forced pass, and nothing else: no
rollout is stored as a training target and no generated code is executed. ``--show OUT`` prints the summary of
a finished measurement.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from chat_vpa.backbone import RealizerSpec, Stage0Realizer, load_gemma4
from chat_vpa.data import CorpusStream, Example, HFTokenizer
from chat_vpa.device import resolve_device
from chat_vpa.evaluate import autocast
from chat_vpa.rollouts import STOPS, Rollout, rollouts_for, teacher_forced_stats

__all__ = ["measure_batch_size", "projection", "summarize_rollouts", "summary_text"]

PROJECTIONS: tuple[int, ...] = (100_000, 320_000)


def _quantiles(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {"mean": round(statistics.fmean(ordered), 2), "median": float(statistics.median(ordered)),
            "p95": float(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]), "max": float(ordered[-1])}


def summarize_rollouts(rollouts: Sequence[Rollout]) -> dict[str, Any]:
    """Shares of ``STOPS`` (over all rollouts and per prompt format), the kept share and the kept programs'
    lengths in tokens (without the <eos>)."""
    total = len(rollouts)
    stops = Counter(r.stop for r in rollouts)
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({r.kind for r in rollouts}):
        rows = [r for r in rollouts if r.kind == kind]
        counts = Counter(r.stop for r in rows)
        by_kind[kind] = {"rollouts": len(rows), **{s: round(counts[s] / len(rows), 4) for s in STOPS}}
    kept = [r for r in rollouts if r.kept]
    lengths = [len(r.canvas_ids) - 1 for r in kept if r.canvas_ids is not None]
    return {"rollouts": total, "stops": {s: round(stops[s] / total, 4) for s in STOPS}, "stops_by_kind": by_kind,
            "kept": len(kept), "kept_share": round(len(kept) / total, 4), "program_tokens": _quantiles(lengths),
            "empty_programs": sum(1 for r in kept if not r.text.strip())}


def measure_batch_size(model: Any, hf_tokenizer: Any, tokenizer: HFTokenizer, examples: Sequence[Example],
                       order: Sequence[int], batch_size: int, args: argparse.Namespace, device: torch.device
                       ) -> tuple[dict[str, Any], list[Rollout]]:
    """All prompts at one batch size. Returns the measurements and the rollouts in the examples' own order."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    rollouts: list[Rollout | None] = [None] * len(examples)
    seeds, steps, generated = [], 0, 0
    start = time.perf_counter()
    for index, first in enumerate(range(0, len(order), batch_size)):
        rows = list(order[first:first + batch_size])
        seed = args.seed * 100_003 + batch_size * 1_009 + index
        batch, batch_steps = rollouts_for(model, hf_tokenizer, tokenizer, [examples[i] for i in rows],
                                          args.canvas_len, args.max_new_tokens, args.temperature, args.top_p,
                                          args.top_k, seed, args.check_every)
        for i, rollout in zip(rows, batch, strict=True):
            rollouts[i] = rollout
        seeds.append(seed)
        steps += batch_steps * len(rows)
        generated += sum(r.generated for r in batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - start
    done = [r for r in rollouts if r is not None]
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    entry = {"batch_size": batch_size, "batches": len(seeds), "seeds": seeds, "wall_seconds": round(wall, 2),
             "rollouts_per_second": round(len(done) / wall, 4), "generated_tokens": generated,
             "generated_tokens_per_second": round(generated / wall, 2),
             # row-steps the batches ran, finished rows included: generated_tokens / row_steps is the share of
             # the decoding work that produced a token
             "row_steps": steps, "useful_share_of_row_steps": round(generated / max(steps, 1), 4),
             "peak_memory_gib": round(peak, 2), **summarize_rollouts(done)}
    return entry, done


def projection(entry: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Hours for the rollouts of n prompts at this entry's rate, and for n kept rollouts."""
    rate, kept = entry["rollouts_per_second"], entry["kept_share"]
    out = {}
    for n in PROJECTIONS:
        out[str(n)] = {"hours_for_n_prompts": round(n / rate / 3600, 2),
                       "hours_for_n_kept_rollouts": round(n / (rate * kept) / 3600, 2) if kept > 0 else float("inf")}
    return out


def _pooled(stats: Sequence[dict[str, float]]) -> dict[str, float]:
    tokens = sum(s["tokens"] for s in stats)
    programs = len(stats)
    if not programs or not tokens:
        return {"programs": programs, "tokens": tokens}
    return {"programs": programs, "tokens": tokens,
            "entropy_per_token": round(sum(s["entropy"] for s in stats) / tokens, 5),
            "entropy_per_program": round(sum(s["entropy"] for s in stats) / programs, 4),
            "nll_per_token": round(sum(s["nll"] for s in stats) / tokens, 5),
            "nll_per_program": round(sum(s["nll"] for s in stats) / programs, 4),
            "eos_entropy": round(sum(s["eos_entropy"] for s in stats) / programs, 5),
            "eos_nll": round(sum(s["eos_nll"] for s in stats) / programs, 5)}


def _kind_note(result: dict[str, Any]) -> str:
    """For a run with ``--kind``: which format, and its share of the stream examples read to find the prompts."""
    kind = result["settings"].get("kind", "both")              # files of earlier runs have no such key
    if kind == "both":
        return ""
    count, read = result["prompts"]["count"], result["prompts"].get("stream_examples_read", 0)
    share = f" ({100 * count / read:.1f}%)" if read else ""
    return f"; only '{kind}' prompts: the first {count} of {read} stream examples{share}"


def summary_text(result: dict[str, Any]) -> str:
    """The measurement in a few lines (what ``--show`` prints)."""
    s = result["settings"]
    lines = [f"rollout throughput: {result['prompts']['count']} prompts ({result['prompts']['kinds']}), "
             f"{result['device']}, budget {s['max_new_tokens']}, T {s['temperature']}, top-k {s['top_k']}, "
             f"top-p {s['top_p']}, sorted by prompt length: {s['sorted_by_prompt_length']}"
             + _kind_note(result),
             "batch  wall_s  rollouts/s  tokens/s  peak_GiB  kept    trunc   eos     budget  too_long  "
             "prog_tokens(mean/median)"]
    for e in result["batch_sizes"]:
        if e.get("out_of_memory"):
            lines.append(f"{e['batch_size']:>5}  out of memory: {e['error']}")
            continue
        st, pt = e["stops"], e["program_tokens"]
        lines.append(f"{e['batch_size']:>5}  {e['wall_seconds']:>6.1f}  {e['rollouts_per_second']:>10.3f}  "
                     f"{e['generated_tokens_per_second']:>8.1f}  {e['peak_memory_gib']:>8.2f}  "
                     f"{e['kept_share']:.3f}   {st['truncation']:.3f}   {st['eos']:.3f}   {st['budget']:.3f}   "
                     f"{st['too_long']:.3f}     {pt.get('mean', float('nan'))}/{pt.get('median', float('nan'))}")
    best = result.get("best")
    if best:
        lines.append(f"best batch size: {best['batch_size']} ({best['rollouts_per_second']} rollouts/s, kept share "
                     f"{best['kept_share']})")
        for kind, row in best["stops_by_kind"].items():
            lines.append(f"  {kind}: {row['rollouts']} prompts, truncation {row['truncation']}, eos {row['eos']}, "
                         f"budget {row['budget']}, too_long {row['too_long']}")
        for n, hours in best["projected"].items():
            lines.append(f"  projected (measured rate, this batch size): {int(n):,} prompts in "
                         f"{hours['hours_for_n_prompts']} h; {int(n):,} kept rollouts in "
                         f"{hours['hours_for_n_kept_rollouts']} h")
    entropy = result.get("entropy")
    if entropy:
        r, h = entropy["rollouts"], entropy["references"]
        lines.append(f"entropy of the decoding distribution along the rollouts ({r['programs']} programs, "
                     f"{int(r['tokens'])} tokens): {r.get('entropy_per_token')} nats per token, "
                     f"{r.get('entropy_per_program')} per program")
        lines.append(f"backbone NLL at temperature 1: rollouts {r.get('nll_per_token')} nats per token "
                     f"({r.get('nll_per_program')} per program); human references of the same prompts "
                     f"{h.get('nll_per_token')} per token ({h.get('nll_per_program')} per program, "
                     f"{int(h['tokens'])} tokens)")
    return "\n".join(lines)


def take_examples(stream: CorpusStream, count: int, kind: str | None, patience: int = 200
                  ) -> tuple[list[Example], int]:
    """The next ``count`` examples of ``stream``, or of one format only; and how many examples were read."""
    if kind is None:
        return stream.take(count), count
    out: list[Example] = []
    read = 0
    while len(out) < count:
        if read >= patience * max(count, 1):
            raise SystemExit(f"fewer than {count} '{kind}' examples among the first {read} of the stream")
        example = stream.take(1)[0]
        read += 1
        if example.kind == kind:
            out.append(example)
    return out, read


def write_rollouts(path: Path, examples: Sequence[Example], rollouts: Sequence[Rollout], tokenizer: HFTokenizer
                   ) -> None:
    """One JSON line per rollout, in the examples' order (the keys ``inspect_rollouts.py`` reads)."""
    with open(path, "w", encoding="utf-8") as handle:
        for index, (example, rollout) in enumerate(zip(examples, rollouts, strict=True)):
            handle.write(json.dumps({"index": index, "kind": example.kind, "stop": rollout.stop,
                                     "generated": rollout.generated, "kept": rollout.kept,
                                     "prompt": tokenizer.decode(example.prompt_ids),
                                     "reference": tokenizer.decode(example.canvas_ids),
                                     "text": rollout.text}) + "\n")


def _examples_text(examples: Sequence[Example], rollouts: Sequence[Rollout], tokenizer: HFTokenizer,
                   count: int) -> str:
    parts = []
    for index, (example, rollout) in enumerate(zip(examples[:count], rollouts[:count], strict=True)):
        parts += [f"===== example {index} ({example.kind}; rollout stopped by {rollout.stop} after "
                  f"{rollout.generated} tokens; {'kept' if rollout.kept else 'DROPPED'})",
                  "----- prompt", tokenizer.decode(example.prompt_ids),
                  "----- human reference", tokenizer.decode(example.canvas_ids),
                  "----- rollout (truncated)", rollout.text, ""]
    return "\n".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E0: throughput of autoregressive rollouts", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--show", type=Path, default=None, help="print the summary of OUT/throughput.json and exit")
    p.add_argument("--model", default=None, help="local directory (or Hub id) of the backbone")
    p.add_argument("--corpus", type=Path, default=None, help="corpus directory (train-*.jsonl.gz)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--max-new-tokens", type=int, default=256, help="the AR baseline's budget")
    p.add_argument("--canvas-len", type=int, default=256)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--write-fraction", type=float, default=0.5)
    p.add_argument("--kind", choices=("complete", "write"), default=None,
                   help="only prompts of this format: the first ones of the stream (a diagnostic)")
    p.add_argument("--stream-seed", type=int, default=0, help="CorpusStream seed (runs/stage0-a: 0)")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=1024)
    p.add_argument("--check-every", type=int, default=8, help="tokens between checks of the truncation rule")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-sort", action="store_true", help="batch the prompts in stream order")
    p.add_argument("--entropy-batch", type=int, default=8, help="examples per teacher-forced forward pass")
    p.add_argument("--examples", type=int, default=20)
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.show is not None:
        print(summary_text(json.loads((args.show / "throughput.json").read_text())))
        return 0
    if args.model is None or args.corpus is None or args.out is None:
        raise SystemExit("--model, --corpus and --out are required")
    device = resolve_device(args.device)
    model, hf_tokenizer = load_gemma4(args.model)
    tokenizer = HFTokenizer(hf_tokenizer)
    model.to(device)
    stream = CorpusStream(args.corpus, tokenizer, args.canvas_len, args.max_prompt_tokens, args.write_fraction,
                          args.stream_seed)
    examples, read = take_examples(stream, args.num_prompts, args.kind)
    order = list(range(len(examples)))
    if not args.no_sort:
        order.sort(key=lambda i: len(examples[i].prompt_ids))        # stable: ties keep the stream's order
    kinds = Counter(ex.kind for ex in examples)
    result: dict[str, Any] = {
        "what": "throughput of autoregressive rollouts of the frozen backbone (measured)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"), "model": str(args.model),
        "weights_dtype": str(next(model.parameters()).dtype),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "settings": {"num_prompts": args.num_prompts, "max_new_tokens": args.max_new_tokens,
                     "temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                     "canvas_len": args.canvas_len, "max_prompt_tokens": args.max_prompt_tokens,
                     "write_fraction": args.write_fraction, "stream_seed": args.stream_seed, "seed": args.seed,
                     "check_every": args.check_every, "sorted_by_prompt_length": not args.no_sort,
                     "kind": args.kind or "both"},
        "prompts": {"count": len(examples), "kinds": dict(sorted(kinds.items())),
                    "prompt_tokens": _quantiles([len(ex.prompt_ids) for ex in examples]),
                    "reference_program_tokens": _quantiles([len(ex.canvas_ids) - 1 for ex in examples]),
                    "stream_dropped": dict(stream.dropped), "stream_rows_read": stream.rows_read,
                    "stream_examples_read": read},
        "batch_sizes": [],
    }
    args.out.mkdir(parents=True, exist_ok=True)

    # One short batch first: CUDA start-up must not be charged to the first batch size.
    rollouts_for(model, hf_tokenizer, tokenizer, examples[:2], args.canvas_len, 8, args.temperature, args.top_p,
                 args.top_k, seed=args.seed, check_every=args.check_every)
    measured: dict[int, list[Rollout]] = {}
    for batch_size in args.batch_sizes:
        print(f"== batch size {batch_size}", file=sys.stderr, flush=True)
        try:
            entry, rollouts = measure_batch_size(model, hf_tokenizer, tokenizer, examples, order, batch_size, args,
                                                 device)
        except torch.OutOfMemoryError as exc:
            result["batch_sizes"].append({"batch_size": batch_size, "out_of_memory": True, "error": str(exc)[:160]})
            print(f"   out of memory at batch size {batch_size}; stopping here", file=sys.stderr, flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
        measured[batch_size] = rollouts
        result["batch_sizes"].append(entry)
        (args.out / "throughput.json").write_text(json.dumps(result, indent=2))
        print(f"   {entry['wall_seconds']} s, {entry['rollouts_per_second']} rollouts/s, kept share "
              f"{entry['kept_share']}", file=sys.stderr, flush=True)

    done = [e for e in result["batch_sizes"] if not e.get("out_of_memory")]
    if not done:
        (args.out / "throughput.json").write_text(json.dumps(result, indent=2))
        raise SystemExit("no batch size fitted the device")
    best = max(done, key=lambda e: e["rollouts_per_second"])
    result["best"] = {"batch_size": best["batch_size"], "rollouts_per_second": best["rollouts_per_second"],
                      "kept_share": best["kept_share"], "stops_by_kind": best["stops_by_kind"],
                      "projected": projection(best)}
    rollouts = measured[best["batch_size"]]
    (args.out / "throughput.json").write_text(json.dumps(result, indent=2))
    (args.out / "examples.txt").write_text(_examples_text(examples, rollouts, tokenizer, args.examples))
    write_rollouts(args.out / "rollouts.jsonl", examples, rollouts, tokenizer)

    # Corollary 1: the entropy of the decoding distribution along the kept rollouts; the backbone's NLL of the
    # rollouts and of the human references of the same prompts.
    print("== teacher-forced pass over the kept rollouts and their prompts' references", file=sys.stderr, flush=True)
    realizer = Stage0Realizer.from_pretrained_model(model, RealizerSpec(mask_id=tokenizer.mask_id))
    for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(model.model, tower):
            setattr(model.model, tower, None)                        # text only from here on
    realizer.to(device).eval()
    kept = [i for i, r in enumerate(rollouts) if r.kept]
    as_examples = [Example(examples[i].prompt_ids, list(rollouts[i].canvas_ids or []), examples[i].kind) for i in kept]
    common = {"device": device, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
              "micro_batch": args.entropy_batch}
    of_rollouts = teacher_forced_stats(realizer, tokenizer, as_examples, autocast=autocast(device), **common)
    of_references = teacher_forced_stats(realizer, tokenizer, [examples[i] for i in kept],
                                         autocast=autocast(device), **common)
    references = _pooled(of_references)
    references.pop("entropy_per_token", None)                        # the decoding rule did not write the references
    references.pop("entropy_per_program", None)
    references.pop("eos_entropy", None)
    result["entropy"] = {
        "of": f"the kept rollouts of batch size {best['batch_size']} and the human references of the same prompts",
        "entropy": "of the decoding distribution (temperature, top-k, top-p as in settings) at each program token, "
                   "teacher-forced along the target; its mean per token estimates the rollouts' entropy rate",
        "nll": "-log p(token) of the frozen backbone at temperature 1 without truncation; for the references a "
               "proxy, their own entropy is unknown",
        "rollouts": _pooled(of_rollouts), "references": references}
    (args.out / "throughput.json").write_text(json.dumps(result, indent=2))
    print(summary_text(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
