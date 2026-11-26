# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_rollouts.py -- the data of E1's Step 1 (docs/e1.md, Section 11.3): the frozen target's own greedy rollouts for
the first user turns of UltraChat-200K ``train_sft`` (English first; S2 to S5, confirmed 27 November 2026).

    python -m chat_vpa.e1_rollouts write --model ASSETS/models/gemma-4-E2B-it \\
        --parquet ASSETS/data/ultrachat_200k_train/data/train_sft-*.parquet --start 0 --count 100000 \\
        --dev-prompts ASSETS/prompts/dev.jsonl --out runs/e1-step1/rollouts/train
    python -m chat_vpa.e1_rollouts summary --out runs/e1-step1/rollouts/train

The prompts (S2): the rows of ``train_sft`` in file order whose first user turn has at least one non-blank
character and at most ``--max-chars`` characters (the rule of the development prompts, ``e1_prompts``), numbered
from 0 in that order; ``--start`` and ``--count`` take a range of those numbers (training: 0 to 99,999; held out:
100,000 to 101,999). With ``--dev-prompts`` the writer counts the prompts of its range whose text is that of a
development prompt (the measurement set): the splits are disjoint as rows, which does not make their texts
distinct. The count and the ids are recorded in ``writer.json`` and printed; nothing is excluded (S2 has no such
rule: an overlap is the author's to decide).

The responses (S3): one user turn in the model's chat template, no system prompt, thinking off (``chat_ids``),
greedy, at most ``--max-new-tokens`` new tokens, stopping at the stop tokens of the model's own generation
configuration (Step 0's ``ar`` call: 1, 106 and 50 for E2B-it); a response that reaches the budget is kept; no
content filter. The runtime (S4): the released checkpoint's own ``generate`` in bfloat16, batched with left
padding. A chunk of ``--chunk`` prompts is cut into batches of ``--batch-size`` after sorting by prompt length
(longest first, ties in prompt order), which only saves padding; a batch that runs out of GPU memory is halved and
run again, and the smaller size holds for the rest of the chunk. In bfloat16 the tokens of a prompt can depend on
the batch it shares (Section 10.2, R1): each chunk records the batch size it ran with.

Stored per rollout (S5): ``id``, ``number``, ``prompt_ids`` (the chat prompt as the model saw it), ``response_ids``
(the tokens before the first stop token), ``stop_id`` (that stop token, or null when the budget ended the
response) and ``text`` (the response decoded, for inspection). A chunk is written whole, with a metadata file of
its timing and peak memory, and is skipped when it exists, so a stopped writer resumes at the first missing
chunk. ``--max-chunks`` ends the run after that many new chunks: the timing run of Section 11.8 is
``--max-chunks 1``. ``summary`` adds the chunks up: rollouts, tokens, stops, the measured rate, and the hours the
whole range would take at that rate (*predicted*).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from chat_vpa.e1_common import file_sha256, write_json

__all__ = ["batches_by_length", "chunk_paths", "dev_overlap", "greedy_batch", "qualifying", "read_rollouts",
           "run_chunk", "stop_ids_of", "summarize"]

Generate = Callable[[list[list[int]]], list[tuple[list[int], int | None]]]


def qualifying(rows: Iterable[tuple[str, str | None]], start: int, count: int, max_chars: int, split: str
               ) -> list[dict[str, Any]]:
    """The qualifying rows numbered ``start`` to ``start + count - 1`` (``e1_prompts.select``'s rule), each with
    its ``number`` among the qualifying rows, its ``id`` and its ``prompt``."""
    if start < 0 or count < 1 or max_chars < 1:
        raise ValueError("need start >= 0, count >= 1 and max_chars >= 1")
    out: list[dict[str, Any]] = []
    number = -1
    for prompt_id, prompt in rows:
        if prompt is None or not prompt.strip() or len(prompt) > max_chars:
            continue
        number += 1
        if number < start:
            continue
        out.append({"number": number, "id": f"ultrachat/{split}/{prompt_id}", "prompt": prompt})
        if len(out) == count:
            break
    return out


def dev_overlap(rows: Sequence[dict[str, Any]], dev_prompts: Iterable[str]) -> list[str]:
    """The ids of ``rows`` whose prompt is the text of a development prompt (exactly, or after collapsing runs of
    whitespace and stripping the ends)."""
    def norm(text: str) -> str:
        return " ".join(text.split())

    dev = list(dev_prompts)
    exact, loose = set(dev), {norm(p) for p in dev}
    return [r["id"] for r in rows if r["prompt"] in exact or norm(r["prompt"]) in loose]


def batches_by_length(lengths: Sequence[int], batch_size: int) -> list[list[int]]:
    """Indices into ``lengths`` cut into batches of ``batch_size``, longest first (ties in index order)."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    return [order[k:k + batch_size] for k in range(0, len(order), batch_size)]


def stop_ids_of(model: Any) -> list[int]:
    """The stop tokens of the model's own generation configuration (what Step 0's ``ar`` call stops at)."""
    configured = getattr(model.generation_config, "eos_token_id", None)
    ids = [configured] if isinstance(configured, int) else list(configured or [])
    if not ids:
        raise ValueError("the model's generation configuration names no stop token")
    return sorted({int(i) for i in ids})


def greedy_batch(model: Any, tokenizer: Any, prompts: Sequence[Sequence[int]], max_new_tokens: int,
                 stop_ids: Sequence[int]) -> list[tuple[list[int], int | None]]:
    """One greedy continuation per prompt, in one left-padded batch of the model's own ``generate`` (Step 0's
    ``ar`` call): per prompt, the tokens before its first stop token and that token (None if the budget ended
    it)."""
    import torch

    from chat_vpa.backbone import left_pad

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids, valid, _ = left_pad([list(p) for p in prompts], pad_id, device)
    width = input_ids.shape[1]
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=valid.long(), max_new_tokens=max_new_tokens,
                                do_sample=False, temperature=None, top_p=None, top_k=None,
                                eos_token_id=list(stop_ids), pad_token_id=pad_id)
    stops = {int(s) for s in stop_ids}
    results: list[tuple[list[int], int | None]] = []
    for row in output[:, width:].tolist():
        end = next((i for i, token in enumerate(row) if token in stops), None)
        results.append((row, None) if end is None else (row[:end], row[end]))
    return results


def _is_out_of_memory(error: BaseException) -> bool:
    import torch

    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def run_chunk(rows: Sequence[dict[str, Any]], prompt_ids: Sequence[Sequence[int]], batch_size: int,
              generate: Generate, log: Callable[[str], None] | None = None
              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The rollouts of one chunk, in prompt order, and what the chunk measured. ``generate`` maps a batch of prompt
    id lists to (response, stop) pairs; a batch that runs out of memory is halved and run again, and the halved size
    holds for the rest of the chunk."""
    if len(rows) != len(prompt_ids) or not rows:
        raise ValueError("need one prompt per row and at least one row")
    out: list[dict[str, Any] | None] = [None] * len(rows)
    size, halvings, batches = batch_size, 0, 0
    pending = batches_by_length([len(p) for p in prompt_ids], size)
    start = time.perf_counter()
    while pending:
        batch = pending.pop(0)
        try:
            results = generate([list(prompt_ids[i]) for i in batch])
        except Exception as error:                              # noqa: BLE001 - only out-of-memory is retried
            if not _is_out_of_memory(error) or len(batch) == 1:
                raise
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            size = max(1, len(batch) // 2)
            halvings += 1
            if log is not None:
                log(f"out of memory at a batch of {len(batch)}: batches of {size} from here on in this chunk")
            rest = [i for group in [batch, *pending] for i in group]
            pending = [rest[k:k + size] for k in range(0, len(rest), size)]
            continue
        if len(results) != len(batch):
            raise RuntimeError("generate returned a different number of rollouts than prompts")
        batches += 1
        for i, (response, stop) in zip(batch, results, strict=True):
            out[i] = {"id": rows[i]["id"], "number": rows[i]["number"], "prompt_ids": [int(t) for t in prompt_ids[i]],
                      "response_ids": [int(t) for t in response], "stop_id": None if stop is None else int(stop)}
    seconds = time.perf_counter() - start
    records = [r for r in out if r is not None]
    if len(records) != len(rows):                                # pragma: no cover - every batch is run once
        raise RuntimeError("a prompt of the chunk has no rollout")
    meta = {"prompts": len(rows), "response_tokens": sum(len(r["response_ids"]) for r in records),
            "prompt_tokens": sum(len(r["prompt_ids"]) for r in records),
            "budget_stops": sum(r["stop_id"] is None for r in records), "seconds": seconds,
            "batch_size": batch_size, "batch_size_at_end": size, "halvings": halvings, "batches": batches}
    return records, meta


def chunk_paths(out: Path, index: int) -> tuple[Path, Path]:
    """The records and the metadata of chunk ``index``; the metadata is written last, so it marks a chunk done."""
    return out / f"chunk-{index:05d}.jsonl", out / f"chunk-{index:05d}.json"


def _write(args: argparse.Namespace) -> int:
    import torch

    from chat_vpa.device import resolve_device
    from chat_vpa.e1_common import chat_ids, environment, load_target, read_prompts
    from chat_vpa.e1_prompts import first_turns

    rows = qualifying(first_turns(args.parquet), args.start, args.count, args.max_chars, args.split)
    if len(rows) < args.count:
        raise SystemExit(f"only {len(rows)} of {args.count} qualifying prompts from number {args.start}")
    overlap = None
    if args.dev_prompts is not None:
        overlap = dev_overlap(rows, [p["prompt"] for p in read_prompts(args.dev_prompts)])
        print(f"{len(overlap)} of the {len(rows)} prompts are the text of a development prompt"
              + (f": {', '.join(overlap[:10])}{' ...' if len(overlap) > 10 else ''}" if overlap else ""), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    chunks = math.ceil(len(rows) / args.chunk)
    todo = [k for k in range(chunks) if not chunk_paths(args.out, k)[1].exists()]
    if args.max_chunks is not None:
        todo = todo[:args.max_chunks]
    settings = {"split": args.split, "start": args.start, "count": args.count, "max_chars": args.max_chars,
                "chunk": args.chunk, "chunks": chunks, "max_new_tokens": args.max_new_tokens, "dtype": args.dtype,
                "first_id": rows[0]["id"], "last_id": rows[-1]["id"]}
    previous = args.out / "writer.json"
    if previous.exists():
        before = json.loads(previous.read_text(encoding="utf-8"))
        changed = {k: (before.get(k), v) for k, v in settings.items() if before.get(k) != v}
        if changed:
            raise SystemExit(f"{args.out} was written with other settings: {changed}")
    if not todo:
        print(f"{args.out}: all {chunks} chunks are written")
        return 0
    device = resolve_device(args.device)
    model, tokenizer = load_target(args.model, args.dtype, device)
    stops = stop_ids_of(model)
    write_json(previous, {
        "what": "E1 Step 1's rollouts (docs/e1.md, Section 11.3), measured", **settings,
        "batch_size": args.batch_size, "stop_ids": stops, "decoding": "greedy, the model's own generate",
        "prompt": "one user turn in the chat template, no system prompt", "model": str(args.model),
        "sources": {Path(f).name: file_sha256(f) for f in sorted(args.parquet)},
        "dev_prompts": None if args.dev_prompts is None else str(args.dev_prompts),
        "dev_overlap": overlap, "environment": environment(device, args.dtype)})

    def generate(batch: list[list[int]]) -> list[tuple[list[int], int | None]]:
        return greedy_batch(model, tokenizer, batch, args.max_new_tokens, stops)

    for k in todo:
        part = rows[k * args.chunk:(k + 1) * args.chunk]
        ids = [chat_ids(tokenizer, r["prompt"]) for r in part]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        records, meta = run_chunk(part, ids, args.batch_size, generate, lambda text: print(text, flush=True))
        if device.type == "cuda":
            meta["peak_memory_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
        data, info = chunk_paths(args.out, k)
        tmp = data.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            for record in records:
                text = tokenizer.decode(record["response_ids"], skip_special_tokens=True)
                handle.write(json.dumps({**record, "text": text}, ensure_ascii=False) + "\n")
        tmp.replace(data)
        write_json(info, {**meta, "chunk": k, "first_number": part[0]["number"],
                          "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
        print(f"chunk {k + 1} of {chunks}: {meta['prompts']} rollouts, {meta['response_tokens']} response tokens in "
              f"{meta['seconds']:.0f} s ({meta['prompts'] / meta['seconds']:.2f} rollouts/s, "
              f"{meta['response_tokens'] / meta['seconds']:.0f} tokens/s), {meta['budget_stops']} at the budget",
              flush=True)
    return 0


def summarize(out: Path) -> dict[str, Any]:
    """The chunks written so far, added up, with the hours the whole range would take at their rate
    (*predicted*)."""
    writer = json.loads((out / "writer.json").read_text(encoding="utf-8"))
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("chunk-*.json"))]
    done = sum(m["prompts"] for m in metas)
    seconds = sum(m["seconds"] for m in metas)
    tokens = sum(m["response_tokens"] for m in metas)
    rate = done / seconds if seconds > 0 else None
    return {"chunks_written": len(metas), "chunks": writer["chunks"], "rollouts": done, "of": writer["count"],
            "response_tokens": tokens, "mean_response_tokens": tokens / done if done else None,
            "mean_prompt_tokens": sum(m.get("prompt_tokens", 0) for m in metas) / done if done else None,
            "budget_stops": sum(m["budget_stops"] for m in metas), "seconds": seconds,
            "rollouts_per_second": rate, "response_tokens_per_second": tokens / seconds if seconds > 0 else None,
            "halvings": sum(m["halvings"] for m in metas),
            "peak_memory_gib": max((m.get("peak_memory_gib", 0.0) for m in metas), default=None),
            "dev_overlap": None if writer.get("dev_overlap") is None else len(writer["dev_overlap"]),
            "predicted_hours_total": writer["count"] / rate / 3600 if rate else None,
            "predicted_hours_left": (writer["count"] - done) / rate / 3600 if rate else None}


def summary_line(summary: dict[str, Any]) -> str:
    rate = summary["rollouts_per_second"] or 0.0
    return (f"{summary['rollouts']} of {summary['of']} rollouts in {summary['chunks_written']} of "
            f"{summary['chunks']} chunks; mean prompt {summary['mean_prompt_tokens'] or 0:.0f} and response "
            f"{summary['mean_response_tokens'] or 0:.1f} tokens; {summary['budget_stops']} at the budget; measured "
            f"{rate:.2f} rollouts/s ({summary['response_tokens_per_second'] or 0:.0f} response tokens/s), peak "
            f"{summary['peak_memory_gib']} GiB, {summary['halvings']} halvings of the batch; predicted "
            f"{summary['predicted_hours_total'] or 0:.1f} h for all, {summary['predicted_hours_left'] or 0:.1f} h left"
            + ("" if summary["dev_overlap"] is None else f"; {summary['dev_overlap']} prompts are development prompts"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E1 Step 1: the target's own greedy rollouts", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write", help="write the chunks that are missing")
    w.add_argument("--model", type=Path, required=True)
    w.add_argument("--parquet", nargs="+", required=True, help="the split's parquet files")
    w.add_argument("--split", default="train_sft")
    w.add_argument("--start", type=int, default=0, help="the first qualifying prompt's number")
    w.add_argument("--count", type=int, required=True)
    w.add_argument("--max-chars", type=int, default=4000)
    w.add_argument("--max-new-tokens", type=int, default=256)
    w.add_argument("--chunk", type=int, default=2048)
    w.add_argument("--batch-size", type=int, default=128)
    w.add_argument("--max-chunks", type=int, default=None, help="end after this many new chunks (timing: 1)")
    w.add_argument("--dev-prompts", type=Path, default=None, help="count the prompts that are development prompts")
    w.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    w.add_argument("--device", default="auto")
    w.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summary", help="add up the chunks written so far")
    s.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == "write":
        return _write(args)
    summary = summarize(args.out)
    write_json(args.out / "summary.json", summary)
    print(summary_line(summary))
    return 0


def read_rollouts(directory: str | Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """The records of the chunks under ``directory``, chunk after chunk, in prompt order within each."""
    paths = sorted(Path(directory).glob("chunk-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no chunk-*.jsonl under {directory}")
    read = 0
    for path in paths:
        if not path.with_suffix(".json").exists():
            continue                                             # a chunk is done once its metadata is written
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)
                read += 1
                if limit is not None and read >= limit:
                    return


if __name__ == "__main__":
    sys.exit(main())
