# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
rollout_writer.py -- E0's data: for every kept prompt, its human reference and one autoregressive rollout of the
frozen backbone (choice C6 of docs/e0.md, confirmed by the author on 23 November 2026).

    python -m chat_vpa.rollout_writer --model data2/models/gemma-4-E2B --corpus data2/corpus --out runs/e0-data

On the server: ``bash scripts/ncp/e0_data.sh`` (``check`` first: batch 256 against 128).

C6: the prompts of ``runs/stage0-a``'s stream (``CorpusStream``, seed 0) in that order; one rollout per prompt
through the released model's own ``generate`` with the baseline's draw (temperature 0.2, top-k 1,024, top-p 0.95)
and budget (256 new tokens), cut by the shared truncation rule, tokenised again and followed by one <eos>
(``rollouts.finish_rollout``). A rollout that reaches the budget without the rule's stop or an <eos>, or whose
program does not fit the canvas, is dropped, and its prompt leaves both arms. No content filter. The kept set is
stratified by format: ``round(write_share * kept)`` ``write`` examples and the rest ``complete``, each quota
filled in stream order, so that it has the stream's format mix (13.5% ``write``, measured).

How, resumably and deterministically for given settings and batch sizes:

- First the validation set of C7: the first ``val_examples`` kept examples of the corpus's validation file, at
  the same shares (``OUT/validation.jsonl.gz``).
- Then the training set, in chunks of ``chunk_size`` candidates. A candidate is the next example of the source
  whose format's quota was not yet full after the previous chunk (the others are read and passed over). A
  chunk's candidates are sorted by prompt length and generated in batches of ``batch_size``, each seeded from
  (seed, chunk, batch). Its kept rollouts go to ``chunk-NNNNN.jsonl.gz``; ``state.json`` then records the
  source's position, the kept counts and the chunk's log, atomically, so a stopped run resumes after its last
  finished chunk. A batch that runs out of memory makes the chunk start again at half the batch size; the batch
  size of every chunk is in its log. The stream must not wrap into a second pass over the corpus (that would
  repeat prompts) unless ``--allow-wrap``.
- When both quotas are full, ``OUT/train.jsonl.gz`` holds the first quota of kept examples of each format in
  stream order (the last chunk's later kept examples are beyond the quota and left out).
- ``OUT/summary.json``: per set the candidates, the kept examples and the stops by format, the kept rates,
  the source examples read, the hours and the batch sizes; the share of kept ``write`` rollouts that define the
  function their prompt names (``inspect_rollouts``); per format the program lengths of rollouts and references
  and the number of kept rollouts that are empty programs; and Corollary 1's quantities on the first
  ``entropy_examples`` kept examples of each format: the entropy of the decoding distribution along the
  rollouts and the backbone's NLL of the rollouts and of the references, per format and pooled at the kept
  shares.

A record (one JSON line): ``stream_index`` (the example's position among the source's examples), ``kind``,
``prompt_ids`` (with <bos>), ``reference_ids`` and ``rollout_ids`` (program tokens and one <eos>), ``stop`` (how
the rollout ended: ``truncation`` or ``eos``) and ``generated`` (tokens generated before it stopped). Nothing is
executed. ``--show OUT`` prints the progress of a run, or the summary of a finished one.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from chat_vpa.data import CorpusStream, Example, Stage0Tokenizer
from chat_vpa.inspect_rollouts import function_end, named_function
from chat_vpa.rollouts import Rollout, teacher_forced_stats

__all__ = ["FORMATS", "Candidate", "ListSource", "StreamSource", "batch_seed", "fill_quotas", "finalize",
           "generate_chunk", "quotas", "read_records", "write_records"]

FORMATS: tuple[str, ...] = ("complete", "write")
# Settings that decide the data: a run resumes only with the same ones (the batch size may change).
DATA_SETTINGS: tuple[str, ...] = ("model", "corpus", "kept", "write_share", "val_examples", "val_pool", "chunk_size",
                                  "canvas_len", "max_prompt_tokens", "write_fraction", "stream_seed",
                                  "max_new_tokens", "temperature", "top_p", "top_k", "check_every", "seed",
                                  "allow_wrap")

Generate = Callable[[Sequence[Example], int], list[Rollout]]       # one batch's examples and seed -> rollouts


def quotas(kept: int, write_share: float) -> dict[str, int]:
    """Kept examples per format: ``round(write_share * kept)`` ``write``, the rest ``complete``."""
    if kept < 1 or not 0.0 <= write_share <= 1.0:
        raise ValueError("need kept >= 1 and 0 <= write_share <= 1")
    write = round(write_share * kept)
    return {"complete": kept - write, "write": write}


def batch_seed(seed: int, chunk: int, batch: int) -> int:
    return (seed * 1_000_003 + chunk) * 1_009 + batch


@dataclass(frozen=True)
class Candidate:
    index: int                     # position among the source's examples
    example: Example


class StreamSource:
    """The training stream (``CorpusStream``) as numbered examples; refuses to wrap into a second pass."""

    def __init__(self, stream: CorpusStream, read: int = 0, allow_wrap: bool = False) -> None:
        self.stream, self.read, self.allow_wrap = stream, read, allow_wrap

    def next(self) -> tuple[int, Example] | None:
        example = self.stream.take(1)[0]
        if not self.allow_wrap and self.stream.state()["epoch"] > 0:
            raise SystemExit(f"the stream wrapped into a second pass over the corpus after {self.read} examples; "
                             "the quotas do not fit in one pass (--allow-wrap to repeat prompts)")
        index, self.read = self.read, self.read + 1
        return index, example

    def state(self) -> dict[str, Any]:
        return {"stream": self.stream.state(), "read": self.read}


class ListSource:
    """A fixed list of examples (the validation file's) as numbered examples."""

    def __init__(self, examples: Sequence[Example], read: int = 0) -> None:
        self.examples, self.read = list(examples), read

    def next(self) -> tuple[int, Example] | None:
        if self.read >= len(self.examples):
            return None
        index, self.read = self.read, self.read + 1
        return index, self.examples[index]

    def state(self) -> dict[str, Any]:
        return {"read": self.read}


def generate_chunk(candidates: Sequence[Candidate], chunk: int, batch_size: int, seed: int, generate: Generate
                   ) -> list[Rollout]:
    """One rollout per candidate, in the candidates' order: sorted by prompt length (stable) into batches of
    ``batch_size``, batch b seeded with ``batch_seed(seed, chunk, b)``."""
    order = sorted(range(len(candidates)), key=lambda i: len(candidates[i].example.prompt_ids))
    out: list[Rollout | None] = [None] * len(candidates)
    for number, first in enumerate(range(0, len(order), batch_size)):
        rows = order[first:first + batch_size]
        rollouts = generate([candidates[i].example for i in rows], batch_seed(seed, chunk, number))
        for i, rollout in zip(rows, rollouts, strict=True):
            out[i] = rollout
    return [r for r in out if r is not None]


def record(candidate: Candidate, rollout: Rollout) -> dict[str, Any]:
    return {"stream_index": candidate.index, "kind": candidate.example.kind,
            "prompt_ids": list(candidate.example.prompt_ids), "reference_ids": list(candidate.example.canvas_ids),
            "rollout_ids": list(rollout.canvas_ids or []), "stop": rollout.stop, "generated": rollout.generated}


def write_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Atomic gzip JSON lines, without a file name or a time stamp in the header: the same records give the same
    bytes, whatever the file is called."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
        for row in records:
            handle.write((json.dumps(row) + "\n").encode("utf-8"))
    os.replace(tmp, path)


def read_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def fill_quotas(make_source: Callable[[dict[str, Any] | None], Any], quota: dict[str, int], chunk_dir: Path,
                chunk_size: int, batch_size: int, seed: int, generate: Generate, max_chunks: int | None = None,
                log: Callable[[str], None] = print, stall_chunks: int = 50) -> bool:
    """Rollouts in chunks until every format holds its quota of kept examples (see the module docstring).

    ``make_source(saved)`` returns the source positioned after the chunks already finished (``saved`` is its
    state from ``state.json``, or None). Returns True when the quotas are full, False when ``max_chunks`` new
    chunks were made first (a later call resumes). Stops with an error when ``stall_chunks`` chunks in a row keep
    nothing of a format whose quota is open and that had candidates in them."""
    if chunk_size < 1 or batch_size < 1:
        raise ValueError("chunk_size and batch_size must be >= 1")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    state_path = chunk_dir / "state.json"
    state: dict[str, Any] = (json.loads(state_path.read_text()) if state_path.exists() else
                             {"chunks": 0, "kept": {f: 0 for f in FORMATS}, "source": None, "log": []})
    source = make_source(state["source"])
    kept, chunk, made = dict(state["kept"]), int(state["chunks"]), 0
    stalled = {f: 0 for f in FORMATS}
    while any(kept[f] < quota[f] for f in FORMATS):
        if max_chunks is not None and made >= max_chunks:
            return False
        open_formats = {f for f in FORMATS if kept[f] < quota[f]}
        candidates: list[Candidate] = []
        while len(candidates) < chunk_size:
            item = source.next()
            if item is None:
                break
            if item[1].kind in open_formats:
                candidates.append(Candidate(*item))
        if not candidates:
            raise SystemExit(f"the source ran out with kept {kept} of {quota}")
        start, size = time.perf_counter(), batch_size
        while True:
            try:
                rollouts = generate_chunk(candidates, chunk, size, seed, generate)
                break
            except torch.OutOfMemoryError:
                if size == 1:
                    raise
                size //= 2
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log(f"chunk {chunk}: out of memory; again at batch size {size}")
        records = [record(c, r) for c, r in zip(candidates, rollouts, strict=True) if r.kept]
        write_records(chunk_dir / f"chunk-{chunk:05d}.jsonl.gz", records)
        new, tried = Counter(r["kind"] for r in records), Counter(c.example.kind for c in candidates)
        for f in FORMATS:
            kept[f] += new[f]
            stalled[f] = stalled[f] + 1 if tried[f] and not new[f] else (stalled[f] if not tried[f] else 0)
            if stalled[f] >= stall_chunks:
                raise SystemExit(f"{stall_chunks} chunks in a row kept no '{f}' example; kept {kept} of {quota}")
        entry = {"chunk": chunk, "candidates": dict(tried),
                 "kept": {f: new[f] for f in FORMATS},
                 "stops": dict(Counter(f"{c.example.kind}/{r.stop}" for c, r in zip(candidates, rollouts,
                                                                                    strict=True))),
                 "batch_size": size, "seconds": round(time.perf_counter() - start, 2), "read": source.read}
        state = {"chunks": chunk + 1, "kept": kept, "source": source.state(), "log": [*state["log"], entry]}
        _write_json(state_path, state)
        log(f"chunk {chunk}: {len(candidates)} candidates, kept " + ", ".join(
            f"{f} {kept[f]}/{quota[f]}" for f in FORMATS) + f"; batch {size}; {entry['seconds']:.0f} s; "
            f"{source.read} examples read")
        chunk, made = chunk + 1, made + 1
    return True


def finalize(chunk_dir: Path, quota: dict[str, int]) -> list[dict[str, Any]]:
    """The first ``quota[f]`` kept records of each format, in source order, from the finished chunks."""
    state = json.loads((chunk_dir / "state.json").read_text())
    taken = {f: 0 for f in FORMATS}
    out: list[dict[str, Any]] = []
    for chunk in range(int(state["chunks"])):
        for row in read_records(chunk_dir / f"chunk-{chunk:05d}.jsonl.gz"):
            if taken[row["kind"]] < quota[row["kind"]]:
                taken[row["kind"]] += 1
                out.append(row)
    if taken != quota:
        raise ValueError(f"the chunks hold {taken} kept examples, not the quotas {quota}")
    return out


# ---------------------------------------------------------------------------------------------
# What the summary reports
# ---------------------------------------------------------------------------------------------

def set_summary(chunk_dir: Path, records: Sequence[dict[str, Any]], quota: dict[str, int]) -> dict[str, Any]:
    state = json.loads((chunk_dir / "state.json").read_text())
    candidates, kept, stops, sizes = Counter(), Counter(), Counter(), Counter()
    seconds = 0.0
    for entry in state["log"]:
        candidates.update(entry["candidates"])
        kept.update(entry["kept"])
        stops.update(entry["stops"])
        sizes[str(entry["batch_size"])] += 1
        seconds += entry["seconds"]
    return {"quota": quota, "records": dict(Counter(r["kind"] for r in records)),
            "candidates": {f: candidates[f] for f in FORMATS}, "kept_before_trim": {f: kept[f] for f in FORMATS},
            "kept_rate": {f: round(kept[f] / candidates[f], 4) if candidates[f] else None for f in FORMATS},
            "stops": dict(sorted(stops.items())), "source_examples_read": state["source"]["read"],
            "chunks": state["chunks"], "chunks_by_batch_size": dict(sizes), "generation_hours": round(seconds / 3600, 3)}


def named_function_share(records: Sequence[dict[str, Any]], tokenizer: Stage0Tokenizer) -> dict[str, Any]:
    """How many kept ``write`` rollouts define the function their prompt names (C6 reports it; none is removed)."""
    writes = [r for r in records if r["kind"] == "write"]
    defined = 0
    for row in writes:
        name = named_function(tokenizer.decode(row["prompt_ids"]))
        text = tokenizer.decode(row["rollout_ids"][:-1])
        defined += name is not None and function_end(text, name)[0] is not None
    return {"write_kept": len(writes), "define_the_named_function": defined,
            "share": round(defined / len(writes), 4) if writes else None}


def lengths_summary(records: Sequence[dict[str, Any]], tokenizer: Stage0Tokenizer) -> dict[str, Any]:
    """Per format: program tokens of the rollouts and of the references (mean, median), and how many kept
    rollouts are empty programs (C6 keeps them: no content filter)."""
    out: dict[str, Any] = {}
    for f in FORMATS:
        rows = [r for r in records if r["kind"] == f]
        if not rows:
            continue
        rollout = sorted(len(r["rollout_ids"]) - 1 for r in rows)
        reference = sorted(len(r["reference_ids"]) - 1 for r in rows)
        empty = sum(not tokenizer.decode(r["rollout_ids"][:-1]).strip() for r in rows)
        out[f] = {"rollout_tokens_mean": round(sum(rollout) / len(rows), 2), "rollout_tokens_median": rollout[len(rows) // 2],
                  "reference_tokens_mean": round(sum(reference) / len(rows), 2),
                  "reference_tokens_median": reference[len(rows) // 2], "empty_rollouts": empty}
    return out


def _pooled(stats: Sequence[dict[str, float]]) -> dict[str, Any]:
    tokens, programs = sum(s["tokens"] for s in stats), len(stats)
    out: dict[str, Any] = {"programs": programs, "tokens": tokens}
    if programs and tokens:
        out.update(entropy_per_token=round(sum(s["entropy"] for s in stats) / tokens, 5),
                   entropy_per_program=round(sum(s["entropy"] for s in stats) / programs, 4),
                   nll_per_token=round(sum(s["nll"] for s in stats) / tokens, 5),
                   nll_per_program=round(sum(s["nll"] for s in stats) / programs, 4),
                   tokens_per_program=round(tokens / programs, 2))
    return out


def entropy_summary(records: Sequence[dict[str, Any]], realizer: Any, tokenizer: Stage0Tokenizer,
                    device: torch.device, count: int, temperature: float, top_p: float, top_k: int,
                    micro_batch: int, autocast: Any = None) -> dict[str, Any]:
    """Corollary 1 on the first ``count`` kept examples of each format: the rollouts' entropy rate under the
    decoding rule and the backbone's NLL of rollouts and references; pooled per token at the kept shares
    (share-weighted sums per program over share-weighted tokens per program)."""
    shares = Counter(r["kind"] for r in records)
    out: dict[str, Any] = {}
    for f in FORMATS:
        rows = [r for r in records if r["kind"] == f][:count]
        if not rows:
            continue
        common = {"device": device, "temperature": temperature, "top_p": top_p, "top_k": top_k,
                  "micro_batch": micro_batch, "autocast": autocast}
        rollouts = teacher_forced_stats(realizer, tokenizer, [Example(r["prompt_ids"], r["rollout_ids"], f)
                                                              for r in rows], **common)
        references = teacher_forced_stats(realizer, tokenizer, [Example(r["prompt_ids"], r["reference_ids"], f)
                                                                for r in rows], **common)
        refs = _pooled(references)
        for key in ("entropy_per_token", "entropy_per_program"):    # the decoding rule did not write them
            refs.pop(key, None)
        out[f] = {"rollouts": _pooled(rollouts), "references": refs}
    total = sum(shares.values())
    if total and all(f in out and "entropy_per_program" in out[f]["rollouts"] for f in shares):
        weight = {f: shares[f] / total for f in shares}
        per_program = sum(weight[f] * out[f]["rollouts"]["entropy_per_program"] for f in shares)
        tokens = sum(weight[f] * out[f]["rollouts"]["tokens_per_program"] for f in shares)
        out["pooled_at_kept_shares"] = {"shares": {f: round(weight[f], 4) for f in shares},
                                        "rollout_entropy_per_token": round(per_program / tokens, 5)}
    return out


def progress_text(out: Path) -> str:
    """The state of a run: per set the kept counts against the quotas, and a prediction of what is left."""
    lines = [f"rollout writer: {out}"]
    summary = out / "summary.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        for name in ("validation", "train"):
            s = data[name]
            lines.append(f"  {name}: {s['records']} of quota {s['quota']}; kept rate {s['kept_rate']}; "
                         f"{s['source_examples_read']} examples read; {s['generation_hours']} h; batch sizes "
                         f"{s['chunks_by_batch_size']}")
        lines.append(f"  named function in kept write rollouts: {data['named_function']}")
        for f, stats in data.get("lengths", {}).items():
            lines.append(f"  lengths {f}: {stats}")
        for f, stats in data.get("entropy", {}).items():
            lines.append(f"  entropy {f}: {stats}")
        lines.append("  finished")
        return "\n".join(lines)
    for name in ("validation", "train"):
        path = out / f"{name}-chunks" / "state.json"
        if not path.exists():
            lines.append(f"  {name}: not started")
            continue
        state = json.loads(path.read_text())
        log = state["log"]
        candidates = Counter()
        for entry in log:
            candidates.update(entry["candidates"])
        seconds = sum(e["seconds"] for e in log)
        quota = json.loads((out / "settings.json").read_text())["quotas"][name]
        kept, read = state["kept"], int((state["source"] or {}).get("read", 0))
        open_formats = [f for f in FORMATS if kept[f] < quota[f]]
        line = (f"  {name}: kept {kept} of {quota} after {state['chunks']} chunks, {read} examples read, "
                f"{seconds / 3600:.2f} h")
        if (out / f"{name}.jsonl.gz").exists():
            line += "; finished"
        elif all(kept[f] for f in open_formats):
            # At the rates so far, an open format needs (remaining) x (candidates per kept example) more
            # candidates, and the source is read until the last quota is full: read x quota / kept examples.
            left = sum((quota[f] - kept[f]) * candidates[f] / kept[f] for f in open_formats)
            per_candidate = seconds / max(sum(candidates.values()), 1)
            total = max((math.ceil(read * quota[f] / kept[f]) for f in open_formats), default=read)
            line += (f"; about {left * per_candidate / 3600:.1f} h left and {total:,} examples read in all "
                     "(predicted from the rates so far)")
        else:
            line += "; too early to predict (a format with an open quota has kept nothing yet)"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E0's data: references and rollouts of the kept prompts", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--show", type=Path, default=None, help="print the progress or summary of OUT and exit")
    p.add_argument("--model", default=None, help="local directory of the backbone")
    p.add_argument("--corpus", type=Path, default=None, help="corpus directory (train-*.jsonl.gz, val.jsonl.gz)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--kept", type=int, default=100_000, help="N_r: kept training examples (C6)")
    p.add_argument("--write-share", type=float, default=0.135, help="share of write examples in the kept set (C6)")
    p.add_argument("--val-examples", type=int, default=256, help="kept validation examples (C7)")
    p.add_argument("--val-pool", type=int, default=10_000, help="validation-file examples to draw them from")
    p.add_argument("--chunk-size", type=int, default=2048, help="candidates per chunk (the unit of resumption)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--canvas-len", type=int, default=256)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--write-fraction", type=float, default=0.5, help="CorpusStream's (runs/stage0-a: 0.5)")
    p.add_argument("--stream-seed", type=int, default=0, help="CorpusStream's (runs/stage0-a: 0)")
    p.add_argument("--max-new-tokens", type=int, default=256, help="the AR baseline's budget")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=1024)
    p.add_argument("--check-every", type=int, default=8, help="tokens between checks of the truncation rule")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--entropy-examples", type=int, default=2000, help="kept examples per format for Corollary 1")
    p.add_argument("--entropy-batch", type=int, default=8)
    p.add_argument("--allow-wrap", action="store_true", help="let the stream start a second pass (repeats prompts)")
    p.add_argument("--max-chunks", type=int, default=None, help="stop after this many new training chunks")
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    keys = {k: getattr(args, k) for k in DATA_SETTINGS}
    keys["model"], keys["corpus"] = str(args.model), str(args.corpus)
    return keys


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.show is not None:
        print(progress_text(args.show))
        return 0
    if args.model is None or args.corpus is None or args.out is None:
        raise SystemExit("--model, --corpus and --out are required")
    from chat_vpa.backbone import RealizerSpec, Stage0Realizer, load_gemma4
    from chat_vpa.data import HFTokenizer, load_validation
    from chat_vpa.device import resolve_device
    from chat_vpa.evaluate import autocast
    from chat_vpa.rollouts import rollouts_for

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    settings = _settings(args)
    settings["quotas"] = {"validation": quotas(args.val_examples, args.write_share),
                          "train": quotas(args.kept, args.write_share)}
    saved_path = out / "settings.json"
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())
        changed = {k: (saved.get(k), v) for k, v in settings.items() if saved.get(k) != v and k != "tokenizer"}
        if changed:
            raise SystemExit(f"{out} was started with other settings: {changed} (a new OUT, or the same settings)")
    device = resolve_device(args.device)
    model, hf_tokenizer = load_gemma4(args.model)
    tokenizer = HFTokenizer(hf_tokenizer)
    model.to(device)
    ids = {"bos": tokenizer.bos_id, "eos": tokenizer.eos_id, "pad": tokenizer.pad_id, "mask": tokenizer.mask_id,
           "vocab_size": tokenizer.vocab_size}
    if saved_path.exists() and json.loads(saved_path.read_text()).get("tokenizer") != ids:
        raise SystemExit(f"{out} was written with another tokenizer")
    _write_json(saved_path, {**settings, "tokenizer": ids})

    def generate(examples: Sequence[Example], seed: int) -> list[Rollout]:
        rollouts, _ = rollouts_for(model, hf_tokenizer, tokenizer, examples, args.canvas_len, args.max_new_tokens,
                                   args.temperature, args.top_p, args.top_k, seed, args.check_every)
        return rollouts

    def log(line: str) -> None:
        print(f"[{time.strftime('%F %T')}] {line}", file=sys.stderr, flush=True)

    sets = {}
    validation_file = out / "validation.jsonl.gz"
    if not validation_file.exists():
        pool = load_validation(args.corpus, tokenizer, args.canvas_len, args.max_prompt_tokens, args.val_pool)
        log(f"validation: {len(pool)} examples of the validation file, quotas {settings['quotas']['validation']}")
        fill_quotas(lambda saved: ListSource(pool, saved["read"] if saved else 0), settings["quotas"]["validation"],
                    out / "validation-chunks", args.chunk_size, args.batch_size, args.seed + 1, generate, log=log)
        write_records(validation_file, finalize(out / "validation-chunks", settings["quotas"]["validation"]))
    train_file = out / "train.jsonl.gz"
    if not train_file.exists():
        def stream_source(saved: dict[str, Any] | None) -> StreamSource:
            stream = CorpusStream(args.corpus, tokenizer, args.canvas_len, args.max_prompt_tokens,
                                  args.write_fraction, args.stream_seed, state=saved["stream"] if saved else None)
            return StreamSource(stream, saved["read"] if saved else 0, args.allow_wrap)

        log(f"training set: quotas {settings['quotas']['train']}, batch size {args.batch_size}")
        if not fill_quotas(stream_source, settings["quotas"]["train"], out / "train-chunks", args.chunk_size,
                           args.batch_size, args.seed, generate, args.max_chunks, log=log):
            log(f"stopped after {args.max_chunks} new chunks; the same command resumes")
            print(progress_text(out))
            return 0
        write_records(train_file, finalize(out / "train-chunks", settings["quotas"]["train"]))
    for name in ("validation", "train"):
        sets[name] = read_records(out / f"{name}.jsonl.gz")

    log("summary: the named function in kept write rollouts, and Corollary 1 on the first kept examples")
    realizer = Stage0Realizer.from_pretrained_model(model, RealizerSpec(mask_id=tokenizer.mask_id))
    for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(model.model, tower):
            setattr(model.model, tower, None)                        # text only from here on
    realizer.to(device).eval()
    summary = {"what": "E0's data (docs/e0.md, C6): kept prompts with their reference and the backbone's rollout "
                       "(measured)",
               "date": time.strftime("%Y-%m-%d %H:%M:%S %z"), "settings": settings,
               "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
               "validation": set_summary(out / "validation-chunks", sets["validation"],
                                         settings["quotas"]["validation"]),
               "train": set_summary(out / "train-chunks", sets["train"], settings["quotas"]["train"]),
               "named_function": named_function_share(sets["train"], tokenizer),
               "lengths": lengths_summary(sets["train"], tokenizer),
               "entropy": entropy_summary(sets["train"], realizer, tokenizer, device, args.entropy_examples,
                                          args.temperature, args.top_p, args.top_k, args.entropy_batch,
                                          autocast(device))}
    _write_json(out / "summary.json", summary)
    print(progress_text(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
