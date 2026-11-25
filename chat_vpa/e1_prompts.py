# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_prompts.py -- E1's development prompts (docs/e1.md, Section 3): the first user turns of UltraChat-200K's
``test_sft`` split, as a prompt file.

    python -m chat_vpa.e1_prompts --parquet ASSETS/data/ultrachat_200k/data/test_sft-*.parquet \\
        --count 200 --out ASSETS/prompts/dev.jsonl

The rule: read the files in name order and the rows in file order; keep a row when its first user turn (the
``prompt`` column) has at least one non-blank character and at most ``--max-chars`` characters; stop at
``--count`` rows. Nothing is drawn at random and no model is consulted. A row becomes
``{"id": "ultrachat/test_sft/<prompt_id>", "prompt": <text>}``.

Why this split: the drafters of E1's Step 1 are trained on rollouts for prompts of ``train_sft``, so ``test_sft``
is disjoint from every training prompt, and everything that is tuned (a draft length, a block size) is tuned on
this file. IFEval is not read here: it has no development split and stays untouched until a protocol run
(working agreement 3).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from chat_vpa.e1_common import file_sha256, write_json

__all__ = ["first_turns", "select"]


def first_turns(paths: Sequence[str | Path]) -> Iterator[tuple[str, str]]:
    """(prompt_id, first user turn) of every row, files in name order, rows in file order."""
    import pyarrow.parquet as pq

    for path in sorted(str(p) for p in paths):
        table = pq.ParquetFile(path)
        for batch in table.iter_batches(batch_size=1024, columns=["prompt", "prompt_id"]):
            prompts, ids = batch.column("prompt").to_pylist(), batch.column("prompt_id").to_pylist()
            yield from zip(ids, prompts, strict=True)


def select(rows: Iterator[tuple[str, str]], count: int, max_chars: int, split: str = "test_sft"
           ) -> list[dict[str, str]]:
    """The first ``count`` rows whose prompt is non-blank and at most ``max_chars`` characters long."""
    if count < 1 or max_chars < 1:
        raise ValueError("need count >= 1 and max_chars >= 1")
    out: list[dict[str, str]] = []
    for prompt_id, prompt in rows:
        if prompt is None or not prompt.strip() or len(prompt) > max_chars:
            continue
        out.append({"id": f"ultrachat/{split}/{prompt_id}", "prompt": prompt})
        if len(out) == count:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E1's development prompts from UltraChat-200K", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", nargs="+", required=True, help="the split's parquet files")
    p.add_argument("--split", default="test_sft", help="the split's name, for the ids")
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=4000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    rows = select(first_turns(args.parquet), args.count, args.max_chars, args.split)
    if len(rows) < args.count:
        raise SystemExit(f"only {len(rows)} of {args.count} prompts found in {len(args.parquet)} file(s)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(args.out.with_suffix(".meta.json"), {
        "what": "E1's development prompts (docs/e1.md, Section 3)", "split": args.split, "count": len(rows),
        "max_chars": args.max_chars, "sources": {Path(f).name: file_sha256(f) for f in sorted(args.parquet)},
        "sha256": file_sha256(args.out)})
    print(f"{args.out}: {len(rows)} prompts, mean {sum(len(r['prompt']) for r in rows) / len(rows):.0f} characters",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
