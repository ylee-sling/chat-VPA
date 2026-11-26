# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""E1 Step 1's rollout writer (``chat_vpa.e1_rollouts``, docs/e1.md Section 11.3): the prompts are the qualifying
first turns numbered in file order; a batch is the released model's own greedy ``generate`` and gives, prompt for
prompt, what that call gives alone (float32, the tiny model); a batch that runs out of memory is halved; a stopped
writer resumes at the first missing chunk and refuses other settings."""

from __future__ import annotations

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")
if not hasattr(transformers, "Gemma4ForConditionalGeneration"):      # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4", allow_module_level=True)

from test_e1 import CHAT_TEMPLATE  # noqa: E402
from test_stage0_cli import write_checkpoint  # noqa: E402

from chat_vpa import e1_rollouts as ro  # noqa: E402
from chat_vpa.e1_common import chat_ids  # noqa: E402


def test_the_prompts_are_numbered_among_the_qualifying_rows():
    rows = [("a", "first"), ("b", "  "), ("c", None), ("d", "x" * 11), ("e", "second"), ("f", "third"),
            ("g", "fourth"), ("h", "fifth")]
    # By hand: qualifying rows a, e, f, g, h are numbers 0 to 4; blank, missing and 11-character prompts are not.
    got = ro.qualifying(iter(rows), start=1, count=3, max_chars=10, split="train_sft")
    assert got == [{"number": 1, "id": "ultrachat/train_sft/e", "prompt": "second"},
                   {"number": 2, "id": "ultrachat/train_sft/f", "prompt": "third"},
                   {"number": 3, "id": "ultrachat/train_sft/g", "prompt": "fourth"}]
    assert [r["number"] for r in ro.qualifying(iter(rows), 0, 100, 10, "s")] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        ro.qualifying(iter(rows), -1, 1, 10, "s")
    # The overlap with the development prompts: exact, or equal after collapsing whitespace.
    picked = [{"id": "p1", "prompt": "Name a  colour."}, {"id": "p2", "prompt": "Count to three"},
              {"id": "p3", "prompt": " Count to three"}]
    assert ro.dev_overlap(picked, ["Count to three", "Name a colour.\n"]) == ["p1", "p2", "p3"]
    assert ro.dev_overlap(picked, ["something else"]) == []


def test_batches_are_cut_longest_first_and_halved_on_out_of_memory():
    assert ro.batches_by_length([3, 7, 7, 1, 5], 2) == [[1, 2], [4, 0], [3]]           # ties in index order
    with pytest.raises(ValueError):
        ro.batches_by_length([1], 0)
    rows = [{"id": f"r{i}", "number": i} for i in range(5)]
    prompts = [[2] * n for n in (3, 7, 7, 1, 5)]
    seen: list[int] = []

    def generate(batch):
        seen.append(len(batch))
        if len(batch) > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (a stand-in)")
        return [(list(range(len(p))), 1 if len(p) % 2 else None) for p in batch]

    logs: list[str] = []
    records, meta = ro.run_chunk(rows, prompts, 4, generate, logs.append)
    # By hand: [1, 2, 4, 0] fails at 4, then batches of 2 from the same order: [1, 2], [4, 0], [3].
    assert seen == [4, 2, 2, 1] and len(logs) == 1 and "batches of 2" in logs[0]
    assert [r["id"] for r in records] == ["r0", "r1", "r2", "r3", "r4"]                  # prompt order
    assert [r["response_ids"] for r in records] == [[0, 1, 2], list(range(7)), list(range(7)), [0], list(range(5))]
    assert [r["stop_id"] for r in records] == [1, 1, 1, 1, 1]
    assert meta["halvings"] == 1 and meta["batch_size_at_end"] == 2 and meta["batches"] == 3
    assert meta["response_tokens"] == 23 and meta["prompt_tokens"] == 23 and meta["budget_stops"] == 0

    def broken(batch):
        raise ValueError("not a memory error")

    with pytest.raises(ValueError, match="not a memory error"):
        ro.run_chunk(rows, prompts, 4, broken)

    def always(batch):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(torch.cuda.OutOfMemoryError):                 # a batch of one is not halved again
        ro.run_chunk(rows[:1], prompts[:1], 1, always)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("e1-rollouts")
    write_checkpoint(root / "model")
    tokenizer = transformers.AutoTokenizer.from_pretrained(root / "model")
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(root / "model")
    return root


def test_a_batch_gives_what_the_models_own_generate_gives_alone(tiny):
    model = transformers.Gemma4ForConditionalGeneration.from_pretrained(tiny / "model", dtype=torch.float32).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny / "model")
    prompts = [chat_ids(tokenizer, text) for text in ("Hi.", "Count to five, slowly.", "Name a colour, then two.")]

    def alone(ids, stops):
        out = model.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                             max_new_tokens=12, do_sample=False, temperature=None, top_p=None, top_k=None,
                             eos_token_id=stops)
        return out[0, len(ids):].tolist()

    # A stop token the random model does emit: the fourth token of the second prompt's continuation alone.
    stop = alone(prompts[1], [1])[3]
    stops = sorted({1, stop})
    expected = []
    for ids in prompts:
        tokens = alone(ids, stops)
        end = next((i for i, t in enumerate(tokens) if t in stops), None)
        expected.append((tokens, None) if end is None else (tokens[:end], tokens[end]))
    got = ro.greedy_batch(model, tokenizer, prompts, 12, stops)
    assert got == expected
    assert any(s is not None for _, s in got)                          # the stop path is exercised
    assert ro.stop_ids_of(model) == [1]


def _write_parquet(path, rows):
    pq.write_table(pa.table({"prompt": [r[1] for r in rows], "prompt_id": [r[0] for r in rows],
                             "messages": [[] for _ in rows]}), path)


def test_the_writer_resumes_by_chunks_and_adds_them_up(tiny, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_parquet(data / "train_sft-0000.parquet", [(f"t{i}", f"Say {i} twice.") for i in range(5)] + [("blank", " ")])
    _write_parquet(data / "train_sft-0001.parquet", [(f"u{i}", f"Spell {i}.") for i in range(4)])
    with open(tmp_path / "dev.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "d0", "prompt": "Say 3 twice."}) + "\n")
    out = tmp_path / "rollouts"
    common = ["write", "--model", str(tiny / "model"), "--parquet", str(data / "train_sft-0001.parquet"),
              str(data / "train_sft-0000.parquet"), "--start", "2", "--count", "6", "--chunk", "4",
              "--batch-size", "3", "--max-new-tokens", "6", "--dtype", "fp32", "--device", "cpu",
              "--dev-prompts", str(tmp_path / "dev.jsonl"), "--out", str(out)]
    assert ro.main([*common, "--max-chunks", "1"]) == 0
    assert (out / "chunk-00000.json").exists() and not (out / "chunk-00001.json").exists()
    first = [json.loads(line) for line in open(out / "chunk-00000.jsonl", encoding="utf-8")]
    # Files in name order; qualifying numbers 0 to 8 are t0..t4, u0..u3, so 2 to 7 are t2, t3, t4, u0, u1, u2.
    assert [r["id"] for r in first] == ["ultrachat/train_sft/t2", "ultrachat/train_sft/t3",
                                        "ultrachat/train_sft/t4", "ultrachat/train_sft/u0"]
    assert [r["number"] for r in first] == [2, 3, 4, 5]
    writer = json.loads((out / "writer.json").read_text())
    assert writer["dev_overlap"] == ["ultrachat/train_sft/t3"] and writer["chunks"] == 2 and writer["stop_ids"] == [1]
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny / "model")
    texts = {f"ultrachat/train_sft/t{i}": f"Say {i} twice." for i in range(5)}
    texts.update({f"ultrachat/train_sft/u{i}": f"Spell {i}." for i in range(4)})
    for record in first:
        assert record["prompt_ids"] == chat_ids(tokenizer, texts[record["id"]])
        assert len(record["response_ids"]) <= 6 and (record["stop_id"] is not None or len(record["response_ids"]) == 6)
        assert record["text"] == tokenizer.decode(record["response_ids"], skip_special_tokens=True)
    before = (out / "chunk-00000.jsonl").read_text()
    assert ro.main(common) == 0                                       # resumes: only chunk 1 is written
    assert (out / "chunk-00000.jsonl").read_text() == before
    second = [json.loads(line) for line in open(out / "chunk-00001.jsonl", encoding="utf-8")]
    assert [r["id"] for r in second] == ["ultrachat/train_sft/u1", "ultrachat/train_sft/u2"]
    assert [r["id"] for r in ro.read_rollouts(out)] == [r["id"] for r in first + second]
    assert [r["id"] for r in ro.read_rollouts(out, limit=5)] == [r["id"] for r in (first + second)[:5]]
    assert ro.main(common) == 0                                       # nothing left to write
    with pytest.raises(SystemExit, match="other settings"):
        ro.main([*common[:-1], str(out), "--max-new-tokens", "7"])
    assert ro.main(["summary", "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    records = first + second
    assert summary["rollouts"] == 6 and summary["chunks_written"] == 2 and summary["dev_overlap"] == 1
    assert summary["response_tokens"] == sum(len(r["response_ids"]) for r in records)
    assert summary["budget_stops"] == sum(r["stop_id"] is None for r in records)
    metas = [json.loads((out / f"chunk-0000{k}.json").read_text()) for k in (0, 1)]
    assert summary["rollouts_per_second"] == pytest.approx(6 / sum(m["seconds"] for m in metas))
    # A chunk whose metadata is missing (stopped while writing it) is not read, and is written again.
    (out / "chunk-00001.json").unlink()
    assert [r["id"] for r in ro.read_rollouts(out)] == [r["id"] for r in first]
    with pytest.raises(SystemExit, match="only 7 of 8"):                # numbers 2 to 8 exist
        ro.main([*common[:common.index("--count") + 1], "8", *common[common.index("--count") + 2:]])
