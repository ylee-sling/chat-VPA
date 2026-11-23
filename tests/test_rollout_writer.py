# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""E0's rollout writer (``chat_vpa.rollout_writer``, C6 of docs/e0.md): the kept set is the first quota of kept
examples of each format in source order, whatever the chunking; a stopped run resumes to the same set; a batch
that runs out of memory is redone at half the size; the stream may not wrap. The selection is checked with a
generator whose decisions are fixed by each example's own index (an explicit loop gives the expected set), the
whole command on a tiny random Gemma 4 saved in the released layout."""

from __future__ import annotations

import gzip
import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4ForConditionalGeneration"):      # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4", allow_module_level=True)

from test_stage0 import CharTokenizer, _write_corpus  # noqa: E402
from test_stage0_cli import write_checkpoint  # noqa: E402

from chat_vpa import rollout_writer as rw  # noqa: E402
from chat_vpa.data import CorpusStream, Example  # noqa: E402
from chat_vpa.rollouts import Rollout  # noqa: E402


def test_quotas_and_batch_seeds():
    assert rw.quotas(100_000, 0.135) == {"complete": 86_500, "write": 13_500}      # C6
    assert rw.quotas(256, 0.135) == {"complete": 221, "write": 35}                  # round(34.56) = 35
    with pytest.raises(ValueError):
        rw.quotas(0, 0.1)
    seeds = {rw.batch_seed(0, c, b) for c in range(50) for b in range(50)}
    assert len(seeds) == 2_500 and rw.batch_seed(1, 0, 0) != rw.batch_seed(0, 0, 0)


def _examples(count: int) -> list[Example]:
    """Example i has kind 'write' when i % 7 == 0 and a prompt [2, 100 + i, 5 x (i % 4)] (lengths differ, so a
    chunk's sort by prompt length reorders it)."""
    return [Example([2, 100 + i, *([5] * (i % 4))], [9, 9, 1], "write" if i % 7 == 0 else "complete")
            for i in range(count)]


def _keeps(index: int, kind: str) -> bool:
    return index % 2 == 0 if kind == "write" else index % 3 != 0


class _Fake:
    """Rollouts decided by the example's index (its prompt's second token - 100); records every call."""

    def __init__(self, fail_above: int | None = None) -> None:
        self.fail_above, self.calls = fail_above, []

    def __call__(self, examples, seed):
        self.calls.append(([ex.prompt_ids[1] - 100 for ex in examples], [len(ex.prompt_ids) for ex in examples],
                           seed))
        if self.fail_above is not None and len(examples) > self.fail_above:
            raise torch.OutOfMemoryError("fake")
        out = []
        for ex in examples:
            index = ex.prompt_ids[1] - 100
            if _keeps(index, ex.kind):
                out.append(Rollout(ex.kind, "eos", 2, "x\n", [7, index % 50 + 10, 1]))
            else:
                out.append(Rollout(ex.kind, "budget", 256, "x\n", None))
        return out


def _expected(examples: list[Example], quota: dict[str, int]) -> list[int]:
    taken, out = {"complete": 0, "write": 0}, []
    for i, ex in enumerate(examples):
        if _keeps(i, ex.kind) and taken[ex.kind] < quota[ex.kind]:
            taken[ex.kind] += 1
            out.append(i)
    return out


@pytest.mark.parametrize("chunk_size", [1, 4, 9, 64])
def test_the_kept_set_is_the_first_quota_of_each_format_in_source_order(tmp_path, chunk_size):
    examples, quota = _examples(300), {"complete": 30, "write": 6}
    fake = _Fake()
    done = rw.fill_quotas(lambda saved: rw.ListSource(examples, saved["read"] if saved else 0), quota,
                          tmp_path / "chunks", chunk_size, 4, 11, fake, log=lambda line: None)
    assert done
    records = rw.finalize(tmp_path / "chunks", quota)
    assert [r["stream_index"] for r in records] == _expected(examples, quota)
    assert all(r["rollout_ids"] == [7, r["stream_index"] % 50 + 10, 1] and r["reference_ids"] == [9, 9, 1]
               and r["prompt_ids"] == examples[r["stream_index"]].prompt_ids for r in records)
    # Each chunk is sorted by prompt length into batches of 4 with the seeds of (seed, chunk, batch).
    state = json.loads((tmp_path / "chunks" / "state.json").read_text())
    calls = iter(fake.calls)
    for entry in state["log"]:
        size = sum(entry["candidates"].values())
        for number in range((size + 3) // 4):
            indices, lengths, seed = next(calls)
            assert seed == rw.batch_seed(11, entry["chunk"], number) and lengths == sorted(lengths)
    assert next(calls, None) is None
    # A format whose quota was full after a chunk has no candidates in later chunks.
    kept = {"complete": 0, "write": 0}
    for entry in state["log"]:
        for f in kept:
            if kept[f] >= quota[f]:
                assert entry["candidates"].get(f, 0) == 0
            kept[f] += entry["kept"][f]


def test_a_stopped_run_resumes_to_the_same_set(tmp_path):
    examples, quota = _examples(300), {"complete": 30, "write": 6}

    def source(saved):
        return rw.ListSource(examples, saved["read"] if saved else 0)

    whole = tmp_path / "whole"
    assert rw.fill_quotas(source, quota, whole, 5, 4, 3, _Fake(), log=lambda line: None)
    parts = tmp_path / "parts"
    assert not rw.fill_quotas(source, quota, parts, 5, 4, 3, _Fake(), max_chunks=2, log=lambda line: None)
    assert json.loads((parts / "state.json").read_text())["chunks"] == 2
    assert not rw.fill_quotas(source, quota, parts, 5, 4, 3, _Fake(), max_chunks=3, log=lambda line: None)
    assert rw.fill_quotas(source, quota, parts, 5, 4, 3, _Fake(), log=lambda line: None)
    assert rw.finalize(parts, quota) == rw.finalize(whole, quota)
    strip = [{k: v for k, v in e.items() if k != "seconds"} for e in json.loads((whole / "state.json").read_text())["log"]]
    assert strip == [{k: v for k, v in e.items() if k != "seconds"}
                     for e in json.loads((parts / "state.json").read_text())["log"]]


def test_a_batch_out_of_memory_redoes_the_chunk_at_half_the_size(tmp_path):
    examples, quota = _examples(120), {"complete": 10, "write": 2}

    def source(saved):
        return rw.ListSource(examples, saved["read"] if saved else 0)

    fake = _Fake(fail_above=2)
    assert rw.fill_quotas(source, quota, tmp_path / "a", 8, 8, 0, fake, log=lambda line: None)
    first_chunk = [len(indices) for indices, _, _ in fake.calls[:3]]
    assert first_chunk == [8, 4, 2]                                   # 8 fails, 4 fails, 2 runs
    log = json.loads((tmp_path / "a" / "state.json").read_text())["log"]
    assert {e["batch_size"] for e in log} == {2}
    assert rw.fill_quotas(source, quota, tmp_path / "b", 8, 2, 0, _Fake(), log=lambda line: None)
    assert rw.finalize(tmp_path / "a", quota) == rw.finalize(tmp_path / "b", quota)
    with pytest.raises(torch.OutOfMemoryError):
        rw.fill_quotas(source, quota, tmp_path / "c", 8, 8, 0, _Fake(fail_above=0), log=lambda line: None)


def test_a_format_that_keeps_nothing_stops_the_run(tmp_path):
    examples = [Example([2, 100 + i], [9, 1], "complete") for i in range(50)]

    def never(batch, seed):
        return [Rollout(ex.kind, "budget", 256, "x\n", None) for ex in batch]

    with pytest.raises(SystemExit, match="3 chunks in a row kept no 'complete' example"):
        rw.fill_quotas(lambda saved: rw.ListSource(examples), {"complete": 1, "write": 0}, tmp_path, 2, 2, 0, never,
                       log=lambda line: None, stall_chunks=3)
    with pytest.raises(SystemExit, match="ran out"):
        rw.fill_quotas(lambda saved: rw.ListSource(examples[:3]), {"complete": 5, "write": 0}, tmp_path / "x", 2, 2,
                       0, _Fake(), log=lambda line: None)


def test_the_stream_source_does_not_wrap(tmp_path):
    _write_corpus(tmp_path / "corpus", functions=10, shards=2)
    tok = CharTokenizer()
    source = rw.StreamSource(CorpusStream(tmp_path / "corpus", tok, 64, 128, seed=0))
    for _ in range(10):
        source.next()
    with pytest.raises(SystemExit, match="wrapped into a second pass"):
        source.next()
    lenient = rw.StreamSource(CorpusStream(tmp_path / "corpus", tok, 64, 128, seed=0), allow_wrap=True)
    assert [lenient.next()[0] for _ in range(25)] == list(range(25))
    # The state resumes the stream exactly.
    first = rw.StreamSource(CorpusStream(tmp_path / "corpus", tok, 64, 128, seed=0), allow_wrap=True)
    for _ in range(7):
        first.next()
    saved = json.loads(json.dumps(first.state()))
    again = rw.StreamSource(CorpusStream(tmp_path / "corpus", tok, 64, 128, seed=0, state=saved["stream"]),
                            saved["read"], allow_wrap=True)
    for _ in range(9):
        a, b = first.next(), again.next()
        assert a[0] == b[0] and a[1].prompt_ids == b[1].prompt_ids and a[1].canvas_ids == b[1].canvas_ids


def write_validation_file(corpus, functions: int = 200) -> None:
    """A validation file of ``functions`` rows (the tiny corpus writes 8): enough kept rollouts of both formats
    for the writer's quotas whatever the tiny model draws."""
    rows = []
    for i in range(functions):
        context = f'def g{i}(a, b):\n    """Subtract {i} from the product of a and b."""\n'
        rows.append({"context": context, "target": context + f"    c = a * b - {i}\n    return c\n",
                     "documented": True, "method": False})
    with gzip.open(corpus / "val.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_records_round_trip_through_reproducible_gzip(tmp_path):
    rows = [{"stream_index": 3, "kind": "write", "prompt_ids": [2, 5], "reference_ids": [9, 1],
             "rollout_ids": [8, 1], "stop": "eos", "generated": 2}]
    rw.write_records(tmp_path / "a.jsonl.gz", rows)
    rw.write_records(tmp_path / "b.jsonl.gz", rows)
    assert rw.read_records(tmp_path / "a.jsonl.gz") == rows
    assert (tmp_path / "a.jsonl.gz").read_bytes() == (tmp_path / "b.jsonl.gz").read_bytes()
    with gzip.open(tmp_path / "a.jsonl.gz", "rt") as handle:
        assert json.loads(handle.readline()) == rows[0]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("writer")
    write_checkpoint(root / "model")
    _write_corpus(root / "corpus", functions=60, shards=3)
    write_validation_file(root / "corpus")
    return root


def _cli(tiny, out, *extra):
    return rw.main(["--model", str(tiny / "model"), "--corpus", str(tiny / "corpus"), "--out", str(out),
                    "--kept", "8", "--write-share", "0.25", "--val-examples", "4", "--chunk-size", "6",
                    "--batch-size", "4", "--max-new-tokens", "32", "--canvas-len", "64", "--max-prompt-tokens", "128",
                    "--top-k", "50", "--temperature", "1.0", "--check-every", "4", "--entropy-examples", "3",
                    "--entropy-batch", "2", "--allow-wrap", "--device", "cpu", *extra])


def test_the_writer_on_the_tiny_model(tiny, tmp_path, capsys):
    out = tmp_path / "data"
    assert _cli(tiny, out, "--max-chunks", "1") == 0                  # the validation set and one chunk
    assert (out / "validation.jsonl.gz").exists() and not (out / "train.jsonl.gz").exists()
    assert _cli(tiny, out) == 0                                       # resumes and finishes
    train, validation = rw.read_records(out / "train.jsonl.gz"), rw.read_records(out / "validation.jsonl.gz")
    assert [r["kind"] for r in train].count("write") == 2 and len(train) == 8
    assert [r["kind"] for r in validation].count("write") == 1 and len(validation) == 4
    for row in train + validation:
        assert row["stop"] in ("truncation", "eos") and row["rollout_ids"][-1] == 1 and row["reference_ids"][-1] == 1
        assert row["prompt_ids"][0] == 2 and len(row["rollout_ids"]) <= 64
    indices = [r["stream_index"] for r in train]
    assert indices == sorted(indices)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["train"]["records"] == {"complete": 6, "write": 2}
    assert summary["train"]["quota"] == {"complete": 6, "write": 2}
    assert summary["named_function"]["write_kept"] == 2
    for f, count in (("complete", 6), ("write", 2)):
        rows = [r for r in train if r["kind"] == f]
        lengths = summary["lengths"][f]
        assert lengths["rollout_tokens_mean"] == round(sum(len(r["rollout_ids"]) - 1 for r in rows) / count, 2)
        assert 0 <= lengths["empty_rollouts"] <= count
    for f in ("complete", "write"):
        stats = summary["entropy"][f]
        assert stats["rollouts"]["programs"] == min(3, summary["train"]["records"][f])
        assert "entropy_per_token" in stats["rollouts"] and "entropy_per_token" not in stats["references"]
    assert set(summary["entropy"]["pooled_at_kept_shares"]["shares"]) == {"complete", "write"}
    # An uninterrupted run with the same settings writes the same files.
    other = tmp_path / "again"
    assert _cli(tiny, other) == 0
    for name in ("train.jsonl.gz", "validation.jsonl.gz"):
        assert (other / name).read_bytes() == (out / name).read_bytes()
    capsys.readouterr()
    assert rw.main(["--show", str(out)]) == 0
    assert "finished" in capsys.readouterr().out
    # A run does not resume with other settings.
    with pytest.raises(SystemExit, match="other settings"):
        _cli(tiny, out, "--kept", "9")


def test_the_progress_display_predicts_the_hours_left_and_the_examples_read(tmp_path):
    """The prediction of an unfinished run, against arithmetic done by hand."""
    out = tmp_path / "out"
    quotas = {"validation": {"complete": 3, "write": 1}, "train": {"complete": 100, "write": 20}}
    (out / "validation-chunks").mkdir(parents=True)
    (out / "train-chunks").mkdir()
    (out / "settings.json").write_text(json.dumps({"quotas": quotas}))
    (out / "validation-chunks" / "state.json").write_text(json.dumps(
        {"chunks": 1, "kept": quotas["validation"], "source": {"read": 9},
         "log": [{"candidates": {"complete": 6, "write": 3}, "seconds": 36.0}]}))
    (out / "validation.jsonl.gz").write_bytes(b"")
    # Two chunks: 100 complete and 25 write candidates from 125 examples, 50 and 5 kept, in 3,600 s.
    log = [{"candidates": {"complete": 60, "write": 15}, "seconds": 2_000.0},
           {"candidates": {"complete": 40, "write": 10}, "seconds": 1_600.0}]
    state = {"chunks": 2, "kept": {"complete": 50, "write": 5}, "source": {"read": 125}, "log": log}
    (out / "train-chunks" / "state.json").write_text(json.dumps(state))
    lines = rw.progress_text(out).splitlines()
    assert lines[1] == ("  validation: kept {'complete': 3, 'write': 1} of {'complete': 3, 'write': 1} after 1 chunks, "
                        "9 examples read, 0.01 h; finished")
    # complete: 50 more x 100/50 = 100 candidates; write: 15 more x 25/5 = 75; 175 candidates at 3,600/125 s each
    # = 1.4 h. The source is read until write is full: 125 x 20/5 = 500 examples (complete would need 250).
    assert lines[2] == ("  train: kept {'complete': 50, 'write': 5} of {'complete': 100, 'write': 20} after 2 chunks, "
                        "125 examples read, 1.00 h; about 1.4 h left and 500 examples read in all "
                        "(predicted from the rates so far)")
    # Once complete is full, only write counts.
    state["kept"] = {"complete": 100, "write": 10}
    (out / "train-chunks" / "state.json").write_text(json.dumps(state))
    assert rw.progress_text(out).splitlines()[2].endswith(
        "about 0.2 h left and 250 examples read in all (predicted from the rates so far)")   # 10 x 25/10 x 28.8 s = 720 s
    # An open format that has kept nothing yet gives no prediction.
    state["kept"] = {"complete": 50, "write": 0}
    (out / "train-chunks" / "state.json").write_text(json.dumps(state))
    assert rw.progress_text(out).splitlines()[2].endswith(
        "; too early to predict (a format with an open quota has kept nothing yet)")
