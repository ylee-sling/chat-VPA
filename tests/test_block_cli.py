# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""The block evaluator and the sample comparison through their command lines, on a tiny random Gemma 4 checkpoint
saved to disk: a block that spans the canvas reproduces the whole-canvas evaluator's samples (the port check of
Task D in miniature), short blocks run under the pass audit, and the cache check compares the fused commits with
the frozen backbone."""

from __future__ import annotations

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
if not hasattr(transformers, "Gemma4ForConditionalGeneration"):      # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4", allow_module_level=True)

from test_stage0 import _write_corpus  # noqa: E402
from test_stage0_cli import write_benchmarks, write_checkpoint  # noqa: E402

from chat_vpa import block_evaluate, compare_samples, evaluate, train  # noqa: E402
from chat_vpa.block_evaluate import block_tag  # noqa: E402
from chat_vpa.block_sampler import BlockSamplerConfig  # noqa: E402


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A tiny checkpoint, benchmarks and a Stage 0 run of a few steps (random adapters would do; these are trained)."""
    root = tmp_path_factory.mktemp("block-cli")
    write_checkpoint(root / "model")
    _write_corpus(root / "corpus", functions=24, shards=2)
    write_benchmarks(root / "eval")
    settings = [f"model={root / 'model'}", "canvas_len=32", "max_prompt_tokens=128", "lora_rank=4", "lora_alpha=4",
                "steps=3", "micro_batch=2", "grad_accum=1", "warmup_steps=1", "save_every=3", "eval_every=3",
                "val_examples=2", "probe_samples=1", "probe_steps=2", "log_every=3", "lr=0.05"]
    assert train.main(["--corpus", str(root / "corpus"), "--run", str(root / "stage0"), "--device", "cpu",
                       "--no-memory-check", "--set", *settings]) == 0
    return root


def test_a_block_spanning_the_canvas_reproduces_the_whole_canvas_evaluator(run, capsys):
    shared = ["--model", str(run / "model"), "--checkpoint", str(run / "stage0" / "last.pt"), "--eval-dir",
              str(run / "eval"), "--benchmarks", "mbpp", "--mbpp-split", "validation", "--num-samples", "3",
              "--k", "1", "--device", "cpu"]
    assert evaluate.main(["--decoder", "realizer", "--k-steps", "4", "8", "--out", str(run / "canvas"), *shared]) == 0
    assert block_evaluate.main(["--block-size", "32", "--canvas-len", "32", "--k-steps", "4", "8", "--out",
                                str(run / "block"), *shared]) == 0             # one block: the run's canvas of 32
    capsys.readouterr()
    for steps in (4, 8):
        canvas = run / "canvas" / "mbpp" / f"realizer-K{steps}" / "samples.jsonl"
        block = run / "block" / "mbpp" / f"block-B32-K{steps}" / "samples.jsonl"
        assert compare_samples.main([str(canvas), str(block)]) == 0
        printed = capsys.readouterr().out
        assert "identical raw text: 6/6 (100.0%)" in printed and "RESULT: identical" in printed
        assert "settings recorded by both: equal" in printed
        assert compare_samples.setting_differences(canvas, block) == []
        a, b = compare_samples.load_samples(canvas), compare_samples.load_samples(block)
        assert len(a) == 6 and all(a[key]["raw"] == b[key]["raw"] and a[key]["passed"] == b[key]["passed"] for key in a)
        assert len({row["raw"] for row in a.values()}) > 1                       # not one degenerate text
        summary = json.loads((block.parent / "summary.json").read_text())
        assert summary["decoder"] == "block" and summary["max_blocks"] == 1 and summary["pass_audit"] == "passed"
        assert summary["mean_passes_per_block"] == steps and summary["mean_sequential_depth"] == steps + 1
    # Samples of another K differ, and the comparison says so.
    other = run / "block" / "mbpp" / "block-B32-K4" / "samples.jsonl"
    assert compare_samples.main([str(canvas), str(other)]) == 1
    printed = capsys.readouterr().out
    assert "RESULT: DIFFERENT" in printed
    # ... and that the two were not run with the same settings: the summaries next to the samples are compared.
    steps_of = json.loads(canvas.with_name("summary.json").read_text())["num_steps"]
    assert compare_samples.setting_differences(canvas, other) == [("num_steps", steps_of, 4)]
    assert f"SETTINGS DIFFER (the samples are not expected to agree): num_steps {steps_of} against 4" in printed
    # Without a summary next to a file the settings are not compared, and the comparison says so.
    alone = run / "alone" / "samples.jsonl"
    alone.parent.mkdir()
    alone.write_text(other.read_text())
    assert compare_samples.setting_differences(alone, other) is None
    assert compare_samples.main([str(alone), str(other)]) == 0
    assert "settings: not compared" in capsys.readouterr().out


def test_short_blocks_under_the_pass_audit_with_a_cache_check(run):
    shared = ["--model", str(run / "model"), "--checkpoint", str(run / "stage0" / "last.pt"), "--eval-dir",
              str(run / "eval"), "--benchmarks", "mbpp", "--num-samples", "2", "--k", "1", "--device", "cpu",
              "--canvas-len", "32", "--out", str(run / "short")]
    assert block_evaluate.main(["--block-size", "4", "--k-steps", "2", "--orders", "exact", "prefix", "confidence",
                                "--cache-check", "2", *shared]) == 0
    assert block_evaluate.main(["--block-size", "4", "--k-steps", "2", "--separate-commit", *shared]) == 0
    assert block_evaluate.main(["--block-size", "4", "--k-steps", "4", "--orders", "prefix", "--exit-budget", "50",
                                *shared]) == 0
    tags = sorted(d.name for d in (run / "short" / "mbpp").iterdir())
    assert tags == ["block-B4-K2", "block-B4-K2-confidence", "block-B4-K2-prefix", "block-B4-K2-separate",
                    "block-B4-K4-prefix-exit50"]
    assert block_tag(BlockSamplerConfig(16, 4)) == "block-B16-K4"
    assert block_tag(BlockSamplerConfig(32, 2, order="prefix", exit_budget=0.05), fused=False) == \
        "block-B32-K2-prefix-exit0.05-separate"
    exact = json.loads((run / "short" / "mbpp" / "block-B4-K2" / "summary.json").read_text())
    assert exact["problems"] == 2 and exact["samples_per_problem"] == 2 and exact["max_blocks"] == 8
    assert exact["pass_audit"] == "passed" and exact["fused_commit"] is True and exact["registers"] == 0
    assert 1.0 <= exact["mean_passes_per_block"] <= 2.0 and 1 <= exact["mean_blocks"] <= 8
    assert abs(sum(exact["stops"].values()) - 1.0) < 1e-3 and exact["seconds_per_block"] > 0
    assert [c["task_id"] for c in exact["cache_check"]] == ["MBPP/511", "MBPP/512"]
    for check in exact["cache_check"]:                                           # (bfloat16 weights, as released)
        assert check["blocks"] < 2 or {"fused_commit", "separate_commit", "adapters_on"} <= set(check)
    # A10, a separate commit pass: the same samples in float32 (tests/test_block_dfm.py). These weights are
    # bfloat16, as released, where the two differ by rounding and this random model's samples may differ with it.
    separate = json.loads((run / "short" / "mbpp" / "block-B4-K2-separate" / "summary.json").read_text())
    assert separate["fused_commit"] is False and separate["pass_audit"] == "passed"
    exited = json.loads((run / "short" / "mbpp" / "block-B4-K4-prefix-exit50" / "summary.json").read_text())
    assert exited["mean_passes_per_block"] == 1.0 and exited["exit_budget"] == 50.0   # every block exits at once
    with pytest.raises(SystemExit):                                              # not a Stage 0 checkpoint
        path = run / "other.pt"
        torch.save({"stage": "e0", "spec": {}, "trainable": {}, "config": {}, "step": 0}, path)
        block_evaluate.main(["--checkpoint", str(path), "--model", str(run / "model"), "--eval-dir", str(run / "eval"),
                             "--out", str(run / "x"), "--device", "cpu"])
