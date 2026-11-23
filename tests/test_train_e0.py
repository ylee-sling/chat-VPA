# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""E0's trainer (``chat_vpa.train_e0``): its configuration holds docs/e0.md's confirmed choices, the training order is
a seeded permutation per epoch that both arms share, a step pools the block loss of its micro-batches as
``dfm_loss`` computes it, only the trained parameters move, a resumed run continues an uninterrupted one exactly,
and the command line trains, stops, resumes and hands its checkpoint to the block evaluator. On a tiny random
Gemma 4; nothing is downloaded."""

from __future__ import annotations

import json
import math
import shutil

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4ForConditionalGeneration"):      # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4", allow_module_level=True)

from test_rollout_writer import write_validation_file  # noqa: E402
from test_stage0 import CharTokenizer, _write_corpus, tiny_realizer  # noqa: E402
from test_stage0_cli import write_benchmarks, write_checkpoint  # noqa: E402

from chat_vpa import block_evaluate, rollout_writer, train_e0  # noqa: E402
from chat_vpa.block_data import block_training_state, collate_blocks  # noqa: E402
from chat_vpa.block_dfm import BlockDFM  # noqa: E402
from chat_vpa.block_objective import dfm_loss  # noqa: E402
from chat_vpa.data import Example  # noqa: E402
from chat_vpa.flow import KappaSchedule  # noqa: E402
from chat_vpa.runtime import save_checkpoint  # noqa: E402
from chat_vpa.train import learning_rate  # noqa: E402


def test_the_configuration_holds_the_confirmed_choices():
    cfg = train_e0.E0Config()
    # C1 to C5, C7 of docs/e0.md (confirmed 23 November 2026)
    assert (cfg.seam_slots, cfg.seam_weight, cfg.pad_weight) == (4, 1.0, 0.1)
    assert (cfg.step_budgets, cfg.prefix_fraction, cfg.schedule) == ([2, 4], 0.5, "cosine")
    assert (cfg.arm, cfg.block_size, cfg.num_registers) == ("rollouts", 16, 0)
    assert (cfg.lora_rank, cfg.lora_alpha, cfg.modulation) == (64, 64.0, True)
    assert (cfg.lr, cfg.min_lr_fraction, cfg.warmup_steps, cfg.weight_decay, cfg.grad_clip) == (2e-4, 0.1, 500, 0.0, 1.0)
    assert (cfg.micro_batch * cfg.grad_accum, cfg.steps, cfg.schedule_steps, cfg.seed) == (32, 10_000, 25_000, 0)
    assert (cfg.eval_every, cfg.timing_steps, cfg.canvas_len, cfg.max_prompt_tokens) == (2_500, 100, 256, 512)
    # 1.41e-4 at the last step (docs/e0.md, C7): the cosine over 24,500 steps after the warm-up, towards 2e-5
    expected = 2e-4 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (9_999 - 500) / 24_500)))
    assert learning_rate(9_999, cfg) == pytest.approx(expected) and round(expected, 6) == 1.41e-4
    assert train_e0.E0Config.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(SystemExit):
        train_e0.E0Config(arm="teacher").check()


def _examples(count: int, target_offset: int = 0) -> list[Example]:
    return [Example([2, 6 + i % 40, 7, 8], [10 + (i + target_offset) % 50, 11, 12, 1], "complete")
            for i in range(count)]


def test_the_epoch_stream_is_a_seeded_permutation_shared_by_both_arms():
    examples = _examples(10)
    stream = train_e0.EpochStream(examples, seed=0)
    taken = stream.take(25)
    for epoch in (taken[:10], taken[10:20]):
        assert sorted(tuple(e.prompt_ids) for e in epoch) == sorted(tuple(e.prompt_ids) for e in examples)
    assert taken[:10] != taken[10:20] and stream.state() == {"epoch": 2, "position": 5}
    # Resumed from its state, the stream goes on exactly.
    first = train_e0.EpochStream(examples, seed=0)
    first.take(13)
    again = train_e0.EpochStream(examples, seed=0, state=json.loads(json.dumps(first.state())))
    assert [e.prompt_ids for e in first.take(12)] == [e.prompt_ids for e in again.take(12)]
    # The other arm (other targets, same prompts) sees the prompts in the same order; another seed does not.
    other_arm = train_e0.EpochStream(_examples(10, target_offset=7), seed=0)
    assert [e.prompt_ids for e in other_arm.take(25)] == [e.prompt_ids for e in taken]
    assert [e.prompt_ids for e in train_e0.EpochStream(examples, seed=1).take(10)] != [e.prompt_ids for e in taken[:10]]


def _trainer(tmp_path, registers: int = 0, examples=None, **settings):
    realizer, _ = tiny_realizer()
    model = BlockDFM(realizer, num_registers=registers, register_seed=3)
    defaults = {"block_size": 4, "micro_batch": 2, "grad_accum": 2, "steps": 6, "warmup_steps": 1, "lr": 0.05,
                "eval_every": 1_000, "save_every": 1_000, "log_every": 1_000, "lora_rank": 4, "lora_alpha": 4.0,
                "num_registers": registers}
    cfg = train_e0.E0Config(**{**defaults, **settings})
    examples = examples or _examples(12)
    stream = train_e0.EpochStream(examples, seed=0)
    validation = {"references": _examples(5), "rollouts": _examples(5, target_offset=3)}
    trainer = train_e0.E0Trainer(model, CharTokenizer(), cfg, stream, validation, tmp_path, torch.device("cpu"))
    return trainer, model, examples


def test_a_step_pools_the_block_loss_of_its_micro_batches(tmp_path):
    trainer, model, examples = _trainer(tmp_path)
    # The reference: the same examples and draws through the lower-level functions, before the update.
    stream = train_e0.EpochStream(examples, seed=0)
    generator = torch.Generator().manual_seed(trainer.cfg.seed)
    nll = weight = 0.0
    with torch.no_grad():
        for _ in range(2):
            batch = collate_blocks(stream.take(2), 4, 1)
            state = block_training_state(batch, KappaSchedule("cosine"), [2, 4], CharTokenizer.mask_id, 0.5, 0.1, 4,
                                         1.0, generator)
            _, stats = dfm_loss(model, batch, state, torch.device("cpu"))
            nll, weight = nll + stats["nll_sum"], weight + stats["weight_sum"]
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    trained = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    metrics = trainer.train_step()
    assert metrics["loss"] == pytest.approx(nll / weight, rel=1e-5, abs=1e-6)
    assert trainer.step == 1 and trainer.examples_seen == 4 and metrics["blocks"] >= 4
    assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if not p.requires_grad)
    assert any(not torch.equal(p, trained[n]) for n, p in model.named_parameters() if p.requires_grad)


def test_training_learns_one_repeated_example(tmp_path):
    """The masked tokens become the most probable ones. The loss falls by more than half a nat but stays near
    4.6: the tiny random model's frozen output layer is small, so even a right argmax gets about 1% of the
    probability (ln 320 = 5.8 is the uniform loss)."""
    one = [Example([2, 9, 7, 8], [20, 21, 22, 23, 24, 25, 1], "complete")]
    trainer, _, _ = _trainer(tmp_path, examples=one, steps=50, lr=0.03)
    steps = [trainer.train_step() for _ in range(50)]
    first, last = steps[:5], steps[-5:]
    assert steps[0]["accuracy"] <= 0.2 and min(m["accuracy"] for m in last) >= 0.9
    assert sum(m["loss"] for m in last) / 5 < sum(m["loss"] for m in first) / 5 - 0.5


def test_a_resumed_run_continues_an_uninterrupted_one_exactly(tmp_path):
    whole, whole_model, _ = _trainer(tmp_path / "whole", registers=2)
    for _ in range(4):
        whole.train_step()
    part, _, _ = _trainer(tmp_path / "part", registers=2)
    for _ in range(2):
        part.train_step()
    part.save()
    payload = torch.load(tmp_path / "part" / "last.pt", weights_only=True)
    assert payload["stage"] == "e0" and payload["registers"].shape[0] == 2 and payload["step"] == 2
    again, again_model, _ = _trainer(tmp_path / "again", registers=2)
    again.stream = train_e0.EpochStream(again.stream.examples, 0, payload["stream"])
    again.restore(payload)
    for _ in range(2):
        again.train_step()
    for (name, a), (_, b) in zip(whole_model.named_parameters(), again_model.named_parameters(), strict=True):
        assert torch.equal(a, b), name
    assert again.stream.state() == whole.stream.state() and again.step == whole.step == 4


def test_validation_reports_both_targets_at_every_budget(tmp_path):
    trainer, _, _ = _trainer(tmp_path)
    first, second = trainer.validate(), trainer.validate()
    assert first == second                                            # fixed draws
    halved, _, _ = _trainer(tmp_path / "halved", micro_batch=1, grad_accum=4)
    assert halved.validate() == first                                 # whatever the memory check made of it
    for target in ("references", "rollouts"):
        for budget in (2, 4):
            assert math.isfinite(first[f"val_{target}_loss_K{budget}"])
            assert 0.0 <= first[f"val_{target}_accuracy_K{budget}"] <= 1.0
        assert math.isfinite(first[f"val_{target}_loss"])
    assert first["val_references_loss"] != first["val_rollouts_loss"]
    assert trainer.probe() == {}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """A tiny model, benchmarks and the rollout writer's data for it."""
    root = tmp_path_factory.mktemp("e0")
    write_checkpoint(root / "model")
    _write_corpus(root / "corpus", functions=60, shards=3)
    write_validation_file(root / "corpus")
    write_benchmarks(root / "eval")
    assert rollout_writer.main([
        "--model", str(root / "model"), "--corpus", str(root / "corpus"), "--out", str(root / "data"),
        "--kept", "10", "--write-share", "0.2", "--val-examples", "4", "--chunk-size", "8", "--batch-size", "4",
        "--max-new-tokens", "32", "--canvas-len", "64", "--max-prompt-tokens", "128", "--top-k", "50",
        "--temperature", "1.0", "--check-every", "4", "--entropy-examples", "2", "--allow-wrap", "--device", "cpu"]) == 0
    return root


def test_the_command_line_trains_stops_resumes_and_hands_over_to_the_evaluator(data, capsys):
    run = data / "runs" / "T4"
    settings = ["arm=rollouts", "block_size=4", "num_registers=2", "steps=6", "micro_batch=2", "grad_accum=1",
                "warmup_steps=1", "save_every=3", "eval_every=3", "log_every=1", "timing_steps=2", "lr=0.05",
                "lora_rank=4", "lora_alpha=4", "canvas_len=64", "max_prompt_tokens=128"]
    args = ["--data", str(data / "data"), "--run", str(run), "--model", str(data / "model"), "--device", "cpu"]
    assert train_e0.main([*args, "--set", *settings, "--stop-at", "3"]) == 0
    payload = torch.load(run / "last.pt", weights_only=True)
    assert payload["step"] == 3 and payload["config"]["steps"] == 6 and payload["config"]["arm"] == "rollouts"
    timing = json.loads((run / "timing.json").read_text())
    assert timing["steps"] == 2 and timing["steps_per_s"] > 0
    assert timing["projected_hours_for_the_run"] == pytest.approx(6 / timing["steps_per_s"] / 3600, abs=1e-4)
    assert json.loads((run / "memory.json").read_text())["micro_batch"] == 2
    assert train_e0.main(args) == 0                                   # resumes to the configured 6 steps
    payload = torch.load(run / "last.pt", weights_only=True)
    config = json.loads((run / "config.json").read_text())
    assert payload["step"] == 6 and payload["data_fingerprint"] == config["data_fingerprint"]
    assert config["train_examples"] == 10 and config["validation_examples"] == 4 and config["tokenizer"]["mask"] == 4
    assert (run / "step-000003.pt").exists() and (run / "step-000006.pt").exists()
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in rows if "eval/val_rollouts_loss" in r]
    assert [r["step"] for r in evals] == [0, 3, 6]
    assert all("eval/val_references_loss_K2" in r and "eval/val_rollouts_loss_K4" in r for r in evals)
    assert [r["step"] for r in rows if "train/loss" in r] == [1, 2, 3, 4, 5, 6]
    # The other arm's run reads the same prompts with the references.
    other = data / "runs" / "R4"
    assert train_e0.main(["--data", str(data / "data"), "--run", str(other), "--model", str(data / "model"),
                          "--device", "cpu", "--no-memory-check", "--set", "arm=references",
                          *[s for s in settings if not s.startswith("arm=")], "--stop-at", "1"]) == 0
    assert torch.load(other / "last.pt", weights_only=True)["stream"] == {"epoch": 0, "position": 2}
    # The block evaluator reads the E0 checkpoint: its registers, and its block size unless told otherwise.
    out = data / "eval-out"
    assert block_evaluate.main(["--model", str(data / "model"), "--checkpoint", str(run / "last.pt"), "--eval-dir",
                                str(data / "eval"), "--benchmarks", "mbpp", "--mbpp-split", "validation",
                                "--num-samples", "2", "--k", "1", "--k-steps", "2", "--canvas-len", "16", "--out",
                                str(out), "--device", "cpu"]) == 0
    summary = json.loads((out / "mbpp" / "block-B4-K2" / "summary.json").read_text())
    assert (summary["checkpoint_stage"], summary["arm"], summary["train_block_size"], summary["registers"]) == (
        "e0", "rollouts", 4, 2)
    assert summary["pass_audit"] == "passed" and summary["step"] == 6
    # A run does not resume on other data, and a Stage 0 checkpoint is not an E0 run.
    changed = data / "data-changed"
    shutil.copytree(data / "data", changed)
    records = rollout_writer.read_records(changed / "train.jsonl.gz")
    rollout_writer.write_records(changed / "train.jsonl.gz", records[1:] + records[:1])
    with pytest.raises(SystemExit, match="not the file this run started with"):
        train_e0.main(["--data", str(changed), "--run", str(run), "--model", str(data / "model"), "--device", "cpu"])
    stage0 = data / "runs" / "stage0"
    save_checkpoint(stage0 / "last.pt", {"stage": "stage0", "step": 1})
    with pytest.raises(SystemExit, match="not an E0 run"):
        train_e0.main(["--data", str(data / "data"), "--run", str(stage0), "--model", str(data / "model"),
                       "--device", "cpu"])
    capsys.readouterr()
