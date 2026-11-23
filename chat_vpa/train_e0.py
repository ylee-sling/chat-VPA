# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
train_e0.py -- E0's trainer: Block-DFM on the frozen Gemma 4 E2B, one run per arm and block size (docs/e0.md).

    python -m chat_vpa.train_e0 --model data2/models/gemma-4-E2B --data runs/e0-data --run runs/e0-T16 \\
        --set arm=rollouts block_size=16

On the server: ``bash scripts/ncp/e0_train.sh T16`` (T: the rollout arm, R: the reference arm; 16 or 32).

What docs/e0.md fixes (confirmed by the author on 23 November 2026), and where it lives here:

    C1  L_DFM (Eq. 12) alone; weights 1 + lambda_seam = 2 on the first b0 = 4 slots of every block, 0.1 on the
        <eos> filling, 1 elsewhere                             ``seam_slots``, ``seam_weight``, ``pad_weight``
    C2  K drawn per block from {2, 4}, t = k/K on its grid, prefix-shaped states with probability 0.5, cosine
                                                               ``step_budgets``, ``prefix_fraction``, ``schedule``
    C3  one run per arm and block size                         ``arm`` (references | rollouts), ``block_size``
    C4  no registers                                           ``num_registers`` = 0
    C5  fresh adapters: LoRA rank 64 at its initial values, the mask vector at the <mask> embedding, the (t, h)
        modulation at zero                                     a new run never loads another run's weights
    C6  the data of ``rollout_writer.py``: the same kept prompts for both arms; ``arm`` picks the target
    C7  runs/stage0-a's optimisation: AdamW (0.9, 0.999) without weight decay, learning rate 2e-4 with 500 warm-up
        steps and cosine decay towards 2e-5 over a horizon of 25,000 steps, clipping at 1.0, bfloat16 autocast,
        32 examples per step (8 x 4, changed only by the memory check), seed 0, 10,000 steps; held-out loss every
        2,500 steps on the validation set, on both targets; the final checkpoint is the one evaluated

The loop, the optimiser, the learning-rate schedule, checkpoints and logging are Stage 0's (``train.Stage0Trainer``);
what E0 changes is behind its hooks. A micro-batch is collated into blocks (``block_data.collate_blocks``: an
example of L target tokens, its <eos> included, has ceil(L / B) blocks), every block draws its state
(``block_training_state``), and the loss is ``block_objective.dfm_loss`` on one staircase training pass: the clean
stream is the frozen backbone's prefill without gradients, the window rows of all blocks one pass with them. The
examples come from ``EpochStream``: a seeded permutation of the kept set per epoch, the same for both arms and both
block sizes. Each run sees 320,000 examples, so each of the 100,000 kept examples about 3.2 times.

Held-out loss (C7): on the validation set's references and on its rollouts, at every budget K of the run, with a
fixed draw of states per (target, K): ``val_<target>_loss_K<K>``, its accuracy, and both pooled over K. Read with
H2: a run whose held-out loss on its own target rises while its training loss falls is fitting the repeated
examples.

``--stop-at N`` ends the loop after step N without changing the run's configuration, and a later start resumes
from ``last.pt``. After the first ``timing_steps`` steps of a run (100), ``RUN/timing.json`` holds the steps per
second, the peak memory and the hours projected for the run (docs/e0.md, Section 4: sent to the author before
the run continues).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from chat_vpa.block_data import BlockBatch, BlockState, block_training_state, collate_blocks
from chat_vpa.block_dfm import BlockDFM
from chat_vpa.block_objective import dfm_loss
from chat_vpa.data import Example, Stage0Tokenizer
from chat_vpa.evaluate import autocast
from chat_vpa.runtime import MetricLogger, load_checkpoint
from chat_vpa.train import Stage0Config, Stage0Trainer, _overrides, learning_rate

__all__ = ["ARMS", "E0Config", "E0Trainer", "EpochStream", "RolloutData"]

ARMS: dict[str, str] = {"references": "reference_ids", "rollouts": "rollout_ids"}


@dataclass
class E0Config(Stage0Config):
    """Stage 0's settings with E0's values (docs/e0.md, C1 to C7) and E0's own fields."""

    step_budgets: list[int] = field(default_factory=lambda: [2, 4])          # C2
    prefix_fraction: float = 0.5                                             # C2
    steps: int = 10_000                                                      # C7
    schedule_steps: int = 25_000                                             # C7: the horizon of the cosine
    arm: str = "rollouts"                                                    # C3: references | rollouts
    block_size: int = 16                                                     # C3
    num_registers: int = 0                                                   # C4
    register_seed: int = 0
    seam_slots: int = 4                                                      # C1: b0
    seam_weight: float = 1.0                                                 # C1: lambda_seam
    data_seed: int = 0                     # the permutation of the kept set per epoch, the same in every run
    timing_steps: int = 100                # docs/e0.md, Section 4
    val_micro_batch: int = 8               # fixed, so that the memory check cannot change the held-out states
    data: str = ""                         # the rollout writer's directory (recorded)

    def check(self) -> E0Config:
        if self.arm not in ARMS:
            raise SystemExit(f"arm must be one of {sorted(ARMS)}, not {self.arm!r}")
        if self.block_size < 1 or self.num_registers < 0 or not self.step_budgets:
            raise SystemExit("need block_size >= 1, num_registers >= 0 and at least one step budget")
        return self


class RolloutData:
    """The rollout writer's output: ``train.jsonl.gz`` and ``validation.jsonl.gz``, each record with both targets."""

    def __init__(self, directory: str | Path) -> None:
        from chat_vpa.rollout_writer import read_records

        self.directory = Path(directory)
        for name in ("settings.json", "train.jsonl.gz", "validation.jsonl.gz"):
            if not (self.directory / name).exists():
                raise SystemExit(f"{self.directory / name} is missing: the rollout writer has not finished")
        self.settings = json.loads((self.directory / "settings.json").read_text())
        self.records = {name: read_records(self.directory / f"{name}.jsonl.gz") for name in ("train", "validation")}

    def examples(self, split: str, arm: str) -> list[Example]:
        key = ARMS[arm]
        return [Example(list(r["prompt_ids"]), list(r[key]), r["kind"]) for r in self.records[split]]

    def fingerprint(self, split: str) -> str:
        return hashlib.sha256((self.directory / f"{split}.jsonl.gz").read_bytes()).hexdigest()


class EpochStream:
    """The kept examples, epoch after epoch, each epoch a seeded permutation. The permutation depends only on the
    seed, the epoch and the number of examples, so both arms (whose examples differ in the target only) and both
    block sizes see the prompts in the same order. ``state()`` -- (epoch, position) -- resumes it exactly."""

    def __init__(self, examples: Sequence[Example], seed: int, state: dict[str, int] | None = None) -> None:
        if not examples:
            raise ValueError("no examples")
        self.examples, self.seed = list(examples), seed
        self.epoch, self.position = (int(state["epoch"]), int(state["position"])) if state else (0, 0)
        self._order = self.permutation(self.epoch)

    def permutation(self, epoch: int) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
        return torch.randperm(len(self.examples), generator=generator).tolist()

    def take(self, count: int) -> list[Example]:
        out: list[Example] = []
        while len(out) < count:
            if self.position == len(self._order):
                self.epoch, self.position = self.epoch + 1, 0
                self._order = self.permutation(self.epoch)
            out.append(self.examples[self._order[self.position]])
            self.position += 1
        return out

    def state(self) -> dict[str, int]:
        return {"epoch": self.epoch, "position": self.position}


class E0Trainer(Stage0Trainer):
    """``Stage0Trainer`` with E0's batches, states and loss (see the module docstring)."""

    stage = "e0"

    def __init__(self, model: BlockDFM, tokenizer: Stage0Tokenizer, cfg: E0Config, stream: EpochStream,
                 validation: dict[str, Sequence[Example]], run_dir: str | Path, device: torch.device,
                 logger: MetricLogger | None = None) -> None:
        self.model = model                                   # before the optimiser is built from it
        super().__init__(model.realizer, tokenizer, cfg, stream, [], run_dir, device, (), logger)
        self.cfg: E0Config = cfg
        self.validation_sets = {target: list(examples) for target, examples in validation.items()}
        self.data_fingerprint: str | None = None             # recorded in every checkpoint when set
        self._timing: dict[str, Any] | None = None

    # -- the hooks -----------------------------------------------------------------------------
    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return self.model.trainable_parameters()

    def set_training(self, mode: bool) -> None:
        self.model.train(mode)

    def collate(self, examples: Sequence[Example]) -> BlockBatch:
        return collate_blocks(examples, self.cfg.block_size, self.tokenizer.eos_id)

    def state_for(self, batch: BlockBatch, budget: int | None = None, step_index: int | None = None,
                  generator: torch.Generator | None = None) -> BlockState:
        cfg = self.cfg
        return block_training_state(batch, self.schedule, cfg.step_budgets, self.tokenizer.mask_id,
                                    cfg.prefix_fraction, cfg.pad_weight, cfg.seam_slots, cfg.seam_weight,
                                    generator or self.generator, budget, step_index)

    def loss_on(self, batch: BlockBatch, budget: int | None = None, step_index: int | None = None,
                generator: torch.Generator | None = None, state: BlockState | None = None,
                mode: str = "bidirectional", adapters: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
        if mode != "bidirectional":
            raise ValueError("E0's windows are bidirectional")
        if state is None:
            state = self.state_for(batch, budget, step_index, generator)
        return dfm_loss(self.model, batch, state, self.device,
                        checkpointing=self.cfg.grad_checkpoint and torch.is_grad_enabled(), adapters=adapters,
                        autocast=autocast(self.device))

    def extra_state(self) -> dict[str, Any]:
        out: dict[str, Any] = {"registers": self.model.registers.detach().cpu()} if self.model.num_registers else {}
        if self.data_fingerprint is not None:
            out["data_fingerprint"] = self.data_fingerprint
        return out

    def load_extra_state(self, payload: dict[str, Any]) -> None:
        if self.model.num_registers:
            with torch.no_grad():
                self.model.registers.copy_(payload["registers"].to(self.model.registers.device))

    # -- one step ------------------------------------------------------------------------------
    def train_step(self) -> dict[str, float]:
        cfg = self.cfg
        lr = learning_rate(self.step, cfg)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        pooled = {"nll_sum": 0.0, "weight_sum": 0.0, "hits": 0.0, "masked": 0.0}
        blocks = 0
        for _ in range(cfg.grad_accum):
            batch = self.collate(self.stream.take(cfg.micro_batch))
            loss, stats = self.loss_on(batch)
            (loss / cfg.grad_accum).backward()
            for key in pooled:
                pooled[key] += stats[key]
            blocks += int(batch.block_valid.sum())
            self.examples_seen += cfg.micro_batch
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), cfg.grad_clip)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite gradient norm at step {self.step}")
        self.optimizer.step()
        self.step += 1
        return {"loss": pooled["nll_sum"] / max(pooled["weight_sum"], 1e-8),
                "accuracy": pooled["hits"] / max(pooled["masked"], 1.0), "lr": lr, "grad_norm": float(grad_norm),
                "blocks": float(blocks), "masked_slots": pooled["masked"]}

    # -- evaluation ----------------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """L_DFM and accuracy on the validation set's two targets at every budget of the run, each (target, K)
        with its own fixed draw of states, in batches of ``val_micro_batch`` whatever the training micro-batch."""
        cfg = self.cfg
        self.set_training(False)
        out: dict[str, float] = {}
        for target, examples in self.validation_sets.items():
            total = {"nll_sum": 0.0, "weight_sum": 0.0, "hits": 0.0, "masked": 0.0}
            for budget in cfg.step_budgets:
                generator = torch.Generator().manual_seed(3_000 + 10 * budget + (target == "rollouts"))
                at_k = {"nll_sum": 0.0, "weight_sum": 0.0, "hits": 0.0, "masked": 0.0}
                for start in range(0, len(examples), cfg.val_micro_batch):
                    batch = self.collate(examples[start:start + cfg.val_micro_batch])
                    _, stats = self.loss_on(batch, budget=budget, generator=generator)
                    for key in at_k:
                        at_k[key] += stats[key]
                out[f"val_{target}_loss_K{budget}"] = at_k["nll_sum"] / max(at_k["weight_sum"], 1e-8)
                out[f"val_{target}_accuracy_K{budget}"] = at_k["hits"] / max(at_k["masked"], 1.0)
                for key in total:
                    total[key] += at_k[key]
            out[f"val_{target}_loss"] = total["nll_sum"] / max(total["weight_sum"], 1e-8)
            out[f"val_{target}_accuracy"] = total["hits"] / max(total["masked"], 1.0)
        self.set_training(True)
        return out

    def probe(self) -> dict[str, float]:
        return {}                          # C7 has no probe; the final checkpoint is evaluated (block_evaluate.py)

    # -- the loop ------------------------------------------------------------------------------
    def fit(self, stop_at: int | None = None) -> None:
        """Stage 0's loop, ending at ``stop_at`` if given (the configuration keeps its ``steps``)."""
        cfg = self.cfg
        end = cfg.steps if stop_at is None else min(cfg.steps, stop_at)
        self.set_training(True)
        if self.step == 0:
            self.evaluate()                                          # the untrained model: the curves' origin
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        timing_path = self.run_dir / "timing.json"
        timing_start = (time.perf_counter(), self.step) if self.step < cfg.timing_steps else None
        window: list[dict[str, float]] = []
        clock, clock_step = time.perf_counter(), self.step
        while self.step < end:
            window.append(self.train_step())
            peak = torch.cuda.max_memory_allocated(self.device) / 2**30 if cuda else 0.0
            if self.step % cfg.log_every == 0 or self.step == end:
                elapsed = time.perf_counter() - clock
                rate = (self.step - clock_step) / max(elapsed, 1e-9)
                mean = {k: sum(m[k] for m in window) / len(window) for k in window[0]}
                self.logger.log(self.step, {**mean, "steps_per_s": rate, "peak_gib": peak,
                                            "eta_h": (cfg.steps - self.step) / max(rate, 1e-9) / 3600,
                                            "examples": self.examples_seen, "epoch": self.stream.epoch},
                                prefix="train/")
                window, clock, clock_step = [], time.perf_counter(), self.step
            if timing_start is not None and self.step >= cfg.timing_steps and not timing_path.exists():
                seconds = time.perf_counter() - timing_start[0]
                rate = (self.step - timing_start[1]) / max(seconds, 1e-9)
                self._timing = {"what": "speed of this run's first steps (measured)", "steps": self.step,
                                "seconds": round(seconds, 2), "steps_per_s": round(rate, 4),
                                "peak_gib": round(peak, 2), "device": str(self.device),
                                "micro_batch": cfg.micro_batch, "grad_accum": cfg.grad_accum,
                                "grad_checkpoint": cfg.grad_checkpoint,
                                "projected_hours_for_the_run": round(cfg.steps / max(rate, 1e-9) / 3600, 4),
                                "note": "without the held-out evaluations every eval_every steps"}
                timing_path.write_text(json.dumps(self._timing, indent=2))
            if self.step % cfg.save_every == 0 or self.step == end:
                self.save()
            if self.step % cfg.eval_every == 0 or self.step == cfg.steps:
                self.evaluate()
                shutil.copyfile(self.run_dir / "last.pt", self.run_dir / f"step-{self.step:06d}.pt")
                clock, clock_step = time.perf_counter(), self.step
        self.save()


# ---------------------------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E0: train Block-DFM on the frozen Gemma 4 E2B", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="the rollout writer's directory")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", default=None, help="local directory of the backbone (default: the config's)")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-memory-check", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="E0Config fields of a new run, e.g. arm=references block_size=32")
    p.add_argument("--stop-at", type=int, default=None, help="end after this step (a later start resumes)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from chat_vpa.backbone import RealizerSpec, Stage0Realizer, load_gemma4
    from chat_vpa.data import HFTokenizer
    from chat_vpa.device import resolve_device

    args = parse_args(argv)
    device = resolve_device(args.device)
    args.run.mkdir(parents=True, exist_ok=True)
    last = args.run / "last.pt"
    payload = load_checkpoint(last) if last.exists() else None
    if payload is not None:
        if payload.get("stage") != "e0":
            raise SystemExit(f"{last} is a {payload.get('stage', 'stage0')!r} checkpoint, not an E0 run")
        cfg = E0Config.from_dict(payload["config"]).check()
        if args.set:
            print("resuming: --set is ignored (the run's configuration is in its checkpoint)", file=sys.stderr)
    else:
        cfg = _overrides(E0Config(), args.set).check()
        cfg.data = str(args.data)
        if args.model:
            cfg.model = args.model
    data = RolloutData(args.data)
    model, hf_tokenizer = load_gemma4(args.model or cfg.model, revision=cfg.revision,
                                      token=os.environ.get("HF_TOKEN") or None)
    tokenizer = HFTokenizer(hf_tokenizer)
    ids = {"bos": tokenizer.bos_id, "eos": tokenizer.eos_id, "pad": tokenizer.pad_id, "mask": tokenizer.mask_id,
           "vocab_size": tokenizer.vocab_size}
    if data.settings.get("tokenizer") != ids:
        raise SystemExit(f"{args.data} was written with another tokenizer ({data.settings.get('tokenizer')})")
    fingerprint = data.fingerprint("train")
    if payload is not None and payload.get("data_fingerprint") not in (None, fingerprint):
        raise SystemExit(f"{args.data}/train.jsonl.gz is not the file this run started with")
    spec = RealizerSpec(mask_id=tokenizer.mask_id, lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                        modulation=cfg.modulation)
    torch.manual_seed(cfg.seed)
    realizer = Stage0Realizer.from_pretrained_model(model, spec)
    for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(model.model, tower):
            setattr(model.model, tower, None)                    # text only: frees the encoders' memory
    block_model = BlockDFM(realizer, num_registers=cfg.num_registers, register_seed=cfg.register_seed)
    block_model.to(device)
    stream = EpochStream(data.examples("train", cfg.arm), cfg.data_seed, payload["stream"] if payload else None)
    validation = {target: data.examples("validation", target) for target in ARMS}
    trainer = E0Trainer(block_model, tokenizer, cfg, stream, validation, args.run, device)
    trainer.data_fingerprint = fingerprint
    if payload is not None:
        trainer.restore(payload)
        print(f"resumed {last} at step {trainer.step}", file=sys.stderr)
    elif not args.no_memory_check:
        report = trainer.fit_memory()
        (args.run / "memory.json").write_text(json.dumps(report, indent=2))
        print(f"memory check: {json.dumps(report)}", file=sys.stderr)
    trainable = sum(p.numel() for p in block_model.trainable_parameters())
    (args.run / "config.json").write_text(json.dumps({
        **cfg.to_dict(), "trainable_parameters": trainable, "tokenizer": ids, "data_fingerprint": fingerprint,
        "train_examples": len(stream.examples), "validation_examples": len(validation["rollouts"])}, indent=2))
    print(f"E0 ({cfg.arm}, B = {cfg.block_size}): {trainable / 1e6:.1f}M trainable parameters, steps {trainer.step} "
          f"-> {cfg.steps if args.stop_at is None else min(cfg.steps, args.stop_at)} of {cfg.steps}, "
          f"{cfg.micro_batch} x {cfg.grad_accum} examples per step, {len(stream.examples)} kept examples",
          file=sys.stderr)
    trainer.fit(stop_at=args.stop_at)
    return 0


if __name__ == "__main__":
    sys.exit(main())
