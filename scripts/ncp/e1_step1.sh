#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# e1_step1.sh -- the timing runs of E1's Step 1 (docs/e1.md, Section 11.8; approved 27 November 2026): what the full
# runs cost on the L4, measured before any of them starts. About half an hour to an hour of the GPU.
#
#   bash scripts/ncp/e1_step1.sh          in tmux (session e1-timing): the three timing runs, then the summary
#   bash scripts/ncp/e1_step1.sh run      the same in the foreground
#   bash scripts/ncp/e1_step1.sh show     the summary the author pastes back; no GPU
#
# 1. the rollout writer (chat_vpa/e1_rollouts.py) on the first chunk of 2,048 of the 100,000 training prompts:
#    the first chunk of the full run, which resumes after it; it also counts the training prompts that are the
#    text of a development prompt. UltraChat-200K's train_sft split is downloaded first if it is missing
#    (bash scripts/ncp/e1_setup.sh train-data, the one step that uses the network).
# 2. 100 steps of A2's trainer (chat_vpa/e1_train.py --arm a2) from Google's drafter, on that chunk, with the
#    schedule of the full run (12,500 steps); before its first step, the held-out curve of Google's drafter as
#    shipped on TIMING_EVAL rollouts of the chunk (it was not trained on them: a check of the trainer's counts
#    against Step 0's tokens per cycle, and the time of an evaluation).
# 3. 100 steps of A3-0's trainer (--arm a3-0) from Google's drafter (A2 does not exist yet; the timing does not
#    depend on the starting weights), likewise.
# Nothing here is a result: the trained weights are thrown away (runs/e1-step1/timing/), and B1's timing waits
# behind B1's gate (docs/e1.md, Section 11.6).
#
# Variables (defaults in brackets):
#   OUT [runs/e1-step1]   ASSETS [runs/assets]   TRAIN_PROMPTS [100000]   BATCH_SIZE [128] (the writer's; halved
#   on out-of-memory)   STOP_AT [100]   TIMING_EVAL [256]   EXTRA_TRAIN [] (more --set KEY=VALUE for both trainers)
# A part that has finished is skipped. Never shuts the server down. Do not git pull while it runs.
set -euo pipefail
PASS_VARS="OUT ASSETS TRAIN_PROMPTS BATCH_SIZE STOP_AT TIMING_EVAL EXTRA_TRAIN SESSION"
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ACTION="${1:-launch}"
OUT="$(runs_dir "${OUT:-runs/e1-step1}")"
ASSETS="$(runs_dir "${ASSETS:-runs/assets}")"
TRAIN_PROMPTS="${TRAIN_PROMPTS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
STOP_AT="${STOP_AT:-100}"
TIMING_EVAL="${TIMING_EVAL:-256}"
EXTRA_TRAIN="${EXTRA_TRAIN:-}"
SESSION="${SESSION:-e1-timing}"
IT_MODEL="$ASSETS/models/gemma-4-E2B-it"
DRAFTER="$ASSETS/models/gemma-4-E2B-it-assistant"
PROMPTS="$ASSETS/prompts/dev.jsonl"
TRAIN_DATA="$ASSETS/data/ultrachat_200k_train"
ROLLOUTS="$OUT/rollouts/train"

timing_runs() {
  check_python
  check_gpu_free
  [[ -f "$IT_MODEL/config.json" && -f "$DRAFTER/config.json" && -f "$PROMPTS" ]] \
    || die "the assets are missing under $ASSETS: bash scripts/ncp/e1_setup.sh"
  if ! compgen -G "$TRAIN_DATA/data/train_sft-*.parquet" > /dev/null; then
    log "train_sft is not downloaded: bash scripts/ncp/e1_setup.sh train-data"
    ASSETS="$ASSETS" bash "$ROOT/scripts/ncp/e1_setup.sh" train-data || die "the download of train_sft failed"
  fi
  mkdir -p "$OUT"
  if [[ -f "$ROLLOUTS/chunk-00000.json" ]]; then
    log "done already: the writer's first chunk"
  else
    log "start: the rollout writer, first chunk of 2,048 prompts (batch $BATCH_SIZE)"
    "$PY" -m chat_vpa.e1_rollouts write --model "$IT_MODEL" --parquet "$TRAIN_DATA"/data/train_sft-*.parquet \
      --start 0 --count "$TRAIN_PROMPTS" --dev-prompts "$PROMPTS" --batch-size "$BATCH_SIZE" --max-chunks 1 \
      --out "$ROLLOUTS" || die "failed: the rollout writer"
  fi
  "$PY" -m chat_vpa.e1_rollouts summary --out "$ROLLOUTS" > /dev/null
  local arm
  for arm in a2 a3-0; do
    if [[ -f "$OUT/timing/$arm/timing.json" ]]; then log "done already: $arm's $STOP_AT steps"; continue; fi
    log "start: $STOP_AT steps of $arm's trainer"
    # shellcheck disable=SC2086
    "$PY" -m chat_vpa.e1_train --arm "$arm" --model "$IT_MODEL" --assistant "$DRAFTER" --rollouts "$ROLLOUTS" \
      --held-out "$ROLLOUTS" --held-out-limit "$TIMING_EVAL" --run "$OUT/timing/$arm" --stop-at "$STOP_AT" \
      --set "horizon_examples=$TRAIN_PROMPTS" $EXTRA_TRAIN || die "failed: $arm's trainer"
  done
  log "timing runs finished: bash scripts/ncp/e1_step1.sh show"
}

show() {
  [[ -f "$ROLLOUTS/summary.json" ]] || die "no timing run yet: bash scripts/ncp/e1_step1.sh"
  "$PY" - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
roll = json.loads((out / "rollouts" / "train" / "summary.json").read_text())
writer = json.loads((out / "rollouts" / "train" / "writer.json").read_text())
env = writer["environment"]
print(f"== E1 Step 1, timing runs (docs/e1.md, Section 11.8): {env.get('gpu', env['device'])}, torch {env['torch']}, "
      f"transformers {env['transformers']}")
rate = roll["rollouts_per_second"]
print(f"rollout writer (measured, {roll['chunks_written']} of {roll['chunks']} chunks): {roll['rollouts']} rollouts in "
      f"{roll['seconds']:.0f} s, {rate:.2f} rollouts/s, {roll['response_tokens_per_second']:.0f} response tokens/s, mean "
      f"prompt {roll['mean_prompt_tokens']:.0f} and response {roll['mean_response_tokens']:.1f} tokens, "
      f"{roll['budget_stops']} at the {writer['max_new_tokens']}-token budget, peak {roll['peak_memory_gib']} GiB, "
      f"{roll['halvings']} halvings of batch {writer['batch_size']}")
print(f"  predicted: {102_000 / rate / 3600:.1f} h for 100,000 + 2,000 prompts")
print(f"  development prompts among the {writer['count']:,} training prompts: {len(writer['dev_overlap'] or [])}"
      + (f" ({', '.join(writer['dev_overlap'][:10])})" if writer["dev_overlap"] else ""))
held_out = 2_000
for arm in ("a2", "a3-0"):
    run = out / "timing" / arm
    if not (run / "timing.json").exists():
        print(f"{arm}: not run")
        continue
    timing = json.loads((run / "timing.json").read_text())
    config = json.loads((run / "config.json").read_text())
    rows = [json.loads(line) for line in open(run / "metrics.jsonl", encoding="utf-8")]
    train = [r for r in rows if "train/loss" in r]
    evals = [r for r in rows if "eval/seconds" in r]
    peak = max((r.get("train/peak_memory_gib", 0.0) for r in train), default=0.0)
    cfg = config["config"]
    per_step = timing["seconds_per_step"]
    steps = timing["horizon_steps"]
    print(f"{arm} (measured): {timing['steps_run']} steps of {cfg['batch']} rollouts, {per_step:.3f} s per step after the "
          f"first 10, peak {peak:.2f} GiB; {config['trained_parameters'] / 1e6:.1f}M trained parameters; first loss "
          f"{train[0]['train/loss']:.3f}, last {train[-1]['train/loss']:.3f}")
    if evals:
        e = evals[0]
        per_rollout = e["eval/seconds"] / e["eval/rollouts"]
        count = len(range(0, steps + 1, cfg["eval_every"])) + (0 if steps % cfg["eval_every"] == 0 else 1)
        eval_hours = per_rollout * held_out * count / 3600
        print(f"  held-out pass (measured): {e['eval/seconds']:.0f} s for {e['eval/rollouts']:.0f} rollouts")
        cycles = "  ".join(f"G={k.rsplit('_', 1)[1]}: {v:.2f}" for k, v in e.items()
                           if k.startswith("eval/tokens_per_cycle_"))
        hits = "  ".join(f"{k.rsplit('_', 1)[1]}: {v:.3f}" for k, v in e.items()
                         if k.startswith("eval/hit_") and not k.startswith("eval/hit_overall"))
        label = "Google's drafter as shipped" if arm == "a2" else "A3-0 before training (no meaning)"
        print(f"  step 0, {label}, counted along the rollouts' continuations: tokens per cycle {cycles}; hit rate by "
              f"depth {hits}")
    else:
        eval_hours = 0.0
    print(f"  predicted: {per_step * steps / 3600:.1f} h for {steps:,} steps, plus {eval_hours:.1f} h for the held-out "
          f"curves on {held_out:,} rollouts every {cfg['eval_every']:,} steps")
print("B1: not timed (behind B1's gate, Section 11.6)")
PYEOF
}

case "$ACTION" in
  launch) start_tmux "$SESSION" "$OUT/timing.log" "bash scripts/ncp/e1_step1.sh run" ;;
  run)    timing_runs; show ;;
  show)   show ;;
  *)      die "usage: e1_step1.sh [launch|run|show]" ;;
esac
