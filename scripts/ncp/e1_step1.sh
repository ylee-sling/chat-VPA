#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# e1_step1.sh -- E1's Step 1 on the L4 (docs/e1.md, Section 11).
#
#   bash scripts/ncp/e1_step1.sh          the timing runs of Section 11.8 in tmux (session e1-timing); done
#   bash scripts/ncp/e1_step1.sh run      the same in the foreground
#   bash scripts/ncp/e1_step1.sh show     their summary; no GPU
#   bash scripts/ncp/e1_step1.sh a2       A2 (approved 27 November 2026), in tmux (session e1-a2): the rest of the
#                                         rollouts and the held-out set, A2's training, its measurement (S9 with the
#                                         accepted-prefix arms) and the readings; about 26 hours (predicted)
#   bash scripts/ncp/e1_step1.sh run-a2   the same in the foreground
#   bash scripts/ncp/e1_step1.sh show-a2  where it stands, and the readings once they exist; no GPU
#   bash scripts/ncp/e1_step1.sh breakeven
#                                         A3's break-even from A2's float32 table and Step 0's float32 ceiling
#                                         (docs/e1.md, Sections 6.1, 11.5, 11.12): raw rates, cycle times, the
#                                         drafter's call time and tau* against A1 (the gate) and A2; no GPU, seconds
#   bash scripts/ncp/e1_step1.sh a3 add   A3-0 in tmux (session e1-a3), its slot input named: "add" (x_q's
#   bash scripts/ncp/e1_step1.sh a3 replace  embedding kept, the code added) or "replace" (D4 as registered); the
#                                         break-even, A3-0's training from A2's weights (D4) and the counts along the
#                                         development prompts' continuations; about 21 hours (predicted)
#   bash scripts/ncp/e1_step1.sh run-a3 add   the same in the foreground
#   bash scripts/ncp/e1_step1.sh show-a3  where every A3 run stands: the break-even, the training curves, the counts,
#                                         the oracle gate and the one-pass tables' readings; no GPU
#   bash scripts/ncp/e1_step1.sh a3-measure
#                                         A3-0's speed (docs/e1.md, Section 11.13), in tmux (session e1-a3m): the
#                                         float32 L4 table of ar, A1 and A2 at their best G (both kinds of keys) and
#                                         A3-0 at G = 2, 4, 6, 8 through its own loop; its readings; about 2.2 hours
#   bash scripts/ncp/e1_step1.sh a3-oracle
#                                         A3-oracle (Section 11.13), in tmux (session e1-a3o): its training from the
#                                         weights A3-0 started from, with A3-0's slot input, then the counts of A1,
#                                         A2, A3-0 and A3-oracle with the oracle gate of Section 6.1; about 21 hours
#   bash scripts/ncp/e1_step1.sh a3-predict
#                                         A3 (Section 11.13), in tmux (session e1-a3p): its predictor trained on
#                                         A3-oracle's frozen realiser and codebook, the counts, its float32 table and
#                                         readings; it refuses to start unless the oracle gate let A3 be trained
#
# A2's parts, each skipped once done, so the command can be given again after a failure:
#   rollouts  chat_vpa/e1_rollouts.py: the training prompts 0 to 99,999 (resuming after the timing run's first
#             chunk), then the held-out prompts 100,000 to 101,999 (S2 to S5)
#   train     chat_vpa/e1_train.py --arm a2 (S6 to S8): 12,500 steps of 16 rollouts, held-out curves every 2,500
#             steps; a stopped run resumes from runs/e1-step1/a2/last.pt
#   measure   S9: chat_vpa/e1_decode.py with A1 (a1) and A2 (a2) on the development prompts: the L4 in float32,
#             100 x 256 (ar, both drafters at G = 2, 4, 6, 8, both at G = 4 with the accepted prefix's keys); the
#             L4 in bfloat16 (ar, each drafter at its best G of the float32 table); the CPU in float32 with
#             THREADS threads, 20 x 128 (ar, both at G = 2, 4), if CPU_FP32_GIB GiB are free; vLLM (a smoke
#             decode of A2, then off-1, both at G = 2, 4, 6, 8, off-2)
#   read      chat_vpa/e1_readings.py: R1, R2, R6, R7, R9 and S11 on those tables (runs/e1-step1/measure/readings.md)
#
# A3-0's parts (A3_PARTS), each skipped once done:
#   breakeven  chat_vpa/e1_readings.py breakeven on runs/e1-step1/measure/l4-fp32 and runs/e1-step0/ceiling/gpu-fp32:
#              the bar fixed before A3-0 trains (runs/e1-step1/breakeven/breakeven.md)
#   train      chat_vpa/e1_train.py --arm a3-0 (D4, D6, D8; Section 11.12): 12,500 steps of 16 rollouts in
#              micro-batches of 8, the cluster term at weight 1, from A2's final weights unless R6 found A2
#              accepting fewer tokens than A1 in the float32 table (then from A1's); held-out curves every 2,500 steps
#   count      chat_vpa/e1_count.py: A1, A2 and A3-0 counted along the float32 table's continuations of the
#              development prompts (runs/e1-step1/count/a3-0/count.md); counted, not timed
# Every later A3 action is one command of its own, for the author to start (each needs GPU time): a3-measure
# (runs/e1-step1/measure/l4-fp32-a30), a3-oracle (runs/e1-step1/a3-oracle, runs/e1-step1/count/a3-oracle) and
# a3-predict (runs/e1-step1/a3, runs/e1-step1/count/a3, runs/e1-step1/measure/l4-fp32-a3).
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
#   OUT [runs/e1-step1]   ASSETS [runs/assets]   TRAIN_PROMPTS [100000]   HELD_OUT [2000]   BATCH_SIZE [128] (the
#   writer's; halved on out-of-memory)   STOP_AT [100]   TIMING_EVAL [256]   EXTRA_TRAIN [] (more --set KEY=VALUE)
#   PARTS ["rollouts train measure read"]   THREADS [4]   CPU_FP32_GIB [24]   MEASURE_PROMPTS [100]   MAX_NEW [256]
#   MAIN_ARMS ["ar a1:2 a2:2 a1:4 a2:4 a1:4:clean a2:4:clean a1:6 a2:6 a1:8 a2:8"]   CPU_PROMPTS [20]
#   CPU_MAX_NEW [128]   CPU_ARMS ["ar a1:2 a2:2 a1:4 a2:4"]   VLLM_LENGTHS ["2 4 6 8"]   VLLM_TIMEOUT [7200]
#   EXTRA_DECODE [], EXTRA_VLLM [] (dry runs)   A3_PARTS ["breakeven train count"]   STEP0 [runs/e1-step0]
#   EXTRA_COUNT [] (dry runs)
# A part that has finished is skipped. Never shuts the server down. Do not git pull while it runs.
set -euo pipefail
PASS_VARS="OUT ASSETS TRAIN_PROMPTS HELD_OUT BATCH_SIZE STOP_AT TIMING_EVAL EXTRA_TRAIN PARTS THREADS CPU_FP32_GIB MEASURE_PROMPTS MAX_NEW MAIN_ARMS CPU_PROMPTS CPU_MAX_NEW CPU_ARMS VLLM_LENGTHS VLLM_TIMEOUT EXTRA_DECODE EXTRA_VLLM A3_PARTS STEP0 EXTRA_COUNT SESSION"
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
HELD_OUT="${HELD_OUT:-2000}"
PARTS="${PARTS:-rollouts train measure read}"
THREADS="${THREADS:-4}"
CPU_FP32_GIB="${CPU_FP32_GIB:-24}"
MEASURE_PROMPTS="${MEASURE_PROMPTS:-100}"
MAX_NEW="${MAX_NEW:-256}"
MAIN_ARMS="${MAIN_ARMS:-ar a1:2 a2:2 a1:4 a2:4 a1:4:clean a2:4:clean a1:6 a2:6 a1:8 a2:8}"
CPU_PROMPTS="${CPU_PROMPTS:-20}"
CPU_MAX_NEW="${CPU_MAX_NEW:-128}"
CPU_ARMS="${CPU_ARMS:-ar a1:2 a2:2 a1:4 a2:4}"
VLLM_LENGTHS="${VLLM_LENGTHS:-2 4 6 8}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-7200}"
EXTRA_DECODE="${EXTRA_DECODE:-}"
EXTRA_VLLM="${EXTRA_VLLM:-}"
A3_PARTS="${A3_PARTS:-breakeven train count}"
STEP0="$(runs_dir "${STEP0:-runs/e1-step0}")"
EXTRA_COUNT="${EXTRA_COUNT:-}"
case "$ACTION" in
  a2|run-a2|show-a2) SESSION="${SESSION:-e1-a2}" ;;
  a3|run-a3|show-a3) SESSION="${SESSION:-e1-a3}" ;;
  a3-measure|run-a3-measure) SESSION="${SESSION:-e1-a3m}" ;;
  a3-oracle|run-a3-oracle)   SESSION="${SESSION:-e1-a3o}" ;;
  a3-predict|run-a3-predict) SESSION="${SESSION:-e1-a3p}" ;;
  *)                 SESSION="${SESSION:-e1-timing}" ;;
esac
IT_MODEL="$ASSETS/models/gemma-4-E2B-it"
DRAFTER="$ASSETS/models/gemma-4-E2B-it-assistant"
PROMPTS="$ASSETS/prompts/dev.jsonl"
TRAIN_DATA="$ASSETS/data/ultrachat_200k_train"
ROLLOUTS="$OUT/rollouts/train"
HELD="$OUT/rollouts/held-out"
A2="$OUT/a2"
A3="$OUT/a3-0"
A3O="$OUT/a3-oracle"
A3P="$OUT/a3"
MEASURE="$OUT/measure"
VLLM_PY="$ASSETS/venvs/vllm/bin/python"

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

# ---------------------------------------------------------------------------------------------
# A2: the rollouts, the training, the measurement and the readings
# ---------------------------------------------------------------------------------------------

# step <directory> <soft|hard> <command...>: run once; a "soft" failure is logged and the script goes on.
step() {
  local dir="$1" mode="$2"; shift 2
  if [[ -f "$dir/.done" ]]; then log "done already: ${dir#"$ROOT"/}"; return 0; fi
  mkdir -p "$dir"
  log "start: ${dir#"$ROOT"/}"
  if "$@"; then
    touch "$dir/.done"
    log "finished: ${dir#"$ROOT"/}"
  elif [[ "$mode" == soft ]]; then
    log "FAILED, going on: ${dir#"$ROOT"/}"
  else
    die "failed: ${dir#"$ROOT"/}"
  fi
}

# memory_for_cpu_fp32: CPU_FP32_GIB GiB are available now (the model in float32 is about 19 GiB; a run that does not
# fit would be killed by the kernel, which may pick another process).
memory_for_cpu_fp32() {
  local have
  have="$(awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo)"
  if (( have >= CPU_FP32_GIB )); then return 0; fi
  log "skipped, not run: float32 on the CPU needs about $CPU_FP32_GIB GiB of free memory, $have GiB are available"
  return 1
}

# foreign <command...>: vLLM, with a home of its own under runs/assets and its own CUDA allocator settings.
foreign() {
  mkdir -p "$ASSETS/home"
  env -u PYTORCH_CUDA_ALLOC_CONF HOME="$ASSETS/home" XDG_CACHE_HOME="$ASSETS/home/.cache" \
    XDG_CONFIG_HOME="$ASSETS/home/.config" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 "$@"
}

rollouts() {
  if ! compgen -G "$TRAIN_DATA/data/train_sft-*.parquet" > /dev/null; then
    ASSETS="$ASSETS" bash "$ROOT/scripts/ncp/e1_setup.sh" train-data || die "the download of train_sft failed"
  fi
  step "$ROLLOUTS" hard "$PY" -m chat_vpa.e1_rollouts write --model "$IT_MODEL" \
    --parquet "$TRAIN_DATA"/data/train_sft-*.parquet --start 0 --count "$TRAIN_PROMPTS" --dev-prompts "$PROMPTS" \
    --batch-size "$BATCH_SIZE" --out "$ROLLOUTS"
  step "$HELD" hard "$PY" -m chat_vpa.e1_rollouts write --model "$IT_MODEL" \
    --parquet "$TRAIN_DATA"/data/train_sft-*.parquet --start "$TRAIN_PROMPTS" --count "$HELD_OUT" \
    --dev-prompts "$PROMPTS" --batch-size "$BATCH_SIZE" --out "$HELD"
  "$PY" -m chat_vpa.e1_rollouts summary --out "$ROLLOUTS"
  "$PY" -m chat_vpa.e1_rollouts summary --out "$HELD"
}

train() {
  [[ -f "$ROLLOUTS/.done" && -f "$HELD/.done" ]] || die "the rollouts are not written: PARTS=rollouts first"
  local extra=()
  if [[ -n "$EXTRA_TRAIN" ]]; then read -r -a extra <<< "$EXTRA_TRAIN"; extra=(--set "${extra[@]}"); fi
  step "$A2" hard "$PY" -m chat_vpa.e1_train --arm a2 --model "$IT_MODEL" --assistant "$DRAFTER" \
    --rollouts "$ROLLOUTS" --held-out "$HELD" --run "$A2" "${extra[@]}"
  [[ -f "$A2/final/config.json" ]] || die "A2's training ended without its final checkpoint"
}

measure() {
  [[ -f "$A2/final/config.json" ]] || die "A2 is not trained: PARTS=train first"
  local drafters=(--drafter "a1=$DRAFTER" --drafter "a2=$A2/final") device=() cpu=()
  if [[ "$REQUIRE_GPU" != 1 ]]; then device=(--device cpu); fi
  # shellcheck disable=SC2086
  step "$MEASURE/l4-fp32" hard "$PY" -m chat_vpa.e1_decode --model "$IT_MODEL" "${drafters[@]}" \
    --prompts "$PROMPTS" --limit "$MEASURE_PROMPTS" --max-new-tokens "$MAX_NEW" --dtype fp32 --arms $MAIN_ARMS \
    --out "$MEASURE/l4-fp32" "${device[@]}" $EXTRA_DECODE
  local g1 g2
  g1="$("$PY" -m chat_vpa.e1_readings best --table "$MEASURE/l4-fp32" --drafter a1)"
  g2="$("$PY" -m chat_vpa.e1_readings best --table "$MEASURE/l4-fp32" --drafter a2)"
  log "best G in float32: A1 $g1, A2 $g2"
  local bf16=(--dtype bf16)
  if [[ "$REQUIRE_GPU" != 1 ]]; then bf16=(--dtype fp32); fi
  # shellcheck disable=SC2086
  step "$MEASURE/l4-bf16" hard "$PY" -m chat_vpa.e1_decode --model "$IT_MODEL" "${drafters[@]}" \
    --prompts "$PROMPTS" --limit "$MEASURE_PROMPTS" --max-new-tokens "$MAX_NEW" "${bf16[@]}" \
    --arms ar "a1:$g1" "a2:$g2" --out "$MEASURE/l4-bf16" "${device[@]}" $EXTRA_DECODE
  if memory_for_cpu_fp32; then
    cpu=(--device cpu --dtype fp32 --threads "$THREADS")
    # shellcheck disable=SC2086
    step "$MEASURE/cpu-fp32" soft "$PY" -m chat_vpa.e1_decode --model "$IT_MODEL" "${drafters[@]}" \
      --prompts "$PROMPTS" --limit "$CPU_PROMPTS" --max-new-tokens "$CPU_MAX_NEW" --warmup 1 --arms $CPU_ARMS \
      --out "$MEASURE/cpu-fp32" "${cpu[@]}" $EXTRA_DECODE
  fi
  measure_vllm
}

measure_vllm() {
  if [[ "$REQUIRE_GPU" != 1 ]]; then log "vLLM needs the GPU: skipped in a dry run"; return 0; fi
  if [[ ! -x "$VLLM_PY" ]]; then log "vLLM is not set up ($VLLM_PY): skipping its table"; return 0; fi
  local dir="$MEASURE/vllm" g
  # S9: A2 enters vLLM's table after a smoke decode through vLLM's MTP path, as A1 did in the preflight.
  # shellcheck disable=SC2086
  step "$dir/smoke-a2" soft foreign timeout --kill-after=60 "$VLLM_TIMEOUT" "$VLLM_PY" -m chat_vpa.e1_vllm smoke \
    --model "$IT_MODEL" --assistant "$A2/final" --prompts "$PROMPTS" --arm mtp:4 $EXTRA_VLLM
  local labels=(a1) a2_ok=0
  if [[ -f "$dir/smoke-a2/.done" ]]; then labels=(a1 a2); a2_ok=1; else log "A2 did not decode in vLLM: A1 only"; fi
  vllm_arm off off-1 ""
  for g in $VLLM_LENGTHS; do
    vllm_arm "mtp:$g" "a1-$g" "$DRAFTER"
    if [[ "$a2_ok" == 1 ]]; then vllm_arm "mtp:$g" "a2-$g" "$A2/final"; fi
  done
  vllm_arm off off-2 ""
  if [[ -f "$dir/arms/off-1/.done" ]]; then
    "$PY" -m chat_vpa.e1_vllm summarize --out "$dir" || log "FAILED, going on: the summary of the vLLM arms"
  fi
  log "vLLM: ${labels[*]}"
}

# vllm_arm <arm> <tag> <drafter or "">: one engine, one process; an arm takes minutes, the timeout only keeps a hung
# engine from holding the GPU unattended.
vllm_arm() {
  local arm="$1" tag="$2" drafter="$3" assistant=()
  if [[ -n "$drafter" ]]; then assistant=(--assistant "$drafter"); fi
  # shellcheck disable=SC2086
  step "$MEASURE/vllm/arms/$tag" soft foreign timeout --kill-after=60 "$VLLM_TIMEOUT" "$VLLM_PY" -m chat_vpa.e1_vllm \
    run --arm "$arm" --tag "$tag" --model "$IT_MODEL" "${assistant[@]}" --prompts "$PROMPTS" \
    --limit "$MEASURE_PROMPTS" --max-new-tokens "$MAX_NEW" --out "$MEASURE/vllm" $EXTRA_VLLM
}

read_a2() {
  local tables=() t
  for t in l4-fp32 l4-bf16 cpu-fp32; do
    if [[ -f "$MEASURE/$t/summary.json" ]]; then tables+=("$MEASURE/$t"); fi
  done
  "$PY" -m chat_vpa.e1_readings read --tables "${tables[@]}" --vllm "$MEASURE/vllm" --out "$MEASURE" > /dev/null \
    || die "the readings failed"
  log "readings: ${MEASURE#"$ROOT"/}/readings.md"
}

a2_runs() {
  check_python
  check_gpu_free
  [[ -f "$IT_MODEL/config.json" && -f "$DRAFTER/config.json" && -f "$PROMPTS" ]] \
    || die "the assets are missing under $ASSETS: bash scripts/ncp/e1_setup.sh"
  [[ -f "$ROLLOUTS/chunk-00000.json" ]] || log "no timing run before this one: the writer starts at chunk 0"
  mkdir -p "$OUT"
  local part
  for part in $PARTS; do
    case "$part" in
      rollouts) rollouts ;;
      train)    train ;;
      measure)  measure ;;
      read)     read_a2 ;;
      *)        die "unknown part: $part (rollouts, train, measure, read)" ;;
    esac
  done
  log "A2 finished: bash scripts/ncp/e1_step1.sh show-a2"
}

show_a2() {
  "$PY" - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
print("== E1 Step 1, A2 (docs/e1.md, Sections 11.3, 11.4)")
for name in ("train", "held-out"):
    d = out / "rollouts" / name
    if (d / "summary.json").exists():
        s = json.loads((d / "summary.json").read_text())
        print(f"rollouts/{name}: {s['rollouts']} of {s['of']} ({s['chunks_written']} of {s['chunks']} chunks), "
              f"{s['budget_stops']} at the budget, {s['rollouts_per_second']:.2f} rollouts/s (measured)"
              + (", done" if (d / ".done").exists() else ""))
    else:
        print(f"rollouts/{name}: not started")
run = out / "a2"
if (run / "metrics.jsonl").exists():
    rows = [json.loads(line) for line in open(run / "metrics.jsonl", encoding="utf-8")]
    train = [r for r in rows if "train/loss" in r]
    evals = [r for r in rows if "eval/seconds" in r]
    config = json.loads((run / "config.json").read_text())["config"]
    if train:
        t = train[-1]
        print(f"a2: step {t['step']} of {config['steps']}, loss {t['train/loss']:.3f}, {t['train/steps_per_s']:.3f} "
              f"steps/s, {t.get('train/eta_h', 0):.1f} h left (predicted), peak {t.get('train/peak_memory_gib', 0):.1f} "
              f"GiB, micro-batch {t.get('train/micro_batch', config['batch']):.0f}"
              + (", done" if (run / "final" / "config.json").exists() else ""))
    for e in evals:
        cycles = "  ".join(f"G={k.rsplit('_', 1)[1]}: {v:.2f}" for k, v in e.items()
                           if k.startswith("eval/tokens_per_cycle_"))
        print(f"  held-out at step {e['step']}: tokens per cycle along the continuations {cycles}; hit at depth 1 "
              f"{e.get('eval/hit_1', float('nan')):.3f}")
else:
    print("a2: not started")
measure = out / "measure"
for table in ("l4-fp32", "l4-bf16", "cpu-fp32", "vllm"):
    d = measure / table
    state = "done" if (d / ".done").exists() or (d / "summary.json").exists() else (
        "running or stopped" if d.exists() else "not started")
    print(f"measure/{table}: {state}")
if (measure / "readings.md").exists():
    print()
    print((measure / "readings.md").read_text())
PYEOF
}

# ---------------------------------------------------------------------------------------------
# A3-0: the break-even, the training and the counts (docs/e1.md, Section 11.12)
# ---------------------------------------------------------------------------------------------

breakeven() {
  [[ -f "$MEASURE/l4-fp32/summary.json" ]] || die "A2's float32 table is missing: bash scripts/ncp/e1_step1.sh a2"
  [[ -f "$STEP0/ceiling/gpu-fp32/summary.json" ]] || die "Step 0's float32 ceiling is missing under $STEP0"
  step "$OUT/breakeven" hard "$PY" -m chat_vpa.e1_readings breakeven --table "$MEASURE/l4-fp32" \
    --ceiling "$STEP0/ceiling/gpu-fp32" --against a1 a2 --out "$OUT/breakeven"
}

# a3_start: D4's rule, applied to the float32 readings: A2's weights unless R6 found A2 accepting fewer tokens.
a3_start() {
  "$PY" - "$MEASURE/readings.json" "$MEASURE/l4-fp32" <<'PYEOF'
import json, sys
readings = json.load(open(sys.argv[1], encoding="utf-8"))
table = next(r for r in readings if r["table"].rstrip("/").endswith("l4-fp32"))
fewer = [x for x in table["r6"] if x["reading"].endswith("accepts fewer")]
print("a1" if fewer else "a2")
PYEOF
}

train_a3() {
  local variant="$1" start source sets=("slot_input=$variant") more=()
  [[ -f "$ROLLOUTS/.done" && -f "$HELD/.done" ]] || die "the rollouts are not written: bash scripts/ncp/e1_step1.sh a2"
  [[ -f "$A2/final/config.json" && -f "$MEASURE/readings.json" ]] || die "A2's run has not finished"
  start="$(a3_start)" || die "R6 could not be read from $MEASURE/readings.json"
  if [[ "$start" == a2 ]]; then source="$A2/final"; else source="$DRAFTER"; fi
  log "A3-0 starts from $start's weights (D4: R6 in the float32 table), slot input '$variant'"
  if [[ -n "$EXTRA_TRAIN" ]]; then read -r -a more <<< "$EXTRA_TRAIN"; sets+=("${more[@]}"); fi
  step "$A3" hard "$PY" -m chat_vpa.e1_train --arm a3-0 --model "$IT_MODEL" --assistant "$source" \
    --rollouts "$ROLLOUTS" --held-out "$HELD" --run "$A3" --set "${sets[@]}"
  [[ -f "$A3/final/realiser.pt" ]] || die "A3-0's training ended without its final checkpoint"
}

count_a3() {
  [[ -f "$A3/final/realiser.pt" ]] || die "A3-0 is not trained: A3_PARTS=train first"
  local device=()
  if [[ "$REQUIRE_GPU" != 1 ]]; then device=(--device cpu); fi
  # shellcheck disable=SC2086
  step "$OUT/count/a3-0" hard "$PY" -m chat_vpa.e1_count --model "$IT_MODEL" --table "$MEASURE/l4-fp32" \
    --prompts "$PROMPTS" --drafter "a1=$DRAFTER" --drafter "a2=$A2/final" --drafter "a3-0=$A3/final" \
    --out "$OUT/count/a3-0" "${device[@]}" $EXTRA_COUNT
}

a3_runs() {
  local variant="${1:-}"
  [[ "$variant" == add || "$variant" == replace ]] \
    || die "name A3-0's slot input (docs/e1.md, Section 11.12): e1_step1.sh a3 add, or e1_step1.sh a3 replace"
  check_python
  check_gpu_free
  [[ -f "$IT_MODEL/config.json" && -f "$DRAFTER/config.json" && -f "$PROMPTS" ]] \
    || die "the assets are missing under $ASSETS: bash scripts/ncp/e1_setup.sh"
  local part
  for part in $A3_PARTS; do
    case "$part" in
      breakeven) breakeven ;;
      train)     train_a3 "$variant" ;;
      count)     count_a3 ;;
      *)         die "unknown part: $part (breakeven, train, count)" ;;
    esac
  done
  log "A3-0 finished: bash scripts/ncp/e1_step1.sh show-a3"
}

# a3_variant: A3-0's slot input, which A3-oracle and A3 keep (Section 11.12).
a3_variant() {
  "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["config"]["slot_input"])' "$A3/config.json"
}

# measure_one_pass <label> <run> <table>: a float32 L4 table of ar, A1 and A2 at their best G (both kinds of keys),
# the one-pass drafters before it at their best G, and <label> at G = 2, 4, 6, 8; then its readings.
measure_one_pass() {
  local label="$1" run="$2" name="$3" g1 g2 g30 device=()
  [[ -f "$run/final/realiser.pt" ]] || die "$label is not trained ($run)"
  if [[ "$REQUIRE_GPU" != 1 ]]; then device=(--device cpu); fi
  g1="$("$PY" -m chat_vpa.e1_readings best --table "$MEASURE/l4-fp32" --drafter a1)"
  g2="$("$PY" -m chat_vpa.e1_readings best --table "$MEASURE/l4-fp32" --drafter a2)"
  local drafters=(--drafter "a1=$DRAFTER" --drafter "a2=$A2/final" --drafter "a30=$A3/final")
  local arms=(ar "a1:$g1" "a1:$g1:clean" "a2:$g2" "a2:$g2:clean")
  if [[ "$label" == a3 ]]; then
    [[ -f "$MEASURE/l4-fp32-a30/summary.json" ]] || die "A3-0's table is missing: bash scripts/ncp/e1_step1.sh a3-measure"
    g30="$("$PY" -m chat_vpa.e1_readings best --table "$MEASURE/l4-fp32-a30" --drafter a30)"
    drafters+=(--drafter "a3=$run/final")
    arms+=("a30:$g30")
  fi
  arms+=("$label:2" "$label:4" "$label:6" "$label:8")
  log "the $label table: ${arms[*]}"
  # shellcheck disable=SC2086
  step "$MEASURE/$name" hard "$PY" -m chat_vpa.e1_decode --model "$IT_MODEL" "${drafters[@]}" \
    --prompts "$PROMPTS" --limit "$MEASURE_PROMPTS" --max-new-tokens "$MAX_NEW" --dtype fp32 --arms "${arms[@]}" \
    --out "$MEASURE/$name" "${device[@]}" $EXTRA_DECODE
  "$PY" -m chat_vpa.e1_readings one-pass --table "$MEASURE/$name" --drafter "$label" \
    --breakeven "$OUT/breakeven/breakeven.json" --out "$MEASURE/$name" > /dev/null
}

a3_measure_runs() {
  check_python
  check_gpu_free
  [[ -f "$A3/final/realiser.pt" ]] || die "A3-0 has not finished: bash scripts/ncp/e1_step1.sh show-a3"
  measure_one_pass a30 "$A3" l4-fp32-a30
  log "A3-0's table finished: bash scripts/ncp/e1_step1.sh show-a3"
}

# count_with_gate <out> [--drafter LABEL=PATH...]: A1, A2, A3-0 and the given drafters counted, with the oracle
# gate of Section 6.1 read at G = B = 8 and the bars beside it.
count_with_gate() {
  local out="$1"; shift
  local device=()
  if [[ "$REQUIRE_GPU" != 1 ]]; then device=(--device cpu); fi
  # shellcheck disable=SC2086
  step "$out" hard "$PY" -m chat_vpa.e1_count --model "$IT_MODEL" --table "$MEASURE/l4-fp32" --prompts "$PROMPTS" \
    --drafter "a1=$DRAFTER" --drafter "a2=$A2/final" --drafter "a3-0=$A3/final" "$@" --gate a3-oracle:a3-0 \
    --gate-length 8 --breakeven "$OUT/breakeven/breakeven.json" --out "$out" "${device[@]}" $EXTRA_COUNT
}

a3_oracle_runs() {
  check_python
  check_gpu_free
  [[ -f "$A3/final/realiser.pt" && -f "$A3/config.json" ]] || die "A3-0 has not finished: bash scripts/ncp/e1_step1.sh show-a3"
  local variant start source more=()
  variant="$(a3_variant)" || die "A3-0's slot input could not be read from $A3/config.json"
  start="$(a3_start)" || die "R6 could not be read from $MEASURE/readings.json"
  if [[ "$start" == a2 ]]; then source="$A2/final"; else source="$DRAFTER"; fi
  log "A3-oracle starts from $start's weights, as A3-0 did, with A3-0's slot input '$variant'"
  local sets=("slot_input=$variant")
  if [[ -n "$EXTRA_TRAIN" ]]; then read -r -a more <<< "$EXTRA_TRAIN"; sets+=("${more[@]}"); fi
  step "$A3O" hard "$PY" -m chat_vpa.e1_train --arm a3-oracle --model "$IT_MODEL" --assistant "$source" \
    --rollouts "$ROLLOUTS" --held-out "$HELD" --run "$A3O" --set "${sets[@]}"
  [[ -f "$A3O/final/realiser.pt" ]] || die "A3-oracle's training ended without its final checkpoint"
  count_with_gate "$OUT/count/a3-oracle" --drafter "a3-oracle=$A3O/final"
  log "A3-oracle finished: bash scripts/ncp/e1_step1.sh show-a3"
}

# oracle_allows: the gate of Section 6.1 as the counts read it.
oracle_allows() {
  "$PY" -c 'import json, sys; g = json.load(open(sys.argv[1]))["gate"]; sys.exit(0 if g["difference"]["low"] > 0 else 1)' \
    "$OUT/count/a3-oracle/count.json"
}

a3_predict_runs() {
  check_python
  check_gpu_free
  [[ -f "$A3O/final/realiser.pt" && -f "$OUT/count/a3-oracle/count.json" ]] \
    || die "A3-oracle has not finished: bash scripts/ncp/e1_step1.sh a3-oracle"
  oracle_allows || die "the oracle gate stops A3 (docs/e1.md, Section 6.1): bash scripts/ncp/e1_step1.sh show-a3"
  local more=() sets=("slot_input=$(a3_variant)")
  if [[ -n "$EXTRA_TRAIN" ]]; then read -r -a more <<< "$EXTRA_TRAIN"; sets+=("${more[@]}"); fi
  step "$A3P" hard "$PY" -m chat_vpa.e1_train --arm a3 --model "$IT_MODEL" --assistant "$A3O/final" \
    --rollouts "$ROLLOUTS" --held-out "$HELD" --run "$A3P" --set "${sets[@]}"
  [[ -f "$A3P/final/realiser.pt" ]] || die "A3's training ended without its final checkpoint"
  count_with_gate "$OUT/count/a3" --drafter "a3-oracle=$A3O/final" --drafter "a3=$A3P/final"
  measure_one_pass a3 "$A3P" l4-fp32-a3
  log "A3 finished: bash scripts/ncp/e1_step1.sh show-a3"
}

show_a3() {
  "$PY" - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
print("== E1 Step 1, A3 (docs/e1.md, Sections 11.5, 11.12, 11.13)")
if (out / "breakeven" / "breakeven.md").exists():
    print((out / "breakeven" / "breakeven.md").read_text())
else:
    print("breakeven: not computed (bash scripts/ncp/e1_step1.sh breakeven)")
for name in ("a3-0", "a3-oracle", "a3"):
    run = out / name
    if not (run / "metrics.jsonl").exists():
        print(f"{name}: not started")
        continue
    rows = [json.loads(line) for line in open(run / "metrics.jsonl", encoding="utf-8")]
    train = [r for r in rows if "train/loss" in r]
    evals = [r for r in rows if "eval/seconds" in r]
    meta = json.loads((run / "config.json").read_text())
    config = meta["config"]
    print(f"{name}: slot input '{config.get('slot_input', 'replace')}', from {meta['assistant']}")
    if train:
        t = train[-1]
        codes = "".join(f", {label} {t[key]:.3f}" for key, label in (
            ("train/code_perplexity", "code perplexity"), ("train/codes_used", "codes used"),
            ("train/predictor_accuracy", "predictor accuracy")) if key in t)
        print(f"  step {t['step']} of {config['steps']}, loss {t['train/loss']:.3f}{codes}, "
              f"{t['train/steps_per_s']:.3f} steps/s, {t.get('train/eta_h', 0):.1f} h left (predicted), peak "
              f"{t.get('train/peak_memory_gib', 0):.1f} GiB, micro-batch {t.get('train/micro_batch', 0):.0f}"
              + (", done" if (run / "final" / "realiser.pt").exists() else ""))
    for e in evals:
        cycles = "  ".join(f"G={k.rsplit('_', 1)[1]}: {v:.2f}" for k, v in e.items()
                           if k.startswith("eval/tokens_per_cycle_"))
        hits = " ".join(f"{e[f'eval/hit_{k}']:.3f}" for k in range(1, 9) if f"eval/hit_{k}" in e)
        extra = "".join(f"; {label} {e[key]:.3f}" for key, label in (
            ("eval/code_perplexity", "code perplexity"), ("eval/predictor_accuracy", "predictor accuracy"))
            if key in e)
        print(f"  held-out at step {e['step']}: tokens per cycle along the continuations {cycles}; hit by slot "
              f"{hits}{extra}")
for name in ("a3-0", "a3-oracle", "a3"):
    if (out / "count" / name / "count.md").exists():
        print()
        print((out / "count" / name / "count.md").read_text())
for table, label in (("l4-fp32-a30", "a30"), ("l4-fp32-a3", "a3")):
    path = out / "measure" / table / f"one-pass-{label}.md"
    if path.exists():
        print()
        print(path.read_text())
PYEOF
}

case "$ACTION" in
  launch)  start_tmux "$SESSION" "$OUT/timing.log" "bash scripts/ncp/e1_step1.sh run" ;;
  run)     timing_runs; show ;;
  show)    show ;;
  a2)      start_tmux "$SESSION" "$OUT/a2.log" "bash scripts/ncp/e1_step1.sh run-a2" ;;
  run-a2)  a2_runs; show_a2 ;;
  show-a2) show_a2 ;;
  breakeven) (export REQUIRE_GPU=0; check_python) > /dev/null; breakeven > /dev/null; cat "$OUT/breakeven/breakeven.md" ;;
  a3)      [[ "${2:-}" == add || "${2:-}" == replace ]] \
             || die "name A3-0's slot input (docs/e1.md, Section 11.12): e1_step1.sh a3 add, or e1_step1.sh a3 replace"
           start_tmux "$SESSION" "$OUT/a3.log" "bash scripts/ncp/e1_step1.sh run-a3 $2" ;;
  run-a3)  a3_runs "${2:-}"; show_a3 ;;
  show-a3) show_a3 ;;
  a3-measure)     start_tmux "$SESSION" "$OUT/a3-measure.log" "bash scripts/ncp/e1_step1.sh run-a3-measure" ;;
  run-a3-measure) a3_measure_runs; show_a3 ;;
  a3-oracle)      start_tmux "$SESSION" "$OUT/a3-oracle.log" "bash scripts/ncp/e1_step1.sh run-a3-oracle" ;;
  run-a3-oracle)  a3_oracle_runs; show_a3 ;;
  a3-predict)     start_tmux "$SESSION" "$OUT/a3-predict.log" "bash scripts/ncp/e1_step1.sh run-a3-predict" ;;
  run-a3-predict) a3_predict_runs; show_a3 ;;
  *)       die "usage: e1_step1.sh [launch|run|show|a2|run-a2|show-a2|breakeven|a3 add|a3 replace|run-a3 VARIANT|show-a3|a3-measure|a3-oracle|a3-predict]" ;;
esac
