#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# stage0_sweep.sh -- text-VPA's diagnostic sweep of a Stage 0 realizer, through the ported code, on MBPP's
# validation split (90 problems; the test sets stay untouched). In chat-VPA its purpose is the port check of
# docs/provenance.md: the same checkpoint, settings and seeds must give the predecessor's samples. Stages
# (STAGES runs them in the order given):
#
#   baseline     the released model's own generate.
#   code         code check: the realizer's code path with the adapters off, causal masks and left_to_right at
#                K = 256 (exactly one token per step) is the backbone's own decoding.
#   adaptation   the trained realizer, left_to_right at K = 256.
#   matched      the same with (t, h) as seen in training, and the exact jump kernel at K = 32 likewise.
#   l2r          left_to_right at K_STEPS.
#   parallel     the orders in ORDERS at K_STEPS (random: the exact jump kernel; confidence).
#
#   bash scripts/ncp/stage0_sweep.sh          start in a detached tmux session (SESSION) and return
#   bash scripts/ncp/stage0_sweep.sh run      run in the foreground
#
# Results: OUT/mbpp/<tag>/{samples.jsonl, summary.json}, OUT/report.md, OUT/analysis.md (rewritten after each
# stage). Rerunning skips finished configurations.
#
# Variables (defaults in brackets; scripts/ncp/env.sh has the server's):
#   CHECKPOINT  [$TEXT_VPA/runs/stage0-a/last.pt]   the Stage 0 realizer (read only)
#   OUT         [runs/port-check/sweep]            must lie under this repository's runs/
#   K_STEPS     [8 32 128]    ORDERS [random confidence]    NUM_SAMPLES [5]
#   STAGES      [baseline code adaptation matched l2r parallel]
#   LIMIT       []            first problems only (a smoke run)
#   SESSION     [port-check-sweep]
# Never shuts the server down.
set -euo pipefail
PASS_VARS="CHECKPOINT OUT K_STEPS ORDERS NUM_SAMPLES STAGES LIMIT SESSION"
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CHECKPOINT="${CHECKPOINT:-$TEXT_VPA/runs/stage0-a/last.pt}"
OUT="$(runs_dir "${OUT:-runs/port-check/sweep}")"
K_STEPS="${K_STEPS:-8 32 128}"
ORDERS="${ORDERS:-random confidence}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
STAGES="${STAGES:-baseline code adaptation matched l2r parallel}"
LIMIT="${LIMIT:-}"
SESSION="${SESSION:-port-check-sweep}"
LOG="$OUT/sweep.log"
KNOWN_STAGES="baseline code adaptation matched l2r parallel"

summarize() {
  "$PY" -m chat_vpa.report --eval "$OUT" --no-verdict >/dev/null
  "$PY" -m chat_vpa.analyze --eval "$OUT" --mbpp "$DATA/eval/mbpp.jsonl" >/dev/null
  log "tables updated: $OUT/report.md, $OUT/analysis.md"
}

run() {
  mkdir -p "$OUT"
  check_python
  check_gpu_free
  check_model
  [[ -f "$CHECKPOINT" ]] || die "$CHECKPOINT is missing (set CHECKPOINT)"
  [[ -f "$DATA/eval/mbpp.jsonl" ]] || die "$DATA/eval/mbpp.jsonl is missing (set DATA)"
  local workers=$(( $(nproc) < 8 ? $(nproc) : 8 ))
  local ks stages orders stage
  read -r -a ks <<< "$K_STEPS"
  read -r -a stages <<< "$STAGES"
  read -r -a orders <<< "$ORDERS"
  for stage in "${stages[@]}"; do
    [[ " $KNOWN_STAGES " == *" $stage "* ]] || die "unknown stage '$stage' (known: $KNOWN_STAGES)"
  done
  local common=(--model "$MODEL_DIR" --eval-dir "$DATA/eval" --out "$OUT" --benchmarks mbpp
                --mbpp-split validation --num-samples "$NUM_SAMPLES" --exec-workers "$workers")
  if [[ -n "$LIMIT" ]]; then common+=(--limit "$LIMIT"); fi
  if [[ "$REQUIRE_GPU" != 1 ]]; then common+=(--device cpu); fi
  local realizer=(--decoder realizer --checkpoint "$CHECKPOINT")
  log "sweep of $CHECKPOINT: MBPP validation, $NUM_SAMPLES samples per problem, K = ${ks[*]}, stages: ${stages[*]}"

  local i=0
  for stage in "${stages[@]}"; do
    i=$((i + 1))
    case "$stage" in
      baseline)
        log "$i/${#stages[@]} baseline: the released model's own generate"
        "$PY" -m chat_vpa.evaluate --decoder ar --canvas-len 256 "${common[@]}" ;;
      code)
        log "$i/${#stages[@]} code check: the backbone through the realizer's code path (adapters off, causal, one token per step)"
        "$PY" -m chat_vpa.evaluate "${realizer[@]}" --k-steps 256 --orders left_to_right --canvas-mode causal \
          --no-adapters "${common[@]}" ;;
      adaptation)
        log "$i/${#stages[@]} the trained realizer, one token per step"
        "$PY" -m chat_vpa.evaluate "${realizer[@]}" --k-steps 256 --orders left_to_right "${common[@]}" ;;
      matched)
        log "$i/${#stages[@]} (t, h) as seen in training: one token per step, and the exact jump kernel at K = 32"
        "$PY" -m chat_vpa.evaluate "${realizer[@]}" --k-steps 256 --orders left_to_right --time-input matched \
          "${common[@]}"
        "$PY" -m chat_vpa.evaluate "${realizer[@]}" --k-steps 32 --orders random --time-input matched \
          "${common[@]}" ;;
      l2r)
        log "$i/${#stages[@]} the trained realizer, left_to_right at K = ${ks[*]}"
        "$PY" -m chat_vpa.evaluate "${realizer[@]}" --k-steps "${ks[@]}" --orders left_to_right "${common[@]}" ;;
      parallel)
        log "$i/${#stages[@]} the trained realizer, ${orders[*]} at K = ${ks[*]}"
        "$PY" -m chat_vpa.evaluate "${realizer[@]}" --k-steps "${ks[@]}" --orders "${orders[@]}" "${common[@]}" ;;
    esac
    summarize
  done
  log "sweep finished: $OUT/report.md, $OUT/analysis.md"
}

case "${1:-launch}" in
  launch) start_tmux "$SESSION" "$LOG" "bash scripts/ncp/stage0_sweep.sh run" ;;
  run)    run ;;
  *)      die "usage: stage0_sweep.sh [launch|run]" ;;
esac
