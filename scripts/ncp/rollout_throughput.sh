#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# rollout_throughput.sh -- how many autoregressive rollouts of the frozen backbone are affordable (about 15
# minutes on the L4; chat_vpa/rollout_throughput.py says what is measured). It reads text-VPA's checkpoint and
# corpus, writes OUT/throughput.json, OUT/examples.txt and OUT/rollouts.jsonl (and, when started in tmux,
# OUT/throughput.log), stores no rollout as a training target and executes no generated code.
#
#   bash scripts/ncp/rollout_throughput.sh          start in a detached tmux session (SESSION) and return
#   bash scripts/ncp/rollout_throughput.sh run      run in the foreground
#   bash scripts/ncp/rollout_throughput.sh show     print the summary of a finished measurement
#   bash scripts/ncp/rollout_throughput.sh inspect  how its rollouts ended (chat_vpa/inspect_rollouts.py), from
#                                                   OUT/rollouts.jsonl, or OUT/examples.txt for an earlier run
#
# Variables (defaults in brackets; scripts/ncp/env.sh has the server's):
#   OUT          [runs/e0-throughput]     must lie under this repository's runs/
#   NUM_PROMPTS  [256]                    the first training examples in CorpusStream order
#   BATCH_SIZES  [16 32 64 128]           tried in this order; stops at the first that runs out of memory
#   EXTRA_ARGS   []                       further arguments of chat_vpa.rollout_throughput, e.g. --no-sort, or
#                                         "--kind write --examples 64" for the write prompts only
#   SESSION      [e0-throughput]
# Never shuts the server down.
set -euo pipefail
PASS_VARS="OUT NUM_PROMPTS BATCH_SIZES EXTRA_ARGS SESSION"
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

OUT="$(runs_dir "${OUT:-runs/e0-throughput}")"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
BATCH_SIZES="${BATCH_SIZES:-16 32 64 128}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SESSION="${SESSION:-e0-throughput}"
LOG="$OUT/throughput.log"

run() {
  mkdir -p "$OUT"
  check_python
  check_gpu_free
  check_model
  ls "$DATA"/corpus/train-*.jsonl.gz >/dev/null 2>&1 || die "$DATA/corpus has no train-*.jsonl.gz shards (set DATA)"
  local sizes device=()
  read -r -a sizes <<< "$BATCH_SIZES"
  if [[ "$REQUIRE_GPU" != 1 ]]; then device=(--device cpu); fi
  log "rollout throughput: $NUM_PROMPTS prompts, batch sizes ${sizes[*]}, model $MODEL_DIR"
  # shellcheck disable=SC2086
  "$PY" -m chat_vpa.rollout_throughput --model "$MODEL_DIR" --corpus "$DATA/corpus" --out "$OUT" \
    --num-prompts "$NUM_PROMPTS" --batch-sizes "${sizes[@]}" "${device[@]}" $EXTRA_ARGS
  log "finished: $OUT/throughput.json, $OUT/examples.txt"
}

case "${1:-launch}" in
  launch) start_tmux "$SESSION" "$LOG" "bash scripts/ncp/rollout_throughput.sh run" ;;
  run)    run ;;
  show)   [[ -f "$OUT/throughput.json" ]] || die "$OUT/throughput.json is missing: the measurement has not finished"
          "$PY" -m chat_vpa.rollout_throughput --show "$OUT" ;;
  inspect) source_file="$OUT/rollouts.jsonl"
          [[ -f "$source_file" ]] || source_file="$OUT/examples.txt"
          [[ -f "$source_file" ]] || die "neither $OUT/rollouts.jsonl nor $OUT/examples.txt exists"
          "$PY" -m chat_vpa.inspect_rollouts "$source_file" ;;
  *)      die "usage: rollout_throughput.sh [launch|run|show|inspect]" ;;
esac
