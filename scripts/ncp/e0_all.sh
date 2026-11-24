#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# e0_all.sh -- E0's runs one after another in C12's order (docs/e0.md, Section 4), so that the GPU does not wait for
# the author between them. For each run: its training to the configured steps (resumed where it stands, skipped
# when finished), its evaluation (skipped when all six configurations exist), then the report of everything so far.
#
#   bash scripts/ncp/e0_all.sh          in tmux (session e0-all; log runs/e0-all.log)
#   bash scripts/ncp/e0_all.sh run      the same in the foreground
#   bash scripts/ncp/e0_all.sh show     where each run stands, and the end of the log
#
# Variables (defaults in brackets):
#   NAMES    ["T16 R16 T32 R32"]   the runs, in this order; NAMES="T16 R16" stops after the first two (C12)
#   SESSION  [e0-all]
# Each step is the script that does it alone, with the protocol's settings: e0_train.sh without STOP_AT or
# EXTRA_ARGS (a resumed run keeps its checkpoint's configuration), e0_eval.sh without PARTIAL or EXTRA_ARGS. A
# failure stops the chain, and running it again continues where it stopped. Do not git pull while it runs (bash
# reads a running script from disk). Never shuts the server down.
set -euo pipefail
PASS_VARS="NAMES SESSION"
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ACTION="${1:-launch}"
NAMES="${NAMES:-T16 R16 T32 R32}"
SESSION="${SESSION:-e0-all}"
LOG="$(runs_dir runs/e0-all.log)"
for name in $NAMES; do
  case "$name" in T16|R16|T32|R32) ;; *) die "NAMES holds '$name'; the runs are T16, R16, T32, R32" ;; esac
done

# finished NAME: the run has reached its configured steps (its final checkpoint exists).
finished() {
  local run="$ROOT/runs/e0-$1" steps
  [[ -f "$run/config.json" ]] || return 1
  steps="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["steps"])' "$run/config.json")"
  [[ -f "$run/$(printf 'step-%06d.pt' "$steps")" ]]
}

# evaluated NAME: all six configurations of the run's block size have a summary.
evaluated() {
  local dir="$ROOT/runs/e0-$1/eval/mbpp" block="${1:1}" k order tag
  for k in 4 2; do
    for order in "" -prefix -confidence; do
      tag="block-B$block-K$k$order"
      [[ -f "$dir/$tag/summary.json" ]] || return 1
    done
  done
}

# wait_gpu: the last step's process has released the GPU (up to a minute).
wait_gpu() {
  [[ "$REQUIRE_GPU" == 1 ]] || return 0
  local i used
  for i in $(seq 1 30); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    [[ "${used:-0}" -ge 1024 ]] || return 0
    sleep 2
  done
}

run() {
  local name
  log "chain: $NAMES"
  for name in $NAMES; do
    if finished "$name"; then
      log "$name: trained"
    else
      log "$name: training (bash scripts/ncp/e0_train.sh $name show)"
      wait_gpu
      env -u RUN -u OUT -u STOP_AT -u EXTRA_ARGS bash scripts/ncp/e0_train.sh "$name" run
      finished "$name" || die "$name stopped before its last step"
    fi
    if evaluated "$name"; then
      log "$name: evaluated"
    else
      log "$name: evaluating"
      wait_gpu
      env -u RUN -u OUT -u PARTIAL -u EXTRA_ARGS bash scripts/ncp/e0_eval.sh "$name" run
    fi
    env -u REPORT bash scripts/ncp/e0_eval.sh report > /dev/null
    log "$name: done; the report so far is runs/e0-report/report.md"
  done
  log "chain finished: $NAMES"
}

show() {
  local name state run
  for name in $NAMES; do
    run="$ROOT/runs/e0-$name"
    if [[ ! -d "$run" ]]; then
      state="not started"
    elif finished "$name"; then
      state="trained"
    else
      state="training, $("$PY" - "$run/metrics.jsonl" <<'PYEOF'
import json, sys
try:
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
except FileNotFoundError:
    rows = []
t = [r for r in rows if "train/loss" in r]
print(f"step {t[-1]['step']}, eta {t[-1]['train/eta_h']:.1f} h" if t else "no step logged yet")
PYEOF
)"
    fi
    if evaluated "$name"; then state+=", evaluated"; fi
    echo "  $name: $state"
  done
  if [[ -f "$LOG" ]]; then echo "== the end of $LOG"; tail -n 5 "$LOG"; fi
}

case "$ACTION" in
  launch) start_tmux "$SESSION" "$LOG" "bash scripts/ncp/e0_all.sh run" ;;
  run)    run ;;
  show)   show ;;
  *)      die "usage: e0_all.sh [launch|run|show]" ;;
esac
