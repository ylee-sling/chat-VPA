#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# first_checks.sh -- everything the L4 has to say before E0 is confirmed, one after the other in one tmux
# session (one GPU job at a time). Predicted, not measured: one to two hours (text-VPA's sweep took 25 to 40
# minutes per configuration, as the author reports, and the port check holds three):
#
#   1. rollout_throughput.sh   how many AR rollouts are affordable            runs/e0-throughput
#   2. port_check.sh           the port reproduces the predecessor's samples  runs/port-check
#   3. block_check.sh          the block machinery on the real weights        runs/block-check
#
#   bash scripts/ncp/first_checks.sh          start in a detached tmux session ("first-checks") and return
#   bash scripts/ncp/first_checks.sh show     print the three summaries (what to paste back)
#
# The steps run with their defaults; to change one (LIMIT, STAGES, ...), run that step's own script. A step that
# fails does not stop the next one. Nothing is trained and no rollout is kept as a training target. Never shuts
# the server down.
set -euo pipefail
PASS_VARS=""
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LOG="$(runs_dir runs/first-checks)/first_checks.log"

run() {
  local step
  for step in rollout_throughput port_check block_check; do
    log "=== $step"
    bash "scripts/ncp/$step.sh" run || log "$step stopped with an error; continuing"
  done
  log "first checks finished: bash scripts/ncp/first_checks.sh show"
}

show() {
  local step
  for step in rollout_throughput port_check block_check; do
    echo "################ $step"
    bash "scripts/ncp/$step.sh" show 2>&1 || true
    echo
  done
  if command -v tmux >/dev/null && tmux has-session -t "=first-checks" 2>/dev/null; then
    echo "(tmux session 'first-checks' is still running)"
  fi
}

case "${1:-launch}" in
  launch) start_tmux "first-checks" "$LOG" "bash scripts/ncp/first_checks.sh run" ;;
  run)    run ;;
  show)   show ;;
  *)      die "usage: first_checks.sh [launch|run|show]" ;;
esac
