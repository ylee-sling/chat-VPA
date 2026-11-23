#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# e0_train.sh -- one E0 run (docs/e0.md, C3 and C12: T16 and R16 first, then T32 and R32). T is the rollout arm,
# R the reference arm; 16 or 32 is the block size B. chat_vpa/train_e0.py says what is trained.
#
#   STOP_AT=100 bash scripts/ncp/e0_train.sh T16    the first 100 steps only, in tmux: RUN/timing.json holds the
#                                                    speed, the peak memory and the hours projected for the run
#   bash scripts/ncp/e0_train.sh T16                 start, or resume to the configured 10,000 steps, in tmux
#   bash scripts/ncp/e0_train.sh T16 run             the same in the foreground
#   bash scripts/ncp/e0_train.sh T16 show            timing, memory check, the last training and held-out metrics
#
# Variables (defaults in brackets; scripts/ncp/env.sh has the server's):
#   DATA_DIR    [runs/e0-data]    the rollout writer's output
#   RUN         [runs/e0-NAME]    must lie under this repository's runs/
#   STOP_AT     []                end after this step; a later start resumes
#   EXTRA_ARGS  []                further KEY=VALUE settings of a new run, or options of chat_vpa.train_e0
#   SESSION     [e0-NAME]
# Never shuts the server down.
set -euo pipefail
PASS_VARS="DATA_DIR RUN STOP_AT EXTRA_ARGS SESSION"
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

NAME="${1:-}"
ACTION="${2:-launch}"
case "$NAME" in
  T16|R16|T32|R32) ;;
  *) die "usage: e0_train.sh T16|R16|T32|R32 [launch|run|show]" ;;
esac
if [[ "${NAME:0:1}" == T ]]; then ARM=rollouts; else ARM=references; fi
BLOCK="${NAME:1}"
DATA_DIR="$(runs_dir "${DATA_DIR:-runs/e0-data}")"
RUN="$(runs_dir "${RUN:-runs/e0-$NAME}")"
STOP_AT="${STOP_AT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SESSION="${SESSION:-e0-$NAME}"

run() {
  check_python
  check_gpu_free
  check_model
  [[ -f "$DATA_DIR/train.jsonl.gz" ]] || die "$DATA_DIR/train.jsonl.gz is missing: bash scripts/ncp/e0_data.sh first"
  local extra=()
  if [[ -n "$STOP_AT" ]]; then extra+=(--stop-at "$STOP_AT"); fi
  if [[ "$REQUIRE_GPU" != 1 ]]; then extra+=(--device cpu); fi
  # shellcheck disable=SC2086
  "$PY" -m chat_vpa.train_e0 --data "$DATA_DIR" --run "$RUN" --model "$MODEL_DIR" "${extra[@]}" \
    --set arm="$ARM" block_size="$BLOCK" $EXTRA_ARGS
}

show() {
  [[ -d "$RUN" ]] || die "$RUN does not exist: the run has not started"
  local name
  for name in timing.json memory.json; do
    if [[ -f "$RUN/$name" ]]; then echo "== $name"; cat "$RUN/$name"; echo; fi
  done
  if [[ -f "$RUN/metrics.jsonl" ]]; then
    "$PY" - "$RUN/metrics.jsonl" <<'PYEOF'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
train = [r for r in rows if "train/loss" in r]
if train:
    t = train[-1]
    print(f"== step {t['step']}: loss {t['train/loss']:.4f}, accuracy {t['train/accuracy']:.4f}, "
          f"{t['train/steps_per_s']:.3f} steps/s, peak {t['train/peak_gib']:.1f} GiB, eta {t['train/eta_h']:.1f} h, "
          f"epoch {int(t['train/epoch'])}")
keys = ("val_references_loss", "val_rollouts_loss", "val_references_accuracy", "val_rollouts_accuracy")
for r in rows:
    if f"eval/{keys[0]}" in r:
        print(f"   held out at step {r['step']}: " + ", ".join(f"{k[4:]} {r['eval/' + k]:.4f}" for k in keys))
PYEOF
  fi
}

case "$ACTION" in
  launch) start_tmux "$SESSION" "$RUN/train.log" "bash scripts/ncp/e0_train.sh $NAME run" ;;
  run)    mkdir -p "$RUN"; run ;;
  show)   show ;;
  *)      die "usage: e0_train.sh T16|R16|T32|R32 [launch|run|show]" ;;
esac
