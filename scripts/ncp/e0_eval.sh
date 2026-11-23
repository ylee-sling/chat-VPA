#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# e0_eval.sh -- E0's evaluation of one run (docs/e0.md, Sections 5 and 6), and the report of all runs.
#
#   bash scripts/ncp/e0_eval.sh T16          the final checkpoint of runs/e0-T16 on MBPP validation, 90 x 5: K = 4
#                                            and 2 at the run's own B, with the exact kernel, the prefix-growing and
#                                            the confidence orders (C8), then the outcome analysis; in tmux
#   bash scripts/ncp/e0_eval.sh T16 run      the same in the foreground
#   bash scripts/ncp/e0_eval.sh T16 show     Pass@1, parses, K-bar, stops and seconds per block per configuration
#   bash scripts/ncp/e0_eval.sh report       the report of every run evaluated so far (REPORT/report.md); no GPU
#
# Variables (defaults in brackets; scripts/ncp/env.sh has the server's):
#   RUN          [runs/e0-NAME]     the training run: it must have reached its configured steps (C7: the final
#                                   checkpoint is the one evaluated)
#   PARTIAL      [0]                1: evaluate last.pt of an unfinished run into RUN/eval-partial, a diagnostic
#                                   that the report does not read
#   OUT          [RUN/eval]         must lie under this repository's runs/
#   EXTRA_ARGS   []                 further options of chat_vpa.block_evaluate (dry runs: the report lists every
#                                   change of what Section 5 and C8 fix as a deviation)
#   SESSION      [e0-eval-NAME]
#   BASELINE     [$TEXT_VPA/runs/stage0-a/sweep/mbpp/ar]   the AR baseline's samples (read only)
#   PREDECESSOR  [$TEXT_VPA/runs/stage0-a/sweep/mbpp]      the whole-canvas rows of the table (read only)
#   DATA_DIR     [runs/e0-data]     REPORT [runs/e0-report]
# Never shuts the server down.
set -euo pipefail
PASS_VARS="RUN PARTIAL OUT EXTRA_ARGS SESSION BASELINE PREDECESSOR DATA_DIR REPORT"
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

NAME="${1:-}"
ACTION="${2:-launch}"
BASELINE="${BASELINE:-$TEXT_VPA/runs/stage0-a/sweep/mbpp/ar}"
PREDECESSOR="${PREDECESSOR:-$TEXT_VPA/runs/stage0-a/sweep/mbpp}"
DATA_DIR="$(runs_dir "${DATA_DIR:-runs/e0-data}")"
REPORT="$(runs_dir "${REPORT:-runs/e0-report}")"

report() {
  export REQUIRE_GPU=0                       # reads files only: no GPU check, so it may run beside a training run
  check_python
  [[ -f "$BASELINE/samples.jsonl" ]] || die "$BASELINE/samples.jsonl is missing (set BASELINE)"
  "$PY" -m chat_vpa.e0_report --runs "$ROOT/runs" --baseline "$BASELINE" --predecessor "$PREDECESSOR" \
    --data "$DATA_DIR" --mbpp "$DATA/eval/mbpp.jsonl" --out "$REPORT"
  echo "written: $REPORT/report.md, $REPORT/report.json"
}

if [[ "$NAME" == report ]]; then
  report
  exit 0
fi
case "$NAME" in
  T16|R16|T32|R32) ;;
  *) die "usage: e0_eval.sh T16|R16|T32|R32 [launch|run|show], or e0_eval.sh report" ;;
esac
RUN="$(runs_dir "${RUN:-runs/e0-$NAME}")"
PARTIAL="${PARTIAL:-0}"
if [[ "$PARTIAL" == 1 ]]; then OUT="$(runs_dir "${OUT:-$RUN/eval-partial}")"; else OUT="$(runs_dir "${OUT:-$RUN/eval}")"; fi
EXTRA_ARGS="${EXTRA_ARGS:-}"
SESSION="${SESSION:-e0-eval-$NAME}"

run() {
  check_python
  check_gpu_free
  check_model
  [[ -f "$RUN/last.pt" && -f "$RUN/config.json" ]] || die "$RUN has no checkpoint: bash scripts/ncp/e0_train.sh $NAME first"
  local steps final
  steps="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["steps"])' "$RUN/config.json")"
  final="$(printf 'step-%06d.pt' "$steps")"
  if [[ ! -f "$RUN/$final" && "$PARTIAL" != 1 ]]; then
    die "$RUN has not reached its $steps steps ($final is missing): the final checkpoint is the one evaluated (C7); PARTIAL=1 evaluates last.pt as a diagnostic"
  fi
  mkdir -p "$OUT"
  local device=()
  if [[ "$REQUIRE_GPU" != 1 ]]; then device=(--device cpu); fi
  log "evaluating $RUN/last.pt into $OUT"
  # shellcheck disable=SC2086
  "$PY" -m chat_vpa.block_evaluate --model "$MODEL_DIR" --checkpoint "$RUN/last.pt" --eval-dir "$DATA/eval" \
    --benchmarks mbpp --mbpp-split validation --num-samples 5 --k 1 --k-steps 4 2 \
    --orders exact prefix confidence --out "$OUT" "${device[@]}" $EXTRA_ARGS
  "$PY" -m chat_vpa.analyze --eval "$OUT" --mbpp "$DATA/eval/mbpp.jsonl" > /dev/null
  log "finished: $OUT (bash scripts/ncp/e0_eval.sh $NAME show)"
}

show() {
  [[ -d "$OUT/mbpp" ]] || die "$OUT/mbpp does not exist: $NAME has not been evaluated"
  "$PY" - "$OUT/mbpp" <<'PYEOF'
import json, sys
from pathlib import Path
for d in sorted(Path(sys.argv[1]).iterdir()):
    if not (d / "summary.json").exists():
        continue
    s = json.loads((d / "summary.json").read_text())
    stops = ", ".join(f"{k} {100 * v:.0f}%" for k, v in s["stops"].items())
    print(f"  {d.name}: Pass@1 {100 * s['pass@1']:.1f}%, parses {100 * s['parse_rate']:.1f}%, K-bar "
          f"{s['mean_passes_per_block']}, {s['mean_blocks']} blocks, stops {stops}, {s['seconds_per_block']} s per "
          f"block, audit {s['pass_audit']}, step {s['step']}")
PYEOF
}

case "$ACTION" in
  launch) start_tmux "$SESSION" "$OUT/eval.log" "bash scripts/ncp/e0_eval.sh $NAME run" ;;
  run)    run ;;
  show)   show ;;
  *)      die "usage: e0_eval.sh T16|R16|T32|R32 [launch|run|show], or e0_eval.sh report" ;;
esac
