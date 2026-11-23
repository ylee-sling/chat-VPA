#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# port_check.sh -- does the ported code reproduce the predecessor on the real model? (docs/provenance.md, Task D.1)
# With text-VPA's checkpoint, settings and seeds, on MBPP's validation split (90 problems x 5 samples; no test
# set), each stage generates samples and compares them, sample by sample, with those of text-VPA's own sweep:
#
#   tests    the repository's test suite in the server's environment
#   canvas   the Stage 0 realizer through the ported whole-canvas sampler, K = 8, exact kernel
#              against REFERENCE/realizer-K8/samples.jsonl
#   block    the same checkpoint through the Block-DFM path with one block of B = 256, K = 8, exact kernel
#              against REFERENCE/realizer-K8/samples.jsonl
#   ar       the autoregressive baseline, the released model's own generate
#              against REFERENCE/ar/samples.jsonl
#
# Identical samples are expected. A stage that differs does not stop the next one; OUT/port_check.txt holds every
# comparison, and `show` prints it.
#
#   bash scripts/ncp/port_check.sh          start in a detached tmux session (SESSION) and return
#   bash scripts/ncp/port_check.sh run      run in the foreground
#   bash scripts/ncp/port_check.sh show     print the comparisons of a finished (or running) check
#
# Variables (defaults in brackets; scripts/ncp/env.sh has the server's):
#   CHECKPOINT  [$TEXT_VPA/runs/stage0-a/last.pt]      the Stage 0 realizer (read only)
#   REFERENCE   [$TEXT_VPA/runs/stage0-a/sweep/mbpp]   text-VPA's samples (read only)
#   OUT         [runs/port-check]                     must lie under this repository's runs/
#   STAGES      [tests canvas block ar]    NUM_SAMPLES [5]    K [8]
#   CANVAS_LEN  [the reference's]           N of the canvas, the block size of `block` and the baseline's budget of
#                                           new tokens: read from REFERENCE's summaries, 256 if they are missing
#   LIMIT       []                          first problems only (the comparison then covers those)
#   SESSION     [port-check]
# Never shuts the server down.
set -euo pipefail
PASS_VARS="CHECKPOINT REFERENCE OUT STAGES NUM_SAMPLES CANVAS_LEN K LIMIT SESSION"
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CHECKPOINT="${CHECKPOINT:-$TEXT_VPA/runs/stage0-a/last.pt}"
REFERENCE="${REFERENCE:-$TEXT_VPA/runs/stage0-a/sweep/mbpp}"
OUT="$(runs_dir "${OUT:-runs/port-check}")"
STAGES="${STAGES:-tests canvas block ar}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
K="${K:-8}"
LIMIT="${LIMIT:-}"
SESSION="${SESSION:-port-check}"
LOG="$OUT/port_check.log"
RESULT="$OUT/port_check.txt"

reference_setting() {   # reference_setting <configuration> <key>: what text-VPA's summary recorded, or 256
  "$PY" - "$REFERENCE/$1/summary.json" "$2" <<'PYEOF'
import json, sys
try:
    print(int(json.load(open(sys.argv[1]))[sys.argv[2]]))
except (OSError, KeyError, ValueError):
    print(256)
PYEOF
}

compare() {   # compare <label> <reference samples> <new samples>
  { echo "== $1"; "$PY" -m chat_vpa.compare_samples "$2" "$3" || true; echo; } | tee -a "$RESULT"
}

run() {
  mkdir -p "$OUT"
  check_python
  check_gpu_free
  check_model
  [[ -f "$CHECKPOINT" ]] || die "$CHECKPOINT is missing (set CHECKPOINT)"
  [[ -d "$REFERENCE" ]] || die "$REFERENCE is missing (set REFERENCE)"
  local canvas="${CANVAS_LEN:-$(reference_setting "realizer-K$K" canvas_len)}"
  local budget="${CANVAS_LEN:-$(reference_setting ar max_new_tokens)}"
  local workers=$(( $(nproc) < 8 ? $(nproc) : 8 ))
  local common=(--model "$MODEL_DIR" --eval-dir "$DATA/eval" --out "$OUT" --benchmarks mbpp
                --mbpp-split validation --num-samples "$NUM_SAMPLES" --exec-workers "$workers")
  if [[ -n "$LIMIT" ]]; then common+=(--limit "$LIMIT"); fi
  if [[ "$REQUIRE_GPU" != 1 ]]; then common+=(--device cpu); fi
  { echo "port check of $(git rev-parse --short HEAD 2>/dev/null || echo '?') on $(date '+%F %T')"
    echo "checkpoint $CHECKPOINT"; echo "reference  $REFERENCE"; echo; } > "$RESULT"
  local stage
  for stage in $STAGES; do
    case "$stage" in
      tests)
        log "tests: the repository's suite in this environment"
        if run_tests; then echo "== tests: passed" | tee -a "$RESULT"; else echo "== tests: FAILED" | tee -a "$RESULT"; fi
        echo >> "$RESULT" ;;
      canvas)
        log "canvas: the ported whole-canvas sampler, K = $K, exact kernel"
        "$PY" -m chat_vpa.evaluate --decoder realizer --checkpoint "$CHECKPOINT" --k-steps "$K" --orders random \
          --canvas-len "$canvas" "${common[@]}" > /dev/null
        compare "whole-canvas sampler, K = $K (ported code against text-VPA)" \
          "$REFERENCE/realizer-K$K/samples.jsonl" "$OUT/mbpp/realizer-K$K/samples.jsonl" ;;
      block)
        log "block: the Block-DFM path with one block of B = $canvas, K = $K, exact kernel"
        "$PY" -m chat_vpa.block_evaluate --checkpoint "$CHECKPOINT" --block-size "$canvas" \
          --canvas-len "$canvas" --k-steps "$K" --orders exact "${common[@]}" > /dev/null
        compare "Block-DFM path, B = $canvas, K = $K (against text-VPA's whole-canvas samples)" \
          "$REFERENCE/realizer-K$K/samples.jsonl" "$OUT/mbpp/block-B$canvas-K$K/samples.jsonl" ;;
      ar)
        log "ar: the autoregressive baseline, the released model's own generate"
        "$PY" -m chat_vpa.evaluate --decoder ar --canvas-len "$budget" "${common[@]}" > /dev/null
        compare "autoregressive baseline (ported code against text-VPA)" \
          "$REFERENCE/ar/samples.jsonl" "$OUT/mbpp/ar/samples.jsonl" ;;
      *) die "unknown stage '$stage' (known: tests canvas block ar)" ;;
    esac
  done
  log "port check finished: $RESULT"
}

case "${1:-launch}" in
  launch) start_tmux "$SESSION" "$LOG" "bash scripts/ncp/port_check.sh run" ;;
  run)    run ;;
  show)   [[ -f "$RESULT" ]] || die "$RESULT is missing: the check has not started"
          cat "$RESULT" ;;
  *)      die "usage: port_check.sh [launch|run|show]" ;;
esac
