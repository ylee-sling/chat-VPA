#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# block_check.sh -- the block machinery on the real weights (Task D.2): a diagnostic, no training, NOT a result.
# text-VPA's Stage 0 adapters are put inside block windows on the first problems of MBPP's validation split:
#
#   (B, K) = (16, 16), prefix-growing   one token per pass inside each block
#   (B, K) = (16, 4), exact kernel      the design point of H1
#
# Those adapters were trained on whole canvases and never saw a window that ends after B slots, so Pass@1 is
# expected to be low and says nothing about Block-DFM. What the check is for: every block runs under the pass
# audit (K-bar passes, no commit row above the KV-storing blocks), the cache built by the fused commits is
# compared with the frozen backbone's on a few problems, and the seconds per block are measured.
#
#   bash scripts/ncp/block_check.sh          start in a detached tmux session (SESSION) and return
#   bash scripts/ncp/block_check.sh run      run in the foreground
#   bash scripts/ncp/block_check.sh show     print the summary
#
# Variables (defaults in brackets; scripts/ncp/env.sh has the server's):
#   CHECKPOINT  [$TEXT_VPA/runs/stage0-a/last.pt]   the Stage 0 adapters (read only)
#   OUT         [runs/block-check]                 must lie under this repository's runs/
#   LIMIT       [10]     NUM_SAMPLES [5]     BLOCK [16]     CACHE_CHECK [3]     CANVAS_LEN [256]
#   SESSION     [block-check]
# Never shuts the server down.
set -euo pipefail
PASS_VARS="CHECKPOINT OUT LIMIT NUM_SAMPLES BLOCK CACHE_CHECK CANVAS_LEN SESSION"
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CHECKPOINT="${CHECKPOINT:-$TEXT_VPA/runs/stage0-a/last.pt}"
OUT="$(runs_dir "${OUT:-runs/block-check}")"
LIMIT="${LIMIT:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
BLOCK="${BLOCK:-16}"
CACHE_CHECK="${CACHE_CHECK:-3}"
CANVAS_LEN="${CANVAS_LEN:-256}"
SESSION="${SESSION:-block-check}"
LOG="$OUT/block_check.log"

show() {
  "$PY" - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]) / "mbpp"
tags = sorted(d for d in root.iterdir() if (d / "summary.json").exists()) if root.exists() else []
if not tags:
    sys.exit(f"no finished configuration under {root}")
print("block machinery on the real weights: a diagnostic of the code, NOT a result of Block-DFM")
for d in tags:
    s = json.loads((d / "summary.json").read_text())
    print(f"{s['tag']}: {s['problems']} problems x {s['samples_per_problem']} samples, checkpoint step {s['step']}")
    print(f"  pass audit {s['pass_audit']}; blocks per sample {s['mean_blocks']}; passes per block (K-bar) "
          f"{s['mean_passes_per_block']}; seconds per block {s['seconds_per_block']}, per pass {s['seconds_per_pass']} "
          f"(batches of {min(s['samples_per_problem'], 10)} rows)")
    print(f"  stops {s['stops']}; mean tokens {s['mean_tokens']}; parses {100 * s['parse_rate']:.1f}%; "
          f"Pass@1 {100 * s['pass@1']:.1f}% (not a result)")
    for c in s.get("cache_check", []):
        if c.get("blocks", 0) < 2:
            print(f"  cache check {c['task_id']}: fewer than two blocks, nothing committed")
            continue
        fused, separate, leak = c["fused_commit"], c["separate_commit"], c["adapters_on"]
        # The rule of docs/e0.md, Section 2.3, written before the run.
        if leak < 10 * separate:
            verdict = "UNINFORMATIVE (adapters on the commit rows move the cache by less than 10 times the rounding)"
        elif fused <= 4 * separate:
            verdict = "consistent with rounding"
        else:
            verdict = "DIFFERS BEYOND ROUNDING: look at this before anything else"
        print(f"  cache check {c['task_id']}: {int(c['committed_positions'])} committed positions; largest difference to "
              f"the frozen backbone's prefill: fused commit {fused:.4g}, separate commit {separate:.4g}, with adapters "
              f"on the commit rows {leak:.4g} (reference RMS {c['reference_rms']:.3g}): {verdict}")
PYEOF
}

run() {
  mkdir -p "$OUT"
  check_python
  check_gpu_free
  check_model
  [[ -f "$CHECKPOINT" ]] || die "$CHECKPOINT is missing (set CHECKPOINT)"
  local workers=$(( $(nproc) < 8 ? $(nproc) : 8 ))
  local common=(--model "$MODEL_DIR" --checkpoint "$CHECKPOINT" --eval-dir "$DATA/eval" --out "$OUT"
                --benchmarks mbpp --mbpp-split validation --num-samples "$NUM_SAMPLES" --limit "$LIMIT"
                --block-size "$BLOCK" --canvas-len "$CANVAS_LEN" --cache-check "$CACHE_CHECK"
                --exec-workers "$workers")
  if [[ "$REQUIRE_GPU" != 1 ]]; then common+=(--device cpu); fi
  log "(B, K) = ($BLOCK, $BLOCK), prefix-growing: one token per pass inside each block"
  "$PY" -m chat_vpa.block_evaluate --k-steps "$BLOCK" --orders prefix "${common[@]}" > /dev/null
  log "(B, K) = ($BLOCK, 4), exact kernel"
  "$PY" -m chat_vpa.block_evaluate --k-steps 4 --orders exact "${common[@]}" > /dev/null
  show
  log "block check finished: $OUT"
}

case "${1:-launch}" in
  launch) start_tmux "$SESSION" "$LOG" "bash scripts/ncp/block_check.sh run" ;;
  run)    run ;;
  show)   show ;;
  *)      die "usage: block_check.sh [launch|run|show]" ;;
esac
