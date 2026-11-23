#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# e0_data.sh -- E0's data (docs/e0.md, C6): the validation set and 100,000 kept training examples, each with its
# human reference and one rollout of the frozen backbone, stratified by format (13.5% write). Predicted 15 to 17 h
# at batch 128 on the L4; chat_vpa/rollout_writer.py says what is written. Nothing generated is executed.
#
#   bash scripts/ncp/e0_data.sh check    first: rollouts per second and peak memory at batch 128 and 256 on the
#                                        first 1,024 prompts, in tmux (session e0-batch-check), about 15 minutes
#   bash scripts/ncp/e0_data.sh          start, or resume after the last finished chunk, in tmux (SESSION)
#   bash scripts/ncp/e0_data.sh run      the same in the foreground
#   bash scripts/ncp/e0_data.sh show     the check's result, and the progress or the summary of the data
#
# Variables (defaults in brackets; scripts/ncp/env.sh has the server's):
#   OUT         [runs/e0-data]   must lie under this repository's runs/
#   BATCH_SIZE  [128]            256 if the check finds it faster; halved by the writer on out-of-memory
#   KEPT [100000]  WRITE_SHARE [0.135]  VAL_EXAMPLES [256]  CHUNK_SIZE [2048]   C6; a resumed run keeps them
#   EXTRA_ARGS  []               further arguments of chat_vpa.rollout_writer
#   CHECK_PROMPTS [1024]  CHECK_BATCH_SIZES ["128 256"]   the check's
#   SESSION     [e0-data]
# Never shuts the server down.
set -euo pipefail
PASS_VARS="OUT BATCH_SIZE KEPT WRITE_SHARE VAL_EXAMPLES CHUNK_SIZE EXTRA_ARGS CHECK_PROMPTS CHECK_BATCH_SIZES SESSION"
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

OUT="$(runs_dir "${OUT:-runs/e0-data}")"
BATCH_SIZE="${BATCH_SIZE:-128}"
KEPT="${KEPT:-100000}"
WRITE_SHARE="${WRITE_SHARE:-0.135}"
VAL_EXAMPLES="${VAL_EXAMPLES:-256}"
CHUNK_SIZE="${CHUNK_SIZE:-2048}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CHECK_PROMPTS="${CHECK_PROMPTS:-1024}"
CHECK_BATCH_SIZES="${CHECK_BATCH_SIZES:-128 256}"
SESSION="${SESSION:-e0-data}"
CHECK="$(runs_dir runs/e0-batch-check)"

check_result() {
  [[ -f "$CHECK/throughput.json" ]] || { echo "the check has not finished ($CHECK/throughput.json)"; return 0; }
  "$PY" - "$CHECK/throughput.json" <<'PYEOF'
import json, sys
result = json.load(open(sys.argv[1]))
done = []
for e in result["batch_sizes"]:
    if e.get("out_of_memory"):
        print(f"  batch {e['batch_size']}: out of memory")
        continue
    done.append(e)
    print(f"  batch {e['batch_size']}: {e['rollouts_per_second']} rollouts/s, peak {e['peak_memory_gib']} GiB, "
          f"kept share {e['kept_share']}")
if done:
    best = max(done, key=lambda e: e["rollouts_per_second"])
    print(f"  faster: batch {best['batch_size']}; start the data with "
          f"BATCH_SIZE={best['batch_size']} bash scripts/ncp/e0_data.sh")
PYEOF
}

run() {
  mkdir -p "$OUT"
  check_python
  check_gpu_free
  check_model
  local device=()
  if [[ "$REQUIRE_GPU" != 1 ]]; then device=(--device cpu); fi
  # shellcheck disable=SC2086
  "$PY" -m chat_vpa.rollout_writer --model "$MODEL_DIR" --corpus "$DATA/corpus" --out "$OUT" --kept "$KEPT" \
    --write-share "$WRITE_SHARE" --val-examples "$VAL_EXAMPLES" --chunk-size "$CHUNK_SIZE" \
    --batch-size "$BATCH_SIZE" "${device[@]}" $EXTRA_ARGS
}

case "${1:-launch}" in
  launch)    start_tmux "$SESSION" "$OUT/rollout_writer.log" "bash scripts/ncp/e0_data.sh run" ;;
  run)       run ;;
  check)     start_tmux "e0-batch-check" "$CHECK/check.log" "bash scripts/ncp/e0_data.sh check-run" ;;
  check-run) OUT="$CHECK" NUM_PROMPTS="$CHECK_PROMPTS" BATCH_SIZES="$CHECK_BATCH_SIZES" \
               bash scripts/ncp/rollout_throughput.sh run
             check_result ;;
  show)      echo "== the batch-size check"; check_result
             echo "== the data"; "$PY" -m chat_vpa.rollout_writer --show "$OUT" ;;
  *)         die "usage: e0_data.sh [launch|run|check|show]" ;;
esac
