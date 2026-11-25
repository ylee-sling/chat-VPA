#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# e1_step0.sh -- E1's Step 0 (docs/e1.md, Section 3): four measurements, nothing trained.
#
#   bash scripts/ncp/e1_step0.sh           all parts, one after another, in tmux (session e1-step0)
#   bash scripts/ncp/e1_step0.sh run       the same in the foreground
#   bash scripts/ncp/e1_step0.sh show      the tables of what has finished (also written to OUT/step0.md); no GPU
#
#   bridge    T16 and R16 as block drafters that the frozen backbone verifies greedily (chat_vpa/e1_bridge.py):
#             MBPP validation, K = 1, 2, 4, prefix-growing and confidence orders; passes counted, not timed
#   ceiling   the target alone: the time of a pass over n positions and what it allows (chat_vpa/e1_ceiling.py),
#             on the GPU in bfloat16 and, if it fits, float32, and on the CPU with THREADS threads
#   decode    greedy decoding of the development prompts, alone and with Google's drafter at draft lengths
#             2, 4, 6, 8, and with prompt lookup (chat_vpa/e1_decode.py): the GPU in bfloat16 (all arms), a
#             profiled slice, the GPU in float32 if it fits, and a CPU slice
#   litert    the same prompts in LiteRT-LM, Google's on-device runtime, with its build of the model: speculative
#             decoding off, on and adaptive (chat_vpa/e1_litert.py, in LiteRT-LM's own environment), on the CPU
#             with THREADS threads and, if the runtime can use it, on the GPU
#
# A part that has finished is skipped (its .done marker), so the script can be run again after a failure. Some
# runs may not be possible on this machine: float32 on the GPU may not fit the L4's memory, and LiteRT-LM's GPU
# backend needs a driver it can use. A failure there is logged and the script goes on. The vLLM arm of Step 0 is
# separate (docs/e1.md, Section 3.4): it is written after the preflight has shown the installed version's flags.
#
# Variables (defaults in brackets):
#   PARTS ["bridge ceiling decode litert"]   OUT [runs/e1-step0]   ASSETS [runs/assets]   THREADS [4]
#   DECODE_PROMPTS [100]  MAX_NEW [256]  ARMS ["ar mtp:2 mtp:4 mtp:6 mtp:8 lookup:10"]
#   FP32_PROMPTS [50]     CPU_PROMPTS [20]  CPU_MAX_NEW [128]  CPU_ARMS ["ar mtp:2 mtp:4"]
#   CEILING_PROMPTS [20]  CPU_CEILING_PROMPTS [5]  LITERT_PROMPTS [50]  LITERT_BACKENDS ["cpu gpu"]
#   EXTRA_BRIDGE [], EXTRA_DECODE [], EXTRA_CEILING [], EXTRA_LITERT [] (dry runs)
# Never shuts the server down. Do not git pull while it runs.
set -euo pipefail
PASS_VARS="PARTS OUT ASSETS THREADS DECODE_PROMPTS MAX_NEW ARMS FP32_PROMPTS CPU_PROMPTS CPU_MAX_NEW CPU_ARMS CEILING_PROMPTS CPU_CEILING_PROMPTS LITERT_PROMPTS LITERT_BACKENDS EXTRA_BRIDGE EXTRA_DECODE EXTRA_CEILING EXTRA_LITERT SESSION"
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ACTION="${1:-launch}"
PARTS="${PARTS:-bridge ceiling decode litert}"
OUT="$(runs_dir "${OUT:-runs/e1-step0}")"
ASSETS="$(runs_dir "${ASSETS:-runs/assets}")"
THREADS="${THREADS:-4}"
DECODE_PROMPTS="${DECODE_PROMPTS:-100}"
MAX_NEW="${MAX_NEW:-256}"
ARMS="${ARMS:-ar mtp:2 mtp:4 mtp:6 mtp:8 lookup:10}"
FP32_PROMPTS="${FP32_PROMPTS:-50}"
CPU_PROMPTS="${CPU_PROMPTS:-20}"
CPU_MAX_NEW="${CPU_MAX_NEW:-128}"
CPU_ARMS="${CPU_ARMS:-ar mtp:2 mtp:4}"
CEILING_PROMPTS="${CEILING_PROMPTS:-20}"
CPU_CEILING_PROMPTS="${CPU_CEILING_PROMPTS:-5}"
EXTRA_BRIDGE="${EXTRA_BRIDGE:-}"
EXTRA_DECODE="${EXTRA_DECODE:-}"
EXTRA_CEILING="${EXTRA_CEILING:-}"
LITERT_PROMPTS="${LITERT_PROMPTS:-50}"
LITERT_BACKENDS="${LITERT_BACKENDS:-cpu gpu}"
EXTRA_LITERT="${EXTRA_LITERT:-}"
LITERT_PY="$ASSETS/venvs/litert/bin/python"
LITERT_MODEL="$ASSETS/litert/gemma-4-E2B-it.litertlm"
SESSION="${SESSION:-e1-step0}"
IT_MODEL="$ASSETS/models/gemma-4-E2B-it"
DRAFTER="$ASSETS/models/gemma-4-E2B-it-assistant"
PROMPTS="$ASSETS/prompts/dev.jsonl"

# On the server the first device is the GPU in bfloat16. A dry run (REQUIRE_GPU=0) has only the CPU.
if [[ "$REQUIRE_GPU" == 1 ]]; then
  MAIN_TAG="gpu-bf16"; MAIN_ARGS=(--dtype bf16)
else
  MAIN_TAG="dry-cpu"; MAIN_ARGS=(--device cpu --dtype fp32 --threads "$THREADS")
fi

# step <directory> <soft|hard> <command...>: run once; "soft" failures are logged and the script goes on.
step() {
  local dir="$1" mode="$2"; shift 2
  if [[ -f "$dir/.done" ]]; then log "done already: ${dir#"$ROOT"/}"; return 0; fi
  mkdir -p "$dir"
  log "start: ${dir#"$ROOT"/}"
  if "$@"; then
    touch "$dir/.done"
    log "finished: ${dir#"$ROOT"/}"
  elif [[ "$mode" == soft ]]; then
    log "FAILED, going on (float32 may not fit the GPU; LiteRT-LM may lack a GPU backend here): ${dir#"$ROOT"/}"
  else
    die "failed: ${dir#"$ROOT"/}"
  fi
}

bridge() {
  check_model
  local name ran=0
  for name in T16 R16; do
    local ckpt="$ROOT/runs/e0-$name/last.pt"
    if [[ ! -f "$ckpt" ]]; then log "no $ckpt: the bridge skips $name"; continue; fi
    ran=1
    local device=()
    if [[ "$REQUIRE_GPU" != 1 ]]; then device=(--device cpu --dtype fp32); fi
    # shellcheck disable=SC2086
    step "$OUT/bridge/$name" hard "$PY" -m chat_vpa.e1_bridge --model "$MODEL_DIR" --checkpoint "$ckpt" \
      --eval-dir "$DATA/eval" --out "$OUT/bridge/$name" "${device[@]}" $EXTRA_BRIDGE
  done
  [[ "$ran" == 1 ]] || log "the bridge found no E0 checkpoint under runs/e0-T16 or runs/e0-R16"
}

ceiling() {
  # shellcheck disable=SC2086
  step "$OUT/ceiling/$MAIN_TAG" hard "$PY" -m chat_vpa.e1_ceiling --model "$IT_MODEL" --prompts "$PROMPTS" \
    --limit "$CEILING_PROMPTS" --out "$OUT/ceiling/$MAIN_TAG" "${MAIN_ARGS[@]}" $EXTRA_CEILING
  if [[ "$REQUIRE_GPU" == 1 ]]; then
    # shellcheck disable=SC2086
    step "$OUT/ceiling/gpu-fp32" soft "$PY" -m chat_vpa.e1_ceiling --model "$IT_MODEL" --prompts "$PROMPTS" \
      --limit "$CEILING_PROMPTS" --dtype fp32 --out "$OUT/ceiling/gpu-fp32" $EXTRA_CEILING
    # shellcheck disable=SC2086
    step "$OUT/ceiling/cpu-fp32" hard "$PY" -m chat_vpa.e1_ceiling --model "$IT_MODEL" --prompts "$PROMPTS" \
      --limit "$CPU_CEILING_PROMPTS" --repeats 3 --warmup 1 --device cpu --dtype fp32 --threads "$THREADS" \
      --out "$OUT/ceiling/cpu-fp32" $EXTRA_CEILING
  fi
}

decode() {
  # shellcheck disable=SC2086
  step "$OUT/decode/$MAIN_TAG" hard "$PY" -m chat_vpa.e1_decode --model "$IT_MODEL" --assistant "$DRAFTER" \
    --prompts "$PROMPTS" --limit "$DECODE_PROMPTS" --max-new-tokens "$MAX_NEW" --arms $ARMS \
    --out "$OUT/decode/$MAIN_TAG" "${MAIN_ARGS[@]}" $EXTRA_DECODE
  # shellcheck disable=SC2086
  step "$OUT/decode/$MAIN_TAG/profile" hard "$PY" -m chat_vpa.e1_decode --model "$IT_MODEL" --assistant "$DRAFTER" \
    --prompts "$PROMPTS" --limit 10 --max-new-tokens "$MAX_NEW" --arms ar mtp:4 --profile \
    --out "$OUT/decode/$MAIN_TAG" "${MAIN_ARGS[@]}" $EXTRA_DECODE
  if [[ "$REQUIRE_GPU" == 1 ]]; then
    # shellcheck disable=SC2086
    step "$OUT/decode/gpu-fp32" soft "$PY" -m chat_vpa.e1_decode --model "$IT_MODEL" --assistant "$DRAFTER" \
      --prompts "$PROMPTS" --limit "$FP32_PROMPTS" --max-new-tokens "$MAX_NEW" --arms ar mtp:4 --dtype fp32 \
      --out "$OUT/decode/gpu-fp32" $EXTRA_DECODE
    # shellcheck disable=SC2086
    step "$OUT/decode/cpu-fp32" hard "$PY" -m chat_vpa.e1_decode --model "$IT_MODEL" --assistant "$DRAFTER" \
      --prompts "$PROMPTS" --limit "$CPU_PROMPTS" --max-new-tokens "$CPU_MAX_NEW" --arms $CPU_ARMS --warmup 1 \
      --device cpu --dtype fp32 --threads "$THREADS" --out "$OUT/decode/cpu-fp32" $EXTRA_DECODE
  fi
}

# LiteRT-LM runs in its own environment, with a home of its own under runs/ (a runtime may keep caches there).
litert() {
  if [[ ! -x "$LITERT_PY" || ! -s "$LITERT_MODEL" ]]; then
    log "LiteRT-LM is not set up ($LITERT_PY, $LITERT_MODEL): skipping (bash scripts/ncp/e1_setup.sh)"
    return 0
  fi
  mkdir -p "$ASSETS/home"
  local backend
  for backend in $LITERT_BACKENDS; do
    local threads=()
    if [[ "$backend" == cpu ]]; then threads=(--threads "$THREADS"); fi
    # shellcheck disable=SC2086
    step "$OUT/litert/$backend" soft env HOME="$ASSETS/home" XDG_CACHE_HOME="$ASSETS/home/.cache" "$LITERT_PY" \
      -m chat_vpa.e1_litert --model "$LITERT_MODEL" --prompts "$PROMPTS" --limit "$LITERT_PROMPTS" \
      --max-new-tokens "$MAX_NEW" --backend "$backend" "${threads[@]}" --out "$OUT/litert/$backend" $EXTRA_LITERT
  done
}

run() {
  check_python
  check_gpu_free
  [[ -f "$IT_MODEL/config.json" && -f "$DRAFTER/config.json" && -f "$PROMPTS" ]] \
    || die "the assets are missing under $ASSETS: bash scripts/ncp/e1_setup.sh assets"
  mkdir -p "$OUT"
  local part
  for part in $PARTS; do
    case "$part" in
      bridge|ceiling|decode|litert) log "== $part"; "$part" ;;
      *) die "PARTS holds '$part'; the parts are bridge, ceiling, decode, litert" ;;
    esac
  done
  REQUIRE_GPU=0 "$PY" -m chat_vpa.e1_report --out "$OUT" > /dev/null
  log "Step 0 finished: bash scripts/ncp/e1_step0.sh show"
}

case "$ACTION" in
  launch) start_tmux "$SESSION" "$OUT/step0.log" "bash scripts/ncp/e1_step0.sh run" ;;
  run)    run ;;
  show)   "$PY" -m chat_vpa.e1_report --out "$OUT" ;;
  *)      die "usage: e1_step0.sh [launch|run|show]" ;;
esac
