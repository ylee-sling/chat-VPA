#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# e1_p2.sh -- P2 of E1's Step 1 (docs/e1.md, Section 11.7): LiteRT-LM's GPU backend at lenience 0, run again with
# one engine and one process per run, to tell whether its speculation's changed texts (Step 0: 49 of 50) are a
# precision effect or a defect. About ten minutes of the GPU.
#
#   bash scripts/ncp/e1_p2.sh          in tmux (session e1-p2): off-a, on-a, off-b, on-b, then the reading
#   bash scripts/ncp/e1_p2.sh run      the same in the foreground
#   bash scripts/ncp/e1_p2.sh show     the reading (OUT/p2.md); no GPU
#
# Each run is one LiteRT-LM engine of one mode in its own process (chat_vpa/e1_p2.py run, LiteRT-LM's own
# environment), on the first P2_PROMPTS development prompts, with the GPU backend shown only NVIDIA's Vulkan
# driver. The reading (chat_vpa/e1_p2.py read, text-VPA's .venv) compares the texts and scores the first token at
# which two responses part with the released checkpoint in float32 on the GPU, after the LiteRT-LM processes have
# ended; it reads Step 0's GPU row as well.
#
# Variables (defaults in brackets):
#   OUT [runs/e1-p2]   ASSETS [runs/assets]   P2_PROMPTS [10]   MAX_NEW [256]   TAGS ["off-a on-a off-b on-b"]
#   STEP0 [runs/e1-step0/litert/gpu/samples.jsonl]   BACKEND [gpu]   EXTRA_RUN [], EXTRA_READ [] (dry runs)
# A run that has finished is skipped (its .done marker). Never shuts the server down. Do not git pull while it runs.
set -euo pipefail
PASS_VARS="OUT ASSETS P2_PROMPTS MAX_NEW TAGS STEP0 BACKEND EXTRA_RUN EXTRA_READ SESSION"
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ACTION="${1:-launch}"
OUT="$(runs_dir "${OUT:-runs/e1-p2}")"
ASSETS="$(runs_dir "${ASSETS:-runs/assets}")"
P2_PROMPTS="${P2_PROMPTS:-10}"
MAX_NEW="${MAX_NEW:-256}"
TAGS="${TAGS:-off-a on-a off-b on-b}"
STEP0="${STEP0:-$ROOT/runs/e1-step0/litert/gpu/samples.jsonl}"
BACKEND="${BACKEND:-gpu}"
EXTRA_RUN="${EXTRA_RUN:-}"
EXTRA_READ="${EXTRA_READ:-}"
SESSION="${SESSION:-e1-p2}"
LITERT_PY="$ASSETS/venvs/litert/bin/python"
LITERT_MODEL="$ASSETS/litert/gemma-4-E2B-it.litertlm"
IT_MODEL="$ASSETS/models/gemma-4-E2B-it"
PROMPTS="$ASSETS/prompts/dev.jsonl"

# A runtime of another environment, with a home of its own under runs/ (as in e1_step0.sh).
foreign() {
  mkdir -p "$ASSETS/home"
  env -u PYTORCH_CUDA_ALLOC_CONF HOME="$ASSETS/home" XDG_CACHE_HOME="$ASSETS/home/.cache" \
    XDG_CONFIG_HOME="$ASSETS/home/.config" DO_NOT_TRACK=1 "$@"
}

run() {
  check_python
  check_gpu_free
  [[ -x "$LITERT_PY" && -s "$LITERT_MODEL" ]] \
    || die "LiteRT-LM is not set up ($LITERT_PY, $LITERT_MODEL): bash scripts/ncp/e1_setup.sh"
  [[ -f "$IT_MODEL/config.json" && -f "$PROMPTS" ]] || die "the assets are missing under $ASSETS"
  mkdir -p "$OUT"
  local vulkan=() icd tag device=()
  if [[ "$BACKEND" == gpu ]]; then
    # The GPU backend is WebGPU over Vulkan; shown only NVIDIA's driver, it cannot fall back to Mesa's software
    # renderer (lvp) and call it the GPU.
    for icd in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
      if [[ -f "$icd" ]]; then vulkan=(VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd"); break; fi
    done
    [[ ${#vulkan[@]} -gt 0 ]] || die "no NVIDIA Vulkan driver file found: P2 needs the GPU backend on the L4"
    log "LiteRT-LM's GPU backend sees one Vulkan driver: $icd"
  fi
  for tag in $TAGS; do
    if [[ -f "$OUT/$tag/.done" ]]; then log "done already: $tag"; continue; fi
    log "start: $tag (one engine of mode ${tag%%-*}, its own process)"
    # shellcheck disable=SC2086
    foreign env "${vulkan[@]}" "$LITERT_PY" -m chat_vpa.e1_p2 run --mode "${tag%%-*}" --tag "$tag" \
      --model "$LITERT_MODEL" --prompts "$PROMPTS" --limit "$P2_PROMPTS" --max-new-tokens "$MAX_NEW" \
      --backend "$BACKEND" --out "$OUT" $EXTRA_RUN || die "failed: $tag"
    touch "$OUT/$tag/.done"
    log "finished: $tag"
  done
  if [[ "$REQUIRE_GPU" != 1 ]]; then device=(--device cpu); fi
  local step0=()
  if [[ -f "$STEP0" ]]; then step0=(--step0 "$STEP0"); else log "no $STEP0: Step 0's GPU row is not read"; fi
  # shellcheck disable=SC2086
  "$PY" -m chat_vpa.e1_p2 read --out "$OUT" --tags $TAGS --model "$IT_MODEL" --prompts "$PROMPTS" \
    "${step0[@]}" "${device[@]}" $EXTRA_READ > /dev/null
  log "P2 finished: bash scripts/ncp/e1_p2.sh show"
}

case "$ACTION" in
  launch) start_tmux "$SESSION" "$OUT/p2.log" "bash scripts/ncp/e1_p2.sh run" ;;
  run)    run ;;
  show)   cat "$OUT/p2.md" ;;
  *)      die "usage: e1_p2.sh [launch|run|show]" ;;
esac
