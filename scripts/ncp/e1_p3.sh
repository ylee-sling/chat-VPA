#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# e1_p3.sh -- P3 of E1's Step 1 (docs/e1.md, Section 11.7): what LiteRT-LM's model file holds, and the null test of
# repacking it with the builder that ships with LiteRT-LM. CPU and disk only: no GPU, so it may run while a GPU job
# runs (it runs niced); minutes.
#
#   bash scripts/ncp/e1_p3.sh            inspect, then repack, in the foreground
#   bash scripts/ncp/e1_p3.sh inspect    the header, every section, every TFLite graph's signatures and weights, the
#                                        drafter's tensors (extracted to OUT/inspect/drafter.tflite) and which of
#                                        A1's weights each holds (chat_vpa/e1_p3.py inspect)
#   bash scripts/ncp/e1_p3.sh repack     the builder's unpack and pack, every section compared byte for byte
#                                        (chat_vpa/e1_p3.py repack); about 5 GB of disk while it runs, 2.4 GB after
#                                        (the repacked file, kept for the engine test of 11.7)
#   bash scripts/ncp/e1_p3.sh show       both reports; what the author pastes back
#
# Both run in LiteRT-LM's environment (runs/assets/venvs/litert), whose litert-lm 0.20.0 brings litert-lm-builder
# 0.20.0 with it: nothing is installed. Variables (defaults in brackets): OUT [runs/e1-p3]   ASSETS [runs/assets]
# A part that has finished is skipped (its .done marker). Never shuts the server down.
set -euo pipefail
PASS_VARS="OUT ASSETS"
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ACTION="${1:-all}"
OUT="$(runs_dir "${OUT:-runs/e1-p3}")"
ASSETS="$(runs_dir "${ASSETS:-runs/assets}")"
LITERT_PY="$ASSETS/venvs/litert/bin/python"
LITERT_MODEL="$ASSETS/litert/gemma-4-E2B-it.litertlm"
DRAFTER="$ASSETS/models/gemma-4-E2B-it-assistant"

# A runtime of another environment, with a home of its own under runs/assets, at a lower priority than a GPU job.
foreign() {
  mkdir -p "$ASSETS/home"
  local quiet=(nice -n 10)
  if command -v ionice > /dev/null; then quiet+=(ionice -c 3); fi
  env -u PYTORCH_CUDA_ALLOC_CONF HOME="$ASSETS/home" XDG_CACHE_HOME="$ASSETS/home/.cache" \
    XDG_CONFIG_HOME="$ASSETS/home/.config" DO_NOT_TRACK=1 "${quiet[@]}" "$@"
}

ready() {
  [[ -x "$LITERT_PY" && -s "$LITERT_MODEL" ]] \
    || die "LiteRT-LM is not set up ($LITERT_PY, $LITERT_MODEL): bash scripts/ncp/e1_setup.sh"
  foreign "$LITERT_PY" -I -c "import litert_lm_builder, importlib.metadata as m; print('litert-lm-builder', m.version('litert-lm-builder'))" \
    || die "litert-lm-builder is missing from LiteRT-LM's environment"
}

part() {
  local name="$1"; shift
  if [[ -f "$OUT/$name/.done" ]]; then log "done already: $name"; return 0; fi
  mkdir -p "$OUT/$name"
  log "start: $name"
  foreign "$LITERT_PY" -m chat_vpa.e1_p3 "$name" "$@" --out "$OUT/$name" > "$OUT/$name/stdout.md" \
    || die "failed: $name ($OUT/$name/stdout.md)"
  touch "$OUT/$name/.done"
  log "finished: $name"
}

show() {
  local name
  for name in inspect repack; do
    if [[ -f "$OUT/$name/$name.md" ]]; then cat "$OUT/$name/$name.md"; echo; else echo "$name: not run"; fi
  done
}

case "$ACTION" in
  all)     ready; part inspect --model "$LITERT_MODEL" --assistant "$DRAFTER"; part repack --model "$LITERT_MODEL"
           log "P3 finished: bash scripts/ncp/e1_p3.sh show" ;;
  inspect) ready; part inspect --model "$LITERT_MODEL" --assistant "$DRAFTER" ;;
  repack)  ready; part repack --model "$LITERT_MODEL" ;;
  show)    show ;;
  *)       die "usage: e1_p3.sh [all|inspect|repack|show]" ;;
esac
