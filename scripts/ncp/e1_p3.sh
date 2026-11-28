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
#   bash scripts/ncp/e1_p3.sh route      route (a) (docs/e1.md, Section 11.13; chat_vpa/e1_p3_route.py): relate (the
#                                        LiteRT drafter's weights against A1's; the proposed mapping), codec (the
#                                        codec's null test), the null patch (the file's own weights written back must
#                                        give the file; the copy is then deleted), and A1's and A2's weights written
#                                        into copies of Google's file (OUT/a1, OUT/a2; 2.4 GB each). CPU and disk;
#                                        minutes; it stops where a step fails, and what failed is in its report
#   bash scripts/ncp/e1_p3.sh engine     LiteRT-LM on Google's file and on the two copies (chat_vpa/e1_litert.py,
#                                        modes off and on, ENGINE_BACKENDS ["cpu"], ENGINE_PROMPTS [50] x 256 tokens):
#                                        exactness against off and the speed of on. Its speeds need a machine with
#                                        nothing else running: it refuses to start beside a GPU job
#   bash scripts/ncp/e1_p3.sh show-route   the route's reports        bash scripts/ncp/e1_p3.sh show-engine  the engine's
#
# Both run in LiteRT-LM's environment (runs/assets/venvs/litert), whose litert-lm 0.20.0 brings litert-lm-builder
# 0.20.0 with it: nothing is installed. Variables (defaults in brackets): OUT [runs/e1-p3]   ASSETS [runs/assets]
# A2 [runs/e1-step1/a2/final]   ENGINE_BACKENDS ["cpu"]   ENGINE_PROMPTS [50]   THREADS [4]   MAX_NEW [256]
# SHARED [mean]: a buffer that the file shares between weights whose values differ in the drafter (A2 trained them
# apart) gets their mean, and the report says by how much they differed; "refuse" stops there instead.
# ROUTE_ROLES: a JSON file of matrix names for relate if the file's are not Google's (chat_vpa/e1_p3_route.py)
# A part that has finished is skipped (its .done marker). Never shuts the server down.
set -euo pipefail
PASS_VARS="OUT ASSETS A2 ENGINE_BACKENDS ENGINE_PROMPTS THREADS MAX_NEW SHARED ROUTE_ROLES"
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ACTION="${1:-all}"
OUT="$(runs_dir "${OUT:-runs/e1-p3}")"
ASSETS="$(runs_dir "${ASSETS:-runs/assets}")"
LITERT_PY="$ASSETS/venvs/litert/bin/python"
LITERT_MODEL="$ASSETS/litert/gemma-4-E2B-it.litertlm"
DRAFTER="$ASSETS/models/gemma-4-E2B-it-assistant"
A2="$(runs_dir "${A2:-runs/e1-step1/a2/final}")"
ENGINE_BACKENDS="${ENGINE_BACKENDS:-cpu}"
ENGINE_PROMPTS="${ENGINE_PROMPTS:-50}"
THREADS="${THREADS:-4}"
MAX_NEW="${MAX_NEW:-256}"
SHARED="${SHARED:-mean}"
PROMPTS="$ASSETS/prompts/dev.jsonl"
NAME="$(basename "$LITERT_MODEL")"

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

# route_step <name> <command...>: run once in LiteRT-LM's environment, its report in OUT/route/<name>.md.
route_step() {
  local name="$1"; shift
  if [[ -f "$OUT/route/$name.done" ]]; then log "done already: route $name"; return 0; fi
  mkdir -p "$OUT/route"
  log "start: route $name"
  if foreign "$LITERT_PY" -m chat_vpa.e1_p3_route "$@" > "$OUT/route/$name.stdout.md" 2> "$OUT/route/$name.log"; then
    touch "$OUT/route/$name.done"
    log "finished: route $name"
  else
    tail -n 5 "$OUT/route/$name.log" >&2
    die "failed: route $name ($OUT/route/$name.log)"
  fi
}

route() {
  ready
  [[ -f "$DRAFTER/config.json" ]] || die "A1 is missing: $DRAFTER"
  local roles=()
  if [[ -n "${ROUTE_ROLES:-}" ]]; then roles=(--roles "$ROUTE_ROLES"); fi
  route_step relate relate --model "$LITERT_MODEL" --assistant "$DRAFTER" "${roles[@]}" --out "$OUT/route/relate"
  route_step codec codec --model "$LITERT_MODEL" --out "$OUT/route/codec"
  route_step null patch --model "$LITERT_MODEL" --null --out "$OUT/route/null/$NAME"
  rm -f "$OUT/route/null/$NAME"                       # it was the original byte for byte: nothing to keep
  route_step a1 patch --model "$LITERT_MODEL" --weights "$DRAFTER" --mapping "$OUT/route/relate/mapping.json" \
    --shared "$SHARED" --out "$OUT/a1/$NAME"
  if [[ -f "$A2/config.json" ]]; then
    route_step a2 patch --model "$LITERT_MODEL" --weights "$A2" --mapping "$OUT/route/relate/mapping.json" \
      --shared "$SHARED" --out "$OUT/a2/$NAME"
  else
    log "A2 is not trained ($A2): only A1's copy is written"
  fi
  log "route (a) finished: bash scripts/ncp/e1_p3.sh show-route"
}

engine() {
  ready
  check_gpu_free                                       # the speeds need an otherwise idle machine
  local file tag backend threads vulkan icd
  for tag in google a1 a2; do
    case "$tag" in
      google) file="$LITERT_MODEL" ;;
      *)      file="$OUT/$tag/$NAME" ;;
    esac
    if [[ ! -s "$file" ]]; then log "no file for $tag ($file): skipped"; continue; fi
    for backend in $ENGINE_BACKENDS; do
      threads=(); vulkan=()
      if [[ "$backend" == cpu ]]; then threads=(--threads "$THREADS"); fi
      if [[ "$backend" == gpu ]]; then
        for icd in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
          if [[ -f "$icd" ]]; then vulkan=(VK_DRIVER_FILES="$icd" VK_ICD_FILENAMES="$icd"); break; fi
        done
      fi
      local dir="$OUT/engine/$tag-$backend"
      if [[ -f "$dir/.done" ]]; then log "done already: engine $tag-$backend"; continue; fi
      mkdir -p "$dir"
      log "start: engine $tag-$backend"
      env -u PYTORCH_CUDA_ALLOC_CONF HOME="$ASSETS/home" XDG_CACHE_HOME="$ASSETS/home/.cache" \
        XDG_CONFIG_HOME="$ASSETS/home/.config" DO_NOT_TRACK=1 "${vulkan[@]}" "$LITERT_PY" -m chat_vpa.e1_litert \
        --model "$file" --prompts "$PROMPTS" --limit "$ENGINE_PROMPTS" --max-new-tokens "$MAX_NEW" --modes off on \
        --backend "$backend" "${threads[@]}" --out "$dir" > "$dir/stdout.md" 2> "$dir/log" \
        || { log "FAILED, going on: engine $tag-$backend ($dir/log)"; continue; }
      touch "$dir/.done"
      log "finished: engine $tag-$backend"
    done
  done
  log "engine finished: bash scripts/ncp/e1_p3.sh show-engine"
}

show_route() {
  local name
  for name in relate codec; do
    if [[ -f "$OUT/route/$name/$name.md" ]]; then cat "$OUT/route/$name/$name.md"; echo; else echo "$name: not run"; fi
  done
  for name in null a1 a2; do
    local report
    case "$name" in null) report="$OUT/route/null/patch.md" ;; *) report="$OUT/$name/patch.md" ;; esac
    if [[ -f "$report" ]]; then cat "$report"; echo; else echo "patch $name: not run"; fi
  done
}

show_engine() {
  local dir
  for dir in "$OUT"/engine/*/; do
    [[ -d "$dir" ]] || { echo "engine: not run"; return 0; }
    echo "== ${dir%/}"
    if [[ -f "$dir/stdout.md" ]]; then cat "$dir/stdout.md"; else echo "(no output)"; fi
    echo
  done
}

case "$ACTION" in
  route)       route ;;
  engine)      engine ;;
  show-route)  show_route ;;
  show-engine) show_engine ;;
  all)     ready; part inspect --model "$LITERT_MODEL" --assistant "$DRAFTER"; part repack --model "$LITERT_MODEL"
           log "P3 finished: bash scripts/ncp/e1_p3.sh show" ;;
  inspect) ready; part inspect --model "$LITERT_MODEL" --assistant "$DRAFTER" ;;
  repack)  ready; part repack --model "$LITERT_MODEL" ;;
  show)    show ;;
  *)       die "usage: e1_p3.sh [all|inspect|repack|show|route|engine|show-route|show-engine]" ;;
esac
