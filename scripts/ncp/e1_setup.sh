#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# e1_setup.sh -- what E1 needs on the server before Step 0 (docs/e1.md, Section 2): the downloads, two inference
# runtimes in environments of their own, and a preflight whose output the author pastes back.
#
#   bash scripts/ncp/e1_setup.sh             all three steps, one after another, in tmux (session e1-setup)
#   bash scripts/ncp/e1_setup.sh show        the preflight's output: what the author pastes back
#
#   bash scripts/ncp/e1_setup.sh assets      the downloads (about 14 GB; needs the network; resumable)
#   bash scripts/ncp/e1_setup.sh runtimes    vLLM, LiteRT-LM and the IFEval checker, each in its own environment
#   bash scripts/ncp/e1_setup.sh preflight   the checks (needs the GPU free)
#   bash scripts/ncp/e1_setup.sh run         the three in the foreground
#
# A step that fails does not stop the next one (a runtime that will not install must not hide the preflight of
# the rest); the preflight says what is missing, and every step can be run again.
#
# Everything lands under runs/assets/ (never delete it with a run's outputs):
#   models/gemma-4-E2B-it              google/gemma-4-E2B-it, the instruction-tuned target
#   models/gemma-4-E2B-it-assistant    google/gemma-4-E2B-it-assistant, Google's Multi-Token Prediction drafter
#   litert/gemma-4-E2B-it.litertlm     litert-community/gemma-4-E2B-it-litert-lm (LiteRT-LM's build, drafter inside)
#   data/ultrachat_200k                HuggingFaceH4/ultrachat_200k, the test_sft split only (the development prompts)
#   ifeval/                            google-research's instruction_following_eval: the checker and its 541 prompts
#   prompts/dev.jsonl                  the development prompts (chat_vpa/e1_prompts.py)
#   venvs/{vllm,litert,ifeval}         the three environments
#
# This script is the one place where the server's rules are relaxed, by the author's decision of 25 November 2026
# (docs/e1.md, Section 7): it uses the network, it creates environments besides text-VPA's .venv (inside runs/,
# so that nothing of another torch reaches the training environment), and it adds pyarrow to text-VPA's .venv if
# that is missing (it reads UltraChat's parquet files; it pins nothing). It still writes nowhere outside runs/
# (pip without its cache, the Hub's cache under runs/assets) and never shuts the server down.
#
# Variables: ASSETS [runs/assets]; HF_TOKEN, if the Hub asks for one; VLLM_SPEC ["vllm>=0.21"];
# LITERT_SPEC [litert-lm]; DEV_PROMPTS [200]; SESSION [e1-setup].
set -euo pipefail
PASS_VARS="ASSETS VLLM_SPEC LITERT_SPEC DEV_PROMPTS SESSION"
# shellcheck source=scripts/ncp/env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ACTION="${1:-launch}"
SESSION="${SESSION:-e1-setup}"
ASSETS="$(runs_dir "${ASSETS:-runs/assets}")"
VLLM_SPEC="${VLLM_SPEC:-vllm>=0.21}"
LITERT_SPEC="${LITERT_SPEC:-litert-lm}"
DEV_PROMPTS="${DEV_PROMPTS:-200}"
IT_MODEL="$ASSETS/models/gemma-4-E2B-it"
DRAFTER="$ASSETS/models/gemma-4-E2B-it-assistant"
PROMPTS="$ASSETS/prompts/dev.jsonl"
LOG="$ASSETS/preflight.log"

online() {
  # Downloads only: the Hub's caches stay under runs/assets, pip keeps no cache.
  export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_CACHE="$ASSETS/hub-cache" HF_XET_CACHE="$ASSETS/hub-cache/xet"
  export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
}

assets() {
  online
  check_python_only
  mkdir -p "$ASSETS/models" "$ASSETS/data" "$ASSETS/litert" "$ASSETS/prompts"
  log "free space under runs/: $(df -h --output=avail "$ROOT/runs" | tail -n 1 | tr -d ' ') (the downloads need about 14 GB, the runtimes about 12 GB more)"
  # Each download stands alone: one that fails is named at the end and does not keep the others from being used.
  local failed=""
  "$PY" - "$ASSETS" <<'PYEOF' || failed+=" hub"
import os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

assets = Path(sys.argv[1])
token = os.environ.get("HF_TOKEN") or None
failures = []


def fetch(repo_ids, target, filename=None, **kwargs):
    """A repository (or one file of it) into ``target``; a ``.complete`` marker says that the download finished."""
    target = Path(target)
    if (target / ".complete").exists():
        print(f"  {target}: complete, skipping", flush=True)
        return
    last = None
    for repo_id in repo_ids:                       # the Hub's spelling of the id, then lower case
        try:
            if filename is None:
                snapshot_download(repo_id, local_dir=target, token=token, **kwargs)
            else:
                hf_hub_download(repo_id, filename, local_dir=target, token=token, **kwargs)
        except Exception as error:                 # noqa: BLE001 - the next spelling is tried, the last error is shown
            last = error
            continue
        (target / ".complete").write_text(repo_id + "\n")
        print(f"  {target}: downloaded {repo_id}", flush=True)
        return
    failures.append(repo_ids[0])
    print(f"  FAILED {repo_ids[0]} into {target}: {str(last)[:300]}", flush=True)


fetch(["google/gemma-4-E2B-it", "google/gemma-4-e2b-it"], assets / "models" / "gemma-4-E2B-it")
fetch(["google/gemma-4-E2B-it-assistant", "google/gemma-4-e2b-it-assistant"], assets / "models" / "gemma-4-E2B-it-assistant")
fetch(["HuggingFaceH4/ultrachat_200k"], assets / "data" / "ultrachat_200k", repo_type="dataset",
      allow_patterns=["data/test_sft-*.parquet", "README.md"])
fetch(["litert-community/gemma-4-E2B-it-litert-lm"], assets / "litert", filename="gemma-4-E2B-it.litertlm")
if failures:
    sys.exit("not downloaded: " + ", ".join(failures) + "\n(if the Hub asks for a login or a licence, accept it "
             "on the repository's page and set HF_TOKEN; then run this step again: what is complete is skipped)")
PYEOF
  if [[ ! -d "$ASSETS/ifeval/instruction_following_eval" ]]; then
    log "IFEval: a sparse checkout of google-research (the checker and its prompts, a few MB)"
    rm -rf "$ASSETS/ifeval"
    git clone -q --depth 1 --filter=blob:none --sparse https://github.com/google-research/google-research.git "$ASSETS/ifeval" \
      && git -C "$ASSETS/ifeval" sparse-checkout set instruction_following_eval || failed+=" ifeval"
  fi
  if [[ -f "$ASSETS/ifeval/instruction_following_eval/data/input_data.jsonl" ]]; then
    [[ "$(wc -l < "$ASSETS/ifeval/instruction_following_eval/data/input_data.jsonl")" == 541 ]] \
      || log "note: IFEval's prompt file does not have 541 lines (the checkout is newer than the one this was written against)"
  fi
  if compgen -G "$ASSETS/data/ultrachat_200k/data/test_sft-*.parquet" > /dev/null; then
    if ! "$PY" -c "import pyarrow" 2>/dev/null; then
      log "adding pyarrow to $("$PY" -c 'import sys; print(sys.prefix)') (it reads UltraChat's parquet files; it pins nothing)"
      "$PY" -m pip install pyarrow || failed+=" pyarrow"
    fi
    "$PY" -m chat_vpa.e1_prompts --parquet "$ASSETS"/data/ultrachat_200k/data/test_sft-*.parquet \
      --count "$DEV_PROMPTS" --out "$PROMPTS" || failed+=" prompts"
  else
    failed+=" prompts(no UltraChat files)"
  fi
  log "assets under $ASSETS: $(du -sh "$ASSETS" | cut -f1)"
  [[ -z "$failed" ]] || die "incomplete:$failed (the lines above say why; run 'bash scripts/ncp/e1_setup.sh assets' again)"
  log "assets ready"
}

# make_env <name> <packages...>: an environment of its own under runs/assets/venvs.
make_env() {
  local name="$1"; shift
  local dir="$ASSETS/venvs/$name"
  if [[ ! -x "$dir/bin/python" ]]; then
    "$PY" -m venv "$dir" 2>/dev/null || { command -v uv >/dev/null && uv venv -q --python "$PY" --seed "$dir"; } \
      || die "could not create $dir (neither 'python -m venv' nor uv worked)"
  fi
  "$dir/bin/python" -m pip install -q --upgrade pip && "$dir/bin/python" -m pip install "$@"
}

runtimes() {
  online
  check_python_only
  mkdir -p "$ASSETS/venvs"
  local failed=""
  log "vLLM ($VLLM_SPEC): its own torch, several GB, a few minutes"
  ( make_env vllm "$VLLM_SPEC" ) || failed+=" vllm"
  log "LiteRT-LM ($LITERT_SPEC)"
  ( make_env litert "$LITERT_SPEC" ) || failed+=" litert"
  log "the IFEval checker's packages"
  if [[ -n "${https_proxy:-}${HTTPS_PROXY:-}" ]]; then export NLTK_ALLOW_PROXIED_URLOPEN=1; fi   # nltk refuses proxies otherwise
  # "&&" throughout: inside "( ... ) ||" bash ignores set -e, so each step must hand its failure on itself.
  ( make_env ifeval absl-py langdetect nltk immutabledict \
    && NLTK_DATA="$ASSETS/nltk_data" "$ASSETS/venvs/ifeval/bin/python" -c "
import nltk, sys
ok = [nltk.download(name, download_dir='$ASSETS/nltk_data', quiet=True) for name in ('punkt', 'punkt_tab')]
sys.exit(0 if all(ok) else 'nltk could not download its tokenizer data')" ) || failed+=" ifeval"
  log "environments under $ASSETS/venvs: $(du -sh "$ASSETS/venvs" | cut -f1)"
  [[ -z "$failed" ]] || die "could not set up:$failed (the lines above say why; the others are in place)"
}

# ifeval_selftest <python>: the official checker on the responses it ships. One of the 541 prompts was edited after
# those responses were made, so the test runs on the 540 they cover. The checker draws from Python's random module
# and langdetect is randomised: both are seeded, or two runs differ.
ifeval_selftest() {
  local python="$1" out="$ASSETS/ifeval-selftest" data="$ASSETS/ifeval/instruction_following_eval/data"
  mkdir -p "$out"
  "$python" - "$data" "$out/input_covered.jsonl" <<'PYEOF'
import json, sys
data, out = sys.argv[1], sys.argv[2]
have = {json.loads(line)["prompt"] for line in open(f"{data}/input_response_data_gpt4_20231107_145030.jsonl")}
rows = [line for line in open(f"{data}/input_data.jsonl") if json.loads(line)["prompt"] in have]
open(out, "w").writelines(rows)
print(f"  {len(rows)} of 541 prompts have a shipped response")
PYEOF
  ( cd "$ASSETS/ifeval" && NLTK_DATA="$ASSETS/nltk_data" "$python" -c "
import random
import langdetect
random.seed(0)
langdetect.DetectorFactory.seed = 0
from absl import app
from instruction_following_eval import evaluation_main
app.run(evaluation_main.main)" --input_data="$out/input_covered.jsonl" \
      --input_response_data="$data/input_response_data_gpt4_20231107_145030.jsonl" --output_dir="$out" 2>&1 \
      | grep -E "^prompt-level|^instruction-level|Accuracy Scores" | sed 's/^/  /' )
}

# apart <command...>: a runtime of another environment, with a home of its own under runs/ (so that whatever it
# caches or records stays inside this repository's runs/) and without usage reports.
apart() {
  mkdir -p "$ASSETS/home"
  HOME="$ASSETS/home" XDG_CACHE_HOME="$ASSETS/home/.cache" XDG_CONFIG_HOME="$ASSETS/home/.config" \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 "$@"
}

# check_python_only: the interpreter and its packages, without asking for the GPU.
check_python_only() { REQUIRE_GPU=0 check_python > /dev/null; }

section() { echo; echo "== $*"; }

preflight() {
  check_python
  check_gpu_free
  mkdir -p "$ASSETS"
  {
    set +e                                   # a diagnostic: every section prints, whatever the one before did
    echo "E1 preflight, $(date '+%F %T %Z'), $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo 'no git')"
    section "machine"
    echo "  $(nproc) CPUs: $(lscpu | sed -n 's/^Model name: *//p' | head -n 1)"
    echo "  memory: $(free -g | awk '/^Mem:/ {print $2 " GiB total, " $7 " GiB available"}')"
    echo "  disk under runs/: $(df -h --output=avail "$ROOT/runs" | tail -n 1 | tr -d ' ') free"
    if [[ "$REQUIRE_GPU" == 1 ]]; then
      echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n 1)"
    fi
    echo "  Vulkan drivers (LiteRT-LM's GPU backend needs one): $(ls /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json 2>/dev/null | tr '\n' ' ')"
    section "assets"
    for path in "$IT_MODEL" "$DRAFTER" "$ASSETS/litert" "$ASSETS/data/ultrachat_200k" "$ASSETS/ifeval" "$PROMPTS"; do
      if [[ -e "$path" ]]; then echo "  $(du -sh "$path" | cut -f1)  ${path#"$ROOT"/}"; else echo "  MISSING  ${path#"$ROOT"/}"; fi
    done
    section "the target and Google's drafter in Transformers"
    local device=()
    if [[ "$REQUIRE_GPU" != 1 ]]; then device=(--device cpu --dtype fp32); fi
    "$PY" -m chat_vpa.e1_preflight --model "$IT_MODEL" --assistant "$DRAFTER" --prompts "$PROMPTS" "${device[@]}" \
      || echo "UNEXPECTED: the Transformers preflight failed (see above)"
    section "vLLM"
    local vllm="$ASSETS/venvs/vllm/bin"
    if [[ -x "$vllm/python" ]]; then
      apart "$vllm/python" -c "import torch, vllm; print('  vllm', vllm.__version__, '/ torch', torch.__version__, '/ CUDA', torch.cuda.is_available())" 2>&1 | tail -n 1
      apart "$vllm/vllm" serve --help=all 2>/dev/null | grep -i -A 6 -- "--speculative-config" | head -n 14 \
        || apart "$vllm/vllm" serve --help 2>/dev/null | grep -i -A 6 "speculative" | head -n 14 || echo "  no --speculative-config in the help"
      echo "-- a smoke decode with Google's drafter (mtp:4, the first prompt, 32 tokens): a check, not a result"
      if [[ "$REQUIRE_GPU" != 1 ]]; then
        echo "  skipped: no GPU in a dry run"
      elif ! apart timeout 1200 env -u PYTORCH_CUDA_ALLOC_CONF VLLM_LOGGING_LEVEL=WARNING "$vllm/python" -m chat_vpa.e1_vllm smoke \
          --model "$IT_MODEL" --assistant "$DRAFTER" --prompts "$PROMPTS" 2> "$ASSETS/vllm-smoke.log"; then
        echo "UNEXPECTED: the vLLM smoke decode failed; the last lines of runs/assets/vllm-smoke.log:"
        tail -n 25 "$ASSETS/vllm-smoke.log" | sed 's/^/    /'
      fi
    else
      echo "  not installed (bash scripts/ncp/e1_setup.sh runtimes)"
    fi
    section "LiteRT-LM"
    local litert="$ASSETS/venvs/litert/bin"
    if [[ -x "$litert/python" ]]; then
      apart "$litert/python" - "$ASSETS/litert/gemma-4-E2B-it.litertlm" <<'PYEOF'
import importlib.metadata, inspect, os, sys
print("  litert-lm", importlib.metadata.version("litert-lm"), "(chat_vpa/e1_litert.py was written against 0.20.0)")
import litert_lm
names = ("Engine", "SpeculativeDecodingConfig", "SpeculativeDecodingMode", "SamplerConfig", "ThinkingConfig", "ModelInfo")
missing = [name for name in names if not hasattr(litert_lm, name)]
print("  UNEXPECTED: litert_lm lacks " + ", ".join(missing) if missing else "  the interface e1_litert.py uses is there")
print("  speculative decoding modes:", [m for m in dir(litert_lm.SpeculativeDecodingMode) if m.isupper()])
print("  SpeculativeDecodingConfig" + str(inspect.signature(litert_lm.SpeculativeDecodingConfig)))
path = sys.argv[1]
if os.path.isfile(path) and os.path.getsize(path) > 0:
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
    info = litert_lm.ModelInfo(path)
    llm = info.llm
    support = llm.has_speculative_decoding_support()
    print(f"  {os.path.basename(path)}: {os.path.getsize(path) / 2**30:.2f} GiB, context {info.max_context_tokens}, "
          f"drafter inside: {support}, thinking: {llm.supports_thinking()}, default sampler: {llm.default_sampler_params}")
    if not support:
        print("  UNEXPECTED: this model file does not support speculative decoding")
    info.close()
else:
    print("  UNEXPECTED: no model file at", path)
PYEOF
    else
      echo "  not installed (bash scripts/ncp/e1_setup.sh runtimes)"
    fi
    section "the IFEval checker on the example responses it ships (GPT-4, November 2023): a self-test"
    local ifeval="$ASSETS/venvs/ifeval/bin/python"
    if [[ -x "$ifeval" && -d "$ASSETS/ifeval/instruction_following_eval" ]]; then
      ifeval_selftest "$ifeval" || echo "UNEXPECTED: the IFEval checker did not run"
      echo "  expected with these seeds: strict 0.7722 / 0.8377, loose 0.7981 / 0.8570 (prompt-level / instruction-level;"
      echo "  measured in the development sandbox on google-research 20fadc5; the paper reports 76.89 / 83.57 and"
      echo "  79.30 / 85.37 for GPT-4 on all 541). Without the seeds the scores move by a prompt or two per run."
    else
      echo "  not installed (bash scripts/ncp/e1_setup.sh assets; bash scripts/ncp/e1_setup.sh runtimes)"
    fi
    echo
    echo "== end of the E1 preflight"
  } 2>&1 | tee "$LOG"
  local unexpected
  unexpected="$(grep -c "UNEXPECTED" "$LOG" || true)"
  echo "$unexpected line(s) with UNEXPECTED; paste the whole output of: bash scripts/ncp/e1_setup.sh show"
}

# everything: the three steps; a failure is reported at the end and does not stop the next step.
everything() {
  local failed="" name
  for name in assets runtimes preflight; do
    log "== $name"
    # A process of its own: inside "( ... ) ||" bash would ignore set -e and a step would run on after an error.
    ASSETS="$ASSETS" VLLM_SPEC="$VLLM_SPEC" LITERT_SPEC="$LITERT_SPEC" DEV_PROMPTS="$DEV_PROMPTS" TEXT_VPA="$TEXT_VPA" \
      PY="$PY" DATA="$DATA" MODEL_DIR="$MODEL_DIR" bash "$ROOT/scripts/ncp/e1_setup.sh" "$name" \
      || { failed+=" $name"; log "FAILED: $name (going on)"; }
  done
  [[ -z "$failed" ]] || die "failed:$failed (run that step again: bash scripts/ncp/e1_setup.sh <step>)"
  log "setup finished: bash scripts/ncp/e1_setup.sh show"
}

case "$ACTION" in
  launch)    start_tmux "$SESSION" "$ASSETS/setup.log" "bash scripts/ncp/e1_setup.sh run" ;;
  run)       everything ;;
  assets)    assets ;;
  runtimes)  runtimes ;;
  preflight) preflight ;;
  show)      [[ -f "$LOG" ]] && cat "$LOG" || die "no preflight yet: bash scripts/ncp/e1_setup.sh preflight" ;;
  *)         die "usage: e1_setup.sh [launch|run|assets|runtimes|preflight|show]" ;;
esac
