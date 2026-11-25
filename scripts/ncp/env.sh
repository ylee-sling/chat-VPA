# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# env.sh -- sourced by every script in scripts/ncp/: the rules of the NCP server (vpa-gpu-l4-01, one NVIDIA L4)
# in one place. From text-VPA's scripts/ncp/stage0.sh (its environment and GPU checks).
#
#   * No script shuts the server down, and none reads SHUTDOWN_ON_SUCCESS / SHUTDOWN_ON_FAILURE (monthly plan).
#   * One interpreter: text-VPA's .venv (Python 3.12). Nothing is installed by these scripts; a missing package
#     stops the script with its name. The one exception, by the author's decision of 25 November 2026
#     (docs/e1.md, Section 7), is e1_setup.sh: it downloads E1's assets and creates the environments of two other
#     runtimes (vLLM, LiteRT-LM) and of the IFEval checker under runs/assets/, and adds pyarrow to text-VPA's
#     .venv if it is missing. Training and every PyTorch measurement still use text-VPA's .venv only.
#   * PYTHONPATH and friends are cleared: ROS 2 Humble's Python 3.10 paths must not reach Python 3.12.
#   * Everything is written under <repository>/runs/ (E1's downloads and environments: runs/assets/, which is
#     not a run's output and is not deleted with one). text-VPA's corpus, benchmarks, checkpoint and runs are
#     read where they are and never written.
#   * One GPU job at a time: a script that needs the GPU refuses to start while another process holds its memory.
#   * tmux targets are matched exactly ("=name"; a bare -t matches by prefix).
#
# Variables (defaults in brackets):
#   TEXT_VPA   [$HOME/text-VPA]                 the predecessor's checkout: its .venv, data2/ and runs/ (read only)
#   PY         [$TEXT_VPA/.venv/bin/python]     the interpreter
#   DATA       [$TEXT_VPA/data2]                DATA/corpus, DATA/eval, DATA/models (read only)
#   MODEL_DIR  [$DATA/models/gemma-4-E2B]       the released checkpoint (with its .complete marker)
#   REQUIRE_GPU [1]                             0: run on the CPU (dry runs with a tiny checkpoint)
#   ALLOW_BUSY_GPU [0]                          1: start although another process holds GPU memory

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

unset PYTHONPATH PYTHONHOME PYTHONSTARTUP VIRTUAL_ENV CONDA_PREFIX
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
# Nothing is written outside runs/: no bytecode next to the sources, temporary files (the harness's program
# directories, pytest's) under runs/tmp, and no download (checkpoint and data are local).
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$ROOT/runs/tmp"
mkdir -p "$TMPDIR"
# Expandable segments keep the caching allocator from failing on fragmentation when batches of varying shape
# follow each other.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TEXT_VPA="${TEXT_VPA:-$HOME/text-VPA}"
PY="${PY:-$TEXT_VPA/.venv/bin/python}"
DATA="${DATA:-$TEXT_VPA/data2}"
MODEL_DIR="${MODEL_DIR:-$DATA/models/gemma-4-E2B}"
export REQUIRE_GPU="${REQUIRE_GPU:-1}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "error: $*" >&2; exit 1; }

# runs_dir <path>: the absolute path of an output directory, which must lie under <repository>/runs/.
runs_dir() {
  local path
  path="$(realpath -m "$1")"
  [[ "$path" == "$ROOT/runs" || "$path" == "$ROOT/runs/"* ]] || die "$1 is not under $ROOT/runs/ (this repository writes nowhere else)"
  echo "$path"
}

# check_python: the interpreter exists, is Python 3.12 or later, has no foreign paths, and has the packages.
check_python() {
  [[ -x "$PY" ]] || die "$PY is missing: these scripts use text-VPA's .venv (set TEXT_VPA or PY)"
  "$PY" - <<'PYEOF'
import importlib, os, sys
assert sys.version_info[:2] >= (3, 12), f"{sys.executable} runs Python {sys.version.split()[0]}; expected 3.12 or later"
stray = [p for p in sys.path if "/opt/ros" in p or "python3.10" in p]
assert not stray, f"foreign paths on sys.path: {stray}"
missing = []
for name in ("torch", "transformers", "tokenizers", "numpy"):
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    sys.exit(f"missing in {sys.prefix}: {', '.join(missing)} (add them to this environment and tell the author)")
import torch, transformers
assert hasattr(transformers, "Gemma4ForConditionalGeneration"), f"transformers {transformers.__version__} has no Gemma 4"
if os.environ.get("REQUIRE_GPU", "1") != "1":
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}, "
          "no GPU required (REQUIRE_GPU=0)")
    sys.exit(0)
assert torch.cuda.is_available(), "PyTorch does not see the GPU"
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
(x @ x).float().sum().item()
print(f"python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}, "
      f"{torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
PYEOF
}

# run_tests [file]: the repository's test suite in this environment (about a minute on the CPU), its output also
# written to file. Returns pytest's exit status, or 127 if pytest is missing; it never ends the calling script, so
# a check that runs the tests first still runs its other stages.
run_tests() {
  local out="${1:-/dev/null}"
  if ! "$PY" -c "import pytest" 2>/dev/null; then
    echo "pytest is missing in $PY's environment" | tee "$out" >&2
    return 127
  fi
  "$PY" -m pytest -q -p no:cacheprovider --basetemp "$TMPDIR/pytest" 2>&1 | tee "$out"
  return "${PIPESTATUS[0]}"
}

# check_gpu_free: one GPU job at a time.
check_gpu_free() {
  [[ "$REQUIRE_GPU" == 1 && "$ALLOW_BUSY_GPU" != 1 ]] || return 0
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ "${used:-0}" -ge 1024 ]]; then
    command -v tmux >/dev/null && tmux ls 2>/dev/null || true
    die "another process holds ${used} MiB of GPU memory (nvidia-smi, tmux ls); wait for it or set ALLOW_BUSY_GPU=1"
  fi
}

# check_model: the released checkpoint is complete.
check_model() {
  [[ -f "$MODEL_DIR/config.json" ]] || die "$MODEL_DIR has no config.json (set MODEL_DIR)"
  [[ -f "$MODEL_DIR/.complete" || "$REQUIRE_GPU" != 1 ]] || die "$MODEL_DIR has no .complete marker: the download did not finish"
}

# start_tmux <session> <log> <command...>: run a command detached, its variables passed explicitly (a running
# tmux server would otherwise hand its own, older environment to the new session).
start_tmux() {
  local session="$1" logfile="$2"; shift 2
  command -v tmux >/dev/null || die "tmux is not installed; run the script with 'run' in the foreground"
  if tmux has-session -t "=$session" 2>/dev/null; then
    die "tmux session '$session' already exists (tmux attach -t '=$session')"
  fi
  mkdir -p "$(dirname "$logfile")"
  local env_args=() name
  for name in $PASS_VARS TEXT_VPA PY DATA MODEL_DIR REQUIRE_GPU ALLOW_BUSY_GPU HF_TOKEN; do
    if [[ -n "${!name:-}" ]]; then env_args+=(-e "$name=${!name}"); fi
  done
  tmux new-session -d -s "$session" -c "$ROOT" "${env_args[@]}" "$* 2>&1 | tee -a '$logfile'"
  echo "started in tmux session '$session'"
  echo "  watch:  tmux attach -t '=$session'      (detach: Ctrl-b d)"
  echo "  log:    tail -n 20 $logfile"
}
