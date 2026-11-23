# chat-VPA: reference code of the Chat-VPA preprint

Implementation of *Chat-VPA: Semi-Autoregressive Block Discrete Flow Matching with Autoregressive Trajectory
Distillation* (Yeonseok Lee, SLING AI Inc.; draft of 22 November 2026, `preprint_261122_chat_vpa.pdf`, kept in
the author's project files and not in this repository). **The preprint is the ground truth** for every formula
and for the design of each experiment. If the code and the preprint disagree, the preprint wins, except where a
decision recorded in `docs/` or in the list at the end of this file says otherwise. Read the relevant section
before changing any math, and cite the equation in a comment.

**Status: nothing of Chat-VPA has been measured.** The code on `main` is text-VPA's verified Stage 0 code
(`docs/provenance.md`). The first experiment is E0 (preprint Sec. 6.2), developed on branch `e0`.

## Layout

| Path | Preprint | Contents |
|---|---|---|
| `chat_vpa/backbone.py` | Sec. 3.2 | `Stage0Realizer` on Gemma 4 E2B: `prefill` (the frozen backbone on the prompt, KV-storing blocks only, into a `PrefixCache`), `canvas_hidden` (canvas rows with canvas-only LoRA, the mask embedding, zero-initialised (t, h) modulation, the AR shift with a clamped anchor), `logits`, `weighted_nll` |
| `chat_vpa/flow.py` | Sec. 3.1, Eqs. 3-4, Lemma 1 | `KappaSchedule`, `mixture_path_sample`, `uniform_time_grid`, the jump kernel (`tilted_jump_kernel`, `kernel_distribution`, `sample_jump`), `timestep_embedding` |
| `chat_vpa/data.py` | Sec. 6.2 | prompt formats `complete` and `write`, `CorpusStream`, canvases, whole-canvas training states |
| `chat_vpa/canvas_sampler.py` | Sec. 3.1 with M = 1 | text-VPA's whole-canvas sampler (`realize`): exact kernel, confidence and left-to-right orders; the B = N reference of the block sampler |
| `chat_vpa/autoregressive.py` | Sec. 6.2 | the AR baseline: the released model's own `generate` |
| `chat_vpa/postprocess.py` | Sec. 6.2 | one truncation rule for both decoders |
| `chat_vpa/train.py` | | text-VPA's Stage 0 trainer and its hooks |
| `chat_vpa/evaluate.py`, `report.py`, `analyze.py`, `preflight.py` | Sec. 6.2 | Pass@k of either decoder; paired bootstrap over problems; outcomes and the def-line check; checkpoint, tokenizer and data against what the code assumes |
| `chat_vpa/runtime.py`, `device.py` | | checkpoints (`chat-vpa/1`, reads `text-vpa/1`), metric logging, backend selection |
| `chat_vpa/evaluation/` | Sec. 6.2 | resource-limited execution (isolation, not a security sandbox), unbiased Pass@k, HumanEval and MBPP tasks |
| `scripts/ncp/env.sh`, `stage0_sweep.sh` | | the server's rules in one place; the predecessor's sweep through the ported code |
| `docs/provenance.md` | | what was imported from text-VPA at `6f97346`, what changed, what was dropped; the corpus |

## Commands

```bash
pip install -r requirements-dev.txt     # torch first for CUDA
python -m pytest                        # about a minute on a CPU; must pass on Python 3.12 and 3.13
ruff check .                            # must pass
```

## Working agreements

1. **Trustworthy over flattering.** Report failures, surprises and uncertainty plainly.
2. **Rules before runs.** Decision rules are written in `docs/` before a run, and results are read by them.
3. **Nothing is selected on a test set.** Diagnostics use MBPP's validation split (90 problems). HumanEval and
   MBPP's test split stay untouched until a protocol run.
4. **One change per run** where possible (the predecessor's second Stage 0 run changed two things and their
   effects could not be separated).
5. **Anything that conditions on the reference is labelled as such** and never reported as the method.
6. **Every number in a document says what it is:** measured (by which run), reported (by which source) or
   predicted (by which model).
7. **The author decides** the license, the open choices of an experiment and any deviation from the preprint.
   Ask first; record every decision in `docs/` with its date.
8. **Git:** `main` holds the import; work on E0 goes to branch `e0`; pull requests when the author asks.

## Rules of the code (text-VPA's Stage 0 rules, for E0)

1. **The backbone is frozen, and prompt rows and commit rows are the frozen backbone.** Its parameters never
   change; a checkpoint holds only what is trained (LoRA factors, the mask embedding, the modulation). No
   adapter, modulation or mask embedding acts on a prompt position or on a committed token, so the KV cache is
   the frozen backbone's causal keys and values of the committed tokens (preprint Prop. 1). `Stage0Realizer`
   must reproduce `Gemma4ForCausalLM` under causal masks with the adapters at their initial values (tested); keep
   that test when transformers is upgraded.
2. **One prompt cache.** `prefill` is the frozen backbone. Do not add anything trainable to the prefill path.
3. **The same treatment for both decoders.** Prompt, token budget, token draw and truncation are shared by the
   block sampler, the whole-canvas sampler and the autoregressive baseline (`evaluate.py`, `postprocess.py`). A
   change to one is a change to all, and the baseline stays the released model's own `generate`.
4. **Adapter state is an argument.** The switch that turns the adapters on is set inside
   `Stage0Realizer._block`, because activation checkpointing calls the block again during the backward pass.
5. **Nothing is selected on a test set** (agreement 3). The final checkpoint is the one evaluated.
6. **A training prompt never shows what the canvas must contain.** The `write` format's assert names the function
   only (`assert name(...) == ...`). The predecessor's first run listed the parameter names there and the
   realizer learned to copy an assert's arguments into its `def` line (38% invalid `def` lines on MBPP). After any
   change to a prompt format, run `python -m chat_vpa.analyze --mbpp` on samples before reading Pass@1.
7. **Do not rename a parameter or a submodule of the realizer.** `~/text-VPA/runs/stage0-a/last.pt` must load
   (`tests/test_port.py`).
8. **No stubs** in core logic (losses, kernels, samplers, trainers).

## Tests

- Every test compares against a reference computed independently (explicit loops, float64 where possible, or
  transformers' own Gemma 4), never with the function under test. Fixed seeds; values, not only shapes; CPU.
  float32 against float64: `rtol=1e-5, atol=1e-6` unless a stated numerical reason requires more. Monte Carlo
  checks: 5 standard errors.
- The tiny random Gemma 4 of `tests/test_stage0.py` has every feature of E2B that the code relies on: local and
  global blocks with different head sizes, a 6-token sliding window, per-layer embeddings, KV sharing in its
  upper 3 blocks (double-wide MLPs) and the final logit soft cap. New tests of the realizer use it, with random
  non-zero adapters where the adapters must matter.
- A test that exposes a library defect is kept as `pytest.mark.xfail(strict=True, reason=...)` until fixed, then
  the marker is removed. Never weaken a correct reference.
- No ported test is dropped or weakened silently; `docs/provenance.md` holds the counts.

## The server (NCP `vpa-gpu-l4-01`; `scripts/ncp/env.sh` enforces what a script can)

- Ubuntu 22.04, one NVIDIA L4 (24 GB), user `ylee`. **Monthly plan: never shut it down.** No script may shut
  it down or read `SHUTDOWN_ON_SUCCESS` / `SHUTDOWN_ON_FAILURE`.
- **Python:** `~/text-VPA/.venv/bin/python` only (Python 3.12, torch 2.14.1+cu130, transformers 5.21.0). Do
  not create another environment. If a package is missing, add it to this environment and tell the author
  which. Install nothing that pins another torch.
- The system Python 3.10 carries ROS 2 Humble: every script unsets `PYTHONPATH PYTHONHOME PYTHONSTARTUP
  VIRTUAL_ENV CONDA_PREFIX` and sets `PYTHONNOUSERSITE=1`; one-off commands run as
  `env -u PYTHONPATH ~/text-VPA/.venv/bin/python ...`.
- `~/Vision-Predictor-Action` belongs to another project: never touch it.
- This repository is cloned to `~/chat-VPA` and writes only under `~/chat-VPA/runs/`. Read only, where they
  are: `~/text-VPA/data2/corpus`, `~/text-VPA/data2/eval`, `~/text-VPA/data2/models/gemma-4-E2B`, and the
  predecessor's runs under `~/text-VPA/runs/`.
- One GPU job at a time: check `tmux ls` and `nvidia-smi` first. Long jobs run detached in tmux; tmux targets
  match by prefix, so always use `-t "=name"`. Bash reads a running script from disk: never `git pull` while
  one runs.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the scripts set it).
- The evaluation harness executes generated code with resource limits; it is not a security sandbox.
- Commands for the author are short: anything long goes into a script in the repository.

## Decisions where the paper is silent (do not change silently)

None yet.
