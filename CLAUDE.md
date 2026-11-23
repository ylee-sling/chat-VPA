# chat-VPA: reference code of the Chat-VPA preprint

Implementation of *Chat-VPA: Semi-Autoregressive Block Discrete Flow Matching with Autoregressive Trajectory
Distillation* (Yeonseok Lee, SLING AI Inc.; draft of 22 November 2026, `preprint_261122_chat_vpa.pdf`, kept in
the author's project files and not in this repository). **The preprint is the ground truth** for every formula
and for the design of each experiment. If the code and the preprint disagree, the preprint wins, except where a
decision recorded in `docs/` or in the list at the end of this file says otherwise. Read the relevant section
before changing any math, and cite the equation in a comment. `docs/preprint-corrections.md` lists what the draft
is known to get wrong (the "two tokens per pass" reading of K = 128, the finished latent oracle, two points of
Table 5): do not repeat those statements, and do not edit the preprint unasked.

**Status: no Chat-VPA model has been trained or measured.** The code on `main` is text-VPA's verified Stage 0
code (`docs/provenance.md`). The first experiment is E0 (preprint Sec. 6.2), developed on branch `e0`: the
Block-DFM core, its tests, the rollout generator and the checks for the L4 exist, and the checks have run (the port
is exact: `docs/e0.md`, Section 2.4). The author has confirmed all of `docs/e0.md` (C1 to C12, 23 November 2026);
it is frozen, and a later change is a deviation recorded in its Section 9. The rollout writer and the E0 trainer
exist and are tested on the CPU with the tiny model; neither has run on the L4. Nothing has been trained yet.

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
| `chat_vpa/block_masks.py` | Sec. 3.3 Eq. 5, Sec. 3.5 | `staircase_masks` (window rows of a training pass), `fused_masks` (commit and window rows of a sampling pass), `window_positions` |
| `chat_vpa/block_dfm.py` | Secs. 3.2-3.5, Props. 1-2 | `BlockDFM` (the realizer inside block windows, the frozen backbone on everything committed): `prefill`, `fused_pass`, `window_pass`, `commit_pass`, `training_hidden`, registers; `BlockCache`; `PassAudit` |
| `chat_vpa/block_data.py` | Sec. 3.3, Eq. 11, Sec. 4.3 | `collate_blocks`, `block_training_state` (per-block K, t, prefix-shaped or independent reveals, seam and filling weights), `right_open` (O_t) |
| `chat_vpa/block_objective.py` | Eq. 12 | `dfm_loss` (L_DFM); L_Block-KD, L_Boundary, L_Step, L_Plan are not built |
| `chat_vpa/block_sampler.py` | Algorithm 1, Prop. 3 | `sample_blocks`: fused commit, exact kernel / prefix / confidence orders, early exit, stopping, passes per block (K-bar), the audit |
| `chat_vpa/block_evaluate.py`, `compare_samples.py` | Sec. 6.2 | Pass@k of the block sampler in `evaluate.py`'s format (Stage 0 and E0 checkpoints), the cache spot check; sample-by-sample comparison of two evaluations |
| `chat_vpa/rollouts.py`, `rollout_throughput.py` | Sec. 4.1, Cor. 1, Sec. 6.2 | batched AR rollouts of the frozen backbone as training targets; the entropy of the decoding distribution along a target; the throughput measurement (`--kind` for one prompt format; writes `rollouts.jsonl`) |
| `chat_vpa/inspect_rollouts.py` | | how the rollouts of a throughput run ended: for `write` prompts, where the named function ends and what follows it (a diagnostic for C6; changes no rule) |
| `chat_vpa/rollout_writer.py` | Sec. 6.2 | E0's data (C6): the validation set and the kept training set, stratified by format, each example with its reference and one rollout; resumable chunks, halving the batch on out-of-memory; the summary of `docs/e0.md` Section 7 (kept shares, stops, lengths, entropy) |
| `chat_vpa/train_e0.py` | Secs. 3.3, 6.2, Eq. 12 | E0's trainer: text-VPA's Stage 0 loop with `collate_blocks`, `block_training_state` and `dfm_loss` behind its hooks; `RolloutData`, `EpochStream`; held-out L_DFM on both targets; checkpoints of stage `e0` (registers, the data's fingerprint), read by `block_evaluate` |
| `scripts/ncp/` | | `env.sh` (the server's rules in one place), `stage0_sweep.sh` (the predecessor's sweep through the ported code), `rollout_throughput.sh` (`inspect` runs `inspect_rollouts.py` on its output), `port_check.sh` (the port against text-VPA's samples), `block_check.sh` (pass audit, cache check and seconds per block on the real weights; not a result), `first_checks.sh` (the three in one tmux session; `show` prints what the author pastes back), `e0_data.sh` (`check`: batch 128 against 256; then the data), `e0_train.sh` (one run of T16, R16, T32, R32; `STOP_AT=100` for the timing) |
| `docs/e0.md` | Secs. 6.2, 6.4 | the pre-registration of E0: the checks before the run and their reading rules, the open choices C1 to C12, H1 to H3 verbatim, the table E0 fills, the decision record |
| `docs/preprint-corrections.md` | | what to raise at the preprint's next revision |
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
   Ask first; record every decision in `docs/` with its date. Everything else is decided without asking (the
   author's request of 23 November 2026): routine engineering, repository formatting and minor implementation
   choices are made, recorded where they belong and reported, not proposed. Pause for the author only when a
   decision affects a pre-registered protocol, a scientific conclusion or a significant amount of GPU time.
8. **Git:** `main` holds the import; work on E0 goes to branch `e0`; pull requests when the author asks.
9. **Commit trailers** (the author's request of 23 November 2026). Every commit message ends with one block of
   trailers, with no blank line inside it (git and GitHub read only the last paragraph as trailers): first
   `Co-authored-by: Yeonseok Lee <ylee@sling.ai.kr>`, then the attribution lines that the session's instructions
   give. The commits up to `2076e2d` on `main` and `cfb51e7` on `e0` lack the first line; they are not rewritten
   or force-pushed unless the author asks.

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
9. **A block costs exactly its passes.** `sample_blocks` runs under `PassAudit`, which counts with hooks what
   the network evaluated: K-bar passes per block, the commit rows through the KV-storing blocks only (Prop. 2).
   Never remove or weaken it. Call networks as `module(...)`, never `module.forward(...)`.
10. **The block sampler with B = N is the whole-canvas sampler.** One block that spans the canvas, without
    registers, must reproduce `canvas_sampler.realize` token for token from the same seed (tested). This ties the
    new code to the verified one; keep the test.

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
- So that nothing is written outside `runs/`, `env.sh` also sets `TMPDIR` to `runs/tmp`,
  `PYTHONDONTWRITEBYTECODE=1` and the Hub's offline switches. Run the tests on the server through a script
  (`port_check.sh` with `STAGES=tests`), not with a bare `pytest`, which would write its cache into the checkout.
- The evaluation harness executes generated code with resource limits; it is not a security sandbox.
- Commands for the author are short: anything long goes into a script in the repository.

## Decisions where the paper is silent (do not change silently)

Made while building the Block-DFM core (23 November 2026). Those marked *(E0)* are choices of the first experiment
as well, listed in `docs/e0.md` with their alternatives; the author confirmed them on 23 November 2026 (C6, C8,
C9).

- **Local blocks inside a window.** Eq. 5 says the sliding window "applies on top of M" and that a window always
  lies inside it. The code keeps the two-sided rule |n(u) - n(v)| < W of the Stage 0 realizer among window rows
  (`block_masks`): it never binds for B + 1 <= W, and it makes one block of B = N the whole-canvas realizer
  (rule 10).
- **Per-layer embeddings of non-tokens.** A masked slot takes the trainable mask vector and a register its own
  vector as the main embedding; the per-layer embedding of both is looked up at the tokenizer's `<mask>` id (for
  masked slots this is Stage 0's decision 5).
- **Registers.** R vectors in `BlockDFM` (not in the realizer, so a Stage 0 checkpoint still loads), initialised
  to the mask vector plus seeded Gaussian noise of standard deviation 0.1 times that vector's RMS, so that they
  differ from one another. A causal window (the AR fallback) has no registers.
- **Commit rows inside a fused pass** are the frozen backbone by a per-row adapter mask (`AdapterState.active`
  as a tensor): LoRA and the modulation are computed and discarded on those rows. The commit rows leave the
  pass before the first KV-sharing block.
- **The clean stream of a training pass** is `Stage0Realizer.prefill` on the clean sequence, without gradients
  and through the KV-storing blocks only; the window rows of all blocks then run in one pass (Prop. 1(b)). In E0
  the clean rows are the frozen backbone, so L_AR (Eq. 14) has nothing to train.
- **Blocks of a training response** *(E0)*. N = M B with M = ceil((program tokens + 1) / B): the block that
  holds the first `<eos>` is the last one trained, because the sampler stops there.
- **Prefix-shaped states** have ceil(kappa_t B - 1e-9) revealed slots: an integer product (the linear schedule's
  k B / K) is not pushed up by rounding.
- **Loss weights** *(E0)*. A slot that is both among the first b_0 of its block and `<eos>` filling gets the
  filling's 0.1.
- **Scheduled count of the prefix and confidence orders** *(E0)*. Sec. 3.5 says they "reveal the same expected
  number of slots per pass". After the pass at t the block has ceil(kappa_{t+h} B) revealed slots, at least one
  more than before, and all of them after the last pass; for the prefix order these are the prefix-shaped
  training states of Sec. 3.3. (The whole-canvas sampler rounds pi times the masked count instead. Computed with
  both rules: at (16, 4) that rule starts the second pass from 1 revealed slot where training has 2, at (32, 4)
  the second and third from 2 and 9 where training has 3 and 10, and at (32, 2) the second from 9 where
  training has 10.)
- **Early exit.** Off at delta = 0: in exact arithmetic eps > 0, and float32 must not decide otherwise. eps uses
  the posterior at temperature 1 without the tokens that are never drawn.
- **Token draw of the block sampler** *(E0)*. Algorithm 1 says "draw x^i from the posterior"; the code draws with
  the baseline's rule (temperature, top-k, top-p; `canvas_sampler.draw_tokens`), as rule 3 requires, and never
  draws `[MASK]`, `<pad>` or `<bos>`.
- **Stopping** *(E0)*. A row stops at the first block that holds `<eos>` (its output ends before that token) or
  once the truncation rule has found the program's end in the blocks finished so far (checked after every
  block); the budget is ceil(256 / B) blocks, the baseline's 256 new tokens.
- **Batches.** Rows that have stopped leave the batch. A block of a batch runs until no row has a masked slot;
  `row_passes` records for each row the passes after which its own block was complete, and K-bar is their mean.
- **Rollout targets** *(E0)*. The generated text is cut by the truncation rule, tokenised again and followed by
  one `<eos>`; a rollout that reaches the budget without a stop, or whose program does not fit the canvas, is
  dropped (`rollouts.py`), and its prompt leaves both arms. No content filter. The kept set is stratified by
  format at the stream's `write` share (13.5%), each quota filled in stream order; 100,000 kept examples
  (`rollout_writer.py`, C6).
- **Order of the training examples** *(E0)*. A seeded permutation of the kept set per epoch, the same for both
  arms and both block sizes (`train_e0.EpochStream`); with 100,000 kept examples and 10,000 steps of 32 each is
  seen about 3.2 times.
- **Checkpoints of an AR-shift model in bfloat16.** A cache built block by block equals the one-pass prefill up
  to rounding (float32: 1e-5 or less, tested; bfloat16: a few units in the last place), so a separate commit
  pass (ablation A10) gives the fused commit's samples in float32 and need not in bfloat16.
