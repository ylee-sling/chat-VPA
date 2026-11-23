# chat-VPA

**Semi-autoregressive block discrete flow matching with autoregressive trajectory distillation.**

[![License: GPL-3.0-only](https://img.shields.io/badge/License-GPL--3.0--only-blue.svg)](LICENSE)

Reference code of the preprint *Chat-VPA: Semi-Autoregressive Block Discrete Flow Matching with Autoregressive
Trajectory Distillation* (Yeonseok Lee, SLING AI Inc.; draft of 22 November 2026).

The paper asks whether one language model can commit a block of 16 or 32 tokens in 2 to 4 parallel passes: a
causal committed stream writes a key-value cache, a bidirectional active window reads it and is denoised with
the exact jump kernel of the mixture path, and the model is trained on the low-entropy rollouts of an
autoregressive teacher.

## Status

> [!IMPORTANT]
> **Nothing of Chat-VPA has been measured.** No Block-DFM model has been trained, and this repository holds no
> result of the method.

What exists today:

- **The predecessor's verified Stage 0 code,** imported from [text-VPA](https://github.com/ylee-sling/text-VPA)
  at commit `6f97346`: Gemma 4 E2B as a discrete flow-matching realizer on a frozen backbone (prompt prefill
  into a KV cache, canvas-only rank-64 adapters, a trainable mask embedding, zero-initialised (t, h)
  modulation, the AR shift), the whole-canvas sampler with the exact jump kernel, the autoregressive baseline
  (the released model's own `generate`), one truncation rule for both decoders, the Pass@k harness, the paired
  bootstrap report and the failure analysis. [`docs/provenance.md`](docs/provenance.md) lists what was kept,
  what was changed and what was dropped.
- **Its tests:** 103 imported tests and 2 guards of the import pass on CPU with Python 3.12 and 3.13; among
  them, the realizer reproduces transformers' `Gemma4ForCausalLM` under causal masks on a tiny random model.
- **On branch `e0`, the Block-DFM core** (not on `main` yet): the staircase, fused and window masks; the fused
  commit; the block sampler with the exact kernel, the prefix-growing and the confidence orders and a pass
  audit; the staircase training pass and the block loss; batched autoregressive rollouts as training targets.
  Its tests check the masks against an explicit loop over the preprint's Eq. 5, the training pass against the
  sampling passes and against transformers' own Gemma 4 under that mask, the cache against the frozen
  backbone's, one block spanning the canvas against the whole-canvas sampler, and a causal window against
  `generate`. These tests verify the implementation, not the method.

The numbers that motivate the design are measurements of the predecessor (text-VPA's `docs/stage0.md` and
`docs/stage1.md`), not of this code.

## Next

The first experiment is **E0**, the bridge experiment of the preprint's Sec. 6.2: Block-DFM on the frozen
Gemma 4 E2B, trained under the staircase block-causal mask on human references or on the backbone's own
autoregressive rollouts, evaluated on MBPP's validation split. Its pre-registration is
[`docs/e0.md`](docs/e0.md) on branch `e0`, a draft whose open choices are the author's; no training starts
before he has confirmed it. Three checks on the L4 come first (`bash scripts/ncp/first_checks.sh`, then
`bash scripts/ncp/first_checks.sh show`): the ported code against the predecessor's samples, the throughput of
autoregressive rollouts, and the block machinery on the real weights. What the preprint draft is known to get
wrong is listed in [`docs/preprint-corrections.md`](docs/preprint-corrections.md).

## Install and test

Python 3.12 or later. Install PyTorch for your backend first, then:

```bash
pip install -r requirements-dev.txt
python -m pytest          # about a minute on a CPU; nothing is downloaded
ruff check .
```

The tests build a tiny random Gemma 4 with every feature of E2B the code relies on (local and global blocks,
a sliding window, per-layer embeddings, shared keys and values in the upper blocks, the final logit soft cap).

## Layout

| Path | Contents |
|---|---|
| `chat_vpa/backbone.py` | `Stage0Realizer`: the frozen backbone's prefill and the adapted canvas pass |
| `chat_vpa/flow.py` | mixture path, exact jump kernel, (t, h) features |
| `chat_vpa/data.py` | prompt formats, corpus stream, canvases, whole-canvas training states |
| `chat_vpa/canvas_sampler.py` | K parallel steps over one whole canvas |
| `chat_vpa/autoregressive.py`, `postprocess.py` | the baseline decoder; one truncation rule for both decoders |
| `chat_vpa/train.py`, `evaluate.py`, `report.py`, `analyze.py`, `preflight.py` | command lines |
| `chat_vpa/evaluation/` | resource-limited execution, unbiased Pass@k, HumanEval and MBPP tasks |
| `scripts/ncp/` | scripts for the project's GPU server |
| `docs/` | provenance; decisions and pre-registrations are recorded here before a run |

Generated programs are executed by the Pass@k harness in resource-limited processes. That is isolation, not a
security sandbox: run evaluations on a machine you can discard.

## Citation

See [`CITATION.cff`](CITATION.cff). The predecessor: Yeonseok Lee, *Text-VPA: Adapting Vision-Predictor-Action
to Few-Step Non-Autoregressive Code Generation via Flow Matching*, preprint, SLING AI Inc., 2026 (version of
17 November 2026).

## License

GPL-3.0-only. Copyright (C) 2026 SLING AI Inc.
