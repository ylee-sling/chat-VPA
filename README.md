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
> **The first experiment, E0, is closed with a negative result** (branch `e0`, measured on one NVIDIA L4, 24 to
> 25 November 2026; [`docs/e0.md`](docs/e0.md), Section 10). Block-DFM on the frozen Gemma 4 E2B, with 16-token
> blocks in 4 passes and the exact kernel, reaches Pass@1 0.4% [0.0, 1.1] on MBPP validation against 23.8% for the
> autoregressive baseline (ratio 0.019 [0.000, 0.047]): the pre-registered hypothesis H1 clearly fails, and by
> the protocol's own rule the preprint's student model is not trained. The runs with 32-token blocks were not run
> (the author's decision of 25 November 2026). One seed, Python only, 450 samples per cell.
>
> **From E1 on the project asks another question** ([`docs/e1.md`](docs/e1.md), its pre-registration): can a
> drafter whose tokens the frozen `google/gemma-4-E2B-it` verifies exactly produce the model's own text faster
> than the model alone and than the drafter Google ships for it? Step 0, which trains nothing, is measured
> (26 November 2026; `docs/e1.md`, Section 10). Google's drafter decodes 1.43x faster than the model alone in
> PyTorch on one NVIDIA L4 in bfloat16, where its output is not token-identical to the model's own, and 1.42x in
> float32, where it is identical on all 50 prompts; 1.57x on the server's CPU (4 threads, float32, identical); and
> 1.05x in LiteRT-LM, Google's on-device runtime, on its CPU and GPU backends. Step 1 is registered and confirmed
> (27 November 2026; `docs/e1.md`, Section 11). Its first trained arm, A2 (Google's drafter distilled on the
> model's own greedy rollouts), is measured (28 November 2026; Section 11.11): exact in float32 wherever Google's is,
> it decodes 1.536x faster than the model alone in PyTorch on the L4 in float32 (Google's: 1.451x) and pays over
> Google's drafter clearly in every table, by 5.8% there and 8.9% in vLLM.

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
- **On branch `e0`, the Block-DFM core, E0's pipeline and E1's Step 0** (not on `main` yet). The core: the staircase, fused and window masks; the fused
  commit; the block sampler with the exact kernel, the prefix-growing and the confidence orders and a pass
  audit; the staircase training pass and the block loss; batched autoregressive rollouts as training targets.
  Its tests check the masks against an explicit loop over the preprint's Eq. 5, the training pass against the
  sampling passes and against transformers' own Gemma 4 under that mask, the cache against the frozen
  backbone's, one block spanning the canvas against the whole-canvas sampler, and a causal window against
  `generate`. These tests verify the implementation, not the method.

The numbers that motivated the design are measurements of the predecessor (text-VPA's `docs/stage0.md` and
`docs/stage1.md`); E0's are this code's.

## Next

**E1, Step 1** ([`docs/e1.md`](docs/e1.md), Section 11, confirmed by the author on 27 November 2026). First a
check of LiteRT-LM's GPU backend, whose speculative decoding did not keep the model's text in Step 0
(`bash scripts/ncp/e1_p2.sh`, run: undecided) and timing runs (`bash scripts/ncp/e1_step1.sh`, run); then, in one
run (`bash scripts/ncp/e1_step1.sh a2`, run), the model's own greedy rollouts on English prompts and the first
trained arm by Step 0's rules, A2: Google's drafter distilled on those rollouts, measured beside it. Now A3, a
drafter that decides one latent code per block and drafts the block in one call: its control A3-0 is training
(`bash scripts/ncp/e1_step1.sh a3 add`, running), and beside it, without the GPU, P3 read what LiteRT-LM's model
file holds (`bash scripts/ncp/e1_p3.sh`, run: its drafter is not Google's Transformers drafter quantised). Built for
what follows (Section 11.13): A3-0's speed through A3's own decoding loop (`bash scripts/ncp/e1_step1.sh
a3-measure`), A3-oracle with the oracle gate (`a3-oracle`), A3 itself if the gate allows it (`a3-predict`), and a
drafter's weights written into a copy of LiteRT-LM's file to measure it in that runtime (`bash scripts/ncp/e1_p3.sh
route`, then `engine`). B1, E0's Block-DFM as a drafter the model
verifies, which Step 0's bridge admitted (E0's checkpoints, unchanged, yield 2.6 tokens per backbone pass on
MBPP code), waits for the author's sign-off after A2 and A3. Before any product claim the LiteRT-LM measurement
runs on a physical laptop. Step 0 itself (`bash scripts/ncp/e1_setup.sh`, then `bash scripts/ncp/e1_step0.sh`) measured
the model alone and with Google's drafter in PyTorch, LiteRT-LM and vLLM, the ceiling that the cost of a pass sets
for any drafter, and E0's checkpoints as verified block drafters.

E0's record: its pre-registration, the checks before the run, the decisions and the result are in
[`docs/e0.md`](docs/e0.md). What the preprint draft is known to get wrong is listed in
[`docs/preprint-corrections.md`](docs/preprint-corrections.md).

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
