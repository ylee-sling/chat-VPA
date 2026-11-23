# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""chat-VPA: semi-autoregressive block discrete flow matching with autoregressive trajectory distillation.

Reference code of the Chat-VPA preprint (Yeonseok Lee, SLING AI Inc., draft of 22 November 2026). The modules
below were imported from text-VPA's verified Stage 0 code at commit 6f97346 (docs/provenance.md):

    backbone.py        Gemma 4 text decoder as a realizer on a frozen backbone: prompt prefill (one KV cache),
                       canvas-only LoRA, bidirectional canvas with two-sided local windows, mask embedding,
                       zero-initialised (t, h) modulation, the AR shift with a clamped anchor token
    flow.py            the mixture path, the exact jump kernel, sinusoidal (t, h) features
    data.py            prompt formats (complete / write), corpus stream, canvases, whole-canvas training states
    canvas_sampler.py  K parallel steps over one whole canvas (the B = N reference of the block sampler)
    autoregressive.py  the baseline: the released model's own ``generate``
    postprocess.py     one truncation rule for both decoders
    train.py           the whole-canvas trainer and its hooks (CLI)
    evaluate.py        Pass@k of either decoder in the repository's harness (CLI)
    report.py          paired bootstrap over problems; text-VPA's Stage 0 verdicts (CLI)
    analyze.py         how samples failed; the def-line check (CLI)
    preflight.py       checkpoint, tokenizer and data against what the code assumes (CLI)
    runtime.py         checkpoints and metric logging
    device.py          backend selection
    evaluation/        resource-limited execution, unbiased Pass@k, HumanEval and MBPP tasks
"""

__version__ = "0.1.0"
