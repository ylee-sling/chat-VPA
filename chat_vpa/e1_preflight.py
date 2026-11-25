# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_preflight.py -- what E1's Step 0 assumes about the downloaded checkpoints, checked on the machine before
anything is measured (docs/e1.md, Section 2).

    python -m chat_vpa.e1_preflight --model ASSETS/models/gemma-4-E2B-it \\
        --assistant ASSETS/models/gemma-4-E2B-it-assistant --prompts ASSETS/prompts/dev.jsonl

It prints, for the author to paste back: the versions and the machine; the target's and the drafter's
architecture and parameter counts (lookup tables apart); the drafter's default draft length; the end-of-sequence
ids; the chat prompt exactly as the tokenizer renders it (thinking must be off: no ``<|think|>``); and a smoke
decode of a few prompts, alone and with the drafter, that says whether the verified output equals the model's
own and how many tokens a cycle yields. The smoke decode is a check that the pieces fit, *not a result*: a few
prompts, a few tokens, no warm-up worth the name.

A failed expectation is printed as a line that starts with ``UNEXPECTED`` and the command exits with status 1.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from chat_vpa.e1_common import DTYPES, chat_ids, load_target, read_prompts
from chat_vpa.e1_decode import decode_once, first_difference, load_assistant, parse_arm

__all__ = ["parameter_counts"]


def parameter_counts(module: torch.nn.Module) -> dict[str, int]:
    """Parameters of a module, with those of its embedding tables (lookups, and the tied output layer) apart."""
    seen: set[int] = set()
    total = tables = 0
    table_ids = {id(p) for m in module.modules() if isinstance(m, torch.nn.Embedding) for p in m.parameters()}
    for p in module.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if id(p) in table_ids:
            tables += p.numel()
    return {"total": total, "embedding_tables": tables, "rest": total - tables}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E1: preflight of the instruction-tuned checkpoint and its drafter",
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--assistant", required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    p.add_argument("--smoke-prompts", type=int, default=3)
    p.add_argument("--smoke-tokens", type=int, default=48)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    import transformers

    from chat_vpa.device import resolve_device

    unexpected: list[str] = []
    device = resolve_device(args.device)
    print("== machine")
    print(f"  python {platform.python_version()}, torch {torch.__version__}, transformers {transformers.__version__}")
    print(f"  device {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    print(f"  {os.cpu_count()} CPUs visible, {torch.get_num_threads()} torch threads, {ram:.0f} GiB of memory")

    model, tokenizer = load_target(args.model, args.dtype, device)
    assistant = load_assistant(args.assistant, args.dtype, device)
    text = model.config.get_text_config()
    print("== target")
    counts = parameter_counts(model)
    print(f"  {type(model).__name__}: {text.num_hidden_layers} layers, width {text.hidden_size}, vocabulary "
          f"{text.vocab_size}, {getattr(text, 'num_kv_shared_layers', '?')} KV-sharing layers")
    print(f"  parameters (text only): {counts['total'] / 1e9:.3f}B, of which embedding tables "
          f"{counts['embedding_tables'] / 1e9:.3f}B and the rest {counts['rest'] / 1e9:.3f}B")
    drafter_text: Any = assistant.config.get_text_config()
    counts = parameter_counts(assistant)
    print("== drafter")
    print(f"  {type(assistant).__name__}: {drafter_text.num_hidden_layers} layers ({', '.join(drafter_text.layer_types)}), "
          f"width {drafter_text.hidden_size}, for a target of width {assistant.config.backbone_hidden_size}")
    print(f"  ordered embeddings: {assistant.config.use_ordered_embeddings} ({assistant.config.num_centroids} clusters, "
          f"top {assistant.config.centroid_intermediate_top_k})")
    print(f"  parameters: {counts['total'] / 1e6:.1f}M, of which embedding tables {counts['embedding_tables'] / 1e6:.1f}M "
          f"and the rest {counts['rest'] / 1e6:.1f}M")
    generation = assistant.generation_config
    print(f"  default draft length: {generation.num_assistant_tokens} "
          f"(schedule {getattr(generation, 'num_assistant_tokens_schedule', None)})")
    if assistant.config.backbone_hidden_size != text.hidden_size:
        unexpected.append("the drafter was made for another target width")

    print("== tokens and the chat prompt")
    print(f"  bos {tokenizer.bos_token_id}, eos {tokenizer.eos_token_id}, pad {tokenizer.pad_token_id}; generate stops "
          f"at {model.generation_config.eos_token_id}")
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "Hello."}], add_generation_prompt=True,
                                             tokenize=False)
    print(f"  one user turn renders as {rendered!r}")
    if "think" in rendered.lower():
        unexpected.append("the rendered prompt mentions thinking: E1 decodes with thinking off")

    prompts = read_prompts(args.prompts, args.smoke_prompts)
    print(f"== smoke decode ({len(prompts)} prompts, {args.smoke_tokens} new tokens, {args.dtype}; not a result)")
    arms = [parse_arm("ar"), parse_arm("mtp:4")]
    for row in prompts:
        ids = torch.tensor([chat_ids(tokenizer, row["prompt"])], dtype=torch.long, device=device)
        alone, drafted = (decode_once(model, assistant, ids, arm, args.smoke_tokens) for arm in arms)
        difference = first_difference(drafted["tokens"], alone["tokens"])
        cycles = max(drafted["target_passes"] - 1, 1)
        print(f"  {row['id']}: {len(alone['tokens'])} tokens; with the drafter "
              + ("the same tokens" if difference is None else f"DIFFERENT from token {difference}")
              + f", {(len(drafted['tokens']) - drafted['first_tokens']) / cycles:.2f} tokens per cycle, "
              f"{alone['seconds']:.2f} s alone and {drafted['seconds']:.2f} s drafted")
        if difference is not None and args.dtype == "fp32":
            unexpected.append(f"{row['id']}: verified decoding differs from the model's own in float32")
        if not alone["tokens"]:
            unexpected.append(f"{row['id']}: the model generated nothing")
    print(f"  the last response begins {tokenizer.decode(alone['tokens'])[:200]!r}")
    for line in unexpected:
        print(f"UNEXPECTED: {line}")
    print("== preflight " + ("FAILED" if unexpected else "passed"))
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
