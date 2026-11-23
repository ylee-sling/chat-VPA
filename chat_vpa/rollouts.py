# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
rollouts.py -- autoregressive rollouts of the frozen backbone as training targets (Chat-VPA Sec. 4.1, Sec. 6.2).

E0's second arm trains on "the backbone's own AR rollouts for the same prompts at tau_T = 0.2" (Sec. 6.2). A
rollout is one continuation of a training prompt by the released model's own ``generate``, under the decoding
rule of the baseline and of every sampler in this repository (temperature, then the ``top_k`` most probable
tokens, then the smallest nucleus of mass >= ``top_p`` among them), with the baseline's budget of new tokens
and the baseline's truncation rule (``postprocess``). Its target is

    tokens of the truncated program, then one <eos>

which is the canvas format of the human-reference arm (``data.Example.canvas_ids``), so the two arms differ in
the target only.

``generate_rollouts`` is ``autoregressive.generate_autoregressive`` for a batch of distinct prompts: the prompts
are left-padded and masked, every row stops on its own (its <eos>, or the truncation rule's stop in its decoded
text, checked every few tokens), and the batch ends when every row has stopped or the budget is spent. Gemma 4's
positions are rotary, so the offset that left padding gives a row's positions changes no attention score; with
greedy decoding the batch reproduces per-prompt ``generate`` token for token (tested on the tiny model).

How a row ended (``Rollout.stop``):

    truncation   the truncation rule found the program's end in the generated text
    eos          the model ended the text itself before the rule fired
    budget       neither within ``max_new_tokens``: the program is cut off; the rollout is dropped, and its
                 prompt is dropped from the human-reference arm as well (decided in docs/e0.md)
    too_long     the truncated program and its <eos> do not fit the canvas (dropped likewise)

``teacher_forced_stats`` measures what Corollary 1 asks for before any student is trained: along a stored
target, the entropy of the decoding distribution s_i = q_T(. | c, w^{<i}) at every program position (its mean per
token estimates the entropy rate h_T of the rollouts), and the negative log-likelihood of the target's tokens at
temperature 1 (for human references a proxy: their own entropy is unknown). It runs the frozen backbone through
the realizer's code path (adapters off, causal canvas), which reproduces ``Gemma4ForCausalLM`` (tested).

Shapes: B rows, P padded prompt length, S vocabulary size.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from chat_vpa.autoregressive import _eos_ids, cut_at_eos
from chat_vpa.backbone import Stage0Realizer, left_pad
from chat_vpa.data import Example, Stage0Tokenizer, collate
from chat_vpa.postprocess import is_complete, truncate_complete, truncate_write

__all__ = ["STOPS", "Rollout", "decoding_entropy", "finish_rollout", "generate_rollouts", "rollouts_for",
           "teacher_forced_stats"]

STOPS: tuple[str, ...] = ("truncation", "eos", "budget", "too_long")


@dataclass(frozen=True)
class Rollout:
    """One continuation of one prompt and the training target made from it."""

    kind: str                      # the prompt's format, "complete" | "write"
    stop: str                      # one of ``STOPS``
    generated: int                 # tokens generated before the row stopped
    text: str                      # the truncated program (what the harness would run for this continuation)
    canvas_ids: list[int] | None   # program tokens + one <eos>; None when the rollout is dropped

    @property
    def kept(self) -> bool:
        return self.canvas_ids is not None


def finish_rollout(raw_text: str, ended: bool, generated: int, kind: str, tokenizer: Stage0Tokenizer,
                   canvas_len: int) -> Rollout:
    """The rollout of one row from its decoded continuation.

    Args:
        raw_text: the generated tokens before the row's first <eos>, decoded.
        ended: the row emitted <eos> within the budget.
        generated: tokens generated before the row stopped.
        kind: the prompt's format; it selects the truncation rule (``postprocess``).
        canvas_len: N; the program and its <eos> must fit.
    """
    if kind not in ("complete", "write"):
        raise ValueError("kind must be 'complete' or 'write'")
    text = truncate_complete(raw_text) if kind == "complete" else truncate_write(raw_text)
    if is_complete(raw_text, kind):
        stop = "truncation"
    elif ended:
        stop = "eos"
    else:
        return Rollout(kind, "budget", generated, text, None)
    ids = tokenizer.encode_batch([text])[0]
    if len(ids) + 1 > canvas_len:
        return Rollout(kind, "too_long", generated, text, None)
    return Rollout(kind, stop, generated, text, [*ids, tokenizer.eos_id])


@torch.no_grad()
def generate_rollouts(model: Any, hf_tokenizer: Any, prompts: Sequence[Sequence[int]], kinds: Sequence[str],
                      max_new_tokens: int, temperature: float, top_p: float, top_k: int, seed: int,
                      check_every: int = 8) -> tuple[list[list[int]], list[bool], int]:
    """One continuation of each prompt, all in one batch of the released model's ``generate``.

    Args:
        model: the released causal model, on its device.
        hf_tokenizer: its Hugging Face tokenizer.
        prompts: B id lists, <bos> included, of any lengths (left-padded here).
        kinds: B prompt formats; a row stops once ``postprocess.is_complete`` holds for its decoded text.
        seed: seeds the token draws of this batch (a batch is reproducible from its prompts, their order and
            this seed).

    Returns:
        for each row the tokens generated before its first <eos> (or padding), whether it emitted <eos> within
        the budget, and the number of decoding steps the batch ran.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    if len(prompts) != len(kinds) or not prompts:
        raise ValueError("need one kind per prompt and at least one prompt")
    device = next(model.parameters()).device
    eos_ids = _eos_ids(model, hf_tokenizer)
    pad_id = hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else 0
    stops = [*eos_ids, pad_id]                                     # rows that stopped early are padded
    input_ids, valid, _ = left_pad([list(p) for p in prompts], pad_id, device)
    width = input_ids.shape[1]
    rows = len(prompts)

    def decode(row: Tensor) -> str:
        return hf_tokenizer.decode(cut_at_eos(row[width:].tolist(), stops), skip_special_tokens=True)

    class _RowComplete(StoppingCriteria):
        def __init__(self) -> None:
            self.finished = torch.zeros(rows, dtype=torch.bool, device=device)

        def __call__(self, ids: Tensor, scores: Tensor, **kwargs: Any) -> Tensor:
            if (ids.shape[1] - width) % check_every == 0:
                for row in range(rows):
                    if not bool(self.finished[row]) and is_complete(decode(ids[row]), kinds[row]):
                        self.finished[row] = True
            return self.finished.clone()

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sampling: dict[str, Any] = {"do_sample": False}
    if temperature > 0:
        sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k}
    output = model.generate(
        input_ids=input_ids, attention_mask=valid.long(), max_new_tokens=max_new_tokens,
        eos_token_id=eos_ids, pad_token_id=pad_id,
        stopping_criteria=StoppingCriteriaList([_RowComplete()]), **sampling,
    )
    new = output[:, width:].tolist()
    generated = [cut_at_eos(row, stops) for row in new]
    ended = [any(token in eos_ids for token in row) for row in new]
    return generated, ended, output.shape[1] - width


def rollouts_for(model: Any, hf_tokenizer: Any, tokenizer: Stage0Tokenizer, examples: Sequence[Example],
                 canvas_len: int, max_new_tokens: int, temperature: float, top_p: float, top_k: int, seed: int,
                 check_every: int = 8) -> tuple[list[Rollout], int]:
    """A rollout for the prompt of each example (one batch), and the decoding steps the batch ran."""
    generated, ended, steps = generate_rollouts(
        model, hf_tokenizer, [ex.prompt_ids for ex in examples], [ex.kind for ex in examples], max_new_tokens,
        temperature, top_p, top_k, seed, check_every)
    rollouts = [finish_rollout(tokenizer.decode(ids), done, len(ids), ex.kind, tokenizer, canvas_len)
                for ids, done, ex in zip(generated, ended, examples, strict=True)]
    return rollouts, steps


# ---------------------------------------------------------------------------------------------
# What Corollary 1 asks to measure
# ---------------------------------------------------------------------------------------------
def decoding_entropy(logits: Tensor, temperature: float, top_p: float, top_k: int) -> Tensor:
    """Entropy in nats of the distribution a token is drawn from under the decoding rule: logits / temperature,
    the ``top_k`` most probable tokens, the smallest nucleus of mass >= ``top_p`` among them, renormalised (the
    rule of ``generate`` and of ``canvas_sampler.draw_tokens``). Temperature 0 is the argmax: entropy 0.

    Shapes: ``[M, S]`` -> ``[M]`` float64.
    """
    if temperature < 0 or not 0 < top_p <= 1 or top_k < 1:
        raise ValueError("need temperature >= 0, 0 < top_p <= 1 and top_k >= 1")
    if temperature == 0.0:
        return torch.zeros(logits.shape[0], dtype=torch.float64, device=logits.device)
    top, _ = torch.topk(logits.double() / temperature, min(top_k, logits.shape[-1]), dim=-1)   # sorted, descending
    probs = torch.softmax(top, dim=-1)
    before = probs.cumsum(dim=-1) - probs                          # mass of the more probable tokens
    probs = torch.where(before < top_p, probs, torch.zeros_like(probs))
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return 0.0 - (probs * torch.log(probs.clamp_min(torch.finfo(torch.float64).tiny))).sum(dim=-1)


@torch.no_grad()
def teacher_forced_stats(realizer: Stage0Realizer, tokenizer: Stage0Tokenizer, examples: Sequence[Example],
                         device: torch.device, temperature: float, top_p: float, top_k: int,
                         micro_batch: int = 8, chunk: int = 256, autocast: Any = None) -> list[dict[str, float]]:
    """The frozen backbone's view of each example's canvas, read causally after its prompt (one forward pass per
    micro-batch, adapters off).

    Returns one dict per example:
        tokens        program tokens (the canvas without its <eos>)
        entropy       sum over the program tokens of the decoding distribution's entropy, nats
        nll           sum over the program tokens of -log p(token) at temperature 1 (no truncation), nats
        eos_entropy, eos_nll   the same two quantities at the <eos> that ends the canvas
    """
    out: list[dict[str, float]] = []
    for start in range(0, len(examples), micro_batch):
        group = list(examples[start:start + micro_batch])
        width = max(len(ex.canvas_ids) for ex in group)
        batch = collate(group, width, tokenizer.eos_id)
        canvas = torch.cat([batch.anchor[:, None], batch.target], dim=1).to(device)
        zeros = torch.zeros(len(group), device=device)

        def forward(batch=batch, canvas=canvas, zeros=zeros, width=width) -> Tensor:
            cache = realizer.prefill(*left_pad(batch.prefix_rows, tokenizer.pad_id, device))
            return realizer.canvas_hidden(cache, canvas, torch.zeros_like(canvas, dtype=torch.bool), zeros,
                                          zeros + 1.0, mode="causal", adapters=False)[:, :width]

        if autocast is None:
            hidden = forward()
        else:
            with autocast:
                hidden = forward()
        positions = torch.arange(width)[None, :] < batch.length[:, None]            # program tokens and <eos>
        chosen = positions.to(device)
        states, targets = hidden[chosen], batch.target.to(device)[chosen]
        entropy, nll = [], []
        for first in range(0, states.shape[0], chunk):
            logits = realizer.logits(states[first:first + chunk]).double()
            entropy.append(decoding_entropy(logits, temperature, top_p, top_k))
            nll.append(-torch.log_softmax(logits, dim=-1).gather(-1, targets[first:first + chunk, None])[:, 0])
        entropy_all, nll_all = torch.cat(entropy).cpu(), torch.cat(nll).cpu()
        offset = 0
        for length in batch.length.tolist():
            e, n = entropy_all[offset:offset + length], nll_all[offset:offset + length]
            out.append({"tokens": float(length - 1), "entropy": float(e[:-1].sum()), "nll": float(n[:-1].sum()),
                        "eos_entropy": float(e[-1]), "eos_nll": float(n[-1])})
            offset += length
    return out
