# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_train.py -- the trained drafters of E1's Step 1 (docs/e1.md, Section 11): A2, Google's drafter distilled on the
target's own greedy rollouts (11.4: S6 to S8), and A3-0, the one-pass realiser without a latent (11.5: D4, D6, D8).

    python -m chat_vpa.e1_train --arm a2 --model ASSETS/models/gemma-4-E2B-it \\
        --assistant ASSETS/models/gemma-4-E2B-it-assistant --rollouts runs/e1-step1/rollouts/train \\
        --held-out runs/e1-step1/rollouts/held-out --run runs/e1-step1/a2
    python -m chat_vpa.e1_train --arm a3-0 ... --assistant runs/e1-step1/a2/final --run runs/e1-step1/a3-0

``--stop-at 100`` is the timing run of Section 11.8: the schedule keeps the full run's horizon, and ``timing.json``
holds the measured seconds per step and the hours the full run would take (*predicted*). A run resumes from
``RUN/last.pt``.

**What the drafter sees.** Google's drafter (``Gemma4AssistantForCausalLM``) has no keys and values of its own:
each of its layers attends to the target's, those of the target's last KV-storing layer of the same attention
type, and every query it drafts sits at the position q of the last token the cycle knows (Transformers'
``SinglePositionMultiTokenCandidateGenerator``). In the first cycle after a prefill, Transformers 5.21 hands the
drafter the target's keys of every position before q in the full-attention layers and of the W + 1 positions
before q in the sliding ones (its bidirectional sliding mask is inclusive, distance <= W, over the last entries).
This trainer trains under that rule at every position (``visibility``), and ``tests/test_e1_train.py`` checks that
a batched call under it reproduces Transformers' own single-query call. After a partly accepted cycle Transformers
also hands the drafter the keys of rejected drafts (docs/e1.md, Section 11.9); vLLM's and LiteRT-LM's paths are not
read here.

**A2 (S6, S7).** At every position q of a response, from its first token to the one before its last, the drafter is
unrolled over G_train = 4 depths as it runs on the accepted path: depth 1 takes the target's input embedding of x_q
concatenated with the target's last hidden state at q - 1; depth k > 1 takes the embedding of the true token
x_{q+k-1} (the accepted draft) with the drafter's own projected state from depth k - 1; every query sits at q.
Depth k is scored on x_{q+k}, the target's greedy token (a response's stop token is a target; when the budget
ended a response, its last token has none): the cross-entropy over the whole vocabulary from the drafter's output
layer (its embedding table), plus ``cluster_weight`` times the cross-entropy of x_{q+k}'s cluster over the
centroid logits (at inference only the top clusters are scored, so the right one has to rank among them). A
depth's loss is the mean over its scored positions; the depths weigh equally. Every parameter of the drafter is
trained; the assignment of tokens to clusters (``token_ordering``, a buffer) is not.

**A3-0 (D4, D6, D8).** One call of the drafter over B = 8 slots per position q: slot j (1 to B) takes a slot
vector in the token embedding's half of the input, concatenated with the target's last hidden state at q - 1, at
position q, under the same visibility; the slots cannot attend to each other (the drafter's layers have no keys
of their own). ``slot_input`` says what the slot vector is (docs/e1.md, Section 11.12): ``replace`` (D4 as
registered) puts ``code + depth[j]`` in place of the token embedding, so no slot sees x_q, the last token of the
cycle, which Google's drafter receives only through that embedding (the hidden state at q - 1 is the one that chose
x_q); ``add`` adds ``code + depth[j]`` to x_q's embedding, so that with both at zero every slot is A2's depth 1.
``code`` is one learned vector (A3-0 has no latent; A3-oracle and A3 put a codebook's vector in its place),
initialised under ``replace`` to the target's input embedding of the tokenizer's mask token (zero if it has none),
as text-VPA's mask vector was, and under ``add`` to zero; ``depth`` holds one learned vector per slot, zero at the
start. Slot j is scored on x_{q+j} by A2's per-position loss, the cluster term included (the author's decision of
27 November 2026, so that A3-0's one change from A2 is the one call over B slots; docs/e1.md, Section 11.9); the
slots weigh equally. Its weights start from ``--assistant`` (D4: A2's, or A1's if A2 accepts fewer tokens).

**The target (S5, S8)**: frozen, bfloat16, without gradients. One pass of its text model over a batch gives every
position's last hidden state and the keys and values the drafter attends to (recomputed in every step, not
stored). The drafter's weights are float32. The vocabulary-wide cross-entropy runs over chunks of rows, each
chunk's logits recomputed in the backward pass, so the logits of a batch never exist at once. A batch can be run as
micro-batches (``micro_batch``; A3-0's default is 8 of its 16) whose gradients add up to the batch's exactly: every
depth's loss is divided by that depth's count over the whole batch, known before the pass. A batch that runs out of
GPU memory is run again in micro-batches of half the size, which hold for the rest of the run; the step's gradient
is the same either way.

**Optimisation (S8)**: AdamW (0.9, 0.999) at 1e-4 without weight decay, linear warm-up over 3% of the steps and
cosine decay to zero (``train.learning_rate``), gradients clipped at norm 1 (text-VPA's trainers; S8 does not
say), batches of 16 rollouts, ``epochs`` passes in a seeded order per epoch (``train_e0.EpochStream``). The final
checkpoint is the one measured (agreement 3): ``RUN/final`` is a ``Gemma4AssistantForCausalLM`` directory, with
the other files of the drafter it started from, that ``generate`` and vLLM load as they load A1; A3-0 adds
``realiser.pt``.

**Held-out curves** (S2's 2,000 prompts, ``--held-out``): per depth (or slot), how often the drafter's greedy
choice -- the best token among its top clusters, as it drafts at inference -- is the target's token, the same for
the best token over the whole vocabulary, the losses, and the tokens per cycle those choices give along the
rollouts' greedy continuations (``tokens_per_cycle``: a cycle at q accepts the leading drafts that equal the
continuation and adds the next token; the first token comes from the prefill), at G in {2, 4, 6, 8} for both arms
and at G = B for A3-0 (A3-0 at G < B verifies the first G of its B slots). They are counts along the continuations with the training
pass's keys, not decode measurements; ``chat_vpa.e1_count`` makes the same counts along the development prompts'
continuations (``choice_hits``).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

__all__ = ["ARMS", "SLOT_INPUTS", "BlockEncoder", "CodePredictor", "CodedRealiser", "OnePassRealiser", "RolloutBatch",
           "TrainConfig", "VectorQuantiser", "a2_hidden", "attention_masks", "choice_hits", "cluster_map", "collate",
           "depth_loss", "drafter_choices", "drafter_hidden", "load_realiser", "one_layer_drafter", "prepare",
           "query_positions", "target_states", "tokens_per_cycle", "visibility", "vocab_cross_entropy"]

ARMS = ("a2", "a3-0", "a3-oracle", "a3")
CODED = ("a3-oracle", "a3")
CODE_COUNT_LENGTH = 4_096                  # longer than any codebook here: the histogram of chosen codes
SLOT_INPUTS = ("replace", "add")
EVAL_LENGTHS = (2, 4, 6, 8)


# ---------------------------------------------------------------------------------------------
# Data: a rollout is its chat prompt, its response and, if it stopped, its stop token
# ---------------------------------------------------------------------------------------------

def prepare(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Training sequences from rollout records: ``ids`` (prompt, response, stop token if any) and ``start`` (the
    first response position). A record with no scored position (fewer than two tokens after the prompt, counting
    the stop token) is dropped; the count of those is returned with the rest."""
    out: list[dict[str, Any]] = []
    dropped = 0
    for record in records:
        prompt, response = list(record["prompt_ids"]), list(record["response_ids"])
        stop = record.get("stop_id")
        ids = prompt + response + ([int(stop)] if stop is not None else [])
        if not prompt or len(ids) - len(prompt) < 2:
            dropped += 1
            continue
        out.append({"id": record["id"], "ids": ids, "start": len(prompt)})
    return out, dropped


@dataclass
class RolloutBatch:
    """Right-padded sequences: ``ids`` and ``valid`` ``[B, T]``, ``start`` and ``length`` ``[B]``."""

    ids: Tensor
    valid: Tensor
    start: Tensor
    length: Tensor

    def to(self, device: torch.device) -> RolloutBatch:
        return RolloutBatch(self.ids.to(device), self.valid.to(device), self.start.to(device), self.length.to(device))


def collate(examples: Sequence[dict[str, Any]], pad_id: int) -> RolloutBatch:
    if not examples:
        raise ValueError("an empty batch")
    width = max(len(e["ids"]) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    valid = torch.zeros((len(examples), width), dtype=torch.bool)
    for b, example in enumerate(examples):
        n = len(example["ids"])
        ids[b, :n] = torch.tensor(example["ids"], dtype=torch.long)
        valid[b, :n] = True
    start = torch.tensor([e["start"] for e in examples], dtype=torch.long)
    length = torch.tensor([len(e["ids"]) for e in examples], dtype=torch.long)
    return RolloutBatch(ids, valid, start, length)


def query_positions(batch: RolloutBatch) -> tuple[Tensor, Tensor]:
    """Per row the drafted positions q = start .. length - 2 (each has a next token), right-padded: ``positions``
    ``[B, Q]`` (padding repeats ``start``, which has keys before it) and ``valid`` ``[B, Q]``."""
    counts = batch.length - 1 - batch.start
    if bool((counts < 1).any()) or bool((batch.start < 1).any()):
        raise ValueError("every row needs a prompt and at least one scored position")
    width = int(counts.max())
    offset = torch.arange(width, device=batch.ids.device)
    valid = offset[None, :] < counts[:, None]
    positions = torch.where(valid, batch.start[:, None] + offset[None, :], batch.start[:, None])
    return positions, valid


def visibility(positions: Tensor, keys: int, window: int) -> tuple[Tensor, Tensor]:
    """Which of the target's ``keys`` key positions a query at each of ``positions`` ``[B, Q]`` attends to, as
    ``[B, 1, Q, keys]`` booleans: every position before it in the full-attention layers, the ``window`` + 1 before
    it in the sliding ones (Transformers' first cycle after a prefill; module docstring)."""
    j = torch.arange(keys, device=positions.device)
    q = positions[..., None]
    full = j < q
    sliding = full & (j >= q - window - 1)
    return full[:, None], sliding[:, None]


def attention_masks(full: Tensor, sliding: Tensor, implementation: str, dtype: torch.dtype) -> dict[str, Tensor]:
    """The masks in the form the drafter's attention takes: booleans for SDPA, additive floats for eager."""
    if implementation == "sdpa":
        return {"full_attention": full, "sliding_attention": sliding}
    if implementation == "eager":
        low = torch.finfo(dtype).min
        return {name: torch.zeros(m.shape, dtype=dtype, device=m.device).masked_fill(~m, low)
                for name, m in (("full_attention", full), ("sliding_attention", sliding))}
    raise ValueError(f"the drafter's attention is {implementation!r}: only 'sdpa' and 'eager' take explicit masks")


# ---------------------------------------------------------------------------------------------
# The frozen target and the drafter
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def target_states(target: Any, batch: RolloutBatch) -> tuple[Tensor, dict[str, tuple[Tensor, Tensor]], Tensor]:
    """One pass of the target's text model: the last hidden state of every position ``[B, T, D]`` (what
    ``generate`` hands the drafter as ``hidden_states[-1]``), the keys and values of its last KV-storing layer of
    each attention type, and the target's input embeddings of the tokens ``[B, T, D]``."""
    text = getattr(target.model, "language_model", target.model)
    out = text(input_ids=batch.ids, attention_mask=batch.valid.long(), return_shared_kv_states=True, use_cache=False)
    if not out.shared_kv_states:
        raise RuntimeError("the target returned no shared keys and values (a Gemma 4 text model is needed)")
    return out.last_hidden_state, dict(out.shared_kv_states), target.get_input_embeddings()(batch.ids)


def drafter_hidden(drafter: Any, inputs: Tensor, positions: Tensor, kv: dict[str, tuple[Tensor, Tensor]],
                   masks: dict[str, Tensor]) -> Tensor:
    """One call of the drafter on queries ``inputs`` ``[B, Q, 2 D]`` at ``positions``: its last hidden state
    ``[B, Q, d]`` (before ``post_projection``), as ``Gemma4AssistantForCausalLM.forward`` computes it, with the
    masks given instead of built."""
    x = drafter.pre_projection(inputs)
    out = drafter.model(input_ids=None, inputs_embeds=x, attention_mask=masks, position_ids=positions,
                        shared_kv_states=kv, use_cache=False)
    return out.last_hidden_state


def _drafter_inputs(drafter: Any, states: tuple[Tensor, dict[str, tuple[Tensor, Tensor]], Tensor]
                    ) -> tuple[Tensor, dict[str, tuple[Tensor, Tensor]], Tensor]:
    dtype = drafter.pre_projection.weight.dtype
    hidden, kv, embeds = states
    return hidden.to(dtype), {k: (v[0].to(dtype), v[1].to(dtype)) for k, v in kv.items()}, embeds.to(dtype)


def _masks_for(drafter: Any, positions: Tensor, keys: int) -> dict[str, Tensor]:
    text = drafter.config.get_text_config()
    full, sliding = visibility(positions, keys, int(text.sliding_window))
    return attention_masks(full, sliding, drafter.model.config._attn_implementation, drafter.pre_projection.weight.dtype)


def _gather(values: Tensor, index: Tensor) -> Tensor:
    """values ``[B, T, D]`` at index ``[B, Q]`` (clamped to the sequence) -> ``[B, Q, D]``."""
    index = index.clamp(0, values.shape[1] - 1)
    return values.gather(1, index[..., None].expand(*index.shape, values.shape[-1]))


def _targets(batch: RolloutBatch, positions: Tensor, valid: Tensor, count: int) -> tuple[Tensor, Tensor]:
    """x_{q+k} for k = 1 .. count ``[B, Q, count]`` and whether it exists (``scored``)."""
    steps = torch.arange(1, count + 1, device=positions.device)
    index = positions[..., None] + steps
    scored = valid[..., None] & (index <= (batch.length - 1)[:, None, None])
    tokens = batch.ids.gather(1, index.clamp(max=batch.ids.shape[1] - 1).flatten(1)).view(index.shape)
    return tokens, scored


def a2_hidden(drafter: Any, states: tuple[Tensor, dict[str, tuple[Tensor, Tensor]], Tensor], batch: RolloutBatch,
              depths: int, target_embeddings: Any = None) -> tuple[Tensor, Tensor, Tensor]:
    """Google's drafter unrolled over ``depths`` on the accepted path at every drafted position: its last hidden
    state per depth ``[B, Q, depths, d]``, the targets ``[B, Q, depths]`` and which are scored."""
    hidden, kv, embeds = _drafter_inputs(drafter, states)
    positions, valid = query_positions(batch)
    masks = _masks_for(drafter, positions, hidden.shape[1])
    targets, scored = _targets(batch, positions, valid, depths)
    previous = _gather(hidden, positions - 1)
    out = []
    for k in range(1, depths + 1):
        token = _gather(embeds, positions + k - 1)                  # x_{q+k-1}: x_q at depth 1, then the drafts
        last = drafter_hidden(drafter, torch.cat([token, previous], dim=-1), positions, kv, masks)
        out.append(last)
        if k < depths:
            previous = drafter.post_projection(last)
    return torch.stack(out, dim=2), targets, scored


class OnePassRealiser(nn.Module):
    """A3's realiser (D4): Google's drafter called once over ``slots`` query slots per position, slot j taking
    ``code + depth[j]`` in place of the token embedding (``slot_input="replace"``, D4 as registered) or added to the
    embedding of x_q, the cycle's last token (``"add"``; the module docstring). A3-0's ``code`` is one learned
    vector."""

    def __init__(self, drafter: Any, slots: int, code: Tensor, slot_input: str = "replace") -> None:
        super().__init__()
        if slots < 1:
            raise ValueError("need at least one slot")
        if slot_input not in SLOT_INPUTS:
            raise ValueError(f"slot_input must be one of {SLOT_INPUTS}")
        self.drafter = drafter
        self.slots = slots
        self.slot_input = slot_input
        dtype = drafter.pre_projection.weight.dtype
        self.code = nn.Parameter(code.detach().to(dtype).clone())
        self.depth = nn.Parameter(torch.zeros(slots, code.shape[-1], dtype=dtype))

    def forward(self, states: tuple[Tensor, dict[str, tuple[Tensor, Tensor]], Tensor], batch: RolloutBatch
                ) -> tuple[Tensor, Tensor, Tensor]:
        """The last hidden state of every slot ``[B, Q, slots, d]``, the targets ``[B, Q, slots]`` and which are
        scored."""
        last, targets, scored, _ = self._draft(states, batch, lambda *_: (self.code, {}))
        return last, targets, scored

    def decode_code(self, token: Tensor, previous: Tensor, kv: dict[str, tuple[Tensor, Tensor]], position: int
                    ) -> Tensor:
        """The code of one decoding cycle ``[D]`` (``e1_decode``'s one-pass drafting): A3-0's one vector."""
        return self.code

    def _draft(self, states: tuple[Tensor, dict[str, tuple[Tensor, Tensor]], Tensor], batch: RolloutBatch,
               code_at: Any) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        """The slots' pass with the code ``code_at(hidden, positions, valid)`` gives: one vector ``[D]`` for every
        position (A3-0) or one per position ``[B, Q, D]`` (A3-oracle, A3), with what else it computed."""
        hidden, kv, embeds = _drafter_inputs(self.drafter, states)
        positions, valid = query_positions(batch)
        rows, width = positions.shape
        previous = _gather(hidden, positions - 1)                     # [B, Q, D]
        code, aux = code_at(hidden, positions, valid)
        code = code.to(previous.dtype)
        if code.dim() == 1:
            slot = (code + self.depth).expand(rows, width, -1, -1)    # [B, Q, S, D]
        else:
            slot = code[:, :, None, :] + self.depth
        if self.slot_input == "add":
            first = _gather(embeds, positions)[:, :, None] + slot     # x_q's embedding + slot j
        else:
            first = slot
        inputs = torch.cat([first, previous[:, :, None].expand(-1, -1, self.slots, -1)], dim=-1)
        flat = positions.repeat_interleave(self.slots, dim=1)         # [B, Q * S], slot-minor
        masks = _masks_for(self.drafter, flat, hidden.shape[1])
        last = drafter_hidden(self.drafter, inputs.reshape(rows, width * self.slots, -1), flat, kv, masks)
        targets, scored = _targets(batch, positions, valid, self.slots)
        return last.view(rows, width, self.slots, -1), targets, scored, aux


# ---------------------------------------------------------------------------------------------
# A3-oracle and A3: a code per position (D1 to D3, D5, D8; docs/e1.md, Section 11.13)
# ---------------------------------------------------------------------------------------------

class BlockEncoder(nn.Module):
    """A3's encoder (D1), used only in training: at each drafted position q, the target's last hidden states at the
    block's own positions q + 1 .. q + B (the tokens the slots draft), each normalised without parameters and
    mapped to ``width`` through a GELU, concatenated in order and mapped to the code's space (the token
    embedding's width). A position past the sequence's end contributes zeros."""

    def __init__(self, hidden: int, slots: int, code_dim: int, width: int = 64) -> None:
        super().__init__()
        self.hidden, self.slots, self.code_dim, self.width = hidden, slots, code_dim, width
        self.inner = nn.Linear(hidden, width)
        self.outer = nn.Linear(slots * width, code_dim)

    def forward(self, hidden: Tensor, positions: Tensor, length: Tensor) -> Tensor:
        rows, width = positions.shape
        index = positions[..., None] + torch.arange(1, self.slots + 1, device=positions.device)   # [B, Q, S]
        exists = index <= (length - 1)[:, None, None]
        states = _gather(hidden, index.flatten(1)).view(rows, width, self.slots, -1).to(self.inner.weight.dtype)
        states = F.layer_norm(states, (states.shape[-1],))
        inner = F.gelu(self.inner(states)) * exists[..., None]
        return self.outer(inner.flatten(-2))                                                      # [B, Q, D]


class VectorQuantiser(nn.Module):
    """D2 and D3: ``codes`` vectors of ``dim``; each encoding goes to its nearest code (Euclidean). The codes are
    exponential moving averages (``decay``) of the encodings assigned to them. They are updated once per training
    step from all of that step's assignments (``apply``), so that a batch run again in halves after running out of
    memory is not counted twice (``discard``); a code that has gone unused (its moving count below
    ``restart_below``) restarts at a random encoding of the step. The first training batch starts every code at
    one of its encodings. The encoder learns through the straight-through estimator and the commitment loss
    ``beta`` x the squared distance to its code (a mean over the code's dimensions, summed over positions; the
    trainer divides it by the batch's positions, as it does the slots' losses)."""

    def __init__(self, codes: int, dim: int, decay: float = 0.99, beta: float = 0.25, restart_below: float = 0.1,
                 eps: float = 1e-5) -> None:
        super().__init__()
        if codes < 2:
            raise ValueError("a codebook needs at least two codes")
        self.codes, self.dim, self.decay, self.beta = codes, dim, decay, beta
        self.restart_below, self.eps = restart_below, eps
        self.register_buffer("embed", torch.zeros(codes, dim))
        self.register_buffer("cluster_size", torch.zeros(codes))
        self.register_buffer("embed_sum", torch.zeros(codes, dim))
        self.register_buffer("initialised", torch.zeros((), dtype=torch.bool))
        self._pending: list[tuple[Tensor, Tensor]] = []

    def nearest(self, encoding: Tensor) -> Tensor:
        flat = encoding.reshape(-1, self.dim).to(self.embed.dtype)
        distances = (flat.pow(2).sum(-1, keepdim=True) - 2 * flat @ self.embed.t()
                     + self.embed.pow(2).sum(-1)[None])
        return distances.argmin(-1).view(encoding.shape[:-1])

    def forward(self, encoding: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """The code of every position with the straight-through gradient ``[B, Q, D]``, its index ``[B, Q]`` and the
        commitment loss summed over the valid positions."""
        if self.training and not bool(self.initialised):
            self._initialise(encoding.detach()[valid])
        index = self.nearest(encoding.detach())
        quantised = self.embed[index].to(encoding.dtype)
        if self.training:
            self._pending.append((index[valid], encoding.detach()[valid].to(self.embed.dtype)))
        commitment = self.beta * ((encoding - quantised.detach()).pow(2).mean(-1) * valid).sum()
        return encoding + (quantised - encoding).detach(), index, commitment

    @torch.no_grad()
    def _initialise(self, rows: Tensor) -> None:
        if rows.shape[0] == 0:
            raise ValueError("the first batch has no position to start the codebook from")
        rows = rows.to(self.embed.dtype)
        if rows.shape[0] >= self.codes:
            pick = torch.randperm(rows.shape[0], device=rows.device)[: self.codes]
            start = rows[pick]
        else:                                           # fewer encodings than codes: repeat them, apart by a little
            pick = torch.randint(0, rows.shape[0], (self.codes,), device=rows.device)
            start = rows[pick]
            start = start + 1e-3 * start.pow(2).mean().sqrt() * torch.randn_like(start)
        self.embed.copy_(start)
        self.embed_sum.copy_(start)
        self.cluster_size.fill_(1.0)
        self.initialised.fill_(True)

    @torch.no_grad()
    def apply(self) -> dict[str, float]:
        """The step's update of the codebook; what it did."""
        if not self._pending:
            return {}
        index = torch.cat([i for i, _ in self._pending])
        rows = torch.cat([r for _, r in self._pending])
        self._pending = []
        counts = torch.bincount(index, minlength=self.codes).float()
        sums = torch.zeros_like(self.embed).index_add_(0, index, rows)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_sum.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        total = self.cluster_size.sum()
        size = (self.cluster_size + self.eps) / (total + self.codes * self.eps) * total
        self.embed.copy_(self.embed_sum / size[:, None])
        dead = self.cluster_size < self.restart_below
        restarted = int(dead.sum())
        if restarted and rows.shape[0]:
            fresh = rows[torch.randint(0, rows.shape[0], (restarted,), device=rows.device)]
            self.embed[dead] = fresh
            self.embed_sum[dead] = fresh
            self.cluster_size[dead] = 1.0
        share = counts / counts.sum().clamp(min=1.0)
        perplexity = torch.exp(-(share * torch.log(share.clamp(min=1e-12))).sum())
        return {"code_perplexity": float(perplexity), "codes_used": float((counts > 0).sum()),
                "codes_restarted": float(restarted)}

    def discard(self) -> None:
        self._pending = []


def one_layer_drafter(drafter: Any) -> Any:
    """D5's body: a drafter of one layer, with its own copies of the realiser's input projection, its last
    full-attention layer (which sees the whole prefix) and its final norm. Its token embedding, output layer and
    post-projection are not used by the predictor: the first two are the realiser drafter's own modules (no memory
    of their own), the third a frozen copy."""
    from transformers import Gemma4AssistantForCausalLM

    whole = drafter.config.to_dict()
    text = dict(whole["text_config"])
    kinds = list(text["layer_types"])
    if "full_attention" not in kinds:
        raise ValueError("the predictor copies a full-attention layer, and the drafter has none")
    source = max(i for i, kind in enumerate(kinds) if kind == "full_attention")
    overrides = (text.get("per_layer_config") or {}).get(str(source), {})
    text.update(num_hidden_layers=1, layer_types=["full_attention"], per_layer_config={"0": dict(overrides)})
    if text.get("num_kv_shared_layers"):
        text["num_kv_shared_layers"] = 1
    whole["text_config"] = text
    config = type(drafter.config).from_dict(whole)
    weight = drafter.pre_projection.weight
    one = Gemma4AssistantForCausalLM(config)
    one.model.embed_tokens = drafter.model.embed_tokens
    one.lm_head = drafter.lm_head
    if hasattr(drafter, "masked_embedding"):
        one.masked_embedding = drafter.masked_embedding
    one = one.to(device=weight.device, dtype=weight.dtype)
    one.pre_projection.load_state_dict(drafter.pre_projection.state_dict())
    one.post_projection.load_state_dict(drafter.post_projection.state_dict())
    one.post_projection.requires_grad_(False)
    one.model.layers[0].load_state_dict(drafter.model.layers[source].state_dict())
    one.model.norm.load_state_dict(drafter.model.norm.state_dict())
    return one


class CodePredictor(nn.Module):
    """D5: one drafter-style layer (``one_layer_drafter``) on the realiser's own inputs without a code -- x_q's
    embedding and the target's last hidden state at q - 1, at position q, under the same visibility -- and a linear
    map to the ``codes`` logits, started at zero (every code equally likely)."""

    UNUSED = ("body.model.embed_tokens.", "body.lm_head.", "body.masked_embedding.", "body.post_projection.")

    def __init__(self, body: Any, codes: int) -> None:
        super().__init__()
        self.body = body
        width = body.pre_projection.weight.shape[0]
        self.head = nn.Linear(width, codes).to(device=body.pre_projection.weight.device,
                                               dtype=body.pre_projection.weight.dtype)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def trainable(self) -> list[nn.Parameter]:
        """What A3 trains: the body's own projection, layer and norm, and the head."""
        return [p for name, p in self.named_parameters() if not name.startswith(self.UNUSED)]

    def trained_state(self) -> dict[str, Tensor]:
        """The state that ``realiser.pt`` keeps (without the modules the body shares with the realiser's drafter)."""
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith(self.UNUSED)}

    def load_trained(self, state: dict[str, Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith(self.UNUSED) for key in missing):
            raise ValueError(f"the predictor's saved state does not fit: missing {missing}, unexpected {unexpected}")

    def forward(self, states: tuple[Tensor, dict[str, tuple[Tensor, Tensor]], Tensor], positions: Tensor) -> Tensor:
        hidden, kv, embeds = _drafter_inputs(self.body, states)
        inputs = torch.cat([_gather(embeds, positions), _gather(hidden, positions - 1)], dim=-1)
        masks = _masks_for(self.body, positions, hidden.shape[1])
        return self.head(drafter_hidden(self.body, inputs, positions, kv, masks))           # [B, Q, C]

    def decode_logits(self, token: Tensor, previous: Tensor, full: tuple[Tensor, Tensor], position: int) -> Tensor:
        """One decoding cycle's logits ``[C]``: x_q's embedding ``token`` and the target's last hidden state
        ``previous`` (``[1, 1, D]`` each) at ``position`` q, attending to the target's full-attention keys and values
        of the accepted prefix (positions 0 .. q - 1), all of which a full-attention query at q sees."""
        dtype = self.body.pre_projection.weight.dtype
        keys = full[0].shape[2]
        if keys != position:
            raise ValueError(f"the predictor attends to the {position} positions before q, not to {keys}")
        positions = torch.tensor([[position]], device=token.device)
        masks = _masks_for(self.body, positions, keys)
        inputs = torch.cat([token, previous], dim=-1).to(dtype)
        hidden = drafter_hidden(self.body, inputs, positions, {"full_attention": (full[0].to(dtype), full[1].to(dtype))},
                                {"full_attention": masks["full_attention"]})
        return self.head(hidden)[0, 0]


class CodedRealiser(OnePassRealiser):
    """A3-oracle (no ``predictor``): the code at q is the quantised encoding of the true next block, so it
    conditions on the reference and is never the method (agreement 5). A3 (a ``predictor``): the code at q is the
    codebook's vector at the predictor's argmax, and the predictor learns the encoder's codes by cross-entropy (D8).
    Either code takes the place of A3-0's one vector; the slots are A3-0's otherwise (``slot_input``, ``depth``)."""

    def __init__(self, drafter: Any, slots: int, slot_input: str, encoder: BlockEncoder, quantiser: VectorQuantiser,
                 predictor: CodePredictor | None = None) -> None:
        super().__init__(drafter, slots, torch.zeros(quantiser.dim), slot_input)
        self.code.requires_grad_(False)                 # unused: the code comes from the codebook
        self.encoder, self.quantiser, self.predictor = encoder, quantiser, predictor

    @property
    def source(self) -> str:
        return "oracle" if self.predictor is None else "predicted"

    def codes(self, states: tuple[Tensor, dict[str, tuple[Tensor, Tensor]], Tensor], batch: RolloutBatch,
              hidden: Tensor, positions: Tensor, valid: Tensor) -> tuple[Tensor, dict[str, Any]]:
        """The code of every drafted position and what came with it: the commitment loss (A3-oracle), or the
        predictor's cross-entropy against the encoder's codes and its hits (A3)."""
        if self.predictor is None:
            code, index, commitment = self.quantiser(self.encoder(hidden, positions, batch.length), valid)
            return code, {"commitment": commitment, "index": index}
        with torch.no_grad():
            truth = self.quantiser.nearest(self.encoder(hidden, positions, batch.length))
        logits = self.predictor(states, positions)
        index = logits.argmax(-1)
        loss = F.cross_entropy(logits[valid].float(), truth[valid], reduction="sum")
        return self.quantiser.embed[index], {"predictor": loss, "predictor_hits": (index == truth)[valid].sum(),
                                             "index": index, "truth": truth}

    def forward(self, states: tuple[Tensor, dict[str, tuple[Tensor, Tensor]], Tensor], batch: RolloutBatch
                ) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        return self._draft(states, batch, lambda hidden, positions, valid: self.codes(states, batch, hidden,
                                                                                    positions, valid))

    def decode_code(self, token: Tensor, previous: Tensor, kv: dict[str, tuple[Tensor, Tensor]], position: int
                    ) -> Tensor:
        """A3's code of one decoding cycle: the codebook's vector at the predictor's argmax. A3-oracle has no code to
        decode with: its code is the true block's."""
        if self.predictor is None:
            raise ValueError("A3-oracle's code is the true next block's: it is counted, never decoded")
        logits = self.predictor.decode_logits(token, previous, kv["full_attention"], position)
        return self.quantiser.embed[int(logits.argmax())]

    def code_loss(self, states: tuple[Tensor, dict[str, tuple[Tensor, Tensor]], Tensor], batch: RolloutBatch
                  ) -> dict[str, Any]:
        """A3's training pass: the predictor's cross-entropy against the encoder's codes, without the slots."""
        if self.predictor is None:
            raise ValueError("only A3 has a predictor to train")
        hidden, _, _ = _drafter_inputs(self.drafter, states)
        positions, valid = query_positions(batch)
        _, aux = self.codes(states, batch, hidden, positions, valid)
        return {**aux, "valid": valid}


# ---------------------------------------------------------------------------------------------
# The per-position loss and the drafter's choices
# ---------------------------------------------------------------------------------------------

def cluster_map(drafter: Any) -> Tensor | None:
    """The cluster of every token under the drafter's clustered output layer (``token_ordering`` lists cluster c's
    tokens at c * V / C .. (c + 1) * V / C - 1), or None if its output layer is not clustered."""
    head = getattr(drafter, "masked_embedding", None)
    if head is None:
        return None
    ordering = head.token_ordering.long()
    vocab = ordering.numel()
    if not torch.equal(torch.sort(ordering).values, torch.arange(vocab, device=ordering.device)):
        raise ValueError("the drafter's token_ordering is not a permutation of the vocabulary: S7's cluster loss "
                         "needs every token in one cluster")
    clusters = torch.empty(vocab, dtype=torch.long, device=ordering.device)
    clusters[ordering] = torch.arange(vocab, device=ordering.device) // head.vocab_size_per_centroid
    return clusters


def _chunk_cross_entropy(hidden: Tensor, weight: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(hidden @ weight.t(), targets, reduction="sum")


def vocab_cross_entropy(hidden: Tensor, weight: Tensor, targets: Tensor, chunk: int) -> Tensor:
    """Sum over rows of the cross-entropy of ``targets`` under logits ``hidden @ weight.T``, chunk by chunk; each
    chunk's logits are recomputed in the backward pass instead of kept."""
    if chunk < 1:
        raise ValueError("chunk must be at least 1")
    total = hidden.new_zeros(())
    for s in range(0, hidden.shape[0], chunk):
        part = (hidden[s:s + chunk], weight, targets[s:s + chunk])
        if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
            total = total + checkpoint(_chunk_cross_entropy, *part, use_reentrant=False)
        else:
            total = total + _chunk_cross_entropy(*part)
    return total


def depth_loss(drafter: Any, last: Tensor, targets: Tensor, scored: Tensor, clusters: Tensor | None,
               cluster_weight: float, chunk: int, counts: Sequence[float] | None = None
               ) -> tuple[Tensor, dict[str, list[float]]]:
    """The mean over depths (or slots) that have a scored position of (the token's cross-entropy over the
    vocabulary + ``cluster_weight`` x its cluster's cross-entropy), each averaged over its scored positions; per
    depth, the sums and the count. ``counts``, the scored positions per depth of the whole batch when this is one of
    its micro-batches, replace this one's own in both averages, so that the micro-batches' losses add up to the
    batch's."""
    if cluster_weight and clusters is None:
        raise ValueError("a cluster loss needs a clustered output layer")
    depths = last.shape[2]
    if counts is None:
        counts = [float(scored[:, :, k].sum()) for k in range(depths)]
    if len(counts) != depths:
        raise ValueError("one count per depth")
    active = sum(1 for c in counts if c > 0)
    if not active:
        raise ValueError("no scored position in the batch")
    weight = drafter.lm_head.weight
    total = last.new_zeros(())
    stats: dict[str, list[float]] = {"count": [], "token_nll": [], "cluster_nll": []}
    for k in range(depths):
        chosen = scored[:, :, k]
        n = int(chosen.sum())
        rows, tokens = last[:, :, k][chosen], targets[:, :, k][chosen]
        token = vocab_cross_entropy(rows, weight, tokens, chunk) if n else rows.new_zeros(())
        cluster = rows.new_zeros(())
        if clusters is not None and n:
            cluster = F.cross_entropy(drafter.masked_embedding.centroids(rows), clusters[tokens], reduction="sum")
        stats["count"].append(float(n))
        stats["token_nll"].append(float(token.detach()))
        stats["cluster_nll"].append(float(cluster.detach()))
        if n:
            if counts[k] <= 0:
                raise ValueError("a micro-batch scores a depth that the batch's counts leave out")
            total = total + (token + cluster_weight * cluster) / counts[k]
    return total / active, stats


@torch.no_grad()
def drafter_choices(drafter: Any, hidden: Tensor, targets: Tensor, chunk: int, clusters: Tensor | None = None
                    ) -> tuple[Tensor, Tensor, Tensor]:
    """For rows of the drafter's last hidden state ``[N, d]``: the best token among its top clusters (what
    ``Gemma4AssistantMaskedEmbedder`` drafts; the best token overall if the head is not clustered), the best token
    over the whole vocabulary, and the cross-entropy of ``targets`` over the vocabulary, per row. ``clusters`` is
    ``cluster_map(drafter)``, computed here if not given."""
    head = getattr(drafter, "masked_embedding", None)
    if head is not None and clusters is None:
        clusters = cluster_map(drafter)
    clustered, overall, nll = [], [], []
    for s in range(0, hidden.shape[0], chunk):
        h = hidden[s:s + chunk]
        logits = h @ drafter.lm_head.weight.t()
        overall.append(logits.argmax(-1))
        nll.append(F.cross_entropy(logits, targets[s:s + chunk], reduction="none"))
        if head is None:
            clustered.append(overall[-1])
            continue
        top = head.centroids(h).topk(head.centroid_intermediate_top_k, dim=-1).indices
        allowed = torch.zeros(h.shape[0], head.num_centroids, dtype=torch.bool, device=h.device)
        allowed.scatter_(1, top, True)
        clustered.append(logits.masked_fill(~allowed[:, clusters], float("-inf")).argmax(-1))
    if not overall:
        empty = torch.zeros(0, dtype=torch.long, device=hidden.device)
        return empty, empty, torch.zeros(0, device=hidden.device)
    return torch.cat(clustered), torch.cat(overall), torch.cat(nll)


def tokens_per_cycle(hits: Sequence[Sequence[Sequence[bool]]], length: int) -> tuple[int, int]:
    """Cycles and the tokens they add along one greedy continuation. ``hits[i][k]``: whether the draft at depth
    k + 1 from the i-th drafted position (q = start + i) equals the continuation. A cycle at q accepts the leading
    hits among its first ``length`` depths, adds the next token, and the next cycle starts after them; the first
    token after the prompt comes from the prefill and is not counted. Returns (cycles, tokens)."""
    positions = len(hits)                                     # q = start .. start + positions - 1; tokens to add
    i, cycles = 0, 0
    while i < positions:
        accepted = 0
        while accepted < length and accepted < len(hits[i]) and hits[i][accepted] and i + accepted + 1 < positions:
            accepted += 1
        cycles += 1
        i += accepted + 1
    return cycles, positions


def _code_statistics(sums: dict[str, Any], aux: dict[str, Any], valid: Tensor) -> None:
    """How often each code is chosen at the valid positions, and how often the predictor's code is the encoder's."""
    index = aux.get("index")
    if index is not None:
        chosen = index[valid].flatten().cpu()
        counts = torch.bincount(chosen, minlength=CODE_COUNT_LENGTH).double()
        sums["code_counts"] = sums["code_counts"] + counts if "code_counts" in sums else counts
    if "predictor_hits" in aux:
        sums["predictor_hits"] = sums.get("predictor_hits", 0.0) + float(aux["predictor_hits"])
        sums["predictor_positions"] = sums.get("predictor_positions", 0.0) + float(valid.sum())


def code_perplexity(counts: Tensor) -> float:
    """exp of the entropy of the codes' shares."""
    share = counts / counts.sum().clamp(min=1.0)
    return float(torch.exp(-(share * torch.log(share.clamp(min=1e-12))).sum()))


@torch.no_grad()
def choice_hits(target: Any, drafter: Any, module: nn.Module | None, count: int, examples: Sequence[dict[str, Any]],
                batch_size: int, pad_id: int, device: torch.device, vocab_chunk: int, clusters: Tensor | None
                ) -> tuple[list[list[list[bool]]], list[list[list[bool]]], dict[str, list[float]]]:
    """Along each example's continuation, at each drafted position q (``query_positions``) and each depth (``module``
    None: Google's drafter unrolled on the accepted path, ``a2_hidden``, over ``count`` depths) or slot (``module``
    an ``OnePassRealiser``): whether the drafter's greedy choice -- the best token among its top clusters, as it
    drafts at inference -- is the continuation's token (``hits[example][position][depth]``; ``tokens_per_cycle``
    reads them), the same for the best token over the whole vocabulary, and the totals per depth: the positions
    scored (``n``), both kinds of hits and the summed cross-entropy (``nll``). A depth without a target there is a
    miss."""
    hits_clustered: list[list[list[bool]]] = []
    hits_overall: list[list[list[bool]]] = []
    sums = {"n": [0.0] * count, "clustered": [0.0] * count, "overall": [0.0] * count, "nll": [0.0] * count}
    for s in range(0, len(examples), batch_size):
        batch = collate(examples[s:s + batch_size], pad_id).to(device)
        states = target_states(target, batch)
        if module is None:
            last, targets, scored = a2_hidden(drafter, states, batch, count)
        else:
            out = module(states, batch)
            last, targets, scored = out[:3]
            if last.shape[2] != count:
                raise ValueError(f"the realiser has {last.shape[2]} slots, not {count}")
            if len(out) > 3:
                _code_statistics(sums, out[3], query_positions(batch)[1])
        rows, width = scored.shape[:2]
        hit_c = torch.zeros(rows, width, count, dtype=torch.bool, device=device)
        hit_o = torch.zeros_like(hit_c)
        for k in range(count):
            chosen = scored[:, :, k]
            clustered, overall, nll = drafter_choices(drafter, last[:, :, k][chosen], targets[:, :, k][chosen],
                                                      vocab_chunk, clusters)
            truth = targets[:, :, k][chosen]
            hit_c[:, :, k][chosen] = clustered == truth
            hit_o[:, :, k][chosen] = overall == truth
            sums["n"][k] += float(chosen.sum())
            sums["clustered"][k] += float((clustered == truth).sum())
            sums["overall"][k] += float((overall == truth).sum())
            sums["nll"][k] += float(nll.sum())
        positions = (batch.length - 1 - batch.start).tolist()
        for b, n in enumerate(positions):
            hits_clustered.append(hit_c[b, :n].tolist())
            hits_overall.append(hit_o[b, :n].tolist())
        del states, last, targets, scored
    return hits_clustered, hits_overall, sums


# ---------------------------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Everything that defines a run (``RUN/config.json`` and every checkpoint). ``steps`` is the run's horizon
    (from ``epochs`` over ``horizon_examples`` rollouts at ``batch`` per step when 0)."""

    arm: str = "a2"
    batch: int = 16
    epochs: float = 2.0
    horizon_examples: int = 0              # 0: the training rollouts given
    steps: int = 0                         # 0: from the three above
    lr: float = 1e-4
    warmup_fraction: float = 0.03
    min_lr_fraction: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    depths: int = 4                        # A2: G_train
    slots: int = 8                         # A3-0: B
    cluster_weight: float = 1.0            # S7's cluster term (A3-0 too: the author's decision of 27 November 2026)
    slot_input: str = "replace"            # A3-0: "replace" (D4 as registered) or "add" (x_q's embedding kept)
    codes: int = 256                       # A3-oracle, A3: C (D3)
    code_width: int = 64                   # A3-oracle: the encoder's width per position
    code_scale: float = 0.1                # A3-oracle: the encodings' RMS at the start / the token embedding's
    commitment: float = 0.25               # A3-oracle: beta of the commitment loss (D8)
    codebook_decay: float = 0.99           # A3-oracle: the codebook's moving averages
    micro_batch: int = 0                   # 0: the arm's own (A2: the whole batch, as timed; A3-0: 8)
    eval_depths: int = 8
    eval_every: int = 2_500
    save_every: int = 500
    log_every: int = 20
    seed: int = 0
    vocab_chunk: int = 1_024

    def finish(self, examples: int) -> TrainConfig:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if self.cluster_weight < 0:
            raise ValueError("cluster_weight must not be negative")
        if self.slot_input not in SLOT_INPUTS:
            raise ValueError(f"slot_input must be one of {SLOT_INPUTS}")
        if self.codes < 2 or self.code_width < 1 or self.code_scale <= 0 or not 0 < self.codebook_decay < 1:
            raise ValueError("codes >= 2, code_width >= 1, code_scale > 0 and 0 < codebook_decay < 1")
        if self.micro_batch <= 0:
            self.micro_batch = self.batch if self.arm == "a2" else min(self.batch, 8)
        self.micro_batch = min(self.micro_batch, self.batch)
        if self.horizon_examples <= 0:
            self.horizon_examples = examples
        if self.steps <= 0:
            self.steps = math.ceil(self.epochs * self.horizon_examples / self.batch)
        return self

    @property
    def warmup_steps(self) -> int:
        return round(self.warmup_fraction * self.steps)

    @property
    def schedule_steps(self) -> int:
        return self.steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def _is_out_of_memory(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def learning_rate(step: int, cfg: TrainConfig) -> float:
    from chat_vpa.train import learning_rate as stage0_learning_rate

    return stage0_learning_rate(step, cfg)  # type: ignore[arg-type]   # the same fields, duck-typed


def fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def mask_code(target: Any, tokenizer: Any) -> Tensor:
    """A3-0's starting code: the target's input embedding of the tokenizer's mask token, or zero."""
    width = target.get_input_embeddings().weight.shape[1]
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        return torch.zeros(width)
    with torch.no_grad():
        ids = torch.tensor([[int(mask_id)]], device=target.get_input_embeddings().weight.device)
        return target.get_input_embeddings()(ids)[0, 0].float().cpu()


class DrafterTrainer:
    """The loop: one batch of ``cfg.batch`` rollouts per step, the target's pass without gradients, the drafter's
    pass and loss with them."""

    def __init__(self, cfg: TrainConfig, target: Any, module: nn.Module, drafter: Any, stream: Any, pad_id: int,
                 run_dir: Path, device: torch.device, held_out: Sequence[dict[str, Any]] = (),
                 logger: Any = None) -> None:
        self.cfg, self.target, self.module, self.drafter = cfg, target, module, drafter
        self.stream, self.pad_id, self.run_dir, self.device = stream, pad_id, Path(run_dir), device
        self.held_out, self.logger = list(held_out), logger
        self.clusters = cluster_map(drafter)
        if cfg.cluster_weight and self.clusters is None:
            raise ValueError("cluster_weight > 0 needs a drafter with a clustered output layer")
        self.parameters = [p for p in module.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=cfg.lr, betas=(0.9, 0.999),
                                           weight_decay=cfg.weight_decay)
        self.step = 0
        self.step_seconds: list[float] = []
        self.micro_batch = cfg.micro_batch
        self.halvings = 0
        self.quantiser = module.quantiser if cfg.arm == "a3-oracle" else None

    def hidden(self, batch: RolloutBatch) -> tuple[Tensor, Tensor, Tensor]:
        states = target_states(self.target, batch)
        if self.cfg.arm == "a2":
            return a2_hidden(self.drafter, states, batch, self.cfg.depths)
        return self.module(states, batch)

    def counts(self, examples: Sequence[dict[str, Any]]) -> list[float]:
        """The scored positions per depth (or slot) of a batch, from its lengths alone."""
        batch = collate(examples, self.pad_id)
        positions, valid = query_positions(batch)
        count = self.cfg.depths if self.cfg.arm == "a2" else self.cfg.slots
        _, scored = _targets(batch, positions, valid, count)
        return [float(scored[:, :, k].sum()) for k in range(count)]

    def accumulate(self, examples: Sequence[dict[str, Any]], counts: Sequence[float]) -> dict[str, Any]:
        """The batch's gradient in ``self.micro_batch``-sized parts, added up in the parameters' ``grad``. A3-oracle
        adds its commitment loss and A3 has only its predictor's cross-entropy, each divided by the batch's drafted
        positions (``counts[0]``: every drafted position scores slot 1), so that the parts add up here too."""
        cfg = self.cfg
        self.optimizer.zero_grad(set_to_none=True)
        loss_total, positions, tokens = 0.0, 0, 0
        sums: dict[str, Any] = {"token_nll": [0.0] * len(counts), "cluster_nll": [0.0] * len(counts),
                                "commitment": 0.0, "predictor_nll": 0.0, "predictor_hits": 0.0}
        for s in range(0, len(examples), self.micro_batch):
            batch = collate(examples[s:s + self.micro_batch], self.pad_id).to(self.device)
            if cfg.arm == "a3":
                aux = self.module.code_loss(target_states(self.target, batch), batch)
                loss = aux["predictor"] / counts[0]
                sums["predictor_nll"] += float(aux["predictor"].detach())
                sums["predictor_hits"] += float(aux["predictor_hits"])
                scored_first = int(aux["valid"].sum())
            else:
                out = self.hidden(batch)
                last, targets, scored = out[:3]
                loss, stats = depth_loss(self.drafter, last, targets, scored, self.clusters, cfg.cluster_weight,
                                         cfg.vocab_chunk, counts)
                if len(out) > 3 and "commitment" in out[3]:
                    loss = loss + out[3]["commitment"] / counts[0]
                    sums["commitment"] += float(out[3]["commitment"].detach())
                for key in ("token_nll", "cluster_nll"):
                    sums[key] = [a + b for a, b in zip(sums[key], stats[key], strict=True)]
                scored_first = int(scored[:, :, 0].sum())
                del out, last, targets, scored
            if loss.requires_grad:
                loss.backward()
            loss_total += float(loss.detach())
            positions += scored_first
            tokens += int(batch.valid.sum())
            del loss
        return {"loss": loss_total, "positions": positions, "tokens": tokens, **sums}

    def train_step(self) -> dict[str, float]:
        cfg = self.cfg
        lr = learning_rate(self.step, cfg)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.module.train()
        examples = self.stream.take(cfg.batch)
        counts = self.counts(examples)
        while True:
            try:
                result = self.accumulate(examples, counts)
                break
            except Exception as error:                          # noqa: BLE001 - only out-of-memory is retried
                if not _is_out_of_memory(error) or self.micro_batch == 1:
                    raise
            self.optimizer.zero_grad(set_to_none=True)
            if self.quantiser is not None:
                self.quantiser.discard()                    # the halves count the batch's assignments again
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.micro_batch = max(1, self.micro_batch // 2)
            self.halvings += 1
            print(f"out of memory at step {self.step}: micro-batches of {self.micro_batch} from here on "
                  "(the same gradient)", flush=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, cfg.grad_clip)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite gradient norm at step {self.step}")
        self.optimizer.step()
        self.step += 1
        row: dict[str, float] = {"loss": result["loss"], "lr": lr, "grad_norm": float(grad_norm),
                                 "positions": float(result["positions"]), "tokens": float(result["tokens"]),
                                 "micro_batch": float(self.micro_batch), "halvings": float(self.halvings)}
        if self.quantiser is not None:
            row.update(self.quantiser.apply())
            row["commitment"] = result["commitment"] / max(counts[0], 1.0)
        if cfg.arm == "a3":
            row["predictor_nll"] = result["predictor_nll"] / max(counts[0], 1.0)
            row["predictor_accuracy"] = result["predictor_hits"] / max(counts[0], 1.0)
            return row
        for k, n in enumerate(counts):
            if n:
                row[f"token_nll_{k + 1}"] = result["token_nll"][k] / n
                row[f"cluster_nll_{k + 1}"] = result["cluster_nll"][k] / n
        return row

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """The held-out curves of the module docstring."""
        cfg = self.cfg
        self.module.eval()
        start = time.perf_counter()
        count = cfg.eval_depths if cfg.arm == "a2" else cfg.slots
        hits_clustered, hits_overall, sums = choice_hits(
            self.target, self.drafter, None if cfg.arm == "a2" else self.module, count, self.held_out,
            self.micro_batch, self.pad_id, self.device, cfg.vocab_chunk, self.clusters)
        metrics: dict[str, float] = {"rollouts": float(len(self.held_out))}
        for k in range(count):
            if sums["n"][k]:
                metrics[f"hit_{k + 1}"] = sums["clustered"][k] / sums["n"][k]
                metrics[f"hit_overall_{k + 1}"] = sums["overall"][k] / sums["n"][k]
                metrics[f"token_nll_{k + 1}"] = sums["nll"][k] / sums["n"][k]
        if "code_counts" in sums:
            metrics["code_perplexity"] = code_perplexity(sums["code_counts"])
            metrics["codes_used"] = float((sums["code_counts"] > 0).sum())
        if sums.get("predictor_positions"):
            metrics["predictor_accuracy"] = sums["predictor_hits"] / sums["predictor_positions"]
        lengths = sorted({g for g in EVAL_LENGTHS if g <= count} | (set() if cfg.arm == "a2" else {count}))
        for g in lengths:
            for name, hits in (("", hits_clustered), ("overall_", hits_overall)):
                cycles = tokens = 0
                for sequence in hits:
                    c, t = tokens_per_cycle(sequence, g)
                    cycles, tokens = cycles + c, tokens + t
                if cycles:
                    metrics[f"{name}tokens_per_cycle_{g}"] = tokens / cycles
        metrics["seconds"] = time.perf_counter() - start
        self.module.train()
        return metrics

    # -- persistence ---------------------------------------------------------------------------
    def save(self, name: str = "last.pt") -> Path:
        from chat_vpa.runtime import save_checkpoint

        return save_checkpoint(self.run_dir / name, {
            "stage": f"e1-{self.cfg.arm}", "step": self.step, "config": self.cfg.to_dict(),
            "trainable": self.module.state_dict(), "optimizer": self.optimizer.state_dict(),
            "stream": self.stream.state()})

    def restore(self, payload: dict[str, Any]) -> None:
        self.module.load_state_dict(payload["trainable"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.step = int(payload["step"])

    def fit(self, stop_at: int | None = None) -> None:
        cfg = self.cfg
        end = cfg.steps if stop_at is None else min(stop_at, cfg.steps)
        if self.step == 0 and self.held_out:
            self.log(self.evaluate(), "eval/")
        window: list[dict[str, float]] = []
        while self.step < end:
            begin = time.perf_counter()
            window.append(self.train_step())
            self.step_seconds.append(time.perf_counter() - begin)
            if self.step % cfg.log_every == 0 or self.step == end:
                mean = {k: sum(m.get(k, 0.0) for m in window) / len(window) for k in window[0]}
                rate = len(window) / max(sum(self.step_seconds[-len(window):]), 1e-9)
                extra = {"steps_per_s": rate, "eta_h": (cfg.steps - self.step) / max(rate, 1e-9) / 3600}
                if self.device.type == "cuda":
                    extra["peak_memory_gib"] = torch.cuda.max_memory_allocated(self.device) / 2**30
                self.log({**mean, **extra}, "train/")
                window = []
            if self.step % cfg.save_every == 0 or self.step == end:
                self.save()
            if self.held_out and (self.step % cfg.eval_every == 0 or self.step == cfg.steps):
                self.log(self.evaluate(), "eval/")

    def log(self, metrics: dict[str, float], prefix: str) -> None:
        if self.logger is not None:
            self.logger.log(self.step, metrics, prefix=prefix)


def embedding_rms(target: Any, samples: int = 4_096, seed: int = 0) -> float:
    """The root mean square of the target's input embeddings (as the drafter receives them) over random tokens."""
    embed = target.get_input_embeddings()
    vocab = embed.weight.shape[0]
    ids = torch.randint(0, vocab, (1, min(samples, vocab)), generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        return float(embed(ids.to(embed.weight.device)).float().pow(2).mean().sqrt())


def start_encoder(encoder: BlockEncoder, scale: float, seed: int = 0) -> float:
    """Scales the encoder's output layer so that its encodings of standard normal states have the root mean square
    ``scale`` (the code starts small beside the token embedding it is added to); returns the factor."""
    generator = torch.Generator().manual_seed(seed)
    weight = encoder.inner.weight
    states = torch.randn(1, encoder.slots + 2, encoder.hidden, generator=generator).to(weight.device, weight.dtype)
    with torch.no_grad():
        positions = torch.zeros(1, 1, dtype=torch.long, device=weight.device)
        length = torch.tensor([encoder.slots + 2], device=weight.device)
        rms = float(encoder(states, positions, length).pow(2).mean().sqrt())
        factor = scale / max(rms, 1e-12)
        encoder.outer.weight.mul_(factor)
        encoder.outer.bias.mul_(factor)
    return factor


def build_module(cfg: TrainConfig, target: Any, tokenizer: Any, drafter: Any, source: Path,
                 device: torch.device) -> nn.Module:
    """What an arm trains: A2, the drafter itself; A3-0, the one-pass realiser around it; A3-oracle, the coded
    realiser with a new encoder and codebook; A3, the predictor of the coded realiser that ``source`` holds (A3-oracle's
    final directory: its drafter, depth vectors, encoder and codebook stay as they are)."""
    if cfg.arm == "a2":
        return drafter
    if cfg.arm == "a3-0":
        code = mask_code(target, tokenizer)
        if cfg.slot_input == "add":
            code = torch.zeros_like(code)                         # every slot starts as A2's depth 1
        return OnePassRealiser(drafter, cfg.slots, code, cfg.slot_input).to(device)
    if cfg.arm == "a3-oracle":
        width = target.get_input_embeddings().weight.shape[1]
        hidden = int(target.config.get_text_config().hidden_size)
        dtype = drafter.pre_projection.weight.dtype
        encoder = BlockEncoder(hidden, cfg.slots, width, cfg.code_width).to(device=device, dtype=dtype)
        start_encoder(encoder, cfg.code_scale * embedding_rms(target), cfg.seed)
        quantiser = VectorQuantiser(cfg.codes, width, cfg.codebook_decay, cfg.commitment).to(device)
        return CodedRealiser(drafter, cfg.slots, cfg.slot_input, encoder, quantiser).to(device)
    base = load_realiser(source, drafter) if (Path(source) / "realiser.pt").exists() else None
    if not isinstance(base, CodedRealiser) or base.source != "oracle":
        raise SystemExit(f"A3 trains the predictor of A3-oracle's realiser, and {source} holds none")
    if (base.slots, base.slot_input, base.quantiser.codes) != (cfg.slots, cfg.slot_input, cfg.codes):
        raise SystemExit(f"{source} has {base.slots} slots, slot input {base.slot_input!r} and "
                         f"{base.quantiser.codes} codes; this run asks for {cfg.slots}, {cfg.slot_input!r}, {cfg.codes}")
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    predictor = CodePredictor(one_layer_drafter(drafter), base.quantiser.codes)
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    for parameter in predictor.trainable():
        parameter.requires_grad_(True)
    module = CodedRealiser(drafter, base.slots, base.slot_input, base.encoder, base.quantiser, predictor)
    with torch.no_grad():
        module.depth.copy_(base.depth)
    module.depth.requires_grad_(False)
    module.quantiser.eval()
    return module.to(device)


def save_final(trainer: DrafterTrainer, source: Path, out: Path) -> Path:
    """``RUN/final``: the drafter as a ``Gemma4AssistantForCausalLM`` directory beside the other files of the drafter
    it started from (A3-0: and ``realiser.pt``)."""
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(source, tmp, ignore=shutil.ignore_patterns("*.safetensors", "*.safetensors.index.json", "*.bin",
                                                               ".complete", ".cache", "realiser.pt", "training.json"))
    trainer.drafter.save_pretrained(tmp)
    if isinstance(trainer.module, OnePassRealiser):
        torch.save(realiser_payload(trainer.module), tmp / "realiser.pt")
    with open(tmp / "training.json", "w", encoding="utf-8") as handle:
        json.dump({"arm": trainer.cfg.arm, "step": trainer.step, "config": trainer.cfg.to_dict(),
                   "started_from": str(source)}, handle, indent=2)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


def realiser_payload(module: OnePassRealiser) -> dict[str, Any]:
    """What ``realiser.pt`` holds: A3-0's slots, slot input, code and depth vectors; A3-oracle's and A3's encoder
    and codebook with their sizes, and A3's predictor."""
    out: dict[str, Any] = {"slots": module.slots, "slot_input": module.slot_input,
                           "code": module.code.detach().cpu(), "depth": module.depth.detach().cpu()}
    if isinstance(module, CodedRealiser):
        e, q = module.encoder, module.quantiser
        out["source"] = module.source
        out["encoder"] = {"hidden": e.hidden, "slots": e.slots, "code_dim": e.code_dim, "width": e.width,
                          "state": {k: v.detach().cpu() for k, v in e.state_dict().items()}}
        out["quantiser"] = {"codes": q.codes, "dim": q.dim, "decay": q.decay, "beta": q.beta,
                            "restart_below": q.restart_below,
                            "state": {k: v.detach().cpu() for k, v in q.state_dict().items()}}
        if module.predictor is not None:
            out["predictor"] = {"codes": q.codes, "state": module.predictor.trained_state()}
    return out


def load_realiser(directory: Path, drafter: Any) -> OnePassRealiser:
    """A3's realiser of a final directory (``save_final``) around its drafter, loaded already: A3-0's (a run saved
    before ``slot_input`` existed is ``replace``), or A3-oracle's or A3's with their encoder, codebook and
    predictor."""
    saved = torch.load(Path(directory) / "realiser.pt", map_location="cpu", weights_only=True)
    device, dtype = drafter.pre_projection.weight.device, drafter.pre_projection.weight.dtype
    slot_input = str(saved.get("slot_input", "replace"))
    if "encoder" in saved:
        e, q = saved["encoder"], saved["quantiser"]
        encoder = BlockEncoder(int(e["hidden"]), int(e["slots"]), int(e["code_dim"]), int(e["width"]))
        encoder.load_state_dict(e["state"])
        quantiser = VectorQuantiser(int(q["codes"]), int(q["dim"]), float(q["decay"]), float(q["beta"]),
                                    float(q["restart_below"]))
        quantiser.load_state_dict(q["state"])
        predictor = None
        if "predictor" in saved:
            predictor = CodePredictor(one_layer_drafter(drafter), int(saved["predictor"]["codes"]))
            predictor.load_trained(saved["predictor"]["state"])
        realiser: OnePassRealiser = CodedRealiser(drafter, int(saved["slots"]), slot_input,
                                                  encoder.to(device=device, dtype=dtype),
                                                  quantiser.to(device), predictor)
    else:
        realiser = OnePassRealiser(drafter, int(saved["slots"]), saved["code"], slot_input)
    with torch.no_grad():
        realiser.depth.copy_(saved["depth"].to(realiser.depth.dtype))
    return realiser.to(device)


def timing(trainer: DrafterTrainer, skip: int = 10) -> dict[str, Any]:
    """Seconds per step after the first ``skip`` (measured) and the hours of the run's whole horizon
    (*predicted*)."""
    seconds = trainer.step_seconds[skip:] if len(trainer.step_seconds) > skip else trainer.step_seconds
    per_step = sum(seconds) / len(seconds) if seconds else None
    return {"steps_run": len(trainer.step_seconds), "steps_timed": len(seconds), "seconds_per_step": per_step,
            "horizon_steps": trainer.cfg.steps,
            "predicted_hours": None if per_step is None else per_step * trainer.cfg.steps / 3600}


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------

def _overrides(cfg: TrainConfig, pairs: Sequence[str]) -> TrainConfig:
    data = cfg.to_dict()
    for pair in pairs:
        key, _, value = pair.partition("=")
        if key not in data:
            raise SystemExit(f"unknown setting {key!r}; known: {', '.join(sorted(data))}")
        kind = type(data[key])
        data[key] = value if kind is str else kind(float(value)) if kind is int else kind(value)
    return TrainConfig.from_dict(data)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 Step 1: train A2 or A3-0", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--model", type=Path, required=True, help="the instruction-tuned target")
    p.add_argument("--assistant", type=Path, required=True, help="the drafter to start from (A1, or A2 for A3-0)")
    p.add_argument("--rollouts", type=Path, required=True, help="the training rollouts (e1_rollouts' chunks)")
    p.add_argument("--limit", type=int, default=None, help="use the first N training rollouts only")
    p.add_argument("--held-out", type=Path, default=None, help="the held-out rollouts, for the curves")
    p.add_argument("--held-out-limit", type=int, default=None)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stop-at", type=int, default=None, help="end at this step (the timing run: 100)")
    p.add_argument("--device", default="auto")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override a TrainConfig field")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from transformers import Gemma4AssistantForCausalLM

    from chat_vpa.device import resolve_device
    from chat_vpa.e1_common import environment, load_target
    from chat_vpa.e1_rollouts import read_rollouts
    from chat_vpa.runtime import MetricLogger, load_checkpoint
    from chat_vpa.train_e0 import EpochStream

    args = parse_args(argv)
    device = resolve_device(args.device)
    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    records, dropped = prepare(read_rollouts(args.rollouts, args.limit))
    if not records:
        raise SystemExit(f"no usable rollout under {args.rollouts}")
    chunks = sorted(p for p in args.rollouts.glob("chunk-*.jsonl") if p.with_suffix(".json").exists())
    cfg = _overrides(TrainConfig(arm=args.arm), args.set).finish(len(records))
    if cfg.arm != args.arm:
        raise SystemExit("the arm is set by --arm, not by --set")
    held_out: list[dict[str, Any]] = []
    if args.held_out is not None:
        held_out, _ = prepare(read_rollouts(args.held_out, args.held_out_limit))
    payload = load_checkpoint(run / "last.pt", map_location="cpu") if (run / "last.pt").exists() else None
    if payload is not None:
        saved = TrainConfig.from_dict(payload["config"])
        if saved != cfg:
            raise SystemExit(f"{run} holds a run with another configuration: {saved} against {cfg}")
    target, tokenizer = load_target(args.model, "bf16", device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    drafter = Gemma4AssistantForCausalLM.from_pretrained(str(args.assistant), dtype=torch.float32).to(device)
    module = build_module(cfg, target, tokenizer, drafter, args.assistant, device)
    stream = EpochStream(records, cfg.seed, payload["stream"] if payload else None)
    logger = MetricLogger(run, enabled=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    trainer = DrafterTrainer(cfg, target, module, drafter, stream, pad_id, run, device, held_out, logger)
    if payload is not None:
        trainer.restore(payload)
        print(f"resumed {run} at step {trainer.step}", flush=True)
    else:
        trained = sum(p.numel() for p in trainer.parameters)
        with open(run / "config.json", "w", encoding="utf-8") as handle:
            json.dump({"config": cfg.to_dict(), "model": str(args.model), "assistant": str(args.assistant),
                       "rollouts": str(args.rollouts), "rollouts_used": len(records), "rollouts_dropped": dropped,
                       "rollouts_fingerprint": fingerprint(chunks), "held_out": str(args.held_out),
                       "held_out_used": len(held_out), "trained_parameters": trained,
                       "environment": environment(device, "bf16 target, fp32 drafter")}, handle, indent=2)
        print(f"{args.arm}: {trained / 1e6:.1f}M trained parameters, {len(records)} rollouts ({dropped} dropped), "
              f"{cfg.steps} steps of {cfg.batch}", flush=True)
    trainer.fit(args.stop_at)
    summary = timing(trainer)
    with open(run / "timing.json", "w", encoding="utf-8") as handle:
        json.dump({**summary, "environment": environment(device, "bf16 target, fp32 drafter"),
                   "what": "seconds per step measured; predicted_hours = seconds per step x horizon steps"},
                  handle, indent=2)
    print(f"step {trainer.step} of {cfg.steps}; {summary['seconds_per_step'] or 0:.3f} s per step measured; "
          f"{summary['predicted_hours'] or 0:.1f} h for {cfg.steps} steps predicted", flush=True)
    if trainer.step == cfg.steps:
        print(f"saved {save_final(trainer, args.assistant, run / 'final')}", flush=True)
    logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
