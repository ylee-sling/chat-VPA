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

**A3-0 (D4, D6, D8).** One call of the drafter over B = 8 slots per position q: slot j (1 to B) takes
``code + depth[j]`` in place of the token embedding, concatenated with the target's last hidden state at q - 1, at
position q, under the same visibility; the slots cannot attend to each other (the drafter's layers have no keys
of their own). ``code`` is one learned vector (A3-0 has no latent; A3-oracle and A3 put a codebook's vector in its
place), initialised to the target's input embedding of the tokenizer's mask token (zero if it has none), as
text-VPA's mask vector was; ``depth`` holds one learned vector per slot, zero at the start. Slot j is scored on
x_{q+j} by A2's per-position loss, with ``cluster_weight`` 0 by default (D8 names the token's cross-entropy only);
the slots weigh equally. Its weights start from ``--assistant`` (D4: A2's, or A1's if A2 accepts fewer tokens).

**The target (S5, S8)**: frozen, bfloat16, without gradients. One pass of its text model over a batch gives every
position's last hidden state and the keys and values the drafter attends to (recomputed in every step, not
stored). The drafter's weights are float32. The vocabulary-wide cross-entropy runs over chunks of rows, each
chunk's logits recomputed in the backward pass, so the logits of a batch never exist at once.

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
continuation and adds the next token; the first token comes from the prefill), at G in {2, 4, 6, 8} for A2 and
G = B for A3-0. They are counts along the continuations with the training pass's keys, not decode measurements.
"""

from __future__ import annotations

import argparse
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

__all__ = ["ARMS", "OnePassRealiser", "RolloutBatch", "TrainConfig", "a2_hidden", "attention_masks", "cluster_map",
           "collate", "depth_loss", "drafter_choices", "drafter_hidden", "prepare", "query_positions",
           "target_states", "tokens_per_cycle", "visibility", "vocab_cross_entropy"]

ARMS = ("a2", "a3-0")
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
    ``code + depth[j]`` in place of the token embedding. A3-0's ``code`` is one learned vector."""

    def __init__(self, drafter: Any, slots: int, code: Tensor) -> None:
        super().__init__()
        if slots < 1:
            raise ValueError("need at least one slot")
        self.drafter = drafter
        self.slots = slots
        dtype = drafter.pre_projection.weight.dtype
        self.code = nn.Parameter(code.detach().to(dtype).clone())
        self.depth = nn.Parameter(torch.zeros(slots, code.shape[-1], dtype=dtype))

    def forward(self, states: tuple[Tensor, dict[str, tuple[Tensor, Tensor]], Tensor], batch: RolloutBatch
                ) -> tuple[Tensor, Tensor, Tensor]:
        """The last hidden state of every slot ``[B, Q, slots, d]``, the targets ``[B, Q, slots]`` and which are
        scored."""
        hidden, kv, _ = _drafter_inputs(self.drafter, states)
        positions, valid = query_positions(batch)
        rows, width = positions.shape
        previous = _gather(hidden, positions - 1)                     # [B, Q, D]
        slot = (self.code + self.depth).to(previous.dtype)            # [S, D]
        inputs = torch.cat([slot.expand(rows, width, -1, -1),
                            previous[:, :, None].expand(-1, -1, self.slots, -1)], dim=-1)
        flat = positions.repeat_interleave(self.slots, dim=1)         # [B, Q * S], slot-minor
        masks = _masks_for(self.drafter, flat, hidden.shape[1])
        last = drafter_hidden(self.drafter, inputs.reshape(rows, width * self.slots, -1), flat, kv, masks)
        targets, scored = _targets(batch, positions, valid, self.slots)
        return last.view(rows, width, self.slots, -1), targets, scored


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
               cluster_weight: float, chunk: int) -> tuple[Tensor, dict[str, list[float]]]:
    """The mean over depths (or slots) that have a scored position of (the token's cross-entropy over the
    vocabulary + ``cluster_weight`` x its cluster's cross-entropy), each averaged over its scored positions; per
    depth, the sums and the count."""
    if cluster_weight and clusters is None:
        raise ValueError("a cluster loss needs a clustered output layer")
    weight = drafter.lm_head.weight
    losses: list[Tensor] = []
    stats: dict[str, list[float]] = {"count": [], "token_nll": [], "cluster_nll": []}
    for k in range(last.shape[2]):
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
            losses.append((token + cluster_weight * cluster) / n)
    if not losses:
        raise ValueError("no scored position in the batch")
    return torch.stack(losses).mean(), stats


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
    cluster_weight: float = -1.0           # -1: the arm's own (A2 1.0, S7; A3-0 0.0, D8)
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
            self.cluster_weight = 1.0 if self.arm == "a2" else 0.0
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

    def hidden(self, batch: RolloutBatch) -> tuple[Tensor, Tensor, Tensor]:
        states = target_states(self.target, batch)
        if self.cfg.arm == "a2":
            return a2_hidden(self.drafter, states, batch, self.cfg.depths)
        return self.module(states, batch)

    def train_step(self) -> dict[str, float]:
        cfg = self.cfg
        lr = learning_rate(self.step, cfg)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.module.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = collate(self.stream.take(cfg.batch), self.pad_id).to(self.device)
        last, targets, scored = self.hidden(batch)
        loss, stats = depth_loss(self.drafter, last, targets, scored, self.clusters, cfg.cluster_weight,
                                 cfg.vocab_chunk)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, cfg.grad_clip)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite gradient norm at step {self.step}")
        self.optimizer.step()
        self.step += 1
        row: dict[str, float] = {"loss": float(loss.detach()), "lr": lr, "grad_norm": float(grad_norm),
                                 "positions": float(int(scored[:, :, 0].sum())),
                                 "tokens": float(int(batch.valid.sum()))}
        for k, n in enumerate(stats["count"]):
            if n:
                row[f"token_nll_{k + 1}"] = stats["token_nll"][k] / n
                row[f"cluster_nll_{k + 1}"] = stats["cluster_nll"][k] / n
        return row

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """The held-out curves of the module docstring."""
        cfg = self.cfg
        self.module.eval()
        start = time.perf_counter()
        count = cfg.eval_depths if cfg.arm == "a2" else cfg.slots
        hits_clustered: list[list[list[bool]]] = []
        hits_overall: list[list[list[bool]]] = []
        sums = {"n": [0.0] * count, "clustered": [0.0] * count, "overall": [0.0] * count, "nll": [0.0] * count}
        for s in range(0, len(self.held_out), cfg.batch):
            batch = collate(self.held_out[s:s + cfg.batch], self.pad_id).to(self.device)
            states = target_states(self.target, batch)
            if cfg.arm == "a2":
                last, targets, scored = a2_hidden(self.drafter, states, batch, count)
            else:
                last, targets, scored = self.module(states, batch)
            rows, width = scored.shape[:2]
            hit_c = torch.zeros(rows, width, count, dtype=torch.bool, device=self.device)
            hit_o = torch.zeros_like(hit_c)
            for k in range(count):
                chosen = scored[:, :, k]
                clustered, overall, nll = drafter_choices(self.drafter, last[:, :, k][chosen],
                                                          targets[:, :, k][chosen], cfg.vocab_chunk, self.clusters)
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
        metrics: dict[str, float] = {"rollouts": float(len(self.held_out))}
        for k in range(count):
            if sums["n"][k]:
                metrics[f"hit_{k + 1}"] = sums["clustered"][k] / sums["n"][k]
                metrics[f"hit_overall_{k + 1}"] = sums["overall"][k] / sums["n"][k]
                metrics[f"token_nll_{k + 1}"] = sums["nll"][k] / sums["n"][k]
        lengths = [g for g in EVAL_LENGTHS if g <= count] if cfg.arm == "a2" else [cfg.slots]
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


def save_final(trainer: DrafterTrainer, source: Path, out: Path) -> Path:
    """``RUN/final``: the drafter as a ``Gemma4AssistantForCausalLM`` directory beside the other files of the drafter
    it started from (A3-0: and ``realiser.pt``)."""
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(source, tmp, ignore=shutil.ignore_patterns("*.safetensors", "*.safetensors.index.json", "*.bin",
                                                               ".complete", ".cache", "realiser.pt", "training.json"))
    trainer.drafter.save_pretrained(tmp)
    if isinstance(trainer.module, OnePassRealiser):
        torch.save({"slots": trainer.module.slots, "code": trainer.module.code.detach().cpu(),
                    "depth": trainer.module.depth.detach().cpu()}, tmp / "realiser.pt")
    with open(tmp / "training.json", "w", encoding="utf-8") as handle:
        json.dump({"arm": trainer.cfg.arm, "step": trainer.step, "config": trainer.cfg.to_dict(),
                   "started_from": str(source)}, handle, indent=2)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


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
    if args.arm == "a2":
        module: nn.Module = drafter
    else:
        module = OnePassRealiser(drafter, cfg.slots, mask_code(target, tokenizer)).to(device)
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
