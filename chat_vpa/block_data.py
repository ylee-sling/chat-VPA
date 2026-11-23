# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
block_data.py -- responses cut into blocks, the training states of each block and their loss weights
(Chat-VPA Sec. 3 "Notation", Sec. 3.3 "Training states", Sec. 4.2 Eq. 11, Sec. 4.3 "Seam weights").

A response is an example's canvas (``data.Example.canvas_ids``: the program's tokens and one <eos>) followed by
<eos> up to a multiple of the block size, N = M B. Block m (0-based here) has its anchor at position
a_m = P - 1 + m B (the last prompt token for m = 0, otherwise the last token of block m - 1) and its B tokens at
a_m + 1 .. a_m + B. Blocks after the one that holds the first <eos> do not exist: the sampler stops there.

Training state of a block (each block draws its own): a budget K from the stage's set, k uniform on
0 .. K - 1, t = k/K, h = 1/K; with probability ``prefix_fraction`` (rho_pre) the draft is prefix-shaped, its
first ceil(kappa_t B) slots revealed and the rest masked, otherwise every slot is revealed independently with
probability kappa_t (the state distribution of the exact kernel, Eq. 3).

Loss weights omega of the masked slots (Sec. 4.3): 1 + ``seam_weight`` (lambda_seam) on the first
``seam_slots`` (b_0) slots of every block, ``pad_weight`` (0.1) on the <eos> filling after the first <eos>, 1
elsewhere; a slot that is both among the first b_0 and filling gets ``pad_weight`` (decision in CLAUDE.md).
Revealed slots and blocks that do not exist get 0.

The right-open set O_t of Eq. 11 (a masked slot with every slot of its block to its right masked) is where an
AR teacher's next-token distribution is an unbiased soft target (Prop. 5). L_DFM does not use it; it is computed
here for L_Block-KD.

Everything is built on the CPU. Shapes: B examples, M blocks (the largest count in the batch), Bk block size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from chat_vpa.data import Example
from chat_vpa.flow import KappaSchedule

__all__ = ["BlockBatch", "BlockState", "block_training_state", "collate_blocks", "prefix_count", "right_open"]


@dataclass
class BlockBatch:
    clean_rows: list[list[int]]    # per example y^1 .. y^{a_M - 1}: prompt, then the response before the last anchor
    anchor: Tensor                 # [B, M] anchor token of each block
    anchor_pos: Tensor             # [B, M] position a_m of each anchor (0 where the block does not exist)
    target: Tensor                 # [B, M, Bk] tokens of each block, <eos> after the program
    block_valid: Tensor            # [B, M] bool
    length: Tensor                 # [B] program tokens + 1 (the first <eos>)
    kinds: list[str]


def collate_blocks(examples: Sequence[Example], block_size: int, eos_id: int) -> BlockBatch:
    """Examples -> blocks. An example of ``length`` canvas tokens (its <eos> included) has ceil(length / Bk)
    blocks; shorter examples are padded with blocks marked invalid."""
    if block_size < 1 or not examples:
        raise ValueError("need block_size >= 1 and at least one example")
    counts = [math.ceil(len(ex.canvas_ids) / block_size) for ex in examples]
    b, m = len(examples), max(counts)
    anchor = torch.full((b, m), eos_id, dtype=torch.long)
    anchor_pos = torch.zeros(b, m, dtype=torch.long)
    target = torch.full((b, m, block_size), eos_id, dtype=torch.long)
    valid = torch.zeros(b, m, dtype=torch.bool)
    clean_rows: list[list[int]] = []
    for row, (ex, blocks) in enumerate(zip(examples, counts, strict=True)):
        if len(ex.prompt_ids) < 2 or not ex.canvas_ids or ex.canvas_ids[-1] != eos_id:
            raise ValueError("an example needs a prompt of at least two tokens and a canvas that ends in <eos>")
        response = [*ex.canvas_ids, *([eos_id] * (blocks * block_size - len(ex.canvas_ids)))]
        sequence = [*ex.prompt_ids, *response]                                     # y, 0-based positions
        first = len(ex.prompt_ids) - 1                                             # a_0: the last prompt token
        for block in range(blocks):
            position = first + block * block_size
            anchor[row, block], anchor_pos[row, block], valid[row, block] = sequence[position], position, True
            target[row, block] = torch.tensor(sequence[position + 1: position + 1 + block_size], dtype=torch.long)
        clean_rows.append(sequence[: first + (blocks - 1) * block_size])
    return BlockBatch(clean_rows, anchor, anchor_pos, target, valid,
                      torch.tensor([len(ex.canvas_ids) for ex in examples], dtype=torch.long),
                      [ex.kind for ex in examples])


@dataclass
class BlockState:
    slots: Tensor                  # [B, M, Bk] the draft x_t of each block: its token where revealed, else [MASK]
    is_mask: Tensor                # [B, M, Bk] bool
    t: Tensor                      # [B, M] float32
    h: Tensor                      # [B, M] float32
    budget: Tensor                 # [B, M] int64 K
    step: Tensor                   # [B, M] int64 k
    prefix: Tensor                 # [B, M] bool: the draft is prefix-shaped
    weights: Tensor                # [B, M, Bk] float32 loss weight omega of each masked slot, else 0
    right_open: Tensor             # [B, M, Bk] bool: the set O_t of Eq. 11


def prefix_count(kappa: Tensor, block_size: int) -> Tensor:
    """ceil(kappa_t B), the revealed slots of a prefix-shaped state (Sec. 3.3). A product that is an integer up to
    rounding (the linear schedule's k B / K) is not pushed to the next integer. float64 in, int64 out."""
    return torch.ceil(kappa.double() * block_size - 1e-9).long().clamp(0, block_size)


def right_open(is_mask: Tensor) -> Tensor:
    """O_t (Eq. 11): masked slots with no revealed slot of their block to their right. ``[..., Bk]`` bool."""
    return is_mask.flip(-1).long().cumprod(dim=-1).flip(-1).bool()


def block_training_state(batch: BlockBatch, schedule: KappaSchedule, step_budgets: Sequence[int], mask_id: int,
                         prefix_fraction: float = 0.5, pad_weight: float = 0.1, seam_slots: int = 4,
                         seam_weight: float = 1.0, generator: torch.Generator | None = None,
                         budget: int | None = None, step_index: int | None = None, prefix: bool | None = None
                         ) -> BlockState:
    """A training state for every block of ``batch`` (see the module docstring).

    Draws, in this order: the budget index ``[B, M]``, the step ``[B, M]``, the prefix coin ``[B, M]``, the
    reveal uniforms ``[B, M, Bk]``. ``budget``, ``step_index`` and ``prefix`` pin K, k and the shape of every
    block (validation states); the corresponding draws are still made, so pinned and drawn states of one
    generator differ only in what is pinned.
    """
    if not 0.0 <= prefix_fraction <= 1.0:
        raise ValueError("prefix_fraction must lie in [0, 1]")
    if seam_slots < 0 or seam_weight < 0 or pad_weight < 0:
        raise ValueError("seam_slots, seam_weight and pad_weight must be >= 0")
    budgets = torch.tensor([int(k) for k in step_budgets], dtype=torch.long)
    if budgets.numel() == 0 or bool((budgets < 1).any()):
        raise ValueError("step_budgets must be positive")
    target = batch.target
    b, m, bk = target.shape
    big_k = budgets[torch.randint(len(budgets), (b, m), generator=generator)]
    step_draw = torch.rand(b, m, generator=generator)
    prefix_draw = torch.rand(b, m, generator=generator)
    reveal_draw = torch.rand(b, m, bk, generator=generator)
    if budget is not None:
        big_k = torch.full((b, m), int(budget), dtype=torch.long)
    step = torch.minimum((step_draw.double() * big_k.double()).long(), big_k - 1)
    if step_index is not None:
        if not bool((step_index < big_k).all()) or step_index < 0:
            raise ValueError("need 0 <= step_index < budget")
        step = torch.full((b, m), int(step_index), dtype=torch.long)
    t = step.double() / big_k.double()
    h = 1.0 / big_k.double()
    kappa = schedule.kappa(t)                                                       # [B, M] float64
    is_prefix = prefix_draw < prefix_fraction if prefix is None else torch.full((b, m), bool(prefix))
    slot = torch.arange(bk)[None, None, :]
    revealed_prefix = slot < prefix_count(kappa, bk)[:, :, None]
    revealed_random = reveal_draw.double() < kappa[:, :, None]
    revealed = torch.where(is_prefix[:, :, None], revealed_prefix, revealed_random)
    is_mask = ~revealed
    slots = torch.where(is_mask, torch.full_like(target, mask_id), target)
    in_response = torch.arange(m)[None, :, None] * bk + slot                        # [1, M, Bk] index of w^j, 0-based
    filling = in_response >= batch.length[:, None, None]                            # after the first <eos>
    omega = torch.where(slot < seam_slots, 1.0 + seam_weight, 1.0).expand(b, m, bk)
    omega = torch.where(filling, torch.full_like(omega, pad_weight), omega)
    weights = omega * is_mask * batch.block_valid[:, :, None]
    return BlockState(slots, is_mask, t.float(), h.float(), big_k, step, is_prefix, weights.float(),
                      right_open(is_mask))
