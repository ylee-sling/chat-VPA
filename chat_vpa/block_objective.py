# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
block_objective.py -- the block loss of discrete flow matching on a batch of responses (Chat-VPA Eq. 12).

    L_DFM = (1 / Omega) sum over masked slots i of omega_i ( -log p_theta^i(w^i) ),   Omega = sum of the omega_i

over all blocks of the micro-batch, with p_theta^i the posterior of slot i read through the AR shift from one
staircase training pass (``BlockDFM.training_hidden``) and omega the weights of Sec. 4.3
(``block_data.block_training_state``): the masked cross-entropy on the target's tokens, without an ELBO weight,
as in the predecessor.

Not here: L_Block-KD (it needs a teacher's stored distributions; its slots are ``BlockState.right_open``),
L_Boundary, L_Step and L_Plan. L_AR (Eq. 14) trains the clean stream, which is the frozen backbone in E0.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from chat_vpa.block_data import BlockBatch, BlockState
from chat_vpa.block_dfm import BlockDFM

__all__ = ["dfm_loss"]


def dfm_loss(model: BlockDFM, batch: BlockBatch, state: BlockState, device: torch.device,
             checkpointing: bool = False, adapters: bool = True, autocast: Any = None
             ) -> tuple[Tensor, dict[str, float]]:
    """L_DFM of Eq. 12 on one micro-batch, and its statistics as weighted sums (for pooling over micro-batches).

    Args:
        batch, state: on the CPU (``block_data``); moved to ``device`` here.
        checkpointing: activation checkpointing of the window pass.
        adapters: False evaluates the frozen backbone on the same states.
        autocast: a context manager for the network's passes (bf16 on the GPU), or None.

    Returns:
        the loss (a scalar with gradients) and {"nll_sum", "weight_sum", "hits", "masked"}; "masked" counts
        the slots that carry loss, "hits" the ones whose most probable token is the target.
    """

    def forward() -> tuple[Tensor, Tensor, Tensor, Tensor]:
        hidden = model.training_hidden(
            batch.clean_rows, batch.anchor.to(device), batch.anchor_pos.to(device), batch.block_valid.to(device),
            state.slots.to(device), state.is_mask.to(device), state.t.to(device), state.h.to(device),
            checkpointing=checkpointing, adapters=adapters)
        weights = state.weights.to(device)
        chosen = weights > 0
        nll, hits = model.realizer.weighted_nll(hidden[chosen], batch.target.to(device)[chosen], weights[chosen])
        return nll, hits, weights.sum(), chosen.sum()

    if autocast is None:
        nll, hits, weight_sum, masked = forward()
    else:
        with autocast:
            nll, hits, weight_sum, masked = forward()
    total = weight_sum.clamp_min(1e-8)
    stats = {"nll_sum": float(nll.detach()), "weight_sum": float(weight_sum), "hits": float(hits),
             "masked": float(masked)}
    return nll / total, stats
