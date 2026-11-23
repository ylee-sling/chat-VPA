# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
block_masks.py -- attention masks of Block-DFM (Chat-VPA Sec. 3.3, Eq. 5; Sec. 3.5, Algorithm 1).

Rows of a pass are of three kinds:

    clean / committed   a token of y = (c; w) at its absolute position, read causally (the frozen backbone in E0)
    window              the active window of block m: the anchor at position a_m, B draft slots at
                        a_m + 1 .. a_m + B and R registers, which carry the anchor's position (Sec. 3.2)
    cache               keys and values of committed positions computed earlier (never queries)

Eq. 5, with n(u) the position of row u and W_m the window rows of block m:

    M_uv = 1   u clean,  v clean, n(v) <= n(u)          causal
    M_uv = 1   u in W_m, v clean, n(v) <  a_m            staircase
    M_uv = 1   u in W_m, v in W_m                        bidirectional window
    M_uv = 0   otherwise

Clean rows never read a window and windows never read one another. In this code the clean rows of a training
pass are computed first, by ``Stage0Realizer.prefill`` (its masks are ``backbone.prefill_masks``, the causal
case of Eq. 5), and their keys and values are the cache that the window rows read (Prop. 1(b)); so the masks
built here have window rows, and for a fused pass commit rows, as queries.

Local blocks. The backbone's sliding window applies on top of M: a key is visible in a ``sliding_attention``
block only at distance < W. For committed keys that is the backbone's own causal window; inside a window the
rule is two-sided, |n(u) - n(v)| < W, which never binds for B + 1 <= W (the paper's B + 1 <= 33 against
W = 512) and makes a single block of B = N the whole-canvas realizer of ``backbone.canvas_masks``.
Global blocks use M as it is.

All masks are bool, True = may attend, shape ``[B, 1, Q, KV]`` per layer type, like ``backbone.canvas_masks``.
Positions are 0-based absolute indices. Shapes: batch B, cache length Lc (left-padded), C commit rows, block size
Bk, R registers, Wn = 1 + Bk + R window rows, M blocks.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["fused_masks", "staircase_masks", "window_positions"]


def window_positions(anchor_pos: Tensor, block_size: int, num_registers: int) -> Tensor:
    """Positions of the rows [anchor; Bk slots; R registers] of a window: a, a + 1 .. a + Bk, then a for every
    register. ``anchor_pos`` ``[...]`` -> ``[..., 1 + Bk + R]``."""
    if block_size < 1 or num_registers < 0:
        raise ValueError("need block_size >= 1 and num_registers >= 0")
    offsets = torch.cat([torch.arange(block_size + 1), torch.zeros(num_registers, dtype=torch.long)])
    return anchor_pos[..., None] + offsets.to(anchor_pos.device)


def fused_masks(cache_pos: Tensor, cache_valid: Tensor, commit_pos: Tensor, window_pos: Tensor, window: int,
                causal: bool = False) -> dict[str, Tensor]:
    """Masks of one sampling pass: the rows [commit; window] over the keys [cache; commit; window].

    Commit rows (the anchor of the previous block and its first Bk - 1 tokens, Algorithm 1 line 19) are causal:
    every valid cache key and the commit keys not after them; never a window key. Window rows see every valid
    cache key, every commit key and every window key. With no commit row (``commit_pos`` of width 0) this is
    the mask of a window pass.

    ``causal`` replaces the bidirectional window by the backbone's causal mask in row order (a window without
    registers only): with the adapters off a window pass is then the committed stream itself (Sec. 3.2).

    Shapes: cache ``[B, Lc]``, commit ``[B, C]``, window ``[B, Wn]`` -> ``[B, 1, C + Wn, Lc + C + Wn]``.
    """
    b, c = commit_pos.shape
    wn = window_pos.shape[1]
    lc = cache_pos.shape[1]
    device = cache_pos.device
    row_pos = torch.cat([commit_pos, window_pos], dim=1)                                  # [B, C + Wn]
    key_pos = torch.cat([cache_pos, commit_pos, window_pos], dim=1)                       # [B, Lc + C + Wn]
    allow = torch.zeros(b, c + wn, lc + c + wn, dtype=torch.bool, device=device)
    allow[:, :, :lc] = cache_valid[:, None, :]                                            # the cache: every row
    index = torch.arange(c, device=device)
    allow[:, :c, lc:lc + c] = (index[None, :] <= index[:, None])[None]                    # commit rows: causal
    allow[:, c:, lc:lc + c] = True                                                        # window rows: all of G
    if causal:
        order = torch.arange(wn, device=device)
        allow[:, c:, lc + c:] = (order[None, :] <= order[:, None])[None]
    else:
        allow[:, c:, lc + c:] = True
    near = (row_pos[:, :, None] - key_pos[:, None, :]).abs() < window
    return {"full_attention": allow[:, None], "sliding_attention": (allow & near)[:, None]}


def staircase_masks(cache_pos: Tensor, cache_valid: Tensor, anchor_pos: Tensor, block_valid: Tensor,
                    block_size: int, num_registers: int, window: int
                    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """The window rows of a training pass (Eq. 5): all M windows of every example over [cache; window rows].

    The cache holds the clean stream's keys and values. Window m sees the valid cache keys at positions before
    its anchor (the staircase) and its own rows in both directions; a row of a block that does not exist for
    its example (``block_valid`` False: padding of the batch) sees itself alone, and its output is not used.

    Args:
        cache_pos, cache_valid: ``[B, Lc]`` positions and validity of the clean keys.
        anchor_pos: ``[B, M]`` position a_m of each block's anchor.
        block_valid: ``[B, M]`` bool.

    Returns:
        the masks ``[B, 1, M Wn, Lc + M Wn]``, the positions ``[B, M Wn]`` of the window rows and their block
        index ``[B, M Wn]`` (rows are ordered block by block: anchor, slots, registers).
    """
    b, m = anchor_pos.shape
    device = cache_pos.device
    row_pos = window_positions(anchor_pos, block_size, num_registers).reshape(b, -1)      # [B, M Wn]
    wn = row_pos.shape[1] // m
    block_of = torch.arange(m, device=device).repeat_interleave(wn)[None].expand(b, -1)   # [B, M Wn]
    row_anchor = anchor_pos.repeat_interleave(wn, dim=1)                                  # [B, M Wn]
    row_valid = block_valid.repeat_interleave(wn, dim=1)
    to_cache = cache_valid[:, None, :] & (cache_pos[:, None, :] < row_anchor[:, :, None]) & row_valid[:, :, None]
    same_block = block_of[:, :, None] == block_of[:, None, :]
    to_rows = same_block & row_valid[:, :, None] & row_valid[:, None, :]
    eye = torch.eye(m * wn, dtype=torch.bool, device=device)[None]
    to_rows = torch.where(row_valid[:, :, None], to_rows, eye)                            # padding: itself only
    allow = torch.cat([to_cache, to_rows], dim=2)
    key_pos = torch.cat([cache_pos, row_pos], dim=1)
    near = (row_pos[:, :, None] - key_pos[:, None, :]).abs() < window                     # a row is near itself
    return ({"full_attention": allow[:, None], "sliding_attention": (allow & near)[:, None]}, row_pos, block_of)
