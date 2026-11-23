# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
block_dfm.py -- Block-DFM on the frozen backbone: one network, two streams (Chat-VPA Secs. 3.2, 3.3 and 3.5).

    committed stream   prompt and finished blocks, read causally by the frozen backbone, exactly as released:
                       no adapter, no (t, h) modulation, no mask vector. Its KV-storing blocks write the cache.
    active window      block m: the anchor (the last finished token, clean and held fixed), B draft slots and
                       R registers, bidirectional among themselves and reading the cache; the realizer's
                       adapters, mask vector and (t, h) modulation act on these rows only. The AR shift is kept:
                       the posterior of slot j is read from the output at slot j - 1, the first slot's from the
                       anchor's; the outputs of the last slot and of the registers are not read.

Layout of one sample (0-based absolute positions; the prompt is c_1 .. c_P, the response w^1 .. w^N, N = M B):

    cache before block m    positions 0 .. a_m - 1        (prompt without its last token, then earlier blocks)
    window of block m       anchor at a_m = P - 1 + (m - 1) B, slots at a_m + 1 .. a_m + B, registers at a_m
    commit rows of block m  positions a_m .. a_m + B - 1  (its anchor and its first B - 1 tokens); the block's
                            last token is the anchor of block m + 1 (Algorithm 1, line 19)

Passes (each is one evaluation of the network; ``PassAudit`` counts them with hooks):

    prefill          the frozen backbone on the prompt without its last token, KV-storing blocks only
    fused_pass       pass 1 of a block (Algorithm 1, line 7): the commit rows of the previous block run
                     causally through the KV-storing blocks only and their keys and values are appended to the
                     cache; in the same pass the window runs through all blocks and reads [cache; commit rows;
                     window] (Prop. 2)
    window_pass      passes 2 .. K (line 9): the window alone
    commit_pass      a separate causal pass over commit rows (ablation A10; the reference of the fused pass)
    training_hidden  the window rows of all blocks of a response at once under the staircase mask (Eq. 5),
                     after a prefill of the clean sequence: by Prop. 1 every window computes what the sampler's
                     passes compute for it

In E0 the backbone is frozen, so the clean rows of a training pass are the prefill itself, their outputs are not
needed and L_AR (Eq. 14) has nothing to train.

Registers (Sec. 3.4) are R learned input vectors per window. They carry the anchor's position, are attended to
by the window, and are never among the commit rows, so they never reach the cache.

Shapes: B batch, Lc cache length, Bk block size, R registers, Wn = 1 + Bk + R window rows, M blocks, d hidden size.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from chat_vpa.backbone import PrefixCache, Stage0Realizer, left_pad
from chat_vpa.block_masks import fused_masks, staircase_masks, window_positions

__all__ = ["REGISTER_NOISE", "BlockCache", "BlockDFM", "PassAudit"]

REGISTER_NOISE = 0.1      # std of a register's initial offset from the mask vector, in units of that vector's RMS


class BlockCache(PrefixCache):
    """The committed stream's keys and values: the prompt's (``prefill``), then one block's commit rows at a
    time. Like ``PrefixCache`` it is read through ``update`` and stores nothing during a pass; ``append`` adds
    the rows a fused or commit pass recorded.

    keys[l], values[l]: ``[B, kv_heads, Lc, head_dim_l]`` for each KV-storing block l; pos, valid: ``[B, Lc]``
    (rows are left-padded: a row's valid positions are 0 .. length - 1 at its right end).
    """

    @classmethod
    def from_prefix(cls, cache: PrefixCache) -> BlockCache:
        return cls(dict(cache.keys), dict(cache.values), cache.pos, cache.valid)

    def expand(self, rows: int) -> BlockCache:
        return BlockCache.from_prefix(super().expand(rows))

    def select(self, rows: Tensor) -> BlockCache:
        """The cache of the batch rows ``rows`` (int64 indices), in that order."""
        return BlockCache({i: k.index_select(0, rows) for i, k in self.keys.items()},
                          {i: v.index_select(0, rows) for i, v in self.values.items()},
                          self.pos.index_select(0, rows), self.valid.index_select(0, rows))

    def append(self, keys: dict[int, Tensor], values: dict[int, Tensor], pos: Tensor) -> None:
        """Add committed rows at the right end. keys[l], values[l]: ``[B, kv_heads, C, head_dim_l]``; pos ``[B, C]``."""
        if set(keys) != set(self.keys) or set(values) != set(self.values):
            raise ValueError("commit rows must come from the same KV-storing blocks as the cache")
        if bool((pos[:, 0] != self.lengths).any()) or bool((pos[:, 1:] != pos[:, :-1] + 1).any()):
            raise ValueError("commit rows must continue the committed stream position by position")
        self.keys = {i: torch.cat([k, keys[i].to(k.dtype)], dim=2) for i, k in self.keys.items()}
        self.values = {i: torch.cat([v, values[i].to(v.dtype)], dim=2) for i, v in self.values.items()}
        self.pos = torch.cat([self.pos, pos], dim=1)
        self.valid = torch.cat([self.valid, torch.ones_like(pos, dtype=torch.bool)], dim=1)


class _CommitRecorder:
    """Cache protocol of ``Gemma4TextAttention`` for a pass with commit rows: the pass reads [cache; its rows]
    and the keys and values of its first ``rows`` rows are remembered for ``BlockCache.append``."""

    def __init__(self, cache: BlockCache, rows: int) -> None:
        self.cache, self.rows = cache, rows
        self.keys: dict[int, Tensor] = {}
        self.values: dict[int, Tensor] = {}

    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int) -> tuple[Tensor, Tensor]:
        self.keys[layer_idx] = key_states[:, :, : self.rows]
        self.values[layer_idx] = value_states[:, :, : self.rows]
        return self.cache.update(key_states, value_states, layer_idx)


class BlockDFM(nn.Module):
    """The realizer inside block windows, and the frozen backbone on everything committed.

    Args:
        realizer: a ``Stage0Realizer`` (its adapters, mask vector and modulation are the window's; nothing of it
            is renamed, so a Stage 0 checkpoint loads into ``BlockDFM(realizer).realizer``).
        num_registers: R >= 0. The register vectors start at the mask vector plus Gaussian noise of standard
            deviation ``REGISTER_NOISE`` times that vector's RMS (seeded by ``register_seed``), so that they
            differ from one another.
    """

    def __init__(self, realizer: Stage0Realizer, num_registers: int = 0, register_seed: int = 0) -> None:
        super().__init__()
        if num_registers < 0:
            raise ValueError("num_registers must be >= 0")
        self.realizer = realizer
        self.num_registers = num_registers
        if num_registers:
            base = realizer.mask_embedding.detach().float().cpu()
            noise = torch.randn(num_registers, base.shape[0], generator=torch.Generator().manual_seed(register_seed))
            scale = REGISTER_NOISE * float(base.pow(2).mean().sqrt())
            self.registers = nn.Parameter((base[None] + scale * noise).to(realizer.mask_embedding.device))
        self.passes: dict[str, int] = {"prefill": 0, "fused": 0, "window": 0, "commit": 0, "training": 0}

    def trainable_parameters(self) -> list[nn.Parameter]:
        own = [self.registers] if self.num_registers else []
        return [*self.realizer.trainable_parameters(), *own]

    def train(self, mode: bool = True) -> BlockDFM:
        super().train(mode)
        self.realizer.train(mode)                                 # keeps the backbone in eval mode
        return self

    # -- embeddings ----------------------------------------------------------------------------
    def _embed_windows(self, anchor: Tensor, slots: Tensor, is_mask: Tensor) -> tuple[Tensor, Tensor | None]:
        """Inputs of the rows [anchor; slots; registers] of each window.

        anchor ``[B, M]``, slots and is_mask ``[B, M, Bk]`` -> embeddings ``[B, M Wn, d]`` and per-layer inputs
        ``[B, M Wn, L, d_ple]``. A masked slot takes the mask vector and a register its own vector; the per-layer
        embedding of both is looked up at the tokenizer's <mask> id (they are not tokens; decision in CLAUDE.md).
        """
        realizer, text = self.realizer, self.realizer.text
        b, m, bk = slots.shape
        r, mask_id = self.num_registers, realizer.spec.mask_id
        ids = torch.cat([anchor[:, :, None], torch.where(is_mask, torch.full_like(slots, mask_id), slots),
                         torch.full((b, m, r), mask_id, dtype=slots.dtype, device=slots.device)], dim=2)
        embeds = text.embed_tokens(ids)                                                  # [B, M, Wn, d]
        masked = torch.cat([torch.zeros_like(is_mask[:, :, :1]), is_mask,
                            torch.zeros(b, m, r, dtype=torch.bool, device=is_mask.device)], dim=2)
        embeds = torch.where(masked[..., None], realizer.mask_embedding.to(embeds.dtype), embeds)
        if r:
            registers = self.registers.to(embeds.dtype)[None, None].expand(b, m, -1, -1)
            embeds = torch.cat([embeds[:, :, : bk + 1], registers], dim=2)
        ids, embeds = ids.reshape(b, -1), embeds.reshape(b, m * (bk + 1 + r), -1)
        per_layer = None
        if getattr(text, "hidden_size_per_layer_input", 0):
            per_layer = text.project_per_layer_inputs(embeds, text.get_per_layer_inputs(ids, None))
        return embeds, per_layer

    # -- sampling passes -----------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, prompts: Sequence[Sequence[int]], device: torch.device | None = None) -> BlockCache:
        """The frozen backbone on each prompt without its last token (the first anchor). One pass."""
        if any(len(p) < 2 for p in prompts):
            raise ValueError("a prompt needs at least two tokens: <bos> and the first anchor")
        cache = self.realizer.prefill(*left_pad([list(p[:-1]) for p in prompts], self.realizer.pad_id, device))
        self.passes["prefill"] += 1
        return BlockCache.from_prefix(cache)

    def _sample_pass(self, cache: BlockCache, commit: Tensor | None, anchor: Tensor, slots: Tensor,
                     is_mask: Tensor, t: Tensor, h: Tensor, causal: bool, adapters: bool) -> Tensor:
        realizer = self.realizer
        b, bk = slots.shape
        r = self.num_registers
        if causal and r:
            raise ValueError("a causal window has no registers")
        c = 0 if commit is None else commit.shape[1]
        start = cache.lengths                                                           # [B] position of the first row
        commit_pos = start[:, None] + torch.arange(c, device=slots.device)[None]
        window_pos = window_positions(start + c, bk, r)
        masks = fused_masks(cache.pos, cache.valid, commit_pos, window_pos, realizer.window, causal)
        embeds, per_layer = self._embed_windows(anchor[:, None], slots[:, None], is_mask[:, None])
        position_ids = window_pos
        active: bool | Tensor = adapters
        reader: BlockCache | _CommitRecorder = cache
        if c:
            commit_embeds, commit_layers = realizer._embed(commit, None)
            embeds = torch.cat([commit_embeds, embeds], dim=1)
            per_layer = torch.cat([commit_layers, per_layer], dim=1) if per_layer is not None else None
            position_ids = torch.cat([commit_pos, window_pos], dim=1)
            if adapters:                                                                # commit rows: the frozen backbone
                active = torch.cat([torch.zeros(c, dtype=torch.bool), torch.ones(bk + 1 + r, dtype=torch.bool)]
                                   ).to(slots.device)[None].expand(b, -1)
            reader = _CommitRecorder(cache, c)
        modulation = realizer.modulation(t, h) if adapters else None
        hidden = realizer._run(embeds, per_layer, position_ids, masks, reader, realizer.num_layers,
                               checkpointing=False, active=active, modulation=modulation, commit_rows=c)
        if isinstance(reader, _CommitRecorder):
            if realizer.num_kv_layers == realizer.num_layers:        # no KV sharing: the commit rows ran every block
                hidden = hidden[:, c:]
            cache.append(reader.keys, reader.values, commit_pos)
        return realizer.text.norm(hidden)[:, : bk + 1]

    def fused_pass(self, cache: BlockCache, commit: Tensor | None, anchor: Tensor, slots: Tensor, is_mask: Tensor,
                   t: Tensor, h: Tensor, causal: bool = False, adapters: bool = True) -> Tensor:
        """Pass 1 of a block: commits ``commit`` to ``cache`` (in place) and denoises the window, in one pass.

        Args:
            cache: the committed stream before the commit rows.
            commit: ``[B, C]`` int64, the anchor of the previous block and its first Bk - 1 tokens; None (or
                width 0) for the first block, whose anchor is the last prompt token.
            anchor: ``[B]`` the last finished token.
            slots: ``[B, Bk]`` int64 draft; ids at masked slots are ignored.
            is_mask: ``[B, Bk]`` bool.
            t, h: ``[B]`` flow time and step size.
            causal, adapters: ``causal=True`` with ``adapters=False`` is the committed stream on the window.

        Returns:
            ``[B, Bk + 1, d]`` final hidden states of [anchor; slots]; index j predicts slot j (0-based).
        """
        if commit is not None and commit.shape[1] == 0:
            commit = None
        hidden = self._sample_pass(cache, commit, anchor, slots, is_mask, t, h, causal, adapters)
        self.passes["fused"] += 1
        return hidden

    def window_pass(self, cache: BlockCache, anchor: Tensor, slots: Tensor, is_mask: Tensor, t: Tensor, h: Tensor,
                    causal: bool = False, adapters: bool = True) -> Tensor:
        """Passes 2 .. K of a block: the window alone, reading the cache. Arguments and result as ``fused_pass``."""
        hidden = self._sample_pass(cache, None, anchor, slots, is_mask, t, h, causal, adapters)
        self.passes["window"] += 1
        return hidden

    @torch.no_grad()
    def commit_pass(self, cache: BlockCache, commit: Tensor) -> None:
        """A separate causal pass of the frozen backbone over commit rows, KV-storing blocks only; appends their
        keys and values to ``cache`` (what ``fused_pass`` does inside the window's first pass; ablation A10)."""
        realizer = self.realizer
        c = commit.shape[1]
        commit_pos = cache.lengths[:, None] + torch.arange(c, device=commit.device)[None]
        masks = fused_masks(cache.pos, cache.valid, commit_pos, commit_pos[:, :0], realizer.window)
        embeds, per_layer = realizer._embed(commit, None)
        recorder = _CommitRecorder(cache, c)
        realizer._run(embeds, per_layer, commit_pos, masks, recorder, realizer.num_kv_layers, checkpointing=False)
        cache.append(recorder.keys, recorder.values, commit_pos)
        self.passes["commit"] += 1

    # -- the training pass ---------------------------------------------------------------------
    def training_hidden(self, clean_rows: Sequence[Sequence[int]], anchor: Tensor, anchor_pos: Tensor,
                        block_valid: Tensor, slots: Tensor, is_mask: Tensor, t: Tensor, h: Tensor,
                        checkpointing: bool = False, adapters: bool = True) -> Tensor:
        """All windows of a batch of responses under the staircase mask (Eq. 5), one pass with gradients, after
        a prefill of the clean sequences without gradients.

        Args:
            clean_rows: per example the clean tokens y^1 .. y^{a_M - 1}: its prompt and the response tokens
                before the anchor of its last block (what its windows may read).
            anchor: ``[B, M]`` anchor token of each block (the last prompt token, then each block's last token).
            anchor_pos: ``[B, M]`` position a_m of each anchor.
            block_valid: ``[B, M]`` bool; False pads examples with fewer blocks.
            slots, is_mask: ``[B, M, Bk]`` the noised copy of each block.
            t, h: ``[B, M]`` flow time and step size of each block's state.

        Returns:
            ``[B, M, Bk, d]`` final hidden states; ``[b, m, j]`` predicts slot j of block m (the output of the
            anchor for j = 0 and of slot j - 1 otherwise).
        """
        realizer = self.realizer
        b, m, bk = slots.shape
        device = slots.device
        with torch.no_grad():
            cache = realizer.prefill(*left_pad([list(row) for row in clean_rows], realizer.pad_id, device))
        needed = torch.where(block_valid, anchor_pos, torch.zeros_like(anchor_pos)).max(dim=1).values
        if bool((cache.lengths < needed).any()):
            raise ValueError("the clean rows must reach the position before the last block's anchor")
        masks, row_pos, block_of = staircase_masks(cache.pos, cache.valid, anchor_pos, block_valid, bk,
                                                   self.num_registers, realizer.window)
        embeds, per_layer = self._embed_windows(anchor, slots, is_mask)
        modulation = None
        if adapters and realizer.spec.modulation:
            modulation = realizer.modulation(t.reshape(-1), h.reshape(-1)).view(b, m, realizer.num_layers, 4, -1)
        hidden = realizer._run(embeds, per_layer, row_pos, masks, cache, realizer.num_layers, checkpointing,
                               active=adapters, modulation=modulation, groups=block_of)
        hidden = realizer.text.norm(hidden).view(b, m, bk + 1 + self.num_registers, -1)
        self.passes["training"] += 1
        return hidden[:, :, :bk]


class PassAudit:
    """Counts with forward hooks what the network actually evaluated, block of the backbone by block.

    Inside the context every call of a decoder block is recorded with its number of rows. ``take`` returns and
    clears the record; ``check_block`` raises unless a generated block cost exactly the passes the sampler says
    it used and no commit row entered a KV-sharing block (Prop. 2):

        fused commit      every block of the backbone ran ``passes`` times; in the first pass the KV-storing
                          blocks saw commit + window rows, everything else saw window rows only
        separate commit   the KV-storing blocks ran once more, first, over the commit rows alone
    """

    def __init__(self, realizer: Stage0Realizer) -> None:
        self.realizer = realizer
        self.calls: list[list[int]] = [[] for _ in range(realizer.num_layers)]
        self._handles: list = []

    def __enter__(self) -> PassAudit:
        for index, layer in enumerate(self.realizer.text.layers[: self.realizer.num_layers]):
            self._handles.append(layer.register_forward_pre_hook(
                lambda module, args, index=index: self.calls[index].append(int(args[0].shape[1]))))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def take(self) -> list[list[int]]:
        calls, self.calls = self.calls, [[] for _ in range(self.realizer.num_layers)]
        return calls

    def check_prefill(self, rows: int) -> None:
        calls = self.take()
        kv = self.realizer.num_kv_layers
        want = [[rows] if i < kv else [] for i in range(self.realizer.num_layers)]
        if calls != want:
            raise RuntimeError(f"pass audit: the prefill is one pass of {rows} rows through the {kv} KV-storing "
                               f"blocks; the network ran {calls}")

    def check_block(self, passes: int, window_rows: int, commit_rows: int, fused: bool = True) -> None:
        calls = self.take()
        kv = self.realizer.num_kv_layers
        want = []
        for i in range(self.realizer.num_layers):
            if commit_rows and fused:
                first = [commit_rows + window_rows] if i < kv else [window_rows]
                want.append(first + [window_rows] * (passes - 1))
            elif commit_rows:
                want.append(([commit_rows] if i < kv else []) + [window_rows] * passes)
            else:
                want.append([window_rows] * passes)
        if calls != want:
            raise RuntimeError(f"pass audit: a block of {passes} passes ({window_rows} window rows, {commit_rows} "
                               f"commit rows, fused={fused}) must run the backbone's blocks as {want}; "
                               f"the network ran {calls}")
