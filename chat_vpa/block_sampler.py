# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
block_sampler.py -- fused KV-commit and block-bidirectional sampling (Chat-VPA Algorithm 1, Sec. 3.5).

A response is generated block by block. Block m starts from x = [MASK]^B at t = 0 and takes K passes on the
grid t_k = k/K, h = 1/K; pass 1 is ``BlockDFM.fused_pass`` (it also commits the previous block's rows), passes
2 .. K are ``BlockDFM.window_pass``. After each pass the masked slots are revealed by one of three orders:

    exact        the exact jump kernel of the mixture path (Eq. 4), the primary sampler: every masked slot is
                 revealed with probability pi_{t,h} = (kappa_{t+h} - kappa_t) / (1 - kappa_t), independently,
                 by a token drawn from its posterior; the last pass has pi = 1 (Lemma 1(c))
    prefix       prefix-growing: the leftmost masked slots are revealed, the scheduled count of them
    confidence   the masked slots whose drawn token has the highest posterior probability, the same count

The scheduled count (decision in CLAUDE.md; the paper says "the same expected number of slots per pass"): after
the pass at t the block has ceil(kappa_{t+h} B) revealed slots, which for the prefix order are exactly the
prefix-shaped training states of Sec. 3.3; at least one slot is revealed per pass while any is masked, and the
last pass reveals all.

Early exit (line 12, Prop. 3). With an exit budget delta > 0, a row whose summed residual uncertainty
eps = sum over masked slots of (1 - max_v p_i(v)) is at most delta reveals all its masked slots in that pass.
With delta = 0 (the default) the rule is off: in exact arithmetic eps > 0 always, and float32 must not decide
otherwise. p_i is the posterior at temperature 1 without the tokens that are never drawn.

Token draw (every order; rule 3 of CLAUDE.md): ``canvas_sampler.draw_tokens``, the draw of the autoregressive
baseline: temperature, then top-k, then the nucleus. [MASK] is never drawn.

After a block (lines 18 and 19): a row whose block holds ``stop_id`` (<eos>) is finished and its output ends
before that token; ``row_done`` (the truncation rule on the tokens finished so far) can finish a row as well;
otherwise the block's first B - 1 tokens and its anchor wait as the commit rows of the next block's first pass
and its last token becomes the next anchor. At most ``max_blocks`` blocks are generated.

Passes. A block costs the passes that were run until no slot of any row of the batch was masked: K, or fewer
after an exit. ``passes`` records them per batch and ``row_passes`` per row (the passes after which that row's
block was complete; their mean is the K-bar a single response would have paid). ``PassAudit`` checks with hooks
that the network ran exactly these passes and that no commit row entered a KV-sharing block.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor

from chat_vpa.block_dfm import BlockCache, BlockDFM, PassAudit
from chat_vpa.canvas_sampler import _draw_in_chunks
from chat_vpa.flow import KappaSchedule, uniform_time_grid

__all__ = ["BLOCK_ORDERS", "BlockResult", "BlockSamplerConfig", "residual_uncertainty", "sample_blocks",
           "scheduled_count"]

BLOCK_ORDERS: tuple[str, ...] = ("exact", "prefix", "confidence")


@dataclass(frozen=True)
class BlockSamplerConfig:
    block_size: int = 16           # B
    num_steps: int = 4             # K
    schedule: str = "cosine"
    order: str = "exact"           # "exact": the exact jump kernel; "prefix"; "confidence"
    exit_budget: float = 0.0       # delta; 0 switches the early exit off
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 1024
    chunk: int = 512               # positions whose logits are held at once

    def __post_init__(self) -> None:
        if self.block_size < 1 or self.num_steps < 1 or self.order not in BLOCK_ORDERS:
            raise ValueError(f"need block_size >= 1, num_steps >= 1 and order in {BLOCK_ORDERS}")
        if self.exit_budget < 0:
            raise ValueError("the exit budget must be >= 0")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.top_k < 1:
            raise ValueError("need temperature >= 0, 0 < top_p <= 1 and top_k >= 1")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlockResult:
    tokens: list[list[int]]                    # per row the generated tokens before its <eos> (or all of them)
    stops: list[str]                           # per row "eos" | "done" (row_done) | "budget"
    blocks: list[int]                          # per row the blocks it generated
    passes: list[int]                          # per generated block of the batch, the passes the network ran
    row_passes: list[list[int]] = field(default_factory=list)   # per row, per block: passes until it was complete
    revealed: list[list[int]] = field(default_factory=list)     # per block, per pass: slots revealed (batch sum)

    @property
    def network_passes(self) -> int:
        """Passes after the prefill (the sequential depth of the batch is this plus one)."""
        return sum(self.passes)

    @property
    def mean_row_passes(self) -> float:
        counts = [p for row in self.row_passes for p in row]
        return sum(counts) / max(len(counts), 1)


def scheduled_count(masked: Tensor, kappa_next: float, block_size: int, last: bool) -> Tensor:
    """Slots the prefix and confidence orders reveal per row in a pass: enough for ceil(kappa_{t+h} B) revealed
    slots after it, at least 1 while any is masked, and all of them in the last pass. ``masked`` ``[rows]`` int64."""
    if last:
        return masked.clone()
    target = min(block_size, max(0, math.ceil(kappa_next * block_size - 1e-9)))
    count = (target - (block_size - masked)).clamp(min=1)
    return torch.minimum(count, masked)


def residual_uncertainty(model: BlockDFM, hidden: Tensor, masked: Tensor, banned: Sequence[int], chunk: int) -> Tensor:
    """eps of Algorithm 1, line 11, per row: the sum over its masked slots of 1 - max_v p_i(v).

    hidden ``[rows, Bk, d]`` (index j predicts slot j), masked ``[rows, Bk]`` -> ``[rows]`` float64.
    """
    states = hidden[masked]
    doubt = []
    for start in range(0, states.shape[0], chunk):
        logits = model.realizer.logits(states[start:start + chunk]).double()
        if banned:
            logits = logits.index_fill(-1, torch.as_tensor(list(banned), device=logits.device), float("-inf"))
        log_top = logits.max(dim=-1).values - torch.logsumexp(logits, dim=-1)
        doubt.append(-torch.expm1(log_top))                                       # 1 - max p, without cancellation
    eps = torch.zeros(masked.shape, dtype=torch.float64, device=hidden.device)
    if doubt:
        eps[masked] = torch.cat(doubt)
    return eps.sum(dim=1)


@torch.no_grad()
def sample_blocks(model: BlockDFM, cache: BlockCache, anchor: Tensor, cfg: BlockSamplerConfig, max_blocks: int,
                  forbidden: Sequence[int] = (), generator: torch.Generator | None = None,
                  stop_id: int | None = None, row_done: Callable[[list[int]], bool] | None = None,
                  causal: bool = False, adapters: bool = True, fused: bool = True, audit: bool = True
                  ) -> BlockResult:
    """Algorithm 1 for a batch of rows that share nothing but the code path.

    Args:
        cache: the prompt's cache (``BlockDFM.prefill``), one row per sample; it is not modified (the rows'
            caches grow in copies).
        anchor: ``[rows]`` the last prompt token of each sample.
        max_blocks: M_max; ceil(256 / B) is the autoregressive baseline's budget of 256 new tokens.
        forbidden: token ids that are never drawn in addition to [MASK] (e.g. <pad>, <bos>).
        generator: on the model's device.
        stop_id: <eos>; None generates ``max_blocks`` blocks for every row.
        row_done: tokens finished so far -> True once the row's output is final (the truncation rule's stop);
            checked after every block.
        causal, adapters: ``causal=True`` with ``adapters=False`` and B = K = 1 is the committed stream: the
            backbone's own autoregressive decoding (Sec. 3.5, "the AR fallback").
        fused: False commits in a separate pass before the window's first pass (ablation A10): the same output,
            one more weight read per block.
        audit: check every block's passes with ``PassAudit`` (leave it on).
    """
    realizer = model.realizer
    mask_id = realizer.spec.mask_id
    device = anchor.device
    rows, bk = anchor.shape[0], cfg.block_size
    banned = sorted({mask_id, *forbidden})
    schedule = KappaSchedule(cfg.schedule)
    grid = uniform_time_grid(cfg.num_steps)
    window_rows = bk + 1 + model.num_registers
    cache = BlockCache.from_prefix(cache)
    alive = torch.arange(rows, device=device)
    tokens: list[list[int]] = [[] for _ in range(rows)]
    stops = ["budget"] * rows
    blocks = [0] * rows
    result = BlockResult(tokens, stops, blocks, [], [[] for _ in range(rows)], [])
    commit: Tensor | None = None
    auditor = PassAudit(realizer) if audit else None
    if auditor is not None:
        auditor.__enter__()
    try:
        for _ in range(max_blocks):
            n = alive.numel()
            x = torch.full((n, bk), mask_id, dtype=torch.long, device=device)
            complete_at = torch.zeros(n, dtype=torch.long, device=device)
            revealed: list[int] = []
            executed = 0
            commit_rows = 0 if commit is None else commit.shape[1]
            for k, (t, h) in enumerate(grid):
                last = k == len(grid) - 1
                masked = x == mask_id
                t_in, h_in = torch.full((n,), t, device=device), torch.full((n,), h, device=device)
                if k == 0 and (fused or commit is None):
                    hidden = model.fused_pass(cache, commit, anchor, x, masked, t_in, h_in, causal, adapters)
                elif k == 0:
                    model.commit_pass(cache, commit)
                    hidden = model.window_pass(cache, anchor, x, masked, t_in, h_in, causal, adapters)
                else:
                    hidden = model.window_pass(cache, anchor, x, masked, t_in, h_in, causal, adapters)
                hidden = hidden[:, :bk]                                              # index j predicts slot j
                executed += 1
                pi = 1.0 if last else float(schedule.exact_jump_mass(t, h))
                exiting = torch.zeros(n, dtype=torch.bool, device=device)
                if cfg.exit_budget > 0 and not last:
                    eps = residual_uncertainty(model, hidden, masked, banned, cfg.chunk)
                    exiting = eps <= cfg.exit_budget
                if cfg.order == "exact":
                    coin = torch.rand(masked.shape, generator=generator, device=device)
                    jump = masked if last else masked & ((coin < pi) | exiting[:, None])
                    drawn, _ = _draw_in_chunks(realizer, hidden[jump], cfg, banned, generator)
                    x[jump] = drawn
                else:
                    count = scheduled_count(masked.sum(dim=1), float(schedule.kappa(min(t + h, 1.0))), bk, last)
                    count = torch.where(exiting, masked.sum(dim=1), count)
                    if cfg.order == "prefix":
                        jump = masked & (masked.long().cumsum(dim=1) <= count[:, None])   # the leftmost `count`
                        drawn, _ = _draw_in_chunks(realizer, hidden[jump], cfg, banned, generator)
                        x[jump] = drawn
                    else:
                        drawn, prob = _draw_in_chunks(realizer, hidden[masked], cfg, banned, generator)
                        score = torch.full(masked.shape, -1.0, device=device)
                        score[masked] = prob
                        proposal = x.clone()
                        proposal[masked] = drawn
                        rank = score.argsort(dim=1, descending=True).argsort(dim=1)   # 0 = most confident
                        jump = masked & (rank < count[:, None])
                        x = torch.where(jump, proposal, x)
                revealed.append(int(jump.sum()))
                still = (x == mask_id).any(dim=1)
                complete_at = torch.where((complete_at == 0) & ~still, torch.full_like(complete_at, k + 1),
                                          complete_at)
                if not bool(still.any()):
                    break
            if bool((x == mask_id).any()):
                raise RuntimeError("slots are still [MASK] after the block's last pass")
            if auditor is not None:
                auditor.check_block(executed, window_rows, commit_rows, fused)
            result.passes.append(executed)
            result.revealed.append(revealed)

            # Lines 18 and 19: emit, stop, or hand the block over to the next one's first pass.
            keep = []
            for i, (row, block) in enumerate(zip(alive.tolist(), x.tolist(), strict=True)):
                blocks[row] += 1
                result.row_passes[row].append(int(complete_at[i]))
                if stop_id is not None and stop_id in block:
                    tokens[row] += block[: block.index(stop_id)]
                    stops[row] = "eos"
                    continue
                tokens[row] += block
                if row_done is not None and row_done(tokens[row]):
                    stops[row] = "done"
                    continue
                keep.append(i)
            if not keep:
                break
            index = torch.tensor(keep, dtype=torch.long, device=device)
            commit = torch.cat([anchor[:, None], x[:, : bk - 1]], dim=1).index_select(0, index)
            anchor = x[:, bk - 1].index_select(0, index)
            cache = cache.select(index)
            alive = alive.index_select(0, index)
    finally:
        if auditor is not None:
            auditor.__exit__(None, None, None)
    return result
