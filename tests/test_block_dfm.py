# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Block-DFM (Chat-VPA Sec. 3): the masks against an explicit loop over Eq. 5, the staircase training pass against
the incremental sampling passes (Prop. 1) and against transformers' own Gemma 4 under that mask, the cache
against the frozen backbone's (cache purity), the fused commit against a separate one (Prop. 2), a single block
of B = N against the whole-canvas realizer, a causal window against ``generate``, the exact kernel inside a
block, registers, training states and loss weights, and the pass audit.

All on the tiny random Gemma 4 of ``test_stage0.py``, whose 6-token sliding window is shorter than prompt +
response, so that windows and commits cross it; the realizer's LoRA factors, modulation and mask vector are
random and non-zero wherever the adapters must matter. Networks are float32: where two different computations of
the same states are compared, ``rtol=1e-4, atol=1e-5`` (the tolerance of the Stage 0 tests for the same reason:
the order of float32 sums differs with the rows of a pass).
"""

from __future__ import annotations

import math
import random

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4ForCausalLM"):                    # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4", allow_module_level=True)

from test_stage0 import MASK, VOCAB, WINDOW, randomize_adapters, tiny_realizer  # noqa: E402

from chat_vpa.autoregressive import cut_at_eos  # noqa: E402
from chat_vpa.backbone import canvas_masks, left_pad, prefill_masks  # noqa: E402
from chat_vpa.block_data import (  # noqa: E402
    block_training_state,
    collate_blocks,
    prefix_count,
    right_open,
)
from chat_vpa.block_dfm import BlockCache, BlockDFM, PassAudit  # noqa: E402
from chat_vpa.block_masks import fused_masks, staircase_masks, window_positions  # noqa: E402
from chat_vpa.block_objective import dfm_loss  # noqa: E402
from chat_vpa.block_sampler import (  # noqa: E402
    BlockSamplerConfig,
    residual_uncertainty,
    sample_blocks,
    scheduled_count,
)
from chat_vpa.canvas_sampler import SamplerConfig, realize  # noqa: E402
from chat_vpa.data import Example  # noqa: E402
from chat_vpa.flow import KappaSchedule  # noqa: E402

EOS, PAD, BOS = 1, 0, 2
RTOL, ATOL = 1e-4, 1e-5


def block_model(seed: int = 0, registers: int = 0, adapted: bool = True) -> tuple[BlockDFM, torch.nn.Module]:
    """A ``BlockDFM`` on the tiny realizer and an untouched copy of its backbone (transformers' own model)."""
    realizer, reference = tiny_realizer(seed)
    if adapted:
        randomize_adapters(realizer)
        with torch.no_grad():
            realizer.mask_embedding.add_(0.5 * torch.randn(realizer.mask_embedding.shape,
                                                           generator=torch.Generator().manual_seed(seed + 1)))
    return BlockDFM(realizer, registers, register_seed=seed), reference


def random_example(rng: random.Random, prompt_len: int, canvas_len: int) -> Example:
    """<bos> + random prompt tokens; random program tokens + <eos>."""
    prompt = [BOS] + [rng.randrange(5, VOCAB) for _ in range(prompt_len - 1)]
    canvas = [rng.randrange(5, VOCAB) for _ in range(canvas_len - 1)] + [EOS]
    return Example(prompt, canvas, "complete")


# =============================================================================================
# 1. Masks against an explicit loop over (row, key): Eq. 5, the sliding window, the global blocks
# =============================================================================================
def eq5_rows(prompt_len: int, block: int, blocks: int, registers: int) -> list[tuple[str, int, int | None]]:
    """The rows of one training pass (Sec. 3.3) as (kind, position, block): the clean rows of y = (c; w), then
    for every block its window: the anchor's copy, the draft slots, the registers (at the anchor's position)."""
    rows: list[tuple[str, int, int | None]] = [("c", n, None) for n in range(prompt_len + blocks * block)]
    for m in range(blocks):
        a = prompt_len - 1 + m * block
        rows += [("w", a + j, m) for j in range(block + 1)] + [("r", a, m)] * registers
    return rows


def eq5_masks(prompt_len: int, block: int, blocks: int, registers: int, window: int) -> dict[str, torch.Tensor]:
    """Eq. 5 entry by entry, and on top of it the backbone's sliding window (local blocks)."""
    rows = eq5_rows(prompt_len, block, blocks, registers)
    full = torch.zeros(len(rows), len(rows), dtype=torch.bool)
    sliding = torch.zeros(len(rows), len(rows), dtype=torch.bool)
    for u, (kind_u, n_u, m_u) in enumerate(rows):
        for v, (kind_v, n_v, m_v) in enumerate(rows):
            if kind_u == "c":
                see = kind_v == "c" and n_v <= n_u                                   # causal
            elif kind_v == "c":
                see = n_v < prompt_len - 1 + m_u * block                             # staircase: before a_m
            else:
                see = m_v == m_u                                                     # bidirectional window
            full[u, v] = see
            sliding[u, v] = see and abs(n_u - n_v) < window
    return {"full_attention": full, "sliding_attention": sliding}


def test_staircase_fused_and_window_masks_against_explicit_loops():
    rng = random.Random(0)
    cases = [(5, 3, 3, 1, WINDOW), (2, 1, 4, 0, 3), (9, 5, 2, 2, WINDOW), (4, 8, 2, 0, WINDOW)]   # the last: B + 1 > W
    cases += [(rng.randint(2, 9), rng.randint(1, 6), rng.randint(1, 4), rng.randint(0, 2), rng.randint(2, 7))
              for _ in range(20)]
    for prompt_len, block, blocks, registers, window in cases:
        case = (prompt_len, block, blocks, registers, window)
        want = eq5_masks(prompt_len, block, blocks, registers, window)
        rows = eq5_rows(prompt_len, block, blocks, registers)
        clean = prompt_len + blocks * block
        wn = block + 1 + registers
        anchors = torch.tensor([[prompt_len - 1 + m * block for m in range(blocks)]])
        for kind in ("full_attention", "sliding_attention"):
            assert not bool(want[kind][:clean, clean:].any()), case             # clean rows never read a window

        # The clean rows are the backbone's causal pass (``prefill``).
        pos, valid = torch.arange(clean)[None], torch.ones(1, clean, dtype=torch.bool)
        causal = prefill_masks(pos, valid, window)
        for kind in ("full_attention", "sliding_attention"):
            assert torch.equal(causal[kind][0, 0], want[kind][:clean, :clean]), case

        # The window rows of the training pass read a cache of the clean positions before the last anchor.
        cached = prompt_len - 1 + (blocks - 1) * block
        got, row_pos, block_of = staircase_masks(pos[:, :cached], valid[:, :cached], anchors,
                                                 torch.ones(1, blocks, dtype=torch.bool), block, registers, window)
        assert row_pos[0].tolist() == [n for _, n, _ in rows[clean:]], case
        assert block_of[0].tolist() == [m for _, _, m in rows[clean:]], case
        keep = [*range(cached), *range(clean, len(rows))]
        for kind in ("full_attention", "sliding_attention"):
            assert torch.equal(got[kind][0, 0], want[kind][clean:][:, keep]), (case, kind)
            assert not bool(want[kind][clean:, cached:clean].any()), case       # no window reads beyond the cache

        # Sampling passes of block m: its window, and for m >= 1 the commit rows of block m - 1 (fused).
        for m in range(blocks):
            a = prompt_len - 1 + m * block
            window_rows = [clean + m * wn + i for i in range(wn)]
            commit = list(range(a - block, a)) if m else []
            before = list(range(a - len(commit)))                                # the cache: clean positions before G
            window_pos = window_positions(torch.tensor([a]), block, registers)
            assert window_pos[0].tolist() == [rows[i][1] for i in window_rows], case
            fused = fused_masks(pos[:, :len(before)], valid[:, :len(before)], torch.tensor([commit], dtype=torch.long),
                                window_pos, window)
            alone = fused_masks(pos[:, :a], valid[:, :a], torch.zeros(1, 0, dtype=torch.long), window_pos, window)
            for kind in ("full_attention", "sliding_attention"):
                picked = [*commit, *window_rows]
                assert torch.equal(fused[kind][0, 0], want[kind][picked][:, [*before, *picked]]), (case, m, kind)
                assert torch.equal(alone[kind][0, 0], want[kind][window_rows][:, [*range(a), *window_rows]]), (case, m)
                others = [i for i in range(len(rows)) if i not in before and i not in picked]
                assert not bool(want[kind][picked][:, others].any()), case     # the pass reads nothing else

    # A causal window (the AR fallback): row i of [anchor; slots] reads the window rows up to itself.
    got = fused_masks(torch.arange(4)[None], torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 0, dtype=torch.long),
                      window_positions(torch.tensor([4]), 3, 0), window=3, causal=True)
    for q in range(4):
        for k in range(8):
            see = True if k < 4 else (k - 4) <= q
            assert bool(got["full_attention"][0, 0, q, k]) == see
            assert bool(got["sliding_attention"][0, 0, q, k]) == (see and abs((4 + q) - k) < 3)
    with pytest.raises(ValueError):
        window_positions(torch.tensor([3]), 0, 0)


def test_masks_of_a_batch_with_left_padding_and_missing_blocks():
    """Two examples with prompts and block counts of their own: every valid row is its example's Eq. 5 row; a
    row of a block that does not exist sees itself alone; padded cache keys are invisible."""
    block, registers, window = 3, 1, WINDOW
    shapes = [(7, 3), (4, 2)]                                                     # (prompt length, blocks)
    cached = [p - 1 + (m - 1) * block for p, m in shapes]
    width, most = max(cached), max(m for _, m in shapes)
    _, cache_valid, cache_pos = left_pad([[9] * n for n in cached], PAD)
    anchors = torch.zeros(2, most, dtype=torch.long)
    block_valid = torch.zeros(2, most, dtype=torch.bool)
    for b, (p, m) in enumerate(shapes):
        anchors[b, :m] = torch.tensor([p - 1 + i * block for i in range(m)])
        block_valid[b, :m] = True
    got, _, _ = staircase_masks(cache_pos, cache_valid, anchors, block_valid, block, registers, window)
    wn = block + 1 + registers
    for b, (p, m) in enumerate(shapes):
        want = eq5_masks(p, block, m, registers, window)
        clean = p + m * block
        pad = width - cached[b]
        for kind in ("full_attention", "sliding_attention"):
            mine = got[kind][b, 0]
            assert not bool(mine[: m * wn, :pad].any())                           # left padding of the cache
            keep = [*range(cached[b]), *range(clean, clean + m * wn)]
            columns = [*range(pad, width), *range(width, width + m * wn)]
            assert torch.equal(mine[: m * wn][:, columns], want[kind][clean:][:, keep]), (b, kind)
            assert not bool(mine[: m * wn, width + m * wn:].any())                # nor the padding blocks' rows
            for row in range(m * wn, most * wn):                                  # a row of a missing block
                assert mine[row].nonzero().flatten().tolist() == [width + row]


# =============================================================================================
# 2. Training equals sampling (Prop. 1)
# =============================================================================================
def _batch_and_state(block: int, seed: int = 1):
    rng = random.Random(seed)
    examples = [random_example(rng, 7, 8), random_example(rng, 4, 5), random_example(rng, 9, 3)]
    batch = collate_blocks(examples, block, EOS)
    state = block_training_state(batch, KappaSchedule("cosine"), [2, 4], MASK,
                                 generator=torch.Generator().manual_seed(seed))
    return examples, batch, state


def _training_logits(model: BlockDFM, batch, state, **kwargs) -> torch.Tensor:
    hidden = model.training_hidden(batch.clean_rows, batch.anchor, batch.anchor_pos, batch.block_valid, state.slots,
                                   state.is_mask, state.t, state.h, **kwargs)
    return model.realizer.logits(hidden)                                        # [B, M, Bk, S]


@pytest.mark.parametrize("registers", [0, 2])
def test_training_pass_equals_the_incremental_sampling_passes(registers):
    """Window logits of one staircase training pass over a batch = those of prefill + one fused pass per block,
    example by example, with the same drafts and times and the target's tokens committed (Prop. 1(b))."""
    model, _ = block_model(registers=registers)
    block = 3
    examples, batch, state = _batch_and_state(block)
    assert batch.block_valid.sum(dim=1).tolist() == [3, 2, 1]                   # blocks per example; padding blocks
    assert bool(state.is_mask.any()) and bool((~state.is_mask).any())
    assert len(set(zip(state.t.flatten().tolist(), state.h.flatten().tolist(), strict=True))) >= 3   # times per block
    with torch.no_grad():
        trained = _training_logits(model, batch, state)
        frozen = _training_logits(model, batch, state, adapters=False)
        compared = 0
        for b, example in enumerate(examples):
            cache = model.prefill([example.prompt_ids])
            commit = None
            for m in range(int(batch.block_valid[b].sum())):
                anchor = batch.anchor[b, m][None]
                hidden = model.fused_pass(cache, commit, anchor, state.slots[b, m][None], state.is_mask[b, m][None],
                                          state.t[b, m][None], state.h[b, m][None])
                torch.testing.assert_close(model.realizer.logits(hidden[0, :block]), trained[b, m], rtol=RTOL,
                                           atol=ATOL, msg=f"example {b}, block {m}")
                commit = torch.cat([anchor, batch.target[b, m, : block - 1]])[None]
                compared += 1
            assert int(cache.lengths) == len(batch.clean_rows[b])               # all but the last block committed
    assert compared == 6
    valid = batch.block_valid
    assert float((trained[valid] - frozen[valid]).abs().max()) > 1e-2           # the adapters matter in this test


def test_training_pass_is_transformers_own_forward_under_the_mask_of_eq5():
    """With the adapters at their initial values the window rows of the training pass are one forward pass of the
    unmodified ``Gemma4ForCausalLM`` over [clean rows; windows] under the explicit mask of Eq. 5, with every
    row at its position and <mask> at the masked slots; its clean rows are the causal backbone."""
    model, reference = block_model(adapted=False)
    block = 3
    examples, batch, state = _batch_and_state(block, seed=2)
    with torch.no_grad():
        trained = _training_logits(model, batch, state)
        for b, example in enumerate(examples):
            blocks = int(batch.block_valid[b].sum())
            p = len(example.prompt_ids)
            y = [*example.prompt_ids, *batch.target[b, :blocks].flatten().tolist()]
            rows = eq5_rows(p, block, blocks, 0)
            masks = eq5_masks(p, block, blocks, 0, WINDOW)
            ids = []
            for kind, n, m in rows:
                if kind == "c" or n == p - 1 + m * block:                       # a clean row, or an anchor's copy
                    ids.append(y[n])
                else:
                    ids.append(int(state.slots[b, m, n - (p - 1 + m * block) - 1]))   # the draft: a token or <mask>
            out = reference(input_ids=torch.tensor([ids]), position_ids=torch.tensor([[n for _, n, _ in rows]]),
                            attention_mask={kind: mask[None, None] for kind, mask in masks.items()}).logits[0]
            clean = len(y)
            for m in range(blocks):
                first = clean + m * (block + 1)
                torch.testing.assert_close(trained[b, m], out[first:first + block], rtol=RTOL, atol=ATOL,
                                           msg=f"example {b}, block {m}")
            causal = reference(input_ids=torch.tensor([y])).logits[0]
            torch.testing.assert_close(out[:clean], causal, rtol=RTOL, atol=ATOL)
    assert MASK in state.slots.flatten().tolist()


# =============================================================================================
# 3. Cache purity (Prop. 1(a))    4. Fused commit = separate commit (Prop. 2, ablation A10)
# =============================================================================================
def _draft(generator: torch.Generator, rows: int, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(5, VOCAB, (rows, block), generator=generator)
    return tokens, torch.rand(rows, block, generator=generator) < 0.5


def test_the_cache_is_the_frozen_backbones_after_every_fused_commit():
    """After each fused pass the cache holds, for every row of a left-padded batch, the causal keys and values of
    the frozen backbone over the committed tokens: those of ``prefill`` on that sequence and those in
    transformers' own ``past_key_values`` (whose local blocks keep the last W - 1 positions). The adapters,
    the modulation, the mask vector and the registers are random and must not reach a commit row."""
    model, reference = block_model(registers=1)
    realizer, block = model.realizer, 3
    generator = torch.Generator().manual_seed(3)
    prompts = [[BOS, *torch.randint(5, VOCAB, (n,), generator=generator).tolist()] for n in (8, 3)]
    with torch.no_grad():
        cache = model.prefill(prompts)
        committed = [list(p[:-1]) for p in prompts]
        anchor = torch.tensor([p[-1] for p in prompts])
        commit = None
        for _ in range(4):
            slots, is_mask = _draft(generator, 2, block)
            before = cache.lengths.clone()
            model.fused_pass(cache, commit, anchor, slots, is_mask, torch.tensor([0.25, 0.5]), torch.tensor([0.25, 0.5]))
            if commit is not None:
                for b in range(2):
                    committed[b] += commit[b].tolist()
                assert cache.lengths.tolist() == (before + block).tolist()
            for b in range(2):
                n = len(committed[b])
                assert int(cache.lengths[b]) == n and cache.pos[b, -n:].tolist() == list(range(n))
                assert bool(cache.valid[b, -n:].all()) and not bool(cache.valid[b, :-n].any())
                frozen = realizer.prefill(*left_pad([committed[b]], PAD))
                past = reference(torch.tensor([committed[b]]), use_cache=True).past_key_values
                assert sorted(cache.keys) == sorted(frozen.keys) == list(range(len(past.layers))) == [0, 1, 2, 3]
                for layer in cache.keys:
                    for mine, theirs in ((cache.keys, frozen.keys), (cache.values, frozen.values)):
                        torch.testing.assert_close(mine[layer][b, :, -n:], theirs[layer][0], rtol=RTOL, atol=ATOL)
                    kept = past.layers[layer].keys.shape[2]                     # W - 1 in a local block, n in a global one
                    assert kept == (min(n, WINDOW - 1) if realizer.layer_types[layer] == "sliding_attention" else n)
                    torch.testing.assert_close(cache.keys[layer][b, :, -kept:], past.layers[layer].keys[0], rtol=RTOL,
                                               atol=ATOL)
                    torch.testing.assert_close(cache.values[layer][b, :, -kept:], past.layers[layer].values[0],
                                               rtol=RTOL, atol=ATOL)
            final, _ = _draft(generator, 2, block)                                # the block as it was finished
            commit = torch.cat([anchor[:, None], final[:, : block - 1]], dim=1)
            anchor = final[:, -1]
    assert [len(c) for c in committed] == [8 + 3 * block, 3 + 3 * block]        # commits crossed the 6-token window


def test_fused_commit_equals_a_separate_commit_pass_and_runs_every_block_once():
    model, _ = block_model(registers=2)
    realizer, block = model.realizer, 4
    window_rows = block + 1 + 2
    kv, layers = realizer.num_kv_layers, realizer.num_layers
    generator = torch.Generator().manual_seed(4)
    prompts = [[BOS, *torch.randint(5, VOCAB, (n,), generator=generator).tolist()] for n in (6, 9)]
    t, h = torch.tensor([0.0, 0.5]), torch.tensor([0.25, 0.5])
    with torch.no_grad(), PassAudit(realizer) as audit:
        fused, separate = model.prefill(prompts), model.prefill(prompts)
        audit.take()
        anchor = torch.tensor([p[-1] for p in prompts])
        commit = None
        for m in range(3):
            slots, is_mask = _draft(generator, 2, block)
            one = model.fused_pass(fused, commit, anchor, slots, is_mask, t, h)
            calls = audit.take()
            rows = (block if m else 0) + window_rows                              # |G| + |W| below, |W| above
            assert calls == [[rows] if i < kv else [window_rows] for i in range(layers)], m
            if commit is not None:
                model.commit_pass(separate, commit)
                assert audit.take() == [[block] if i < kv else [] for i in range(layers)]
            two = model.window_pass(separate, anchor, slots, is_mask, t, h)
            assert audit.take() == [[window_rows]] * layers
            torch.testing.assert_close(realizer.logits(one), realizer.logits(two), rtol=RTOL, atol=ATOL)
            assert torch.equal(fused.pos, separate.pos) and torch.equal(fused.valid, separate.valid)
            for layer in fused.keys:
                torch.testing.assert_close(fused.keys[layer], separate.keys[layer], rtol=RTOL, atol=ATOL)
                torch.testing.assert_close(fused.values[layer], separate.values[layer], rtol=RTOL, atol=ATOL)
            final, _ = _draft(generator, 2, block)
            commit = torch.cat([anchor[:, None], final[:, : block - 1]], dim=1)
            anchor = final[:, -1]
    assert kv == 4 and layers == 7 and fused.pos.shape[1] == 9 + 2 * block     # 9 prompt keys, two commits of B
    assert model.passes == {"prefill": 2, "fused": 3, "window": 3, "commit": 2, "training": 0}


# =============================================================================================
# 5. A single block of B = N is the whole-canvas realizer    6. A causal window is the committed stream
# =============================================================================================
def test_one_block_spanning_the_canvas_is_the_stage0_realizer():
    """B = N, M = 1, R = 0: the window pass is ``Stage0Realizer.canvas_hidden`` on the same state (N = 14 exceeds
    the 6-token window, so the two-sided local rule is in play), and the block sampler with the exact kernel
    reproduces ``canvas_sampler.realize`` token for token from the same seed."""
    model, _ = block_model()
    realizer, n, rows = model.realizer, 14, 3
    prompt = [BOS, 30, 31, 32, 33, 34, 35, 36, 37]
    generator = torch.Generator().manual_seed(5)
    tokens, is_mask = _draft(generator, rows, n)
    t, h = torch.tensor([0.0, 0.25, 0.5]), torch.tensor([0.25, 0.25, 0.5])
    anchor = torch.full((rows,), prompt[-1])
    front = torch.zeros(rows, 1, dtype=torch.bool)
    with torch.no_grad():
        cache = model.prefill([prompt]).expand(rows)
        prefix = realizer.prefill(*left_pad([prompt[:-1]], PAD)).expand(rows)
        for causal, mode in ((False, "bidirectional"), (True, "causal")):
            got = fused_masks(cache.pos, cache.valid, cache.pos[:, :0], window_positions(cache.lengths, n, 0), WINDOW,
                              causal)
            want = canvas_masks(prefix.pos, prefix.valid, prefix.lengths[:, None] + torch.arange(n + 1)[None],
                                WINDOW, mode)
            assert torch.equal(got["full_attention"], want["full_attention"])
            assert torch.equal(got["sliding_attention"], want["sliding_attention"])
            hidden = model.window_pass(cache, anchor, tokens, is_mask, t, h, causal=causal)
            canvas = realizer.canvas_hidden(prefix, torch.cat([anchor[:, None], tokens], dim=1),
                                            torch.cat([front, is_mask], dim=1), t, h, mode=mode)
            torch.testing.assert_close(hidden, canvas, rtol=0, atol=0)          # the same operations
            torch.testing.assert_close(realizer.logits(hidden), realizer.logits(canvas), rtol=0, atol=0)
        for steps in (2, 4, 8):
            block = sample_blocks(model, cache, anchor, BlockSamplerConfig(block_size=n, num_steps=steps,
                                                                           temperature=0.7), max_blocks=1,
                                  forbidden=[PAD, BOS], generator=torch.Generator().manual_seed(7))
            canvas = realize(realizer, prefix, anchor, n, SamplerConfig(num_steps=steps, temperature=0.7),
                             forbidden=[PAD, BOS], generator=torch.Generator().manual_seed(7))
            assert block.tokens == canvas.tokens.tolist() and block.stops == ["budget"] * rows
            assert block.passes == [canvas.canvas_passes] == [steps] and block.blocks == [1] * rows
    assert len({tuple(row) for row in block.tokens}) == rows                    # sampled, not degenerate


def test_a_causal_window_without_adapters_is_the_committed_stream():
    """(a) A window of clean tokens under the causal mask, adapters and modulation off, gives the backbone's own
    next-token logits. (b) With B = 1 and K = 1 the block sampler is then greedy ``generate``, token for token."""
    model, reference = block_model(seed=1)                                       # random adapters: must not matter
    realizer = model.realizer
    prompt, n = [BOS, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69], 12
    reference.generation_config.eos_token_id = EOS
    with torch.no_grad():
        ids = torch.tensor([[*prompt, 70, 71, 72, 73, 74]])
        want = reference(ids).logits[0, len(prompt) - 1:]                       # from the anchor on
        cache = model.prefill([prompt])
        hidden = model.window_pass(cache, torch.tensor([prompt[-1]]), ids[:, len(prompt):],
                                   torch.zeros(1, 5, dtype=torch.bool), torch.zeros(1), torch.ones(1), causal=True,
                                   adapters=False)
        torch.testing.assert_close(realizer.logits(hidden[0]), want, rtol=RTOL, atol=ATOL)
        adapted = model.window_pass(cache, torch.tensor([prompt[-1]]), ids[:, len(prompt):],
                                    torch.zeros(1, 5, dtype=torch.bool), torch.zeros(1), torch.ones(1), causal=True)
        assert float((realizer.logits(adapted[0]) - want).abs().max()) > 1e-2   # with them it is not

        out = reference.generate(torch.tensor([prompt]), max_new_tokens=n, do_sample=False,
                                 suppress_tokens=[MASK, PAD, BOS], pad_token_id=PAD)
        greedy = cut_at_eos(out[0, len(prompt):].tolist(), [EOS])
        result = sample_blocks(model, cache, torch.tensor([prompt[-1]]),
                               BlockSamplerConfig(block_size=1, num_steps=1, temperature=0.0), max_blocks=n,
                               forbidden=[PAD, BOS], stop_id=EOS, causal=True, adapters=False)
    assert result.tokens == [greedy] and len(greedy) >= 3
    assert result.passes == [1] * result.blocks[0] and result.network_passes == result.blocks[0]
    assert result.stops == (["eos"] if len(greedy) < n else ["budget"])
    with pytest.raises(ValueError, match="registers"):
        BlockDFM(realizer, 1).window_pass(cache, torch.tensor([9]), ids[:, :2], torch.zeros(1, 2, dtype=torch.bool),
                                          torch.zeros(1), torch.ones(1), causal=True)


# =============================================================================================
# 7. The kernel inside a block; the orders; stopping
# =============================================================================================
def _spy(model: BlockDFM) -> list[torch.Tensor]:
    """Record the mask of every pass's draft (``is_mask`` of ``fused_pass`` and ``window_pass``)."""
    seen: list[torch.Tensor] = []
    fused, window = model.fused_pass, model.window_pass

    def spy_fused(cache, commit, anchor, slots, is_mask, *args, **kwargs):
        seen.append(is_mask.clone())
        return fused(cache, commit, anchor, slots, is_mask, *args, **kwargs)

    def spy_window(cache, anchor, slots, is_mask, *args, **kwargs):
        seen.append(is_mask.clone())
        return window(cache, anchor, slots, is_mask, *args, **kwargs)

    model.fused_pass, model.window_pass = spy_fused, spy_window                 # type: ignore[method-assign]
    return seen


@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_exact_kernel_reveals_each_slot_of_a_block_with_the_schedules_probability(schedule):
    """Monte Carlo over 4,000 rows of one block: before pass k every slot is revealed with probability
    kappa_{k/K}, independently of its place (5 standard errors); the last pass leaves no [MASK]."""
    model, _ = block_model()
    rows, block, steps = 4000, 5, 4
    seen = _spy(model)
    with torch.no_grad():
        cache = model.prefill([[BOS, 40, 41, 42]]).expand(rows)
        result = sample_blocks(model, cache, torch.full((rows,), 42), BlockSamplerConfig(
            block_size=block, num_steps=steps, schedule=schedule, temperature=1.0, top_p=1.0, top_k=VOCAB),
            max_blocks=1, generator=torch.Generator().manual_seed(0))
    assert len(seen) == steps and result.passes == [steps]
    assert all(len(row) == block and MASK not in row for row in result.tokens)
    for k, is_mask in enumerate(seen):
        kappa = 1 - math.cos(math.pi * (k / steps) / 2) if schedule == "cosine" else k / steps
        revealed = 1.0 - is_mask.float().mean(dim=0)                            # per slot
        error = 5 * math.sqrt(kappa * (1 - kappa) / rows) + 1e-12
        assert bool(((revealed - kappa).abs() <= error).all()), (k, revealed.tolist(), kappa)
    for before, after in zip(seen[:-1], seen[1:], strict=True):
        assert bool((after <= before).all())                                    # a revealed slot stays revealed
    assert sum(sum(r) for r in result.revealed) == rows * block


class ScriptedBlocks:
    """A stand-in for ``BlockDFM`` whose posterior for slot j of block m of row r is (almost) all on
    ``scripts[r][m B + j]``, with a peak height of its own per slot, so that what the sampler reveals, when, and
    what it commits is known exactly. The row's identity travels in the cache's ``pos``."""

    def __init__(self, scripts: list[list[int]], block: int, sharpness: list[float] | None = None, vocab: int = 40):
        import types

        self.realizer, self.num_registers, self.block, self.vocab = self, 0, block, vocab
        self.spec = types.SimpleNamespace(mask_id=MASK)
        self.scripts = torch.tensor(scripts)
        self.sharpness = torch.tensor(sharpness or [40.0] * block)
        self.block_index = -1
        self.masks: list[torch.Tensor] = []
        self.firsts: list[tuple[list | None, list]] = []                        # (commit rows, anchors) per block
        self.passes = 0

    def cache(self) -> BlockCache:
        rows = self.scripts.shape[0]
        return BlockCache({}, {}, torch.arange(rows)[:, None], torch.ones(rows, 1, dtype=torch.bool))

    def _hidden(self, cache, is_mask):
        self.passes += 1
        self.masks.append(is_mask.clone())
        start = self.block_index * self.block
        script = self.scripts[cache.pos[:, 0], start:start + self.block]         # [rows, Bk]
        hidden = torch.zeros(is_mask.shape[0], self.block + 1, self.vocab)
        hidden[:, : self.block] = torch.nn.functional.one_hot(script, self.vocab) * self.sharpness[None, :, None]
        return hidden                                                           # index j: the posterior of slot j

    def fused_pass(self, cache, commit, anchor, slots, is_mask, t, h, causal=False, adapters=True):
        self.block_index += 1
        self.firsts.append((None if commit is None else commit.tolist(), anchor.tolist()))
        return self._hidden(cache, is_mask)

    def window_pass(self, cache, anchor, slots, is_mask, t, h, causal=False, adapters=True):
        return self._hidden(cache, is_mask)

    def logits(self, hidden):
        return hidden


@pytest.mark.parametrize("block,steps", [(16, 4), (16, 2), (32, 4), (32, 2), (5, 3)])
def test_prefix_and_confidence_orders_reveal_the_scheduled_count(block, steps):
    """Prefix-growing reveals the leftmost slots and the confidence order the most confident ones; before pass k
    the block has ceil(kappa_{k/K} B) revealed slots, the prefix-shaped training states of Sec. 3.3."""
    script = [[10 + (j % 20) for j in range(block)]]
    cumulative = [math.ceil((1 - math.cos(math.pi * (k / steps) / 2)) * block - 1e-9) for k in range(steps)] + [block]
    assert all(b > a for a, b in zip(cumulative[:-1], cumulative[1:], strict=True))    # at least one slot per pass
    schedule = KappaSchedule("cosine")
    batch = collate_blocks([Example([BOS, 9], [*script[0][: block - 1], EOS], "complete")], block, EOS)

    prefix = ScriptedBlocks(script, block)
    result = sample_blocks(prefix, prefix.cache(), torch.tensor([9]),
                           BlockSamplerConfig(block_size=block, num_steps=steps, order="prefix"), max_blocks=1,
                           audit=False)
    assert result.tokens == script and result.passes == [steps]
    assert result.revealed == [[b - a for a, b in zip(cumulative[:-1], cumulative[1:], strict=True)]]
    for k, is_mask in enumerate(prefix.masks):
        assert is_mask[0].tolist() == [j >= cumulative[k] for j in range(block)]
        state = block_training_state(batch, schedule, [steps], MASK, budget=steps, step_index=k, prefix=True)
        assert torch.equal(state.is_mask[0, 0], is_mask[0])                    # the state the model was trained on

    order = list(range(block))
    random.Random(block * steps).shuffle(order)                                 # order[i]: the i-th most confident
    sharpness = [0.0] * block
    for rank, slot in enumerate(order):
        sharpness[slot] = 12.0 - 8.0 * rank / block                             # distinct peak heights, 4 to 12
    confident = ScriptedBlocks(script, block, sharpness)
    result = sample_blocks(confident, confident.cache(), torch.tensor([9]),
                           BlockSamplerConfig(block_size=block, num_steps=steps, order="confidence", temperature=0.0),
                           max_blocks=1, audit=False)
    assert result.tokens == script
    for k, is_mask in enumerate(confident.masks):
        assert sorted(j for j in range(block) if not is_mask[0, j]) == sorted(order[: cumulative[k]])

    # scheduled_count itself, against the explicit rule, for every number of masked slots
    for k in range(steps):
        last = k == steps - 1
        for masked in range(block + 1):
            want = masked if last else min(masked, max(1, cumulative[k + 1] - (block - masked)))
            got = scheduled_count(torch.tensor([masked]), float(schedule.kappa((k + 1) / steps)), block, last)
            assert got.tolist() == [want]


def test_blocks_are_emitted_committed_and_stopped_as_in_algorithm_1():
    """Lines 18 and 19 on three rows: row 0 meets <eos> in its second block (stops there, emits up to it), row 1
    is ended by ``row_done`` after its first block, row 2 runs to the block limit. The commit rows of a block's
    first pass are the previous anchor and the previous block's first B - 1 tokens; the new anchor is its last."""
    block = 4
    scripts = [[10, 11, 12, 13, 14, EOS, 16, 17, 18, 19, 20, 21],
               [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
               [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 30, 31]]
    model = ScriptedBlocks(scripts, block)
    result = sample_blocks(model, model.cache(), torch.tensor([7, 8, 9]),
                           BlockSamplerConfig(block_size=block, num_steps=2, order="prefix"), max_blocks=3,
                           stop_id=EOS, row_done=lambda tokens: tokens[0] == 20, audit=False)
    assert result.tokens == [[10, 11, 12, 13, 14], [20, 21, 22, 23], scripts[2]]
    assert result.stops == ["eos", "done", "budget"] and result.blocks == [2, 1, 3]
    assert result.passes == [2, 2, 2] and result.row_passes == [[2, 2], [2], [2, 2, 2]]
    assert result.network_passes == 6 == model.passes and result.mean_row_passes == 2.0
    assert model.firsts == [(None, [7, 8, 9]),
                            ([[7, 10, 11, 12], [9, 30, 31, 32]], [13, 33]),      # row 1 is gone
                            ([[33, 34, 35, 36]], [37])]                          # row 0 is gone
    # Without a stop token every row generates max_blocks blocks.
    model = ScriptedBlocks(scripts, block)
    result = sample_blocks(model, model.cache(), torch.tensor([7, 8, 9]),
                           BlockSamplerConfig(block_size=block, num_steps=2), max_blocks=2, audit=False)
    assert result.tokens == [s[:8] for s in scripts] and result.stops == ["budget"] * 3
    for bad in ({"order": "left_to_right"}, {"block_size": 0}, {"exit_budget": -1.0}, {"top_p": 0.0}):
        with pytest.raises(ValueError):
            BlockSamplerConfig(**bad)


def test_early_exit_fires_only_with_a_budget_and_only_below_it():
    """Line 12: with delta > 0 a row reveals all its masked slots once eps <= delta; delta = 0 is off, even when
    float32 cannot tell the top probability from 1. eps against an explicit float64 loop."""
    block, steps = 6, 4
    script = [[10, 11, 12, 13, 14, 15]]
    for order in ("exact", "prefix", "confidence"):
        cfg = {"block_size": block, "num_steps": steps, "order": order}
        sharp = ScriptedBlocks(script, block)                                   # peaks of 40: max p rounds to 1 in float32
        assert float(torch.softmax(sharp.sharpness[0] * torch.eye(40)[0], dim=0)[0]) == 1.0
        off = sample_blocks(sharp, sharp.cache(), torch.tensor([9]), BlockSamplerConfig(**cfg), max_blocks=1,
                            audit=False, generator=torch.Generator().manual_seed(0))
        assert off.tokens == script
        if order != "exact":                                                    # (the exact kernel may finish early by chance)
            assert off.passes == [steps] and off.row_passes == [[steps]]
        sharp = ScriptedBlocks(script, block)
        on = sample_blocks(sharp, sharp.cache(), torch.tensor([9]), BlockSamplerConfig(exit_budget=0.05, **cfg),
                           max_blocks=1, audit=False, generator=torch.Generator().manual_seed(0))
        assert on.tokens == script and on.passes == [1] and on.row_passes == [[1]] and on.mean_row_passes == 1.0
        flat = ScriptedBlocks(script, block, sharpness=[2.0] * block)           # uncertain slots: no exit at 0.05
        kept = sample_blocks(flat, flat.cache(), torch.tensor([9]),
                             BlockSamplerConfig(exit_budget=0.05, temperature=0.0, **cfg), max_blocks=1, audit=False,
                             generator=torch.Generator().manual_seed(0))
        if order != "exact":
            assert kept.passes == [steps]

    model, _ = block_model()
    generator = torch.Generator().manual_seed(6)
    hidden = torch.randn(3, 4, 64, generator=generator)
    masked = torch.tensor([[True, False, True, True], [False, False, False, False], [True, True, True, True]])
    got = residual_uncertainty(model, hidden, masked, banned=[MASK, PAD], chunk=3)
    logits = model.realizer.logits(hidden).double()
    for row in range(3):
        want = 0.0
        for slot in range(4):
            if masked[row, slot]:
                p = torch.softmax(torch.tensor([x for v, x in enumerate(logits[row, slot].tolist())
                                                if v not in (MASK, PAD)], dtype=torch.float64), dim=0)
                want += 1.0 - float(p.max())
        assert math.isclose(float(got[row]), want, rel_tol=1e-9, abs_tol=1e-12)
    assert float(got[1]) == 0.0


# =============================================================================================
# 8. Registers
# =============================================================================================
def test_registers_change_the_window_and_never_the_cache():
    """R = 0 and R = 2 on the same weights, driven through the same drafts and commits: the windows' outputs
    differ, and the cache (every committed state) is the same after every fused commit."""
    plain, _ = block_model()
    with_registers = BlockDFM(plain.realizer, 2, register_seed=0)
    assert with_registers.registers.shape == (2, 64) and not hasattr(plain, "registers")
    registers = with_registers.registers.detach()
    assert float((registers[0] - registers[1]).abs().max()) > 0                 # not one vector twice
    assert len(with_registers.trainable_parameters()) == len(plain.trainable_parameters()) + 1
    block = 3
    generator = torch.Generator().manual_seed(8)
    prompts = [[BOS, *torch.randint(5, VOCAB, (n,), generator=generator).tolist()] for n in (7, 4)]
    t, h = torch.tensor([0.25, 0.0]), torch.tensor([0.25, 0.5])
    with torch.no_grad():
        caches = [plain.prefill(prompts), with_registers.prefill(prompts)]
        anchor, commit, changed = torch.tensor([p[-1] for p in prompts]), None, 0.0
        for _ in range(4):
            slots, is_mask = _draft(generator, 2, block)
            outs = [model.fused_pass(cache, commit, anchor, slots, is_mask, t, h)
                    for model, cache in zip((plain, with_registers), caches, strict=True)]
            assert outs[0].shape == outs[1].shape == (2, block + 1, 64)       # register rows are not returned
            changed = max(changed, float((outs[0] - outs[1]).abs().max()))
            assert torch.equal(caches[0].pos, caches[1].pos) and torch.equal(caches[0].valid, caches[1].valid)
            for layer in caches[0].keys:
                torch.testing.assert_close(caches[0].keys[layer], caches[1].keys[layer], rtol=RTOL, atol=ATOL)
                torch.testing.assert_close(caches[0].values[layer], caches[1].values[layer], rtol=RTOL, atol=ATOL)
            final, _ = _draft(generator, 2, block)
            commit = torch.cat([anchor[:, None], final[:, : block - 1]], dim=1)
            anchor = final[:, -1]
    assert changed > 1e-2 and caches[1].pos.shape[1] == 7 + 3 * block          # B rows per commit, no register row
    # In training the registers are learned: the loss reaches them.
    examples, batch, state = _batch_and_state(block)
    loss, _ = dfm_loss(with_registers, batch, state, torch.device("cpu"))
    loss.backward()
    assert float(with_registers.registers.grad.abs().sum()) > 0
    with pytest.raises(ValueError):
        BlockDFM(plain.realizer, -1)


# =============================================================================================
# 9. Blocks, training states, loss weights and the right-open set
# =============================================================================================
def test_collate_blocks_against_an_explicit_layout():
    rng = random.Random(9)
    block = 4
    examples = [random_example(rng, 6, 9), random_example(rng, 3, 4), random_example(rng, 2, 1)]
    batch = collate_blocks(examples, block, EOS)
    assert batch.target.shape == (3, 3, block) and batch.length.tolist() == [9, 4, 1]
    for b, example in enumerate(examples):
        blocks = -(-len(example.canvas_ids) // block)                            # ceil
        response = example.canvas_ids + [EOS] * (blocks * block - len(example.canvas_ids))
        y = example.prompt_ids + response                                       # position n holds y[n]
        p = len(example.prompt_ids)
        assert batch.block_valid[b].tolist() == [m < blocks for m in range(3)]
        for m in range(blocks):
            a = p - 1 + m * block
            assert int(batch.anchor_pos[b, m]) == a and int(batch.anchor[b, m]) == y[a]
            assert batch.target[b, m].tolist() == y[a + 1: a + 1 + block]
        assert batch.clean_rows[b] == y[: p - 1 + (blocks - 1) * block]         # what the last window may read
        assert response.index(EOS) == len(example.canvas_ids) - 1
    assert batch.anchor[0, 1] == examples[0].canvas_ids[block - 1]              # a block's last token anchors the next
    with pytest.raises(ValueError):
        collate_blocks([Example([BOS, 7], [8, 9], "complete")], block, EOS)     # a canvas without its <eos>
    with pytest.raises(ValueError):
        collate_blocks(examples, 0, EOS)


def test_block_training_states_weights_and_right_open_slots_against_loops():
    block, budgets = 8, [2, 4]
    rng = random.Random(10)
    examples = [random_example(rng, 5, rng.choice([3, 8, 11, 17, 24])) for _ in range(400)]
    batch = collate_blocks(examples, block, EOS)
    b, m, _ = batch.target.shape
    schedule = KappaSchedule("cosine")
    state = block_training_state(batch, schedule, budgets, MASK, prefix_fraction=0.5, pad_weight=0.1, seam_slots=3,
                                 seam_weight=1.0, generator=torch.Generator().manual_seed(10))
    valid = batch.block_valid
    counts = {"prefix": 0, "blocks": 0, "revealed": {}, "slots": {}}
    for i in range(b):
        for j in range(m):
            big_k, k = int(state.budget[i, j]), int(state.step[i, j])
            assert big_k in budgets and 0 <= k < big_k
            assert math.isclose(float(state.t[i, j]), k / big_k, abs_tol=1e-7)
            assert math.isclose(float(state.h[i, j]), 1 / big_k, abs_tol=1e-7)
            kappa = 1 - math.cos(math.pi * (k / big_k) / 2)
            mask = state.is_mask[i, j].tolist()
            if bool(state.prefix[i, j]):                                        # first ceil(kappa B) slots revealed
                assert mask == [s >= math.ceil(kappa * block - 1e-9) for s in range(block)]
            for s in range(block):
                token = int(state.slots[i, j, s])
                assert token == (MASK if mask[s] else int(batch.target[i, j, s]))
                # weights: 0 unless masked in an existing block; 0.1 on the filling; 1 + lambda on the first b_0
                filling = j * block + s >= int(batch.length[i])
                want = 0.0 if not (mask[s] and bool(valid[i, j])) else (0.1 if filling else (2.0 if s < 3 else 1.0))
                assert math.isclose(float(state.weights[i, j, s]), want, rel_tol=1e-6)
                assert bool(state.right_open[i, j, s]) == all(mask[s:])         # Eq. 11
            if bool(valid[i, j]):
                counts["blocks"] += 1
                counts["prefix"] += bool(state.prefix[i, j])
                if not bool(state.prefix[i, j]):
                    key = (big_k, k)
                    counts["revealed"][key] = counts["revealed"].get(key, 0) + mask.count(False)
                    counts["slots"][key] = counts["slots"].get(key, 0) + block
    share = counts["prefix"] / counts["blocks"]                                 # rho_pre, 5 standard errors
    assert abs(share - 0.5) < 5 * math.sqrt(0.25 / counts["blocks"])
    assert set(counts["slots"]) == {(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)}     # every (K, k) was drawn
    for (big_k, k), slots in counts["slots"].items():                           # independent reveals at rate kappa_t
        kappa = 1 - math.cos(math.pi * (k / big_k) / 2)
        assert abs(counts["revealed"][(big_k, k)] / slots - kappa) <= 5 * math.sqrt(kappa * (1 - kappa) / slots) + 1e-12
    # The first b_0 slots of a block that are filling keep the filling's weight (the last block of a short program).
    short = collate_blocks([Example([BOS, 7], [9, EOS], "complete")], block, EOS)
    masked = block_training_state(short, schedule, budgets, MASK, seam_slots=3, budget=2, step_index=0)
    assert bool(masked.is_mask.all()) and masked.weights[0, 0].tolist()[:4] == pytest.approx([2.0, 2.0, 0.1, 0.1])
    assert torch.equal(masked.right_open, masked.is_mask)                       # t = 0: every slot is right-open
    # Pinned states; the linear schedule's exact products are not pushed up by rounding.
    pinned = block_training_state(batch, KappaSchedule("linear"), budgets, MASK, budget=4, step_index=2, prefix=True,
                                  generator=torch.Generator().manual_seed(1))
    assert set(pinned.t.flatten().tolist()) == {0.5} and bool(pinned.prefix.all())
    assert bool((pinned.is_mask == (torch.arange(block) >= 4)).all())           # ceil(0.5 * 8) = 4
    kappas = torch.tensor([0.0, 0.25, 0.3, 1.0, 3 * (1 / 3) / 3], dtype=torch.float64)
    assert prefix_count(kappas, 12).tolist() == [0, 3, 4, 12, 4]
    assert right_open(torch.tensor([[True, False, True, True], [False, True, True, False]])).tolist() == \
        [[False, False, True, True], [False, False, False, False]]
    for bad in ({"prefix_fraction": 1.5}, {"seam_slots": -1}, {"budget": 2, "step_index": 2}):
        with pytest.raises(ValueError):
            block_training_state(batch, schedule, budgets, MASK, **bad)
    with pytest.raises(ValueError):
        block_training_state(batch, schedule, [], MASK)


def test_dfm_loss_is_the_weighted_cross_entropy_of_eq12_and_trains_only_the_adapters():
    """With the adapters at their initial values, L_DFM equals the weighted mean of -log p over the masked slots
    with p from transformers' own model under the mask of Eq. 5 (float64 sum); its gradient reaches the LoRA
    factors, the mask vector and the modulation, and activation checkpointing does not change it."""
    model, reference = block_model(adapted=False)
    block = 3
    examples, batch, state = _batch_and_state(block, seed=3)
    loss, stats = dfm_loss(model, batch, state, torch.device("cpu"))
    total, weight, hits = 0.0, 0.0, 0
    with torch.no_grad():
        for b, example in enumerate(examples):
            blocks, p = int(batch.block_valid[b].sum()), len(example.prompt_ids)
            y = [*example.prompt_ids, *batch.target[b, :blocks].flatten().tolist()]
            rows = eq5_rows(p, block, blocks, 0)
            ids = [y[n] if kind == "c" or n == p - 1 + m * block else int(state.slots[b, m, n - (p - 1 + m * block) - 1])
                   for kind, n, m in rows]
            masks = eq5_masks(p, block, blocks, 0, WINDOW)
            logits = reference(input_ids=torch.tensor([ids]), position_ids=torch.tensor([[n for _, n, _ in rows]]),
                               attention_mask={kind: mask[None, None] for kind, mask in masks.items()}).logits[0]
            logp = torch.log_softmax(logits.double(), dim=-1)
            for m in range(blocks):
                for s in range(block):
                    omega = float(state.weights[b, m, s])
                    if omega > 0:
                        row = len(y) + m * (block + 1) + s                      # the row whose output predicts slot s
                        target = int(batch.target[b, m, s])
                        total += omega * -float(logp[row, target])
                        weight += omega
                        hits += int(logp[row].argmax()) == target
    assert math.isclose(float(loss.detach()), total / weight, rel_tol=1e-4)
    assert math.isclose(stats["nll_sum"], total, rel_tol=1e-4) and math.isclose(stats["weight_sum"], weight, rel_tol=1e-6)
    assert stats["hits"] == hits and stats["masked"] == float((state.weights > 0).sum())

    model, _ = block_model()                                                    # random adapters from here on
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    grads = []
    for checkpointing in (False, True):
        model.zero_grad()
        loss, _ = dfm_loss(model, batch, state, torch.device("cpu"), checkpointing=checkpointing)
        loss.backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad})
    assert set(grads[0]) == set(grads[1]) and len(grads[0]) == 98
    for name, grad in grads[0].items():
        torch.testing.assert_close(grads[1][name], grad, rtol=1e-4, atol=1e-6, msg=name)
    for family in ("lora_b", "lora_a", "mask_embedding", "mod_tables", "mod_proj.weight", "t_mlp.0.weight"):
        assert any(family in n and float(g.abs().sum()) > 0 for n, g in grads[0].items()), family
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None and torch.equal(p, frozen[name]), name
    model.zero_grad()                                                           # adapters off: the frozen backbone
    off, _ = dfm_loss(model, batch, state, torch.device("cpu"), adapters=False)  # reads the same states, and only
    off.backward()                                                              # the mask vector is still an input
    reached = {n for n, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert reached == {"realizer.mask_embedding"}


# =============================================================================================
# 10. The pass audit
# =============================================================================================
def test_a_response_costs_the_prefill_and_the_blocks_passes_and_the_audit_catches_any_other_count():
    model, _ = block_model(registers=1)
    realizer, block, steps, blocks = model.realizer, 3, 3, 3
    kv, layers, window_rows = realizer.num_kv_layers, realizer.num_layers, block + 1 + 1
    prompt = [BOS, 50, 51, 52, 53, 54, 55, 56]
    cfg = BlockSamplerConfig(block_size=block, num_steps=steps, order="prefix", temperature=0.7)

    def run(**kwargs):
        with torch.no_grad():
            cache = model.prefill([prompt]).expand(2)
            return sample_blocks(model, cache, torch.full((2,), prompt[-1]), cfg, max_blocks=blocks,
                                 forbidden=[PAD, BOS], generator=torch.Generator().manual_seed(2), **kwargs)

    with PassAudit(realizer) as outer:                                          # an audit of our own, around it all
        result = run()
        calls = outer.take()
    assert result.passes == [steps] * blocks and result.network_passes == steps * blocks
    assert result.row_passes == [[steps] * blocks] * 2 and result.mean_row_passes == steps
    for i in range(layers):                                                     # prefill + sum of the blocks' passes
        assert len(calls[i]) == (1 if i < kv else 0) + result.network_passes
    assert calls[0][0] == len(prompt) - 1 and calls[kv] == [window_rows] * (steps * blocks)
    assert calls[0][1:] == ([window_rows] * steps + ([block + window_rows] + [window_rows] * (steps - 1)) * (blocks - 1))
    assert max(calls[layers - 1]) == window_rows                                # no commit row above the KV-storing blocks

    separate = run(fused=False)                                                 # A10: the same output, one more read
    assert separate.tokens == result.tokens and separate.passes == result.passes
    with PassAudit(realizer) as outer:
        run(fused=False)
        calls = outer.take()
    assert len(calls[0]) == 1 + steps * blocks + (blocks - 1) and len(calls[kv]) == steps * blocks

    window = model.window_pass

    def twice(*args, **kwargs):
        window(*args, **kwargs)
        return window(*args, **kwargs)

    model.window_pass = twice                                                   # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="pass audit"):
        run()                                                                   # an extra pass
    model.window_pass = lambda cache, anchor, slots, *a, **k: torch.zeros(slots.shape[0], block + 1, 64)  # type: ignore
    with pytest.raises(RuntimeError, match="pass audit"):
        run()                                                                   # a missing pass
    model.window_pass = window                                                  # type: ignore[method-assign]
    fused = model.fused_pass

    def leaking(cache, commit, anchor, slots, is_mask, t, h, *args, **kwargs):
        """One pass per block as claimed, but the commit rows go through every block of the backbone."""
        if commit is None:
            return fused(cache, commit, anchor, slots, is_mask, t, h, *args, **kwargs)
        rows = torch.cat([commit, anchor[:, None], slots, slots[:, :1]], dim=1)     # |G| + |W| rows
        hidden = realizer.canvas_hidden(cache, rows, torch.zeros_like(rows, dtype=torch.bool), t, h)
        return hidden[:, block: 2 * block + 1]

    model.fused_pass = leaking                                                  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="pass audit"):
        run()
    model.fused_pass = fused                                                    # type: ignore[method-assign]
    assert run().tokens == result.tokens                                        # and the sampler is seeded

    with PassAudit(realizer) as audit:
        with torch.no_grad():
            model.prefill([prompt])
        audit.check_prefill(len(prompt) - 1)
        with torch.no_grad():
            model.prefill([prompt])
        with pytest.raises(RuntimeError, match="pass audit"):
            audit.check_prefill(len(prompt))


def test_cache_check_tells_a_pure_commit_from_a_leak():
    """The spot check of the real-model runs, in float32: the fused commits' cache is the frozen backbone's to
    rounding, like a cache built by separate commit passes, and adapters acting on commit rows would show."""
    from chat_vpa.block_evaluate import cache_check

    model, _ = block_model()
    generator = torch.Generator().manual_seed(11)
    prompt = [BOS, *torch.randint(5, VOCAB, (9,), generator=generator).tolist()]
    response = torch.randint(5, VOCAB, (22,), generator=generator).tolist()
    found = cache_check(model, prompt, response, 4, torch.device("cpu"))
    assert found["blocks"] == 5 and found["committed_positions"] == len(prompt) + 4 * 4 - 1
    assert found["fused_commit"] < 1e-4 and found["separate_commit"] < 1e-4 and 0.5 < found["reference_rms"] < 2.0
    assert found["adapters_on"] > 100 * max(found["fused_commit"], found["separate_commit"])
    assert cache_check(model, prompt, response[:7], 4, torch.device("cpu")) == {"blocks": 1.0}


def test_block_cache_only_grows_by_the_next_positions():
    model, _ = block_model()
    with torch.no_grad():
        cache = model.prefill([[BOS, 5, 6, 7], [BOS, 8]])
        one = cache.expand(1) if cache.pos.shape[0] == 1 else cache.select(torch.tensor([1, 0]))
    assert one.lengths.tolist() == [1, 3] and cache.lengths.tolist() == [3, 1]
    keys = {i: k[:, :, :2] for i, k in cache.keys.items()}
    with pytest.raises(ValueError, match="continue"):
        cache.append(keys, keys, torch.tensor([[3, 4], [2, 3]]))               # row 1 would skip a position
    with pytest.raises(ValueError, match="same KV-storing blocks"):
        cache.append({0: keys[0]}, {0: keys[0]}, torch.tensor([[3, 4], [1, 2]]))
    cache.append(keys, keys, torch.tensor([[3, 4], [1, 2]]))
    assert cache.lengths.tolist() == [5, 3] and cache.pos[1].tolist() == [0, 0, 0, 1, 2]
    with pytest.raises(ValueError):
        model.prefill([[BOS]])
