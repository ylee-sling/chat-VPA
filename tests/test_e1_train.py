# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""E1 Step 1's trainer of A2 and A3-0 (``chat_vpa.e1_train``, docs/e1.md Sections 11.4 and 11.5), against
references computed independently: the drafter's batched, unrolled training pass against Transformers' own
single-query calls of ``Gemma4AssistantForCausalLM`` as its candidate generator makes them (float64, the tiny
Gemma 4 with a sliding window shorter than the sequences, both attention implementations); the loss against an
explicit loop; the drafter's clustered choice against ``Gemma4AssistantMaskedEmbedder``; the chunked
cross-entropy against the full one, values and gradients; the cycle count by hand; the command line resuming to
the same weights, and its final directory loading as A1 does and verifying exactly in float32."""

from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4AssistantForCausalLM"):          # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4 drafter", allow_module_level=True)

from test_e1 import CHAT_TEMPLATE  # noqa: E402
from test_stage0 import VOCAB, WINDOW  # noqa: E402
from test_stage0_cli import write_checkpoint  # noqa: E402

from chat_vpa import e1_rollouts as ro  # noqa: E402
from chat_vpa import e1_train as tr  # noqa: E402
from chat_vpa.e1_common import chat_ids, load_target  # noqa: E402

CENTROIDS, TOP = 16, 4                                  # 320 tokens in 16 clusters of 20, the top 4 scored


def _drafter_config():
    text = transformers.Gemma4TextConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=1, head_dim=16, global_head_dim=32, sliding_window=WINDOW,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        final_logit_softcapping=None, vocab_size_per_layer_input=0, hidden_size_per_layer_input=0,
        num_kv_shared_layers=4, use_double_wide_mlp=False, max_position_embeddings=1024)
    return transformers.Gemma4AssistantConfig(text_config=text, backbone_hidden_size=64, use_ordered_embeddings=True,
                                              num_centroids=CENTROIDS, centroid_intermediate_top_k=TOP)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """The tiny instruction-tuned target with a chat template, and a drafter for it with a clustered output layer,
    random weights of a useful size and a random assignment of tokens to clusters."""
    root = tmp_path_factory.mktemp("e1-train")
    write_checkpoint(root / "model")
    tokenizer = transformers.AutoTokenizer.from_pretrained(root / "model")
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(root / "model")
    torch.manual_seed(5)
    drafter = transformers.Gemma4AssistantForCausalLM(_drafter_config())
    generator = torch.Generator().manual_seed(6)
    with torch.no_grad():
        for parameter in drafter.parameters():
            parameter.add_(0.2 * torch.randn(parameter.shape, generator=generator))
        drafter.masked_embedding.token_ordering.copy_(torch.randperm(VOCAB, generator=generator))
    drafter.save_pretrained(root / "assistant")
    (root / "assistant" / "README.md").write_text("A1's other files travel with A2.\n")
    return root


def _models(tiny, dtype, implementation="sdpa"):
    target = transformers.Gemma4ForConditionalGeneration.from_pretrained(tiny / "model", dtype=dtype).eval()
    drafter = transformers.Gemma4AssistantForCausalLM.from_pretrained(
        tiny / "assistant", dtype=dtype, attn_implementation=implementation).eval()
    return target, drafter


def _sequences():
    """Two rollouts longer than the 6-token window: prompts of 5 and 9 tokens, responses of 12 and 6 (the second
    ended by its stop token 1, the first by the budget)."""
    generator = torch.Generator().manual_seed(7)
    first = [2] + torch.randint(6, VOCAB, (16,), generator=generator).tolist()
    second = [2] + torch.randint(6, VOCAB, (13,), generator=generator).tolist() + [1]
    return [{"id": "a", "ids": first, "start": 5}, {"id": "b", "ids": second, "start": 9}]


def _reference_calls(target, drafter, ids, q, inputs_of):
    """Transformers' own calls at one drafted position q: the first cycle after a prefill of x_0 .. x_{q-1}, as
    ``SinglePositionMultiTokenCandidateGenerator.get_candidates`` makes it (the keys cut at the sequence length,
    the 2D mask of its length, every query at q). ``inputs_of(k, previous)`` gives depth k's input embedding."""
    with torch.no_grad():
        out = target(input_ids=torch.tensor([ids[:q]]), output_hidden_states=True, return_shared_kv_states=True,
                     use_cache=False)
        kv = {name: (v[0][:, :, :q + 1], v[1][:, :, :q + 1]) for name, v in out.shared_kv_states.items()}
        previous = out.hidden_states[-1][:, -1:]
        inner, projected, logits = [], [], []
        k = 1
        while (inputs := inputs_of(k, previous)) is not None:
            call = drafter(inputs_embeds=inputs, attention_mask=torch.ones(1, q + 1, dtype=torch.long),
                           position_ids=torch.tensor([[q]]), shared_kv_states=kv, use_cache=False,
                           output_hidden_states=True)
            inner.append(call.hidden_states[-1][0, 0])
            projected.append(call.last_hidden_state[0, 0])
            logits.append(call.logits[0, 0])
            previous = call.last_hidden_state
            k += 1
    return inner, projected, logits


# Gemma 4's eager attention takes its softmax in float32 whatever the precision of its inputs, so in float64 its
# two code paths agree only to float32's rounding, carried through four normalised layers (seen: 2.4e-6 on states
# of order 1); SDPA computes in float64 throughout. A wrong mask would differ at order 1.
TOLERANCE = {"sdpa": {"rtol": 1e-10, "atol": 1e-12}, "eager": {"rtol": 1e-4, "atol": 2e-5}}


@pytest.mark.parametrize("implementation", ["sdpa", "eager"])
def test_the_unrolled_pass_is_transformers_own_calls_of_the_drafter(tiny, implementation):
    target, drafter = _models(tiny, torch.float64, implementation)
    tolerance = TOLERANCE[implementation]
    examples = _sequences()
    batch = tr.collate(examples, pad_id=0)
    depths = 4
    with torch.no_grad():
        last, targets, scored = tr.a2_hidden(drafter, tr.target_states(target, batch), batch, depths)
    embed = target.get_input_embeddings()
    rows = float(drafter.lm_head.weight.detach().norm(dim=1).max())
    checked = 0
    for b, example in enumerate(examples):
        ids, start = example["ids"], example["start"]
        for i, q in enumerate(range(start, len(ids) - 1)):

            def inputs_of(k, previous, ids=ids, q=q):
                if k > depths or q + k > len(ids) - 1:
                    return None
                return torch.cat([embed(torch.tensor([[ids[q + k - 1]]])), previous], dim=-1)

            inner, projected, logits = _reference_calls(target, drafter, ids, q, inputs_of)
            for k in range(depths):
                assert bool(scored[b, i, k]) == (k < len(inner))
                if k >= len(inner):
                    continue
                assert int(targets[b, i, k]) == ids[q + k + 1]
                torch.testing.assert_close(last[b, i, k], inner[k], **tolerance)
                torch.testing.assert_close(drafter.post_projection(last[b, i, k]), projected[k], **tolerance)
                # The clustered output: the same logits on the tokens it scores, the same choice. A logit's error
                # is at most the hidden state's, in the 2-norm, times the norm of its row of the output layer.
                full = drafter.lm_head(last[b, i, k])
                kept = logits[k] > logits[k].min()
                torch.testing.assert_close(full[kept], logits[k][kept], rtol=tolerance["rtol"],
                                           atol=tolerance["atol"] * math.sqrt(last.shape[-1]) * rows)
                clustered, _, _ = tr.drafter_choices(drafter, last[b, i, k][None], targets[b, i, k][None], 7)
                assert int(clustered[0]) == int(logits[k].argmax())
                checked += 1
    # By hand: the first rollout drafts at q = 5 .. 15 (11 positions), depth k + 1 having a target where
    # q + k + 1 <= 16; the second at q = 9 .. 13 (5 positions), where q + k + 1 <= 14.
    assert checked == (11 + 10 + 9 + 8) + (5 + 4 + 3 + 2)
    assert int(scored[:, :, 0].sum()) == 11 + 5


def test_a3_slots_are_single_calls_at_the_last_known_position(tiny):
    target, drafter = _models(tiny, torch.float64)
    generator = torch.Generator().manual_seed(8)
    realiser = tr.OnePassRealiser(drafter, 3, torch.randn(64, generator=generator, dtype=torch.float64))
    with torch.no_grad():
        realiser.depth.copy_(torch.randn(3, 64, generator=generator, dtype=torch.float64))
    examples = _sequences()
    batch = tr.collate(examples, pad_id=0)
    with torch.no_grad():
        last, targets, scored = realiser(tr.target_states(target, batch), batch)
    assert last.shape[2] == 3
    for b, example in enumerate(examples):
        ids, start = example["ids"], example["start"]
        for i, q in enumerate(range(start, len(ids) - 1)):
            for j in range(3):

                def inputs_of(k, previous, j=j):
                    if k > 1:
                        return None
                    slot = (realiser.code + realiser.depth[j]).view(1, 1, -1)
                    return torch.cat([slot, previous], dim=-1)

                inner, _, _ = _reference_calls(target, drafter, ids, q, inputs_of)
                torch.testing.assert_close(last[b, i, j], inner[0], rtol=1e-10, atol=1e-12)
                assert bool(scored[b, i, j]) == (q + j + 1 <= len(ids) - 1)
                if scored[b, i, j]:
                    assert int(targets[b, i, j]) == ids[q + j + 1]


def test_the_loss_is_the_mean_over_depths_of_each_depths_mean(tiny):
    target, drafter = _models(tiny, torch.float64)
    batch = tr.collate(_sequences(), pad_id=0)
    with torch.no_grad():
        last, targets, scored = tr.a2_hidden(drafter, tr.target_states(target, batch), batch, 4)
    clusters = tr.cluster_map(drafter)
    # The cluster of a token by an explicit walk over token_ordering: cluster c lists 20 tokens from 20 c.
    ordering = drafter.masked_embedding.token_ordering.tolist()
    walk = {}
    for position, token in enumerate(ordering):
        walk[token] = position // (VOCAB // CENTROIDS)
    assert clusters.tolist() == [walk[t] for t in range(VOCAB)]
    loss, stats = tr.depth_loss(drafter, last, targets, scored, clusters, 0.5, chunk=3)
    per_depth = []
    for k in range(4):
        terms = []
        for b in range(last.shape[0]):
            for i in range(last.shape[1]):
                if not scored[b, i, k]:
                    continue
                h, t = last[b, i, k], int(targets[b, i, k])
                token = -torch.log_softmax(drafter.lm_head(h), -1)[t]
                cluster = -torch.log_softmax(drafter.masked_embedding.centroids(h), -1)[walk[t]]
                terms.append(token + 0.5 * cluster)
        assert stats["count"][k] == len(terms)
        per_depth.append(sum(terms) / len(terms))
    torch.testing.assert_close(loss, sum(per_depth) / 4, rtol=1e-10, atol=1e-12)
    with pytest.raises(ValueError, match="clustered"):
        tr.depth_loss(drafter, last, targets, scored, None, 1.0, chunk=3)
    bad = drafter.masked_embedding.token_ordering.clone()
    drafter.masked_embedding.token_ordering[0] = bad[1]
    with pytest.raises(ValueError, match="permutation"):
        tr.cluster_map(drafter)
    drafter.masked_embedding.token_ordering.copy_(bad)


def _three_sequences():
    generator = torch.Generator().manual_seed(11)
    return [*_sequences(), {"id": "c", "ids": [2] + torch.randint(6, VOCAB, (9,), generator=generator).tolist() + [1],
                            "start": 4}]


def test_micro_batches_add_up_to_the_batchs_loss_and_gradient(tiny):
    target, drafter = _models(tiny, torch.float64)
    clusters = tr.cluster_map(drafter)
    examples = _three_sequences()
    params = [p for p in drafter.parameters() if p.requires_grad]

    def run(groups, counts):
        drafter.zero_grad(set_to_none=True)
        total = 0.0
        for group in groups:
            batch = tr.collate(group, 0)
            last, targets, scored = tr.a2_hidden(drafter, tr.target_states(target, batch), batch, 4)
            loss, _ = tr.depth_loss(drafter, last, targets, scored, clusters, 0.7, 5, counts)
            loss.backward()
            total += float(loss.detach())
        return total, [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]

    # The reference: the whole batch at once, its own counts.
    whole, reference = run([examples], None)
    batch = tr.collate(examples, 0)
    positions, valid = tr.query_positions(batch)
    counts = [float(tr._targets(batch, positions, valid, 4)[1][:, :, k].sum()) for k in range(4)]
    # By hand: the third rollout (11 tokens, start 4) drafts at q = 4 .. 9, depth k having 7 - k positions.
    assert counts == [11 + 5 + 6, 10 + 4 + 5, 9 + 3 + 4, 8 + 2 + 3]
    for groups in ([examples[:1], examples[1:]], [[e] for e in examples], [examples[:2], examples[2:]]):
        total, grads = run(groups, counts)
        assert total == pytest.approx(whole, rel=1e-12)                      # the forward is the batch's exactly
        for got, want in zip(grads, reference, strict=True):
            # Gemma 4's RMSNorm computes in float32 whatever its input's precision, and the backward sums its
            # rows over a micro-batch instead of the batch: float32's rounding, relative to each gradient's scale
            # (seen: 1.7e-7). A wrong normalisation would differ by a factor of order one.
            assert float((got - want).abs().max()) <= 1e-5 * float(want.abs().max())


def test_a_batch_that_runs_out_of_memory_is_run_again_in_halves(tiny, tmp_path):
    from chat_vpa.train_e0 import EpochStream

    examples = _three_sequences()

    def trainer(fail_above=None, error=torch.cuda.OutOfMemoryError):
        target, drafter = _models(tiny, torch.float64)
        cfg = tr.TrainConfig(arm="a2", batch=3, steps=2, vocab_chunk=5).finish(len(examples))
        made = tr.DrafterTrainer(cfg, target, drafter, drafter, EpochStream(examples, 0), 0, tmp_path,
                                 torch.device("cpu"))
        if fail_above is not None:
            original = made.hidden

            def hidden(batch):
                if batch.ids.shape[0] > fail_above:
                    raise error("CUDA out of memory (a stand-in)")
                return original(batch)

            made.hidden = hidden
        return made

    plain = trainer()
    assert plain.micro_batch == 3                                     # A2: the whole batch, as timed
    expected = [plain.train_step()["loss"] for _ in range(2)]
    halved = trainer(fail_above=1)
    rows = [halved.train_step() for _ in range(2)]
    assert halved.micro_batch == 1 and halved.halvings == 1           # 3, then 3 // 2 = 1 for the rest of the run
    assert [r["micro_batch"] for r in rows] == [1.0, 1.0]
    assert rows[0]["loss"] == pytest.approx(expected[0], rel=1e-12)    # the same weights: the same forward
    assert rows[1]["loss"] == pytest.approx(expected[1], rel=1e-5)
    # The gradients agree to float32's rounding (Gemma 4's RMSNorm, above); AdamW moves an element by at most about
    # lr per step, so an element whose gradient is near zero may move the other way: at most 2 lr per step apart.
    for a, b in zip(plain.drafter.parameters(), halved.drafter.parameters(), strict=True):
        assert float((a - b).detach().abs().max()) <= 2 * 2 * plain.cfg.lr * 1.01
    with pytest.raises(ValueError, match="stand-in"):                 # anything else is not retried
        trainer(fail_above=0, error=ValueError).train_step()


def test_the_chunked_cross_entropy_is_the_full_one_with_its_gradients():
    generator = torch.Generator().manual_seed(9)
    hidden = torch.randn(11, 5, dtype=torch.float64, generator=generator, requires_grad=True)
    weight = torch.randn(13, 5, dtype=torch.float64, generator=generator, requires_grad=True)
    targets = torch.randint(0, 13, (11,), generator=generator)
    full = F.cross_entropy(hidden @ weight.t(), targets, reduction="sum")
    grads = torch.autograd.grad(full, (hidden, weight))
    for chunk in (1, 3, 4, 11, 50):
        value = tr.vocab_cross_entropy(hidden, weight, targets, chunk)
        torch.testing.assert_close(value, full, rtol=1e-12, atol=1e-12)
        for got, want in zip(torch.autograd.grad(value, (hidden, weight)), grads, strict=True):
            torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        tr.vocab_cross_entropy(hidden, weight, targets, 0)


def test_the_drafters_choice_is_its_masked_embedders(tiny):
    _, drafter = _models(tiny, torch.float64)
    generator = torch.Generator().manual_seed(10)
    hidden = torch.randn(17, 32, dtype=torch.float64, generator=generator)
    targets = torch.randint(0, VOCAB, (17,), generator=generator)
    clustered, overall, nll = tr.drafter_choices(drafter, hidden, targets, chunk=5)
    with torch.no_grad():
        masked = drafter.masked_embedding(hidden[None], drafter.lm_head.weight)[0]
        logits = drafter.lm_head(hidden)
    assert clustered.tolist() == masked.argmax(-1).tolist()
    assert overall.tolist() == logits.argmax(-1).tolist()
    assert (clustered != overall).any()                                  # the top clusters do bind here
    torch.testing.assert_close(nll, F.cross_entropy(logits, targets, reduction="none"), rtol=1e-12, atol=1e-12)


def test_tokens_per_cycle_by_hand():
    t, f = True, False
    hits = [[t, t, t, f], [f, f, f, f], [f, f, f, f], [f, t, t, t], [t, t, f, f], [t, f, f, f]]
    # G = 2: q0 accepts 2 (adds 3), q3 none (adds 1), q4 one (its second draft would be the last token: adds 2).
    assert tr.tokens_per_cycle(hits, 2) == (3, 6)
    assert tr.tokens_per_cycle(hits, 4) == (2, 6)                      # q0 accepts 3, then q4 one
    assert tr.tokens_per_cycle(hits, 1) == (4, 6)                      # q0 one, q2 none, q3 none, q4 one
    assert tr.tokens_per_cycle([[t, t]], 2) == (1, 1)
    assert tr.tokens_per_cycle([], 2) == (0, 0)


def test_positions_targets_and_visibility_by_hand():
    batch = tr.collate([{"id": "a", "ids": [2, 7, 8, 9, 1], "start": 2}, {"id": "b", "ids": [2, 7, 8], "start": 1}], 0)
    positions, valid = tr.query_positions(batch)
    # By hand: row a drafts at q = 2, 3 (each has a next token: 9, then 1); row b at q = 1, padded with its start.
    assert valid.tolist() == [[True, True], [True, False]]
    assert positions.tolist() == [[2, 3], [1, 1]]
    full, sliding = tr.visibility(positions, keys=5, window=1)
    assert full[0, 0].int().tolist() == [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]]
    assert sliding[0, 0].int().tolist() == [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]]     # the 2 = W + 1 before q
    targets, scored = tr._targets(batch, positions, valid, 3)
    assert scored.int().tolist() == [[[1, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 0]]]
    assert targets[scored].tolist() == [9, 1, 1, 8]                     # x3, x4 from q = 2; x4 from 3; row b's x2
    examples, dropped = tr.prepare([{"id": "x", "prompt_ids": [2, 5], "response_ids": [7], "stop_id": None},
                                    {"id": "y", "prompt_ids": [2, 5], "response_ids": [7], "stop_id": 1},
                                    {"id": "z", "prompt_ids": [2], "response_ids": [], "stop_id": 1}])
    assert dropped == 2 and examples == [{"id": "y", "ids": [2, 5, 7, 1], "start": 2}]
    with pytest.raises(ValueError, match="sdpa"):
        tr.attention_masks(full, sliding, "flash_attention_2", torch.float32)


# ---------------------------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rollouts(tiny, tmp_path_factory):
    """Six rollouts of the tiny target, written by the rollout writer."""
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    root = tmp_path_factory.mktemp("e1-train-rollouts")
    texts = ["Say hello.", "Count to four.", "Name two colours.", "Spell cat.", "Add two and two.", "Say no."]
    pq.write_table(pa.table({"prompt": texts, "prompt_id": [f"p{i}" for i in range(6)],
                             "messages": [[] for _ in texts]}), root / "train_sft-0.parquet")
    assert ro.main(["write", "--model", str(tiny / "model"), "--parquet", str(root / "train_sft-0.parquet"),
                    "--count", "6", "--chunk", "4", "--batch-size", "3", "--max-new-tokens", "10", "--dtype", "fp32",
                    "--device", "cpu", "--out", str(root / "train")]) == 0
    return root / "train"


def _train(tiny, rollouts, run, arm, stop_at=None, extra=()):
    args = ["--arm", arm, "--model", str(tiny / "model"), "--assistant", str(tiny / "assistant"), "--rollouts",
            str(rollouts), "--held-out", str(rollouts), "--held-out-limit", "3", "--run", str(run), "--device", "cpu",
            "--set", "batch=2", "steps=6", "log_every=1", "save_every=2", "eval_every=3", "vocab_chunk=7", *extra]
    if stop_at is not None:
        args += ["--stop-at", str(stop_at)]
    return tr.main(args)


def test_a2_resumes_to_the_same_weights_and_its_final_directory_is_a_drafter(tiny, rollouts, tmp_path):
    assert _train(tiny, rollouts, tmp_path / "whole", "a2") == 0
    assert _train(tiny, rollouts, tmp_path / "parts", "a2", stop_at=4) == 0
    assert not (tmp_path / "parts" / "final").exists()
    timing = json.loads((tmp_path / "parts" / "timing.json").read_text())
    assert timing["steps_run"] == 4 and timing["horizon_steps"] == 6 and timing["predicted_hours"] > 0
    assert _train(tiny, rollouts, tmp_path / "parts", "a2") == 0                # resumes at step 4
    whole = transformers.Gemma4AssistantForCausalLM.from_pretrained(tmp_path / "whole" / "final")
    parts = transformers.Gemma4AssistantForCausalLM.from_pretrained(tmp_path / "parts" / "final")
    start = transformers.Gemma4AssistantForCausalLM.from_pretrained(tiny / "assistant")
    for (name, a), (_, b), (_, c) in zip(whole.state_dict().items(), parts.state_dict().items(),
                                         start.state_dict().items(), strict=True):
        assert torch.equal(a, b), name
        if name.endswith("token_ordering"):
            assert torch.equal(a, c)                                     # the clusters stay as they were (S6)
    assert any(not torch.equal(a, c) for a, c in zip(whole.parameters(), start.parameters(), strict=True))
    assert (tmp_path / "whole" / "final" / "README.md").exists()           # A1's other files travel with it
    # The metrics: train rows at every step, held-out curves at steps 0, 3 and 6.
    rows = [json.loads(line) for line in open(tmp_path / "whole" / "metrics.jsonl", encoding="utf-8")]
    evals = [r for r in rows if "eval/hit_1" in r]
    assert [r["step"] for r in evals] == [0, 3, 6]
    assert all(1.0 <= r["eval/tokens_per_cycle_2"] <= 3.0 for r in evals)
    assert [r["step"] for r in rows if "train/loss" in r] == [1, 2, 3, 4, 5, 6]
    config = json.loads((tmp_path / "whole" / "config.json").read_text())
    assert config["config"]["cluster_weight"] == 1.0 and config["rollouts_used"] == 6
    # Another configuration in the same run directory is refused.
    with pytest.raises(SystemExit, match="another configuration"):
        _train(tiny, rollouts, tmp_path / "parts", "a2", extra=("lr=0.001",))
    # A2 loads as A1 does and, verified by the target, changes no token (float32).
    target = transformers.Gemma4ForConditionalGeneration.from_pretrained(tiny / "model", dtype=torch.float32).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny / "model")
    ids = torch.tensor([chat_ids(tokenizer, "Count to four.")])
    whole.generation_config.num_assistant_tokens = 3
    whole.generation_config.num_assistant_tokens_schedule = "constant"
    kwargs = {"max_new_tokens": 12, "do_sample": False, "temperature": None, "top_p": None, "top_k": None,
              "attention_mask": torch.ones_like(ids)}
    with torch.no_grad():
        plain = target.generate(ids, **kwargs)
        drafted = target.generate(ids, assistant_model=whole.eval(), **kwargs)
    assert torch.equal(plain, drafted)


def test_a3_0_trains_its_code_and_saves_the_realiser(tiny, rollouts, tmp_path):
    assert _train(tiny, rollouts, tmp_path / "a3", "a3-0", extra=("slots=3", "steps=2")) == 0
    saved = torch.load(tmp_path / "a3" / "final" / "realiser.pt", weights_only=True)
    assert saved["slots"] == 3 and saved["depth"].shape == (3, 64)
    assert saved["depth"].abs().sum() > 0                                  # zero at the start, trained since
    target, tokenizer = load_target(tiny / "model", "bf16", torch.device("cpu"))     # as the trainer loads it
    start = tr.mask_code(target, tokenizer)
    moved = (saved["code"] - start).abs().max()
    assert 0 < moved < 1e-3                         # it starts at the mask token's embedding; two small steps since
    rows = [json.loads(line) for line in open(tmp_path / "a3" / "metrics.jsonl", encoding="utf-8")]
    evals = [r for r in rows if "eval/hit_1" in r]
    assert [r["step"] for r in evals] == [0, 2] and all("eval/tokens_per_cycle_3" in r for r in evals)
    config = json.loads((tmp_path / "a3" / "config.json").read_text())
    # A2's per-position loss, cluster term included (the author's decision of 27 November 2026), in micro-batches
    # of at most 8.
    assert config["config"]["cluster_weight"] == 1.0 and config["config"]["micro_batch"] == 2
    assert transformers.Gemma4AssistantForCausalLM.from_pretrained(tmp_path / "a3" / "final") is not None


# ---------------------------------------------------------------------------------------------
# What Transformers hands its drafter after a partly accepted cycle
# ---------------------------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Transformers 5.21's SinglePositionMultiTokenCandidateGenerator cuts the "
                   "target's shared keys at the sequence length, so after a partly accepted cycle the drafter "
                   "attends to the keys of rejected drafts (docs/e1.md, Section 11.9)")
def test_transformers_hands_its_drafter_only_the_accepted_prefixs_keys(tiny):
    target, drafter = _models(tiny, torch.float32)
    calls = []
    original = drafter.forward

    def spy(*args, **kwargs):
        calls.append({"q": int(kwargs["position_ids"][0, 0]),
                      "kv": {n: (v[0].clone(), v[1].clone()) for n, v in kwargs["shared_kv_states"].items()}})
        return original(*args, **kwargs)

    drafter.forward = spy
    drafter.generation_config.num_assistant_tokens = 3
    drafter.generation_config.num_assistant_tokens_schedule = "constant"
    prompt = torch.tensor([[2] + list(range(10, 21))])
    with torch.no_grad():
        output = target.generate(prompt, attention_mask=torch.ones_like(prompt), assistant_model=drafter,
                                 max_new_tokens=16, do_sample=False, temperature=None, top_p=None, top_k=None)
    sequence = output[0].tolist()
    assert len({c["q"] for c in calls}) > 2
    for call in calls:
        q = call["q"]
        with torch.no_grad():
            clean = target.model.language_model(input_ids=torch.tensor([sequence[:q]]), return_shared_kv_states=True,
                                                use_cache=False).shared_kv_states
        full = call["kv"]["full_attention"][0][:, :, :q + 1]
        assert full.shape[2] == q                                   # the keys of x_0 .. x_{q-1}, nothing at q
        # float32's rounding between the cached passes and one fresh pass, relative to the keys' scale
        torch.testing.assert_close(full, clean["full_attention"][0], rtol=1e-5,
                                   atol=1e-5 * float(clean["full_attention"][0].abs().max()))
        sliding = call["kv"]["sliding_attention"][0][:, :, -(WINDOW + 1):]
        torch.testing.assert_close(sliding, clean["sliding_attention"][0][:, :, -(WINDOW + 1):], rtol=1e-5,
                                   atol=1e-5 * float(clean["sliding_attention"][0].abs().max()))
