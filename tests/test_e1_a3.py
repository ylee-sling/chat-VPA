# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""E1 Step 1's A3-oracle and A3 (``chat_vpa.e1_train``, docs/e1.md Sections 6.1, 11.5 and 11.13) and A3's one-pass
decoding (``chat_vpa.e1_decode``), against references computed independently: the block encoder against an explicit
loop; the codebook's nearest code against brute force, its moving averages and restarts by hand, its gradient as
the straight-through estimator; the coded slots and the predictor's layer against Transformers' own single-query
calls of the drafter (float64); the command line's runs of A3-oracle and A3 (what each trains and what it leaves as
it was); micro-batches adding up to the batch; and the one-pass decoding loop, whose batched call equals separate
calls and whose output is the model's own greedy text (float32)."""

from __future__ import annotations

import json
import math
import shutil

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4AssistantForCausalLM"):          # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4 drafter", allow_module_level=True)

from test_e1_train import _models, _reference_calls, _sequences, _train, rollouts, tiny  # noqa: E402,F401

from chat_vpa import e1_decode as de  # noqa: E402
from chat_vpa import e1_train as tr  # noqa: E402
from chat_vpa.e1_common import chat_ids  # noqa: E402

SLOTS = 3


# ---------------------------------------------------------------------------------------------
# The encoder and the codebook
# ---------------------------------------------------------------------------------------------

def test_the_block_encoder_is_an_explicit_loop():
    torch.manual_seed(0)
    encoder = tr.BlockEncoder(hidden=8, slots=3, code_dim=5, width=4).double()
    hidden = torch.randn(2, 7, 8, dtype=torch.float64)
    positions = torch.tensor([[1, 2, 3], [4, 5, 5]])
    length = torch.tensor([7, 6])
    got = encoder(hidden, positions, length)
    for b in range(2):
        for i in range(3):
            q = int(positions[b, i])
            parts = []
            for j in range(1, 4):
                if q + j <= int(length[b]) - 1:
                    x = hidden[b, q + j]
                    x = (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + 1e-5)
                    parts.append(F.gelu(encoder.inner(x)))
                else:
                    parts.append(torch.zeros(4, dtype=torch.float64))
            torch.testing.assert_close(got[b, i], encoder.outer(torch.cat(parts)), rtol=1e-10, atol=1e-12)


def test_the_codebook_by_hand():
    torch.manual_seed(1)
    q = tr.VectorQuantiser(codes=3, dim=2, decay=0.5, beta=0.25, restart_below=0.1)
    with torch.no_grad():
        q.embed.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]))
        q.embed_sum.copy_(q.embed)
        q.cluster_size.fill_(1.0)
        q.initialised.fill_(True)
    encoding = torch.tensor([[[0.1, 0.1], [0.9, -0.2], [0.2, 1.7], [5.0, 5.0]]], requires_grad=True)
    valid = torch.tensor([[True, True, True, False]])
    q.train()
    code, index, commitment = q(encoding, valid)
    brute = torch.stack([((encoding.detach()[0, n] - q.embed) ** 2).sum(-1).argmin() for n in range(4)])
    assert index[0].tolist() == brute.tolist() == [0, 1, 2, 2]
    torch.testing.assert_close(code.detach(), q.embed[index])
    # The commitment loss over the valid positions: beta x mean over dimensions of the squared distance.
    by_hand = 0.25 * sum(((encoding.detach()[0, n] - q.embed[brute[n]]) ** 2).mean() for n in range(3))
    torch.testing.assert_close(commitment.detach(), torch.as_tensor(float(by_hand)))
    # Straight through: the code's gradient reaches the encoding unchanged.
    (code * torch.arange(8.0).view(1, 4, 2)).sum().backward()
    torch.testing.assert_close(encoding.grad, torch.arange(8.0).view(1, 4, 2))
    # One update per step, from the valid positions only: counts (1, 1, 1); decay 0.5.
    stats = q.apply()
    assert stats["codes_used"] == 3.0
    size = torch.tensor([1.0, 1.0, 1.0])                                  # 0.5 x 1 + 0.5 x 1 each
    sums = 0.5 * torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]) + 0.5 * encoding.detach()[0, :3]
    total = size.sum()
    smoothed = (size + 1e-5) / (total + 3e-5) * total
    torch.testing.assert_close(q.cluster_size, size)
    torch.testing.assert_close(q.embed, sums / smoothed[:, None])
    # A code that the steps leave unused decays (1, 0.5, 0.25, 0.125, 0.0625 at decay 0.5); below restart_below it
    # restarts at one of the step's encodings: the two codes these encodings never reach, at the fourth step.
    step_rows = torch.tensor([[[0.1, 0.0], [0.0, 0.1]]])
    restarts, after = [], []
    for _ in range(5):
        q(step_rows, torch.tensor([[True, True]]))
        restarts.append(q.apply()["codes_restarted"])
        after.append(q.embed.clone())
    assert restarts == [0.0, 0.0, 0.0, 2.0, 0.0] and float(q.cluster_size.min()) >= 0.1 - 1e-9
    assert sum(any(torch.equal(code, row) for row in step_rows[0]) for code in after[3]) == 2
    # Assignments of a batch that is run again are discarded, not counted twice.
    q(torch.tensor([[[0.1, 0.0]]]), torch.tensor([[True]]))
    q.discard()
    assert q.apply() == {}
    # The first training batch starts the codes at its encodings.
    fresh = tr.VectorQuantiser(codes=2, dim=2)
    fresh.train()
    rows = torch.tensor([[[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
    fresh(rows, torch.ones(1, 3, dtype=torch.bool))
    assert bool(fresh.initialised)
    assert all(any(torch.equal(code, row) for row in rows[0]) for code in fresh.embed)


# ---------------------------------------------------------------------------------------------
# The coded slots and the predictor against Transformers' own calls
# ---------------------------------------------------------------------------------------------

def _coded(drafter, target, seed=8, predictor=False):
    generator = torch.Generator().manual_seed(seed)
    width = target.get_input_embeddings().weight.shape[1]
    hidden = target.config.get_text_config().hidden_size
    encoder = tr.BlockEncoder(hidden, SLOTS, width, 4).double()
    quantiser = tr.VectorQuantiser(5, width).double()
    with torch.no_grad():
        quantiser.embed.copy_(torch.randn(5, width, generator=generator, dtype=torch.float64))
        quantiser.initialised.fill_(True)
        for parameter in encoder.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=torch.float64))
    body = tr.CodePredictor(tr.one_layer_drafter(drafter), 5) if predictor else None
    realiser = tr.CodedRealiser(drafter, SLOTS, "add", encoder, quantiser, body).eval()
    with torch.no_grad():
        realiser.depth.copy_(torch.randn(SLOTS, width, generator=generator, dtype=torch.float64))
        if body is not None:
            body.head.weight.copy_(torch.randn(body.head.weight.shape, generator=generator, dtype=torch.float64))
    return realiser


def test_the_oracles_slots_are_transformers_calls_with_the_true_blocks_code(tiny):  # noqa: F811
    target, drafter = _models(tiny, torch.float64)
    realiser = _coded(drafter, target)
    examples = _sequences()
    batch = tr.collate(examples, pad_id=0)
    with torch.no_grad():
        states = tr.target_states(target, batch)
        last, targets, scored, aux = realiser(states, batch)
        full = target(input_ids=batch.ids, attention_mask=batch.valid.long(), output_hidden_states=True,
                      use_cache=False).hidden_states[-1]
    embed = target.get_input_embeddings()
    for b, example in enumerate(examples):
        ids, start = example["ids"], example["start"]
        for i, q in enumerate(range(start, len(ids) - 1)):
            # The code by hand: the encoder on the target's states at q + 1 .. q + 3, then the nearest code.
            encoding = realiser.encoder(full[b:b + 1], torch.tensor([[q]]), torch.tensor([len(ids)]))[0, 0]
            code = realiser.quantiser.embed[((encoding - realiser.quantiser.embed) ** 2).sum(-1).argmin()]
            assert int(aux["index"][b, i]) == int(((encoding - realiser.quantiser.embed) ** 2).sum(-1).argmin())
            for j in range(SLOTS):

                def inputs_of(k, previous, j=j, q=q, ids=ids, code=code):
                    if k > 1:
                        return None
                    slot = embed(torch.tensor([[ids[q]]])) + (code + realiser.depth[j]).view(1, 1, -1)
                    return torch.cat([slot, previous], dim=-1)

                inner, _, _ = _reference_calls(target, drafter, ids, q, inputs_of)
                torch.testing.assert_close(last[b, i, j], inner[0], rtol=1e-10, atol=1e-12)
                assert bool(scored[b, i, j]) == (q + j + 1 <= len(ids) - 1)


def test_the_predictors_layer_is_transformers_own_call_of_a_one_layer_drafter(tiny):  # noqa: F811
    target, drafter = _models(tiny, torch.float64)
    realiser = _coded(drafter, target, predictor=True)
    body = realiser.predictor.body
    # A copy of the realiser's last full-attention layer, input projection and final norm.
    kinds = drafter.config.get_text_config().layer_types
    source = max(i for i, kind in enumerate(kinds) if kind == "full_attention")
    for (name, a), (_, b) in zip(body.model.layers[0].state_dict().items(),
                                 drafter.model.layers[source].state_dict().items(), strict=True):
        assert torch.equal(a, b), name
    assert torch.equal(body.pre_projection.weight, drafter.pre_projection.weight)
    assert body.model.layers[0].state_dict()["self_attn.q_proj.weight"].data_ptr() != \
        drafter.model.layers[source].state_dict()["self_attn.q_proj.weight"].data_ptr()
    examples = _sequences()
    batch = tr.collate(examples, pad_id=0)
    positions, valid = tr.query_positions(batch)
    with torch.no_grad():
        logits = realiser.predictor(tr.target_states(target, batch), positions)
    embed = target.get_input_embeddings()
    for b, example in enumerate(examples):
        ids, start = example["ids"], example["start"]
        for i, q in enumerate(range(start, len(ids) - 1)):

            def inputs_of(k, previous, q=q, ids=ids):
                return None if k > 1 else torch.cat([embed(torch.tensor([[ids[q]]])), previous], dim=-1)

            inner, _, _ = _reference_calls(target, body, ids, q, inputs_of)
            torch.testing.assert_close(logits[b, i], realiser.predictor.head(inner[0]), rtol=1e-10, atol=1e-12)
    # Its head starts at zero: every code equally likely.
    fresh = tr.CodePredictor(tr.one_layer_drafter(drafter), 5)
    assert torch.count_nonzero(fresh.head.weight) == 0 and torch.count_nonzero(fresh.head.bias) == 0


# ---------------------------------------------------------------------------------------------
# The command line: A3-oracle, then A3 from it
# ---------------------------------------------------------------------------------------------

def test_a3_oracle_then_a3_train_what_they_should_and_nothing_else(tiny, rollouts, tmp_path):  # noqa: F811
    extra = ("slots=3", "steps=2", "slot_input=add", "codes=8", "code_width=4")
    assert _train(tiny, rollouts, tmp_path / "oracle", "a3-oracle", extra=extra) == 0
    oracle = torch.load(tmp_path / "oracle" / "final" / "realiser.pt", weights_only=True)
    assert oracle["source"] == "oracle" and oracle["slot_input"] == "add" and "predictor" not in oracle
    assert bool(oracle["quantiser"]["state"]["initialised"]) and oracle["quantiser"]["codes"] == 8
    rows = [json.loads(line) for line in open(tmp_path / "oracle" / "metrics.jsonl", encoding="utf-8")]
    trained = [r for r in rows if "train/loss" in r]
    assert all("train/code_perplexity" in r and r["train/commitment"] >= 0 for r in trained)
    assert all("eval/code_perplexity" in r for r in rows if "eval/hit_1" in r)
    # A3 trains only its predictor; it starts from A3-oracle's final directory.
    args = ["--arm", "a3", "--model", str(tiny / "model"), "--assistant", str(tmp_path / "oracle" / "final"),
            "--rollouts", str(rollouts), "--held-out", str(rollouts), "--held-out-limit", "3", "--run",
            str(tmp_path / "a3"), "--device", "cpu", "--set", "batch=2", "steps=2", "log_every=1", "save_every=2",
            "eval_every=3", "vocab_chunk=7", *extra]
    assert tr.main(args) == 0
    a3 = torch.load(tmp_path / "a3" / "final" / "realiser.pt", weights_only=True)
    assert a3["source"] == "predicted" and "predictor" in a3
    for part in ("encoder", "quantiser"):
        for key, value in oracle[part]["state"].items():
            assert torch.equal(a3[part]["state"][key], value), (part, key)
    assert torch.equal(a3["depth"], oracle["depth"])
    before = transformers.Gemma4AssistantForCausalLM.from_pretrained(tmp_path / "oracle" / "final")
    after = transformers.Gemma4AssistantForCausalLM.from_pretrained(tmp_path / "a3" / "final")
    for (name, a), (_, b) in zip(before.state_dict().items(), after.state_dict().items(), strict=True):
        assert torch.equal(a, b), name
    assert torch.count_nonzero(a3["predictor"]["state"]["head.weight"]) > 0         # zero at the start, trained
    rows = [json.loads(line) for line in open(tmp_path / "a3" / "metrics.jsonl", encoding="utf-8")]
    assert all(0.0 <= r["train/predictor_accuracy"] <= 1.0 for r in rows if "train/loss" in r)
    assert any("eval/predictor_accuracy" in r for r in rows)
    loaded = tr.load_realiser(tmp_path / "a3" / "final", after)
    assert loaded.source == "predicted" and loaded.quantiser.codes == 8
    # A3 refuses anything but A3-oracle's realiser.
    with pytest.raises(SystemExit, match="A3-oracle's realiser"):
        tr.main([*args[:5], str(tiny / "assistant"), *args[6:13], str(tmp_path / "bad"), *args[14:]])


@pytest.mark.parametrize("arm", ["a3-oracle", "a3"])
def test_micro_batches_add_up_for_the_coded_arms(tiny, tmp_path, arm):  # noqa: F811
    target, drafter = _models(tiny, torch.float64)
    examples = [*_sequences(), {"id": "c", "ids": _sequences()[0]["ids"][:12], "start": 4}]

    def gradients(micro):
        torch.manual_seed(3)
        d = transformers.Gemma4AssistantForCausalLM.from_pretrained(tiny / "assistant", dtype=torch.float64)
        realiser = _coded(d, target, predictor=(arm == "a3"))
        if arm == "a3":
            for parameter in realiser.parameters():
                parameter.requires_grad_(False)
            for parameter in realiser.predictor.trainable():
                parameter.requires_grad_(True)
        cfg = tr.TrainConfig(arm=arm, batch=3, steps=1, slots=SLOTS, slot_input="add", codes=5, code_width=4,
                             micro_batch=micro, vocab_chunk=7).finish(3)
        trainer = tr.DrafterTrainer(cfg, target, realiser, d, iter(()), 0, tmp_path, torch.device("cpu"))
        trainer.module.train()
        result = trainer.accumulate(examples, trainer.counts(examples))
        if trainer.quantiser is not None:
            trainer.quantiser.discard()
        # The drafter's post-projection takes no part in a one-pass draft, so it has no gradient.
        return result["loss"], [None if p.grad is None else p.grad.clone() for p in trainer.parameters]

    whole_loss, whole = gradients(3)
    part_loss, parts = gradients(1)
    assert math.isclose(whole_loss, part_loss, rel_tol=1e-6)
    assert sum(g is not None for g in whole) > 0
    # Gemma 4's RMSNorm computes in float32 whatever its input's precision (CLAUDE.md, E1 decisions).
    for a, b in zip(whole, parts, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5 * float(a.abs().max()) + 1e-12)


# ---------------------------------------------------------------------------------------------
# One-pass decoding
# ---------------------------------------------------------------------------------------------

def _a3_dir(tiny, tmp_path, predictor):  # noqa: F811
    """A drafter directory with an A3 realiser (A3-0's when ``predictor`` is False, A3's when True), float32."""
    path = tmp_path / ("a3" if predictor else "a30")
    shutil.copytree(tiny / "assistant", path)
    target, drafter = _models(tiny, torch.float32)
    generator = torch.Generator().manual_seed(21)
    width = target.get_input_embeddings().weight.shape[1]
    if predictor:
        realiser = _coded(drafter.double(), target.double(), predictor=True).float()
    else:
        realiser = tr.OnePassRealiser(drafter, SLOTS, 0.5 * torch.randn(width, generator=generator), "add")
        with torch.no_grad():
            realiser.depth.copy_(0.5 * torch.randn(SLOTS, width, generator=generator))
    torch.save(tr.realiser_payload(realiser), path / "realiser.pt")
    return path


@pytest.mark.parametrize("predictor", [False, True])
def test_one_pass_decoding_drafts_in_one_call_and_keeps_the_models_text(tiny, tmp_path, predictor):  # noqa: F811
    path = _a3_dir(tiny, tmp_path, predictor)
    target = transformers.Gemma4ForConditionalGeneration.from_pretrained(tiny / "model", dtype=torch.float32).eval()
    drafter = de.load_assistant(path, "fp32", torch.device("cpu"))
    realiser = de._ONE_PASS[id(drafter)]
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny / "model")
    ids = torch.tensor([chat_ids(tokenizer, "Count to four, then stop.")])
    calls = []
    original = de.one_pass_drafts

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        calls.append((args, kwargs, out))
        return out

    de.one_pass_drafts = spy
    try:
        record = de.decode_once(target, drafter, ids, de.Arm("mtp", SLOTS, "a3" if predictor else "a30"), 14)
    finally:
        de.one_pass_drafts = original
    plain = target.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=14, do_sample=False,
                            temperature=None, top_p=None, top_k=None)
    assert record["tokens"] == plain[0, ids.shape[1]:].tolist()                     # the model's own text
    assert calls and record["drafter_passes"] == len(calls)                         # one call per cycle
    if predictor:
        assert record["predictor_passes"] == len(calls)
    # Each cycle's batched call equals one call per slot (the reference: one row at a time).
    for (r, embeddings, last_token, last_hidden, kv, mask, position, count), _, out in calls[:3]:
        token = embeddings(last_token)
        code = r.decode_code(token, last_hidden, kv, position)
        for j in range(count):
            inputs = torch.cat([token + (code + r.depth[j]), last_hidden], dim=-1)
            single = r.drafter(inputs_embeds=inputs, attention_mask=mask, position_ids=torch.tensor([[position]]),
                               shared_kv_states=kv, use_cache=False)
            assert int(single.logits[0, 0].argmax()) == int(out[0, j])
        # The keys are the accepted prefix's: positions 0 .. q - 1 in full attention.
        assert kv["full_attention"][0].shape[2] == position
    assert realiser.slots == SLOTS
    with pytest.raises(ValueError, match="accepted prefix"):
        de.decode_once(target, drafter, ids, de.Arm("mtp", SLOTS, "a3", True), 4)


def test_an_oracle_is_never_decoded(tiny, tmp_path):  # noqa: F811
    path = tmp_path / "oracle"
    shutil.copytree(tiny / "assistant", path)
    target, drafter = _models(tiny, torch.float64)
    torch.save(tr.realiser_payload(_coded(drafter, target)), path / "realiser.pt")
    with pytest.raises(ValueError, match="never decoded"):
        de.load_assistant(path, "fp32", torch.device("cpu"))
