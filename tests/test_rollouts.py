# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Autoregressive rollouts as training targets (``chat_vpa.rollouts``, ``chat_vpa.rollout_throughput``): the batched
generator against transformers' own per-prompt ``generate``, the targets against the truncation rule, the
entropy measurements against transformers' logits warpers and ``Gemma4ForCausalLM``. On a tiny random Gemma 4
saved to disk in the released layout; nothing is downloaded."""

from __future__ import annotations

import json
import math

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4ForConditionalGeneration"):      # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4", allow_module_level=True)

from test_stage0 import MASK, VOCAB, CharTokenizer, _write_corpus, tiny_realizer  # noqa: E402
from test_stage0_cli import write_checkpoint  # noqa: E402

from chat_vpa import rollout_throughput  # noqa: E402
from chat_vpa.autoregressive import cut_at_eos  # noqa: E402
from chat_vpa.backbone import load_gemma4  # noqa: E402
from chat_vpa.data import Example, HFTokenizer, collate, tokenize_examples  # noqa: E402
from chat_vpa.postprocess import is_complete, truncate_complete, truncate_write  # noqa: E402
from chat_vpa.rollouts import (  # noqa: E402
    STOPS,
    Rollout,
    decoding_entropy,
    finish_rollout,
    generate_rollouts,
    rollouts_for,
    teacher_forced_stats,
)


@pytest.fixture(scope="module")
def released(tmp_path_factory):
    """The tiny checkpoint as ``load_gemma4`` returns the released one, in float32."""
    path = tmp_path_factory.mktemp("model")
    write_checkpoint(path)
    model, hf_tokenizer = load_gemma4(str(path), dtype=torch.float32)
    model.generation_config.eos_token_id = 1
    return model, hf_tokenizer


def _prompts(tokenizer: HFTokenizer) -> list[list[int]]:
    texts = ["def f(a):\n", "def add_two_numbers(first, second):\n    # sum\n", "x", "import os\n\n\ndef g():\n",
             "def h(value, *, scale=1):\n    \"\"\"Scale the value.\"\"\"\n"]
    return [[tokenizer.bos_id, *ids] for ids in tokenizer.encode_batch(texts)]


def test_greedy_batched_rollouts_equal_per_prompt_generate(released):
    """Left padding, the attention mask and the positions of a batch of distinct prompts: with greedy decoding
    every row is transformers' own ``generate`` on that prompt alone, token for token."""
    model, hf_tokenizer = released
    tokenizer = HFTokenizer(hf_tokenizer)
    prompts = _prompts(tokenizer)
    assert len({len(p) for p in prompts}) == len(prompts) and min(map(len, prompts)) == 2      # all padded but one
    budget = 24
    want = []
    for prompt in prompts:                                           # the reference: one unpadded prompt at a time
        with torch.no_grad():
            out = model.generate(input_ids=torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt)).long(),
                                 max_new_tokens=budget, do_sample=False, eos_token_id=[1], pad_token_id=0)
        want.append(cut_at_eos(out[0, len(prompt):].tolist(), [1, 0]))
    # "write": its truncation rule does not fire on these continuations, so every row runs to its <eos> or budget.
    got, ended, steps = generate_rollouts(model, hf_tokenizer, prompts, ["write"] * len(prompts), budget, 0.0, 1.0,
                                          VOCAB, seed=0, check_every=4)
    assert got == want and steps == max(len(w) + (len(w) < budget) for w in want)
    assert ended == [len(w) < budget for w in want]
    assert sum(len(w) for w in want) >= 3 * budget                   # the comparison is not vacuous
    # A different order of the rows changes nothing for a row (greedy): padding is all that differs.
    again, _, _ = generate_rollouts(model, hf_tokenizer, prompts[::-1], ["write"] * len(prompts), budget, 0.0, 1.0,
                                    VOCAB, seed=5, check_every=4)
    assert again == want[::-1]


def test_a_row_stops_once_the_truncation_rule_has_found_its_stop(released):
    """``complete`` rows stop at the first check after their text holds a line that starts in column 0; what
    they generated until then is a prefix of the unstopped continuation, and the truncated program is the same."""
    model, hf_tokenizer = released
    tokenizer = HFTokenizer(hf_tokenizer)
    prompts = _prompts(tokenizer)
    budget, every = 40, 4
    full, _, _ = generate_rollouts(model, hf_tokenizer, prompts, ["write"] * len(prompts), budget, 0.0, 1.0, VOCAB,
                                   seed=0, check_every=every)
    short, _, _ = generate_rollouts(model, hf_tokenizer, prompts, ["complete"] * len(prompts), budget, 0.0, 1.0,
                                    VOCAB, seed=0, check_every=every)
    stopped_early = 0
    for whole, part in zip(full, short, strict=True):
        assert part == whole[:len(part)]
        first = next((n for n in range(every, len(whole) + 1, every)             # an explicit loop over the checks
                      if is_complete(tokenizer.decode(whole[:n]), "complete")), None)
        if first is None:
            assert part == whole
        else:
            assert len(part) == first
            stopped_early += len(part) < len(whole)
            assert truncate_complete(tokenizer.decode(part)) == truncate_complete(tokenizer.decode(whole))
    assert stopped_early >= 1                                        # the rule did fire on a row before its end


def test_finish_rollout_gives_the_reference_arms_canvas_format():
    tok = CharTokenizer()
    body = "    return a + b\n\n\ndef other():\n    pass\n"
    done = finish_rollout(body, ended=False, generated=41, kind="complete", tokenizer=tok, canvas_len=32)
    assert done.stop == "truncation" and done.kept and done.text == "    return a + b\n"
    assert done.canvas_ids == [*tok.encode_batch(["    return a + b\n"])[0], tok.eos_id]
    # The same format as a human reference: tokens of the text, then one <eos> (``tokenize_examples``).
    example, _ = tokenize_examples([("def add(a, b):\n", "    return a + b\n", "complete")], tok, 32, 64)
    assert done.canvas_ids == example[0].canvas_ids
    collate([Example(example[0].prompt_ids, done.canvas_ids, "complete")], 32, tok.eos_id)
    ended = finish_rollout("    return a - b   \n", ended=True, generated=18, kind="complete", tokenizer=tok,
                           canvas_len=32)
    assert ended.stop == "eos" and ended.text == "    return a - b\n" and ended.canvas_ids[-1] == tok.eos_id
    cut = finish_rollout("    return (a +\n        b", ended=False, generated=32, kind="complete", tokenizer=tok,
                         canvas_len=32)
    assert cut.stop == "budget" and not cut.kept and cut.canvas_ids is None
    program = "def add(a, b):\n    return a + b\n\nassert add(1, 2) == 3\n"
    written = finish_rollout(program, ended=False, generated=50, kind="write", tokenizer=tok, canvas_len=40)
    assert written.stop == "truncation" and written.text == truncate_write(program) == "def add(a, b):\n    return a + b\n"
    long = finish_rollout(program, ended=False, generated=50, kind="write", tokenizer=tok, canvas_len=len(written.text))
    assert long.stop == "too_long" and not long.kept                 # the program fits, its <eos> does not
    assert finish_rollout(program, False, 50, "write", tok, len(written.text) + 1).kept
    assert {done.stop, ended.stop, cut.stop, long.stop} == set(STOPS)
    with pytest.raises(ValueError):
        finish_rollout(body, False, 1, "chat", tok, 32)


def test_sampled_rollouts_are_seeded(released):
    model, hf_tokenizer = released
    tokenizer = HFTokenizer(hf_tokenizer)
    examples = [Example(p, [7, tokenizer.eos_id], "complete") for p in _prompts(tokenizer)]
    args = (model, hf_tokenizer, tokenizer, examples, 48, 32, 1.0, 0.95, 50)
    first, steps = rollouts_for(*args, seed=3)
    second, _ = rollouts_for(*args, seed=3)
    other, _ = rollouts_for(*args, seed=4)
    assert first == second and first != other and 1 <= steps <= 32
    assert all(isinstance(r, Rollout) and r.stop in STOPS and r.kind == "complete" for r in first)
    for rollout in first:
        if rollout.kept:
            assert rollout.canvas_ids[-1] == tokenizer.eos_id and len(rollout.canvas_ids) <= 48
            assert tokenizer.eos_id not in rollout.canvas_ids[:-1]
            assert tokenizer.decode(rollout.canvas_ids) == rollout.text


def _warped(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    """The distribution ``generate`` draws from: transformers' own warpers, in ``generate``'s order."""
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

    scores = TemperatureLogitsWarper(temperature)(None, logits.double())
    scores = TopKLogitsWarper(top_k)(None, scores)
    scores = TopPLogitsWarper(top_p)(None, scores)
    return torch.softmax(scores, dim=-1)


def _entropy(probs: torch.Tensor) -> list[float]:
    return [-math.fsum(p * math.log(p) for p in row if p > 0) for row in probs.tolist()]


@pytest.mark.parametrize("temperature,top_p,top_k", [(0.2, 0.95, 50), (1.0, 0.9, 7), (0.7, 1.0, 400), (1.3, 0.5, 1)])
def test_decoding_entropy_is_the_entropy_of_what_generate_draws_from(temperature, top_p, top_k):
    logits = 3.0 * torch.randn(64, VOCAB, generator=torch.Generator().manual_seed(1))
    got = decoding_entropy(logits, temperature, top_p, top_k)
    want = torch.tensor(_entropy(_warped(logits, temperature, top_p, top_k)), dtype=torch.float64)
    torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-12)
    assert float(got.min()) >= 0.0 and (float(got.max()) > 0.0) == (top_k > 1)   # one token: no entropy
    assert float(decoding_entropy(logits, 0.0, top_p, top_k).abs().max()) == 0.0     # greedy: Corollary 1(a)
    with pytest.raises(ValueError):
        decoding_entropy(logits, temperature, 0.0, top_k)


def test_teacher_forced_stats_against_the_reference_model():
    """Entropy of the decoding distribution and NLL at temperature 1 at every program position and at the <eos>,
    against transformers' ``Gemma4ForCausalLM`` on <bos> prompt + program, one example at a time."""
    realizer, reference = tiny_realizer(seed=2)
    tok = CharTokenizer()
    pairs = [("def f(a):\n", "    return a\n", "complete"), ("def longer_name(x, y):\n", "    z = x * y\n    return z\n",
                                                             "complete"), ("q", "w = 1\n", "write")]
    examples, _ = tokenize_examples(pairs, tok, canvas_len=40, max_prompt_tokens=40)
    temperature, top_p, top_k = 0.5, 0.9, 40
    got = teacher_forced_stats(realizer, tok, examples, torch.device("cpu"), temperature, top_p, top_k, micro_batch=2,
                               chunk=7)
    assert len(got) == 3
    for example, stats in zip(examples, got, strict=True):
        ids = torch.tensor([example.prompt_ids + example.canvas_ids[:-1]])
        with torch.no_grad():
            logits = reference(ids).logits[0, len(example.prompt_ids) - 1:].double()   # one row per canvas token
        assert logits.shape[0] == len(example.canvas_ids)
        nll = (-torch.log_softmax(logits, dim=-1))[torch.arange(len(example.canvas_ids)), example.canvas_ids].tolist()
        entropy = _entropy(_warped(logits, temperature, top_p, top_k))
        assert stats["tokens"] == len(example.canvas_ids) - 1
        # float32 networks compared in float64; logits enter the entropy divided by the temperature
        assert math.isclose(stats["nll"], math.fsum(nll[:-1]), rel_tol=1e-4)
        assert math.isclose(stats["entropy"], math.fsum(entropy[:-1]), rel_tol=1e-4)
        assert math.isclose(stats["eos_nll"], nll[-1], rel_tol=1e-4)
        assert math.isclose(stats["eos_entropy"], entropy[-1], rel_tol=1e-4)
    assert MASK not in examples[0].canvas_ids


def test_throughput_cli_on_the_tiny_model(tmp_path, capsys):
    write_checkpoint(tmp_path / "model")
    _write_corpus(tmp_path / "corpus", functions=40, shards=2)
    out = tmp_path / "runs" / "throughput"
    args = ["--model", str(tmp_path / "model"), "--corpus", str(tmp_path / "corpus"), "--out", str(out),
            "--num-prompts", "12", "--batch-sizes", "4", "8", "--max-new-tokens", "24", "--canvas-len", "64",
            "--max-prompt-tokens", "128", "--top-k", "50", "--temperature", "1.0", "--check-every", "4",
            "--entropy-batch", "5", "--examples", "3", "--device", "cpu"]
    assert rollout_throughput.main(args) == 0
    assert sorted(p.name for p in out.iterdir()) == ["examples.txt", "throughput.json"]      # and nothing else
    result = json.loads((out / "throughput.json").read_text())
    assert result["prompts"]["count"] == 12 and sum(result["prompts"]["kinds"].values()) == 12
    assert result["settings"]["sorted_by_prompt_length"] is True and result["settings"]["max_new_tokens"] == 24
    assert [e["batch_size"] for e in result["batch_sizes"]] == [4, 8]
    for entry in result["batch_sizes"]:
        assert entry["batches"] == math.ceil(12 / entry["batch_size"]) == len(entry["seeds"])
        assert entry["rollouts"] == 12 and math.isclose(sum(entry["stops"].values()), 1.0, abs_tol=1e-3)
        assert entry["kept"] == round(12 * (entry["stops"]["truncation"] + entry["stops"]["eos"]))
        assert entry["generated_tokens"] <= entry["row_steps"] <= 12 * 24 and entry["wall_seconds"] > 0
        assert sum(k["rollouts"] for k in entry["stops_by_kind"].values()) == 12
    best = result["best"]
    assert best["batch_size"] in (4, 8) and set(best["projected"]) == {"100000", "320000"}
    rate, kept = best["rollouts_per_second"], best["kept_share"]
    assert math.isclose(best["projected"]["320000"]["hours_for_n_prompts"], 320000 / rate / 3600, abs_tol=0.01)
    if kept > 0:
        assert math.isclose(best["projected"]["100000"]["hours_for_n_kept_rollouts"], 100000 / rate / kept / 3600,
                            abs_tol=0.01)
        entropy = result["entropy"]
        assert entropy["rollouts"]["programs"] == entropy["references"]["programs"] == round(12 * kept)
        assert "entropy_per_token" not in entropy["references"] and entropy["references"]["nll_per_token"] > 0
    text = (out / "examples.txt").read_text()
    assert text.count("===== example") == 3 and "----- human reference" in text and "----- rollout" in text
    capsys.readouterr()
    assert rollout_throughput.main(["--show", str(out)]) == 0
    shown = capsys.readouterr().out
    assert "best batch size" in shown and "projected" in shown
    with pytest.raises(SystemExit):
        rollout_throughput.main(["--out", str(out)])
