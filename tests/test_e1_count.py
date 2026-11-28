# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""E1 Step 1's counted tokens per cycle along the development prompts' continuations (``chat_vpa.e1_count``,
docs/e1.md Sections 6.1 and 11.5): every count against a cycle-by-cycle walk of the continuation whose drafts come
from Transformers' own single-query calls of the drafter (``test_e1_train._reference_calls``), for Google's drafter
unrolled on the accepted path and for an A3 realiser in one call, along continuations built so that both drafters'
drafts are accepted for several tokens at a time (random drafters on the target's own continuations would never
be); the comparisons by hand from the per-prompt counts; the continuations and the measured counts beside them on a
decode table that ``e1_decode`` writes with the tiny Gemma 4."""

from __future__ import annotations

import json
import shutil

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4AssistantForCausalLM"):          # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4 drafter", allow_module_level=True)

from test_e1_measure import tiny  # noqa: E402,F401  (the fixture: the tiny target, drafters a1 and a2, prompts)
from test_e1_train import _reference_calls  # noqa: E402

from chat_vpa import e1_count as ec  # noqa: E402
from chat_vpa import e1_decode as de  # noqa: E402
from chat_vpa.e1_common import chat_ids, file_sha256  # noqa: E402

MAX_NEW = 14


@pytest.fixture(scope="module")
def table(tiny, tmp_path_factory):  # noqa: F811
    """A decode table of the tiny target in float32 on the CPU: ar, a1 at G = 2 with Transformers' keys and with
    the accepted prefix's."""
    out = tmp_path_factory.mktemp("e1-count") / "decode"
    assert de.main(["--model", str(tiny / "model"), "--drafter", f"a1={tiny / 'a1'}", "--prompts",
                    str(tiny / "prompts.jsonl"), "--arms", "ar", "a1:2", "a1:2:clean", "--max-new-tokens",
                    str(MAX_NEW), "--dtype", "fp32", "--device", "cpu", "--warmup", "0", "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def realiser(tiny, tmp_path_factory):  # noqa: F811
    """An A3 realiser beside a2's drafter: 3 slots added to x_q's embedding, a random code and depth vectors."""
    path = tmp_path_factory.mktemp("e1-count-a3") / "a3"
    shutil.copytree(tiny / "a2", path)
    generator = torch.Generator().manual_seed(11)
    torch.save({"slots": 3, "slot_input": "add", "code": 0.3 * torch.randn(64, generator=generator),
                "depth": 0.3 * torch.randn(3, 64, generator=generator)}, path / "realiser.pt")
    return path


def _drafts(target, drafter, ids, q, saved=None):
    """Transformers' own drafts at q, with the keys of a cycle that starts there: Google's drafter on its own
    accepted path over 4 depths, or each slot of a realiser in one call."""
    embed = target.get_input_embeddings()
    if saved is None:
        drafts = []

        def inputs_of(k, previous):
            if k > 4:
                return None
            token = ids[q] if k == 1 else drafts[-1]
            return torch.cat([embed(torch.tensor([[token]])), previous], dim=-1)

        # _reference_calls asks for depth k's input after depth k - 1's call; record each draft as it is made.
        original = drafter.forward

        def recording(*args, **kwargs):
            out = original(*args, **kwargs)
            drafts.append(int(out.logits[0, 0].argmax()))
            return out

        drafter.forward = recording
        try:
            _reference_calls(target, drafter, ids, q, inputs_of)
        finally:
            drafter.forward = original
        return drafts
    drafts = []
    for j in range(saved["slots"]):

        def one(k, previous, j=j):
            if k > 1:
                return None
            slot = (saved["code"] + saved["depth"][j]).view(1, 1, -1)
            return torch.cat([embed(torch.tensor([[ids[q]]])) + slot, previous], dim=-1)

        _, _, logits = _reference_calls(target, drafter, ids, q, one)
        drafts.append(int(logits[0].argmax()))
    return drafts


@pytest.fixture(scope="module")
def built(tiny, realiser, tmp_path_factory):  # noqa: F811
    """A table of three continuations built cycle by cycle, alternately from a1's and from the realiser's drafts:
    each cycle keeps 1 to 3 of a1's 4 drafts, or 1 to 2 of the realiser's 3, and then one token that differs from
    the drafter's next draft."""
    root = tmp_path_factory.mktemp("e1-count-built")
    target = transformers.Gemma4ForConditionalGeneration.from_pretrained(tiny / "model", dtype=torch.float32).eval()
    a1 = transformers.Gemma4AssistantForCausalLM.from_pretrained(tiny / "a1", dtype=torch.float32).eval()
    a3 = transformers.Gemma4AssistantForCausalLM.from_pretrained(realiser, dtype=torch.float32).eval()
    saved = torch.load(realiser / "realiser.pt", weights_only=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny / "model")
    generator = torch.Generator().manual_seed(12)
    prompts = [json.loads(line) for line in open(tiny / "prompts.jsonl", encoding="utf-8")]
    samples = []
    for p, row in enumerate(prompts):
        prompt = chat_ids(tokenizer, row["prompt"])
        ids = prompt + [int(torch.randint(6, 300, (1,), generator=generator))]
        turn = p
        while len(ids) < len(prompt) + MAX_NEW:
            q = len(ids) - 1
            drafts = _drafts(target, a1, ids, q) if turn % 2 == 0 else _drafts(target, a3, ids, q, saved)
            keep = int(torch.randint(1, len(drafts), (1,), generator=generator))     # leaves a next draft
            miss = (drafts[keep] + 1 + int(torch.randint(0, 200, (1,), generator=generator))) % 300
            ids += drafts[:keep] + [max(miss, 6)]
            turn += 1
        samples.append({"id": row["id"], "arm": "ar", "prompt_tokens": len(prompt),
                        "tokens": ids[len(prompt):len(prompt) + MAX_NEW]})
    out = root / "table"
    out.mkdir()
    with open(out / "samples.jsonl", "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")
    (out / "summary.json").write_text(json.dumps({
        "environment": {"device": "cpu", "dtype": "fp32", "runtime": "PyTorch, eager"},
        "prompts_sha256": file_sha256(tiny / "prompts.jsonl"), "arms": {"ar": {"tokens_per_cycle": 1.0}}}))
    return out


def _walk(hits_at, length, positions):
    """Cycles along one continuation, by hand: a cycle at drafted position i accepts the leading hits among its first
    ``length`` drafts (never past the continuation's end) and adds the next token."""
    i, cycles = 0, 0
    while i < positions:
        drafts = hits_at(i)
        accepted = 0
        while accepted < min(length, len(drafts)) and drafts[accepted] and i + accepted + 1 < positions:
            accepted += 1
        cycles += 1
        i += accepted + 1
    return cycles


def _reference_hits(tiny, ids, start, drafter_dir, saved=None):  # noqa: F811
    """Per drafted position q = start .. len(ids) - 2, whether each draft equals the continuation, the drafts made by
    Transformers' own calls of the drafter (its clustered logits' argmax): unrolled on the accepted path up to 8
    depths, or one call per slot of a realiser."""
    target = transformers.Gemma4ForConditionalGeneration.from_pretrained(tiny / "model", dtype=torch.float32).eval()
    drafter = transformers.Gemma4AssistantForCausalLM.from_pretrained(drafter_dir, dtype=torch.float32).eval()
    embed = target.get_input_embeddings()
    out = []
    for q in range(start, len(ids) - 1):
        if saved is None:
            def inputs_of(k, previous, q=q):
                if k > 8 or q + k > len(ids) - 1:
                    return None
                return torch.cat([embed(torch.tensor([[ids[q + k - 1]]])), previous], dim=-1)

            _, _, logits = _reference_calls(target, drafter, ids, q, inputs_of)
            out.append([int(logit.argmax()) == ids[q + k + 1] for k, logit in enumerate(logits)])
        else:
            row = []
            for j in range(saved["slots"]):
                if q + j + 1 > len(ids) - 1:
                    break

                def one(k, previous, j=j, q=q):
                    if k > 1:
                        return None
                    slot = (saved["code"] + saved["depth"][j]).view(1, 1, -1)
                    return torch.cat([embed(torch.tensor([[ids[q]]])) + slot, previous], dim=-1)

                _, _, logits = _reference_calls(target, drafter, ids, q, one)
                row.append(int(logits[0].argmax()) == ids[q + j + 1])
            out.append(row)
    return out


def test_counts_are_a_walk_of_transformers_own_drafts(tiny, built, realiser, tmp_path):  # noqa: F811
    table = built
    out = tmp_path / "count"
    assert ec.main(["--model", str(tiny / "model"), "--table", str(table), "--prompts", str(tiny / "prompts.jsonl"),
                    "--drafter", f"a1={tiny / 'a1'}", "--drafter", f"a3={realiser}", "--batch-size", "2",
                    "--device", "cpu", "--out", str(out)]) == 0
    result = json.loads((out / "count.json").read_text())
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny / "model")
    samples = [json.loads(line) for line in open(table / "samples.jsonl", encoding="utf-8")]
    prompts = {json.loads(line)["id"]: json.loads(line)["prompt"] for line in open(tiny / "prompts.jsonl")}
    ar = [s for s in samples if s["arm"] == "ar"]
    assert result["prompts"] == 3 and result["environment"]["dtype"] == "fp32"
    assert result["tokens"] == sum(len(s["tokens"]) - 1 for s in ar)
    saved = torch.load(realiser / "realiser.pt", weights_only=True)
    for label, path, kept in (("a1", tiny / "a1", None), ("a3", realiser, saved)):
        c = result["drafters"][label]
        assert c["kind"] == ("one pass" if kept else "sequential") and c["depths"] == (3 if kept else 8)
        walked = {g: [0, 0] for g in (2, 4, 6, 8) if g <= c["depths"]}
        reached = [[0, 0] for _ in range(c["depths"])]
        for p, s in enumerate(ar):
            ids = chat_ids(tokenizer, prompts[s["id"]]) + s["tokens"]
            start = len(ids) - len(s["tokens"])
            hits = _reference_hits(tiny, ids, start, path, kept)
            for g in walked:
                cycles = _walk(lambda i, hits=hits: hits[i], g, len(hits))
                assert c["per_prompt"][str(g)][p] == [cycles, len(hits)], (label, g, s["id"])
                walked[g][0] += cycles
                walked[g][1] += len(hits)
            for row in hits:
                for k in range(len(row)):
                    if all(row[:k]):
                        reached[k][0] += 1
                        reached[k][1] += row[k]
        for g, (cycles, tokens) in walked.items():
            assert c["tokens_per_cycle"][str(g)] == pytest.approx(tokens / cycles)
        assert c["profile"] == [pytest.approx(a / n) if n else None for n, a in reached]
        # Not a trivial count: drafts are accepted, several at a time (the continuations were built so).
        assert c["tokens_per_cycle"]["2"] > 1.2 and reached[1][1] > 0
    # a3 minus a1 at the lengths both have (2 only: three slots), from the per-prompt counts by hand.
    (x,) = result["comparisons"]
    assert (x["later"], x["earlier"], x["g"]) == ("a3", "a1", 2)
    a3, a1 = result["drafters"]["a3"]["per_prompt"]["2"], result["drafters"]["a1"]["per_prompt"]["2"]
    by_hand = sum(t for _, t in a3) / sum(c for c, _ in a3) - sum(t for _, t in a1) / sum(c for c, _ in a1)
    assert x["difference"]["point"] == pytest.approx(by_hand)
    assert x["difference"]["low"] <= by_hand <= x["difference"]["high"]
    assert result["measured_beside"] == []
    assert "| a3 | one pass, 3 slots (add) |" in (out / "count.md").read_text()


def test_the_continuations_are_the_tables_and_its_prompt_file(tiny, table, tmp_path):  # noqa: F811
    out = tmp_path / "count"
    assert ec.main(["--model", str(tiny / "model"), "--table", str(table), "--prompts", str(tiny / "prompts.jsonl"),
                    "--drafter", f"a1={tiny / 'a1'}", "--lengths", "2", "--device", "cpu", "--out", str(out)]) == 0
    result = json.loads((out / "count.json").read_text())
    # Beside the count: what decoding measured with the accepted prefix's keys at G = 2 (described, not read).
    summary = json.loads((table / "summary.json").read_text())
    (beside,) = result["measured_beside"]
    assert beside == {"arm": "a1-2-clean", "measured": summary["arms"]["a1-2-clean"]["tokens_per_cycle"],
                      "counted": result["drafters"]["a1"]["tokens_per_cycle"]["2"]}
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny / "model")
    records, summary = ec.continuations(table, tiny / "prompts.jsonl", tokenizer)
    ar = [json.loads(line) for line in open(table / "samples.jsonl", encoding="utf-8")]
    ar = [s for s in ar if s["arm"] == "ar"]
    assert [r["id"] for r in records] == [s["id"] for s in ar]
    assert all(r["response_ids"] == s["tokens"] and r["stop_id"] is None for r, s in zip(records, ar, strict=True))
    other = tmp_path / "other.jsonl"
    other.write_text((tiny / "prompts.jsonl").read_text().replace("Say yes.", "Say no."))
    with pytest.raises(ValueError, match="SHA-256"):
        ec.continuations(table, other, tokenizer)
    # By hand: depth 2 has a token to match only at the first of the first continuation's two positions.
    assert ec.acceptance_profile([[[True, False], [False, True]], [[True, True]]], 2) == [2 / 3, 0.0]
    with pytest.raises(SystemExit, match="LABEL=PATH"):
        ec.main(["--model", str(tiny / "model"), "--table", str(table), "--prompts", str(tiny / "prompts.jsonl"),
                 "--drafter", "a1", "--out", str(tmp_path / "x")])
