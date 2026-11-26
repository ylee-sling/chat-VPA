# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""P2 of E1's Step 1 (docs/e1.md, Section 11.7) without the runtime: one LiteRT-LM engine of one mode per process
(the stand-in of ``test_e1_litert``), the comparison of texts, the classes and the verdict of the registered rule,
and the parting token scored by a tiny random Gemma 4 in float32, against log-probabilities and ranks computed
here independently (float64, an explicit sort). Nothing is downloaded."""

from __future__ import annotations

import json
import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
if not hasattr(transformers, "Gemma4ForConditionalGeneration"):      # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4", allow_module_level=True)

from test_e1_litert import PROMPTS, TABLE, _fake_runtime  # noqa: E402
from test_stage0_cli import write_checkpoint  # noqa: E402

from chat_vpa import e1_p2 as p2  # noqa: E402
from chat_vpa.e1_common import chat_ids, load_target  # noqa: E402

CHAT_TEMPLATE = ("{{ bos_token }}{% for m in messages %}<{{ m['role'] }}>\n{{ m['content'] }}\n{% endfor %}"
                 "{% if add_generation_prompt %}<model>\n{% endif %}")


def test_the_first_differing_token():
    assert p2.first_token_difference([1, 2, 3], [1, 2, 3]) is None
    assert p2.first_token_difference([1, 2, 3], [1, 9, 3]) == 1
    assert p2.first_token_difference([5], [6, 7]) == 0
    assert p2.first_token_difference([1, 2], [1, 2, 3]) == 2                 # a prefix: the shorter length
    assert p2.first_token_difference([], [4]) == 0
    assert p2.first_token_difference([], []) is None


def test_the_classes_and_the_verdict_follow_the_registered_rule():
    assert p2.classify(1, 2, 0.4) == "near tie"
    assert p2.classify(2, 1, 1.0) == "near tie"                             # the bound itself is a near tie
    assert p2.classify(1, 2, 1.0001) == "other"                             # top two, but more than 1 nat apart
    assert p2.classify(1, 3, 0.2) == "other"                                # close, but not the top two
    assert p2.classify(1, 5, 4.0) == "other"                                # rank 5 is not below 5
    assert p2.classify(2, 6, 4.0) == "far"
    assert p2.classify(3, 60, 9.0) == "other"                               # the plain side is not in the top two

    nine, one = ["near tie"] * 9, ["other"]
    assert p2.verdict(nine + one, True) == "consistent with a precision effect"            # 90% near ties, no far
    assert p2.verdict(nine + ["far"], True) == "a defect of the GPU backend's speculation"  # 10% far
    assert p2.verdict(["near tie"] * 8 + one * 2, True) == "undecided"
    assert p2.verdict(["near tie"] * 19 + ["far"], True) == "undecided"                     # one far in twenty
    assert p2.verdict([], None).startswith("nothing to read")
    shaky = p2.verdict(nine + one, False)
    assert shaky.startswith("consistent with a precision effect; the backend does not repeat itself")


def test_runs_are_compared_text_by_text_and_paired_by_mode():
    a = {"p0": {"text": "same"}, "p1": {"text": "abcd"}, "p2": {"text": "x"}}
    b = {"p0": {"text": "same"}, "p1": {"text": "abXd"}, "p2": {"text": "xyz"}}
    assert p2.compare(a, b, ["p0", "p1", "p2"]) == {"prompts": 3, "same_text": 1,
                                                    "first_differences": {"p1": 2, "p2": 1}}
    with pytest.raises(ValueError, match="missing"):
        p2.compare(a, {"p0": {"text": "same"}}, ["p0", "p1"])
    assert p2._pairs(["off-a", "on-a", "off-b", "on-b"]) == ([("off-a", "off-b"), ("on-a", "on-b")],
                                                             [("off-a", "on-a"), ("off-b", "on-b")])
    with pytest.raises(ValueError, match="mode"):
        p2._mode_of("fast-a")


def test_one_engine_of_one_mode_runs_in_the_process(tmp_path, monkeypatch, capsys):
    log: list = []
    fake = _fake_runtime(log, fail=())
    monkeypatch.setitem(sys.modules, "litert_lm", fake)
    model = tmp_path / "litert" / "model.litertlm"
    model.parent.mkdir()
    model.write_bytes(b"not a model")
    prompts = tmp_path / "dev.jsonl"
    prompts.write_text("".join(json.dumps(row) + "\n" for row in PROMPTS))
    monkeypatch.setenv("VK_DRIVER_FILES", "/x/nvidia_icd.json")
    out = tmp_path / "p2"
    assert p2.main(["run", "--mode", "on", "--tag", "on-a", "--model", str(model), "--prompts", str(prompts),
                    "--out", str(out), "--limit", "2", "--warmup", "1", "--max-new-tokens", "9"]) == 0
    engines = [entry for entry in log if entry[0] == "engine"]
    assert [entry[1] for entry in engines] == ["on"]                       # no off engine beside it
    assert engines[0][2] == fake.Backend.GPU() and engines[0][4] == 0.0     # the GPU backend, lenience 0
    turns = [(mode, prompt) for mode, prompt, *_ in log if mode != "engine"]
    assert turns == [("on", "zero"), ("on", "zero"), ("on", "one")]        # the warm-up, then the two prompts
    samples = [json.loads(line) for line in (out / "on-a" / "samples.jsonl").read_text().splitlines()]
    assert [(s["id"], s["mode"], s["tag"], s["text"]) for s in samples] == [
        ("p0", "on", "on-a", TABLE[("on", "zero")]["text"]), ("p1", "on", "on-a", TABLE[("on", "one")]["text"])]
    run = json.loads((out / "on-a" / "run.json").read_text())
    assert run["mode"] == "on" and run["lenience"] == 0.0 and run["prompts"]["count"] == 2
    assert run["environment"]["vulkan_driver_files"] == "/x/nvidia_icd.json"
    assert "on-a: 2 prompts decoded by mode on" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        p2.main(["run", "--mode", "off", "--tag", "on-b", "--model", str(model), "--prompts", str(prompts),
                 "--out", str(out)])
    failing = _fake_runtime(log, fail=("off",))
    monkeypatch.setitem(sys.modules, "litert_lm", failing)
    with pytest.raises(SystemExit, match="could not be created"):
        p2.main(["run", "--mode", "off", "--tag", "off-a", "--model", str(model), "--prompts", str(prompts),
                 "--out", str(out)])


# ---------------------------------------------------------------------------------------------
# The parting token, scored by a tiny model
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("p2")
    write_checkpoint(root / "model")
    tokenizer = transformers.AutoTokenizer.from_pretrained(root / "model")
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(root / "model")
    return root


def _reference(model, ids, tokens):
    """Log-probabilities in float64 from the full logits of the last position, and ranks by an explicit sort."""
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1].to(torch.float64)
    logp = logits - torch.logsumexp(logits, dim=-1)
    order = sorted(range(logp.numel()), key=lambda t: -float(logp[t]))
    return [float(logp[t]) for t in tokens], [order.index(t) + 1 for t in tokens], order[0]


def test_the_parting_token_is_scored_as_an_independent_computation_scores_it(tiny):
    model, tokenizer = load_target(tiny / "model", "fp32", torch.device("cpu"))
    plain, speculating = "Sure, the sum is 12.", "Sure, the sum is 21."
    where = p2.parting(tokenizer, plain, speculating)
    assert where["kind"] == "token" and where["index"] == plain.index("12")     # one token per character
    assert tokenizer.decode(where["shared"]) == "Sure, the sum is "
    assert (tokenizer.decode([where["plain_token"]]), tokenizer.decode([where["speculating_token"]])) == ("1", "2")
    assert p2.parting(tokenizer, "abc", "abc") == {"kind": "same tokens"}
    assert p2.parting(tokenizer, "abc", "abcd") == {"kind": "ends", "index": 3}

    prompt = chat_ids(tokenizer, "Add five and seven.")
    got = p2.score_parting(model, prompt, where["shared"], where["plain_token"], where["speculating_token"],
                           torch.device("cpu"))
    logp, ranks, argmax = _reference(model, prompt + where["shared"], [where["plain_token"],
                                                                       where["speculating_token"]])
    assert (got["plain_rank"], got["speculating_rank"], got["argmax"]) == (ranks[0], ranks[1], argmax)
    assert got["plain_logp"] == pytest.approx(logp[0], rel=1e-6, abs=1e-6)
    assert got["speculating_logp"] == pytest.approx(logp[1], rel=1e-6, abs=1e-6)
    assert got["gap"] == pytest.approx(abs(logp[0] - logp[1]), rel=1e-6, abs=1e-6)


def _write_run(directory, tag, texts):
    directory.mkdir(parents=True)
    with open(directory / "samples.jsonl", "w", encoding="utf-8") as handle:
        for i, text in texts.items():
            handle.write(json.dumps({"id": i, "mode": tag.split("-")[0], "tag": tag, "text": text}) + "\n")
    env = {"runtime": "LiteRT-LM 0.20.0", "backend": "gpu", "vulkan_driver_files": "/x/nvidia_icd.json"}
    (directory / "run.json").write_text(json.dumps({"tag": tag, "environment": env}))


def test_the_reading_compares_scores_once_and_applies_the_rule(tiny, tmp_path, capsys):
    prompts = tmp_path / "dev.jsonl"
    rows = [{"id": "p0", "prompt": "Name a colour."}, {"id": "p1", "prompt": "Count to three."}]
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    off = {"p0": "Blue is a colour.", "p1": "One, two, three."}
    on = {"p0": "Blue is a color.", "p1": "One, two, three."}
    out = tmp_path / "p2"
    for tag, texts in (("off-a", off), ("on-a", on), ("off-b", off), ("on-b", on)):
        _write_run(out / tag, tag, texts)
    step0 = tmp_path / "step0.jsonl"
    with open(step0, "w", encoding="utf-8") as handle:
        for mode, texts in (("off", off), ("on", on), ("adaptive", {"p0": "Blue is a colour!", "p1": "One, two"})):
            for i, text in texts.items():
                handle.write(json.dumps({"id": i, "mode": mode, "text": text}) + "\n")

    assert p2.main(["read", "--out", str(out), "--model", str(tiny / "model"), "--prompts", str(prompts),
                    "--step0", str(step0), "--device", "cpu"]) == 0
    summary = json.loads((out / "p2.json").read_text())
    assert summary["p2a_repeats"] is True
    assert summary["p2a"]["off-a / off-b"]["same_text"] == 2 and summary["p2a"]["on-a / on-b"]["same_text"] == 2
    assert summary["p2b"]["off-a / on-a"] == {"prompts": 2, "same_text": 1, "first_differences": {"p0": 14}}
    assert summary["against_step0"]["on-a / Step 0 on"]["same_text"] == 2
    counts = summary["p2c"]["counts"]
    # Pairs: 2 per P2b pair (4), 2 for Step 0's on and 2 for its adaptive. Same text: p1 three times. The p0 pair
    # of on against off appears three times (P2b twice, Step 0 once) and is scored once; adaptive's p0 parts at
    # "!" against "."; its p1, "One, two", ends where off's goes on.
    assert counts == {"pairs": 8, "same text": 3, "same tokens": 0, "ends": 1, "duplicates": 2, "scored": 2}
    rows_scored = summary["p2c"]["rows"]
    assert [(r["source"], r["id"], r["index"]) for r in rows_scored] == [("P2 on-a / off-a", "p0", 14),
                                                                        ("Step 0 adaptive / off", "p0", 16)]

    model, tokenizer = load_target(tiny / "model", "fp32", torch.device("cpu"))
    classes = []
    for row, (plain, other) in zip(rows_scored, ((off["p0"], on["p0"]), (off["p0"], "Blue is a colour!")),
                                   strict=True):
        index = row["index"]
        shared = list(tokenizer(plain[:index], add_special_tokens=False)["input_ids"])
        tokens = [tokenizer(plain[index], add_special_tokens=False)["input_ids"][0],
                  tokenizer(other[index], add_special_tokens=False)["input_ids"][0]]
        logp, ranks, _ = _reference(model, chat_ids(tokenizer, rows[0]["prompt"]) + shared, tokens)
        assert (row["plain_rank"], row["speculating_rank"]) == tuple(ranks)
        assert row["gap"] == pytest.approx(abs(logp[0] - logp[1]), rel=1e-6, abs=1e-6)
        classes.append(p2.classify(ranks[0], ranks[1], abs(logp[0] - logp[1])))
    assert [r["class"] for r in rows_scored] == classes
    assert summary["verdict"] == p2.verdict(classes, True)
    printed = capsys.readouterr().out
    assert "| P2b: off-a / on-a | 1 of 2 | 14 |" in printed and "**Verdict (Section 11.7, P2):**" in printed
    assert (out / "p2.md").read_text().startswith("# P2: LiteRT-LM's GPU backend")

    # Without the model: the texts are compared and nothing is scored.
    assert p2.main(["read", "--out", str(out), "--no-ties"]) == 0
    quick = json.loads((out / "p2.json").read_text())
    assert quick["p2c"]["rows"] == [] and quick["verdict"].startswith("not read")
    with pytest.raises(SystemExit):
        p2.main(["read", "--out", str(out)])                                 # scoring needs --model and --prompts
