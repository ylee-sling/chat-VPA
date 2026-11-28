# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""E1 Step 1's measurement of A2 beside A1 (docs/e1.md, Section 11.4, S9 as the author amended it on 27 November
2026): the decode arms of labelled drafters, the arms whose drafter sees only the accepted prefix's keys (checked
against the keys of a fresh pass of the target over that prefix, and token-identical to the model alone in
float32), and the readings R1, R2, R6, R7, R9 and S11 against tables whose answers are worked out by hand."""

from __future__ import annotations

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4AssistantForCausalLM"):          # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4 drafter", allow_module_level=True)

from test_e1 import CHAT_TEMPLATE  # noqa: E402
from test_e1_train import _drafter_config  # noqa: E402
from test_stage0 import VOCAB, WINDOW  # noqa: E402
from test_stage0_cli import write_checkpoint  # noqa: E402

from chat_vpa import e1_decode as de  # noqa: E402
from chat_vpa import e1_readings as rd  # noqa: E402
from chat_vpa.e1_decode import summarize  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """The tiny target with a chat template, two drafters for it (a1, and a2 with other weights) and prompts."""
    root = tmp_path_factory.mktemp("e1-measure")
    write_checkpoint(root / "model")
    tokenizer = transformers.AutoTokenizer.from_pretrained(root / "model")
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(root / "model")
    for name, seed in (("a1", 5), ("a2", 9)):
        torch.manual_seed(seed)
        drafter = transformers.Gemma4AssistantForCausalLM(_drafter_config())
        generator = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for parameter in drafter.parameters():
                parameter.add_(0.2 * torch.randn(parameter.shape, generator=generator))
            drafter.masked_embedding.token_ordering.copy_(torch.randperm(VOCAB, generator=generator))
        drafter.save_pretrained(root / name)
    with open(root / "prompts.jsonl", "w", encoding="utf-8") as handle:
        for i, text in enumerate(["Name a colour.", "Count to five, slowly.", "Say yes."]):
            handle.write(json.dumps({"id": f"p{i}", "prompt": text}) + "\n")
    return root


def test_labelled_and_clean_arms_parse():
    labels = ("mtp", "a1", "a2")
    assert de.parse_arm("a2:4", labels) == de.Arm("mtp", 4, "a2", False) and de.parse_arm("a2:4", labels).name == "a2-4"
    assert de.parse_arm("a1:2:clean", labels).name == "a1-2-clean" and de.parse_arm("a1:2:clean", labels).clean
    assert de.parse_arm("mtp:4:clean").name == "mtp-4-clean" and de.parse_arm("mtp:4").name == "mtp-4"
    for bad in ("a3:4", "a2:0", "lookup:3:clean", "ar:clean", "a2:4:dirty"):
        with pytest.raises(ValueError, match="an arm is"):
            de.parse_arm(bad, labels)


def test_the_accepted_prefix_drops_what_the_verifying_pass_saw_after_it():
    def keys(n):                                       # entries whose value is their position, so cuts are visible
        return torch.arange(n, dtype=torch.float32).view(1, 1, n, 1)

    full_end, sliding_start = 10, 3                    # the verifying pass ended at position 9
    states = {"full_attention": (keys(full_end), keys(full_end)),
              "sliding_attention": (sliding_start + keys(full_end - sliding_start),
                                    sliding_start + keys(full_end - sliding_start))}
    cut = de.accepted_prefix(states, position=8)       # the drafted queries at 8: positions 8 and 9 are stale
    assert cut["full_attention"][0].flatten().tolist() == list(range(8))
    assert cut["sliding_attention"][1].flatten().tolist() == list(range(3, 8))
    assert de.accepted_prefix(states, position=10) is states                 # every draft was accepted
    with pytest.raises(ValueError, match="full-attention"):
        de.accepted_prefix({"sliding_attention": states["sliding_attention"]}, 8)


def test_the_clean_arm_hands_the_drafter_the_accepted_prefixs_keys_and_keeps_the_text(tiny):
    target = transformers.Gemma4ForConditionalGeneration.from_pretrained(tiny / "model", dtype=torch.float32).eval()
    drafter = transformers.Gemma4AssistantForCausalLM.from_pretrained(tiny / "a1", dtype=torch.float32).eval()
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
    kwargs = {"attention_mask": torch.ones_like(prompt), "max_new_tokens": 16, "do_sample": False,
              "temperature": None, "top_p": None, "top_k": None}
    with torch.no_grad():
        plain = target.generate(prompt, **kwargs)
        with de.accepted_prefix_keys():
            drafted = target.generate(prompt, assistant_model=drafter, **kwargs)
    assert torch.equal(drafted, plain)                                    # verified: the model's own tokens
    import transformers.generation.utils as generation

    assert generation.SinglePositionMultiTokenCandidateGenerator.__name__ == "SinglePositionMultiTokenCandidateGenerator"
    sequence = drafted[0].tolist()
    assert len({c["q"] for c in calls}) > 2
    seen = set()
    for call in calls:
        q = call["q"]
        with torch.no_grad():
            clean = target.model.language_model(input_ids=torch.tensor([sequence[:q]]), return_shared_kv_states=True,
                                                use_cache=False).shared_kv_states
        assert call["kv"]["full_attention"][0].shape[2] == q                # x_0 .. x_{q-1}, nothing at q
        # The keys of the cached passes against one fresh pass: float32's rounding between two code paths, relative
        # to the keys' scale (a cache built pass by pass equals a one-pass prefill to 1e-5; seen here: 1.2e-6).
        scale = float(clean["full_attention"][0].abs().max())
        torch.testing.assert_close(call["kv"]["full_attention"][0], clean["full_attention"][0], rtol=1e-5,
                                   atol=1e-5 * scale)
        # The sliding layers see the last W + 1 entries they are given: here the accepted prefix's last ones, all
        # of them before q. After a cycle that accepted no draft the target's sliding cache (W - 1 kept, plus the
        # verifying pass) no longer holds the oldest of the W + 1, so the drafter sees W of them.
        given = call["kv"]["sliding_attention"][0][:, :, -(WINDOW + 1):]
        seen.add(given.shape[2])
        assert given.shape[2] in (WINDOW, WINDOW + 1)
        torch.testing.assert_close(given, clean["sliding_attention"][0][:, :, -given.shape[2]:], rtol=1e-5,
                                   atol=1e-5 * float(clean["sliding_attention"][0].abs().max()))
    assert seen == {WINDOW, WINDOW + 1}                 # the first cycle has W + 1; the random drafter accepts nothing


def test_two_drafters_and_both_kinds_of_keys_in_one_run(tiny, tmp_path):
    out = tmp_path / "table"
    args = ["--model", str(tiny / "model"), "--drafter", f"a1={tiny / 'a1'}", "--drafter", f"a2={tiny / 'a2'}",
            "--prompts", str(tiny / "prompts.jsonl"), "--out", str(out), "--max-new-tokens", "12", "--warmup", "1",
            "--dtype", "fp32", "--device", "cpu", "--arms", "ar", "a1:2", "a2:2", "a1:2:clean", "a2:2:clean"]
    assert de.main(args) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert list(summary["arms"]) == ["ar", "a1-2", "a2-2", "a1-2-clean", "a2-2-clean"]
    assert summary["drafters"] == {"a1": str(tiny / "a1"), "a2": str(tiny / "a2")}
    assert all(s["exact"] == 3 for s in summary["arms"].values())         # float32: every arm keeps the text
    records = [json.loads(line) for line in open(out / "samples.jsonl", encoding="utf-8")]
    assert all(r["drafter_passes"] > 0 for r in records if r["arm"] != "ar")
    with pytest.raises(SystemExit, match="an arm is"):
        de.main([*args[:-5], "ar", "a3:2"])
    with pytest.raises(SystemExit, match="LABEL=PATH"):
        de.main(["--drafter", "mtp=x", *args])
    readings = rd.read_pytorch(out)
    assert set(readings["r1"]) == {"a1-2", "a2-2", "a1-2-clean", "a2-2-clean"}
    assert [x["keys"] for x in readings["r6"]] == ["Transformers'", "accepted prefix"]
    assert [(x["drafter"], x["g"]) for x in readings["r9"]] == [("a1", 2), ("a2", 2)]
    assert rd.main(["best", "--table", str(out), "--drafter", "a2"]) == 0


# ---------------------------------------------------------------------------------------------
# The readings, on tables whose answers are worked out by hand
# ---------------------------------------------------------------------------------------------

#                       target passes, decode seconds; each prompt has 21 tokens, the first one from the prefill
PLAN = {"ar": (21, 2.0), "a1-2": (11, 1.25), "a2-2": (9, 1.0), "a1-4": (9, 1.6), "a2-4": (7, 1.25),
        "a1-4-clean": (8, 1.5), "a2-4-clean": (6, 1.2)}


def _table(path, plan, vary=None):
    """A decode table of 3 prompts. ``vary`` maps (arm, prompt) to other (passes, seconds)."""
    records = []
    for p in ("p0", "p1", "p2"):
        for arm, (passes, seconds) in plan.items():
            passes, seconds = (vary or {}).get((arm, p), (passes, seconds))
            tokens = list(range(21))
            if arm == "a2-4" and p == "p1":
                tokens[5] = 99                          # not token-identical on one prompt
            records.append({"id": p, "arm": arm, "tokens": tokens, "seconds": seconds + 0.5, "first_seconds": 0.5,
                            "first_tokens": 1, "target_passes": passes, "drafter_passes": 0})
    path.mkdir(parents=True)
    with open(path / "samples.jsonl", "w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r) + "\n")
    environment = {"device": "cuda:0", "gpu": "a stand-in", "dtype": "fp32", "runtime": "PyTorch, eager"}
    (path / "summary.json").write_text(json.dumps({"environment": environment, "max_new_tokens": 20,
                                                   "arms": summarize(records, list(plan), resamples=200)}))
    return path


def test_the_pytorch_readings_by_hand(tmp_path):
    r = rd.read_pytorch(_table(tmp_path / "t", PLAN))
    # Per prompt, 20 tokens after the first. Tokens per cycle = 20 / (passes - 1); decode rate = 20 / seconds.
    assert r["r1"]["a2-4"] == {"exact": 2, "of": 3, "reading": "not token-identical"}
    assert r["r1"]["a1-2"]["reading"] == "exact"
    # R2: a1's speed-ups are 16 / 10 = 1.6 at G = 2 and 12.5 / 10 = 1.25 at G = 4; a2's 2.0 and 1.6. Identical
    # prompts: every resample gives the point, so the intervals are the points.
    assert r["r2"]["a1"]["best_g"] == 2 and r["r2"]["a1"]["speedup"] == pytest.approx(1.6)
    assert r["r2"]["a2"]["best_g"] == 2 and r["r2"]["a2"]["reading"] == "pays clearly"
    six = {(x["g"], x["keys"]): x for x in r["r6"]}
    assert six[(2, "Transformers'")]["difference"]["point"] == pytest.approx(20 / 8 - 20 / 10)
    assert six[(4, "Transformers'")]["difference"]["point"] == pytest.approx(20 / 6 - 20 / 8)
    assert six[(4, "accepted prefix")]["difference"]["point"] == pytest.approx(20 / 5 - 20 / 7)
    assert all(x["reading"] == "a2 accepts more" for x in r["r6"])
    assert r["r7"]["second"] == "a2-2" and r["r7"]["first"] == "a1-2"
    assert r["r7"]["ratio"]["point"] == pytest.approx(20.0 / 16.0) and r["r7"]["reading"] == "a2 pays over a1 clearly"
    nine = {x["drafter"]: x for x in r["r9"]}
    assert nine["a1"]["tokens_per_cycle"]["point"] == pytest.approx(20 / 7 - 20 / 8)
    assert nine["a2"]["rate_ratio"]["point"] == pytest.approx((20 / 1.2) / (20 / 1.25))
    assert nine["a1"]["reading"] == "the accepted prefix's keys accept more"
    # Undecided when the prompts disagree: a2 at G = 2 accepts more on p0 and fewer on p1 and p2.
    vary = {("a2-2", "p1"): (13, 1.4), ("a2-2", "p2"): (13, 1.4)}
    u = rd.read_pytorch(_table(tmp_path / "u", PLAN, vary))
    two = next(x for x in u["r6"] if x["g"] == 2 and x["keys"] == "Transformers'")
    assert two["difference"]["point"] == pytest.approx(20 * 3 / (8 + 12 + 12) - 2.0)
    assert two["difference"]["low"] < 0 < two["difference"]["high"] and two["reading"] == "undecided"
    assert rd.verdict(1.01, 1.2, 1.0, "up", "down") == "up" and rd.verdict(0.8, 0.99, 1.0, "up", "down") == "down"
    assert rd.verdict(0.99, 1.01, 1.0, "up", "down") == "undecided"
    text = rd.render([r])
    assert "R7: decode rate a2-2 / a1-2 = 1.250 [1.250, 1.250]: a2 pays over a1 clearly" in text


def test_a_one_pass_drafters_readings_by_hand(tmp_path):
    plan = {"ar": (21, 2.0), "a1-2": (11, 1.25), "a1-2-clean": (11, 1.2), "a2-4": (7, 1.25), "a2-4-clean": (6, 1.2),
            "a30-2": (11, 1.1), "a30-8": (8, 1.0)}
    bars = {"bars": [{"against": "a1", "registered": True, "g": 2, "by_block": [{"b": 8, "tau_star": 1.816}]},
                     {"against": "a2", "registered": False, "g": 4, "by_block": [{"b": 8, "tau_star": 1.922}]}]}
    r = rd.read_one_pass(_table(tmp_path / "t", plan), "a30", bars=bars)
    # 20 tokens per prompt after the first; decode rate = 20 / seconds: ar 10, a1-2 16, a1-2-clean 16.67, a2-4 16,
    # a2-4-clean 16.67, a30-2 18.18, a30-8 20. Identical prompts: every interval is its point.
    assert r["r2"]["a30"]["best_g"] == 8 and r["r2"]["a30"]["speedup"] == pytest.approx(2.0)
    assert r["r1"]["a30-8"]["reading"] == "exact" and r["r1"]["a2-4"]["exact"] == 2
    against = {(x["reference"], x["keys"]): x for x in r["against"]}
    assert against[("a1-2", "Transformers'")]["ratio"]["point"] == pytest.approx(20 / 16)
    assert against[("a1-2-clean", "accepted prefix")]["ratio"]["point"] == pytest.approx(1.2)
    assert against[("a2-4", "Transformers'")]["reading"] == "a30 pays over a1 clearly".replace("a1", "a2")
    assert r["tokens_per_cycle"]["a30-8"] == pytest.approx(20 / 7)
    assert r["bars"][0]["by_block"] == {8: 1.816}
    text = rd.render_one_pass(r)
    assert "R7's rule, a30 at G = 8 against a1-2 (Transformers' keys): decode rate ratio 1.250 [1.250, 1.250]: a30 " \
           "pays over a1 clearly" in text
    assert "tau* 1.922 at B = 8 (predicted)" in text
    assert rd.main(["one-pass", "--table", str(tmp_path / "t"), "--drafter", "a30", "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "one-pass-a30.md").exists()
    with pytest.raises(SystemExit, match="no arm of a3"):
        rd.main(["one-pass", "--table", str(tmp_path / "t"), "--drafter", "a3", "--out", str(tmp_path / "o")])


def _vllm(path, drift):
    arms = {"off-1": {"decode_speedup": None, "same_tokens": 3, "prompts": 3, "tokens_per_cycle": None},
            "a1-2": {"decode_speedup": 1.5, "decode_speedup_interval": [1.4, 1.6], "same_tokens": 1, "prompts": 3,
                     "tokens_per_cycle": 1.9},
            "a1-4": {"decode_speedup": 1.7, "decode_speedup_interval": [1.6, 1.8], "same_tokens": 1, "prompts": 3,
                     "tokens_per_cycle": 2.2},
            "a2-4": {"decode_speedup": 1.9, "decode_speedup_interval": [1.8, 2.0], "same_tokens": 2, "prompts": 3,
                     "tokens_per_cycle": 2.6},
            "off-2": {"decode_speedup": 1.0, "decode_speedup_interval": list(drift), "same_tokens": 3, "prompts": 3,
                      "tokens_per_cycle": None}}
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps({"environment": {"device": "cuda", "dtype": "bf16",
                                                                   "runtime": "vLLM"}, "arms": arms}))
    for tag, seconds in (("a1-4", 2.0), ("a2-4", 1.6)):
        (path / "arms" / tag).mkdir(parents=True)
        with open(path / "arms" / tag / "samples.jsonl", "w", encoding="utf-8") as handle:
            for p in ("p0", "p1", "p2"):
                handle.write(json.dumps({"id": p, "tokens": list(range(41)), "decode_seconds": seconds}) + "\n")
    return path


def test_the_vllm_readings_follow_the_drift_rule(tmp_path):
    read = rd.read_vllm(_vllm(tmp_path / "in", (0.99, 1.01)))
    assert read["drift"]["read"] and read["r2"]["a1"]["best_g"] == 4 and read["r2"]["a2"]["reading"] == "pays clearly"
    # R7 by hand: 40 tokens after the first per prompt; (40 / 1.6) / (40 / 2.0) = 1.25 on every prompt.
    assert read["r7"]["ratio"]["point"] == pytest.approx(1.25) and read["r7"]["reading"] == "a2 pays over a1 clearly"
    out = rd.read_vllm(_vllm(tmp_path / "out", (0.97, 0.99)))
    assert not out["drift"]["read"] and "r7" not in out and out["r2"]["a2"]["reading"] == "not read (S11)"
    assert out["r1"]["a2-4"] == {"same_tokens": 2, "of": 3}
    assert rd.main(["read", "--tables", str(_table(tmp_path / "t", PLAN)), "--vllm", str(tmp_path / "in"),
                    "--out", str(tmp_path / "r")]) == 0
    assert (tmp_path / "r" / "readings.md").read_text().count("## ") == 2


# ---------------------------------------------------------------------------------------------
# A3's break-even (docs/e1.md, Sections 6.1 and 10.3) against Step 0's own computation by hand
# ---------------------------------------------------------------------------------------------

STEP0_BF16 = {"rates": {2: 29.7, 4: 28.9, 6: 26.5, 8: 24.2}, "tau": {2: 1.88, 4: 2.12, 6: 2.21, 8: 2.24},
              "speedup": {2: 1.43, 4: 1.39, 6: 1.28, 8: 1.17},
              "t_ms": {1: 48.72, 2: 53.10, 3: 53.17, 4: 53.41, 5: 53.40, 6: 53.40, 8: 53.65, 12: 53.62, 16: 53.62,
                       24: 53.83, 32: 53.86}}


def _break_even_inputs(tmp_path, label="mtp", gpu="NVIDIA L4", dtype="bf16"):
    """Step 0's bfloat16 decode table and ceiling as Section 10.1 prints them (rounded)."""
    environment = {"device": "cuda", "gpu": gpu, "dtype": dtype, "runtime": "PyTorch, eager"}
    arms = {"ar": {"prompts": 100, "decode_tokens_per_second": 20.7, "tokens_per_cycle": 1.0}}
    for g, rate in STEP0_BF16["rates"].items():
        arms[f"{label}-{g}"] = {"prompts": 100, "decode_tokens_per_second": rate, "tokens_per_cycle":
                                STEP0_BF16["tau"][g], "decode_speedup": STEP0_BF16["speedup"][g]}
    table, ceiling = tmp_path / "decode", tmp_path / "ceiling"
    table.mkdir(parents=True, exist_ok=True)
    ceiling.mkdir(parents=True, exist_ok=True)
    (table / "summary.json").write_text(json.dumps({"environment": environment, "max_new_tokens": 256, "arms": arms}))
    rows = [{"n": n, "seconds": t / 1e3} for n, t in STEP0_BF16["t_ms"].items()]
    (ceiling / "summary.json").write_text(json.dumps({"environment": {**environment, "gpu": "NVIDIA L4"},
                                                      "rows": rows}))
    return table, ceiling


def test_the_break_even_is_step_0s_hand_computation(tmp_path):
    table, ceiling = _break_even_inputs(tmp_path)
    r = rd.break_even(table, ceiling, references=("mtp",), blocks=(2, 4, 8, 16))
    # Section 10.3, item 1: cycles of 63.30, 73.36, 83.40 and 92.56 ms; without t(G + 1) (t(7) and t(9)
    # interpolated), 0.65 ms + 4.81 ms x G remain, residuals within 0.34 ms.
    cycles = {a["g"]: a["cycle_ms"] for a in r["arms"]}
    assert [round(cycles[g], 2) for g in (2, 4, 6, 8)] == [63.30, 73.36, 83.40, 92.56]
    passes = {a["g"]: (a["pass_ms"], a["pass_interpolated"]) for a in r["arms"]}
    assert passes[6] == (pytest.approx((53.40 + 53.65) / 2), True)             # t(7), halfway between 6 and 8
    assert passes[8] == (pytest.approx(53.65 + (53.62 - 53.65) / 4), True)     # t(9), a quarter of 8 to 12
    assert passes[2] == (pytest.approx(53.17), False)
    fit = r["fits"]["mtp"]
    assert round(fit["call_ms"], 2) == 4.81 and round(fit["intercept_ms"], 2) == 0.65
    assert max(abs(x) for x in fit["residuals_ms"]) < 0.34
    assert round(fit["call_over_t1"], 3) == 0.099
    # Item 6: G* = 2, tau_1 = 1.88, T_1 = 63.30 ms, D_1 = 2 calls, D_3 = 1.25 calls: tau* = 1.77 at B = 2, 1.78 at
    # B = 4 and 1.79 at B = 8 and at B = 16.
    bar = r["bars"][0]
    assert bar["registered"] and bar["g"] == 2 and bar["tau"] == 1.88
    assert bar["drafter_ms"] == pytest.approx(2 * fit["call_ms"]) and bar["a3_drafting_ms"] == pytest.approx(
        1.25 * fit["call_ms"])
    assert [round(x["tau_star"], 2) for x in bar["by_block"]] == [1.77, 1.78, 1.79, 1.79]
    # The formula by hand at B = 8: 1.88 (T_1 - D_1 + D_3 + t(9) - t(3)) / T_1.
    t1, d = 1.88 / 29.7 * 1e3, fit["call_ms"]
    by_hand = 1.88 * (t1 - 2 * d + 1.25 * d + (53.65 - 0.03 / 4) - 53.17) / t1
    assert bar["by_block"][2]["tau_star"] == pytest.approx(by_hand)
    assert r["gate"]["blocked"] == [] and "does not stop A3" in r["gate"]["reading"]
    text = rd.render_break_even(r)
    assert "| mtp (the registered gate) | 2 | 1.880 | 63.30 |" in text and "*i*" in text


def test_the_break_even_refuses_costs_from_another_precision_and_marks_a_blocked_gate(tmp_path):
    table, ceiling = _break_even_inputs(tmp_path, dtype="fp32")
    summary = json.loads((ceiling / "summary.json").read_text())
    summary["environment"]["dtype"] = "bf16"
    (ceiling / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="different dtype"):
        rd.break_even(table, ceiling, references=("mtp",))
    # A pass over B + 1 positions far dearer than anything the cycle saves: t(24) = 10 s puts t(17) at 1.30 s, so the
    # bar at B = 16 lies above 17 tokens per cycle and the gate stops A3 there; the fit (G + 1 <= 9) is untouched.
    table, ceiling = _break_even_inputs(tmp_path / "dear")
    summary = json.loads((ceiling / "summary.json").read_text())
    for row in summary["rows"]:
        if row["n"] == 24:
            row["seconds"] = 10.0
    (ceiling / "summary.json").write_text(json.dumps(summary))
    r = rd.break_even(table, ceiling, references=("mtp",), blocks=(8, 16))
    d, t1 = r["fits"]["mtp"]["call_ms"], 1.88 / 29.7 * 1e3
    t17 = 53.62 + (10_000 - 53.62) / 8
    assert r["bars"][0]["by_block"][1]["tau_star"] == pytest.approx(1.88 * (t1 - 0.75 * d + t17 - 53.17) / t1)
    assert r["bars"][0]["by_block"][1]["tau_star"] > 17 and r["gate"]["blocked"] == [16]
    assert "no drafter of A3's kind can pay there" in r["gate"]["reading"]
    assert rd.main(["breakeven", "--table", str(table), "--ceiling", str(ceiling), "--against", "mtp",
                    "--out", str(tmp_path / "out")]) == 0
    assert json.loads((tmp_path / "out" / "breakeven.json").read_text())["bars"][0]["against"] == "mtp"
    with pytest.raises(SystemExit, match="two draft lengths"):
        rd.main(["breakeven", "--table", str(table), "--ceiling", str(ceiling), "--against", "a9",
                 "--out", str(tmp_path / "out")])
    assert rd.pass_seconds([{"n": 1, "seconds": 1.0}, {"n": 3, "seconds": 2.0}], 2) == (1.5, True)
    with pytest.raises(ValueError, match="neither measured"):
        rd.pass_seconds([{"n": 1, "seconds": 1.0}, {"n": 3, "seconds": 2.0}], 4)
