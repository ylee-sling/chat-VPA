# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""E1's Step 0 in vLLM (docs/e1.md, Section 3.4), without vLLM: a stand-in for the ``vllm`` package records what the
measurement asks of it. The engine of each arm gets the settings the docstring states, and an mtp arm Google's
drafter through vLLM's Gemma 4 MTP path; every prompt reaches the engine as the token ids of ``chat_ids``; the
warm-up is decoded first and not recorded; a record is what the engine's output says; the summary is the
arithmetic of the records, computed here by hand; the report prints it with the drift check. The real engine runs
first on the server: vLLM needs a GPU, and it is not installed in the test environment."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field

import pytest
import transformers

from chat_vpa import e1_report
from chat_vpa import e1_vllm as vl


class FakeTokenizer:
    """A chat template of its own: the rendering is ``<u>`` + prompt + ``</u><m>``, one id per character."""

    chat_template = "fake"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert tokenize is False and add_generation_prompt is True and len(messages) == 1
        return f"<u>{messages[0]['content']}</u><m>"

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}


def expected_ids(prompt):
    return [ord(c) for c in f"<u>{prompt}</u><m>"]


@dataclass
class Stats:
    scheduled_ts: float
    first_token_ts: float
    last_token_ts: float


@dataclass
class Spec:
    histogram: list
    num_draft_tokens: int
    num_spec_tokens: int

    def to_dict(self):
        steps = sum(self.histogram)
        accepted = sum(j * c for j, c in enumerate(self.histogram))
        return {"mean_acceptance_length": 1 + accepted / steps if steps else 1.0,
                "draft_acceptance_rate": accepted / self.num_draft_tokens if self.num_draft_tokens else 0.0,
                "acceptance_histogram": list(self.histogram), "num_spec_steps": steps,
                "num_accepted_draft_tokens": accepted, "num_draft_tokens": self.num_draft_tokens,
                "num_spec_tokens": self.num_spec_tokens}


@dataclass
class Completion:
    token_ids: list
    text: str
    finish_reason: str = "stop"
    spec_decode_metrics: Spec | None = None


@dataclass
class Output:
    outputs: list
    metrics: Stats | None


@dataclass
class FakeLLM:
    """Answers a prompt by its length: len(ids) % 7 + 2 tokens, a prefill of 0.01 s and 0.001 s per token after."""

    kwargs: dict
    log: list = field(default_factory=list)
    stats: bool = True

    def generate(self, prompts, sampling_params, use_tqdm):
        assert use_tqdm is False and len(prompts) == 1
        ids = prompts[0]["prompt_token_ids"]
        self.log.append((list(ids), sampling_params))
        n = len(ids) % 7 + 2
        spec = None
        if "speculative_config" in self.kwargs:
            k = self.kwargs["speculative_config"]["num_speculative_tokens"]
            spec = Spec(histogram=[1] + [0] * (k - 1) + [1], num_draft_tokens=2 * k, num_spec_tokens=k)
        stats = Stats(1.0, 1.01, 1.01 + 0.001 * (n - 1)) if self.stats else None
        return [Output([Completion(list(range(n)), "x" * n, spec_decode_metrics=spec)], stats)]


def fake_vllm(engines):
    module = types.ModuleType("vllm")
    module.__version__ = "0.34.0"

    def llm(**kwargs):
        engine = FakeLLM(kwargs)
        engines.append(engine)
        return engine

    @dataclass
    class SamplingParams:
        temperature: float
        max_tokens: int

    module.LLM, module.SamplingParams = llm, SamplingParams
    return module


PROMPTS = [{"id": "a", "prompt": "one"}, {"id": "b", "prompt": "three"}, {"id": "c", "prompt": "fifteen"}]


def test_arms_tags_and_the_engine_of_each_arm(tmp_path):
    assert vl.parse_vllm_arm("off") == vl.VllmArm("off") and vl.parse_vllm_arm("mtp:4") == vl.VllmArm("mtp", 4)
    for bad in ("mtp:0", "mtp", "lookup:3", "mtp:x"):
        with pytest.raises(ValueError):
            vl.parse_vllm_arm(bad)
    assert vl.tag_of(["off", "mtp:2", "mtp:4", "off"]) == ["off-1", "mtp-2", "mtp-4", "off-2"]
    with pytest.raises(ValueError):
        vl.tag_of(["off", "mtp:2", "mtp:2"])

    common = {"model": "M", "dtype": "bfloat16", "seed": 0, "max_model_len": 4096, "gpu_memory_utilization": 0.85,
              "enable_prefix_caching": False, "language_model_only": True, "disable_log_stats": False}
    assert vl.engine_kwargs("M", None, vl.VllmArm("off")) == common
    assert vl.engine_kwargs("M", tmp_path, vl.VllmArm("mtp", 4)) == {
        **common, "speculative_config": {"method": "mtp", "model": str(tmp_path), "num_speculative_tokens": 4},
        "per_request_spec_decode_metrics": "summary"}
    with pytest.raises(ValueError, match="assistant"):
        vl.engine_kwargs("M", None, vl.VllmArm("mtp", 2))


def test_every_prompt_reaches_the_engine_as_chat_ids_after_the_warmup():
    engine = FakeLLM({"speculative_config": {"num_speculative_tokens": 3}})
    seen: list[str] = []
    records = vl.run_arm(engine, "params", FakeTokenizer(), PROMPTS, warmup=2, progress=seen.append)
    sent = [ids for ids, _ in engine.log]
    assert sent == [expected_ids("one"), expected_ids("three")] + [expected_ids(r["prompt"]) for r in PROMPTS]
    assert all(params == "params" for _, params in engine.log)
    assert [r["id"] for r in records] == ["a", "b", "c"]
    first = records[0]                       # "<u>one</u><m>": 13 ids, 13 % 7 + 2 = 8 tokens
    assert first["prompt_tokens"] == 13 and first["tokens"] == list(range(8)) and first["text"] == "x" * 8
    assert first["prefill_seconds"] == pytest.approx(0.01) and first["decode_seconds"] == pytest.approx(0.007)
    assert first["spec"]["acceptance_histogram"] == [1, 0, 0, 1] and first["spec"]["num_draft_tokens"] == 6
    assert first["finish_reason"] == "stop" and first["seconds"] >= 0
    assert seen == ["  3/3 prompts"]
    with pytest.raises(RuntimeError, match="disable_log_stats"):
        vl.run_arm(FakeLLM({}, stats=False), "params", FakeTokenizer(), PROMPTS[:1], warmup=0)


def record(id_, tokens, seconds, decode, prefill=0.05, spec=None):
    return {"id": id_, "prompt_tokens": 10, "tokens": tokens, "text": "", "finish_reason": "stop",
            "seconds": seconds, "prefill_seconds": prefill, "decode_seconds": decode, "spec": spec}


def spec(histogram, drafted):
    return Spec(histogram, drafted, len(histogram) - 1).to_dict()


def test_the_vllm_summary_is_the_arithmetic_of_the_records():
    arms = {
        # off-1: 10 + 20 tokens; after the first, 9 + 19 = 28 in 0.1 + 0.18 = 0.28 s: 100 tok/s; end to end 30 in 0.4
        "off-1": [record("p", list(range(10)), 0.15, 0.10, prefill=0.04),
                  record("q", list(range(20)), 0.25, 0.18, prefill=0.06)],
        # mtp-4: the same 28 decode tokens in 0.07 + 0.07 = 0.14 s: 200 tok/s, 2x; end to end 30 in 0.2: 150, 2x.
        # Its second response differs from off-1's at position 5. Steps 3 + 3; accepted 2 + 4 + 4 + 4 = 14 of 24
        "mtp-4": [record("p", list(range(10)), 0.08, 0.07, spec=spec([1, 0, 1, 0, 1], 12)),
                  record("q", list(range(5)) + [99] + list(range(6, 20)), 0.12, 0.07,
                         spec=spec([1, 0, 0, 0, 2], 12))],
        # off-2: as off-1, a little slower: 28 in 0.308 s
        "off-2": [record("p", list(range(10)), 0.16, 0.11), record("q", list(range(20)), 0.26, 0.198)],
    }
    s = vl.summarize(arms, resamples=2000, seed=0)
    assert list(s) == ["off-1", "mtp-4", "off-2"]
    off, mtp, drift = s["off-1"], s["mtp-4"], s["off-2"]
    assert off["decode_tokens_per_second"] == pytest.approx(100.0)
    assert off["end_to_end_tokens_per_second"] == pytest.approx(75.0)
    assert "decode_speedup" not in off and off["same_tokens"] == 2 and off["first_differences"] == {}
    assert off["median_prefill_seconds"] == pytest.approx(0.05)
    assert mtp["decode_tokens_per_second"] == pytest.approx(200.0) and mtp["decode_speedup"] == pytest.approx(2.0)
    assert mtp["end_to_end_speedup"] == pytest.approx(150.0 / 75.0)
    low, high = mtp["decode_speedup_interval"]
    # Per prompt the decode speed-ups are (9/0.07)/(9/0.10) = 1.43 and (19/0.07)/(19/0.18) = 2.57.
    assert 0.10 / 0.07 - 1e-9 <= low <= mtp["decode_speedup"] <= high <= 0.18 / 0.07 + 1e-9
    assert mtp["same_tokens"] == 1 and mtp["first_differences"] == {"q": 5}
    assert mtp["acceptance_histogram"] == [2, 0, 1, 0, 3]
    assert mtp["tokens_per_cycle"] == pytest.approx(1 + 14 / 6)
    assert mtp["draft_acceptance_rate"] == pytest.approx(14 / 24)
    assert drift["decode_speedup"] == pytest.approx((28 / 0.308) / 100.0)
    assert drift["tokens_per_cycle"] is None and drift["same_tokens"] == 2

    with pytest.raises(ValueError, match="reference"):
        vl.summarize({"mtp-4": arms["mtp-4"]})
    with pytest.raises(ValueError, match="same order"):
        vl.summarize({"off-1": arms["off-1"], "mtp-4": arms["mtp-4"][::-1]})


def test_the_commands_run_each_arm_alone_and_summarize_them(tmp_path, monkeypatch, capsys):
    engines: list = []
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm(engines))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    prompts = tmp_path / "dev.jsonl"
    prompts.write_text("".join(json.dumps(row) + "\n" for row in PROMPTS))
    out = tmp_path / "vllm" / "gpu-bf16"
    common = ["--model", str(tmp_path / "model"), "--assistant", str(tmp_path / "drafter"), "--prompts",
              str(prompts), "--out", str(out), "--max-new-tokens", "9", "--warmup", "1"]
    for arm, tag in zip(["off", "mtp:4", "off"], vl.tag_of(["off", "mtp:4", "off"]), strict=True):
        assert vl.main(["run", "--arm", arm, "--tag", tag, *common]) == 0
    assert [e.kwargs.get("speculative_config") for e in engines] == [
        None, {"method": "mtp", "model": str(tmp_path / "drafter"), "num_speculative_tokens": 4}, None]
    for engine in engines:
        assert [ids for ids, _ in engine.log] == [expected_ids("one")] + [expected_ids(r["prompt"]) for r in PROMPTS]
        assert all((p.temperature, p.max_tokens) == (0.0, 9) for _, p in engine.log)
    meta = json.loads((out / "arms" / "mtp-4" / "arm.json").read_text())
    assert meta["arm"] == "mtp:4" and meta["engine"]["per_request_spec_decode_metrics"] == "summary"
    assert meta["prompts"]["count"] == 3 and meta["environment"]["runtime"].startswith("vLLM 0.34.0")

    assert vl.main(["summarize", "--out", str(out), "--resamples", "500"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["order"] == ["off-1", "mtp-4", "off-2"] and summary["reference"] == "off-1"
    assert summary["speculative_config"]["mtp-4"]["num_speculative_tokens"] == 4
    assert summary["arms"]["mtp-4"]["acceptance_histogram"] == [3, 0, 0, 0, 3]

    # The report prints the table and the drift check (the stand-in's off-2 is as fast as off-1: ratio 1).
    text = e1_report.render(tmp_path)
    assert "## vLLM: gpu-bf16" in text and "one engine per arm, run in the order off-1, mtp-4, off-2" in text
    assert "| off-1 |" in text and "| mtp-4 |" in text and "1 (reference)" in text
    assert "Drift: off-2's decode speed against off-1's is 1.000 [1.000, 1.000], which contains 1" in text
    assert "- mtp-4: verification steps by accepted draft tokens (0, 1, ...): [3, 0, 0, 0, 3]" in text

    # Arms run on other prompts cannot be compared.
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"id": "a", "prompt": "four"}) + "\n" + "".join(json.dumps(r) + "\n"
                                                                              for r in PROMPTS[1:]))
    common[common.index(str(prompts))] = str(other)
    assert vl.main(["run", "--arm", "mtp:2", "--tag", "mtp-2", *common]) == 0
    with pytest.raises(SystemExit, match="different prompt token ids"):
        vl.main(["summarize", "--out", str(out)])


def test_the_smoke_decode_reports_what_it_finds(tmp_path, monkeypatch, capsys):
    engines: list = []
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm(engines))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    prompts = tmp_path / "dev.jsonl"
    prompts.write_text("".join(json.dumps(row) + "\n" for row in PROMPTS))
    argv = ["smoke", "--model", "M", "--assistant", "D", "--prompts", str(prompts)]
    assert vl.main(argv) == 0
    printed = capsys.readouterr().out
    assert "vllm 0.34.0; arm mtp:4; 32 tokens of the first prompt" in printed
    assert "drafting: 2 verification steps, 4 of 8 draft tokens accepted, 3.00 tokens per cycle" in printed
    assert "UNEXPECTED" not in printed and len(engines) == 1 and [ids for ids, _ in engines[0].log] == [
        expected_ids("one")]

    def broken(**kwargs):
        raise RuntimeError("no   engine\nhere")

    sys.modules["vllm"].LLM = broken
    assert vl.main(argv) == 1
    assert "UNEXPECTED: vLLM could not decode with mtp:4: RuntimeError: no engine here" in capsys.readouterr().out

    sys.modules["vllm"].LLM = lambda **kwargs: FakeLLM({})                 # an engine that never drafts
    assert vl.main(argv) == 1
    assert "UNEXPECTED: no acceptance counts" in capsys.readouterr().out


def test_without_vllm_the_command_names_the_interpreter(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    prompts = tmp_path / "dev.jsonl"
    prompts.write_text(json.dumps(PROMPTS[0]) + "\n")
    with pytest.raises(SystemExit, match="vLLM's environment"):
        vl.main(["run", "--arm", "off", "--tag", "off-1", "--model", "M", "--prompts", str(prompts), "--out",
                 str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="no arm has run"):
        vl.main(["summarize", "--out", str(tmp_path / "nothing")])
