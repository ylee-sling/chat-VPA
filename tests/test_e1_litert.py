# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""E1's Step 0 in LiteRT-LM (docs/e1.md, Section 3.4), without the runtime: a stand-in for the ``litert_lm``
package records what the measurement asks of it. Every prompt goes through every mode back to back, after the
warm-up; a mode whose engine cannot be created is recorded and the others are measured; the summary is the
arithmetic of the records, computed here by hand; and the report prints it. The real runtime is exercised by the
dry run of ``scripts/ncp/e1_step0.sh`` (it is not installed in the test environment)."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass

import pytest

from chat_vpa import e1_litert as lt
from chat_vpa import e1_report


def test_the_text_of_a_message_and_the_first_difference():
    assert lt.message_text({"role": "assistant", "content": "plain"}) == "plain"
    parts = [{"type": "text", "text": "a"}, {"type": "image", "data": "..."}, {"type": "text", "text": "b"}]
    assert lt.message_text({"role": "assistant", "content": parts}) == "ab"
    assert lt.message_text({"role": "assistant"}) == ""
    assert lt.first_text_difference("same", "same") is None
    assert lt.first_text_difference("abcd", "abXd") == 2
    assert lt.first_text_difference("abc", "abcde") == 3          # one is a prefix of the other
    assert lt.first_text_difference("", "x") == 0


# A stand-in runtime: a response is a function of (mode, prompt); every call is written to a shared log.
@dataclass
class _Info:
    last_decode_token_count: int
    last_decode_tokens_per_second: float
    time_to_first_token_in_second: float
    last_prefill_token_count: int


class _Conversation:
    def __init__(self, engine, kwargs):
        self.engine, self.kwargs, self.closed, self.last = engine, kwargs, False, None

    def render_message_to_string(self, message):
        return f"<user>{message['content']}</user><model>"

    def send_message(self, prompt):
        self.engine.log.append((self.engine.mode, prompt, dict(self.kwargs)))
        self.last = self.engine.table[(self.engine.mode, prompt)]
        return {"role": "assistant", "content": [{"type": "text", "text": self.last["text"]}]}

    def get_benchmark_info(self):
        return _Info(self.last["tokens"], self.last["rate"], self.last["first"], 7)

    def close(self):
        self.closed = True
        self.engine.open_conversations -= 1


class _Engine:
    def __init__(self, mode, table, log):
        self.mode, self.table, self.log, self.open_conversations, self.closed = mode, table, log, 0, False

    def create_conversation(self, **kwargs):
        self.open_conversations += 1
        return _Conversation(self, kwargs)

    def close(self):
        self.closed = True


PROMPTS = [{"id": "p0", "prompt": "zero"}, {"id": "p1", "prompt": "one"}, {"id": "p2", "prompt": "two"}]
# tokens, the runtime's decode rate, time to the first token, and the text, per (mode, prompt). By hand:
#   off: decode seconds 10/100 + 20/100 + 30/100 = 0.6 for 60 tokens: 100 tok/s
#   on:  decode seconds 10/250 + 20/200 + 30/100 = 0.44 for 60 tokens: 136.36 tok/s, a speed-up of 0.6 / 0.44
TABLE = {("off", "zero"): {"tokens": 10, "rate": 100.0, "first": 0.3, "text": "alpha beta"},
         ("off", "one"): {"tokens": 20, "rate": 100.0, "first": 0.1, "text": "gamma"},
         ("off", "two"): {"tokens": 30, "rate": 100.0, "first": 0.2, "text": "delta epsilon"},
         ("on", "zero"): {"tokens": 10, "rate": 250.0, "first": 0.3, "text": "alpha beta"},
         ("on", "one"): {"tokens": 20, "rate": 200.0, "first": 0.5, "text": "gamma"},
         ("on", "two"): {"tokens": 30, "rate": 100.0, "first": 0.4, "text": "delta zeta"}}


def test_every_prompt_runs_through_every_mode_back_to_back():
    log: list = []
    engines = {mode: _Engine(mode, TABLE, log) for mode in ("off", "on")}
    seen: list[str] = []
    records = lt.run(engines, PROMPTS, max_new_tokens=5, warmup=2, conversation_args={"sampler_config": "argmax"},
                     progress=seen.append)
    order = [(mode, prompt) for mode, prompt, _ in log]
    warm = [("off", "zero"), ("on", "zero"), ("off", "one"), ("on", "one")]
    measured = [(mode, row["prompt"]) for row in PROMPTS for mode in ("off", "on")]
    assert order == warm + measured                                   # the warm-up first, then mode by mode per prompt
    assert all(kwargs == {"max_output_tokens": 5, "sampler_config": "argmax"} for _, _, kwargs in log)
    assert all(engine.open_conversations == 0 for engine in engines.values())      # every conversation was closed
    assert [(r["id"], r["mode"]) for r in records] == [(row["id"], mode) for row in PROMPTS for mode in ("off", "on")]
    first = records[1]
    assert (first["text"], first["decode_tokens"], first["decode_tokens_per_second"], first["first_token_seconds"],
            first["prompt_tokens"]) == ("alpha beta", 10, 250.0, 0.3, 7)
    assert first["seconds"] >= 0.0
    assert seen == ["1 of 3 prompts", "2 of 3 prompts", "3 of 3 prompts"]


def _records(wall):
    return [{"id": row["id"], "mode": mode, "text": TABLE[(mode, row["prompt"])]["text"], "seconds": wall[mode][i],
             "decode_tokens": TABLE[(mode, row["prompt"])]["tokens"],
             "decode_tokens_per_second": TABLE[(mode, row["prompt"])]["rate"],
             "first_token_seconds": TABLE[(mode, row["prompt"])]["first"], "prompt_tokens": 7}
            for i, row in enumerate(PROMPTS) for mode in ("off", "on")]


def test_the_litert_summary_is_the_arithmetic_of_the_records():
    wall = {"off": [0.2, 0.3, 0.5], "on": [0.1, 0.2, 0.5]}            # 60 tokens in 1.0 s and in 0.8 s
    summary = lt.summarize(_records(wall), ["off", "on"], resamples=2000, seed=0)
    off, on = summary["off"], summary["on"]
    assert off["decode_tokens"] == on["decode_tokens"] == 60
    assert off["decode_tokens_per_second"] == pytest.approx(100.0)
    assert on["decode_tokens_per_second"] == pytest.approx(60 / 0.44)
    assert on["decode_speedup"] == pytest.approx(0.6 / 0.44)
    assert off["end_to_end_tokens_per_second"] == pytest.approx(60.0)
    assert on["end_to_end_tokens_per_second"] == pytest.approx(75.0)
    assert on["end_to_end_speedup"] == pytest.approx(1.25)
    assert "decode_speedup" not in off
    low, high = on["decode_speedup_interval"]
    # Per prompt the speed-ups are 2.5, 2 and 1: every resample lies between the smallest and the largest.
    assert 1.0 - 1e-9 <= low <= on["decode_speedup"] <= high <= 2.5 + 1e-9
    assert (off["median_seconds_to_first_token"], on["median_seconds_to_first_token"]) == (0.2, 0.4)
    assert (off["same_text"], on["same_text"], on["same_text_share"]) == (3, 2, pytest.approx(2 / 3))
    assert on["first_differences"] == {"p2": 6} and off["first_differences"] == {}     # "delta e" against "delta z"
    assert on["same_token_count"] == 3

    with pytest.raises(ValueError, match="reference"):
        lt.summarize([r for r in _records(wall) if r["mode"] == "on"], ["on"])
    with pytest.raises(ValueError, match="does not cover"):
        lt.summarize(_records(wall)[:-1], ["off", "on"])
    broken = _records(wall)
    broken[1]["decode_tokens"] = 0
    with pytest.raises(ValueError, match="without decoded tokens"):
        lt.summarize(broken, ["off", "on"])


def _fake_runtime(log, fail=("adaptive",), thinking=True):
    """A module that stands in for ``litert_lm``: the classes the measurement touches, and nothing else."""
    module = types.ModuleType("litert_lm")

    class Mode:
        OFF, ON, ADAPTIVE = "off", "on", "adaptive"

    class Capability:
        default_sampler_params = "SamplerConfig(top_k=64)"

        def has_speculative_decoding_support(self):
            return True

        def supports_thinking(self):
            return thinking

    class ModelInfo:
        max_context_tokens = 4096

        def __init__(self, path):
            self.llm = Capability()

        def close(self):
            pass

    class Backend:
        @dataclass
        class CPU:
            thread_count: int | None = None

        @dataclass
        class GPU:
            pass

    @dataclass
    class SpeculativeDecodingConfig:
        mode: str | None = None
        lenience: float | None = None

    @dataclass
    class SamplerConfig:
        top_k: int | None = None
        temperature: float | None = None
        seed: int | None = None

    @dataclass
    class ThinkingConfig:
        enable_thinking: bool = True

    def engine(path, backend, cache_dir, enable_benchmark, speculative_decoding_config):
        mode = speculative_decoding_config.mode
        log.append(("engine", mode, backend, enable_benchmark, speculative_decoding_config.lenience))
        if mode in fail:
            raise RuntimeError(f"no   engine\nfor {mode}")
        return _Engine(mode, TABLE, log)

    module.SpeculativeDecodingMode, module.ModelInfo, module.Backend = Mode, ModelInfo, Backend
    module.SpeculativeDecodingConfig, module.SamplerConfig, module.ThinkingConfig = (
        SpeculativeDecodingConfig, SamplerConfig, ThinkingConfig)
    module.Engine = engine
    module.LogSeverity = types.SimpleNamespace(ERROR=3)
    module.set_min_log_severity = lambda level: None
    return module


def test_the_command_measures_the_modes_that_start_and_records_the_one_that_does_not(tmp_path, monkeypatch, capsys):
    log: list = []
    fake = _fake_runtime(log)
    monkeypatch.setitem(sys.modules, "litert_lm", fake)
    model = tmp_path / "litert" / "model.litertlm"
    model.parent.mkdir()
    model.write_bytes(b"not a model")
    prompts = tmp_path / "dev.jsonl"
    prompts.write_text("".join(json.dumps(row) + "\n" for row in PROMPTS))
    out = tmp_path / "out"
    assert lt.main(["--model", str(model), "--prompts", str(prompts), "--out", str(out), "--threads", "3",
                    "--max-new-tokens", "9", "--warmup", "1"]) == 0
    printed = capsys.readouterr().out
    assert "mode adaptive could not be started: RuntimeError: no engine for adaptive" in printed

    engines = [entry for entry in log if entry[0] == "engine"]
    assert [entry[1] for entry in engines] == ["off", "on", "adaptive"]
    assert all(entry[2] == fake.Backend.CPU(thread_count=3) and entry[3] is True and entry[4] == 0.0
               for entry in engines)                                   # benchmark on, lenience 0: the lossless setting
    turns = [entry for entry in log if entry[0] != "engine"]
    assert [(mode, prompt) for mode, prompt, _ in turns] == (
        [("off", "zero"), ("on", "zero")] + [(mode, row["prompt"]) for row in PROMPTS for mode in ("off", "on")])
    for _, _, kwargs in turns:
        assert kwargs == {"max_output_tokens": 9, "sampler_config": fake.SamplerConfig(top_k=1, temperature=1.0, seed=0),
                          "thinking_config": fake.ThinkingConfig(enable_thinking=False)}
    assert (model.parent / "cache").is_dir()

    summary = json.loads((out / "summary.json").read_text())
    assert list(summary["modes"]) == ["off", "on"] and list(summary["failed"]) == ["adaptive"]
    assert summary["modes"]["on"]["decode_speedup"] == pytest.approx(0.6 / 0.44)
    assert summary["modes"]["on"]["first_differences"] == {"p2": 6}
    assert summary["environment"]["backend"] == "cpu" and summary["environment"]["cpu_threads"] == 3
    assert summary["model"]["speculative_decoding_support"] is True and summary["thinking"] == "off"
    assert summary["first_prompt_as_rendered"] == "<user>zero</user><model>"
    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    assert [(s["id"], s["mode"], s["text"]) for s in samples] == [
        (row["id"], mode, TABLE[(mode, row["prompt"])]["text"]) for row in PROMPTS for mode in ("off", "on")]

    # The report prints the table, the failed mode and where the texts part (the row by hand: 136.4 tok/s, 1.36x).
    (tmp_path / "step0" / "litert").mkdir(parents=True)
    (out).rename(tmp_path / "step0" / "litert" / "cpu")
    text = e1_report.render(tmp_path / "step0")
    assert "## LiteRT-LM: cpu" in text and "backend cpu, 3 of" in text and "thinking off" in text
    assert "| on | 136.4 | 1.36 [" in text and "| off | 100.0 | 1 (reference) |" in text and "| 2 of 3 |" in text
    assert "- mode `adaptive` could not be started: RuntimeError: no engine for adaptive" in text
    assert "- on: 1 responses differ from `off`'s; first differing character at [6]" in text
    assert "Nothing has been measured yet." in e1_report.render(tmp_path / "empty")


def test_the_gpu_backend_and_a_file_without_thinking(tmp_path, monkeypatch):
    log: list = []
    fake = _fake_runtime(log, fail=(), thinking=False)
    monkeypatch.setitem(sys.modules, "litert_lm", fake)
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"x")
    engines, failed, conversation_args, facts = lt._open_engines(model, ["off", "on"], "gpu", 4, tmp_path)
    assert list(engines) == ["off", "on"] and failed == {}
    assert all(entry[2] == fake.Backend.GPU() for entry in log)
    assert "thinking_config" not in conversation_args and facts["thinking_support"] is False


def test_off_is_required_and_a_missing_runtime_names_the_interpreter(tmp_path, monkeypatch):
    prompts = tmp_path / "dev.jsonl"
    prompts.write_text(json.dumps(PROMPTS[0]) + "\n")
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"x")
    with pytest.raises(SystemExit):
        lt.main(["--model", str(model), "--prompts", str(prompts), "--out", str(tmp_path / "o"), "--modes", "on"])
    monkeypatch.setitem(sys.modules, "litert_lm", None)                # importing it now raises ImportError
    with pytest.raises(SystemExit, match="LiteRT-LM's environment"):
        lt.main(["--model", str(model), "--prompts", str(prompts), "--out", str(tmp_path / "o")])
    log: list = []
    monkeypatch.setitem(sys.modules, "litert_lm", _fake_runtime(log, fail=("off", "on", "adaptive")))
    with pytest.raises(SystemExit, match="'off' engine could not be created"):
        lt.main(["--model", str(model), "--prompts", str(prompts), "--out", str(tmp_path / "o")])
