# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""E1's Step 0 measurements (docs/e1.md, Section 3) on a tiny random Gemma 4 with a tiny random drafter, on the
CPU: the development prompts are the first rows that qualify; speculative decoding along a greedy continuation is
counted as a hand count says; the block sampler as a drafter is accepted exactly where an explicit loop accepts
it, and the committed stream as its own drafter always; a pass over n tokens chooses what n passes choose; the
decode command's autoregressive arm is the model's own greedy ``generate``, every verified arm reproduces it, and
its summary is the arithmetic of its samples. Nothing is downloaded."""

from __future__ import annotations

import json
import math

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4AssistantForCausalLM"):          # pragma: no cover
    pytest.skip("this transformers release has no Gemma 4 drafter", allow_module_level=True)
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

from test_stage0 import VOCAB, WINDOW, randomize_adapters  # noqa: E402
from test_stage0_cli import write_benchmarks, write_checkpoint  # noqa: E402

from chat_vpa import e1_bridge as br  # noqa: E402
from chat_vpa import e1_ceiling as ce  # noqa: E402
from chat_vpa import e1_decode as de  # noqa: E402
from chat_vpa import e1_preflight as pf  # noqa: E402
from chat_vpa import e1_prompts as pr  # noqa: E402
from chat_vpa.backbone import RealizerSpec, Stage0Realizer, load_gemma4  # noqa: E402
from chat_vpa.block_dfm import BlockDFM  # noqa: E402
from chat_vpa.block_sampler import BlockSamplerConfig, sample_blocks  # noqa: E402
from chat_vpa.e1_common import chat_ids, load_target, median, read_prompts  # noqa: E402
from chat_vpa.runtime import save_checkpoint  # noqa: E402

CHAT_TEMPLATE = ("{{ bos_token }}{% for m in messages %}<{{ m['role'] }}>\n{{ m['content'] }}\n{% endfor %}"
                 "{% if add_generation_prompt %}<model>\n{% endif %}")


# ---------------------------------------------------------------------------------------------
# The development prompts
# ---------------------------------------------------------------------------------------------

def test_the_development_prompts_are_the_first_rows_that_qualify(tmp_path):
    files = {"b-0001.parquet": [("b0", "second file, first row"), ("b1", "x" * 31)],
             "a-0000.parquet": [("a0", "   "), ("a1", "the first that counts"), ("a2", None), ("a3", "y" * 30),
                                ("a4", "another")]}
    for name, rows in files.items():
        pq.write_table(pa.table({"prompt": [r[1] for r in rows], "prompt_id": [r[0] for r in rows],
                                 "messages": [[] for _ in rows]}), tmp_path / name)
    out = tmp_path / "prompts" / "dev.jsonl"
    paths = [str(tmp_path / name) for name in files]                   # given in the wrong order on purpose
    assert pr.main(["--parquet", *paths, "--count", "4", "--max-chars", "30", "--out", str(out)]) == 0
    # By hand: files in name order (a, then b); blank, missing and 31-character prompts are skipped.
    expected = [("a1", "the first that counts"), ("a3", "y" * 30), ("a4", "another"), ("b0", "second file, first row")]
    assert read_prompts(out) == [{"id": f"ultrachat/test_sft/{i}", "prompt": text} for i, text in expected]
    assert read_prompts(out, limit=2) == read_prompts(out)[:2]
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["count"] == 4 and sorted(meta["sources"]) == sorted(files)
    with pytest.raises(SystemExit, match="only 4 of 5"):
        pr.main(["--parquet", *paths, "--count", "5", "--max-chars", "30", "--out", str(out)])


# ---------------------------------------------------------------------------------------------
# Speculative decoding along a greedy continuation: the accounting
# ---------------------------------------------------------------------------------------------

def _rule_drafter(prompts, targets, block_size, calls=None):
    """A drafter that is right for the first (s mod 3) tokens after a prefix that ends at position s of its
    target, and wrong from there on."""
    known = {tuple(p): y for p, y in zip(prompts, targets, strict=True)}

    def draft(prefixes):
        if calls is not None:
            calls.append(len(prefixes))
        out = []
        for prefix in prefixes:
            prompt = next(p for p in known if tuple(prefix[:len(p)]) == p)
            s = len(prefix) - len(prompt)
            y = known[prompt]
            assert list(prefix[len(prompt):]) == y[:s]                  # only ever asked about prefixes of y
            right = y[s:s + min(s % 3, block_size)]
            out.append([*right, *[-1] * (block_size - len(right))])
        return out

    return draft


def test_speculative_decoding_is_counted_as_by_hand():
    assert br.accepted_prefix([], [1, 2]) == 0 and br.accepted_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert br.accepted_prefix([1, 2, 3], [1, 2]) == 2 and br.accepted_prefix([5], [1]) == 0
    prompts = [[1, 2], [1, 3, 9]]
    targets = [list(range(100, 110)), list(range(200, 205))]
    # By hand, a = s mod 3 capped by what is left, gained = min(a + 1, left):
    #   10 tokens: s = 0, 1, 3, 4, 6, 7, 9 -> a = 0, 1, 0, 1, 0, 1, 0 and gained 1, 2, 1, 2, 1, 2, 1
    #    5 tokens: s = 0, 1, 3, 4       -> a = 0, 1, 0, 1 and gained 1, 2, 1, 1 (one token was left at s = 4)
    for batch in (1, 2, 8):
        calls: list[int] = []
        traces = br.simulate(prompts, targets, _rule_drafter(prompts, targets, 4, calls), batch)
        assert [t.accepted for t in traces] == [[0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1]]
        assert [t.gained for t in traces] == [[1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 1]]
        assert [t.tokens for t in traces] == [10, 5] and max(calls) <= batch
    summary = br.summarize(traces, num_steps=2, block_size=4, resamples=200)
    assert summary["cycles"] == 11 and summary["tokens"] == 15 and summary["passes_per_cycle"] == 3
    assert summary["accepted_per_cycle"] == pytest.approx(5 / 11)
    assert summary["tokens_per_cycle"] == pytest.approx(15 / 11)
    assert summary["tokens_per_backbone_pass"] == pytest.approx(15 / 33)          # AR would be 1
    assert summary["accepted_histogram"] == [6, 5, 0, 0, 0]
    assert summary["cycles_with_nothing_accepted"] == pytest.approx(6 / 11)
    low, high = summary["tokens_per_backbone_pass_interval"]
    assert 10 / 21 - 1e-12 <= high and low >= 5 / 12 - 1e-12                       # between the two problems' own
    # At training's boundaries s = 0, 4, 8: a = 0, 1, 2 for the first target and 0, 1 for the second; per slot,
    # drafts that hit / drafts that have a target token there.
    per_block, hits, counts = br.aligned(prompts, targets, _rule_drafter(prompts, targets, 4), 4, 3)
    assert per_block == [[0, 1, 2], [0, 1]] and hits == [3, 1, 0, 0] and counts == [5, 4, 3, 3]
    with pytest.raises(ValueError, match="no cycle"):
        br.summarize([br.Trace()], 2, 4)


# ---------------------------------------------------------------------------------------------
# A tiny instruction-tuned model, its drafter, its prompts, and an E0 checkpoint
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("e1")
    write_checkpoint(root / "model")
    tokenizer = transformers.AutoTokenizer.from_pretrained(root / "model")
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(root / "model")
    torch.manual_seed(1)
    text = transformers.Gemma4TextConfig(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=1, head_dim=16, global_head_dim=32, sliding_window=WINDOW,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        final_logit_softcapping=None, vocab_size_per_layer_input=0, hidden_size_per_layer_input=0,
        num_kv_shared_layers=4, use_double_wide_mlp=False, max_position_embeddings=1024)
    drafter = transformers.Gemma4AssistantConfig(text_config=text, backbone_hidden_size=64)
    transformers.Gemma4AssistantForCausalLM(drafter).save_pretrained(root / "assistant")
    with open(root / "prompts.jsonl", "w", encoding="utf-8") as handle:
        for i in range(4):
            handle.write(json.dumps({"id": f"p{i}", "prompt": f"Count to {i + 3}, then count to {i + 3} again."}) + "\n")
    write_benchmarks(root / "eval")
    # An E0 checkpoint: the tiny model's realizer with random adapters, in the trainer's format.
    model, _ = load_gemma4(str(root / "model"), dtype=torch.float32)
    realizer = Stage0Realizer.from_pretrained_model(model, RealizerSpec(mask_id=4, lora_rank=4, lora_alpha=4.0,
                                                                        time_dim=32))
    torch.manual_seed(2)
    randomize_adapters(realizer)
    spec = {**realizer.spec.to_dict(), "lora_targets": list(realizer.spec.lora_targets)}
    save_checkpoint(root / "e0.pt", {"stage": "e0", "step": 7, "spec": spec, "trainable": realizer.trainable_state(),
                                     "config": {"arm": "rollouts", "block_size": 4, "schedule": "cosine",
                                                "num_registers": 0}})
    return root


def test_the_chat_prompt_is_the_models_template(tiny):
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny / "model")
    ids = chat_ids(tokenizer, "Hi")
    assert tokenizer.decode(ids) == "<bos><user>\nHi\n<model>\n" and ids[0] == tokenizer.bos_token_id
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="no chat template"):
        chat_ids(tokenizer, "Hi")
    assert median([3, 1, 2]) == 2 and median([4, 1, 3, 2]) == 2.5


# ---------------------------------------------------------------------------------------------
# The E0 bridge
# ---------------------------------------------------------------------------------------------

def test_an_e0_checkpoint_as_a_drafter_is_accepted_where_an_explicit_loop_accepts_it(tiny, capsys):
    out = tiny / "bridge"
    assert br.main(["--model", str(tiny / "model"), "--checkpoint", str(tiny / "e0.pt"), "--eval-dir",
                    str(tiny / "eval"), "--out", str(out), "--k-steps", "1", "2", "--orders", "prefix",
                    "--batch-size", "2", "--max-new-tokens", "24", "--check", "2", "--dtype", "fp32", "--device",
                    "cpu"]) == 0
    table = capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text())
    reference = [json.loads(line) for line in (out / "reference.jsonl").read_text().splitlines()]
    assert [r["task_id"] for r in reference] == ["MBPP/511", "MBPP/512"] and summary["problems"] == 2
    assert summary["greedy_tokens"] == sum(len(r["greedy_ids"]) for r in reference) > 0
    # The committed stream as its own drafter (B = K = 1, causal, adapters off): each one-token draft is the
    # backbone's next token, so each is accepted, and a cycle yields 2 tokens (or the last one). This holds in
    # float32; with this model's weights cast to bfloat16, generate and the realizer's path differ at 2 of 48
    # positions, which is what the check is for.
    check = summary["committed_stream_check"]
    assert check["share"] == 1.0 and check["drafts"] == sum(math.ceil(len(r["greedy_ids"]) / 2) for r in reference)

    # The reference: the same drafts one prefix at a time, with an explicit loop.
    model, hf_tokenizer = load_gemma4(str(tiny / "model"), dtype=torch.float32)
    ckpt = torch.load(tiny / "e0.pt", weights_only=True)
    realizer = Stage0Realizer.from_pretrained_model(model, RealizerSpec.from_dict(ckpt["spec"]))
    realizer.load_trainable_state(ckpt["trainable"])
    block_model = BlockDFM(realizer).eval()
    device = torch.device("cpu")
    for row in summary["rows"]:
        cfg = BlockSamplerConfig(4, row["num_steps"], "cosine", "prefix", 0.0, 0.0)
        accepted, gained = [], []
        for r in reference:
            y, s = r["greedy_ids"], 0
            while s < len(y):
                prefix = [*r["prompt_ids"], *y[:s]]
                with torch.no_grad():
                    result = sample_blocks(block_model, block_model.prefill([prefix], device),
                                           torch.tensor([prefix[-1]]), cfg, 1, [0, 2], None, stop_id=None)
                draft, a = result.tokens[0], 0
                assert len(draft) == 4
                while a < 4 and s + a < len(y) and draft[a] == y[s + a]:
                    a += 1
                accepted.append(a)
                gained.append(min(a + 1, len(y) - s))
                s += gained[-1]
        spec = row["speculative"]
        assert spec["cycles"] == len(accepted) and spec["tokens"] == sum(gained) == summary["greedy_tokens"]
        assert spec["accepted_per_cycle"] == pytest.approx(sum(accepted) / len(accepted))
        assert spec["tokens_per_backbone_pass"] == pytest.approx(sum(gained) / (len(accepted) * (row["num_steps"] + 1)))
        assert spec["accepted_histogram"] == [accepted.count(a) for a in range(5)]
        assert len(row["aligned"]["slot_agreement"]) == 4
    assert "| rollouts B4 (step 7) | 1 | prefix |" in table and "| rollouts B4 (step 7) | 2 | prefix |" in table
    torch.save({"stage": "stage0"}, tiny / "other.pt")
    with pytest.raises(SystemExit, match="not an E0 checkpoint"):
        br.main(["--model", str(tiny / "model"), "--checkpoint", str(tiny / "other.pt"), "--eval-dir",
                 str(tiny / "eval"), "--out", str(out), "--device", "cpu"])


# ---------------------------------------------------------------------------------------------
# The ceiling
# ---------------------------------------------------------------------------------------------

def test_the_ceiling_is_the_arithmetic_of_the_pass_times():
    assert ce.expected_tokens(0.8, 3) == pytest.approx(1 + 0.8 + 0.64) and ce.expected_tokens(1.0, 5) == 5.0
    assert ce.expected_tokens(0.0, 7) == 1.0
    rows = ce.ceiling_table({4: 0.016, 1: 0.010, 2: 0.011}, alphas=(0.5, 0.9))
    assert [r["n"] for r in rows] == [1, 2, 4]
    assert [r["relative_cost"] for r in rows] == pytest.approx([1.0, 1.1, 1.6])
    assert [r["ceiling"] for r in rows] == pytest.approx([1.0, 2 / 1.1, 4 / 1.6])
    # n = 4, alpha = 0.5: 1 + 0.5 + 0.25 + 0.125 = 1.875 tokens for 1.6 passes' time.
    assert rows[2]["predicted"]["0.5"] == pytest.approx(1.875 / 1.6)
    assert rows[2]["predicted"]["0.9"] == pytest.approx((1 + 0.9 + 0.81 + 0.729) / 1.6)
    assert rows[0]["predicted"] == {"0.5": 1.0, "0.9": 1.0}                  # one position: the target's own token
    with pytest.raises(ValueError, match="include n = 1"):
        ce.ceiling_table({2: 0.01})


def test_a_pass_over_n_tokens_chooses_what_n_passes_choose(tiny, capsys):
    device = torch.device("cpu")
    model, tokenizer = load_target(tiny / "model", "fp32", device)
    ids = chat_ids(tokenizer, "Count to three.")
    result = ce.measure_prompt(model, ids, [1, 3, 8], repeats=2, device=device)
    with torch.no_grad():                                                   # the model's own greedy decoding
        own = model.generate(torch.tensor([ids]), max_new_tokens=9, do_sample=False, temperature=None, top_p=None,
                             top_k=None, eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
    assert result["reference_tokens"] == own[1:9] and result["first_token"] == own[0]
    assert result["mismatches"] == {1: 0, 3: 0, 8: 0} and result["prompt_tokens"] == len(ids)
    assert {n: len(v) for n, v in result["seconds"].items()} == {1: 2, 3: 2, 8: 2}
    assert len(result["sequential_seconds"]) == 8 and min(result["sequential_seconds"]) > 0
    out = tiny / "ceiling"
    assert ce.main(["--model", str(tiny / "model"), "--prompts", str(tiny / "prompts.jsonl"), "--lengths", "1", "2", "4",
                    "--limit", "3", "--repeats", "2", "--warmup", "1", "--dtype", "fp32", "--device", "cpu", "--out",
                    str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert [r["n"] for r in summary["rows"]] == [1, 2, 4] and summary["prompt_count"] == 3
    assert [(r["mismatches"], r["positions"]) for r in summary["rows"]] == [(0, 3), (0, 6), (0, 12)]
    pooled = median([s for p in summary["per_prompt"] for s in p["seconds"]["4"]])
    assert summary["rows"][2]["seconds"] == pooled
    assert summary["rows"][2]["ceiling"] == pytest.approx(4 * summary["rows"][0]["seconds"] / pooled)
    assert summary["environment"]["dtype"] == "fp32" and "| n | t(n), ms |" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="include 1"):
        ce.main(["--model", str(tiny / "model"), "--prompts", str(tiny / "prompts.jsonl"), "--lengths", "2", "--out",
                 str(out)])


# ---------------------------------------------------------------------------------------------
# Decoding with and without a verified drafter
# ---------------------------------------------------------------------------------------------

def test_arms_and_differences():
    assert de.parse_arm("ar") == de.Arm("ar") and de.parse_arm("mtp:4") == de.Arm("mtp", 4)
    assert de.parse_arm("lookup:10").name == "lookup-10" and de.Arm("ar").name == "ar"
    for bad in ("mtp", "mtp:0", "ar:2", "draft:3", "lookup:x"):
        with pytest.raises(ValueError, match="an arm is"):
            de.parse_arm(bad)
    assert de.first_difference([1, 2, 3], [1, 2, 3]) is None and de.first_difference([1, 2, 3], [1, 9, 3]) == 1
    assert de.first_difference([1, 2], [1, 2, 3]) == 2 and de.first_difference([], []) is None


def test_the_summary_is_the_arithmetic_of_the_records():
    def record(prompt, arm, tokens, seconds, first, passes, first_tokens=1, drafter=0):
        return {"id": prompt, "arm": arm, "tokens": tokens, "seconds": seconds, "first_seconds": first,
                "first_tokens": first_tokens, "target_passes": passes, "drafter_passes": drafter}

    a, b = list(range(11)), list(range(21))
    records = [record("p", "ar", a, 1.5, 0.5, 11), record("p", "mtp-4", a, 0.9, 0.5, 5, drafter=16),
               record("q", "ar", b, 2.5, 0.5, 21), record("q", "mtp-4", [*b[:7], 99, *b[8:]], 1.3, 0.5, 9, drafter=32)]
    out = de.summarize(records, ["ar", "mtp-4"], resamples=300)
    ar, mtp = out["ar"], out["mtp-4"]
    # By hand. AR: 30 tokens after the first in 3.0 s, 32 tokens in 4.0 s, 30 passes after the 2 prefills.
    assert ar["decode_tokens_per_second"] == pytest.approx(10.0) and ar["end_to_end_tokens_per_second"] == pytest.approx(8.0)
    assert ar["tokens_per_cycle"] == pytest.approx(1.0) and ar["exact"] == 2 and "decode_speedup" not in ar
    # MTP: 30 tokens in 0.4 + 0.8 = 1.2 s -> 25 tok/s, 2.5 times AR; 32 tokens in 2.2 s; 12 cycles for 30 tokens.
    assert mtp["decode_tokens_per_second"] == pytest.approx(25.0) and mtp["decode_speedup"] == pytest.approx(2.5)
    assert mtp["end_to_end_speedup"] == pytest.approx((32 / 2.2) / 8.0)
    assert mtp["tokens_per_cycle"] == pytest.approx(30 / 12) and mtp["drafter_passes"] == 48
    assert mtp["exact"] == 1 and mtp["exact_share"] == 0.5 and mtp["first_differences"] == {"q": 7}
    low, high = mtp["decode_speedup_interval"]                      # between the two prompts' own speed-ups
    assert low == pytest.approx(2.5) and high == pytest.approx(2.5)  # 10 / 0.4 over 10 and 20 / 0.8 over 10: both 2.5
    assert mtp["median_seconds_to_first_tokens"] == 0.5
    with pytest.raises(ValueError, match="does not cover"):
        de.summarize(records[:3], ["ar", "mtp-4"])


def test_every_verified_arm_reproduces_the_models_own_greedy_decoding(tiny, capsys):
    out = tiny / "decode"
    args = ["--model", str(tiny / "model"), "--assistant", str(tiny / "assistant"), "--prompts",
            str(tiny / "prompts.jsonl"), "--arms", "ar", "mtp:3", "lookup:4", "--max-new-tokens", "20", "--dtype", "fp32",
            "--device", "cpu", "--warmup", "1", "--out", str(out)]
    assert de.main(args) == 0
    table = capsys.readouterr().out
    samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
    summary = json.loads((out / "summary.json").read_text())
    assert [s["arm"] for s in samples] == ["ar", "mtp-3", "lookup-4"] * 4

    device = torch.device("cpu")
    model, tokenizer = load_target(tiny / "model", "fp32", device)
    prompts = read_prompts(tiny / "prompts.jsonl")
    by_arm = {arm: [s for s in samples if s["arm"] == arm] for arm in ("ar", "mtp-3", "lookup-4")}
    for row, ar in zip(prompts, by_arm["ar"], strict=True):
        ids = chat_ids(tokenizer, row["prompt"])
        with torch.no_grad():                                               # the reference: generate itself
            own = model.generate(torch.tensor([ids]), max_new_tokens=20, do_sample=False, temperature=None,
                                 top_p=None, top_k=None)[0, len(ids):].tolist()
        assert ar["tokens"] == own and ar["prompt_tokens"] == len(ids) and ar["text"] == tokenizer.decode(own)
        assert ar["target_passes"] == len(own) and ar["drafter_passes"] == 0   # one pass per token
        assert ar["first_tokens"] == 1 and 0 < ar["first_seconds"] < ar["seconds"]
    for arm in ("mtp-3", "lookup-4"):
        for sample, ar in zip(by_arm[arm], by_arm["ar"], strict=True):
            assert sample["tokens"] == ar["tokens"]                         # float32: verification is exact
            assert 1 <= sample["target_passes"] <= len(sample["tokens"])
    for sample in by_arm["mtp-3"]:                                          # at most 3 drafts per cycle
        assert 0 < sample["drafter_passes"] <= 3 * (sample["target_passes"] - 1)

    # The summary, recomputed from the samples with loops.
    for arm, rows in by_arm.items():
        s = summary["arms"][arm]
        tokens = sum(len(r["tokens"]) for r in rows)
        seconds = sum(r["seconds"] for r in rows)
        late_tokens = sum(len(r["tokens"]) - r["first_tokens"] for r in rows)
        late_seconds = sum(r["seconds"] - r["first_seconds"] for r in rows)
        assert s["new_tokens"] == tokens and s["exact"] == 4 and s["first_differences"] == {}
        assert s["end_to_end_tokens_per_second"] == pytest.approx(tokens / seconds)
        assert s["decode_tokens_per_second"] == pytest.approx(late_tokens / late_seconds)
        assert s["tokens_per_cycle"] == pytest.approx(late_tokens / (sum(r["target_passes"] for r in rows) - 4))
        if arm != "ar":
            base = summary["arms"]["ar"]["decode_tokens_per_second"]
            assert s["decode_speedup"] == pytest.approx(s["decode_tokens_per_second"] / base)
            assert s["decode_speedup_interval"][0] <= s["decode_speedup"] <= s["decode_speedup_interval"][1]
    assert summary["environment"]["runtime"] == "PyTorch, eager" and summary["profiled"] is False
    assert summary["chat_template_of_the_first_prompt"].startswith("<bos><user>\n")
    assert "| ar |" in table and "| mtp-3 |" in table and "1 (reference)" in table

    # A profiled run is written beside the plain one and says where the time went.
    assert de.main([*args, "--profile", "--limit", "2"]) == 0
    profiled = json.loads((out / "profile" / "summary.json").read_text())
    assert profiled["profiled"] is True and json.loads((out / "summary.json").read_text()) == summary
    share = profiled["arms"]["mtp-3"]["profile"]
    assert 0 < share["drafter_share"] < 1 and 0 < share["target_share"] < 1
    assert profiled["arms"]["ar"]["profile"]["drafter_share"] == 0.0
    with pytest.raises(SystemExit, match="must start with 'ar'"):
        de.main([*args[:7], "mtp:3", "ar", *args[10:]])
    with pytest.raises(SystemExit, match="need --assistant"):
        de.main(["--model", str(tiny / "model"), "--prompts", str(tiny / "prompts.jsonl"), "--arms", "ar", "mtp:2",
                 "--out", str(out), "--device", "cpu", "--dtype", "fp32"])


def test_the_preflight_reads_both_checkpoints_and_decodes(tiny, capsys):
    args = ["--model", str(tiny / "model"), "--assistant", str(tiny / "assistant"), "--prompts",
            str(tiny / "prompts.jsonl"), "--dtype", "fp32", "--smoke-prompts", "2", "--smoke-tokens", "12", "--device", "cpu"]
    assert pf.main(args) == 0
    text = capsys.readouterr().out
    assert "== preflight passed" in text and "UNEXPECTED" not in text
    assert "Gemma4AssistantForCausalLM: 4 layers (sliding_attention, sliding_attention, sliding_attention, " \
           "full_attention), width 32, for a target of width 64" in text
    assert "one user turn renders as '<bos><user>\\nHello.\\n<model>\\n'" in text
    assert text.count("with the drafter the same tokens") == 2             # float32: verification is exact
    # Parameter counts against explicit sums: tables are the embedding modules' weights, tied weights count once.
    drafter = transformers.Gemma4AssistantForCausalLM.from_pretrained(tiny / "assistant")
    counts = pf.parameter_counts(drafter)
    unique = {id(p): p.numel() for p in drafter.parameters()}
    tables = {id(m.weight): m.weight.numel() for m in drafter.modules() if isinstance(m, torch.nn.Embedding)}
    assert counts == {"total": sum(unique.values()), "embedding_tables": sum(tables.values()),
                      "rest": sum(unique.values()) - sum(tables.values())}
    assert counts["embedding_tables"] == VOCAB * 32 and f"parameters: {counts['total'] / 1e6:.1f}M" in text
    # A chat template that switches thinking on is flagged.
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny / "model")
    other = tiny / "thinking"
    transformers.Gemma4ForConditionalGeneration.from_pretrained(tiny / "model").save_pretrained(other)
    tokenizer.chat_template = "{{ bos_token }}<|think|>" + CHAT_TEMPLATE[len("{{ bos_token }}"):]
    tokenizer.save_pretrained(other)
    assert pf.main(["--model", str(other), *args[2:]]) == 1
    assert "UNEXPECTED: the rendered prompt mentions thinking" in capsys.readouterr().out


def test_every_command_resolves_its_device_from_auto_by_default(monkeypatch, tmp_path):
    """Without --device each command asks chat_vpa.device for "auto" (cuda, then mps, then cpu), as every other
    command of the repository does; None reached torch.device on the server and stopped the preflight."""
    import chat_vpa.device
    from chat_vpa import e1_bridge, e1_ceiling, e1_decode, e1_preflight

    asked: list[object] = []

    class Resolved(Exception):
        pass

    def record(name="auto"):
        asked.append(name)
        raise Resolved

    monkeypatch.setattr(chat_vpa.device, "resolve_device", record)
    some = str(tmp_path / "unused")
    commands = [
        (e1_bridge, ["--model", some, "--checkpoint", some, "--eval-dir", some, "--out", some]),
        (e1_ceiling, ["--model", some, "--prompts", some, "--out", some]),
        (e1_decode, ["--model", some, "--prompts", some, "--out", some]),
        (e1_preflight, ["--model", some, "--assistant", some, "--prompts", some]),
    ]
    for module, argv in commands:
        with pytest.raises(Resolved):
            module.main(argv)
    assert asked == ["auto"] * len(commands)
    monkeypatch.undo()
    assert isinstance(chat_vpa.device.resolve_device("auto"), torch.device)
