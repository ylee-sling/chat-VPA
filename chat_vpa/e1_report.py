# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_report.py -- the tables of E1's Step 0 as they stand (docs/e1.md, Section 3), from the summaries that the
measurements wrote.

    python -m chat_vpa.e1_report --out runs/e1-step0

It prints, for every measurement found under OUT, where it ran (device, precision, runtime, threads) and its
table: ``bridge/<run>`` (``e1_bridge.py``), ``ceiling/<tag>`` (``e1_ceiling.py``), ``decode/<tag>`` and
``decode/<tag>/profile`` (``e1_decode.py``), ``litert/<tag>`` (``e1_litert.py``), ``vllm/<tag>``
(``e1_vllm.py``). It computes nothing new and reads no rule: the reading rules of docs/e1.md, Section 4, are
applied by hand to these tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from chat_vpa import e1_bridge, e1_ceiling, e1_decode, e1_litert, e1_vllm

__all__ = ["render"]


def _where(env: dict[str, Any]) -> str:
    gpu = f" ({env['gpu']})" if "gpu" in env else ""
    threads = f", {env['cpu_threads']} CPU threads" if str(env.get("device", "")).startswith("cpu") else ""
    return f"{env['device']}{gpu}, {env['dtype']}, {env['runtime']}{threads}, torch {env['torch']}, {env['time']}"


def render(out: Path) -> str:
    lines = [f"# E1, Step 0: what has been measured under {out}", "",
             "Every number below is *measured* where its line says, or *predicted* from such measurements where its "
             "column says so (the ceiling and the alpha columns). Nothing here is read by a rule yet."]
    found = False
    for path in sorted(out.glob("bridge/*/summary.json")):
        s = json.loads(path.read_text())
        found = True
        lines += ["", f"## E0 bridge: {s['drafter']}", "",
                  f"{s['problems']} MBPP {s['split']} problems, {s['greedy_tokens']} greedy tokens, blocks of "
                  f"{s['block_size']}, {s['device']}, {s.get('dtype', 'bf16')}. Passes are counted, not timed; "
                  "autoregressive decoding is 1 token per backbone pass."]
        if "committed_stream_check" in s:
            c = s["committed_stream_check"]
            lines.append(f"Committed-stream check: {c['accepted']} of {c['drafts']} one-token drafts accepted on "
                         f"{c['problems']} problems.")
        if s["rows"]:
            lines += ["", e1_bridge._table(s["rows"])]
            slots = s["rows"][0]["aligned"]["slot_agreement"]
            lines += ["", f"Agreement with the greedy continuation per slot of a block drafted at training's "
                      f"boundaries (K = {s['rows'][0]['num_steps']}, {s['rows'][0]['order']}): "
                      + ", ".join("--" if v is None else f"{100 * v:.0f}%" for v in slots)]
    for path in sorted(out.glob("ceiling/*/summary.json")):
        s = json.loads(path.read_text())
        found = True
        lines += ["", f"## Ceiling: {path.parent.name}", "", _where(s["environment"]) + f"; {s['prompt_count']} prompts, "
                  f"mean {s['mean_prompt_tokens']:.0f} prompt tokens, {s['repeats']} readings per pass length.", "",
                  e1_ceiling._markdown(s["rows"], s["alphas"])]
    for path in [*sorted(out.glob("decode/*/summary.json")), *sorted(out.glob("decode/*/profile/summary.json"))]:
        s = json.loads(path.read_text())
        found = True
        name = path.parent.parent.name + " (profiled: not for speed-ups)" if s["profiled"] else path.parent.name
        prompts = next(iter(s["arms"].values()))["prompts"]
        lines += ["", f"## Decode: {name}", "", _where(s["environment"]) + f"; {prompts} prompts, up to "
                  f"{s['max_new_tokens']} new tokens, greedy.", "", e1_decode._markdown(s["arms"])]
        if s["profiled"]:
            lines += ["", "Share of the wall-clock inside the target's and the drafter's passes: "
                      + "; ".join(f"{arm}: {100 * a['profile']['target_share']:.0f}% and "
                                  f"{100 * a['profile']['drafter_share']:.0f}%" for arm, a in s["arms"].items())]
        differing = {arm: a["first_differences"] for arm, a in s["arms"].items() if a["first_differences"]}
        for arm, where in differing.items():
            lines.append(f"- {arm}: {len(where)} responses differ from `ar`'s; first differing token at "
                         f"{sorted(where.values())[:10]}{' ...' if len(where) > 10 else ''}")
    for path in sorted(out.glob("litert/*/summary.json")):
        s = json.loads(path.read_text())
        found = True
        env = s["environment"]
        threads = f", {env['cpu_threads']} of {env['cpus']} CPU threads" if env["backend"] == "cpu" else ""
        lines += ["", f"## LiteRT-LM: {path.parent.name}", "",
                  f"{env['runtime']}, backend {env['backend']}{threads}, {env['machine']}, {env['time']}; "
                  f"{s['model']['file']} ({s['model']['bytes'] / 2**30:.2f} GiB; drafter inside: "
                  f"{'yes' if s['model']['speculative_decoding_support'] else 'no'}); {s['prompts']['count']} prompts, "
                  f"up to {s['max_new_tokens']} new tokens, {s['sampler']}, lenience {s['lenience']}, thinking "
                  f"{s['thinking']}.", "", e1_litert._markdown(s["modes"])]
        for mode, reason in s["failed"].items():
            lines.append(f"- mode `{mode}` could not be started: {reason}")
        for mode, m in s["modes"].items():
            if m["first_differences"]:
                where = sorted(m["first_differences"].values())
                lines.append(f"- {mode}: {len(where)} responses differ from `off`'s; first differing character at "
                             f"{where[:10]}{' ...' if len(where) > 10 else ''}; the same number of tokens as `off` "
                             f"in {m['same_token_count']} of {m['prompts']}")
    for path in sorted(out.glob("vllm/*/summary.json")):
        s = json.loads(path.read_text())
        found = True
        startup = ", ".join(f"{tag} {seconds:.0f} s" for tag, seconds in s["startup_seconds"].items())
        lines += ["", f"## vLLM: {path.parent.name}", "",
                  _where(s["environment"]) + f"; {s['prompts']['count']} prompts, up to {s['max_new_tokens']} new "
                  f"tokens, greedy, one request at a time; one engine per arm, run in the order "
                  f"{', '.join(s['order'])} (engine start-up: {startup}).", "", e1_vllm._markdown(s["arms"])]
        drift = s["arms"].get("off-2")
        if drift is None:
            lines.append("- No second reference run (off-2): the drift of the machine over the run is not known.")
        else:
            low, high = drift["decode_speedup_interval"]
            verdict = ("contains 1: the machine held its speed" if low <= 1 <= high else
                       "excludes 1: the machine's speed moved during the run, and R2 is not read for this table "
                       "(docs/e1.md, Section 3.4)")
            lines.append(f"- Drift: off-2's decode speed against off-1's is {drift['decode_speedup']:.3f} "
                         f"[{low:.3f}, {high:.3f}], which {verdict}; off-2 has the same tokens as off-1 on "
                         f"{drift['same_tokens']} of {drift['prompts']} prompts.")
        for tag, m in s["arms"].items():
            if m["acceptance_histogram"]:
                lines.append(f"- {tag}: verification steps by accepted draft tokens (0, 1, ...): "
                             f"{m['acceptance_histogram']}")
            if m["first_differences"]:
                where = sorted(m["first_differences"].values())
                lines.append(f"- {tag}: {len(where)} responses differ from off-1's; first differing token at "
                             f"{where[:10]}{' ...' if len(where) > 10 else ''}")
    if not found:
        lines += ["", "Nothing has been measured yet."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E1 Step 0: the tables so far", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    text = render(args.out)
    if args.out.is_dir():
        (args.out / "step0.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
