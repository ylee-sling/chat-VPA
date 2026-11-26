# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
e1_p2.py -- P2 of E1's Step 1 (docs/e1.md, Section 11.7): does LiteRT-LM's GPU backend repeat itself at
lenience 0, and is the text its speculation changes a precision effect or a defect?

In Step 0 (Section 10.4, flag 1) the GPU backend's ``on`` and ``adaptive`` modes kept ``off``'s text on 1 of 50
prompts, while its CPU backend, run by the same script, kept it on 50 of 50. Two commands, each in the
environment it needs:

    ASSETS/venvs/litert/bin/python -m chat_vpa.e1_p2 run --mode off --tag off-a \\
        --model ASSETS/litert/gemma-4-E2B-it.litertlm --prompts ASSETS/prompts/dev.jsonl --limit 10 \\
        --backend gpu --out runs/e1-p2
    python -m chat_vpa.e1_p2 read --out runs/e1-p2 --model ASSETS/models/gemma-4-E2B-it \\
        --prompts ASSETS/prompts/dev.jsonl --step0 runs/e1-step0/litert/gpu/samples.jsonl

``run`` decodes the first ``--limit`` prompts with one engine of one mode in this process (Step 0 created its
three engines in one process), after ``--warmup`` prompts that are thrown away, with Step 0's settings (lenience 0,
top-k 1, thinking off where the file has it); it writes ``<out>/<tag>/samples.jsonl`` and ``run.json``. It runs in
LiteRT-LM's environment and needs no torch.

``read`` compares the texts of the runs and, for every pair of responses that part, scores the first token at
which they part with the released checkpoint in float32; it then applies the rules registered in Section 11.7:

    P2a  repeatability     off-a against off-b, on-a against on-b: the same text on how many prompts
    P2b  speculation       on-a against off-a, on-b against off-b, each in a process of its own
         (beside them: every run against Step 0's GPU run of the same mode on the same prompts)
    P2c  the parting token for every pair of P2b and of Step 0's GPU row (on and adaptive against off) whose texts
         differ: both texts are tokenised again by the target's tokenizer, and where the two token sequences
         first differ the target in float32 scores the prompt (its chat template) with the shared tokens. Recorded:
         the two tokens, their ranks (1 is the target's argmax) and their log-probabilities. The pair is a
         "near tie" if the two tokens are the target's top two and their log-probabilities lie within 1 nat;
         "far" if the speculating side's token ranks below 5 while the plain side's ranks among the top 2; "other"
         otherwise. A pair in which one token sequence ends where the other goes on is counted apart, unscored.
         Identical pairs (the same prompt and the same two texts) are scored once.

    verdict  a precision effect: at least 90% of the scored pairs are near ties and none is far; a defect of the
             GPU backend's speculation: at least 10% are far; otherwise undecided. If P2a fails, the verdict also
             says that the backend does not repeat itself, so that what speculation changes is not isolated.

What this is not. The float32 checkpoint is not LiteRT-LM's quantised build: its ranks say whether the runtime's
two choices were close for the model both approximate, not what the runtime computed. A text tokenised again
need not give the runtime's own tokens; the parting token is where the two re-tokenised texts part.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from chat_vpa.e1_common import file_sha256, read_prompts, write_json
from chat_vpa.e1_litert import first_text_difference

__all__ = ["FAR_RANK", "NEAR_TIE_NATS", "TOP", "classify", "compare", "first_token_difference", "parting",
           "score_parting", "verdict"]

NEAR_TIE_NATS = 1.0          # the two tokens' log-probabilities lie within this many nats
TOP = 2                      # ... and both are among the target's top TOP tokens
FAR_RANK = 5                 # the speculating side's token ranks below this while the plain side's is in the top TOP


def first_token_difference(a: Sequence[int], b: Sequence[int]) -> int | None:
    """The first index at which two token sequences differ: None if they are equal, the shorter length if one is
    a prefix of the other."""
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def classify(plain_rank: int, speculating_rank: int, gap: float) -> str:
    """The class of a parting token (the rule of the module docstring); ranks count from 1, the gap is in nats."""
    if plain_rank <= TOP and speculating_rank <= TOP and gap <= NEAR_TIE_NATS:
        return "near tie"
    if speculating_rank > FAR_RANK and plain_rank <= TOP:
        return "far"
    return "other"


def verdict(classes: Sequence[str], repeats: bool | None) -> str:
    """The reading of P2c over the scored pairs, with P2a's outcome (None: P2a was not run)."""
    if not classes:
        core = "nothing to read: no pair of responses parts at a token"
    else:
        near, far = classes.count("near tie") / len(classes), classes.count("far") / len(classes)
        if near >= 0.9 and classes.count("far") == 0:
            core = "consistent with a precision effect"
        elif far >= 0.1:
            core = "a defect of the GPU backend's speculation"
        else:
            core = "undecided"
    if repeats is False:
        core += ("; the backend does not repeat itself (P2a), so what speculation changes is not isolated from "
                 "what changes between two runs of one mode")
    return core


def compare(reference: dict[str, dict[str, Any]], other: dict[str, dict[str, Any]], ids: Sequence[str]
            ) -> dict[str, Any]:
    """The same text on how many of ``ids``, and the first differing character of the others."""
    missing = [i for i in ids if i not in reference or i not in other]
    if missing:
        raise ValueError(f"prompts missing from a run: {missing[:5]}")
    firsts = {i: first_text_difference(reference[i]["text"], other[i]["text"]) for i in ids}
    return {"prompts": len(ids), "same_text": sum(d is None for d in firsts.values()),
            "first_differences": {i: d for i, d in firsts.items() if d is not None}}


def parting(tokenizer: Any, plain_text: str, speculating_text: str) -> dict[str, Any]:
    """Where the two texts, tokenised again, first differ: ``kind`` is "token" (with the shared tokens and the two
    tokens), "ends" (one sequence is a prefix of the other) or "same tokens"."""
    a = list(tokenizer(plain_text, add_special_tokens=False)["input_ids"])
    b = list(tokenizer(speculating_text, add_special_tokens=False)["input_ids"])
    index = first_token_difference(a, b)
    if index is None:
        return {"kind": "same tokens"}
    if index >= len(a) or index >= len(b):
        return {"kind": "ends", "index": index}
    return {"kind": "token", "index": index, "shared": a[:index], "plain_token": a[index],
            "speculating_token": b[index]}


def score_parting(model: Any, prompt_ids: Sequence[int], shared: Sequence[int], plain_token: int,
                  speculating_token: int, device: Any) -> dict[str, Any]:
    """The target's ranks and log-probabilities (float64 from the model's logits) of the two parting tokens after
    the prompt and the shared tokens."""
    import torch

    ids = torch.tensor([list(prompt_ids) + list(shared)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
    logp = torch.log_softmax(logits.double(), dim=-1)

    def rank(token: int) -> int:
        return int((logp > logp[token]).sum().item()) + 1

    plain, speculating = float(logp[plain_token]), float(logp[speculating_token])
    return {"plain_rank": rank(plain_token), "speculating_rank": rank(speculating_token), "plain_logp": plain,
            "speculating_logp": speculating, "gap": abs(plain - speculating), "argmax": int(torch.argmax(logp))}


# ---------------------------------------------------------------------------------------------
# run: one engine of one mode in this process (LiteRT-LM's environment)
# ---------------------------------------------------------------------------------------------

def _run(args: argparse.Namespace) -> int:
    from chat_vpa import e1_litert as lt

    prompts = read_prompts(args.prompts, args.limit)
    cache_dir = args.cache_dir or args.model.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    engines, failed, conversation_args, facts = lt.open_engines(args.model, [args.mode], args.backend, args.threads,
                                                                cache_dir)
    if args.mode not in engines:
        raise SystemExit(f"the {args.mode} engine could not be created on the {args.backend} backend: "
                         f"{failed.get(args.mode)}")
    started = time.perf_counter()
    try:
        records = lt.run(engines, prompts, args.max_new_tokens, args.warmup, conversation_args,
                         lambda text: print(f"{args.tag}: {text}", flush=True))
    finally:
        for engine in engines.values():
            engine.close()
    out = args.out / args.tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "samples.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({**record, "tag": args.tag}, ensure_ascii=False) + "\n")
    write_json(out / "run.json", {
        "what": "P2: one LiteRT-LM engine of one mode in its own process (docs/e1.md, Section 11.7), measured",
        "tag": args.tag, "mode": args.mode,
        "environment": {"runtime": f"LiteRT-LM {lt._runtime_version()}", "backend": args.backend,
                        "cpu_threads": args.threads, "cpus": os.cpu_count(), "machine": platform.machine(),
                        "vulkan_driver_files": os.environ.get("VK_DRIVER_FILES"), "pid": os.getpid(),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")},
        "model": {"file": args.model.name, "bytes": args.model.stat().st_size, "sha256": file_sha256(args.model),
                  **facts},
        "prompts": {"file": str(args.prompts), "sha256": file_sha256(args.prompts), "count": len(prompts)},
        "max_new_tokens": args.max_new_tokens, "warmup": args.warmup, "sampler": "top-k 1 (argmax), seed 0",
        "lenience": 0.0, "seconds": time.perf_counter() - started})
    print(f"{args.tag}: {len(prompts)} prompts decoded by mode {args.mode} on the {args.backend} backend")
    return 0


# ---------------------------------------------------------------------------------------------
# read: the texts compared, the parting tokens scored, the rules applied (text-VPA's environment)
# ---------------------------------------------------------------------------------------------

def _load_samples(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _by_id(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in records}


def _mode_of(tag: str) -> str:
    mode = tag.split("-", 1)[0]
    if mode not in ("off", "on", "adaptive"):
        raise ValueError(f"tag {tag}: a tag is <mode>-<name>, the mode one of off, on, adaptive")
    return mode


def _pairs(tags: Sequence[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """P2a's pairs (consecutive runs of one mode) and P2b's (the i-th on run against the i-th off run)."""
    by_mode: dict[str, list[str]] = {}
    for tag in tags:
        by_mode.setdefault(_mode_of(tag), []).append(tag)
    repeat = [(runs[k], runs[k + 1]) for runs in by_mode.values() for k in range(len(runs) - 1)]
    speculation = [(off, other) for mode in ("on", "adaptive") for off, other in
                   zip(by_mode.get("off", []), by_mode.get(mode, []), strict=False)]
    return repeat, speculation


def _shown(tokenizer: Any, token: int) -> str:
    return repr(tokenizer.decode([token]))


def _differences(result: dict[str, Any], limit: int = 10) -> str:
    firsts = sorted(result["first_differences"].values())
    return ", ".join(str(d) for d in firsts[:limit]) + (" ..." if len(firsts) > limit else "")


def _read(args: argparse.Namespace) -> int:
    tags = list(args.tags)
    runs = {tag: _by_id(_load_samples(args.out / tag / "samples.jsonl")) for tag in tags}
    environments = {tag: json.loads((args.out / tag / "run.json").read_text(encoding="utf-8")) for tag in tags}
    ids = list(runs[tags[0]])
    for tag in tags:
        if list(runs[tag]) != ids:
            raise SystemExit(f"run {tag} does not cover the prompts of {tags[0]} in the same order")
    repeat_pairs, speculation_pairs = _pairs(tags)

    p2a = {f"{a} / {b}": compare(runs[a], runs[b], ids) for a, b in repeat_pairs}
    p2b = {f"{a} / {b}": compare(runs[a], runs[b], ids) for a, b in speculation_pairs}
    repeats = all(r["same_text"] == r["prompts"] for r in p2a.values()) if p2a else None

    step0: dict[str, dict[str, dict[str, Any]]] = {}
    against_step0: dict[str, Any] = {}
    if args.step0 is not None and args.step0.exists():
        for record in _load_samples(args.step0):
            step0.setdefault(record["mode"], {})[record["id"]] = record
        for tag in tags:
            mode = _mode_of(tag)
            if mode in step0 and all(i in step0[mode] for i in ids):
                against_step0[f"{tag} / Step 0 {mode}"] = compare(step0[mode], runs[tag], ids)

    # The pairs whose parting token is scored: P2b's, then Step 0's GPU row (on and adaptive against off); P2a's
    # pairs that differ are scored too, and reported apart from the verdict.
    candidates: list[tuple[str, str, str, str]] = []           # (source, prompt id, plain text, other text)
    for a, b in speculation_pairs:
        candidates += [(f"P2 {b} / {a}", i, runs[a][i]["text"], runs[b][i]["text"]) for i in ids]
    for mode in ("on", "adaptive"):
        if "off" in step0 and mode in step0:
            candidates += [(f"Step 0 {mode} / off", i, step0["off"][i]["text"], step0[mode][i]["text"])
                           for i in step0["off"] if i in step0[mode]]
    repeat_candidates = [(f"P2a {b} / {a}", i, runs[a][i]["text"], runs[b][i]["text"])
                         for a, b in repeat_pairs for i in ids]

    scored: list[dict[str, Any]] = []
    repeat_scored: list[dict[str, Any]] = []
    counts = {"pairs": 0, "same text": 0, "same tokens": 0, "ends": 0, "duplicates": 0, "scored": 0}
    target = "not loaded (--no-ties)"
    if not args.no_ties:
        from chat_vpa.device import resolve_device
        from chat_vpa.e1_common import chat_ids, load_target

        device = resolve_device(args.device)
        model, tokenizer = load_target(args.model, "fp32", device)
        target = f"{args.model}, float32, {device}"
        prompt_text = {row["id"]: row["prompt"] for row in read_prompts(args.prompts)}

        def score_all(items: list[tuple[str, str, str, str]], into: list[dict[str, Any]], count: bool) -> None:
            seen: set[tuple[str, str, str]] = set()
            for source, i, plain, other in items:
                if count:
                    counts["pairs"] += 1
                if plain == other:
                    if count:
                        counts["same text"] += 1
                    continue
                key = (i, plain, other)
                if key in seen:
                    if count:
                        counts["duplicates"] += 1
                    continue
                seen.add(key)
                where = parting(tokenizer, plain, other)
                if where["kind"] != "token":
                    if count:
                        counts[where["kind"]] += 1
                    continue
                s = score_parting(model, chat_ids(tokenizer, prompt_text[i]), where["shared"], where["plain_token"],
                                  where["speculating_token"], device)
                into.append({"source": source, "id": i, "index": where["index"],
                             "plain_token": _shown(tokenizer, where["plain_token"]),
                             "speculating_token": _shown(tokenizer, where["speculating_token"]),
                             "argmax": _shown(tokenizer, s["argmax"]), **{k: v for k, v in s.items() if k != "argmax"},
                             "class": classify(s["plain_rank"], s["speculating_rank"], s["gap"])})
                if count:
                    counts["scored"] += 1

        score_all(candidates, scored, True)
        score_all(repeat_candidates, repeat_scored, False)

    classes = [row["class"] for row in scored]
    reading = verdict(classes, repeats) if not args.no_ties else "not read: the parting tokens were not scored"
    summary = {"what": "P2: LiteRT-LM's GPU backend at lenience 0, repeated (docs/e1.md, Section 11.7)",
               "tags": tags, "environments": environments, "prompts": len(ids), "target": target,
               "p2a": p2a, "p2a_repeats": repeats, "p2b": p2b, "against_step0": against_step0,
               "p2c": {"counts": counts, "classes": {c: classes.count(c) for c in ("near tie", "far", "other")},
                       "rows": scored, "p2a_rows": repeat_scored},
               "rule": {"near_tie_nats": NEAR_TIE_NATS, "top": TOP, "far_rank": FAR_RANK}, "verdict": reading}
    write_json(args.out / "p2.json", summary)
    text = render(summary)
    (args.out / "p2.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def render(summary: dict[str, Any]) -> str:
    env = summary["environments"][summary["tags"][0]]["environment"]
    driver = env.get("vulkan_driver_files") or "not pinned"
    lines = ["# P2: LiteRT-LM's GPU backend at lenience 0, repeated (docs/e1.md, Section 11.7)", "",
             f"{env['runtime']}, backend {env['backend']}, Vulkan driver {driver}; runs {', '.join(summary['tags'])}, "
             f"one engine and one process each; {summary['prompts']} prompts. Parting tokens scored by "
             f"{summary['target']}. Every number is *measured*.", "",
             "| pair | same text | first differing character (the smallest ten) |", "|---|---|---|"]
    for title, block in (("P2a", summary["p2a"]), ("P2b", summary["p2b"]), ("Step 0", summary["against_step0"])):
        for name, r in block.items():
            lines.append(f"| {title}: {name} | {r['same_text']} of {r['prompts']} | {_differences(r)} |")
    c = summary["p2c"]
    repeats = {True: "repeats itself", False: "does not repeat itself", None: "not run"}[summary["p2a_repeats"]]
    lines += ["", f"P2a: the backend {repeats}."]
    k = c["counts"]
    lines += ["", f"P2c: {k['pairs']} pairs, {k['same text']} with the same text, {k['duplicates']} repeated, "
              f"{k['same tokens']} with the same tokens, {k['ends']} where one ends and the other goes on (not "
              f"scored), {k['scored']} scored: {c['classes']['near tie']} near ties, {c['classes']['far']} far, "
              f"{c['classes']['other']} other.", ""]
    if c["rows"] or c["p2a_rows"]:
        lines += ["| source | prompt | token | plain side | rank, log-prob | speculating side | rank, log-prob "
                  "| gap, nats | target's argmax | class |", "|---|---|---|---|---|---|---|---|---|---|"]
        for row in c["rows"] + c["p2a_rows"]:
            lines.append(f"| {row['source']} | {row['id']} | {row['index']} | `{row['plain_token']}` | "
                         f"{row['plain_rank']}, {row['plain_logp']:.3f} | `{row['speculating_token']}` | "
                         f"{row['speculating_rank']}, {row['speculating_logp']:.3f} | {row['gap']:.3f} | "
                         f"`{row['argmax']}` | {row['class']} |")
        lines.append("")
    lines.append(f"**Verdict (Section 11.7, P2):** {summary['verdict']}.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="P2: LiteRT-LM's GPU backend at lenience 0, repeated", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="one engine of one mode in this process (LiteRT-LM's environment)")
    r.add_argument("--mode", required=True, choices=("off", "on", "adaptive"))
    r.add_argument("--tag", required=True, help="the run's name, <mode>-<name> (off-a, on-a, ...)")
    r.add_argument("--model", type=Path, required=True, help="the .litertlm file")
    r.add_argument("--prompts", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--backend", default="gpu", choices=("cpu", "gpu"))
    r.add_argument("--threads", type=int, default=None, help="CPU threads (the cpu backend)")
    r.add_argument("--max-new-tokens", type=int, default=256)
    r.add_argument("--limit", type=int, default=10)
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--cache-dir", type=Path, default=None, help="compiled artifacts [beside the model, in cache/]")
    d = sub.add_parser("read", help="compare the runs, score the parting tokens, read the rules")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--tags", nargs="+", default=["off-a", "on-a", "off-b", "on-b"])
    d.add_argument("--model", type=Path, help="the released checkpoint (safetensors) that scores the parting tokens")
    d.add_argument("--prompts", type=Path, help="the prompts the runs decoded")
    d.add_argument("--step0", type=Path, default=None, help="Step 0's samples.jsonl of the same backend")
    d.add_argument("--device", default="auto")
    d.add_argument("--no-ties", action="store_true", help="compare the texts only; load no model")
    args = p.parse_args(argv)
    if args.command == "run":
        if _mode_of(args.tag) != args.mode:
            p.error(f"--tag {args.tag} names another mode than --mode {args.mode}")
        return _run(args)
    if not args.no_ties and (args.model is None or args.prompts is None):
        p.error("read needs --model and --prompts to score the parting tokens (or --no-ties)")
    return _read(args)


if __name__ == "__main__":
    sys.exit(main())
