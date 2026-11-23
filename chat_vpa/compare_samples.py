# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
compare_samples.py -- do two evaluations hold the same samples?

    python -m chat_vpa.compare_samples runs/a/mbpp/ar/samples.jsonl runs/b/mbpp/ar/samples.jsonl

Reads two ``samples.jsonl`` files of ``evaluate.py`` or ``block_evaluate.py`` and compares them sample by sample
(same task, same sample index): the raw generated text, the program that was run, and whether it passed. Prints
the agreement rates and the first samples that differ, and exits with status 0 only if every raw text is equal.
The port check of docs/provenance.md uses it: the ported code, with the predecessor's checkpoint, settings and
seeds, must reproduce the predecessor's samples.

If a ``summary.json`` lies next to each file, the settings both record (token draw, seed, split, token budget,
schedule, steps) are compared as well, and those that differ are printed: samples of two different settings are
not expected to agree, and the result should not be read as a defect of the code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["compare", "load_samples", "setting_differences"]

# Settings that decide the samples and that both evaluators record under the same name. The reveal order is not
# among them: the whole-canvas sampler calls the exact kernel "random" and the block sampler "exact".
SETTINGS = ("benchmark", "split", "problems", "samples_per_problem", "seed", "temperature", "top_p", "top_k",
            "canvas_len", "max_new_tokens", "schedule", "num_steps", "adapters", "step")


def load_samples(path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    """(task_id, sample index) -> row."""
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[(str(row["task_id"]), int(row["sample"]))] = row
    return rows


def compare(first: dict[tuple[str, int], dict[str, Any]], second: dict[tuple[str, int], dict[str, Any]]
            ) -> dict[str, Any]:
    """Agreement of two sets of samples on their common keys, and what each has alone."""
    common = sorted(set(first) & set(second))
    different = [key for key in common if first[key].get("raw") != second[key].get("raw")]
    programs = sum(first[key].get("completion") == second[key].get("completion") for key in common)
    outcomes = sum(bool(first[key].get("passed")) == bool(second[key].get("passed")) for key in common)
    tasks = sorted({task for task, _ in common})
    whole_tasks = sum(all(first[key].get("raw") == second[key].get("raw") for key in common if key[0] == task)
                      for task in tasks)
    n = max(len(common), 1)
    return {"samples": len(common), "only_in_first": len(set(first) - set(second)),
            "only_in_second": len(set(second) - set(first)), "identical_raw": len(common) - len(different),
            "raw_agreement": (len(common) - len(different)) / n, "program_agreement": programs / n,
            "outcome_agreement": outcomes / n, "tasks": len(tasks), "tasks_fully_identical": whole_tasks,
            "pass_rate_first": sum(bool(first[key].get("passed")) for key in common) / n,
            "pass_rate_second": sum(bool(second[key].get("passed")) for key in common) / n,
            "first_differences": different[:5]}


def setting_differences(first: str | Path, second: str | Path) -> list[tuple[str, Any, Any]] | None:
    """(name, first value, second value) of every setting recorded by both ``summary.json`` files next to the two
    sample files that differs; None if either summary is missing."""
    summaries = []
    for path in (first, second):
        summary = Path(path).with_name("summary.json")
        if not summary.exists():
            return None
        summaries.append(json.loads(summary.read_text(encoding="utf-8")))
    a, b = summaries
    return [(name, a[name], b[name]) for name in SETTINGS if name in a and name in b and a[name] != b[name]]


def _where(a: str, b: str) -> int:
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return index
    return min(len(a), len(b))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare two samples.jsonl files", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("first", type=Path)
    p.add_argument("second", type=Path)
    p.add_argument("--show", type=int, default=2, help="differing samples to print")
    args = p.parse_args(argv)
    first, second = load_samples(args.first), load_samples(args.second)
    result = compare(first, second)
    identical = (result["samples"] > 0 and result["identical_raw"] == result["samples"]
                 and not result["only_in_first"] and not result["only_in_second"])
    print(f"{args.first}\n{args.second}")
    settings = setting_differences(args.first, args.second)
    if settings is None:
        print("  settings: not compared (a summary.json is missing next to one of the files)")
    elif settings:
        print("  SETTINGS DIFFER (the samples are not expected to agree): "
              + "; ".join(f"{name} {x!r} against {y!r}" for name, x, y in settings))
    else:
        print("  settings recorded by both: equal")
    print(f"  samples compared: {result['samples']} ({result['tasks']} problems; only in the first: "
          f"{result['only_in_first']}, only in the second: {result['only_in_second']})")
    print(f"  identical raw text: {result['identical_raw']}/{result['samples']} ({100 * result['raw_agreement']:.1f}%); "
          f"problems with all samples identical: {result['tasks_fully_identical']}/{result['tasks']}")
    print(f"  same program after truncation: {100 * result['program_agreement']:.1f}%; same test outcome: "
          f"{100 * result['outcome_agreement']:.1f}%; pass rate {100 * result['pass_rate_first']:.1f}% against "
          f"{100 * result['pass_rate_second']:.1f}%")
    for key in result["first_differences"][: args.show]:
        a, b = str(first[key].get("raw", "")), str(second[key].get("raw", ""))
        at = _where(a, b)
        print(f"  differs: {key[0]} sample {key[1]} at character {at}: {a[max(0, at - 30): at + 30]!r} against "
              f"{b[max(0, at - 30): at + 30]!r}")
    print("  RESULT: " + ("identical" if identical else "DIFFERENT"))
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
