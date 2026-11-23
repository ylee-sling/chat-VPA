# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
inspect_rollouts.py -- how did the rollouts of a throughput run end? A diagnostic for choice C6 of docs/e0.md.

    python -m chat_vpa.inspect_rollouts runs/e0-throughput/examples.txt
    python -m chat_vpa.inspect_rollouts runs/e0-write/rollouts.jsonl

On the server: ``bash scripts/ncp/rollout_throughput.sh inspect`` (with ``OUT`` for another run), which writes
nothing outside runs/.

Reads what ``rollout_throughput.py`` wrote, ``examples.txt`` (the first prompts, as text) or ``rollouts.jsonl``
(every rollout of the best batch size), and reports how each rollout ended. For a ``write`` prompt the
truncation rule (``postprocess.py``) keeps lines up to the first line that starts with one of ``WRITE_STOPS``; a
new top-level ``def`` does not stop it, because an MBPP program may define helpers, so a rollout that goes on
writing its module after the function reaches the budget and is dropped as a training target (``rollouts.py``).
For every ``write`` rollout the report says whether the function that the prompt names (its last line
``assert name(...) == ...``) was written, which line ends it, what kind of line that is, and whether the text up
to there is valid Python: whether cutting the rollout at the end of the named function would give a program. For
``complete`` prompts, whose rule stops at the first line in column 0, it reports how often the rollouts that
reached the budget repeat their own lines (a loop) rather than write one long body.

The end of a function is the first non-blank line in column 0 after its ``def`` line that is not a closing
bracket (a black-style signature), not a column-0 comment inside the body, and up to which the text parses (a
column-0 line inside a multi-line string does not). Nothing here changes a rule: the cut at the end of the named
function is a candidate. A rollout that the rule stopped is stored without its stop line, so for those the line
after the function is the rule's stop.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chat_vpa.postprocess import WRITE_STOPS, parses

__all__ = ["REPEAT_MIN_CHARS", "WriteReading", "category", "function_end", "load_rollouts", "named_function",
           "parse_examples_text", "read_write", "repeated_share"]

REPEAT_MIN_CHARS = 8               # shorter lines ("else:", "pass") repeat in any program
_HEADER = re.compile(r"^===== example (\d+) \((complete|write); rollout stopped by (\w+) after (\d+) tokens; "
                     r"(kept|DROPPED)\)$")
_SECTIONS = {"----- prompt": ("prompt", None), "----- human reference": ("reference", "prompt"),
             "----- rollout (truncated)": ("text", "reference")}
_ASSERT = re.compile(r"^\s*assert\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(")


# ---------------------------------------------------------------------------------------------
# Reading the files
# ---------------------------------------------------------------------------------------------

def parse_examples_text(text: str) -> list[dict[str, Any]]:
    """The blocks of an ``examples.txt`` (``rollout_throughput._examples_text``) as rollout records with the
    keys of ``rollouts.jsonl``. The markers are only taken in their order, so a program line that happens to
    read "----- prompt" does not open a section."""
    records: list[dict[str, Any]] = []
    block: dict[str, Any] | None = None
    section: str | None = None
    for line in text.split("\n"):
        header = _HEADER.match(line)
        if header:
            if block is not None:
                records.append(_finish(block))
            index, kind, stop, generated, kept = header.groups()
            block = {"index": int(index), "kind": kind, "stop": stop, "generated": int(generated),
                     "kept": kept == "kept", "prompt": [], "reference": [], "text": []}
            section = None
            continue
        if block is None:
            continue
        opened = _SECTIONS.get(line)
        if opened is not None and opened[1] == section:
            section = opened[0]
            continue
        if section is not None:
            block[section].append(line)
    if block is not None:
        records.append(_finish(block))
    return records


def _finish(block: dict[str, Any]) -> dict[str, Any]:
    out = dict(block)
    out["prompt"] = "\n".join(block["prompt"])
    out["reference"] = "\n".join(block["reference"])
    # The writer puts one empty line after each rollout; a truncated program ends with exactly one newline.
    out["text"] = "\n".join(block["text"]).rstrip("\n") + "\n"
    return out


def load_rollouts(path: str | Path) -> list[dict[str, Any]]:
    """Rollout records from ``rollouts.jsonl`` or ``examples.txt``."""
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return parse_examples_text(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# Reading one rollout
# ---------------------------------------------------------------------------------------------

def named_function(prompt: str) -> str | None:
    """The function a ``write`` prompt names: that of its last line ``assert name(...``. The corpus's prompts end
    their description with ``assert name(...) == ...``; an MBPP prompt holds its first test. A docstring's own
    example asserts come before that line."""
    names = [m.group(1) for m in map(_ASSERT.match, prompt.split("\n")) if m]
    return names[-1] if names else None


def category(line: str) -> str:
    """What kind of line, in column 0, ends a function."""
    for stop in WRITE_STOPS:
        if line.startswith(stop):
            return f"write stop {stop.strip()!r}"
    if line.startswith(("def ", "async def ")):
        return "def"
    if line.startswith("class "):
        return "class"
    if line.startswith("@"):
        return "decorator"
    if line.startswith(("import ", "from ")):
        return "import"
    if line.startswith("#"):
        return "comment"
    return "other code"


def _ends_here(lines: Sequence[str], j: int) -> bool:
    line = lines[j]
    if not line.strip() or line[0].isspace() or line[0] in ")]}":
        return False
    if not line.startswith("#"):
        return True
    for following in lines[j + 1:]:                       # a comment ends the function only if code in column 0
        if not following.strip() or following.startswith("#"):   # (or nothing) comes after it
            continue
        return not following[0].isspace()
    return True


def function_end(text: str, name: str) -> tuple[int | None, int | None, bool]:
    """(line of ``def name``, line that ends it, whether the text up to that line parses) in ``text``.

    The def line is None when the function was not written; the end is None when it runs to the end of the
    text, and then the third value says whether the whole text parses."""
    lines = text.split("\n")
    pattern = re.compile(rf"^(?:async\s+)?def\s+{re.escape(name)}\s*[(\[]")
    start = next((i for i, line in enumerate(lines) if pattern.match(line)), None)
    if start is None:
        return None, None, False
    candidates = [j for j in range(start + 1, len(lines)) if _ends_here(lines, j)]
    for j in candidates:
        if parses("\n".join(lines[:j]) + "\n"):
            return start, j, True
    if candidates:
        return start, candidates[0], False
    return start, None, parses(text)


def repeated_share(text: str, min_chars: int = REPEAT_MIN_CHARS) -> float:
    """Share of the lines (stripped, at least ``min_chars`` long) that repeat an earlier line of the text."""
    seen: set[str] = set()
    total = repeats = 0
    for line in text.split("\n"):
        key = line.strip()
        if len(key) < min_chars:
            continue
        total += 1
        if key in seen:
            repeats += 1
        else:
            seen.add(key)
    return repeats / total if total else 0.0


@dataclass(frozen=True)
class WriteReading:
    """How one rollout of a ``write`` prompt ended."""

    index: int
    stop: str                      # how the rollout stopped (rollouts.STOPS)
    generated: int
    name: str | None               # the function the prompt names
    written: bool                  # its def line is in the rollout
    complete: bool                 # it ends inside the rollout (a line after it, the rule's stop, or <eos>)
    parses: bool                   # the text up to its end (or the whole text) is valid Python
    follow: str                    # what ends it
    follow_text: str               # that line, or the rollout's first line if the function was not written
    repeated: float


def _short(line: str, width: int = 60) -> str:
    line = line.rstrip()
    return line if len(line) <= width else line[: width - 3] + "..."


def read_write(record: dict[str, Any]) -> WriteReading:
    """The reading of one ``write`` rollout record (see the module docstring)."""
    text, stop = record["text"], record["stop"]
    name = named_function(record["prompt"])
    first = next((line for line in text.split("\n") if line.strip()), "")
    common = {"index": record["index"], "stop": stop, "generated": record["generated"], "name": name,
              "repeated": round(repeated_share(text), 3)}
    if name is None:
        return WriteReading(**common, written=False, complete=False, parses=False,
                            follow="no assert line in the prompt", follow_text=_short(first))
    start, end, valid = function_end(text, name)
    if start is None:
        return WriteReading(**common, written=False, complete=False, parses=False,
                            follow="named function not written", follow_text=_short(first))
    if end is not None:
        line = text.split("\n")[end]
        return WriteReading(**common, written=True, complete=True, parses=valid, follow=category(line),
                            follow_text=_short(line))
    if stop == "truncation":
        follow, complete = "write stop (cut from the stored text)", True
    elif stop == "eos":
        follow, complete = "<eos>", True
    elif stop == "too_long":
        follow, complete = "write stop or <eos> (program too long for the canvas)", True
    else:
        follow, complete = "end of text at the budget", False
    return WriteReading(**common, written=True, complete=complete, parses=valid, follow=follow, follow_text="")


# ---------------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------------

def _counts(values: Sequence[str]) -> str:
    return ", ".join(f"{key} {count}" for key, count in Counter(values).most_common()) or "none"


def report(records: Sequence[dict[str, Any]], rows: int | None = None) -> list[str]:
    lines: list[str] = []
    writes = [r for r in records if r["kind"] == "write"]
    readings = [read_write(r) for r in writes]
    budget = [x for x in readings if x.stop == "budget"]
    lines.append(f"write: {len(writes)} rollouts (stops: {_counts([r['stop'] for r in writes])})")
    if readings:
        written = [x for x in readings if x.written]
        complete = [x for x in written if x.complete]
        lines.append(f"  the named function: written in {len(written)}, ends inside the rollout in {len(complete)}, "
                     f"and the text up to its end parses in {sum(x.parses for x in complete)}")
        lines.append(f"  what ends it: {_counts([x.follow for x in readings])}")
        rescued = [x for x in budget if x.written and x.complete]
        lines.append(f"  rollouts at the budget whose named function ends inside them (cut there: a program): "
                     f"{len(rescued)} of {len(budget)} (parses: {sum(x.parses for x in rescued)})")
        if budget:
            shares = [x.repeated for x in budget]
            lines.append(f"  repeated lines in the rollouts at the budget: median {statistics.median(shares):.2f}, "
                         f"largest {max(shares):.2f}")
        lines.append("  index  stop        tokens  named function            ends with                         "
                     "parses  repeated  line after it / first line")
        for x in readings[:rows]:
            lines.append(f"  {x.index:>5}  {x.stop:<10}  {x.generated:>6}  {str(x.name)[:24]:<24}  {x.follow[:32]:<32}  "
                         f"{'yes' if x.parses else 'no':<6}  {x.repeated:>8.2f}  {x.follow_text}")
    completes = [r for r in records if r["kind"] == "complete"]
    line = f"complete: {len(completes)} rollouts (stops: {_counts([r['stop'] for r in completes])})"
    at_budget = [repeated_share(r["text"]) for r in completes if r["stop"] == "budget"]
    if at_budget:
        line += (f"; repeated lines in the rollouts at the budget: median {statistics.median(at_budget):.2f}, "
                 f"share above 0.5: {sum(s > 0.5 for s in at_budget) / len(at_budget):.2f}")
    lines.append(line)
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="How did the rollouts of a throughput run end?", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", type=Path, nargs="+", help="rollouts.jsonl or examples.txt of rollout_throughput")
    p.add_argument("--rows", type=int, default=None, help="write rollouts listed one per line (default: all)")
    args = p.parse_args(argv)
    for path in args.files:
        records = load_rollouts(path)
        print(f"== {path}: {len(records)} rollouts (a diagnostic; no rule is changed)")
        print("\n".join(report(records, args.rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
