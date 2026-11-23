# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Tests of inspect_rollouts.py: how rollouts of ``write`` prompts end. Every expected value is worked out by
hand from the text of its case (line numbers in the comments), never by the function under test."""

from __future__ import annotations

import json

import pytest

from chat_vpa import inspect_rollouts as ir
from chat_vpa.data import Example
from chat_vpa.rollout_throughput import _examples_text, take_examples
from chat_vpa.rollouts import Rollout


def _write_prompt(description: str) -> str:
    return '"""\n' + description + '\n"""\n\n'


def _record(text: str, stop: str, name: str = "add", index: int = 0, generated: int = 256, kind: str = "write"):
    return {"index": index, "kind": kind, "stop": stop, "generated": generated, "kept": stop in ("truncation", "eos"),
            "prompt": _write_prompt(f"Do it.\nassert {name}(...) == ..."), "reference": "", "text": text}


def test_named_function_takes_the_last_assert_line():
    corpus = _write_prompt("Compute things.\n\nExample:\n    assert compute(1) == 2\nassert compute_all(...) == ...")
    assert ir.named_function(corpus) == "compute_all"               # the docstring's own example comes first
    mbpp = _write_prompt("Write a function to find the shared elements.\n"
                         "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)")
    assert ir.named_function(mbpp) == "similar_elements"
    assert ir.named_function(_write_prompt("Nothing to call here.")) is None


def test_category_of_the_line_that_ends_a_function():
    assert ir.category("def sub(a, b):") == ir.category("async def sub(a):") == "def"
    assert ir.category("class Circle:") == "class"
    assert ir.category("@property") == "decorator"
    assert ir.category("import os") == ir.category("from math import pi") == "import"
    assert ir.category("# Example usage") == "comment"
    assert ir.category("VALUE = 3") == "other code"
    assert ir.category("assert add(1, 2) == 3") == "write stop 'assert'"
    assert ir.category("print(add(1, 2))") == "write stop 'print('"
    assert ir.category('"""') == "write stop '\"\"\"'"
    assert ir.category("if __name__ == '__main__':") == "write stop 'if __name__'"


def test_a_function_followed_by_another_def():
    # lines: 0 def add, 1 body, 2 '', 3 '', 4 def sub, 5 body, 6 ''
    text = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
    assert ir.function_end(text, "add") == (0, 4, True)
    reading = ir.read_write(_record(text, "budget"))
    assert (reading.name, reading.written, reading.complete, reading.parses) == ("add", True, True, True)
    assert (reading.follow, reading.follow_text) == ("def", "def sub(a, b):")


def test_a_black_style_signature_is_not_the_end():
    # lines: 0 import, 1 '', 2 '', 3 def area(, 4 arg, 5 ') -> float:' (column 0), 6 body, 7 '', 8 '', 9 class
    text = "import math\n\n\ndef area(\n    r: float,\n) -> float:\n    return math.pi * r * r\n\n\nclass Circle:\n    pass\n"
    assert ir.function_end(text, "area") == (3, 9, True)
    reading = ir.read_write(_record(text, "budget", name="area"))
    assert (reading.follow, reading.parses) == ("class", True)


def test_a_column_zero_comment_inside_the_body_is_not_the_end():
    # lines: 0 def f, 1 body, 2 '# keep going' (indented code follows), 3 body, 4 '', 5 'result = f(2)'
    text = "def f(x):\n    y = x + 1\n# keep going\n    return y\n\nresult = f(2)\n"
    assert ir.function_end(text, "f") == (0, 5, True)
    assert ir.read_write(_record(text, "budget", name="f")).follow == "other code"


def test_a_comment_after_the_function_ends_it():
    # lines: 0 def g, 1 body, 2 '', 3 '# Example usage' (column-0 code follows), 4 'x = g()'
    text = "def g():\n    return 1\n\n# Example usage\nx = g()\n"
    assert ir.function_end(text, "g") == (0, 3, True)
    reading = ir.read_write(_record(text, "budget", name="g"))
    assert (reading.follow, reading.follow_text) == ("comment", "# Example usage")


def test_a_column_zero_line_inside_brackets_is_skipped_because_the_text_before_it_does_not_parse():
    # lines: 0 def doc, 1 's = (', 2 '"flush left"' (column 0, inside the parentheses), 3 '    )', 4 body,
    # 5 '', 6 '', 7 def next_one
    text = 'def doc():\n    s = (\n"flush left"\n    )\n    return s\n\n\ndef next_one():\n    pass\n'
    assert ir.function_end(text, "doc") == (0, 7, True)


def test_helpers_before_the_named_function_belong_to_the_cut():
    # lines: 0 def helper, 1 body, 2 '', 3 '', 4 def main_fn, 5 body, 6 '', 7 '', 8 'VALUE = 3'
    text = "def helper(x):\n    return x + 1\n\n\ndef main_fn(y):\n    return helper(y)\n\n\nVALUE = 3\n"
    assert ir.function_end(text, "main_fn") == (4, 8, True)
    assert ir.read_write(_record(text, "budget", name="main_fn")).follow == "other code"


def test_the_named_function_not_written():
    text = "import os\n\n\ndef helper():\n    return os.sep\n"
    assert ir.function_end(text, "solve") == (None, None, False)
    reading = ir.read_write(_record(text, "budget", name="solve"))
    assert (reading.written, reading.complete, reading.parses) == (False, False, False)
    assert (reading.follow, reading.follow_text) == ("named function not written", "import os")
    no_assert = dict(_record(text, "budget"), prompt=_write_prompt("Nothing to call here."))
    assert ir.read_write(no_assert).follow == "no assert line in the prompt"


def test_how_a_function_that_runs_to_the_end_of_the_text_is_read():
    valid = "def long_one(n):\n    total = 0\n    for i in range(n):\n        total += i\n"
    assert ir.function_end(valid, "long_one") == (0, None, True)
    reading = ir.read_write(_record(valid, "budget", name="long_one"))
    assert (reading.follow, reading.complete, reading.parses) == ("end of text at the budget", False, True)
    broken = "def long_one(n):\n    total = [\n        1,\n"
    assert ir.function_end(broken, "long_one") == (0, None, False)
    done = "def h(a):\n    return a * 2\n"
    stopped = ir.read_write(_record(done, "truncation", name="h"))
    assert (stopped.follow, stopped.complete, stopped.parses) == ("write stop (cut from the stored text)", True, True)
    assert ir.read_write(_record(done, "eos", name="h")).follow == "<eos>"


def test_repeated_share_counts_long_lines_that_occurred_before():
    text = "    total += value\n    total += value\n    x\n    return total\n"
    # lines of at least 8 characters: 'total += value' twice, 'return total' once -> 1 repeat of 3
    assert ir.repeated_share(text) == pytest.approx(1 / 3)
    assert ir.repeated_share("a\nb\n") == 0.0


class _CharTokenizer:
    """ids are code points: decoding is exact, so a round trip through the writer must return every text."""

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_examples_text_round_trips_through_the_writer():
    cases = [
        (_write_prompt("Add.\nassert add(...) == ..."), "def add(a, b):\n    return a + b\n",
         "write", "budget", 256, "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n", False),
        ("def mean(xs):\n    \"\"\"Mean.\n\n    Returns\n    -------\n    float\n    \"\"\"\n", "    return sum(xs)\n",
         "complete", "truncation", 9, "    return sum(xs) / len(xs)\n", True),
        (_write_prompt("Odd.\nassert odd(...) == ..."), "def odd(n):\n    return n % 2\n",
         "write", "budget", 256, "def odd(n):\n----- prompt\n    return n % 2\n", False),     # a marker-like line
        (_write_prompt("Empty.\nassert empty(...) == ..."), "def empty():\n    pass\n",
         "write", "truncation", 3, "\n", True),                                          # an empty program
    ]
    examples = [Example([ord(c) for c in prompt], [ord(c) for c in reference], kind)
                for prompt, reference, kind, *_ in cases]
    rollouts = [Rollout(kind, stop, generated, text, [1] if kept else None)
                for _, _, kind, stop, generated, text, kept in cases]
    written = _examples_text(examples, rollouts, _CharTokenizer(), len(cases))
    records = ir.parse_examples_text(written)
    expected = [{"index": i, "kind": kind, "stop": stop, "generated": generated, "kept": kept, "prompt": prompt,
                 "reference": reference, "text": text}
                for i, (prompt, reference, kind, stop, generated, text, kept) in enumerate(cases)]
    assert records == expected


def test_report_and_cli_on_a_rollouts_file(tmp_path, capsys):
    loop = "".join("    if x == 1:\n        return 1\n" for _ in range(6))
    records = [
        _record("def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n", "budget", index=0),
        _record("def add(a, b):\n    total = [\n        a,\n", "budget", index=1),
        _record("import os\n\n\ndef helper():\n    return os.sep\n", "budget", index=2),
        _record("def add(a, b):\n    return a + b\n", "truncation", index=3, generated=17),
        {"index": 4, "kind": "complete", "stop": "budget", "generated": 256, "kept": False,
         "prompt": "def f(x):\n", "reference": "    return 1\n", "text": loop},
    ]
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert ir.load_rollouts(path) == records
    assert ir.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert f"== {path}: 5 rollouts" in out
    assert "write: 4 rollouts (stops: budget 3, truncation 1)" in out
    # written: 0, 1, 3; ends inside the rollout: 0 (a def after it) and 3 (the rule's stop); both parse
    assert "the named function: written in 3, ends inside the rollout in 2, and the text up to its end parses in 2" in out
    assert "at the budget whose named function ends inside them (cut there: a program): 1 of 3 (parses: 1)" in out
    # the complete rollout repeats 'if x == 1:' and 'return 1' five times each: 10 repeats of 12 long lines
    assert "complete: 1 rollouts (stops: budget 1); repeated lines in the rollouts at the budget: median 0.83" in out
    assert "share above 0.5: 1.00" in out


class _OneKindStream:
    def take(self, count):
        return [Example([2, 7], [8, 1], "complete") for _ in range(count)]


def test_take_examples_gives_up_when_the_stream_has_no_such_format():
    examples, read = take_examples(_OneKindStream(), 3, None)
    assert len(examples) == read == 3
    with pytest.raises(SystemExit, match="fewer than 2 'write' examples among the first 20"):
        take_examples(_OneKindStream(), 2, "write", patience=10)
