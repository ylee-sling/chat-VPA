# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Functional-correctness evaluation: Pass@k on HumanEval and MBPP (from text-VPA's ``text_vpa/evaluation``)."""

from chat_vpa.evaluation.execution import ExecutionResult, run_program, run_programs
from chat_vpa.evaluation.passk import pass_at_k, summarize_pass_at_k
from chat_vpa.evaluation.tasks import EvalTask, load_humaneval_tasks, load_mbpp_tasks

__all__ = [
    "EvalTask",
    "ExecutionResult",
    "load_humaneval_tasks",
    "load_mbpp_tasks",
    "pass_at_k",
    "run_program",
    "run_programs",
    "summarize_pass_at_k",
]
