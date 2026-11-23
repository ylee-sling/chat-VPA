# Provenance: what chat-VPA took from text-VPA

chat-VPA starts from the verified Stage 0 code of its predecessor, imported on 23 November 2026 with a fresh git
history (text-VPA's history would have brought PR-JEPA along).

- **Source:** https://github.com/ylee-sling/text-VPA, branch `stage1-planner`, commit
  `6f973461b76586d188b448a74e1c2e36cb2241a2` ("docs(stage1): the author's decision; work continues as Chat-VPA").
  Its code is identical to that of its parent `ff747c9`; the commit adds the decision to `docs/stage1.md`.
  text-VPA's `main` does not hold this code.
- **Rule of the import:** exactly the import closure of `text_vpa/stage0` and of its tests, with the planner
  stage's plumbing removed. Nothing else.
- **License:** GPL-3.0-only, as in text-VPA (the author's decision, 23 November 2026); SPDX headers and the
  copyright of SLING AI Inc. are carried over.

## Kept

| chat-VPA | text-VPA at `6f97346` | What changed in the import |
|---|---|---|
| `chat_vpa/backbone.py` | `text_vpa/stage0/backbone.py` | planner plumbing removed (below); docstring |
| `chat_vpa/flow.py` | `text_vpa/models/realizers/paths.py`: `KappaSchedule`, `MixtureSample`, `mixture_path_sample`, `uniform_time_grid`; `text_vpa/models/realizers/kernels.py`, whole; `timestep_embedding` from `text_vpa/models/realizers/backbone.py` | three sources in one module; the Logit-KL path functions of `paths.py` and the rest of the PR-JEPA realizer are not imported; equation numbers in docstrings are Chat-VPA's |
| `chat_vpa/data.py` | `text_vpa/stage0/data.py` | docstrings only |
| `chat_vpa/canvas_sampler.py` | `text_vpa/stage0/sampler.py` | renamed (it is the whole-canvas sampler, the B = N reference of the block sampler); planner plumbing removed |
| `chat_vpa/autoregressive.py` | `text_vpa/stage0/autoregressive.py` | none |
| `chat_vpa/postprocess.py` | `text_vpa/stage0/postprocess.py` | none |
| `chat_vpa/evaluate.py` | `text_vpa/stage0/evaluate.py` | planner plumbing removed |
| `chat_vpa/report.py` | `text_vpa/stage0/report.py` | docstring only |
| `chat_vpa/analyze.py` | `text_vpa/stage0/analyze.py` | none |
| `chat_vpa/preflight.py` | `text_vpa/stage0/preflight.py` | wording of two messages |
| `chat_vpa/train.py` | `text_vpa/stage0/train.py` | docstrings only (the generic trainer hooks are kept) |
| `chat_vpa/runtime.py` | `text_vpa/engine/runtime.py`: `MetricLogger`, `save_checkpoint`, `load_checkpoint` | the distributed, optimizer and seeding helpers are not imported; checkpoint format (below) |
| `chat_vpa/device.py` | `text_vpa/utils/device.py` | none |
| `chat_vpa/evaluation/{execution,passk,tasks}.py` | `text_vpa/evaluation/{execution,passk,tasks}.py` | `_read_jsonl` of `text_vpa/data/dataset.py` inlined in `tasks.py`; the temporary directory's prefix is `chat-vpa-exec-` |
| `tests/test_stage0.py` | `tests/test_stage0.py` | imports only |
| `tests/test_stage0_cli.py` | `tests/test_stage0_cli.py` | imports only |
| `tests/test_kernels.py` | `tests/test_kernels.py` | imports only |
| `tests/test_evaluation.py` | `tests/test_evaluation.py`, the tests of kept modules | the six tests of PR-JEPA's command lines and of `diagnose.py` are not imported (below) |
| `scripts/ncp/stage0_sweep.sh`, `scripts/ncp/env.sh` | `scripts/ncp/stage0_sweep.sh`; the environment and GPU checks of `scripts/ncp/stage0.sh` | reads the checkpoint, data and interpreter from `~/text-VPA`, writes under this repository's `runs/` only; the orders of its `parallel` stage are a variable; installs nothing |

In every module the package path changed (`text_vpa.stage0.x` to `chat_vpa.x`, and so on). No class, function,
parameter or submodule of the realizer was renamed: `Stage0Realizer`, `Stage0Config`, `Stage0Trainer` and the
names of all trained tensors are text-VPA's, so that its checkpoints load (`tests/test_port.py`).

### Planner-stage plumbing removed

Branch `stage1-planner` had threaded an optional plan through the Stage 0 code. Removed:

- `backbone.py`: the `plan` argument of `_block`, `_run` and `canvas_hidden`; the `plan` and `residual` fields
  of `AdapterState`; `Stage0Realizer.frozen_states` (used by the latent oracle only).
- `canvas_sampler.py`: the `plan` argument of `realize` and the plan branch of its depth audit.
- `evaluate.py`: `plan_for` of `realizer_samples`; the stage check no longer points to `text_vpa.stage1`
  (it still refuses a checkpoint whose stage is not `stage0`).

Kept, because E0's trainer uses them: the trainer hooks `trainable_parameters`, `set_training`, `collate`,
`canvas_pass`, `extra_state`, `load_extra_state` and the `stage` attribute of `Stage0Trainer`, and the
`tokenize` hooks of `CorpusStream` and `load_validation`.

### Checkpoint format

`runtime.save_checkpoint` writes `"format": "chat-vpa/1"`; `runtime.load_checkpoint` reads `chat-vpa/1` and the
predecessor's `text-vpa/1`. The payload's layout is unchanged.

## Dropped

- **PR-JEPA:** `text_vpa/models/` (encoders, EMA, selector, predictor, bridge, readout, plan, the FS-DFM and
  Logit-KL realizers and their backbone, layers), `text_vpa/engine/` (losses, Stage 1 and Stage 2 trainers, the
  rest of `runtime.py`), `text_vpa/inference/` (sampler, energies, loading), `text_vpa/configs/` (schema, loader,
  YAML groups), `text_vpa/data/` (tokenization, tree-sitter extraction, dataset, synthetic corpus and the corpus
  builder `corpus.py`), `text_vpa/utils/` except `device.py` (profiler, memory probe, verification),
  `text_vpa/evaluation/diagnose.py`, the root command lines `train.py`, `generate.py`, `evaluate.py`,
  `setup.py`, `scripts/gcp/`, `scripts/init_repo.sh`, `scripts/release.sh`, `docs/pr_jepa_preprint.pdf`, and
  their tests.
- **Stage 1 (the planner stage and its oracles):** `text_vpa/stage1/`, `tests/test_stage1.py`,
  `tests/test_stage1_cli.py`, `scripts/ncp/stage1.sh`.
- **Of Stage 0:** `scripts/ncp/stage0.sh` as a pipeline (download, harness check, training, protocol
  evaluation); `Stage0Realizer.frozen_states`.
- **Documents:** `docs/stage0.md` and `docs/stage1.md` are not copied. They remain the record of the Stage 0
  design, of both Stage 0 runs and of the oracle experiments:
  https://github.com/ylee-sling/text-VPA/blob/6f97346/docs/stage0.md and `.../docs/stage1.md`. Comments in the
  imported code that cite "decision N" refer to `docs/stage0.md` there.

## Tests before and after

Run on CPU with torch 2.14.1 and transformers 5.21.0 (23 November 2026).

| file | text-VPA at `6f97346` | chat-VPA | |
|---|---|---|---|
| `tests/test_stage0.py` | 40 | 40 | |
| `tests/test_stage0_cli.py` | 2 | 2 | |
| `tests/test_kernels.py` | 39 | 39 | |
| `tests/test_evaluation.py` | 28 | 22 | 6 not imported |
| total | 109 passed (Python 3.12) | 103 passed (Python 3.12 and 3.13) | |

The six tests not imported test modules that were not imported:
`test_evaluate_cli_reference_mode`, `test_evaluate_cli_with_a_trained_checkpoint` (PR-JEPA's root `evaluate.py`
and `train.py`), `test_diagnose_helpers_find_functions_and_score_programs`,
`test_diagnose_reading_names_the_first_suspect`, `test_diagnose_cli_on_a_trained_checkpoint` (`diagnose.py`) and
`test_inference_threshold_comes_from_the_checkpoint_calibration_unless_disabled` (`inference/loading.py`).
In the kept tests one literal changed (`text-vpa-exec-` to `chat-vpa-exec-`).

New in chat-VPA, `tests/test_port.py` (2 tests): the names and shapes of the tiny realizer's 98 trained tensors
equal those computed with text-VPA's own code at `6f97346` (a digest), and a `text-vpa/1` checkpoint loads. A
state saved by text-VPA's `Stage0Realizer.trainable_state` on the tiny model was also loaded into the ported
realizer directly (checked once, off the test suite).

## The corpus on the server

chat-VPA does not build a corpus. E0 reads the predecessor's, as built, at `~/text-VPA/data2/corpus` on the NCP
server: 1,990,070 training and 9,930 validation Python functions (counts reported by text-VPA's build), in
`train-*.jsonl.gz` shards and `val.jsonl.gz`, rows `{"context", "target", "language", "task_id", "license",
"origin", ...}` with `target` starting with `context`.

It was built by `text_vpa/data/corpus.py` of text-VPA (not imported; see that file at `6f97346` for the exact
procedure) from `codeparrot/github-code-clean` on the Hugging Face Hub:

- **License filter:** files whose license is permissive (MIT, Apache-2.0, BSD, ISC and similar; the list is
  `PERMISSIVE_LICENSES` there); files without a license are dropped by default.
- **Extraction:** self-contained functions (every name read is a parameter, a local, a builtin or a module-level
  import; needed imports are kept). Documented top-level functions first; to reach about 2M functions also
  methods of top-level non-test classes (dedented to column 0, `self` stays a parameter) and undocumented
  functions of at least two statements. Context = needed imports + signature + docstring; target = needed
  imports + the whole function.
- **Deduplication:** exact, on the function's syntax tree without its docstring.
- **Decontamination rule:** a training function is dropped if it shares any 13-word n-gram with a HumanEval or
  MBPP problem (prompt, reference solution, MBPP description and tests), where words are identifiers and numbers
  and punctuation and formatting are ignored (GPT-3's 13-gram rule).
- **Split:** a deterministic hash split into training shards and the validation file.

The benchmarks (`~/text-VPA/data2/eval/HumanEval.jsonl.gz`, `mbpp.jsonl`) and the checkpoint
(`~/text-VPA/data2/models/gemma-4-E2B`, `google/gemma-4-E2B`, pre-trained, Apache 2.0) are read where they are.
