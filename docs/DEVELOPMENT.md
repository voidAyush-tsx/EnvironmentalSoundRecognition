# Development guide

Use [the root README](../README.md) to choose a workflow. This guide explains
where behavior lives, how to change it and how to check the result.

## Code map

The main path is:

`metadata -> waveform Dataset -> adapted CNN14 -> training/validation ->
selected trainable parameters -> held-out evaluation -> reports`.

| File | Responsibility |
| --- | --- |
| `src/__main__.py`, `src/arguments.py` | One command interface, shared argument parsers and lightweight help |
| `src/config.py` | Paths, class IDs, model frontend defaults and separate historical experiment identity |
| `src/environment.py` | Installed-package/GPU checks and explicit dependency snapshots |
| `src/metadata.py` | Official CSV validation, physical-file checks, fold and source-overlap audits |
| `src/dataset.py` | Audio preprocessing, Dataset items and disjoint train/validation/test loaders |
| `src/pann_adapter.py` | Pinned upstream imports and strict pretrained-backbone loading |
| `src/finetune_model.py` | Ten-class head, freeze policies, optimizer groups and trainable-state restoration |
| `src/finetune_train.py` | Training, validation, accumulation, early stopping and checkpoint selection |
| `src/evaluate_selected.py` | Selected-run guards, held-out inference, metrics, predictions and figures |
| `src/runtime.py`, `src/gradient_audit.py` | Device/precision/seeding policy and numerical/gradient checks |
| `src/diagnostics.py` | Explicit model, loader and backward/update diagnostics |
| `src/artifacts.py`, `src/evidence.py` | Small artifact I/O and provenance-checked historical results |
| `src/processes.py`, `src/notebook_helpers.py` | Command logging and concise notebook presentation helpers |
| `src/submission.py` | Faculty audit, archive creation/inspection and the managed README summary |

`src/__init__.py` marks the package; it does not run a workflow. The upstream
source under `external/audioset_tagging_cnn/` retains its own layout and license.
Application changes belong in `src/`, rather than a fork of that dependency.

## Make a change

- **Change asset locations:** pass `--dataset-root`, `--weights` and
  `--pann-source` where applicable. Relative CLI paths use the repository root.
- **Add a training option:** define it in `arguments.py`, consume it in the
  trainer and record it in `config.json`. General defaults are three epochs and
  patience three; the recorded comparison explicitly used six and two.
- **Change preprocessing:** edit `read_waveform` and test temporary WAVs for
  channel conversion, resampling, cropping/padding, shape and label identity.
  A changed method is a new experiment; keep earlier evidence identifiable.
- **Change which layers train:** edit `CNN14ForUrbanSound.configure`, then verify
  optimizer membership, gradients, actual updates and frozen BatchNorm statistics.
- **Change metrics:** edit `compute_metrics` and use known probabilities and
  confusion counts. OvR AUC uses probabilities. Preserve existing artifact fields
  or supply a compatible reader for a deliberate schema change.
- **Change notebook presentation:** call shared helpers. Default review must read
  committed evidence without WAVs, model checkpoints, CUDA, a submission ZIP or
  command side effects. Keep saved research outputs and provenance.

## Verify changes

```powershell
python -m src check --unit
python -m src review
python -m src train --help
```

Unit tests use synthetic WAVs, toy backbones, temporary directories and tiny
checkpoint/archive fixtures. Workflow tests check help/results without site
packages and run notebook default review with reports/figures but no local
data/model assets. They do not train CNN14 or evaluate the held-out fold.

Run the checks relevant to a substantive change. For model integration, use the
explicit diagnostics after restoring assets as described in
[reproducibility](REPRODUCIBILITY.md). Record what was actually executed: toy-model
success does not establish GPU feasibility or reproduce a historical score.
Held-out inference is an experiment action, not a routine development test.

Inspect the diff before committing. Keep raw WAVs, weights, private archives and
Python caches out of Git. Some historical files under `results/` are already
tracked despite the ignore rule; inspect tracked state rather than deleting them
as generic generated clutter. Committed `reports/` and notebook outputs are
research evidence.

## Notebook workflow

Notebook 01 covers data/model foundations, Notebook 02 covers validation and the
selected result, and Notebook 03 covers research/submission. Their default action
flags are off. Notebook 01 additionally starts with `REVIEW_ONLY=True`.

Change an action flag only for the operation you intend to run. Long commands
write logs under `results/notebook_evidence/`. Source code, configuration and
CSV/JSON artifacts remain authoritative when interpreting historical outputs.
Ordinary maintenance such as a kernel restart or Git inspection belongs in the
editor/terminal; it does not need another notebook workflow.

## Command replacements

The unified CLI replaces the scattered root scripts and stage runners:

| Earlier entry point | Current command |
| --- | --- |
| `check_dependencies.py` | `python -m src check --environment` |
| `check_urbansound8k.py` | `python -m src audit-data` |
| `finalize_requirements.py` | `python -m src snapshot-env --out <path>` |
| `src/check_folds.py` | `python -m src audit-folds` |
| `src/check_cnn14.py` | `python -m src check-model --use-safe-weights` |
| `src/check_dataloader.py` | `python -m src check-loader` |
| `src/finetune_check.py` | `python -m src check-gradients --config <mode>` |
| `tools/faculty_submission.py` | `python -m src faculty audit` or `faculty package` |
| Combined stage runners | Choose unit, metadata, audio, model or gradient checks explicitly |

Use `python -m src --help` and command-specific `--help` for the current options.

## Terms

- **Backbone:** pretrained CNN14, providing the audio representation.
- **Head:** the task-specific linear classifier attached to its embedding.
- **Logits:** ten raw class scores consumed by cross-entropy.
- **Fold:** one predefined dataset partition. Validation selects the model; a
  separate held-out fold measures the selected model.
- **Macro-F1:** the average F1 across all ten classes, with equal class weight.
