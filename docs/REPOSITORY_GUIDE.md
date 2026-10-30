# Repository guide

This is a **notebook-first experimental project**. Begin with the root [README](../README.md) for scope and results; the three notebooks show the actual progression and stored output.

| Path | Role |
|---|---|
| `notebooks/01_*.ipynb` | Dataset, reproducibility, PANNs checkpoint and partial-fine-tuning foundations |
| `notebooks/02_*.ipynb` | Validation comparison and held-out evaluation |
| `notebooks/03_*.ipynb` | CO4 literature, CO5 ablation, submission audit |
| `src/dataset.py` | Waveform loading, official class IDs, fold selection, DataLoaders |
| `src/finetune_model.py` | Strict official PANNs loading, adapted head, freeze policies, optimizer |
| `src/finetune_train.py` | Supervised training, gradient checks, validation checkpoint selection |
| `src/evaluate_selected.py` | Held-out metrics from selected trained parameters |
| `src/check_*.py`, `src/audit_audio.py` | Verifications and dataset scans |
| `tests/` | Offline unit tests (some test model logic with mock modules) |
| `reports/faculty_evidence/` | Curated **actual** experiment data and figures |
| `reports/stage4_comparison/` | Consolidated validation plots |
| `docs/` | Research comparison, ablation, methods, reproduction, faculty requirements |
| `models/CHECKPOINT_ACCESS.md` | Pretrained and trained checkpoint provenance |
| `tools/faculty_submission.py` | Faculty-specific audit and export |

## Data and generated artifacts

`data/UrbanSound8K/`, `checkpoints/`, `results/stage4/*/best.pt` and local experiment logs are not committed. Obtain the upstream files via [REPRODUCIBILITY](REPRODUCIBILITY.md). Small curated results are committed for independent reading, but **do not mistake them for a fresh execution on another machine**.

The original PANNs source at `external/audioset_tagging_cnn` is external code, pinned to a commit; obtain it by following the reproduction steps. The Git revision referenced by the project is `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`.

## What was intentionally kept stable

Existing notebook filenames, src imports, faculty packaging script, evidence paths and selected-checkpoint provenance have not been renamed. Changing any of those could invalidate saved notebook outputs and submission instructions. This organization pass prioritizes clarity and a safe reviewer experience over moving working training modules.
