# Reproduction guide

Begin with [README](../README.md) for the overview and purpose-based navigation. This guide covers
external assets and deliberate experiment reproduction. The recorded training
machine was Windows, Python 3.13.15, PyTorch 2.7.0+cu128, RTX 3050 (4 GiB).
Use [DEVELOPMENT](DEVELOPMENT.md) for code changes and
[RESEARCH](RESEARCH.md) for recorded protocols, results and faculty requirements.

## Environment

Create one virtual environment and select its interpreter in your editor:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m src check --unit
```

For the original CUDA wheel use `requirements.txt` instead of
`requirements-cpu.txt` in the same install command. Do not replace an already
working environment just to review evidence. Both recipes share the existing
version ranges in `requirements-common.txt`; neither installs upstream's old
AudioSet-training dependencies. The CPU recipe was added for development and
has not been freshly installed/validated on every supported platform.

`requirements.lock.txt` is the unchanged snapshot of **direct** packages from
the historical Windows machine. It is not a transitive or hash-verified lock.
To attempt that original dependency snapshot, install it explicitly instead of
the range file. It does not guarantee bit-identical results on other hardware.
Normal checks never refresh it. To record a new environment separately:

```powershell
python -m src snapshot-env --out results/dependencies-current.txt
```

Create the output parent directory first if needed. Use an explicit output path
to retain the historical snapshot. Record the Python version, operating system,
CUDA build and source revision along with new experiment results.

## Dataset

Obtain [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html),
preserving its official layout:

```text
data/UrbanSound8K/
  metadata/UrbanSound8K.csv
  audio/fold1/*.wav
  ...
  audio/fold10/*.wav
```

There are 8,732 metadata entries, ten official class IDs 0-9 and ten predefined
folds. Keep those labels/folds rather than alphabetically remapping labels or
redistributing clips. Use `--dataset-root <path>` for another location. All
relative CLI paths resolve against the checkout.

The pipeline averages channels to mono, resamples to 32 kHz using `soxr_hq`,
then center-crops or right-pads to 128,000 float32 samples. CNN14 computes
log-Mel features internally. Five source IDs span official folds; retain the
[source-dependence caveat](RESEARCH.md#recorded-experiment).

Raw WAVs are excluded from Git and faculty archives. Respect dataset terms and
attribution: Salamon, Jacoby and Bello, *A Dataset and Taxonomy for Urban Sound
Research*, ACM Multimedia (2014).

```powershell
python -m src audit-data --dataset-root data/UrbanSound8K
python -m src audit-folds --metadata data/UrbanSound8K/metadata/UrbanSound8K.csv
python -m src audit-audio --max-files 50
```

The first two check metadata/files and partitions; `audit-audio` decodes audio.
Its limited scan writes isolated pilot outputs under `results/audio_audits/`
and `figures/audio_audits/`. Omitting the limit requests the full audio audit.
Explicit `--results-dir`/`--figures-dir` paths allow generating new audit evidence
while preserving the historical reports.

## Pinned source and pretrained weights

1. In a Git checkout initialize the pinned source submodule:

   ```powershell
   git submodule update --init external/audioset_tagging_cnn
   git -C external/audioset_tagging_cnn rev-parse HEAD
   ```

   Expected revision: `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`.
   Existing local checkouts are preserved by this refactor. Do not overwrite a
   modified checkout; inspect it before deliberately updating its revision.
   A submission ZIP has no Git index: in an absent source directory clone the
   [official repository](https://github.com/qiuqiangkong/audioset_tagging_cnn)
   manually, then check out the same revision. `--pann-source` accepts that
   repository root. The adapter verifies its own Git root/revision and isolates
   upstream imports; no upstream code has been edited.
2. Download original `Cnn14_mAP=0.431.pth` from
   [official Zenodo](https://zenodo.org/records/3576403) into `checkpoints/`.
   Expected published MD5: `595633ac2d1cac7ef04ebf70e2fee4e4`.
   The original is a legacy pickle. Conversion is an explicit operation only
   for a trusted official download whose provenance you have verified:

   ```powershell
   python -m src check-model --trust-legacy-checkpoint --export-safe-weights
   ```

   This retains the existing opt-in unrestricted legacy-loading behavior.
   Subsequent operations use `Cnn14_AudioSet_state_dict_safe.pth` with
   `weights_only=True` and strict state matching. The original MD5 is an integrity
   comparison, not a security guarantee.

Check prerequisites independently before running model operations:

```powershell
python -m src audit-data
python -m src check --environment
python -m src check-model --use-safe-weights --cpu
python -m src check-loader --model-forward
```

Override paths on model commands with `--dataset-root`, `--weights` and
`--pann-source` as applicable. All relative CLI paths use this repository root.
A CUDA-required check is explicit: `python -m src check --environment --require-cuda`.

## Diagnostics and new training runs

After restoring assets, a small head-only diagnostic uses training data only:

```powershell
python -m src check-gradients --config head --device auto
python -m src train --config head --epochs 1 --max-train-batches 4 --max-val-batches 4
```

The limited run is marked `diagnostic_only=true` and cannot be used for final
research evaluation. To reproduce the **training budget** of the recorded
comparison, make separate unrestricted runs:

```powershell
python -m src train --config head --epochs 6 --patience 2 --batch-size 1 --accum-steps 4
python -m src train --config fc1 --epochs 6 --patience 2 --batch-size 1 --accum-steps 4
python -m src train --config block6 --epochs 6 --patience 2 --batch-size 1 --accum-steps 4
```

The CLI's general defaults remain three epochs/patience three. The commands
above explicitly use the historical six/two settings, seed 67 and test fold 1.
Model selection uses validation macro-F1, with accuracy as tie-break. Every
invocation creates a new run; it does not replace recorded historical evidence.

## Checkpoint reconstruction

The original backbone and selected task parameters have different roles:

| Resource | Location / identity |
|---|---|
| Original pretrained AudioSet backbone | `checkpoints/Cnn14_mAP=0.431.pth`, from [official Zenodo](https://zenodo.org/records/3576403) |
| Published original MD5 | `595633ac2d1cac7ef04ebf70e2fee4e4` |
| Restricted-loadable backbone export | `checkpoints/Cnn14_AudioSet_state_dict_safe.pth` |
| Selected task parameters | `results/stage4/20261027_144004_603447_block6_test1/best.pt`; private ZIP location `models/selected_block6_best.pt` |
| Selected task SHA-256 | `7c0e01a0078c225a18f4dc5eaa5110d051ef2c8effec7bcebb4e856e6659328c` |

Build the pinned original 527-class AudioSet backbone, attach the project's
ten-class head in the `block6` configuration, then apply the selected trainable
state. The evaluator follows this order. Loading `best.pt` alone cannot restore
the complete model.

Use `python -m src review` for historical metrics. An explicit saved evaluation
can be read with `python -m src review --evaluation-dir <directory>`; the reader
checks that it matches the recorded selected-model provenance.

Real evaluation requires a completed selected-run directory containing
`config.json`, `summary.json`, `history.csv` and `best.pt`, plus the original
backbone state. A ZIP's `models/selected_block6_best.pt` is trainable parameters
only. To reconstruct its run folder, copy that exact file as `best.pt` alongside
metadata from `reports/faculty_evidence/training_runs/block6/`. Verify its SHA-256
against the identity above. `Get-FileHash <best.pt> -Algorithm SHA256` can do this
on PowerShell. Keep the original
archive and artifact directories intact.

Only when intentionally reproducing the selected experiment:

```powershell
python -m src evaluate --run-dir <explicit-selected-run>
```

The evaluator rejects diagnostic checkpoints, mismatches, wrong folds and a
previously evaluated checkpoint recorded under `results/stage5/`. Reuse that
saved evaluation when available. Do not remove evaluation records to bypass the
guard or tune settings from held-out results. No real test inference was run
for this refactor.

## Notebook review and faculty workflow

Default notebook paths read committed historical artifacts. Notebook 01's
`REVIEW_ONLY=True` avoids local data/model workflows; set it false and enable
individual controls only when required. Asset override controls are available
in Notebooks 01/02. Notebook 03's audit and packaging are separately opt-in;
ZIP verification is skipped while packaging is disabled.

See [faculty submission](RESEARCH.md#faculty-submission) for archive identity,
source attribution and exact selected-checkpoint requirements. Saved outputs
remain historical; reviewing them does not establish a fresh reproduction.
