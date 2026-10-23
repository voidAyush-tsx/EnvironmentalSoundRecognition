# Stage 4 — Genuine CNN14 Partial Fine-Tuning

**Project:** Deep Learning Framework for Environmental Sound Recognition Using PyTorch  
**Dataset:** UrbanSound8K (8,732 clips, 10 official folds, 10 official classes)  
**Backbone:** official AudioSet-pretrained PANNs `Cnn14`, Git commit `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`  
**Hardware:** Windows Python 3.13.15, PyTorch 2.7.0+cu128, NVIDIA RTX 3050 Laptop GPU (4 GiB)  
**Status:** Code prepared and tested with CPU toy-backbone unit tests. **Real CNN14 backward-pass and supervised training remain unexecuted** until the local Windows commands below complete.

## What this stage adds

Only the following Stage 4 files need to be copied into the EXISTING project tree:

```text
src/finetune_model.py       Strict CNN14 load, new classifier, freeze policies, optimizer
src/finetune_check.py       One real training-audio backward+update diagnostic
src/finetune_train.py       Train/validate, best-checkpoint selection, CSV/plots
 tests/test_finetune.py     Lightweight model policy and checkpoint tests
 tests/test_finetune_loop.py  Small CPU train/validation loop test
README.md                   Updated ROOT README with full history (replace old version)
run_stage4_checks.py        One-command test/backward verification with logs
STAGE4_README.md            This stage-specific guide
```

**DO NOT overwrite** working Stage 2/3 scripts or the official model/checkpoint. Do not copy the historical ESC-50 project as current training evidence.

## 1. Training architecture and freezing

The code **first** builds the original `Cnn14(classes_num=527)` and loads the AudioSet `state_dict` from the already exported tensor-only checkpoint using `weights_only=True` and `strict=True`. The original PANNs six convolutional blocks, 2,048-dim `fc1`, spectrogram/log-Mel front-end, and original 527-class head are preserved. A separate `nn.Linear(2048,10)` head is attached, producing raw logits for `CrossEntropyLoss`.

| Mode | Trainable params | Frozen params |
| --- | --- | --- |
| `head` | New 10-way classifier | **Entire** pretrained backbone (`bn0`, `conv_block1..6`, `fc1`, original `fc_audioset`) |
| `fc1` | `backbone.fc1`, new classifier | `bn0`, `conv_block1..6`, original AudioSet head |
| `block6` | `backbone.conv_block6`, `backbone.fc1`, new classifier | `bn0`, `conv_block1..5`, original AudioSet head |

The original 527-class AudioSet head is computed by the stock PANNs forward function but is **always frozen and is not used for the UrbanSound8K loss**. `conv_block6` gets actual convolutional gradients in the `block6` mode.

**All original CNN14 BatchNorm running statistics are fixed in `.eval()` mode**, including in trainable `conv_block6`. Its BN affine `weight`/`bias` still receive gradients when their block is unfrozen. This is deliberate for batch size 1. Other standard CNN14 operations remain in the official forward path. In `model.train()`, PANNs' built-in SpecAugmentation and dropout execute **for training only**; evaluation disables them. We have **not** added any external audio augmentation or altered the initial data pipeline.

## 2. Run tests and one-step gradient verification FIRST

**Recommended one-command verification** in Windows Command Prompt at the EXISTING project root:

```bat
cd /d D:\EnvironmentalSoundRecognition
python run_stage4_checks.py
```

This runs the complete unit suite and the three single-update checks, stops on the first failure and saves a log/JSON summary under `results/stage4/checks/`. It does not download/reinstall packages, change the dataset, or run full training. For only CPU unit tests: `python run_stage4_checks.py --unit-only`. After the gradient checks pass, `python run_stage4_checks.py --pilots` includes two **four-batch diagnostic** training runs.

**Equivalent individual commands** for troubleshooting:

```bat
cd /d D:\EnvironmentalSoundRecognition
python -m unittest discover -s tests -v
python -m src.finetune_check --config head --device cuda
python -m src.finetune_check --config fc1 --device cuda
python -m src.finetune_check --config block6 --device cuda
```

Each diagnostic uses **one real training-fold waveform** and one optimizer update (not a trained classification model). It checks `[1,10]`, finite cross-entropy, gradients on intended layers, no gradients/weight updates on frozen early layers, fixed frozen-BN running statistics, actual updates to the classifier/fc1/conv_block6 where appropriate, and GPU peak allocated/reserved memory. The script uses mixed-precision CUDA by default. A test failing with out-of-memory is an actual resource blocker; do not claim the mode passed. Start with `head` and only run `block6` once `head` passes. For CUDA FP16 instability, try `--amp off` (may use more VRAM); CPU is an alternative functionality check, not a substitute for a GPU feasibility test.

## 3. Tiny training smoke test (not academic performance)

```bat
python -m src.finetune_train --config head --test-fold 1 --epochs 1 --batch-size 1 --accum-steps 4 --max-train-batches 4 --max-val-batches 4
```

This touches only **four training batches and four validation batches**. It is marked `diagnostic_only=true` in the output. Do not report the resulting scores or plots as project performance.

Once `block6` gradient test passes, also smoke-test that mode with the same limited batches:

```bat
python -m src.finetune_train --config block6 --test-fold 1 --epochs 1 --batch-size 1 --accum-steps 4 --max-train-batches 4 --max-val-batches 4
```

## 4. First full, single-fold candidate runs (after smoke tests)

These commands use ALL training and validation recordings on test fold 1, preserve fold 1 for later final testing, and compare matched conditions. They do **not** evaluate test accuracy.

```bat
python -m src.finetune_train --config head --test-fold 1 --epochs 6 --patience 2 --batch-size 1 --accum-steps 4
python -m src.finetune_train --config fc1 --test-fold 1 --epochs 6 --patience 2 --batch-size 1 --accum-steps 4
python -m src.finetune_train --config block6 --test-fold 1 --epochs 6 --patience 2 --batch-size 1 --accum-steps 4
```

The `block6` mode is required to meet the project's genuine CNN-layer fine-tuning goal. If 4 GiB GPU memory is insufficient for it, do not quietly substitute only head training or report fc1 adaptation as convolutional fine-tuning; capture the measured OOM, and adjust after analysis.

### Outputs (per run)

Each invocation creates a new `results/stage4/YYYYMMDD_HHMMSS_.../` directory:

- `config.json`: settings, fold IDs/counts, PANNs commit, chosen trainability, GPU/AMP settings.
- `history.csv`: per-epoch train/validation loss and accuracy, **validation macro-F1**, sample counts, runtime, GPU peak allocated memory.
- `learning_curves.png`: training versus validation curves (train accuracy is online with standard augmentation/dropout enabled; validation is eval-mode).
- `best.pt`: only trainable parameters, selected by validation **macro-F1** (tie-break: validation accuracy). Reconstruct with the same pinned pretrained source checkpoint before loading selected parameters. **No test labels are used.**
- `summary.json`: best validation epoch and metric, `diagnostic_only`, exact sample counts and `test_evaluated=false`.

A minimal optimizer uses AdamW with default learning rates **`1e-3` classifier** and **`1e-5` pretrained layers**, weight decay `1e-4`, gradient accumulation 4, gradient clipping norm 1.0, and optional CUDA FP16. These are **pilot settings**, not tuned results. Seed 67 and validation-only early stopping (`patience=2` for full-run example) are specified.

## 5. Evaluation caveats and remaining requirements

- Test fold is never decoded by the Stage 4 trainer. Stage 5 will add final test-only evaluation after configurations are selected by validation.
- Official UrbanSound8K folds are maintained. Five `fsID` values span multiple folds; overlap concern remains open. The Stage 4 trainer does not resolve it.
- This 8-training/1-validation/1-test protocol differs from many published 9/1 UrbanSound8K protocols; note fairness caveats under CO4.
- A one-fold configuration pilot is **not** ten-fold cross-validation. The full CO3/CO5 evaluation should repeat selected configurations on more prescribed folds (preferably all 10) as computationally feasible, and must distinguish completed versus proposed runs.
- No completed training scores, precision/recall/F1, confusion matrix, ROC-AUC, or published comparison figures are supplied with Stage 4. These must be obtained from actual selected checkpoints, later evaluation, and verifiable papers.

## 6. Troubleshooting

- **`CUDA out of memory`**: RTX 3050 has 4 GiB. Close competing GPU programs, use batch size 1/AMP, inspect reported memory. Gradient accumulation reduces update noise but does **not** reduce single-microbatch memory; freezing more layers does.
- **`Missing official ...`**: ensure previous `external/audioset_tagging_cnn` clone exists, has `pytorch/models.py` and `pytorch/pytorch_utils.py` and is at the pinned commit. Do not automatically reclone if the directory exists.
- **`Missing restricted-load AudioSet ...`**: retain `checkpoints/Cnn14_AudioSet_state_dict_safe.pth` from your already successful Stage 2 export.
- **`ModuleNotFoundError: src`**: run `python -m src.finetune_check`, NOT `python src/finetune_check.py`, from the project root.
- **Nonfinite gradients/loss**: record the traceback, CUDA precision and waveform before changing learning rates or dependencies.
