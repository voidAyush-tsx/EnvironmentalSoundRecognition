# Stage 4 Experiment Review — 27 October 2026

**Project:** Deep Learning Framework for Environmental Sound Recognition Using PyTorch  
**Evidence:** the supplied `Stage4_Three_Full_Runs_Review.zip` (three configurations × `config.json`, `summary.json`, `history.csv`, `learning_curves.png`).

## Controlled validation results (not held-out test results)

| Configuration | Best epoch | Validation accuracy | Validation macro-F1 | Trainable parameters | Six-epoch time | Peak CUDA allocated |
|:--|--:|--:|--:|--:|--:|--:|
| Head only | 4 | 87.73% | 0.8638 | 20,490 | 19.80 min | 474.2 MiB |
| fc1 + head | 4 | 88.06% | 0.8668 | 4,216,842 | 18.87 min | 522.3 MiB |
| conv_block6 + fc1 + head | **4** | **90.65%** | **0.9010** | **60,848,138** | 28.65 min | 1278.1 MiB |

The `block6` model exceeded the head-only baseline by **2.93 percentage points** in validation accuracy and **0.0372 absolute macro-F1**. Compared with fc1, accuracy increased by **2.59 percentage points**. These are validation-split descriptive differences, not tested statistical significance or multi-fold stability.

## Protocol checked from configs and histories

- UrbanSound8K official folds: 3–10 training (**6,971 clips**), fold 2 validation (**888 clips**), fold 1 reserved for testing (**873 clips**).
- All three: PyTorch 2.7.0+cu128, CUDA BF16, seed 67, batch size 1, gradient accumulation 4, six completed epochs, 1,743 optimizer steps/epoch.
- Head LR `0.001`, backbone LR `0.00001`, weight decay `0.0001`, patience 2; selected checkpoints maximize validation macro-F1 (with accuracy as tie-break).
- Every selected checkpoint is epoch 4. `summary.json` in all three runs explicitly records `diagnostic_only: false` and `test_evaluated: false`.
- All full epoch runs record nonzero-gradient steps on intended monitored layers; these records complement the earlier one-step weight-update checks.

## Generalization and training dynamics

Each configuration improves up through **epoch 4**. After that, validation loss increases in all three experiments while training loss continues to decline. For `block6`, validation loss goes from **0.3433** (epoch 4) to **0.3986** (epoch 6), and validation macro-F1 goes from **0.9010** to **0.8847**. This is a warning sign of diminishing generalization after the selected epoch; the best validation-selected checkpoint must be restored instead of evaluating final-epoch weights.

**Caution:** The reported training accuracy is an *online training-mode* measurement, with official CNN14 SpecAugmentation and dropout active. Validation is in evaluation mode. The training-versus-validation accuracy gap is therefore not a clean estimate of the generalization gap; use consistent-mode measurements if a rigorous bias–variance comparison is needed.

## Trade-off and limitations

`block6` fine-tunes **74.33%** of the model's parameters and used about **2.70×** the peak CUDA allocation of the head-only baseline. Six-epoch runtime was **1.45×** head-only; these are measured run totals, affected by system-load variability. The experiments share one validation fold and one seed; do not claim statistically reliable superiority across folds from these results alone.

Five Freesound `fsID` values occur in multiple official folds, with some overlapping annotated source time intervals. This remains a source-isolation caveat; preserve the official split for the primary UrbanSound8K comparison and disclose the issue.

## Predeclared next step: final held-out test, once

**Selection made without test evidence:** choose the `block6` checkpoint at epoch 4 on validation macro-F1. Evaluate **only this selected run** on reserved test fold 1 and calculate accuracy, macro/weighted precision/recall/F1, per-class results, one-vs-rest specificity, multiclass ROC-AUC from softmax probabilities, and a confusion matrix. Do not tune hyperparameters after seeing the final test results. Preserve the checkpoint and evaluation artifacts.

`src/evaluate_selected.py` and the companion notebook implement this evaluation on the user's PC. The 873 test WAVs and Stage 4 `best.pt` weights were not provided in the review ZIP, so **held-out test accuracy has not been calculated here**.
