# Verified training details

This document records **what the current source implements**, with run values grounded in the three full-fold configuration files under `reports/faculty_evidence/training_runs/`.

| Item | Recorded value or behavior | Source |
|---|---|---|
| Optimizer | **AdamW**; two parameter groups (head vs later pretrained layers) | `src/finetune_model.py::optimizer_for_model` |
| Classification loss | **Cross-entropy on 10 raw logits** | `src/finetune_train.py::train_epoch`, `validate_epoch` |
| Head learning rate | **1e-3** | Train CLI defaults / `config.json` |
| Pretrained trainable-layer learning rate | **1e-5** | Train CLI defaults / `config.json` |
| Weight decay | **1e-4** | AdamW creation / `config.json` |
| Batch / gradient accumulation | **1 / 4** | `config.json` |
| Maximum epochs / patience | **6 / 2** | `config.json` |
| Precision | **CUDA BF16** on supported GPU | `src/precision_policy.py` / run `config.json` |
| Seed | **67**, deterministic cuDNN flags | `src/finetune_train.py::seed_everything` |
| Gradient clipping | max norm **1.0**; reject nonfinite gradients | `src/finetune_train.py::train_epoch` |
| Scheduler | **None**: the trainer neither constructs nor steps an LR scheduler | `src/finetune_train.py` |
| Checkpoint selection | Best validation **macro-F1**, accuracy as tie-break; epoch **4** | `src/finetune_train.py`, run summaries |
| Freeze policy | Earlier CNN14 blocks frozen; `conv_block6` and `fc1` trainable only in designated configurations | `src/finetune_model.py` |
| BatchNorm | All pretrained BN running statistics held fixed | `CNN14ForUrbanSound.train()` |
| Dropout/SpecAugmentation | Official pretrained CNN14 forward uses its internal training-mode dropout/SpecAugmentation. This project does not add an extra dropout layer. | Official pinned `pytorch/models.py` and wrapper |
| Tuning | Controlled **freezing-depth ablation** (`head`, `fc1`, `block6`); **no LR grid-search claim** | Three run configurations |

**Checkpoint note:** Selected `best.pt` contains trainable parameters only; restore the original AudioSet backbone first. A ten-fold mean was *not* measured.
