# CO5 — Controlled CNN14 fine-tuning-depth study

## Contribution and what is *not* claimed

The contribution is an **experimental evaluation of fine-tuning depth and computational trade-offs** for AudioSet-pretrained PANNs CNN14 on UrbanSound8K. It is **not** an original neural-network architecture, new optimizer, new algorithm, or a claim to parameter-efficient transfer learning. The study tests whether adapting a later convolutional block yields an improvement over freezing the pretrained extractor.

## Controlled setup

The uploaded `config.json` and `history.csv` files confirm: official folds 3–10 train (6,971 clips), fold 2 validation (888), test fold 1 reserved (873), seed 67, BF16 CUDA, 32 kHz 4-second mono waveforms, batch size 1, gradient accumulation 4, 6 requested and completed epochs, patience 2, head learning rate 0.001, backbone learning rate 0.00001, weight decay 0.0001, and checkpoint selection by validation macro-F1. Trainable parameters alone were passed to the optimizer. Frozen layers and pretrained BatchNorm running stats were kept fixed. `conv_block6` received nonzero gradients and changed convolution weights in the real CUDA one-step test and full training history.

| Configuration | Trainable layers | Trainable / total params | Best val accuracy | Best val macro-F1 | Selected epoch | Six-epoch training time | Peak CUDA allocated |
|---|---|---:|---:|---:|---:|---:|---:|
| A — Frozen-feature baseline | New 10-class head | 20,490 / 81,857,561 | 87.73% | 0.8638 | 4 | 19.80 min | 474.2 MiB |
| B — Dense adaptation | `fc1` + head | 4,216,842 / 81,857,561 | 88.06% | 0.8668 | 4 | 18.87 min | 522.3 MiB |
| C — Partial convolutional fine-tuning | `conv_block6` + `fc1` + head | 60,848,138 / 81,857,561 | **90.65%** | **0.9010** | 4 | 28.65 min | 1,278.1 MiB |

Compared with A, C improved validation accuracy by **2.93 percentage points**, macro-F1 by **0.0372**, and raised the six-epoch GPU training time from 19.80 to 28.65 minutes (~1.45×). Its peak GPU allocation was ~2.70× A. Relative to B, C improved validation accuracy by **2.59 percentage points** and macro-F1 by **0.0341**. The B vs A difference is small (0.34 percentage points accuracy; 0.0031 macro-F1).

## Test and generalization

The C configuration was selected **using fold 2 validation only**, then evaluated **once** on held-out fold 1: accuracy **88.55%**, macro-F1 **0.8964**, macro ROC-AUC **0.9909**, 873 recordings. The held-out test was *not* used to select A, B, or C. **A and B have no held-out test scores in the available evidence** and must not be assigned any.

The C validation macro-F1 selected epoch 4 (0.9010); the final epoch-6 validation macro-F1 was 0.8847, and validation loss increased from 0.3433 at epoch 4 to 0.3986 at epoch 6. Thus **checkpoint selection prevented reporting the worse final epoch**. Training accuracy increased while validation performance worsened after epoch 4, suggesting some overfitting/optimization instability; only one seed and one held-out test fold were run.

## Limitations and future controlled investigations

- **Single test fold, single seed:** no estimate of ten-fold mean/standard deviation or multi-seed stability; this is validation evidence, not a new SOTA claim.
- **Five `fsID` values cross official folds**, including potentially overlapping time segments. Report the source-isolation caveat alongside the disjoint clip-level split.
- A prior train-only sample gave zero `fc1` and `conv_block6` gradients because the input to `fc1` was completely zero; other samples and all full-training epochs showed nonzero updates. Keep that exception in the engineering notes.
- **Unfreezing 60.85 million parameters is costly**; the study improves validation metrics at nontrivial training and storage cost.
- For stronger CO5 claims, evaluate all ten prescribed test folds, record variance, optionally repeat seeds, and compare a smaller unfreezing depth or frozen-BN policy under matched conditions. Such experiments are **not already completed**.

**Raw source artifacts:** `reports/faculty_evidence/training_runs/{head,fc1,block6}/{config.json,history.csv,summary.json,learning_curves.png}` and `reports/faculty_evidence/heldout/evaluation_summary.json`.
