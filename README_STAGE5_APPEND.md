<!-- STAGE5_FACULTY_RESULTS_START -->
## Verified UrbanSound8K CNN14 results — 30 October 2026

**Project title:** Deep Learning Framework for Environmental Sound Recognition Using PyTorch  
**Student details:** [ADD STUDENT NAME, REGISTRATION NUMBER AND SECTION]  
**Dataset:** UrbanSound8K, 8,732 clips, 10 classes; fixed official folds. 
**Architecture:** Official PANNs CNN14, AudioSet-pretrained, model-source commit `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`; frozen early layers and fine-tuned `conv_block6`, `fc1`, and a 10-logit classification head. The original 527-way AudioSet head is retained for checkpoint compatibility but not trained as the UrbanSound8K classifier.

### Reproducible evaluation protocol

- Training: folds 3–10, 6,971 clips; validation: fold 2, 888 clips; held-out test: fold 1, 873 clips.
- Audio: mono, 32,000 Hz, four-second center crop or right padding (128,000 samples).
- Classifier: 10 raw logits; single-label cross-entropy.
- Experiment settings: Python 3.13.15, PyTorch 2.7.0+cu128, BF16; seed 67, batch size 1, accumulate 4; max 6 epochs; validation-macro-F1 checkpoint selection, patience 2. Head LR=0.001; pretrained trainable layers LR=0.00001; weight decay=0.0001.
- Three layer-freezing configurations used the same folds and training settings; these are controlled ablations, **not** proof of an invented new neural-network algorithm.

| Configuration | Trainable parameters | Best validation accuracy | Best validation macro-F1 | Best epoch |
|---|---:|---:|---:|---:|
| New head only | 20,490 | 87.73% | 0.8638 | 4 |
| `fc1` + head | 4,216,842 | 88.06% | 0.8668 | 4 |
| `conv_block6` + `fc1` + head | 60,848,138 | 90.65% | 0.9010 | 4 |

The proposed `block6` configuration improved accuracy by 2.93 percentage points and macro-F1 by 0.0372 relative to the head-only baseline on validation fold 2. These are **validation comparisons, not three held-out test scores**.

### One-time held-out test: selected `block6` checkpoint

| Metric | Result |
|---|---:|
| Test fold | 1 (873 clips) |
| Correct | 773 / 873 |
| Accuracy | 88.55% |
| Cross-entropy loss | 0.3800 |
| Macro precision | 0.9008 |
| Macro recall | 0.9005 |
| Macro F1 | 0.8964 |
| Weighted F1 | 0.8838 |
| Mean one-vs-rest specificity | 0.9870 |
| Macro one-vs-rest ROC-AUC | 0.9909 |

The selected checkpoint is from **epoch 4**, chosen using validation fold 2 before test evaluation. Its SHA-256 is `7c0e01a0078c225a18f4dc5eaa5110d051ef2c8effec7bcebb4e856e6659328c`. Validation accuracy was 90.65%; test accuracy was 88.55% (difference 2.11 percentage points). ROC-AUC was computed from softmax probabilities, not predicted class IDs.

### Per-class limitations

- Lowest test recall: drilling 68%, air conditioner 69%.
- Strong test recall: car horn, gunshot, siren all 100% **on this specific test fold**. Do not generalize these as perfect performance across the dataset.
- Five Freesound `fsID` values cross official folds. Potential source-level dependence is documented in `results/cross_fold_source_summary.csv`.
- This is a **single-fold held-out test**, not the official ten-fold average. Published ten-fold benchmark numbers are contextual literature comparisons and should not be treated as directly equivalent.

### Evidence paths on the user's machine

- `results/stage4/<selected-run>/config.json`, `history.csv`, `summary.json`, `learning_curves.png`, `best.pt` (for each ablation)
- `results/stage5/.../evaluation_summary.json`, `per_class_metrics.csv`, confusion-matrix figures, and predictions CSV (confirm exact generated directory)
- `results/cross_fold_source_summary.csv`; `results/verification_runs/`; saved executable notebooks

**Remaining:** CO4 published-method comparison with original citations and protocol caveats; CO5 robustness/ablation interpretation; hyperparameter selection disclosure; final notebook/README/source/package audit and registration-number-only ZIP.
<!-- STAGE5_FACULTY_RESULTS_END -->
