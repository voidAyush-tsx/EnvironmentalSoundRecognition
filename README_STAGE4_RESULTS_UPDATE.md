# Root README update — 27 October 2026

**Do not blindly overwrite your existing root README:** it may contain the execution diary from notebook 01. The new notebook's **Section 9** inserts this same evidence between bounded markers into your existing `README.md`, preserving everything else.

- Stages 1–3: UrbanSound8K, official pretrained PANNs CNN14, safe checkpoint, folds, audio/data pipeline and Windows CUDA checks completed (documented in notebook 01).
- Stage 4 real one-step backward checks and diagnostic pilots passed.
- Stage 4 full training: **three configurations, each six epochs** on 6,971 training samples with 888 validation samples; test fold 1 reserved (873 samples); the model is **not yet finally evaluated**.
- Validation-selected epoch 4: `head` 87.73% / macro-F1 0.8638; `fc1` 88.06% / 0.8668; **`block6` 90.65% / 0.9010**.
- CO5 frozen-depth controlled comparison: `block6` improved accuracy by **2.93 percentage points** versus head-only on this validation fold. It trains **60,848,138** parameters versus **20,490** for head-only and incurs greater training time / memory.
- Validation loss increased after epoch 4 in all three runs. Checkpoint selection by validation macro-F1 is essential.
- Full test accuracy and per-class metrics: **pending held-out Stage 5 evaluation**, not inferred from validation scores.
- Limitation: five Freesound `fsID` values appear in more than one official fold, including interval overlaps. No claims of perfect source isolation.
- Evidence: `reports/stage4_comparison/STAGE4_REVIEW.md`, `stage4_comparison.csv`, `stage4_all_epochs.csv` and saved figures. Stage 5 tools: `src/evaluate_selected.py`, `notebooks/02_Stage4_Review_and_Stage5_HeldOut_Evaluation.ipynb`.
