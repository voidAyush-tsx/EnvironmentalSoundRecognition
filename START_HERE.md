# Stage 6–8: CO4 + CO5 + faculty submission (notebook-first)

**Current project:** Deep Learning Framework for Environmental Sound Recognition Using PyTorch — UrbanSound8K, official PANNs CNN14, PyTorch.

## Install safely

Extract this ZIP into the existing `D:\EnvironmentalSoundRecognition` folder. **Do not delete, recreate, or replace** your Stage 2–5 source, notebooks 01/02, original dataset/checkpoints, or existing root `README.md`. This kit adds **Notebook 03**, `docs/`, `tools/`, `models/CHECKPOINT_ACCESS.md`, `data/README_DATASET.md`, and small copied evidence under `reports/faculty_evidence/`.

The original already executed notebooks 01/02 and trained local checkpoint are required for a complete demonstration. The ZIP provides small **review copies** of the training configurations, histories and held-out metrics that the student previously uploaded; it does not contain the 1.4-GB pretrained backbone or the original WAVs.

## Run from Jupyter in VS Code

1. Open the project folder and `notebooks/03_CO4_CO5_Faculty_Submission.ipynb`.
2. Select the **same** Python 3.13 / torch 2.7.0+cu128 kernel used for actual training.
3. **Restart & Run All**: safe default displays the existing actual evidence, CO4 published papers and CO5 ablations; does not rerun test, start training, or build ZIP.
4. Review `docs/CO4_PUBLISHED_METHODS.md`, `docs/CO5_ABLATION_AND_DISCUSSION.md` and `docs/TRAINING_DETAILS_TO_VERIFY.md`; inspect Notebook 03 tables/figures and **verify the scheduler/dropout details against your local training code** (not fully recorded in uploaded JSON).
5. In Notebook 03 Controls, set `UPDATE_ROOT_README=True` and run the README sync cell **once**. Then set it back `False`. This preserves the existing README and only replaces the managed faculty evidence section.
6. In Controls set `RUN_FACULTY_AUDIT=True`; run its audit cell. Check missing files against `docs/FACULTY_SUBMISSION_CHECKLIST.md`.
7. Enter **your real** `STUDENT_NAME`, `REGISTRATION_NUMBER`, and `SECTION` in Controls. These fields were not supplied and must not be invented.
8. When audit is clean and the local `best.pt` checkpoint exists at the evaluated run, set `CREATE_FINAL_SUBMISSION_ZIP=True` and run the package cell. Reset to False afterward. The package is named `<registration number>.zip` with no project/model/student name in filename.
9. Save Notebook 03 with `Ctrl+S`; reopen the actual output ZIP and check its content.

## Evidence included and academic status

- Full-fold validation: head **87.73% / F1 0.8638**; fc1 **88.06% / 0.8668**; block6 **90.65% / 0.9010**; all best at epoch 4.
- One held-out fold-1 test (selected block6): **88.55%** accuracy, **0.8964** macro-F1, **0.9909** OvR macro ROC-AUC; 873 samples, 773 correct.
- Published same-dataset references (separate protocols): Su et al. 2019 **97.2%**; Ting et al. 2021 **83.03%** mean; Fang et al. 2022 **97.51 ± 0.18%**. These are **published** numbers, not rerun scores.
- Teacher's CO4 original-paper comparison is documented, but a **fair accuracy leaderboard requires 10-fold evaluation**. The UrbanSound8K dataset explicitly recommends this; only one test fold is completed.

## Requirements and Git

Existing `requirements.txt` and `requirements.lock.txt` must remain the versions verified on your PC. `docs/GITIGNORE_APPEND.txt` contains safe additions for `__pycache__`, large data, weights, and output archives (do not overwrite the existing `.gitignore`).

## What's still a real action before calling it submitted?

Student identity fields; execute Notebook 03 on actual Windows project; README merge; local faculty audit; hash of the selected `best.pt`; packaging and ZIP inspection; ideally 10-fold robustness if the faculty expects benchmark-level cross-study claims. **No new experiment is silently assumed complete.**
