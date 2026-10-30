# Stage 4 results → Stage 5 held-out evaluation (notebook-first)

**Project:** Deep Learning Framework for Environmental Sound Recognition Using PyTorch

**Evidence reviewed:** `Stage4_Three_Full_Runs_Review.zip` supplied 27 October 2026. It contained **12 small files**, not the 1.4 GB AudioSet checkpoint or the fine-tuned `best.pt` files. Three genuine (non-pilot) full-fold runs were found, each with six epochs. All runs used folds 3–10 for training, fold 2 for validation, and reserved fold 1 for testing.

## Result reviewed and decision made *before* testing

| Stage 4 config | Validation accuracy | Validation macro-F1 | Best epoch | Trainable params |
|---|---:|---:|---:|---:|
| `head` | 87.73% | 0.8638 | 4 | 20,490 |
| `fc1` | 88.06% | 0.8668 | 4 | 4,216,842 |
| `block6` | **90.65%** | **0.9010** | **4** | 60,848,138 |

**Selection decision:** evaluate the **validation-selected `block6` checkpoint at epoch 4**, once, on test fold **1**. This is a *single-held-out-fold* result. Do not call it a 10-fold benchmark mean or use the test fold to choose another model. The 5 cross-fold Freesound IDs are a known source-isolation limitation.

## Setup

1. Unzip this package **at the existing project root** `D:\EnvironmentalSoundRecognition`. It adds a companion notebook (`notebooks/02_...ipynb`), `src/evaluate_selected.py`, `tests/test_stage5_eval.py`, and `reports/stage4_comparison/`. It **does not overwrite** your existing notebook `01_...ipynb`, Stage 4 trainer, trained checkpoints, data, or root README.
2. In VS Code, open `notebooks/02_Stage4_Review_and_Stage5_HeldOut_Evaluation.ipynb`, select your existing Python 3.13 kernel (`torch==2.7.0+cu128`), and run all with the default controls (**held-out test disabled**). The figures and three-run comparison should display.
3. Run `python -m unittest discover -s tests -v` **only if you want a full unit-test rerun**. The companion notebook itself is the main control surface; it does not force command-line use.
4. In the notebook's **Section 6**, locate the exact full `block6` Stage 4 run directory. It must contain `config.json`, `summary.json`, `history.csv`, `best.pt`. The earlier ZIP did not contain this path or checkpoint. If several matches appear, confirm which was your reported six-epoch run.
5. In the **controls cell**, set `SELECTED_BLOCK6_RUN_DIR = r"D:\EnvironmentalSoundRecognition\results\stage4\<exact_run_folder>"`, then `RUN_HELDOUT_TEST = True`. Run controls and **Section 7 once**. It computes the actual 873-sample fold-1 test metrics and saves them to `results/stage5/<timestamp>_block6_test1/` with a full notebook log in `results/notebook_evidence/`.
6. Reset `RUN_HELDOUT_TEST = False`, run **Section 8** to display metrics/plots, set `UPDATE_ROOT_README = True` and run **Section 9** to synchronize a clearly marked current evidence section *without deleting your older README notes*. Reset `UPDATE_ROOT_README = False`, and save the notebook with **Ctrl+S**.

### Evaluator outputs after an actual successful run

- `evaluation_summary.json` — test accuracy, cross-entropy, macro/weighted precision, recall and F1, macro OvR specificity and ROC-AUC, fold and checkpoint provenance.
- `per_class_metrics.csv` — 10 rows, official classIDs 0–9, support, precision, recall, F1, one-vs-rest specificity/ROC-AUC and TP/FP/FN/TN.
- `test_predictions.csv` — 873 filenames, true/predicted classIDs, confidence and **ten class probabilities** for traceability.
- `confusion_matrix.csv`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `per_class_f1.png`, `roc_curves_ovr.png`.

**Never use `best.pt` alone as a full model checkpoint:** it stores only trainable parameters from Stage 4. The evaluator first loads `checkpoints/Cnn14_AudioSet_state_dict_safe.pth` through the official CNN14 source at commit `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`, then applies the selected trainable state using the Stage 4 helper. The loader uses `weights_only=True`.

## Verification and limitations

- From the uploaded ZIP, we verified exact fold/seed/batch/epoch consistency, selection basis, non-pilot flags and absence of test evaluation.
- `src/evaluate_selected.py` passes four **synthetic metric/guard tests** and the notebook safe-default path executes in a test environment. **No actual user test WAVs or trained checkpoint were available in this sandbox**, so no Stage 5 test performance is claimed.
- Training-time online accuracy has augmentation/dropout enabled, while validation is in evaluation mode. Do **not** calculate a naive generalization-gap claim by subtracting these two accuracies.
- The existing project `.gitignore` excludes `.pyc`, `__pycache__/`, `results/`, `checkpoints/`, and WAV data. The small reviewed comparison figures/CSV are placed in `reports/` so they can be deliberately committed to Git.

## After Stage 5

Next, complete the CO4 comparison using genuine published UrbanSound8K results with original citations and protocol notes, and package the reproducible notebook, source, data access instructions, metrics, and evidence per the teacher's assignment. Additional fold/seed experiments can be presented as subsequent controlled extensions, not retroactive tuning based on the final test fold.
