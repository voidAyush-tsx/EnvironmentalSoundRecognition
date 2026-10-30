# Deep Learning Framework for Environmental Sound Recognition Using PyTorch

**UrbanSound8K · AudioSet-pretrained PANNs CNN14 · PyTorch · Transfer learning**

A reproducible research project for **single-label, ten-class environmental sound classification**. The primary experiment freezes CNN14's earlier feature layers and fine-tunes **conv_block6 + fc1 + a new 10-class classifier**. The project also compares head-only and fc1-only adaptation under matched conditions.

> **Scope of evidence:** The reported test score is from **UrbanSound8K fold 1 only**, not ten-fold cross-validation. Five Freesound source IDs appear in multiple official folds. Please read [limitations](docs/CO5_ABLATION_AND_DISCUSSION.md) before comparing published numbers.

## Results at a glance

| Configuration | Trainable parameters | Best validation accuracy (fold 2) | Best validation macro-F1 |
|---|---:|---:|---:|
| Head only | 20,490 | 87.73% | 0.8638 |
| fc1 + head | 4,216,842 | 88.06% | 0.8668 |
| **conv_block6 + fc1 + head** | **60,848,138** | **90.65%** | **0.9010** |

All three runs used folds 3–10 for training (6,971 clips), fold 2 for validation (888), the same six-epoch budget, and validation macro-F1 checkpoint selection. **Only the selected block6 model** was evaluated on held-out fold 1 (873 clips):

| Held-out metric | Value |
|---|---:|
| Accuracy | **88.55%** (773/873) |
| Macro precision / recall / F1 | 0.9008 / 0.9005 / **0.8964** |
| Macro one-vs-rest ROC-AUC | **0.9909** |

[Held-out evaluation JSON](reports/faculty_evidence/heldout/evaluation_summary.json) · [Per-class metrics](reports/faculty_evidence/heldout/per_class_metrics.csv) · [Confusion matrix](reports/faculty_evidence/heldout/confusion_matrix.png) · [Learning curves and ablations](reports/stage4_comparison/STAGE4_REVIEW.md)

## Explore the project

| Start here | Purpose |
|---|---|
| [Notebook 01 — Data and model foundations](notebooks/01_UrbanSound8K_PANNs_CNN14_Project.ipynb) | Environment, dataset, preprocessing, checkpoint verification, gradient checks and training workflow |
| [Notebook 02 — Experiments and held-out evaluation](notebooks/02_Stage4_Review_and_Stage5_HeldOut_Evaluation.ipynb) | Actual learning curves, comparison and selected test-fold evaluation |
| [Notebook 03 — CO4, CO5 and faculty evidence](notebooks/03_CO4_CO5_Faculty_Submission.ipynb) | Literature comparison, contribution and submission audit |
| [Repository guide](docs/REPOSITORY_GUIDE.md) | Where every important file lives and what to read first |
| [Reproducibility guide](docs/REPRODUCIBILITY.md) | Fresh-PC setup, exact dependencies, external data and checkpoints |
| [CO4 published methods](docs/CO4_PUBLISHED_METHODS.md) | Primary-paper references and protocol differences |
| [CO5 ablation](docs/CO5_ABLATION_AND_DISCUSSION.md) | Experimental contribution, compute trade-offs and limitations |
| [Faculty checklist](docs/FACULTY_SUBMISSION_CHECKLIST.md) | Submission requirements and verified evidence |

**Suggested reading order:** this README → Notebook 02 → Notebook 01 → Notebook 03 → source/tests. Notebooks contain saved outputs; rerunning training or the held-out test is **not necessary just to review this project**.

## Method

- **Dataset:** 8,732 UrbanSound8K WAV clips, ten official classes and ten predefined folds. Mono waveforms resampled to **32 kHz**, center-cropped or right-padded to **128,000 samples**. Dataset metadata supplies official class IDs 0–9.
- **Backbone:** official [PANNs CNN14](https://github.com/qiuqiangkong/audioset_tagging_cnn), pinned revision `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`; [AudioSet checkpoint](https://zenodo.org/records/3576403). CNN14 computes log-Mel features **internally**.
- **Head:** a 2,048-to-10 linear classifier producing raw logits. The original AudioSet head remains part of the strictly loaded backbone but is frozen.
- **Optimization:** AdamW, cross-entropy, classifier LR 1e-3, trainable backbone LR 1e-5, weight decay 1e-4, batch size 1, accumulation 4, BF16 on the verified RTX 3050, seed 67. Pretrained BatchNorm running statistics remain fixed. **No LR scheduler** is used by the recorded training code. Freezing depth—not LR—was the controlled experimental variable.
- **Selection:** best validation macro-F1 (epoch 4 for each configuration). Test fold 1 was used only for evaluating the validation-selected block6 model.

Full implementation: [training](src/finetune_train.py), [model & freezing](src/finetune_model.py), [dataset](src/dataset.py), [evaluation](src/evaluate_selected.py), [verified training details](docs/TRAINING_DETAILS_TO_VERIFY.md).

## Quick start (Windows / VS Code / Jupyter)

See the [reproducibility guide](docs/REPRODUCIBILITY.md) **before running commands**, because UrbanSound8K audio and the 1.4 GB official pretrained checkpoint are not stored in Git.

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_all_checks.py --quick
```

Open the notebooks from the repository root in VS Code with the **same selected Python environment**. Their expensive steps are opt-in. Run the faculty file audit with:

```powershell
python tools/faculty_submission.py audit --project .
```

**Do not rerun the held-out test to tune hyperparameters.** To reproduce it, restore the original pretrained backbone and the separately selected trainable checkpoint as described in [checkpoint access](models/CHECKPOINT_ACCESS.md).

## Important limitations

This is a **single held-out fold**, with no ten-fold mean or confidence interval. Some source recordings overlap across predefined folds (see [source audit](results/cross_fold_source_summary.csv)); source independence is not perfect. Published UrbanSound8K results in [CO4](docs/CO4_PUBLISHED_METHODS.md) use different folds, architectures, external data and training protocols; they are **context only**, not a controlled leaderboard. The block6 approach trains ~74% of the model's parameters, so it is not parameter-efficient.

## Project provenance and submission

Faculty deliverables: **CO3 implementation/evaluation (10 marks), CO4 literature comparison (5), CO5 ablation (5)**. See [faculty checklist](docs/FACULTY_SUBMISSION_CHECKLIST.md). The faculty-specific ZIP is produced locally by [the packaging tool](tools/faculty_submission.py) using the student's actual registration number; it is separate from this lightweight public code repository.

**Data:** [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) (Salamon, Jacoby & Bello, 2014). **Model:** [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) (Kong et al.), AudioSet-pretrained weights from [Zenodo](https://zenodo.org/records/3576403). Please observe upstream licenses and attribution.

<!-- FACULTY_EVIDENCE_START -->
See the actual [CO3 metrics](reports/faculty_evidence/heldout/evaluation_summary.json), [CO4 comparison](docs/CO4_PUBLISHED_METHODS.md), [CO5 discussion](docs/CO5_ABLATION_AND_DISCUSSION.md), and [faculty audit](docs/FACULTY_SUBMISSION_CHECKLIST.md). No unmeasured ten-fold score or new CNN architecture is claimed.
<!-- FACULTY_EVIDENCE_END -->
