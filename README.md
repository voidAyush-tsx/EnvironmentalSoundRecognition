# Deep Learning Framework for Environmental Sound Recognition Using PyTorch

**Dataset:** UrbanSound8K · **Backbone:** official AudioSet-pretrained PANNs CNN14 · **Framework:** PyTorch  
**Task:** Single-label classification of ten urban/environmental sound classes  
**Project status (23 October 2026):** Stage 2 verification complete; Stage 3 implementation prepared and **awaiting complete execution on the local dataset**. Model training, fine-tuning, evaluation, CO4 comparisons, and CO5 ablations are **not yet complete**.

> This is the living README for the current **UrbanSound8K/PANNs CNN14** project. Earlier ESC-50/SimpleCNN notebook experiments are historical and **must not** be reported as results of this project.

## Student and submission details

| Field | Value |
| --- | --- |
| Student name | **[To be supplied]** |
| Registration number | **[To be supplied]** |
| Section | **[To be supplied]** |
| Course | Deep Learning (third-year B.Tech CSE) |
| Assessed learning outcomes | CO3 (10 marks), CO4 (5 marks), CO5 (5 marks) |
| ZIP name required by teacher | `RegistrationNumber.zip` (replace with actual registration number only) |

The teacher's original initial-evaluation deadline in the supplied assignment was **14 September 2026**; confirm any subsequently revised deadline separately. The final package must include source, reproducibility information, required results, figures, comparisons, ablations, README and requirements. Do not submit a verification-only project as a completed trained system.

## 1. Project objective

Build a reproducible PyTorch pipeline that takes a real urban sound recording, processes it into a **32 kHz mono waveform**, passes the waveform into the **official AudioSet-pretrained PANNs CNN14**, and predicts one of UrbanSound8K's ten sound classes using an adapted **2,048 → 10** classifier.

Planned transfer-learning experiments:

1. **Frozen-feature baseline:** pretrained CNN14 backbone frozen; train only the new classifier.
2. **Partial fine-tuning:** freeze early feature-extraction layers; train selected later layers and the classifier, with appropriate BatchNorm handling and lower learning rates on pretrained weights.
3. **Optional full fine-tuning:** only if GPU memory permits; use the same folds and processing when comparing approaches.

These are **planned experiments**, not completed results. We will use `CrossEntropyLoss` on ten logits and evaluate unseen folds using the teacher-required metrics. No architecture will be substituted for CNN14.

## 2. Dataset and label mapping

**Official source:** https://urbansounddataset.weebly.com/urbansound8k.html  
**Local expected path:** `data/UrbanSound8K/`  
**Metadata:** `data/UrbanSound8K/metadata/UrbanSound8K.csv`

UrbanSound8K contains **8,732 WAV files**, **10 predefined folds**, **10 classes**, and metadata with `slice_file_name`, `fsID`, `start`, `end`, `salience`, `fold`, `classID`, `class`.

The user's executed `check_urbansound8k.py` output confirmed:

- 8,732 metadata entries and 8,732 `.wav` files across folders `fold1`–`fold10`.
- All required WAV paths existed; no invalid class IDs or fold values were found in the metadata check.
- **This metadata/file-existence test does not establish that all audio decodes correctly.** That is the purpose of Stage 3's full audio audit.

### Official label IDs — never alphabetically reindex

| classID | Sound class (metadata label) |
| ---: | --- |
| 0 | `air_conditioner` |
| 1 | `car_horn` |
| 2 | `children_playing` |
| 3 | `dog_bark` |
| 4 | `drilling` |
| 5 | `engine_idling` |
| 6 | `gun_shot` |
| 7 | `jackhammer` |
| 8 | `siren` |
| 9 | `street_music` |

### Observed official fold counts

| Fold | WAV clips | Fold | WAV clips |
| ---: | ---: | ---: | ---: |
| 1 | 873 | 6 | 823 |
| 2 | 888 | 7 | 838 |
| 3 | 925 | 8 | 806 |
| 4 | 990 | 9 | 816 |
| 5 | 936 | 10 | 837 |

The counts sum to 8,732.

## 3. Evaluation fold methodology

Preserve UrbanSound8K's **predefined folds**. For every outer fold `f` in `1..10`:

- **Test:** fold `f` (never used for parameter, epoch or hyperparameter selection).
- **Validation:** fold `(f % 10) + 1`.
- **Training:** the remaining eight official folds.

Example: test `fold1`, validation `fold2`, training `fold3`–`fold10` = **873 test**, **888 validation**, **6,971 training**, **8,732 total**. The protocol repeats until every fold is a test fold once. Model initialization, checkpoint selection and hyperparameter selection must be confined to the appropriate training/validation subset of each outer run.

**Protocol caveat:** many published UrbanSound8K benchmarks train on nine folds and test on one. Our eight-train/one-validation/one-test protocol is a deliberate validation-based variation; do **not** claim direct numerical comparability with nine-train/one-test published results.

### Source-recording overlap warning (open)

The user's executed `src/check_folds.py` passed the **clip-level** fold partition checks but reported **five original Freesound source IDs (`fsID`) present in more than one official fold**:

| fsID | Folds | Clips | Pairs of overlapping `start`/`end` intervals across folds |
| ---: | :---: | ---: | ---: |
| 77751 | 2, 7 | 30 | 31 |
| 106905 | 1, 8 | 7 | 0 |
| 132016 | 1, 4 | 15 | 0 |
| 176638 | 1, 4 | 5 | 1 |
| 180937 | 1, 9 | 95 | 57 |

These counts are **pairs of potentially overlapping time ranges**, not confirmed identical WAVs. They mean perfect source independence has **not** been established. Do not reshuffle or delete examples to force the warning away. Preserve the official benchmark folds, retain the CSV audit files, inspect affected recordings and describe the limitation in evaluation/reporting. The warning remains open even if all code checks pass.

## 4. Local environment and dependencies

The following details are based on the user's submitted **Windows command outputs**, not the ChatGPT execution environment:

| Item | Locally observed value |
| --- | --- |
| OS | Windows 11 / Windows NT version `10.0.26300.9711` |
| Python | `3.13.15` (64-bit) |
| PyTorch | `2.7.0+cu128` |
| CUDA build reported by PyTorch | `12.8` |
| CUDA available | `True` |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | `4.0 GiB`, reported during checkpoint verification |
| Dependency installation | `python -m pip install -r requirements.txt` completed |
| Dependency conflicts | `python -m pip check` → `No broken requirements found` |
| Exact-version snapshot | `requirements.lock.txt` generated by `finalize_requirements.py` |

Use the **existing working CUDA build**. Do not replace it with a CPU-only PyTorch installation. A fresh virtual environment needs its own appropriate CUDA PyTorch installation; a generic `pip install torch` alone does not guarantee the desired backend.

- `requirements.txt` is the current compatibility-range requirements file.
- `requirements.lock.txt` records actual package versions on the local PC; use it alongside the Python version and PANNs source commit.
- Core packages: PyTorch, torchlibrosa, NumPy, SciPy, librosa, soundfile, pandas, scikit-learn, Matplotlib, seaborn, tqdm, ipykernel.

**Optional first-time installation (only on a suitably configured machine):**

```bat
cd /d D:\EnvironmentalSoundRecognition
python -m pip install -r requirements.txt
python -m pip check
```

## 5. Official model source, checkpoint and verification history

**Official implementation:** https://github.com/qiuqiangkong/audioset_tagging_cnn  
**Locally recorded Git commit:** `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`  
**Published checkpoint archive:** https://zenodo.org/records/3576403  
**Original filename:** `checkpoints/Cnn14_mAP=0.431.pth`  
**Published MD5 and measured local MD5:** `595633ac2d1cac7ef04ebf70e2fee4e4`

### Why two checkpoint files exist

The original checkpoint was created with an older PyTorch serialization format. Modern restricted `torch.load(..., weights_only=True)` rejected legacy NumPy pickle metadata. This was **not a failed architecture compatibility test**. After confirming the official-source download and MD5, the checkpoint was read **once** using an explicitly trusted legacy loader and only the tensor state dictionary was exported as:

`checkpoints/Cnn14_AudioSet_state_dict_safe.pth`

The exported tensor-only copy **passed restricted reload**. Future checks use `--use-safe-weights`, and **must not routinely load the original checkpoint with `weights_only=False`**. MD5 is an integrity comparison against the source's published value, not a cryptographic security guarantee for arbitrary files.

### Verified local CNN14 forward-pass evidence

The user executed `src/check_cnn14.py` with original trusted checkpoint and safe-copy export, then with safe weights on both CUDA and CPU. All three runs reported `PASS`:

| Verified property | Observed output |
| --- | --- |
| Strict checkpoint load | PASS; no missing/unexpected weight keys |
| State dictionary entries | 84 parameter/buffer keys |
| Total model parameters, backbone + new head | **81,857,561** |
| New 10-class classification head parameters | **20,490** |
| Input waveform | `[1, 128000]` |
| Original AudioSet output | `[1, 527]` |
| CNN14 embedding | `[1, 2048]` |
| New UrbanSound8K logits | `[1, 10]` |
| GPU synthetic forward pass | PASS |
| CPU synthetic forward pass | PASS |
| UrbanSound8K training | **Not performed** |

A NumPy `VisibleDeprecationWarning` was observed during legacy deserialization, but did not prevent load or inference. The **10-class head is randomly initialized**. The verification proves model integration and tensor shapes, not classification accuracy or successful fine-tuning.

## 6. Stage 3 waveform preprocessing (prepared; local full audit pending)

The Stage 3 implementation is supplied in `src/dataset.py`, `src/audit_audio.py`, `src/check_dataloader.py`, and `tests/test_dataset.py`.

| Property | Implementation |
| --- | --- |
| Input representation | Real WAV audio → raw waveform, not an externally prepared Mel spectrogram |
| Channels | Convert to mono by averaging source channels |
| Target sampling rate | **32,000 Hz**, using `librosa` / `soxr_hq` resampling when necessary |
| Target duration | **4 seconds** (`128000` samples) |
| Short clips | Right zero-padding |
| Long clips | Deterministic center crop |
| Waveform dtype | `torch.float32` |
| Labels | `torch.int64`, official class IDs 0–9 |
| Waveform amplitude normalization | None added |
| Data augmentation | None for the initial pipeline |
| PANNs feature extraction | Internal 64-bin log-Mel frontend, not independently fed to CNN14 |
| DataLoader batch | `[batch_size, 128000]` waveforms |
| Windows-friendly workers | Initially `num_workers=0` |

Official CNN14 frontend constructor settings: `sample_rate=32000`, `window_size=1024`, `hop_size=320`, `mel_bins=64`, `fmin=50`, `fmax=14000`. The notebook can visualize log-Mel features, but the model itself takes raw waveforms.

**Important limitation:** center cropping a clip longer than four seconds could discard events at its boundaries. The full audio audit records duration distribution and counts longer-than-4-second clips, so this preprocessing choice should be revisited if the observed distribution makes it problematic.

## 7. Re-run all Stage 2 and Stage 3 checks with one command

### Prerequisites

Extract `Stage3_UrbanSound8K_DataPipeline.zip` **into the project root**, preserving its `src/` and `tests/` directories. Keep the old `src/check_cnn14.py` and `src/check_folds.py`, your original checkpoint, and the safe exported state dictionary. Put `run_all_checks.py` and this README in the project root.

For the local folder specified throughout the project:

```bat
cd /d D:\EnvironmentalSoundRecognition
python run_all_checks.py
```

That is the **single command to repeat all available verification**. On a fresh project checkout, the runner first restores only the missing, small official PANNs source repository (Git required); it does **not** reinstall existing packages, re-download the 1.4 GB checkpoint, or download UrbanSound8K. An incomplete or conflicting existing upstream directory is never overwritten automatically. It will:

1. If the small external PANNs source repository is **absent**, automatically clone it and check out commit `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4` (the revision previously tested locally). Do not replace an existing repository. Then check remaining prerequisites, including the existing dataset and both checkpoint files, and confirm the original Zenodo MD5.
2. Execute `python -m pip check` and `python check_dependencies.py` (imports, versions, GPU tensor computation).
3. Execute `python check_urbansound8k.py` (metadata rows, paths, folds and class IDs).
4. Record the current PANNs Git commit.
5. Execute `python src/check_cnn14.py --use-safe-weights` and the corresponding `--cpu` version.
6. Execute `python src/check_folds.py` (8/1/1 folds, disjoint clips, documented source-ID warnings).
7. Execute `python -m unittest discover -s tests -v` (synthetic preprocessing tests).
8. Execute `python -m src.audit_audio --max-files 50` (pilot only).
9. Execute `python -m src.audit_audio` (**full WAV decoding of all 8,732 files**).
10. Execute `python -m src.check_dataloader` (real batch shapes and labels).
11. Execute `python -m src.check_dataloader --model-forward` (real WAV through CNN14, untrained head).
12. Refresh `requirements.lock.txt` with `python finalize_requirements.py`.

The script **stops on the first failed check**, prints the failing command and its traceback, and saves a timestamped console transcript and JSON summary in `results/verification_runs/`. Earlier generated statistics under `results/` may be overwritten by the rerun; keep a copy when comparing separate versions of preprocessing. The source-ID warning is retained in the summary instead of being silently ignored.

### Useful one-command variations

```bat
python run_all_checks.py --quick
```

Runs the prerequisite, model, fold, unit-test, 50-file pilot and loader checks **without** a full dataset decode. This **cannot** be claimed as completed Stage 3 audio integrity verification.

```bat
python run_all_checks.py --skip-cpu
```

Runs the full checks but skips the additional CPU CNN14 inference (the regular GPU test remains).

```bat
python run_all_checks.py --setup
```

**Opt-in pip installation:** installs `requirements.txt` first (unnecessary when your existing dependencies already work); the runner independently clones the small PANNs source repository if missing. Use cautiously because pip can modify the currently selected Python environment. This option **does not download the large UrbanSound8K dataset or checkpoints**. It also does not use unrestricted pickle; if the safe checkpoint is missing, first reproduce its verified export from the trusted original as described above.

```bat
python run_all_checks.py --offline --quick
```

Runs without cloning any missing PANNs source. This will fail a prerequisite check if the repo is absent, rather than making a network request.

```bat
python run_all_checks.py --dry-run
```

Shows the commands it would execute without changing files or running any checks.

**Correct execution:** run commands from the project root with the same Python interpreter that reports CUDA support (`python -c "import torch; print(torch.cuda.is_available())"`).

### What constitutes Stage 3 passing?

- Local unit tests pass.
- Full audit reports exactly 8,732 **successfully decoded** recordings and **zero** decode/validity errors (`results/dataset_integrity_report.json` has `"full_audio_audit": true`).
- Dataset statistics, waveform/log-Mel figures and distribution files are generated.
- Real train/val/test DataLoaders produce the expected waveform and label tensors.
- A real audio waveform successfully passes through the strict-loaded pretrained CNN14 and untrained 10-class head.

Passing these checks **does not resolve cross-fold source-ID dependencies** and **does not demonstrate supervised classification performance**.

## 8. Folder layout

```text
EnvironmentalSoundRecognition/
├── README.md                            # This project log and reproduction guide
├── requirements.txt                    # Compatibility constraints
├── requirements.lock.txt               # Exact versions captured on local PC
├── run_all_checks.py                   # One-command Stage 2 + Stage 3 runner
├── check_urbansound8k.py               # Original dataset metadata/path audit
├── check_dependencies.py               # Dependency imports and CUDA tensor test
├── finalize_requirements.py            # Regenerate local dependency lock
├── data/
│   └── UrbanSound8K/
│       ├── audio/fold1 ... fold10/      # Original WAV files; keep folds intact
│       └── metadata/UrbanSound8K.csv   # Official CSV, do not alter
├── checkpoints/
│   ├── Cnn14_mAP=0.431.pth             # Verified original AudioSet checkpoint
│   └── Cnn14_AudioSet_state_dict_safe.pth  # Tensor-only exported copy
├── external/
│   └── audioset_tagging_cnn/           # Official source pinned by Git commit
├── src/
│   ├── __init__.py
│   ├── check_cnn14.py                  # Strict load and 10-logit forward test
│   ├── check_folds.py                  # Ten-fold and source-ID checks
│   ├── dataset.py                      # Waveform preprocessing + Dataset/DataLoader
│   ├── audit_audio.py                  # Full WAV decode audit and figures
│   └── check_dataloader.py             # Real batch and CNN14 smoke tests
├── tests/
│   └── test_dataset.py                 # Unit tests
├── results/
│   ├── fold_split_summary.csv
│   ├── cross_fold_source_summary.csv
│   ├── cross_fold_source_details.csv
│   ├── dataset_integrity_report.json  # After Stage 3 audit
│   ├── audio_file_statistics.csv       # After Stage 3 audit
│   └── verification_runs/              # One log + JSON summary per runner invocation
├── figures/                            # Created by Stage 3 audit
└── notebooks/                          # Future NEW UrbanSound8K/CNN14 notebook
```

Large raw datasets and pretrained checkpoints need not be placed in the final teacher ZIP unless explicitly required. Provide trusted acquisition instructions, the original checkpoint filename and hash, and the exact source commit in the final reproducibility documentation. Do not include the old ESC-50 experimental notebook in the current evaluation as if it were evidence for this implementation.

## 9. Execution history — what's been done and why

| Stage | Check / execution | Evidence status |
| --- | --- | --- |
| 1 | Read the teacher's CO3/CO4/CO5 rubric and identified ZIP, README and metrics requirements | **Requirements audited** |
| 1 | Inspected historical ESC-50 notebook; established migration to UrbanSound8K + pretrained CNN14 | **Scope clarified; historical results excluded** |
| 2 | Obtained UrbanSound8K from the official dataset and extracted fold directories | **Completed locally** |
| 2 | Checked 8,732 metadata rows, 10 labels/folds, all WAV references | **PASS (user console output)** |
| 2 | Inspected Python, PyTorch CUDA 12.8, RTX 3050 detection | **PASS (user console output)** |
| 2 | Installed dependencies and checked pip constraints; generated local lockfile | **PASS (`pip check` and lockfile generation)** |
| 2 | Cloned official PANNs source and recorded exact Git revision | **PASS (user console output)** |
| 2 | Downloaded original CNN14 model; checked Zenodo MD5 | **PASS (hash matches)** |
| 2 | Debugged incorrect repository utility path in first verification script | **Resolved** |
| 2 | Debugged legacy NumPy pickle under restricted PyTorch loading | **Resolved via trusted one-time load and safe tensor-only export** |
| 2 | Strict checkpoint load; CPU/GPU synthetic waveform → 2048 embedding → ten logits | **PASS (user console output)** |
| 2 | Verified all ten 8/1/1 partitions and generated CSV audits | **PASS at clip level; source-overlap WARNING open** |
| 3 | Implemented Stage 3 dataset/audit/DataLoader scripts and synthetic tests | **Files prepared; synthetic tests passed in development environment** |
| 3 | Run tests and complete 8,732-file WAV audit on student's PC | **Pending local verification** |
| 3 | Run real-audio CNN14 forward check on student's PC | **Pending local verification** |
| 4–9 | Genuine frozen/partial fine-tuning, real scores, CO4/CO5 evaluations | **Not started / no claims of results** |
| 10–11 | Final reproducibility audit and registration-number ZIP | **Pending** |

## 10. Planned training and hyperparameters (not final results)

The first candidate configuration is for a **pilot only**, subject to measured VRAM use and validation evidence:

| Setting | Initial candidate / rule |
| --- | --- |
| Backbone | Official CNN14, AudioSet-pretrained |
| Output | Ten logits, official UrbanSound8K labels |
| Loss | `torch.nn.CrossEntropyLoss` |
| Optimizer | AdamW |
| Classifier learning rate | Approximately `1e-3` to begin testing |
| Unfrozen pretrained-layer learning rate | Approximately `1e-5` to begin testing |
| Weight decay | Approximately `1e-4` to begin testing |
| Batch size | Start small due to **4 GiB VRAM**, determine empirically |
| Epochs | Pilot first; later validation-based early stopping |
| Scheduler | Validation-based scheduler if needed; not yet selected |
| Augmentation | None initially; optional training-only controlled ablation |
| Dropout | Standard backbone behavior plus any documented head changes; not yet finalized |
| Frozen layers and BatchNorm | To be verified explicitly; no fine-tuning implemented yet |

**Hyperparameter tuning:** not performed for this CNN14/UrbanSound8K implementation. The final README must replace these candidate values with the values actually selected on validation folds and cite `results/hyperparameter_results.csv`.

## 11. Results, CO3, CO4 and CO5 requirements — pending

- **CO3 (implementation and application):** Build and train a real ten-class CNN14 adaptation; save train/validation histories, selected checkpoints, fold test predictions, overall/per-class accuracy, macro/weighted precision/recall/F1, confusion matrix, specificity and defensible multiclass ROC-AUC from probabilities, runtime/complexity evidence. Generate figures/CSVs from actual runs. **Not yet completed.**
- **CO4 (published comparison):** Research approximately 3–5 verified **UrbanSound8K** published results. Required fields include `Model | Accuracy | Precision | Recall | F1 | AUC | Parameters` with citations and clear warnings about differing pretraining/splits/augmentation. **Not yet completed.** Historical ESC-50 literature is not a substitute.
- **CO5 (experimental contribution):** Evaluate **frozen-feature extraction vs selective unfreezing** on matched folds; compare trainable parameter counts, accuracy, macro-F1, generalization gaps, compute cost and cross-fold stability. Report negative findings honestly. **Not yet completed.**
- **Bias–variance and generalization:** interpret actual learning curves and train/validation gaps; fold-to-fold variability is empirical robustness, not the mathematical variance term from a formal bias–variance decomposition. **Awaiting training data.**
- **Reproducibility:** save seeds, dependencies, Git source commit, preprocessing settings, data splits, selected checkpoint identifiers, runs, output CSVs and figures. **Partially established through verification scripts.**

There are **no UrbanSound8K training accuracies, loss curves, ablation metrics, trained 10-class checkpoints, or comparisons to report as completed yet.**

## 12. Troubleshooting history

| Symptom | Cause and resolution |
| --- | --- |
| `Missing the official PANNs source` after a successful clone | First script expected a wrong `utils/pytorch_utils.py` location; corrected to the repository's `pytorch/pytorch_utils.py`. |
| `_pickle.UnpicklingError: Weights only load failed` | Original legacy checkpoint includes NumPy metadata. After verifying official source and hash, performed explicitly trusted one-time legacy loading and exported safe tensor-only state dictionary. |
| `5 fsID values span multiple folds` | A diagnostic assumption of strict source isolation failed on official metadata. Subsequent script **reports** this instead of changing the benchmark folds. Concern remains open. |
| CUDA Out Of Memory during later model development | First check actual VRAM and competing GPU processes; use smaller batches, frozen-feature baseline and tested precision settings where appropriate. Do not claim a training run passed if it failed. |
| `ModuleNotFoundError: src...` | Run `python -m src...` from the **project root**, not from within `src/`; extract Stage 3 ZIP preserving `src/__init__.py`. |

## 13. References and provenance

- UrbanSound8K official dataset: https://urbansounddataset.weebly.com/urbansound8k.html
- Official PANNs implementation: https://github.com/qiuqiangkong/audioset_tagging_cnn
- Official CNN14 checkpoint record: https://zenodo.org/records/3576403
- PyTorch checkpoint security / `torch.load`: https://docs.pytorch.org/docs/stable/generated/torch.load.html
- Course assignment: `project_submission_instructions` (provided as part of the project context; preserve your original copy outside the submission if needed).

---

**Next milestone:** Execute `python run_all_checks.py` on the student PC, review the full audio integrity report and unresolved cross-fold source dependencies, then proceed to **Stage 4: genuine partial freezing and CNN14 fine-tuning** only after real-audio inputs pass all required checks.
