# Deep Learning Framework for Environmental Sound Recognition Using PyTorch

**Dataset:** UrbanSound8K · **Backbone:** official AudioSet-pretrained PANNs CNN14 · **Framework:** PyTorch  
**Task:** Single-label classification of ten urban/environmental sound classes  
**Project status (23 October 2026):** Stage 2 and Stage 3 **PASSED on the student Windows PC** (13/13 full checks; 8,732/8,732 WAV files decoded; CUDA+CPU CNN14 forward); five cross-fold `fsID` source-dependence warnings remain open. Stage 4: **14/14 unit tests PASSED on the PC; BF16 head-only real backward/update PASSED (465.9 MiB peak CUDA allocation); BF16 `fc1` one-step gradient check FAILED (`fc1.weight` norm exactly 0.0) on the first training clip**; `block6` not yet tested. Separate head-only FP32 check PASSED (721.4 MiB allocated). This revision adds a multi-clip gradient diagnostic and gradient-activity logging; its real-CNN14 Windows result is **pending**. No real UrbanSound8K fine-tuning experiment or held-out classification accuracy is complete.

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


### Historical Stage 4 numerical-precision correction (23 October 2026; head-only CUDA verification now PASSED)

The first actual Windows run of `python run_stage4_checks.py` passed all **10 original Stage 3/4 unit tests**, then stopped at the `head` one-step CUDA check:

```text
Device=cuda, config=head, AMP=True
AssertionError: Nonfinite logits
Stage 4 check status: FAIL_INCOMPLETE
```

The old Stage 4 checker and trainer both autocast the **entire official CNN14 forward pass** to CUDA `float16`. CNN14 internally uses `torchlibrosa` to calculate power spectrograms; this operation can overflow under FP16. The checkpoint is already independently confirmed loadable and usable in unautocast forward inference; this is **suspected FP16 numerical overflow**, not yet confirmed by controlled local reruns.

**Earlier precision fix (already installed in the project):** `src/precision_policy.py`, updated Stage 4 scripts and `tests/test_precision_policy.py`. The CURRENT ZIP is an incremental gradient diagnostic patch and does not need to replace that already-working precision policy. `--amp auto` selects CUDA **BF16** only when `torch.cuda.is_bf16_supported()` reports true; otherwise it uses FP32, never automatically FP16. `--amp on`/`--amp bf16` require BF16. `--amp off` forces FP32. BF16 shares FP32's exponent range; GradScaler is disabled for BF16 and FP32. Both training and validation now check finite logits and losses and gradient clipping errors on nonfinite norms. No third-party source files, model layers, initial checkpoints or dataset folds are changed.

Run from project root, after replacing the files:

```bat
python -m unittest discover -s tests -v
python run_stage4_checks.py
```

Optional one-step full-precision diagnostic when BF16 fails:

```bat
python -m src.finetune_check --config head --device cuda --amp off
```

Then, only if head/fc1/block6 one-step checks pass, run diagnostic pilots (four train/validation batches only). **The latest student run has NOT reached this prerequisite:**

```bat
python run_stage4_checks.py --pilots
```

**Verified in this assistant's CPU environment:** 19/19 combined unit tests including this diagnostic patch, syntax validation. **Not verified here:** real CNN14 `fc1`/`block6` nonzero gradients on your GPU, successful end-to-end Stage 4, pilots, full epochs, or classification accuracy; user did verify BF16 head-only update and memory as documented above. Update this record with the real Windows result; do not claim a successful Stage 4 run based on the local unit tests alone.


## 2026-10-23 — Stage 4 single-clip zero-gradient investigation (current)

The user's **actual Windows output** after the BF16 precision fix was:

| Check | Verified observation |
|---|---|
| Unit tests | 14/14 passed locally on the user's PC |
| `head` at CUDA BF16 | Finite logits and gradient; classifier parameters updated; 465.9 MiB peak allocated VRAM |
| `fc1` at CUDA BF16, train-subset first clip (classID 3) | `fc1.weight.grad` exists but its L2 norm was **0.0**; the one-step check stopped before optimizer update |
| `head` at CUDA FP32 | Finite logits, nonzero head gradient, classifier updated; 721.4 MiB peak allocated VRAM |
| `block6` | Not executed because the checker stopped at `fc1` |
| Diagnostic pilots and full training | Not completed |

The official `Cnn14.forward` applies `F.relu_` to `fc1` before emitting its embedding. Thus a single clip's `fc1.weight` gradient can be zero when its pooled input features vanish, even if the head receives a nonzero gradient from activated pretrained `fc1` biases. Other possible explanations include mixed-precision behavior and training-time dropout/SpecAugmentation. **These are hypotheses; the exact real-data cause is not yet determined.** Changing learning rates or disabling the gradient assertion would not be a legitimate fix.

### Diagnostic patch included in this revision

- `src/gradient_audit.py`: reports activation statistics and gradient states, without treating zero gradients as success.
- `src/finetune_check.py`: tests the first training clip and up to 11 additional deterministically selected **training-only** clips (including one representative per class), reports both `fc1` input and pre-ReLU activation statistics, and performs exactly **one** optimizer update **only if** all relevant trainable weight tensors have nonzero finite gradients. If zero-gradient cases occurred before a successful candidate, they remain explicit warnings. If none work, no optimizer update occurs and Stage 4 remains blocked.
- `src/finetune_train.py`: logs how many accumulated optimizer steps have nonzero classifier / `fc1` / `conv_block6.conv1` gradients, and rejects an epoch where an intended layer received none. The training fold is used for gradients; validation still determines checkpoint selection; the held-out test fold is not scored.
- `tests/test_gradient_audit.py`: regression tests for candidate selection and for the mathematically possible combination *nonzero head gradient, zero `fc1.weight` gradient*.

**Run after installing this patch:**

```bat
cd /d D:\EnvironmentalSoundRecognition
python run_stage4_checks.py
```

If `fc1` still fails after scanning the documented candidates, isolate numerical precision and augmentation **without changing official weights or architecture**:

```bat
python -m src.finetune_check --config fc1 --device cuda --amp off --max-candidates 12
python -m src.finetune_check --config fc1 --device cuda --amp auto --feature-mode eval --max-candidates 12
```

`--feature-mode eval` is **diagnostic only**: it disables the official pretrained backbone's *training-time SpecAugmentation and dropout* while retaining gradients for the selected trainable layers. Do not compare its results with normal train-mode runs as though their conditions were identical. Only continue to diagnostic pilots after **all three** real-gradient checks pass. Neither this patch nor its tests establish that the user's CNN14 pretrained convolutional gradients are now nonzero; only execution on the 4 GiB RTX 3050 can do that.

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
- **The separate Stage 3 full audio audit subsequently decoded all 8,732 recordings successfully (0 invalid).** File-existence and decode checks are separate pieces of evidence.

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

## 6. Stage 3 waveform preprocessing (full Windows execution PASS)

The Stage 3 implementation is supplied in `src/dataset.py`, `src/audit_audio.py`, `src/check_dataloader.py`, and `tests/test_dataset.py`. The user ran the full verification on Windows on **23 October 2026**: `results/verification_runs/20261023_011110_002188/summary.json` (13/13 check statuses passing with the open source-isolation warning).

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

**Observed full audio audit:** 8,732/8,732 WAV files decoded, 0 invalid; 11 source sampling rates, 8,000–192,000 Hz; durations min `0.05 s`, median `4.0 s`, mean `3.6075 s`, max `4.03665 s`. **Eight clips exceed 4.001 s.** Center cropping may discard events at boundaries; this is a documented limitation. All 8,732 clips remain in the dataset. The audit generated dataset-statistics reports and figures.

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

### Stage 3 passing criteria (all met locally on 23 October 2026)

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
├── run_stage4_checks.py                 # One-command Stage 4 unit + gradient checker
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
│   ├── check_dataloader.py             # Real batch and CNN14 smoke tests
│   ├── finetune_model.py               # Stage 4: strict load + freeze policies
│   ├── finetune_check.py               # Stage 4: real-gradient/update verification
│   └── finetune_train.py               # Stage 4: train/validate only
├── tests/
│   ├── test_dataset.py                 # Stage 3 unit tests
│   ├── test_finetune.py                # Stage 4 freeze/optimizer tests
│   └── test_finetune_loop.py           # Stage 4 train/validation loop test
├── results/
│   ├── fold_split_summary.csv
│   ├── cross_fold_source_summary.csv
│   ├── cross_fold_source_details.csv
│   ├── dataset_integrity_report.json  # After Stage 3 audit
│   ├── audio_file_statistics.csv       # After Stage 3 audit
│   ├── verification_runs/              # One log + JSON summary per Stage 2/3 run
│   └── stage4/                         # New training run dirs (after local execution)
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
| 3 | Implemented Stage 3 dataset/audit/DataLoader scripts and unit tests | **PASS on Windows (4/4 existing unit tests)** |
| 3 | Full audio decoding and statistics: all 8,732 recordings | **PASS locally; 0 invalid WAVs, 11 sample rates, 8 clips longer than 4.001 s** |
| 3 | Real-audio CNN14 forward with `[1,10]` output | **PASS locally (untrained head)** |
| 3 | One-command `python run_all_checks.py` full audit | **13/13 PASS, source-ID caveat remains; run summary at `results/verification_runs/20261023_011110_002188/summary.json`** |
| 4 | Freeze-policy/optimizer/test-loop implementations | **Prepared; 6 Stage 4 lightweight tests passed in development environment** |
| 4 | Real CNN14 convolutional backward pass, parameter updates, pilot training | **Awaiting local Windows execution** |
| 5–9 | Final held-out test evaluation, CO4/CO5 results | **Not started / no claims of results** |
| 10–11 | Final reproducibility audit and registration-number ZIP | **Pending** |

## 10. Stage 4 fine-tuning hyperparameters (candidate values; not validated results)

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
| Batch size | **1** for initial RTX 3050 GPU pilot, with 4-microbatch gradient accumulation; adjust based on measured memory |
| Epochs | 1-epoch limited-batch smoke test; initial full-fold candidate 6 epochs, patience 2 |
| Scheduler | Validation-based scheduler if needed; not yet selected |
| Augmentation | None initially; optional training-only controlled ablation |
| Dropout | Standard backbone behavior plus any documented head changes; not yet finalized |
| Frozen layers and BatchNorm | Implemented three modes (`head`, `fc1`, `block6`); freeze all pretrained BN running statistics, even in trainable block6 |

**Hyperparameter tuning:** not yet performed for this CNN14/UrbanSound8K implementation. Stage 4 uses pilot candidates and logs `config.json` per training run; do not pretend these are optimally tuned. The final README should point to genuine validation-controlled tuning results.

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

**Next milestone:** Run the Stage 4 unit suite and `python -m src.finetune_check --config block6 --device cuda` on the RTX 3050. If gradient/update and frozen-BN checks pass, run short diagnostic training, then genuine full-fold validation-selected experiments. Continue monitoring the source-overlap warning; do not touch test data during model selection.


## 14. Stage 4 — CNN14 fine-tuning (code added 23 October 2026; local execution pending)

**Read `STAGE4_README.md` for complete commands and implementation rationale.** This stage adds `src/finetune_model.py`, `src/finetune_check.py`, `src/finetune_train.py`, `tests/test_finetune.py`, and `tests/test_finetune_loop.py`, and `run_stage4_checks.py`.

**Architecture:** build official CNN14 with its original 527-class classifier, load the verified tensor-only AudioSet checkpoint with `strict=True`, then attach `Linear(2048,10)` to the pretrained embedding. Use raw logits and `CrossEntropyLoss`; do not use the original AudioSet probabilities for the UrbanSound8K loss. The original AudioSet head stays frozen. No other network replaces CNN14.

| Stage 4 configuration | Trainable components | Intended comparison |
| --- | --- | --- |
| `head` | New 10-way linear classifier | Frozen-feature baseline |
| `fc1` | Pretrained `fc1` + new head | Dense-layer fine-tuning baseline |
| `block6` | Pretrained `conv_block6` + `fc1` + new head | **Genuine partial CNN fine-tuning (main method)** |

All earlier convolutions remain frozen in `block6`; the optimizer receives **only** `requires_grad=True` tensors. All original BatchNorm running means/variances are fixed in evaluation mode, but unfrozen block6 BN affine weights can update. Standard PANNs training-mode SpecAugmentation and dropout remain active only during training; validation uses evaluation mode.

**One-command checker (recommended):**

```bat
cd /d D:\EnvironmentalSoundRecognition
python run_stage4_checks.py
```

This runs all unit tests plus all three single-update checks, stops on failure, and logs under `results/stage4/checks/`. For four-batch diagnostic pilots too, add `--pilots`; for unit tests only, add `--unit-only`. It never evaluates the test fold or starts a full training run.

**Equivalent individual commands:**

```bat
python -m unittest discover -s tests -v
python -m src.finetune_check --config head --device cuda
python -m src.finetune_check --config fc1 --device cuda
python -m src.finetune_check --config block6 --device cuda
```

The `finetune_check` commands perform a single real train-fold forward/backward/update; they verify gradient and parameter changes, no frozen-gradient/update leakage, frozen BatchNorm running-stat invariance, output shape `[1,10]` and peak CUDA allocated/reserved memory. **They are not accuracy measurements or a completed training experiment.**

**Diagnostic training (limited batches; do not report as project performance):**

```bat
python -m src.finetune_train --config head --test-fold 1 --epochs 1 --batch-size 1 --accum-steps 4 --max-train-batches 4 --max-val-batches 4
python -m src.finetune_train --config block6 --test-fold 1 --epochs 1 --batch-size 1 --accum-steps 4 --max-train-batches 4 --max-val-batches 4
```

**First genuine full-fold candidate (only after diagnostics pass):**

```bat
python -m src.finetune_train --config head --test-fold 1 --epochs 6 --patience 2 --batch-size 1 --accum-steps 4
python -m src.finetune_train --config fc1 --test-fold 1 --epochs 6 --patience 2 --batch-size 1 --accum-steps 4
python -m src.finetune_train --config block6 --test-fold 1 --epochs 6 --patience 2 --batch-size 1 --accum-steps 4
```

Each run creates a unique directory under `results/stage4/` with `config.json`, `history.csv`, `learning_curves.png`, `best.pt` (trainable parameters only), and `summary.json`. The best checkpoint is selected exclusively using **validation macro-F1** (tie-break: validation accuracy); the test fold is **never evaluated** in Stage 4. A run with limited train/val batches is marked `diagnostic_only=true`. The saved checkpoint must be combined with the SAME safe AudioSet pretrained backbone to reconstruct the trained classifier. Early stopping is supported.

**Provenance:** Stage 2/3 passed on the student PC; Stage 4 Python code and toy CPU tests were validated in the development environment (six additional tests alongside four Stage 3 tests, ten total), **not yet run locally with real CNN14 gradients**. No training metrics may be claimed until actual local executions are provided.

### Stage 4 limitations and future evidence

With the RTX 3050's 4 GiB VRAM, a genuine `conv_block6` backward pass may exceed memory. Test with batch size 1 first; gradient accumulation changes effective optimizer batch but does not reduce single-microbatch memory. Mixed precision is enabled automatically on CUDA but must be measured and validated.

The Stage 4 training curves compare **online training metrics with training-only SpecAugmentation/dropout** against clean evaluation-mode validation; do not overinterpret a small train–validation gap as a formal bias–variance decomposition.

Stage 5 must evaluate a **validation-selected** checkpoint on untouched test data; Stage 6+ should add the full required metrics, all-fold stability as feasible, published UrbanSound8K comparison, controlled ablation, and submission audit. The five cross-fold `fsID` warnings remain unresolved and must be mentioned alongside benchmark results.
