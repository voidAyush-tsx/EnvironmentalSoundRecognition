# Research and submission

This guide brings together the experiment protocol, measured evidence, published
method context and faculty requirements. For execution and checkpoint access, use
[reproducibility](REPRODUCIBILITY.md); for the application overview, use
[README](../README.md).

## Recorded experiment

The task is single-label classification of UrbanSound8K's ten official classes.
All 8,732 clips retain their predefined folds: folds 3-10 train the model
(6,971 clips), fold 2 provides validation (888), and fold 1 is held out (873).
The experiment used one seed, 67.

The selected `block6` model was chosen using validation macro-F1 before fold-1
evaluation. Its recorded results are:

| Held-out metric | Value |
|---|---:|
| Accuracy | 88.55% (773 / 873) |
| Macro precision | 0.9008 |
| Macro recall | 0.9005 |
| Macro F1 | 0.8964 |
| Macro one-vs-rest specificity | 0.9870 |
| Macro one-vs-rest ROC-AUC | 0.9909 |

Exact aggregate values, weighted metrics and provenance are in
[the evaluation JSON](../reports/faculty_evidence/heldout/evaluation_summary.json).
[Per-class metrics](../reports/faculty_evidence/heldout/per_class_metrics.csv) and
[the confusion matrix](../reports/faculty_evidence/heldout/confusion_matrix.png)
show errors by class. True labels are matrix rows; predictions are columns.
ROC-AUC was calculated from softmax probabilities rather than hard class IDs.

Drilling (68% recall) and air conditioner (69%) were hardest on this fold.
Engine idling reached 82.29%, and jackhammer 94.17%. Car horn (36 clips), gunshot
(35) and siren (86) each reached 100% recall here; those limited supports do not
establish perfect generalization.

This is **one held-out fold**, not ten-fold cross-validation or a multi-seed
estimate. Five Freesound source IDs span official folds, sometimes with overlapping
annotated intervals. WAV paths are disjoint, but source-level independence is
imperfect. Keep this caveat alongside reported performance.

## Freezing-depth study

The controlled variable was the set of trainable pretrained layers. All three
runs used the same folds, seed, six-epoch budget, waveform processing and optimizer
settings. Only the validation-selected `block6` model has a held-out test result;
there are no head-only or `fc1` test scores in the evidence.

| Mode | Trainable layers | Trainable parameters | Best validation accuracy | Best macro-F1 | Selected epoch | Six-epoch time | Peak CUDA allocation |
|---|---|---:|---:|---:|---:|---:|---:|
| `head` | New classifier | 20,490 | 87.73% | 0.8638 | 4 | 19.80 min | 474.2 MiB |
| `fc1` | Final dense layer + classifier | 4,216,842 | 88.06% | 0.8668 | 4 | 18.87 min | 522.3 MiB |
| `block6` | Final convolution block + dense layer + classifier | 60,848,138 | 90.65% | 0.9010 | 4 | 28.65 min | 1,278.1 MiB |

All adapted models contain 81,857,561 parameters in total. `block6` trains about
74.3% of them, so this is partial fine-tuning with a substantial compute/storage
cost. Compared with `head`, it improved validation accuracy by 2.93 percentage
points and macro-F1 by 0.0372, using about 1.45 times the recorded training time
and 2.70 times peak GPU allocation. Against `fc1`, the accuracy difference was
2.59 points and macro-F1 difference 0.0341. The smaller `fc1` versus `head`
differences were 0.34 points and 0.0031.

All three selected epoch 4. For `block6`, validation loss then rose from 0.3433
at epoch 4 to 0.3986 at epoch 6, while macro-F1 fell to 0.8847. Restoring the
validation-selected checkpoint avoids reporting the worse final epoch.
Training accuracy is measured online with official dropout/SpecAugmentation;
validation runs in evaluation mode. Subtracting those accuracies is therefore
not a controlled estimate of a generalization gap.

The contribution is a documented freezing-depth experiment and its computational
trade-off, rather than a new network or optimizer. No statistical-significance,
parameter-efficiency or state-of-the-art claim follows from one split and seed.
A training-only clip previously produced zero input to `fc1` and zero weight
gradients; other clips and full epochs showed nonzero updates. The diagnostics
retain that distinction instead of treating zero gradients as a pass.

Raw evidence is under
[training runs](../reports/faculty_evidence/training_runs/) and
[validation comparison](../reports/stage4_comparison/). Ten-fold evaluation,
additional seeds and smaller unfreezing depths are possible future experiments;
they have not been completed in this record.

## Training details

The implementation preserves the official AudioSet CNN14 and its original
527-class head for strict checkpoint compatibility. Its 2,048-dimensional
embedding feeds a separate ten-class linear classifier.

| Item | Implemented behavior / recorded setting |
|---|---|
| Input | Mono float32 waveform, 32 kHz, four seconds / 128,000 samples |
| Preprocessing | Channel mean; `soxr_hq` resampling when needed; deterministic center crop or right zero-padding; no loudness normalization |
| Features | CNN14 computes spectrogram/log-Mel internally; notebook Mel plots are illustrative |
| Loss | Cross-entropy on ten raw logits |
| Optimizer | AdamW; classifier LR 1e-3, trainable backbone LR 1e-5, weight decay 1e-4 |
| Batch / accumulation | 1 / 4, including a correctly scaled final partial window |
| Budget / patience | Six epochs / two for the recorded runs; CLI defaults remain three / three |
| Precision | CUDA BF16 on the verified hardware; BF16 when supported, otherwise FP32; no whole-model FP16 |
| Seed | 67, deterministic cuDNN flags |
| Gradient clipping | Maximum norm 1.0, with nonfinite-gradient rejection |
| Scheduler | None |
| Checkpoint choice | Highest validation macro-F1, accuracy as tie-break; epoch 4 in all three recorded runs |
| Freeze policy | `head`, `fc1` and `block6` enable the parameter subsets listed above |
| BatchNorm | All pretrained running statistics remain fixed; affine weights may train in the unfrozen block |
| Dropout / augmentation | Official CNN14 training-mode dropout/SpecAugmentation remains active; no extra task dropout or dataset augmentation |
| Saved weights | Trainable parameters only; restore the original backbone before applying them |
| Tuning | Freezing depth varied; LR, weight decay, batch and seed were held fixed rather than grid-searched |

Recorded configurations and histories, rather than a later CLI default, define
the historical experiment. Model/freezing code is in
[`finetune_model.py`](../src/finetune_model.py), the training loop in
[`finetune_train.py`](../src/finetune_train.py), and numerical policy in
[`runtime.py`](../src/runtime.py).

## Published methods

These are **published external results**, not reproduced by this project.
The same dataset does not imply the same splits, pretraining, augmentation,
input representation, training budget or compute regime.

| Original method | Year | Representation and model | Published UrbanSound8K result | Protocol / complexity notes |
|---|---:|---|---|---|
| TSCNN-DS, Su et al. | 2019 | Two CNN streams, combined auditory features, Dempster-Shafer decision fusion | 97.20% accuracy | Paper gives ten measurements, N=10, mean 0.9720 and SD 0.01788; consult its detailed fold protocol. Parameters NR in the primary comparison. |
| Inception-Dense, Ting et al. | 2021 | 16-kHz log-Mel, lightweight Inception-Dense classifier; training-time mixup/augmentation | 83.03% mean accuracy | Ten predefined folds; 97,214 parameters (Table 5). |
| RACNN, Fang et al. | 2022 | Log-Mel, resource-adaptive CNN, alpha=0.5 | 97.51% +/- 0.18 | Paper's multi-fold protocol; 145,700 parameters (Table 2). |
| This project's partial CNN14 | 2026 | 32-kHz waveform, AudioSet-pretrained CNN14, new ten-class head | 88.55% accuracy, fold 1 only | Train 8 folds, validate 1, test 1; 81,857,561 total / 60,848,138 trainable parameters. |

AudioSet pretraining is not reported for the three published methods in this
comparison; this project explicitly uses it. Their matching overall macro
precision, recall, F1 and AUC are **NR** where not reported equivalently.
NR means unavailable in a comparable form, not zero; those values cannot be
inferred from accuracy.

UrbanSound8K recommends predefined ten-fold evaluation and averaging held-out
results. The project's eight-train/one-validation/one-test protocol also differs
from a nine-train/one-test description. Do not present the table as a controlled
leaderboard, subtract published means as fair superiority margins, or claim SOTA.
The much larger pretrained representation and trainable parameter count are
material differences from the lightweight published models.

### Original references

- Su, Zhang, Wang and Madani (2019), *Environment Sound Classification Using a
  Two-Stream CNN Based on Decision-Level Fusion*, Sensors 19(7), 1733.
  [Original paper](https://doi.org/10.3390/s19071733). Tables 2-3 and 7 contain
  the study details; the reported TSCNN-DS result is 97.2%.
- Ting et al. (2021), *Environmental Noise Classification with Inception-Dense
  Blocks for Hearing Aids*, Sensors 21(16), 5406.
  [Original paper](https://doi.org/10.3390/s21165406). Sections 4.1/4.4 and
  Table 5 describe the 83.03% result, protocol and 97,214 parameters.
- Fang et al. (2022), *Fast environmental sound classification based on resource
  adaptive convolutional neural network*, Scientific Reports 12, 6599.
  [Original paper](https://doi.org/10.1038/s41598-022-10382-x). UrbanSound8K
  results and Table 2 give 97.51% (+/-0.18) and 145.7K parameters.
- Salamon, Jacoby and Bello (2014), *A Dataset and Taxonomy for Urban Sound
  Research*, ACM Multimedia. [Dataset and evaluation guidance](https://urbansounddataset.weebly.com/urbansound8k.html).
- Kong et al., *PANNs: Large-Scale Pretrained Audio Neural Networks for Audio
  Pattern Recognition*. [Official source](https://github.com/qiuqiangkong/audioset_tagging_cnn)
  and [AudioSet weights](https://zenodo.org/records/3576403). The exact source
  pin and checkpoint identity are in [checkpoint reconstruction](REPRODUCIBILITY.md#checkpoint-reconstruction).

## Faculty submission

This appendix covers the academic deliverable: CO3 implementation/evaluation
(10 marks), CO4 original-paper context (5), and CO5 justified experimentation (5).
It is separate from ordinary development and does not require rerunning the
held-out experiment to prepare files.

| Requirement | Evidence to include |
|---|---|
| Dataset / distributions / preprocessing | Notebook 01, class/fold/audio figures, official labels and 32-kHz waveform pipeline |
| Architecture / genuine training | Pinned pretrained CNN14, trainable-layer definitions, three full histories, actual gradient/update checks |
| Evaluation | Selected fold-1 aggregate/per-class metrics, confusion matrix and probability-based AUC |
| Convergence / cost | Training/validation curves, epoch-4 selection, parameter count, runtime and memory trade-offs |
| Published-method comparison | References and protocol caveats above; NR for unavailable matching metrics |
| Ablation | Controlled `head` / `fc1` / `block6` comparison with fixed other settings |
| Reproduction | README, three notebooks, application/tests, dependency recipes/snapshot, dataset and checkpoint acquisition instructions |
| Student identity / archive | Actual name, registration and section; exact filename `<registration>.zip` |
| Selected weights | Exact evaluated trainable-parameter checkpoint, with matching SHA-256 |

Prepare a new archive only when needed:

1. Review the committed evidence and saved notebook outputs. A file audit does
   not prove fresh model replay, licensing signoff or ten-fold robustness.
2. Run `python -m src faculty audit`. It checks required files and recorded
   metadata; packaging separately hashes the actual checkpoint.
3. Restore the selected checkpoint using
   [the reconstruction instructions](REPRODUCIBILITY.md#checkpoint-reconstruction).
   Discovery uses the documented location or `models/selected_block6_best.pt`,
   never the latest run. Use `--checkpoint <path>` for another location of the
   exact evaluated file.
4. Supply real identity through CLI options or Notebook 03's environment
   variables `ESR_STUDENT_NAME`, `ESR_REGISTRATION_NUMBER` and `ESR_SECTION`.
   Notebook packaging remains off until `CREATE_FINAL_SUBMISSION_ZIP=True`.
5. Inspect `python -m src faculty package --help`, then explicitly package:

   ```powershell
   python -m src faculty package --registration <digits> --student-name "Actual Name" --section <section> --checkpoint <selected-best.pt>
   ```

6. Existing registration ZIPs are refused rather than overwritten. Preserve the
   earlier private archive before deliberately preparing another version.
   Archive inspection checks required files, identity, integrity and selected
   checkpoint/evidence agreement without unpickling model contents.

The package includes selected trainable parameters and documented dependencies,
but excludes raw WAVs and the original large AudioSet backbone. Those assets and
the pinned upstream source must be obtained separately. A private ZIP was built
in the original workflow; its absence from a public checkout does not mean the
recorded research is pending. Retain the single-fold/source-overlap/pretraining
disclosures and never invent new experiments or missing published metrics.
