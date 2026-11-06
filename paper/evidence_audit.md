# Evidence audit for the revised manuscript

Audit date: 6 November 2026. The supplied LaTeX draft was checked against the current local project, the saved experiment artifacts, and the primary sources below. The original attachment and existing project records were not edited. No training, test inference, or new performance experiment was performed for this revision.

## What changed

The draft's assertion that UrbanSound8K/CNN14 artifacts were missing was obsolete for this workspace. It has been replaced with an empirical account of the three preserved runs and the selected fold-1 evaluation. The assertion that only an ESC-50 notebook was accessible was removed; no ESC-50 measurements are used.

| Manuscript content | Supporting evidence | Verification performed |
| --- | --- | --- |
| 8,732 clips; ten classes; class/fold totals | `data/UrbanSound8K/metadata/UrbanSound8K.csv` | Recounted rows, labels, official class mapping, fold membership, and file-key uniqueness |
| Training 6,971 / validation 888 / test 873 | Same metadata; recorded run configurations | Reconstructed folds 3-10 / 2 / 1 and matched record counts |
| Sample rates, channels, durations | All 8,732 local WAV headers; `results/dataset_integrity_report.json` | Fresh header scan compared with the saved full-decoding report; decoding itself was not repeated |
| Five source IDs crossing folds | Metadata `fsID`, `fold`, `start`, `end` | Recomputed cross-fold membership and positive temporal-overlap pair counts |
| Mono, 32 kHz, four-second crop/pad | `src/dataset.py`, `src/config.py`; saved configurations | Inspected implementation and recorded settings |
| CNN14 computation and stochastic training behavior | Pinned upstream `pytorch/models.py`; `src/finetune_model.py` | Inspected actual code, including active dropout/SpecAugmentation and fixed BatchNorm statistics |
| Three configurations and parameter counts | Run configs; reconstructed model | Checked each `requires_grad` policy and exact parameter-element totals |
| Optimizer, loss, accumulation, patience, seed, precision | Run configs/histories; training implementation | Cross-checked the historical settings; did not substitute later CLI defaults |
| Validation comparison and selected epoch | `reports/faculty_evidence/training_runs/{head,fc1,block6}/` | Recomputed maximum macro-F1, accuracy tie-break, six-epoch totals, and peak allocation |
| Test accuracy, macro/weighted metrics, AUC, classwise metrics | `results/stage5/20261030_132140_664348_block6_test1/test_predictions.csv` | Recomputed from all 873 rows and checked against saved aggregate/per-class/confusion records |
| Test label alignment | Predictions plus original metadata | One-to-one filename/fold join, matching class IDs, complete fold-1 coverage |
| Probability integrity | Saved ten-class probabilities | Checked finite values, bounds, row sums, argmax labels, confidence, and correctness flags |
| Selected checkpoint | `results/stage4/20261027_144004_603447_block6_test1/best.pt` | Recomputed SHA-256, restricted-loaded payload, checked epoch/configuration, restored selected tensors into pinned backbone |
| Original pretrained weights | `checkpoints/Cnn14_mAP=0.431.pth`; official Zenodo entry | Recomputed original MD5 and compared with published value; original legacy object was not unpickled |
| Historical training environment | `results/notebook_evidence/20261027_134247_157394/environment.json` | Read the contemporaneous record; not inferred from the current environment |
| Authors and affiliation | User's explicit replies in this conversation | User supplied; Sandeep Gupta confirmed to share the affiliation; his email intentionally omitted |

The exact recomputed metrics, source-overlap details, current source revision, and verification boundaries are in `audit/verification.json`. `audit/evidence_manifest.csv` hashes the inspected project records. `audit/audio_headers.csv` contains the fresh header measurements. `audit/training_copy_checks.json` confirms that all nine copied training artifacts match the original run directories; `audit/gradient_history_checks.json` checks the applicable recorded gradient counters. These hashes establish the identity of files inspected now; they are not signatures proving when historical files were created.

## Important corrections and qualifications

1. **Single fold only.** The 88.55% figure means 773 correct predictions out of 873 clips in fold 1. It is not a ten-fold average. The manuscript keeps that scope in its title, abstract, results, comparison, and conclusion.
2. **Validation improvement only.** The 2.93-percentage-point improvement over head-only concerns validation fold 2. No head-only or `fc1` held-out test scores were recovered.
3. **Substantial trainable fraction.** `block6` trains 60,848,138 of 81,857,561 registered parameter elements, or 74.33%. It is not described as a tiny or parameter-efficient adaptation. The total includes frozen front-end kernels and the unused original AudioSet head.
4. **Augmentation is active inside CNN14.** The dataset loader itself has no augmentation; the official backbone does apply training-mode SpecAugmentation and dropout. The older dataset-report field `augmentation: none` is interpreted as loader behavior, not absence of all model augmentation.
5. **Memory and time scope.** Runtime sums recorded training-plus-validation epoch timers. Peak memory is allocated CUDA tensor memory, not total device use. Neither is a repeated benchmark or an energy measurement.
6. **Source independence is imperfect.** IDs 106905, 132016, 176638, and 180937 cross train/test; ID 77751 crosses train/validation. A shared source alone is not proof of identical waveforms, while some annotated intervals do overlap. No causal estimate of the impact is asserted.
7. **Measured duration differs slightly from the nominal description.** Eight local WAVs exceed four seconds by a small amount; maximum header duration is 4.036647314949202 s. The model center-crops them after resampling.
8. **Training curves have different measurement conditions.** Online training includes stochastic augmentation/dropout and evolving weights. Validation uses evaluation mode. Their difference is not labeled a controlled generalization gap.
9. **Historical diagnostics are mixed evidence.** The early verification run `20261027_133720_914118` reports `FAIL_OR_INCOMPLETE` after an environment check. It is not presented as an overall success. Later training-only backward checks and full histories provide the relevant positive evidence. A clip with zero later-layer gradients is disclosed rather than hidden.
10. **Metric recomputation is not model replay.** Saved predictions reproduce the reported aggregate metrics, and the checkpoint hash/reconstruction match. This does not independently prove the predictions were generated by that checkpoint; fresh inference was deliberately not performed while preparing the manuscript.
11. **Test cross-entropy is a recorded value.** The paper labels it accordingly. It was calculated from logits in the original evaluator; the exact original logits were not preserved in the probability CSV, so this audit does not claim bit-identical independent loss reconstruction.

## Primary-source literature checks

All links below were checked on 6 November 2026. External performance figures are attributed reports, not replications by this project.

| Reference | Direct location checked | Finding used |
| --- | --- | --- |
| Salamon, Jacoby, Bello (2014) | [Original paper](https://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf); [official dataset page](https://urbansounddataset.weebly.com/urbansound8k.html); local dataset README | Dataset identity, nominal clip limit, class mapping, predefined folds, official recommendation for ten-fold evaluation; license statement from the supplied README |
| Gemmeke et al. (2017) | [Google Research publication](https://research.google/pubs/audio-set-an-ontology-and-human-labeled-dataset-for-audio-events/) | AudioSet paper identity and audio-event annotation context |
| Kong et al. (2020) | [arXiv v5](https://arxiv.org/abs/1912.10211v5); [official source](https://github.com/qiuqiangkong/audioset_tagging_cnn); local pinned `Cnn14` implementation | Pretrained audio models, actual waveform interface, embedding, tagging output, blocks, pooling, dropout, augmentation |
| PANNs pretrained release | [Zenodo version 1](https://zenodo.org/records/3576403), file `Cnn14_mAP=0.431.pth` | Published MD5 `595633ac2d1cac7ef04ebf70e2fee4e4` |
| Piczak (2015) | [Author-hosted PDF](https://www.karolpiczak.com/papers/Piczak2015-ESC-ConvNet.pdf), Section 3.3 and p. 5 paragraph continuing the results | **LP 73.1%** is the CNN result; **US 73.7%** is the unsupervised comparator. The supplied draft misattributed 73.7%. The later ESResNet table repeats 73.70% under Piczak, so the manuscript explicitly notes this disagreement and follows the directly attributed original text. |
| Salamon and Bello (2017) | [Original manuscript PDF](https://arxiv.org/pdf/1608.04363), evaluation setup and Section III | Approximately 0.73 without augmentation and 0.79 with augmentation; ten-fold study. Rounded numbers stay rounded. |
| Guzhov et al. (2020) | [Original PDF](https://arxiv.org/pdf/2004.07301), Tables I-II and Section V-D | **ESResNet-Attention** with ImageNet pretraining gives 84.21% mono / 85.42% stereo. Plain ESResNet gives 83.59% / 84.90%. Reimplementation-based attribution of Su et al.'s results to unofficial splits is identified as these authors' finding. |
| Su et al. (2019) | [Publisher PDF](https://mdpi-res.com/d_attachment/sensors/sensors-19-01733/article_deploy/sensors-19-01733.pdf), Sections 4.1-4.2 and Tables 2-3 | 97.2% reported TSCNN-DS accuracy, mean 0.9720 over ten measurements; original text says ten-fold evaluation. The result is retained only with the subsequent reproducibility caveat. |

Publisher/PMC HTML for Su et al. was intermittently blocked, so the publisher-hosted PDF was downloaded and inspected directly. The PDF, not a search-result snippet, supports the reported value. Bibliography entries omit unverified page ranges rather than guessing them. Preprints are cited as the specific versions actually consulted.

## Rebuilding the audit

From the repository root, with the existing project dependencies and original local assets:

```powershell
python paper/audit_and_figures.py
```

This writes only within `paper/`, plus a Numba cache within `tmp/paper-tools/`. It does not modify source experiment records or execute a new model training/evaluation. `--reuse-headers` can reuse the already generated header table during formatting iterations. PDF graphics contain measured data only. The paper's generated table rows use the unrounded input values before formatting.
