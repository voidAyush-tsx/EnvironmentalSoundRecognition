# CO4 — Original-paper comparison on UrbanSound8K

**Project:** Deep Learning Framework for Environmental Sound Recognition Using PyTorch  
**Evidence date:** 30 October 2026  
**Comparison standard:** published results are **external results**, not reproduced by this project. Report `NR` rather than inventing missing metrics.

## Published methods and our result

| Method / original paper | Year | Input / model | Pretraining / extra data | UrbanSound8K evaluation | Accuracy (%) | Precision / Recall / F1 / AUC | Reported model parameters | Source |
|---|---:|---|---|---|---:|---|---:|---|
| TSCNN-DS — Su, Zhang, Wang & Madani | 2019 | Two CNN streams, combined auditory features and Dempster–Shafer decision-level fusion | AudioSet pretraining **not reported** | Paper reports ten experiment measurements (`N=10`, mean 0.9720, SD 0.01788); paper's detailed fold protocol should be consulted before claiming it is identical to ours | **97.20** | NR for matching overall macro metrics | NR in this primary-paper table | https://doi.org/10.3390/s19071733 |
| Inception-Dense Blocks — Ting et al. | 2021 | Log-Mel features at 16 kHz; lightweight Inception–Dense classifier | AudioSet pretraining **not reported**; training-time mixup and augmentation described | Original predefined UrbanSound8K folds; 10-fold cross-validation, mean accuracy | **83.03** | NR for matching overall macro metrics | **97,214** (Table 5) | https://doi.org/10.3390/s21165406 |
| RACNN — Fang et al. | 2022 | Log-Mel features; resource-adaptive CNN, α=0.5 | AudioSet pretraining **not reported** | Paper's UrbanSound8K protocol, reported mean and variation across folds | **97.51 ± 0.18** | NR for matching overall macro metrics | **145,700** (Table 2) | https://doi.org/10.1038/s41598-022-10382-x |
| **Our partially fine-tuned PANNs CNN14** | **2026** | 32-kHz waveform → AudioSet-pretrained CNN14 → 2048-D embedding → 10-logit head | **AudioSet pretrained** via original PANNs checkpoint; UrbanSound8K used for downstream training | **One held-out test fold 1** (873 samples), chosen using validation fold 2; training folds 3–10 | **88.55** | **Macro precision 0.9008; recall 0.9005; F1 0.8964; OVR ROC-AUC 0.9909** | **81,857,561 total; 60,848,138 trainable** | `reports/faculty_evidence/heldout/evaluation_summary.json` |

## Interpretation and strict fairness limitations

1. All rows concern the same **dataset**, but **not necessarily the same experimental conditions**. In particular, our value is *one fold's test score*, whereas published accuracies summarize multiple experiments/folds. **Do not subtract and present the published numbers as fair head-to-head superiority margins, or call our model SOTA.**
2. UrbanSound8K's creators explicitly advise **ten-fold cross-validation using the predefined splits** and averaging the held-out results. The paper comparison is **contextual** until all ten test folds are evaluated under a compatible protocol. Our 8-training/1-validation/1-test design is also not identical to the dataset guide's 9-training/1-test description. See https://urbansounddataset.weebly.com/urbansound8k.html .
3. **Input representations differ:** the 2021 and 2022 models operate on handcrafted log-Mel inputs; ours accepts raw waveforms and computes log-Mel features inside pretrained CNN14. TSCNN-DS combines multiple feature streams and decision-level fusion. The feature sets, training schedules, augmentation policies, and training budgets are not controlled across studies.
4. **External pretraining matters:** our representation benefits from AudioSet and has a much larger overall parameter count; this is not a from-scratch comparison. The reported efficient methods use far fewer parameters; our approach is **not parameter-efficient** when `conv_block6` is unfrozen (~74.3% of total parameters are trainable).
5. Published articles generally report accuracy and selected complexity metrics, not comparable per-class macro precision/recall/F1/AUC. `NR` means **not reported in a directly equivalent way**, not zero.
6. CO4 conclusion: the exercise establishes placement in the UrbanSound8K literature **with explicit caveats**, not a verified accuracy ranking.

## Original references and study notes

- **Su et al. (2019)**, *Environment Sound Classification Using a Two-Stream CNN Based on Decision-Level Fusion*, Sensors 19(7), 1733. https://doi.org/10.3390/s19071733 . See Tables 2–3 and 7; the paper's TSCNN-DS result is 97.2% and Table 3 lists N=10.
- **Ting et al. (2021)**, *Environmental Noise Classification with Inception-Dense Blocks for Hearing Aids*, Sensors 21(16), 5406. https://doi.org/10.3390/s21165406 . See §4.1, §4.4, and Table 5; 83.03% and 97,214 parameters. In that paper mixup is training-only.
- **Fang et al. (2022)**, *Fast environmental sound classification based on resource adaptive convolutional neural network*, Scientific Reports 12, 6599. https://doi.org/10.1038/s41598-022-10382-x . See UrbanSound8K results and Table 2; 97.51% (±0.18), α=0.5, 145.7K parameters.
- **Salamon, Jacoby & Bello (2014)**, *A Dataset and Taxonomy for Urban Sound Research*, ACM Multimedia. Dataset and official protocol: https://urbansounddataset.weebly.com/urbansound8k.html .
- **Kong et al.**, PANNs, official source and weights: https://github.com/qiuqiangkong/audioset_tagging_cnn and https://zenodo.org/records/3576403 . Commit `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4` used in our runs.

**Faculty comparison matrix applicability:** No original paper has been assumed to report all of precision, recall, F1 and AUC. Retain NR cells and original URLs; never calculate these missing values from accuracy.
