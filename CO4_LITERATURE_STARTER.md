# CO4 — Published UrbanSound8K comparison: verified literature starter

**Interpretation rule:** The results below are from the cited publications. They are not our experiments. Our one-held-out-fold result is not directly comparable to published ten-fold averages. Neither differing audio representations nor training from AudioSet-pretrained weights should be ignored.

| Method | Year / original paper | UrbanSound8K accuracy as reported | Protocol | Notes |
|---|---|---:|---|---|
| TSCNN-DS two-stream, decision-level fusion | 2019; Su, Zhang, Wang & Madani, *Sensors* | 97.2% | Reported performance of their UrbanSound8K experiments; review fold methodology in paper before asserting a fully matched protocol | Multiple auditory features and late fusion; DOI: https://doi.org/10.3390/s19071733 |
| Inception-Dense Blocks for Hearing Aids | 2021; Ting, Ruan & Li, *Sensors* | 83.03% mean accuracy | Uses original UrbanSound8K folds and 10-fold evaluation; their preprocessing resamples to 16 kHz | Lightweight model; DOI: https://doi.org/10.3390/s21165406 |
| RACNN resource-adaptive convolution | 2022; Fang, Yin, Du & Huang, *Scientific Reports* | 97.51% ± 0.18% | Ten-fold cross-validation; reported 120 epochs with SGD | Lightweight architecture (~145.7K parameters in original Table 5); DOI: https://doi.org/10.1038/s41598-022-10382-x |
| **Ours: partially fine-tuned AudioSet PANNs CNN14** | **2026; student implementation** | **88.55%** | **ONE held-out test fold 1**, selection on validation fold 2 | 81,857,561 total parameters, 60,848,138 trainable in `block6` configuration; cannot claim rank relative to ten-fold average. |

## Research tasks still required

1. Confirm the original methods' exact fold protocol, any external pretraining, augmentation and metric aggregation from the full paper. Do not infer omitted F1, precision, recall, ROC-AUC or computational metrics.
2. State that our pretrained CNN14 uses AudioSet and evaluates only one test fold. Published models may have different source data, training duration and input features.
3. Write two paragraphs interpreting similarities, discrepancies, parameter costs and validity limits. Do not claim SOTA, comparable win, or algorithmic novelty.
4. For stronger apples-to-apples benchmarking, execute the official ten-fold protocol for CNN14. This was not performed in the uploaded runs.

## Verified primary sources

- Su et al., 2019: https://doi.org/10.3390/s19071733
- Ting et al., 2021: https://doi.org/10.3390/s21165406
- Fang et al., 2022: https://doi.org/10.1038/s41598-022-10382-x
- Official UrbanSound8K usage notes: https://urbansounddataset.weebly.com/urbansound8k.html
