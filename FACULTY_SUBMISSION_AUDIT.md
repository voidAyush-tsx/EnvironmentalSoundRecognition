# Faculty submission readiness audit — 30 October 2026

**Source:** `project_submission_instructions`, "Deep Learning Project Assignment – 20 Marks". **Original initial-evaluation deadline in the source is 14 September 2026**, now past. The file does not establish any later evaluation or submission deadline. Confirm the current one with faculty.

## Progress and realistic next actions

| Component | Status | Required action |
|---|---|---|
| CO3 /10: data and implementation | Verified | None required beyond clean evidence presentation and reproducibility |
| CO3 /10: selected model and test | Verified, fold 1 only | Report all metrics, confusion matrix and limitations |
| CO4 /5: published methods, same dataset | Unfinished | Complete original-paper comparison and matched-protocol caveats; ideally ten-fold test |
| CO5 /5: layer-freezing ablations | Three controlled validation runs completed | Describe parameter/cost vs gain and single-fold weakness; optional cross-fold robustness |
| Notebook and final project README | In progress | Insert verified test metrics, CO4 table, CO5 contributions, training settings, figure paths |
| Source, requirements and checkpoint instructions | Present, full package not audited | Test reproducibility on a clean environment; document safe checkpoint access instead of blindly ZIPping 1.4GB weights |
| Student metadata | Not supplied | Fill actual name, registration number and section (do not invent) |
| Submission ZIP | Pending | Archive required files with exact filename `RegistrationNumber.zip`; review internal tree |

## Dataset and evaluation caveats

- Official class IDs 0–9, 8732 recordings, 10 original folds; 8 train / 1 validation / 1 test in current run.
- Checkpoint provenance is documented by official PANNs commit and recorded SHA-256.
- Audio input is mono 32 kHz, 128000 samples; fixed padding/cropping rules.
- The source-isolation audit found five `fsID` values crossing original folds. Individual clip partitions are disjoint; full source independence is not established.
- The current 88.55% is from test fold 1 only; don't label as mean official ten-fold UrbanSound8K accuracy.
- `fc1` and `block6` improvement measured on *validation fold 2*, not separate held-out tests of all configurations.

## Mandatory deliverables from the faculty source

- [x] Selected dataset and class/fold analyses
- [x] PyTorch official pretrained CNN14 implementation
- [x] Verified partial fine-tuning and backpropagation
- [x] Three controlled freezing-depth configurations
- [x] Training/validation histories and curves from six epochs
- [x] Selected-checkpoint held-out test summary (accuracy, precision, recall, F1, specificity and ROC-AUC)
- [x] Per-class results and confusion matrix
- [ ] CO4 original-paper comparative analysis with references and protocol labels
- [ ] CO5 discussion: quantified trade-offs, contribution and limitations; optional multi-fold robustness
- [ ] Verified final README with student details, optimizer, LR, batch, epochs, weight decay, scheduler/dropout if any, and rationale for settings
- [ ] Explicit hyperparameter tuning documentation: distinguish values used from values genuinely tested
- [ ] Final reproducible notebooks, source, exact `requirements.lock.txt` or versions, dataset/checkpoint acquisition guidance
- [ ] Consistent result paths and source-audit warning included
- [ ] Final package, opening and replay tests
- [ ] Exact ZIP file name of registration number only

## Time planning (human/student effort estimates, not delivery commitments)

- Literature verification and CO4: about 3–5 focused hours.
- CO5 synthesis, numerical consistency and limitations: about 2–3 focused hours.
- Notebook/README/result integration, final audit, packaging and demonstration rehearsal: about 3–6 focused hours.
- Thus roughly **8–14 hours of hands-on remaining work** for a transparent, single-fold submission, if no new experiments are needed.
- Ten-fold complete benchmark experimentation adds GPU compute, experiment inspection and documentation; estimate separately from the student's writing time. Based on six-epoch single-fold histories, the additional nine-fold `block6` training alone would take roughly 4–5 hours of GPU execution at the previously measured speed; doing all three configurations over the additional nine folds is roughly 10 hours, before failures, setup or analysis. These are extrapolations, not executed results.
