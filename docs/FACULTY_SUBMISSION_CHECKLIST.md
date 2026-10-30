# Faculty compliance checklist — authoritative source: `project_submission_instructions`

## Grade components /20

| Teacher requirement | What this project actually has | Faculty packaging status |
|---|---|---|
| **CO3 (10):** Dataset | UrbanSound8K, 8,732 recordings, 10 classes, original folds | Verified in Stage 2–3 on student's Windows PC; dataset summary included |
| **CO3:** Data distribution | Fold counts, waveform durations/sample rates, per-class distributions from Notebook 01 | Include notebook outputs / figures; verify in ZIP before submission |
| **CO3:** Preprocessing | 32-kHz mono waveform, center crop/right pad to 128,000 samples | Code and tests were executed successfully on student's PC |
| **CO3:** Architecture and pretrained weights | Official AudioSet PANNs CNN14, pinned Git commit, verified strict load | Code and checkpoint provenance; do not upload original 1.4GB weights by accident |
| **CO3:** Actual training | Head, fc1, block6; six full epochs each, selection on validation | Real `history.csv`, best epoch 4, and pretrained conv update proven |
| **CO3:** Test and metrics | block6 selected fold-1 test, 873 clips; 88.55% accuracy, macro-F1 0.8964, OvR ROC-AUC 0.9909 | Evidence files packaged; class-wise metrics and confusion matrix included |
| **CO3:** Train/validation curves and convergence | Three full run histories and figures | Available as evidence; note validation loss rising after epoch 4 |
| **CO4 (5):** Same dataset comparison | Original UrbanSound8K papers: Su 2019, Ting 2021, Fang 2022 | `docs/CO4_PUBLISHED_METHODS.md`; only contextual across protocols |
| **CO4:** Precision/recall/F1/AUC model comparison | Our values available; published papers do not give directly matching macro metrics | Label published cells `NR`; do not fabricate |
| **CO5 (5):** Justified innovation | Controlled layer-freezing depth study, 3 matched runs | Quantified contribution, trainable params, compute costs, limitations |
| **CO5:** Ablations | Frozen head / fc1 / block6 | Only validation split compared; do not claim multi-fold proof |
| **README:** Identity | Project title known; **student name, registration, section not supplied** | Must be filled by actual student; packager requires them |
| **README:** Dataset/model/hyperparameters/hyperparameter tuning/structure | Documented in addendum, CO4/CO5 docs, model/data access docs; **scheduler/dropout details need local source inspection** | Run `tools/update_readme.py` and `docs/TRAINING_DETAILS_TO_VERIFY.md` |
| **requirements.txt** | Working package ranges + exact `requirements.lock.txt` | Include both and verify kernel interpreter; Python 3.13.15 |
| **Code, notebooks, reproducibility** | Original `src`, `tests`, 3 notebooks, usage docs | Run Notebook 03 preflight; verify Notebook 01/02 saved outputs |
| **ZIP only, exact name `RegistrationNumber.zip`** | Built by opt-in packaging script after metadata and checkpoint checks | **Not built yet; student registration missing** |

## Required project-specific disclosures

- Fold 1 is **one** test fold, not 10-fold mean; ten-fold cross-validation is recommended by the dataset creators but **not yet performed**.
- Our chosen configuration used AudioSet pretraining, unlike the published models where such pretraining may not be reported.
- Five original Freesound source IDs span original folds; clip-level splits are disjoint, but source-level independence is not established.
- The actual variable under study was CNN14 fine-tuning depth. AdamW LR, batch size, weight decay and seed were held fixed, **not** separately grid-searched.
- `block6` fine-tuning is **not parameter-efficient** (~74.3% trainable); claim improved validation macro-F1 for the observed split, not an invented algorithm.

## Final signoff on student's PC

1. Open Notebook 01 (dataset + Stage 2–4) and Notebook 02 (Stage 4 comparison + Stage 5). Ensure code cells and saved images render; save both.
2. Extract this kit into the project root without overwriting the existing README. Open Notebook 03 and run its safe default cells; read research notes.
3. Update the root README through Notebook 03's opt-in section, then fill actual name, registration number, section via notebook controls.
4. Ensure `results/stage4/20261027_144004_603447_block6_test1/best.pt` exists and passes the recorded SHA-256; this is the evaluated trainable-parameter checkpoint.
5. Run the faculty audit from Notebook 03 and resolve any missing path; inspect `reports/faculty_evidence` and charts in `figures/`.
6. Enable only the final package cell, yielding `<RegistrationNumber>.zip`, with **no additional words** in the ZIP name. Verify archive contents and document checksum.
7. Open the ZIP, check that `README.md`, requirements, three notebooks, source, tests, metrics, figure, CO4/CO5 papers/analysis, data instructions, pretrained checkpoint access, and selected fine-tuned `best.pt` are present.
8. If the faculty expects a fully offline/portable training environment, the external CNN14 weights and WAVs must be provided separately per their licensing/download instructions. The 1.4GB AudioSet base weights are **not** bundled by default; make this explicit before submission.

**No final submission readiness claim is made by this document alone.** Replaying locally and passing the preflight are still mandatory.
