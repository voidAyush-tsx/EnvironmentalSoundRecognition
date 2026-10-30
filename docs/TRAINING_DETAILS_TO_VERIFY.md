# Final hyperparameter verification — do not fabricate settings

The three uploaded full-fold `config.json` files **prove** the following values (all three runs):

| Training choice | Verified value | Evidence |
|---|---|---|
| Optimizer | **AdamW** was documented in Stage 4 implementation; verify actual `src/finetune_train.py` before finalizing faculty README | Original training source, not exported config field |
| Loss | **CrossEntropyLoss** was documented in Stage 4 implementation; verify actual `src/finetune_loop.py` or trainer before finalizing faculty README | Original training source, not exported config field |
| Learning rate (classifier) | **0.001** | `config.json`: `lr_head` |
| Learning rate (later trainable CNN layers) | **0.00001** | `config.json`: `lr_backbone` |
| Weight decay | **0.0001** | `config.json`: `weight_decay` |
| Batch size | **1** | `config.json`: `batch_size` |
| Gradient accumulation | **4** (effective gradient batch target = 4) | `config.json`: `accum_steps` |
| Epoch budget | **6** requested and completed | `summary.json`, `history.csv` |
| Early stopping patience | **2** | `config.json`: `patience` |
| Precision | **BF16 CUDA** | `config.json`: `precision` |
| Seed | **67** | `config.json`: `seed` |
| Checkpoint criterion | Best validation macro-F1; epoch **4** | `summary.json` |
| Augmentation policy | CNN14 internal SpecAugmentation and dropout operated in training mode according to real gradient-check outputs; validation/test in eval mode | Stage 4 checker logs |
| Scheduler | **NOT recorded in the uploaded config JSON**; inspect the actual optimizer/trainer code to confirm whether one was used and its parameters; do not infer 'None' blindly. | Source inspection required |
| Dropout probability | **NOT recorded in uploaded config JSON**; inspect official CNN14 `pytorch/models.py` and the wrapper forward path to report the active rates and whether train/eval differs. | Source inspection required |

**Action in Notebook 01 or VS Code:** inspect and copy the relevant `optimizer=...`, loss criterion, scheduler initialization/step, and `F.dropout(..., p=...)` source lines into this document/README; don't claim a hyperparameter sweep unless it was actually run. A fixed set of values plus three freezing-depth ablations is a valid controlled design but not a learning-rate search.
