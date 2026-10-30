"""Idempotently synchronize the faculty CO3/CO4/CO5 section into root README.md."""
from pathlib import Path
START = '<!-- FACULTY_EVIDENCE_START -->'
END = '<!-- FACULTY_EVIDENCE_END -->'


def section() -> str:
    return '''## Faculty evaluation summary — CO3 / CO4 / CO5 (30 October 2026)

**Project:** Deep Learning Framework for Environmental Sound Recognition Using PyTorch.  
**Dataset:** UrbanSound8K (8,732 recordings; official 10 classes/folds), **model:** official AudioSet-pretrained PANNs CNN14, **framework:** PyTorch.

**CO3 — Implementation and held-out evaluation.** Full audio audit decoded all 8,732 files. Train folds 3–10: 6,971; validation fold 2: 888; held-out test fold 1: 873. 32 kHz mono four-second waveforms. Selected checkpoint: `conv_block6` + `fc1` + ten-class head, best epoch 4 on validation macro-F1. **Fold-1 test: accuracy 88.55%, macro-F1 0.8964, macro precision 0.9008, macro recall 0.9005, OvR macro specificity 0.9870, OvR macro ROC-AUC 0.9909**. Full figures and exact values: `reports/faculty_evidence/heldout/`.

**CO5 — Ablation.** Head-only / fc1 / block6 validation accuracy **87.73% / 88.06% / 90.65%**, macro-F1 **0.8638 / 0.8668 / 0.9010**. Trainable parameters **20,490 / 4,216,842 / 60,848,138**. Settings held constant: seed 67, six epochs, BF16, 32 kHz, batch 1 + accumulation 4, AdamW, CrossEntropyLoss, head LR 1e-3, backbone LR 1e-5, weight decay 1e-4. See `docs/CO5_ABLATION_AND_DISCUSSION.md` for trainability, cost, and limitations; this is an experimental layer-depth contribution, **not a new CNN algorithm**.

**CO4 — Same-dataset papers.** Su et al. 2019 (TSCNN-DS, 97.2%), Ting et al. 2021 (Inception–Dense, 83.03% mean), Fang et al. 2022 (RACNN, 97.51% ± 0.18%). See `docs/CO4_PUBLISHED_METHODS.md` for original citations, protocol, feature and pretraining differences. Published multi-fold scores **must not be ranked directly against our one-fold result**.

**Known limits.** The official dataset advises ten-fold cross-validation; this project has **one held-out test fold**, not a ten-fold mean. Five original Freesound source IDs cross official folds; source-level isolation is not guaranteed. The selected test fold was evaluated only after choosing the `block6` configuration using validation fold 2.

**Hyperparameter tuning disclosure.** The *freezing depth* was varied across three controlled runs. The learning rates, batch size, weight decay, and seed were **held fixed**, not grid-searched. Checkpoint selection used validation macro-F1.

**Reproduction and submission paths.** `notebooks/01_...ipynb` (dataset/preprocessing/Stage 4), `notebooks/02_...ipynb` (validation/test), `notebooks/03_...ipynb` (CO4/CO5/submission), `src/`, `tests/`, `requirements.txt`, `requirements.lock.txt`, `models/CHECKPOINT_ACCESS.md`, and `data/README_DATASET.md`. Exact project metadata and package naming are enforced by `tools/faculty_submission.py`.
'''


def update_readme(root: Path) -> Path:
    path = Path(root) / 'README.md'
    if not path.is_file():
        raise FileNotFoundError(f'Missing existing root README; refusing to replace other documentation: {path}')
    content = path.read_text(encoding='utf-8')
    managed = f'{START}\n{section()}\n{END}'
    if START in content and END in content:
        a = content.index(START)
        b = content.index(END, a) + len(END)
        content = content[:a] + managed + content[b:]
    elif START in content or END in content:
        raise ValueError('README has only one faculty marker; manual review required')
    else:
        content = content.rstrip() + '\n\n---\n\n' + managed + '\n'
    path.write_text(content, encoding='utf-8')
    return path


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    print('Updated:', update_readme(a.project))
