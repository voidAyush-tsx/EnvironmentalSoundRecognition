"""Audit and package a faculty-ready UrbanSound8K/PANNs CNN14 project.

Only package after testing Notebook 01/02/03 and evaluating actual checkpoints.
This script NEVER trains, reads all WAVs, fetches URLs, or unpickles checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

COMMIT = 'd2f4b8c18eab44737fcc0de1248ae21eb43f6aa4'
SELECTED_SHA256 = '7c0e01a0078c225a18f4dc5eaa5110d051ef2c8effec7bcebb4e856e6659328c'
STANDARD = [
    'README.md', 'requirements.txt', 'requirements.lock.txt',
    'notebooks/01_UrbanSound8K_PANNs_CNN14_Project.ipynb',
    'notebooks/02_Stage4_Review_and_Stage5_HeldOut_Evaluation.ipynb',
    'notebooks/03_CO4_CO5_Faculty_Submission.ipynb',
    'src/dataset.py', 'src/finetune_train.py', 'src/evaluate_selected.py',
    'docs/CO4_PUBLISHED_METHODS.md', 'docs/CO5_ABLATION_AND_DISCUSSION.md',
    'docs/TRAINING_DETAILS_TO_VERIFY.md',
    'reports/faculty_evidence/heldout/evaluation_summary.json',
    'reports/faculty_evidence/heldout/per_class_metrics.csv',
    'reports/faculty_evidence/heldout/confusion_matrix.png',
    'models/CHECKPOINT_ACCESS.md', 'data/README_DATASET.md',
]


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding='utf-8'))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(2 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def locate_checkpoint(root: Path) -> Path | None:
    run = root / 'results' / 'stage4' / '20261027_144004_603447_block6_test1' / 'best.pt'
    if run.is_file():
        return run
    matches = sorted((root / 'results' / 'stage4').glob('*_block6_test1/best.pt'))
    return matches[-1] if matches else None


def audit(root: Path, checkpoint: Path | None = None, *, verify_checkpoint: bool = False) -> list[str]:
    errors = []
    for item in STANDARD:
        if not (root / item).is_file():
            errors.append('MISSING: ' + item)
    for method in ('head', 'fc1', 'block6'):
        folder = root / 'reports' / 'faculty_evidence' / 'training_runs' / method
        for fn in ('config.json', 'summary.json', 'history.csv'):
            if not (folder / fn).is_file():
                errors.append(f'MISSING: reports/faculty_evidence/training_runs/{method}/{fn}')
        cfg = folder / 'config.json'
        summary = folder / 'summary.json'
        if cfg.is_file() and summary.is_file():
            try:
                c, s = read_json(cfg), read_json(summary)
                if c.get('config') != method or c.get('diagnostic_only') is not False or c.get('test_fold') != 1:
                    errors.append(f'INVALID full-fold configuration: {method}')
                if s.get('best_epoch') != 4 or s.get('test_evaluated') is not False:
                    errors.append(f'INVALID validation-selected checkpoint summary: {method}')
            except (OSError, ValueError) as e:
                errors.append(f'CORRUPT JSON for {method}: {e}')
    e = root / 'reports' / 'faculty_evidence' / 'heldout' / 'evaluation_summary.json'
    if e.is_file():
        try:
            j = read_json(e)
            if not (j.get('test_evaluated') is True and j.get('config') == 'block6' and
                    j.get('test_fold') == 1 and j.get('test_samples') == 873 and
                    j.get('checkpoint_epoch') == 4 and j.get('stage4_best_checkpoint_sha256') == SELECTED_SHA256):
                errors.append('INVALID: held-out evidence provenance does not match selected checkpoint')
        except (OSError, ValueError) as exc:
            errors.append('CORRUPT held-out JSON: ' + str(exc))
    if verify_checkpoint:
        if not checkpoint or not checkpoint.is_file():
            errors.append('MISSING selected fine-tuned best.pt. Specify --checkpoint or restore the original run folder.')
        elif sha256_file(checkpoint).lower() != SELECTED_SHA256:
            errors.append('CHECKPOINT SHA256 MISMATCH: selected best.pt is not the evaluated checkpoint')
    return errors


def collect_files(root: Path) -> list[Path]:
    """Allowlist human-authored project files and small evidentiary results; no caches or raw audio."""
    files: set[Path] = set()
    for fname in ('README.md', 'requirements.txt', 'requirements.lock.txt', '.gitignore',
                  'run_all_checks.py', 'run_stage4_checks.py', 'check_dependencies.py',
                  'check_urbansound8k.py', 'finalize_requirements.py',
                  'STAGE3_README.md', 'STAGE4_README.md',
                  'NOTEBOOK_QUICKSTART.md', 'STAGE5_NOTEBOOK_GUIDE.md'):
        f = root / fname
        if f.is_file():
            files.add(f)
    for folder, suffixes in (
        ('src', ('.py',)), ('tests', ('.py',)),
        ('notebooks', ('.ipynb',)), ('docs', ('.md', '.csv', '.json')),
        ('reports/faculty_evidence', ('.md', '.csv', '.json', '.png')),
        ('reports/stage4_comparison', ('.md', '.csv', '.png')),
        ('figures', ('.png', '.jpg', '.jpeg', '.svg')),
        ('tools', ('.py',)), ('models', ('.md',)),
    ):
        d = root / folder
        if d.exists():
            for p in d.rglob('*'):
                if p.is_file() and p.suffix.lower() in suffixes and not any(
                    part in {'__pycache__', '.ipynb_checkpoints', '.git', '.venv'} for part in p.parts
                ):
                    files.add(p)
    f = root / 'data' / 'README_DATASET.md'
    if f.is_file():
        files.add(f)
    # Include only named reports from user's selected held-out evaluation.
    results = root / 'results' / 'stage5'
    if results.exists():
        for p in results.rglob('*'):
            if p.is_file() and p.name in {
                'evaluation_summary.json', 'per_class_metrics.csv', 'confusion_matrix.csv',
                'confusion_matrix.png', 'confusion_matrix_normalized.png',
                'per_class_f1.png', 'roc_curves_ovr.png', 'test_predictions.csv'
            }:
                files.add(p)
    # Do not ship the legacy ESC-50 notebook or stale lockfiles hidden among other files.
    return sorted(files)


def student_section(name: str, registration: str, section: str) -> str:
    return f'''\n\n---\n## Faculty submission metadata and final experimental summary\n\n- **Project title:** Deep Learning Framework for Environmental Sound Recognition Using PyTorch\n- **Student name:** {name}\n- **Registration number:** {registration}\n- **Section:** {section}\n- **Dataset:** UrbanSound8K; 8,732 clips, 10 classes, official folds 1–10.\n- **Split:** 6,971 train (folds 3–10), 888 validation (fold 2), 873 test (fold 1).\n- **Architecture:** Official PANNs CNN14, AudioSet-pretrained at pinned commit `{COMMIT}`; 2048-dimensional embedding and a new ten-class head.\n- **Preprocessing:** mono waveform, 32 kHz, four seconds (128,000 samples), resampling, center cropping/right padding; CNN14 computes its own log-Mel frontend.\n- **Training:** AdamW with head LR 0.001, backbone LR 0.00001, weight decay 0.0001; CrossEntropyLoss; batch size 1, gradient accumulation 4, maximum 6 epochs, early stopping patience 2, BF16, seed 67; checkpoint chosen by validation macro-F1 (epoch 4).\n- **CO3 held-out result:** Accuracy **88.55%**, macro precision **0.9008**, macro recall **0.9005**, macro-F1 **0.8964**, macro specificity **0.9870**, macro one-vs-rest ROC-AUC **0.9909**, 873 examples (test fold 1 only). See `reports/faculty_evidence/heldout/`.\n- **CO5 experimental contribution:** controlled head-only vs `fc1` vs `conv_block6` freezing ablation. Validation accuracies 87.73% / 88.06% / 90.65%; macro-F1 0.8638 / 0.8668 / 0.9010. This is a justified fine-tuning-depth experiment, **not a novel CNN architecture**; see `docs/CO5_ABLATION_AND_DISCUSSION.md`.\n- **CO4 comparison:** see `docs/CO4_PUBLISHED_METHODS.md` for original-paper references, cross-protocol caveats and unreported fields (NR). The single-fold test score is not a 10-fold average.\n- **Limitations:** five Freesound `fsID` values cross predefined folds; train/test are disjoint by filename but perfect original-source isolation is unverified. No ten-fold or multi-seed benchmark was performed.\n- **Reproduction and folder guide:** `notebooks/01_...ipynb` preprocessing and checks; `notebooks/02_...ipynb` selection/test; `notebooks/03_...ipynb` CO4/CO5; `src/` and `tests/` implementations; `models/CHECKPOINT_ACCESS.md` for model access; `data/README_DATASET.md` for data source; `reports/faculty_evidence/` contains verified small experiment evidence. `requirements.txt` and `requirements.lock.txt` describe dependencies.\n- **Scheduler/dropout disclosure:** Their exact values were not stored in the shared run JSON; verify the source and complete `docs/TRAINING_DETAILS_TO_VERIFY.md` before submission.
- **Hyperparameter tuning disclosure:** Three layer-freezing choices were tested at **fixed** LR, weight decay, batch size and seed. These settings were not independently grid-searched. The final block6 method was chosen using validation fold 2; its test fold was not used for tuning.\n'''


def make_archive(root: Path, out: Path, registration: str, name: str, section: str, checkpoint: Path) -> dict:
    if not re.fullmatch(r'[0-9]{6,20}', registration):
        raise ValueError('Registration number must contain 6–20 digits only; no names or prefixes')
    if not name.strip() or not section.strip():
        raise ValueError('Student name and section are mandatory for faculty submission')
    errors = audit(root, checkpoint, verify_checkpoint=True)
    if errors:
        raise ValueError('Faculty preflight failed:\n  ' + '\n  '.join(errors))
    files = collect_files(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.name != f'{registration}.zip':
        raise ValueError(f'ZIP must be named exactly {registration}.zip')
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for f in files:
            rel = f.relative_to(root).as_posix()
            if rel == 'README.md':
                text = f.read_text(encoding='utf-8') + student_section(name, registration, section)
                z.writestr(f'{registration}/README.md', text.encode('utf-8'))
            else:
                z.write(f, f'{registration}/{rel}')
        z.write(checkpoint, f'{registration}/models/selected_block6_best.pt', compress_type=zipfile.ZIP_STORED)
        z.writestr(f'{registration}/external/PANNs_PINNED_SOURCE.md',
                   f'Official repository: https://github.com/qiuqiangkong/audioset_tagging_cnn\nPinned commit: {COMMIT}\n'
                   'This source must be cloned separately. See models/CHECKPOINT_ACCESS.md.\n')
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        if bad:
            raise IOError('ZIP integrity check failed at ' + bad)
    return {'output': str(out), 'count': len(files) + 2, 'bytes': out.stat().st_size, 'checkpoint_sha256': SELECTED_SHA256}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['audit', 'package'])
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--checkpoint', type=Path, default=None, help='Optional path to exact Stage 5 selected block6 best.pt')
    p.add_argument('--registration', default='', help='Registration number (digits only)')
    p.add_argument('--student-name', default='')
    p.add_argument('--section', default='')
    a = p.parse_args()
    root = a.project.resolve()
    chk = (a.checkpoint.resolve() if a.checkpoint is not None else locate_checkpoint(root))
    if a.command == 'audit':
        errors = audit(root, chk, verify_checkpoint=False)
        print(f'Faculty files audit: {len(STANDARD)} required paths + 3 experiment runs')
        print('Evidence files:', 'PASS' if not errors else 'INCOMPLETE')
        for x in errors:
            print('  ', x)
        print('Selected trainable checkpoint:', str(chk) if chk else 'NOT LOCATED locally')
        print('Student details / notebook execution / external-source license: must be checked before submission')
        print('NOTE: Passing this audit does not prove notebook replay or full 10-fold CV.')
        return 1 if errors else 0
    try:
        report = make_archive(root, root / f'{a.registration}.zip', a.registration, a.student_name, a.section, chk)
    except (OSError, ValueError, TypeError) as e:
        print('SUBMISSION BLOCKED:', e)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
