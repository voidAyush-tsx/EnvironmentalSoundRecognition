"""Audit and package a faculty-ready UrbanSound8K/PANNs CNN14 project.

Only package after testing Notebook 01/02/03 and evaluating actual checkpoints.
This script NEVER trains, reads all WAVs, fetches URLs, or unpickles checkpoints.
"""
from __future__ import annotations
import hashlib
import json
import re
import sys
import zipfile
import zlib
from pathlib import Path

from .config import EXPECTED_PANN_COMMIT as COMMIT, SELECTED_SHA256
from .artifacts import read_json, sha256_file
from .evidence import selected_checkpoint

REQUIREMENT_FILES = (
    'requirements.txt', 'requirements.lock.txt', 'requirements-cpu.txt', 'requirements-common.txt',
)
STANDARD = [
    'README.md', *REQUIREMENT_FILES,
    'docs/DEVELOPMENT.md', 'docs/REPRODUCIBILITY.md', 'docs/RESEARCH.md',
    'notebooks/01_Project_Overview.ipynb',
    'notebooks/02_Model_Evaluation.ipynb',
    'notebooks/03_Faculty_Submission.ipynb',
    'src/dataset.py', 'src/finetune_train.py', 'src/evaluate_selected.py',
    'reports/faculty_evidence/heldout/evaluation_summary.json',
    'reports/faculty_evidence/heldout/per_class_metrics.csv',
    'reports/faculty_evidence/heldout/confusion_matrix.png',
]


def locate_checkpoint(root: Path) -> Path | None:
    return selected_checkpoint(root)


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
    for fname in ('README.md', *REQUIREMENT_FILES, '.gitignore', '.gitmodules'):
        f = root / fname
        if f.is_file():
            files.add(f)
    for folder, suffixes in (
        ('src', ('.py',)), ('tests', ('.py',)),
        ('notebooks', ('.ipynb',)), ('docs', ('.md', '.csv', '.json')),
        ('reports/faculty_evidence', ('.md', '.csv', '.json', '.png')),
        ('reports/stage4_comparison', ('.md', '.csv', '.png')),
        ('figures', ('.png', '.jpg', '.jpeg', '.svg')),
    ):
        d = root / folder
        if d.exists():
            for p in d.rglob('*'):
                if p.is_file() and p.suffix.lower() in suffixes and not any(
                    part in {'__pycache__', '.ipynb_checkpoints', '.git', '.venv'} for part in p.parts
                ):
                    files.add(p)
    # Include only named reports from user's selected held-out evaluation.
    results = root / 'results' / 'stage5'
    if results.exists():
        selected_dirs = []
        for summary_path in results.glob('*/evaluation_summary.json'):
            record = read_json(summary_path)
            if record.get('stage4_best_checkpoint_sha256') == SELECTED_SHA256:
                selected_dirs.append(summary_path.parent)
        for p in (file for folder in selected_dirs for file in folder.rglob('*')):
            if p.is_file() and p.name in {
                'evaluation_summary.json', 'per_class_metrics.csv', 'confusion_matrix.csv',
                'confusion_matrix.png', 'confusion_matrix_normalized.png',
                'per_class_f1.png', 'roc_curves_ovr.png', 'test_predictions.csv'
            }:
                files.add(p)
    # Do not ship the legacy ESC-50 notebook or stale lockfiles hidden among other files.
    return sorted(files)


def student_section(name: str, registration: str, section: str) -> str:
    return f'''\n\n---\n## Faculty submission metadata and final experimental summary\n\n- **Project title:** Deep Learning Framework for Environmental Sound Recognition Using PyTorch\n- **Student name:** {name}\n- **Registration number:** {registration}\n- **Section:** {section}\n- **Dataset:** UrbanSound8K; 8,732 clips, 10 classes, official folds 1–10.\n- **Split:** 6,971 train (folds 3–10), 888 validation (fold 2), 873 test (fold 1).\n- **Architecture:** Official PANNs CNN14, AudioSet-pretrained at pinned commit `{COMMIT}`; 2048-dimensional embedding and a new ten-class head.\n- **Preprocessing:** mono waveform, 32 kHz, four seconds (128,000 samples), resampling, center cropping/right padding; CNN14 computes its own log-Mel frontend.\n- **Training:** AdamW with head LR 0.001, backbone LR 0.00001, weight decay 0.0001; CrossEntropyLoss; batch size 1, gradient accumulation 4, maximum 6 epochs, early stopping patience 2, BF16, seed 67; checkpoint chosen by validation macro-F1 (epoch 4).\n- **CO3 held-out result:** Accuracy **88.55%**, macro precision **0.9008**, macro recall **0.9005**, macro-F1 **0.8964**, macro specificity **0.9870**, macro one-vs-rest ROC-AUC **0.9909**, 873 examples (test fold 1 only). See `reports/faculty_evidence/heldout/`.\n- **CO5 experimental contribution:** controlled head-only vs `fc1` vs `conv_block6` freezing ablation. Validation accuracies 87.73% / 88.06% / 90.65%; macro-F1 0.8638 / 0.8668 / 0.9010. This is a justified fine-tuning-depth experiment, **not a novel CNN architecture**; see `docs/RESEARCH.md#freezing-depth-study`.\n- **CO4 comparison:** see `docs/RESEARCH.md#published-methods` for original-paper references, cross-protocol caveats and unreported fields (NR). The single-fold test score is not a 10-fold average.\n- **Limitations:** five Freesound `fsID` values cross predefined folds; train/test are disjoint by filename but perfect original-source isolation is unverified. No ten-fold or multi-seed benchmark was performed.\n- **Reproduction and folder guide:** `notebooks/01_...ipynb` preprocessing and checks; `notebooks/02_...ipynb` selection/test; `notebooks/03_...ipynb` CO4/CO5; `src/` and `tests/` implementations; `docs/REPRODUCIBILITY.md#checkpoint-reconstruction` for model access; `docs/REPRODUCIBILITY.md#dataset` for data source; `reports/faculty_evidence/` contains verified small experiment evidence. `requirements.txt` and `requirements.lock.txt` describe dependencies.\n- **Scheduler/dropout disclosure:** No scheduler is used; official CNN14 training-mode dropout/SpecAugmentation is retained. See `docs/RESEARCH.md#training-details` before submission.
- **Hyperparameter tuning disclosure:** Three layer-freezing choices were tested at **fixed** LR, weight decay, batch size and seed. These settings were not independently grid-searched. The final block6 method was chosen using validation fold 2; its test fold was not used for tuning.\n'''


def inspect_archive(path: Path, registration: str, name: str, section: str) -> dict:
    """Read and validate a faculty archive without extracting or unpickling it.

    Members are streamed for CRC verification. The selected parameter checkpoint
    is read once and its SHA-256 must match the recorded evaluation.
    This deliberately reads the whole archive, so use it only for an explicit
    packaging/verification action, not ordinary notebook review.
    """
    path = Path(path)
    if not re.fullmatch(r'[0-9]{6,20}', registration):
        raise ValueError('Registration number must contain 6–20 digits only')
    if not name.strip() or not section.strip():
        raise ValueError('Student name and section are required to inspect the archive')
    if path.name != f'{registration}.zip':
        raise ValueError(f'ZIP must be named exactly {registration}.zip')
    prefix = f'{registration}/'
    checkpoint_name = prefix + 'models/selected_block6_best.pt'
    required = set(STANDARD) | {'models/selected_block6_best.pt', 'external/PANNs_PINNED_SOURCE.md'}
    for method in ('head', 'fc1', 'block6'):
        for filename in ('config.json', 'summary.json', 'history.csv'):
            required.add(f'reports/faculty_evidence/training_runs/{method}/{filename}')
    digest = hashlib.sha256()
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            filenames = [entry.filename for entry in members]
            if len(filenames) != len(set(filenames)):
                raise ValueError('Submission archive contains duplicate file paths')
            invalid_paths = [filename for filename in filenames
                             if not filename.startswith(prefix) or '..' in filename.split('/')
                             or '\\' in filename]
            if invalid_paths:
                raise ValueError('Unexpected archive paths: ' + ', '.join(invalid_paths))
            missing = sorted(relative for relative in required if prefix + relative not in filenames)
            if missing:
                raise ValueError('Submission archive is missing required files: ' + ', '.join(missing))
            unwanted = [filename for filename in filenames
                        if any(part.lower() in {'__pycache__', '.ipynb_checkpoints', '.git', '.venv'}
                               for part in filename.split('/'))
                        or filename.lower().endswith(('.wav', '.pyc', '.pyo'))]
            if unwanted:
                raise ValueError('Submission archive contains raw audio or caches: ' + ', '.join(unwanted))
            for entry in members:
                if entry.is_dir():
                    continue
                with archive.open(entry) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        if entry.filename == checkpoint_name:
                            digest.update(chunk)
            if digest.hexdigest() != SELECTED_SHA256:
                raise ValueError('CHECKPOINT SHA256 MISMATCH: archived best.pt is not the evaluated checkpoint')
            readme = archive.read(prefix + 'README.md').decode('utf-8')
            identity_lines = (f'**Student name:** {name}', f'**Registration number:** {registration}',
                              f'**Section:** {section}')
            if not all(line in readme for line in identity_lines):
                raise ValueError('Student identity does not match the archived README')

            def archived_json(relative: str) -> dict:
                value = json.loads(archive.read(prefix + relative).decode('utf-8'))
                if not isinstance(value, dict):
                    raise ValueError(f'Expected a JSON object in archived {relative}')
                return value

            evaluation = archived_json('reports/faculty_evidence/heldout/evaluation_summary.json')
            expected = {'test_evaluated': True, 'config': 'block6', 'test_fold': 1,
                        'validation_fold': 2, 'test_samples': 873, 'checkpoint_epoch': 4,
                        'model_source_commit': COMMIT, 'stage4_best_checkpoint_sha256': SELECTED_SHA256}
            if any(evaluation.get(key) != value for key, value in expected.items()):
                raise ValueError('Archived held-out evidence provenance does not match the selected checkpoint')
            for method in ('head', 'fc1', 'block6'):
                base = f'reports/faculty_evidence/training_runs/{method}/'
                config, summary = archived_json(base + 'config.json'), archived_json(base + 'summary.json')
                if (config.get('config') != method or config.get('diagnostic_only') is not False
                        or config.get('test_fold') != 1 or config.get('validation_fold') != 2
                        or config.get('model_source_commit') != COMMIT
                        or summary.get('best_epoch') != 4 or summary.get('test_evaluated') is not False):
                    raise ValueError(f'Archived training provenance is invalid for {method}')
    except (OSError, zipfile.BadZipFile, RuntimeError, EOFError, zlib.error,
            UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Invalid submission archive {path}: {exc}') from exc
    return {'output': str(path), 'count': len(members), 'bytes': path.stat().st_size,
            'checkpoint_sha256': digest.hexdigest(), 'identity_verified': True, 'integrity': 'PASS'}


def make_archive(root: Path, out: Path, registration: str, name: str, section: str, checkpoint: Path) -> dict:
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing submission archive: {out}')
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
    with zipfile.ZipFile(out, 'x', zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
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
                   'This source must be cloned separately. See docs/REPRODUCIBILITY.md#checkpoint-reconstruction.\n')
    return inspect_archive(out, registration, name, section)


def main(argv=None) -> int:
    from .arguments import faculty_parser
    p = faculty_parser()
    a = p.parse_args(argv)
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




START = '<!-- FACULTY_EVIDENCE_START -->'
END = '<!-- FACULTY_EVIDENCE_END -->'


def section() -> str:
    return """## Research evidence

Read the recorded [held-out metrics](reports/faculty_evidence/heldout/evaluation_summary.json),
[method and limitations](docs/RESEARCH.md), and
[faculty submission guidance](docs/RESEARCH.md#faculty-submission).
These are historical results from one held-out fold; no ten-fold mean is claimed.
"""


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
