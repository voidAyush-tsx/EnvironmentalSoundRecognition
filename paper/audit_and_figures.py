"""Recompute manuscript evidence; never train or run held-out inference.

Run from the repository root: python paper/audit_and_figures.py
Requires the existing project dependencies and local metadata/audio/checkpoints.
Report outputs are confined to paper/; the JIT cache uses tmp/paper-tools/.
Original experiment records are read-only.
"""
from pathlib import Path
import hashlib
import itertools
import json
import subprocess
import sys
import os

# Keep JIT caches within the writable workspace, not the system Python install.
os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(__file__).resolve().parents[1] / 'tmp/paper-tools/numba-cache'))

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper'
sys.path.insert(0, str(ROOT))
from src.config import CLASS_NAMES

for folder in ('figures', 'tables', 'audit'):
    (OUT / folder).mkdir(exist_ok=True, parents=True)

sources = set()
def read_csv(path, **kwargs):
    p = ROOT / path
    sources.add(p)
    return pd.read_csv(p, **kwargs)

def read_json(path):
    p = ROOT / path
    sources.add(p)
    return json.loads(p.read_text(encoding='utf-8'))

def digest(path, algorithm='sha256'):
    with path.open('rb') as f:
        return hashlib.file_digest(f, algorithm).hexdigest()

def same(a, b, atol=1e-10):
    assert np.allclose(a, b, rtol=0, atol=atol), (a, b)

meta = read_csv('data/UrbanSound8K/metadata/UrbanSound8K.csv')
assert len(meta) == 8732 and not meta.duplicated(['fold', 'slice_file_name']).any()
assert all(CLASS_NAMES[int(r.classID)] == r['class'] for _, r in meta.iterrows())
class_counts = meta.groupby(['classID', 'class']).size().reset_index(name='clips')
class_counts.to_csv(OUT / 'tables/class_counts.csv', index=False)
fold_counts = meta.groupby('fold').size().reset_index(name='clips')
fold_counts.to_csv(OUT / 'tables/fold_counts.csv', index=False)

overlaps = []
for fsid, g in meta.groupby('fsID'):
    if g.fold.nunique() <= 1:
        continue
    pairs = sum(a.fold != b.fold and min(a.end, b.end) > max(a.start, b.start)
                for a, b in itertools.combinations(g.itertuples(), 2))
    overlaps.append(dict(fsID=int(fsid), folds=','.join(map(str, sorted(g.fold.unique()))),
                         clips=len(g), overlapping_interval_pairs=int(pairs)))
pd.DataFrame(overlaps).to_csv(OUT / 'tables/source_overlap.csv', index=False)
train, val, test = meta[meta.fold >= 3], meta[meta.fold == 2], meta[meta.fold == 1]
train_test_ids = sorted(set(train.fsID) & set(test.fsID))
test_shared = test[test.fsID.isin(train_test_ids)]

# Fresh local WAV-header scan; historical full-decode audit is checked separately.
headers = []
if '--reuse-headers' in sys.argv:
    headers = pd.read_csv(OUT / 'audit/audio_headers.csv')
    assert len(headers) == 8732
else:
    for r in meta.itertuples():
        path = ROOT / 'data/UrbanSound8K/audio' / f'fold{r.fold}' / r.slice_file_name
        info = sf.info(path)
        headers.append(dict(file=r.slice_file_name, fold=r.fold, sample_rate=info.samplerate,
                            channels=info.channels, frames=info.frames,
                            duration=info.frames / info.samplerate))
    headers = pd.DataFrame(headers)
    headers.to_csv(OUT / 'audit/audio_headers.csv', index=False)
print('Metadata partitions, source overlaps, and all 8732 audio headers checked.', flush=True)
recorded_audio = read_json('results/dataset_integrity_report.json')
for field, col in [('sample_rate_counts_in_scanned_audio', 'sample_rate'),
                   ('mono_stereo_channel_counts_in_scanned_audio', 'channels')]:
    assert {str(k): int(v) for k, v in headers[col].value_counts().items()} == recorded_audio[field]
for stat in ('min', 'median', 'mean', 'max'):
    same(getattr(headers.duration, stat)(), recorded_audio['duration_s_in_scanned_audio'][stat])

eval_dir = 'results/stage5/20261030_132140_664348_block6_test1'
pred = read_csv(f'{eval_dir}/test_predictions.csv')
saved = read_json(f'{eval_dir}/evaluation_summary.json')
assert saved == read_json('reports/faculty_evidence/heldout/evaluation_summary.json')
assert len(pred) == len(test) == 873 and pred.fold.eq(1).all()
assert not pred.file.duplicated().any()
joined = pred.merge(test, left_on=['fold', 'file'], right_on=['fold', 'slice_file_name'],
                    validate='one_to_one')
assert len(joined) == 873
assert np.array_equal(joined.true_class_id, joined.classID)
y = pred.true_class_id.to_numpy()
yhat = pred.predicted_class_id.to_numpy()
probs = pred[[f'prob_class_{c}' for c in range(10)]].to_numpy()
assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
same(probs.sum(axis=1), np.ones(len(pred)), atol=2e-7)
assert np.array_equal(probs.argmax(axis=1), yhat)
assert np.array_equal(y == yhat, pred.correct)
same(probs.max(axis=1), pred.confidence)
cm = confusion_matrix(y, yhat, labels=range(10))
assert np.array_equal(cm, read_csv(f'{eval_dir}/confusion_matrix.csv', header=None))
np.savetxt(OUT / 'tables/confusion_matrix.csv', cm, fmt='%d', delimiter=',')
metrics = {'accuracy': float(accuracy_score(y, yhat))}
for avg in ('macro', 'weighted'):
    p, r, f, _ = precision_recall_fscore_support(y, yhat, labels=range(10), average=avg, zero_division=0)
    metrics.update({f'{avg}_precision': float(p), f'{avg}_recall': float(r), f'{avg}_f1': float(f)})
tn = cm.sum() - cm.sum(0) - cm.sum(1) + np.diag(cm)
fp = cm.sum(0) - np.diag(cm)
metrics['macro_specificity_ovr'] = float(np.mean(tn / (tn + fp)))
aucs = [roc_auc_score(y == c, probs[:, c]) for c in range(10)]
metrics['macro_roc_auc_ovr'] = float(np.mean(aucs))
for key, value in metrics.items():
    same(value, saved[key])
print('All saved prediction metrics agree with independent recalculation.', flush=True)
pr, re, f1, support = precision_recall_fscore_support(y, yhat, labels=range(10), zero_division=0)
pc = pd.DataFrame(dict(class_id=range(10), class_name=CLASS_NAMES, support=support,
                       precision=pr, recall=re, f1=f1, roc_auc_ovr=aucs))
original_pc = read_csv(f'{eval_dir}/per_class_metrics.csv')
for col in ('support', 'precision', 'recall', 'f1', 'roc_auc_ovr'):
    same(pc[col], original_pc[col])
pc.to_csv(OUT / 'tables/per_class_metrics.csv', index=False)

rows, histories = [], {}
for mode in ('head', 'fc1', 'block6'):
    base = f'reports/faculty_evidence/training_runs/{mode}'
    config, summary = read_json(f'{base}/config.json'), read_json(f'{base}/summary.json')
    h = read_csv(f'{base}/history.csv')
    assert not config['diagnostic_only'] and len(h) == 6
    assert h.train_samples_seen.eq(6971).all() and h.val_samples_seen.eq(888).all()
    best = h.sort_values(['val_macro_f1', 'val_accuracy'], ascending=False).iloc[0]
    assert best.epoch == summary['best_epoch'] == 4
    same(best.val_macro_f1, summary['best_val_macro_f1'])
    same(best.val_accuracy, summary['best_val_accuracy'])
    histories[mode] = h
    rows.append(dict(mode=mode, epoch=int(best.epoch), validation_accuracy=float(best.val_accuracy),
                     validation_macro_f1=float(best.val_macro_f1), validation_loss=float(best.val_loss),
                     trainable=config['parameter_counts']['trainable'],
                     total=config['parameter_counts']['total'],
                     seconds=float(h.seconds.sum()), peak_mib=float(h.peak_cuda_allocated_mib.max())))
comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / 'tables/validation_comparison.csv', index=False)

# Verify checkpoint identity and parameter counts without running any audio through it.
import torch
from src.finetune_model import load_model, load_trainable_state, parameter_summary
checkpoint_path = ROOT / 'results/stage4/20261027_144004_603447_block6_test1/best.pt'
sources.add(checkpoint_path)
checkpoint_hash = digest(checkpoint_path)
assert checkpoint_hash == saved['stage4_best_checkpoint_sha256']
payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
assert payload['config'] == 'block6' and payload['best_epoch'] == 4
assert payload['test_fold'] == 1 and payload['validation_fold'] == 2
same(payload['best_val_macro_f1'], saved['stage4_best_val_macro_f1'])
model = load_model('block6')
load_trainable_state(model, payload['trainable_parameters'])
print('Pinned model and selected checkpoint reconstructed successfully.', flush=True)
assert parameter_summary(model) == dict(total=81857561, trainable=60848138, frozen=21009423)
components = []
for name, module in model.backbone.named_children():
    components.append(dict(component=name, **parameter_summary(module)))
components.append(dict(component='classifier', **parameter_summary(model.classifier)))
pd.DataFrame(components).to_csv(OUT / 'tables/parameter_counts.csv', index=False)
for row in rows:
    model.configure(row['mode'])
    assert parameter_summary(model)['trainable'] == row['trainable']
del model, payload
backbone = ROOT / 'checkpoints/Cnn14_AudioSet_state_dict_safe.pth'
original = ROOT / 'checkpoints/Cnn14_mAP=0.431.pth'
sources.update([backbone, original])
original_md5 = digest(original, 'md5')
assert original_md5 == '595633ac2d1cac7ef04ebf70e2fee4e4'

# Publication figures are redrawn only from the original numeric records.
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
                     'legend.fontsize': 7, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(3.5, 2.4), layout='constrained')
ax.barh(range(10), class_counts.clips, color='#356d94')
ax.set_yticks(range(10), [x.replace('_', ' ') for x in CLASS_NAMES])
ax.invert_yaxis()
ax.set(xlabel='Number of clips', xlim=(0, 1150))
for i, n in enumerate(class_counts.clips):
    ax.text(n + 15, i, str(n), va='center', fontsize=7)
fig.savefig(OUT / 'figures/class_counts.pdf')
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.3), layout='constrained')
h = histories['block6']
axes[0].plot(h.epoch, h.train_loss, 'o-', label='Training (online)', markersize=3)
axes[0].plot(h.epoch, h.val_loss, 's-', label='Validation', markersize=3)
axes[0].set(ylabel='Cross-entropy', xlabel='Epoch')
axes[1].plot(h.epoch, 100 * h.train_accuracy, 'o-', label='Training (online)', markersize=3)
axes[1].plot(h.epoch, 100 * h.val_accuracy, 's-', label='Validation', markersize=3)
axes[1].set(ylabel='Accuracy (%)', xlabel='Epoch')
for mode, marker in zip(histories, ['o', 's', '^']):
    h = histories[mode]
    axes[2].plot(h.epoch, h.val_macro_f1, marker + '-', label=mode, markersize=3)
axes[2].set(ylabel='Validation macro-F1', xlabel='Epoch')
for ax in axes:
    ax.set_xticks(range(1, 7))
    ax.axvline(4, color='grey', linestyle=':', linewidth=1)
    ax.grid(alpha=.2)
    ax.legend(loc='best')
fig.savefig(OUT / 'figures/training_curves.pdf')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.15), layout='constrained')
ax.imshow(cm, cmap='Blues', vmin=0, vmax=120)
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=6.5,
                color='white' if cm[i, j] > 65 else 'black')
ax.set(xticks=range(10), yticks=range(10), xlabel='Predicted class ID', ylabel='True class ID')
fig.savefig(OUT / 'figures/confusion_matrix.pdf')
plt.close(fig)

# Generated rows keep manuscript rounding traceable to the recomputation.
def tex_name(s):
    return s.replace('_', r'\_')
def tex_rows(filename, lines):
    command = {'classes.tex': 'ClassRows', 'per_class.tex': 'PerClassRows',
               'comparison.tex': 'ComparisonRows', 'sample_rates.tex': 'SampleRateRows'}[filename]
    body = '\\newcommand{\\' + command + '}{%\n' + '\n'.join(lines) + '\n}\n'
    (OUT / 'tables' / filename).write_text(body, encoding='utf-8')
tex_rows('classes.tex', [f"{r.classID} & {tex_name(r['class'])} & {r.clips:,} & {int((test.classID == r.classID).sum())}" + r' \\' for _, r in class_counts.iterrows()])
tex_rows('per_class.tex', [f"{r.class_id} & {r.support} & {100*r.precision:.2f} & {100*r.recall:.2f} & {r.f1:.4f}" + r' \\' for r in pc.itertuples()])
tex_rows('comparison.tex', [f"{r.mode} & {r.trainable:,} & {100*r.validation_accuracy:.2f} & {r.validation_macro_f1:.4f} & {r.epoch} & {r.seconds/60:.2f} & {r.peak_mib:.1f}" + r' \\' for r in comparison.itertuples()])
tex_rows('sample_rates.tex', [f"{int(rate):,} & {int(n):,}" + r' \\' for rate, n in headers.sample_rate.value_counts().sort_index().items()])

audit = dict(audit_date='2026-11-06',
             method='Recomputed saved predictions; fresh WAV headers, checkpoint hashes and strict reconstruction. No new training or inference.',
             root_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             metadata_rows=len(meta), train_samples=len(train), validation_samples=len(val), test_samples=len(test),
             correct_predictions=int((y == yhat).sum()), metrics=metrics,
             probability_sum_max_error=float(np.abs(probs.sum(1)-1).max()),
             cross_fold_sources=overlaps, train_test_shared_ids=list(map(int, train_test_ids)),
             test_clips_sharing_source_with_training=len(test_shared),
             duration_seconds={s:float(getattr(headers.duration, s)()) for s in ('min', 'median', 'mean', 'max')},
             clips_over_four_seconds=int((headers.duration>4).sum()),
             selected_checkpoint_sha256=checkpoint_hash, original_checkpoint_md5=original_md5,
             safe_backbone_sha256=digest(backbone),
             not_reverified=['Historical training execution/timings', 'Historical full audio decoding',
                             'Prediction generation by fresh inference', 'Exact project revision at training time'])
(OUT / 'audit/verification.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
for name in ('src/config.py', 'src/dataset.py', 'src/finetune_model.py', 'src/finetune_train.py',
             'src/evaluate_selected.py', 'external/audioset_tagging_cnn/pytorch/models.py',
             'results/notebook_evidence/20261027_134247_157394/environment.json',
             'results/notebook_evidence/20261027_134247_157394/stage4_real_backward_checks.log'):
    sources.add(ROOT / name)
pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                   sha256=digest(p)) for p in sorted(sources)]).to_csv(OUT / 'audit/evidence_manifest.csv', index=False)
print(json.dumps(audit, indent=2))
print(comparison.to_string(index=False))
print(pd.DataFrame(components).to_string(index=False))
