# Stage 3 — UrbanSound8K raw-waveform data pipeline

Project: **Deep Learning Framework for Environmental Sound Recognition Using PyTorch**

This bundle adds files only; it does not overwrite or retrain the existing model.

## Copy files

Unzip this archive **inside the project root** `D:\EnvironmentalSoundRecognition`, so it creates:

```
src/__init__.py
src/dataset.py
src/audit_audio.py
src/check_dataloader.py
tests/test_dataset.py
STAGE3_README.md
```

Your existing `src/check_cnn14.py` and `src/check_folds.py` are not included and must remain in place. Do not replace `requirements.txt` or your safe checkpoint.

## Run (from the project root)

```
cd /d D:\EnvironmentalSoundRecognition
python -m unittest discover -s tests -v
python -m src.audit_audio --max-files 50
python -m src.audit_audio
python -m src.check_dataloader
python -m src.check_dataloader --model-forward
```

The full audio audit decodes every WAV one at a time and saves integrity, class, fold, duration, sampling-rate, and visual reports under `results/` and `figures/`.

`--max-files 50` is for a **partial smoke test only** and must not be described as a full audit. The second, unrestricted audit is the genuine complete run.

The loader returns **raw float32 mono waveforms of 128,000 samples** and official class IDs 0..9; the official PANNs model performs the spectrogram/log-Mel extraction internally. The current preprocessing uses soxr_hq resampling only when needed, right zero-padding for short clips, deterministic center crop for >4s clips, no custom amplitude normalization, and no data augmentation. Long-audio cropping counts are audited and should be examined before training.

Test fold `f` uses validation fold `(f % 10) + 1`; the other eight are training folds. Reuse the exact same split with both the frozen-feature and partial fine-tuning experiments. This respects official clip folds but **does not resolve the five cross-fold Freesound source IDs** found in Stage 2.

### What counts as passing

- 4 synthetic WAV/split unit tests pass on the user's PC.
- `python -m src.audit_audio` reports exactly 8,732 successfully decoded files and 0 invalid files; `"full_audio_audit": true` in `results/dataset_integrity_report.json`.
- The three loaders return waveform batches of shape `[batch_size,128000]` and `torch.int64` labels in 0..9.
- Optional `--model-forward` runs a real waveform through the previously verified checkpoint and returns shape `[1,10]` (untrained head only).

Do not use the current 10-class outputs as classification accuracy until genuine training occurs.
