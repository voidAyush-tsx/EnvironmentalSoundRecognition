"""Stage 3: stream/decode UrbanSound8K WAV files and generate factual dataset reports.

Run from the project root:
    python -m src.audit_audio --max-files 50  # short smoke test
    python -m src.audit_audio                 # complete 8732-file audit

This script does not train anything, edit audio, change labels, or change folds.
Mel plots are for visual inspection ONLY; pretrained PANNs CNN14 creates
its own log-Mel representation from raw waveforms internally.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from .dataset import (
    DATASET_ROOT, PROJECT_ROOT, CLASS_NAMES, SAMPLE_RATE,
    TARGET_SAMPLES, load_metadata, read_waveform, split_metadata,
)


def save_figures(df: pd.DataFrame, root: Path, figures_dir: Path) -> None:
    """Generate class chart and waveform/Mel visualizations from real audio."""
    class_counts = df.groupby(["classID", "class"], sort=True).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(class_counts["class"], class_counts["count"])
    ax.set(xlabel="Official UrbanSound8K class", ylabel="Clip count", title="Class distribution (metadata)")
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    # Deterministic example: first metadata row, not chosen using labels or results.
    row = df.iloc[0]
    wav_path = root / "audio" / f"fold{int(row['fold'])}" / str(row["slice_file_name"])
    waveform = read_waveform(wav_path)
    time = np.arange(TARGET_SAMPLES) / SAMPLE_RATE
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time, waveform, linewidth=0.35)
    ax.set(xlabel="Time (s)", ylabel="Amplitude", title=f"Example waveform — {row['class']} (32 kHz)")
    fig.tight_layout()
    fig.savefig(figures_dir / "waveform_example.png", dpi=150)
    plt.close(fig)

    # A matching-parameter illustrative Mel plot, not the model's actual frontend output.
    mel_power = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_fft=1024, win_length=1024,
        hop_length=320, n_mels=64, fmin=50, fmax=14_000,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel_power, ref=1.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_db, sr=SAMPLE_RATE, hop_length=320,
        x_axis="time", y_axis="mel", ax=ax,
    )
    ax.set_title("Illustrative log-Mel spectrogram (CNN14 computes its own)")
    fig.colorbar(img, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(figures_dir / "logmel_example.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit UrbanSound8K audio and generate figures")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "figures")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Scan only first N files for a quick smoke test; NOT a full integrity audit")
    args = parser.parse_args()
    if args.max_files is not None and args.max_files <= 0:
        parser.error("--max-files must be positive")

    root = args.dataset_root
    df = load_metadata(root)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # Metadata counts are always from the complete 8732-row table, not only sampled WAVs.
    class_counts = df.groupby(["classID", "class"], sort=True).size().reset_index(name="count")
    fold_counts = df.groupby("fold", sort=True).size().reset_index(name="count")
    fold_by_class = pd.crosstab(df["fold"], df["classID"]).reindex(
        index=range(1, 11), columns=range(10), fill_value=0,
    )
    fold_by_class.columns = [CLASS_NAMES[i] for i in fold_by_class.columns]
    class_counts.to_csv(args.results_dir / "class_distribution.csv", index=False)
    fold_counts.to_csv(args.results_dir / "fold_distribution.csv", index=False)
    fold_by_class.to_csv(args.results_dir / "fold_by_class.csv", index_label="fold")

    scanned = df.head(args.max_files) if args.max_files is not None else df
    records: list[dict] = []
    errors: list[dict] = []
    for row in tqdm(scanned.itertuples(index=False), total=len(scanned), desc="Decoding WAV files"):
        path = root / "audio" / f"fold{int(row.fold)}" / str(row.slice_file_name)
        try:
            # Decode (not just inspect WAV headers), one clip at a time.
            wave, sr = sf.read(path, dtype="float32", always_2d=True)
            if sr <= 0 or wave.shape[0] == 0 or not np.isfinite(wave).all():
                raise ValueError("Invalid sample rate, empty waveform, or nonfinite values")
            mono = wave.mean(axis=1, dtype=np.float32)
            records.append({
                "fold": int(row.fold), "classID": int(row.classID),
                "slice_file_name": str(row.slice_file_name),
                "sample_rate_hz": int(sr), "channels": int(wave.shape[1]),
                "frames": int(wave.shape[0]), "duration_s": wave.shape[0] / sr,
                "peak_abs": float(np.max(np.abs(mono))),
                "rms": float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))),
                "over_4_seconds": bool(wave.shape[0] / sr > 4.001),
            })
        except Exception as exc:
            errors.append({"fold": int(row.fold), "file": str(path), "error": repr(exc)})

    info = pd.DataFrame(records)
    info.to_csv(args.results_dir / "audio_file_statistics.csv", index=False)
    pd.DataFrame(errors, columns=["fold", "file", "error"]).to_csv(
        args.results_dir / "invalid_audio.csv", index=False,
    )

    # A readable histogram from successfully decoded recordings (full or pilot).
    if not info.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(info["duration_s"], bins=30)
        ax.set(xlabel="Duration (seconds)", ylabel="Recordings",
               title=f"Audio duration distribution (decoded n={len(info)})")
        fig.tight_layout()
        fig.savefig(args.figures_dir / "duration_distribution.png", dpi=150)
        plt.close(fig)
        sr_distribution = info["sample_rate_hz"].value_counts().sort_index()
        sr_distribution.rename_axis("sample_rate_hz").reset_index(name="count").to_csv(
            args.results_dir / "sample_rate_distribution.csv", index=False,
        )
    else:
        sr_distribution = pd.Series(dtype=int)

    full = len(scanned) == len(df)
    report = {
        "dataset_name": "UrbanSound8K",
        "expected_metadata_rows": 8732,
        "metadata_rows": len(df),
        "total_classes": len(CLASS_NAMES),
        "official_fold_count": 10,
        "audio_files_requested_for_decoding": len(scanned),
        "audio_files_successfully_decoded": len(records),
        "invalid_or_undecodable_audio_files": len(errors),
        "full_audio_audit": full,
        "sample_rate_counts_in_scanned_audio": {str(k): int(v) for k, v in sr_distribution.items()},
        "mono_stereo_channel_counts_in_scanned_audio": {
            str(k): int(v) for k, v in (info["channels"].value_counts().items() if not info.empty else [])
        },
        "duration_s_in_scanned_audio": (
            {"min": float(info["duration_s"].min()),
             "median": float(info["duration_s"].median()),
             "mean": float(info["duration_s"].mean()),
             "max": float(info["duration_s"].max())} if not info.empty else None
        ),
        "clips_above_4_seconds_in_scanned_audio": int(info["over_4_seconds"].sum()) if not info.empty else None,
        "preprocessing": {
            "input": "raw waveform, not precomputed log-Mel",
            "mono": "mean of channels", "sample_rate_hz": SAMPLE_RATE,
            "clip_samples": TARGET_SAMPLES, "short_clips": "zero-pad on right",
            "long_clips": "center crop, deterministic",
            "resampling": "librosa soxr_hq, only when original sample rate differs",
            "amplitude_normalization": "none",
            "augmentation": "none",
        },
        "source_id_fold_isolation": "WARNING: 5 cross-fold fsIDs detected in Stage 2; no folds changed",
    }
    with (args.results_dir / "dataset_integrity_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if errors:
        print(f"ERROR: {len(errors)} audio files failed decoding; inspect results/invalid_audio.csv")
        raise SystemExit(1)
    # Plot real sample only after audit has established successful decoding.
    save_figures(df, root, args.figures_dir)
    train, val, test = split_metadata(df, test_fold=1)
    print("Metadata:", len(df), "rows; 10 official classes; 10 folds")
    print("Audio scan:", len(records), "decoded,", len(errors), "invalid;")
    print("FULL AUDIT" if full else "PILOT AUDIT ONLY (not full dataset integrity validation)")
    if not info.empty:
        print("Sample rates:", dict(sr_distribution))
        print("Duration (min/median/mean/max seconds):", report["duration_s_in_scanned_audio"])
        print("Clips longer than 4.001 s:", report["clips_above_4_seconds_in_scanned_audio"])
    print("Outer test fold 1: train", len(train), "validation", len(val), "test", len(test))
    print("Reports:", args.results_dir)
    print("Figures:", args.figures_dir)
    print("Reminder: source-level overlap warning is not cleared by this audit.")


if __name__ == "__main__":
    main()
