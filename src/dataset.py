"""UrbanSound8K metadata splits and raw-waveform Dataset for official PANNs CNN14.

CNN14 expects a mono float32 waveform [batch, time], NOT an externally
computed Mel spectrogram. This module deliberately has no data augmentation.

Run diagnostics from project root:
    python -m src.check_dataloader
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "UrbanSound8K"

# Official classID values as given in UrbanSound8K.csv (not alphabetical).
CLASS_NAMES = (
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music",
)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Verified official CNN14 pretrained frontend: raw waveform, 32 kHz.
SAMPLE_RATE = 32_000
CLIP_SECONDS = 4
TARGET_SAMPLES = SAMPLE_RATE * CLIP_SECONDS


def load_metadata(dataset_root: str | Path = DATASET_ROOT) -> pd.DataFrame:
    """Validate metadata, preserving the predefined folds and class IDs."""
    root = Path(dataset_root)
    csv_path = root / "metadata" / "UrbanSound8K.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"slice_file_name", "fold", "classID", "class", "fsID"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if len(df) != 8_732:
        raise ValueError(f"Expected 8732 metadata rows; found {len(df)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Metadata contains null values in required columns")
    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype("int64")
    df["classID"] = pd.to_numeric(df["classID"], errors="raise").astype("int64")
    if set(df["fold"]) != set(range(1, 11)):
        raise ValueError("Fold IDs must be 1-10")
    if set(df["classID"]) != set(range(10)):
        raise ValueError("Class IDs must be 0-9")
    for class_id, name in enumerate(CLASS_NAMES):
        values = set(df.loc[df["classID"] == class_id, "class"])
        if values != {name}:
            raise ValueError(f"classID {class_id}: expected {name}; found {values}")
    keys = df["fold"].astype(str) + "/" + df["slice_file_name"].astype(str)
    if not keys.is_unique:
        raise ValueError("Duplicate (fold, slice_file_name) metadata entries")
    return df.reset_index(drop=True)


def split_metadata(df: pd.DataFrame, test_fold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) using eight, one and one official folds.

    Validation fold = test_fold % 10 + 1, as agreed in Stage 2.
    Does not promise source-level independence: fsID overlaps are documented.
    """
    if test_fold not in range(1, 11):
        raise ValueError("test_fold must be in 1..10")
    val_fold = test_fold % 10 + 1
    train = df.loc[~df["fold"].isin([test_fold, val_fold])].copy()
    val = df.loc[df["fold"] == val_fold].copy()
    test = df.loc[df["fold"] == test_fold].copy()
    assert (len(train) + len(val) + len(test)) == len(df)
    # The same physical file cannot belong to more than one subset.
    def keys(part: pd.DataFrame) -> set[tuple[int, str]]:
        return set(zip(part["fold"].astype(int), part["slice_file_name"].astype(str)))
    a, b, c = keys(train), keys(val), keys(test)
    assert not a.intersection(b) and not a.intersection(c) and not b.intersection(c)
    return train, val, test


def read_waveform(file_path: str | Path, target_samples: int = TARGET_SAMPLES) -> np.ndarray:
    """Read a WAV -> mono -> 32 kHz -> fixed-length float32 [target_samples].

    Mono conversion: arithmetic mean of input channels.
    Resampling: librosa's high-quality soxr_hq backend.
    Short clips: RIGHT zero-padding (no looping).
    Long clips: deterministic CENTER crop (same across train/val/test).
    No per-clip loudness normalization and no precomputed Mel spectrogram.
    """
    path = Path(file_path)
    audio, original_sr = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0 or original_sr <= 0:
        raise ValueError(f"Empty/invalid audio: {path}")
    waveform = audio.mean(axis=1, dtype=np.float32)
    if not np.isfinite(waveform).all():
        raise ValueError(f"NaN or infinite samples in {path}")
    if original_sr != SAMPLE_RATE:
        waveform = librosa.resample(
            waveform, orig_sr=original_sr, target_sr=SAMPLE_RATE,
            res_type="soxr_hq",
        )
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size > target_samples:
        start = (waveform.size - target_samples) // 2
        waveform = waveform[start:start + target_samples]
    elif waveform.size < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.size), mode="constant")
    if waveform.shape != (target_samples,) or not np.isfinite(waveform).all():
        raise ValueError(f"Waveform preprocessing failed for {path}")
    return np.ascontiguousarray(waveform, dtype=np.float32)


class UrbanSound8KDataset(Dataset):
    """Returns waveform [128000] float32 and official classID int (0-9)."""

    def __init__(self, metadata: pd.DataFrame, dataset_root: str | Path = DATASET_ROOT):
        self.metadata = metadata.reset_index(drop=True).copy()
        self.audio_root = Path(dataset_root) / "audio"

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.metadata.iloc[index]
        path = self.audio_root / f"fold{int(row['fold'])}" / str(row["slice_file_name"])
        waveform = read_waveform(path)
        label = int(row["classID"])
        if not 0 <= label < len(CLASS_NAMES):
            raise ValueError(f"Invalid classID {label} in {path}")
        return torch.from_numpy(waveform), label


def make_loaders(
    df: pd.DataFrame,
    *,
    test_fold: int,
    dataset_root: str | Path = DATASET_ROOT,
    batch_size: int = 2,
    num_workers: int = 0,
    seed: int = 67,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return training, validation, test loaders without crossing fold boundaries.

    num_workers=0 is an intentional Windows-friendly starting default.
    Raise batch_size only after checking actual RTX 3050 memory consumption.
    """
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size >= 1 and num_workers >= 0 required")
    train, val, test = split_metadata(df, test_fold)
    generator = torch.Generator().manual_seed(seed + test_fold)
    base = dict(batch_size=batch_size, num_workers=num_workers,
                pin_memory=torch.cuda.is_available())
    return (
        DataLoader(UrbanSound8KDataset(train, dataset_root), shuffle=True,
                   generator=generator, **base),
        DataLoader(UrbanSound8KDataset(val, dataset_root), shuffle=False, **base),
        DataLoader(UrbanSound8KDataset(test, dataset_root), shuffle=False, **base),
    )
