"""Unit tests that use synthetic WAV files; they do NOT imply real UrbanSound8K was audited.

From project root:
    python -m unittest discover -s tests -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.dataset import (
    CLASS_NAMES, SAMPLE_RATE, TARGET_SAMPLES,
    UrbanSound8KDataset, read_waveform, split_metadata,
)


class WaveformTests(unittest.TestCase):
    def test_stereo_resampling_and_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "short.wav"
            sr = 16_000
            t = np.arange(sr // 2, dtype=np.float32) / sr
            left = 0.3 * np.sin(2 * np.pi * 330 * t)
            right = 0.1 * np.sin(2 * np.pi * 330 * t)
            sf.write(path, np.column_stack((left, right)), sr)
            wave = read_waveform(path)
            self.assertEqual(wave.shape, (TARGET_SAMPLES,))
            self.assertEqual(wave.dtype, np.float32)
            self.assertTrue(np.isfinite(wave).all())
            self.assertTrue(np.all(wave[20_000:] == 0))
            self.assertGreater(np.max(np.abs(wave[:16_000])), 0.15)

    def test_center_crop_is_deterministic(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "long.wav"
            signal = np.concatenate([
                np.full(SAMPLE_RATE, 0.1, dtype=np.float32),
                np.full(4 * SAMPLE_RATE, 0.25, dtype=np.float32),
                np.full(SAMPLE_RATE, 0.1, dtype=np.float32),
            ])
            sf.write(path, signal, SAMPLE_RATE, subtype="FLOAT")
            a, b = read_waveform(path), read_waveform(path)
            self.assertTrue(np.array_equal(a, b))
            self.assertEqual(a.shape, (TARGET_SAMPLES,))
            self.assertTrue(np.allclose(a, 0.25))

    def test_dataset_returns_float_waveform_and_official_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "audio" / "fold1").mkdir(parents=True)
            sf.write(root / "audio" / "fold1" / "sample.wav",
                     np.zeros(SAMPLE_RATE // 4, dtype=np.float32), SAMPLE_RATE)
            df = pd.DataFrame([{
                "fold": 1, "slice_file_name": "sample.wav", "classID": 8,
            }])
            x, y = UrbanSound8KDataset(df, root)[0]
            self.assertIsInstance(x, torch.Tensor)
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(tuple(x.shape), (TARGET_SAMPLES,))
            self.assertEqual(y, 8)
            self.assertEqual(CLASS_NAMES[y], "siren")

    def test_splits_are_disjoint(self):
        df = pd.DataFrame([{"fold": i, "slice_file_name": f"s{i}.wav", "classID": 0}
                           for i in range(1, 11)])
        for fold in range(1, 11):
            train, val, test = split_metadata(df, fold)
            self.assertEqual((len(train), len(val), len(test)), (8, 1, 1))
            self.assertEqual(int(val.iloc[0]["fold"]), fold % 10 + 1)
            self.assertEqual(int(test.iloc[0]["fold"]), fold)


if __name__ == "__main__":
    unittest.main()
