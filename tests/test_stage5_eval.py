"""Small source-compatible tests; no real UrbanSound8K dataset required."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json

import numpy as np

from src.evaluate_selected import compute_metrics, validate_selected_run


class Stage5MetricsTests(unittest.TestCase):
    def test_all_metrics_perfect(self):
        labels = np.arange(10, dtype=np.int64)
        scores = np.eye(10) * 0.91 + np.ones((10, 10)) * 0.009
        # make each row sum to 1
        scores /= scores.sum(axis=1, keepdims=True)
        overall, per_class, matrix = compute_metrics(labels, scores)
        self.assertEqual(overall["accuracy"], 1.0)
        self.assertEqual(overall["macro_f1"], 1.0)
        self.assertEqual(overall["macro_roc_auc_ovr"], 1.0)
        self.assertEqual(overall["macro_specificity_ovr"], 1.0)
        self.assertEqual(matrix.shape, (10, 10))
        self.assertEqual([q["class_id"] for q in per_class], list(range(10)))

    def test_reject_bad_probabilities(self):
        labels = np.arange(10)
        scores = np.ones((10, 10)) * 0.1
        scores[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            compute_metrics(labels, scores)

    def test_undefined_auc_for_missing_class(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.full((4, 10), 0.05)
        scores[labels == 0, 0] = 0.55
        scores[labels == 1, 1] = 0.55
        # row sum = 0.55 + 9*0.05 = 1
        overall, classes, _ = compute_metrics(labels, scores)
        self.assertIsNone(overall["macro_roc_auc_ovr"])
        self.assertIsNone(classes[2]["roc_auc_ovr"])

    def test_pilot_rejected_before_loading_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            d = Path(root)
            cfg = {"config":"block6", "test_fold":1, "validation_fold":2,
                   "seed":67, "precision":"bf16", "checkpoint_basis":"x",
                   "max_train_batches":4, "max_val_batches":4, "diagnostic_only":True}
            (d/"config.json").write_text(json.dumps(cfg), encoding="utf-8")
            (d/"summary.json").write_text(json.dumps(cfg), encoding="utf-8")
            (d/"best.pt").write_bytes(b"untrusted-placeholder")
            (d/"history.csv").write_text("epoch\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Diagnostic"):
                validate_selected_run(d)


if __name__ == "__main__":
    unittest.main()
