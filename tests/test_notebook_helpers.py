"""Notebook presentation and opt-in execution checks; no real model/data work."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from src.config import PROJECT_ROOT
from src.evidence import read_heldout_evidence
from src.notebook_helpers import (
    asset_arguments, notebook_assets, read_per_class_metrics,
    read_training_evidence, run_action, show_audio_example,
)


class NotebookHelperTests(unittest.TestCase):
    def test_asset_paths_use_the_notebook_checkout_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            assets = notebook_assets(root, dataset_root="custom-data", weights=root / "base.pth")
            self.assertEqual(assets["dataset_root"], root / "custom-data")
            self.assertEqual(assets["weights"], root / "base.pth")
            self.assertEqual(assets["pann_source"], root / "external/audioset_tagging_cnn")
            self.assertIn(str(root / "custom-data"), asset_arguments(assets))
            self.assertNotIn("--dataset-root", asset_arguments(assets, include_dataset=False))

    def test_disabled_action_has_no_filesystem_or_process_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("src.processes.run_logged", side_effect=AssertionError("must stay disabled")):
                self.assertIsNone(run_action(root, ["train"]))
            self.assertEqual(list(root.iterdir()), [])

    def test_enabled_action_logs_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("src.processes.run_logged", return_value=0) as run:
                log = run_action(root, ["check", "--unit"], enabled=True)
            self.assertTrue(log.is_relative_to(root / "results/notebook_evidence"))
            self.assertEqual(run.call_args.args[1], [sys.executable, "-m", "src", "check", "--unit"])
            self.assertEqual(run.call_args.kwargs["cwd"], root)
            with patch("src.processes.run_logged", return_value=3):
                with self.assertRaisesRegex(RuntimeError, "exit 3"):
                    run_action(root, ["check", "--unit"], enabled=True)

    def test_training_view_reads_curated_full_runs_and_rejects_pilot_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(PROJECT_ROOT / "reports/faculty_evidence", root / "reports/faculty_evidence")
            table, histories = read_training_evidence(root)
            self.assertEqual(table.index.tolist(), ["head", "fc1", "block6"])
            self.assertEqual(set(histories), {"head", "fc1", "block6"})
            self.assertTrue((table["best_epoch"] == 4).all())
            self.assertAlmostEqual(table.loc["block6", "best_val_accuracy_pct"], 90.65315315315315)
            file = root / "reports/faculty_evidence/training_runs/head/config.json"
            config = json.loads(file.read_text(encoding="utf-8"))
            config["diagnostic_only"] = True
            file.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence mismatch"):
                read_training_evidence(root)

    def test_per_class_reader_detects_disagreement_with_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "reports/faculty_evidence/heldout"
            shutil.copytree(PROJECT_ROOT / "reports/faculty_evidence/heldout", directory)
            saved = read_heldout_evidence(root)
            metrics = read_per_class_metrics(saved)
            self.assertEqual(int(metrics["tp"].sum()), 773)
            metrics.loc[0, "support"] += 1
            metrics.to_csv(directory / "per_class_metrics.csv", index=False)
            with self.assertRaisesRegex(ValueError, "disagree"):
                read_per_class_metrics(saved)

    def test_notebook_local_metadata_and_loaders_receive_the_overridden_root(self):
        path = PROJECT_ROOT / "notebooks/01_Project_Overview.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            frame = pd.DataFrame([dict(fold=3, slice_file_name="example.wav", classID=0)])
            namespace = {"REVIEW_ONLY": False, "assets": {"dataset_root": root},
                         "pd": pd, "display": Mock(), "CLASS_NAMES": tuple(str(i) for i in range(10)),
                         "TEST_FOLD": 1}
            with patch("src.metadata.load_metadata", return_value=frame) as metadata:
                exec(cells["701aa671"], namespace)
            metadata.assert_called_once_with(root)
            batch = [(torch.zeros(1, 128000), torch.tensor([0]))]
            reserved_test = Mock()
            with patch("src.dataset.make_loaders", return_value=(batch, batch, reserved_test)) as loaders, \
                 contextlib.redirect_stdout(io.StringIO()):
                exec(cells["043fc3dd"], namespace)
            self.assertIs(loaders.call_args.args[0], frame)
            self.assertEqual(loaders.call_args.kwargs["dataset_root"], root)
            self.assertEqual(reserved_test.mock_calls, [])

    def test_notebook_diagnostics_and_pilots_honor_fold_and_precision(self):
        path = PROJECT_ROOT / "notebooks/01_Project_Overview.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        run = Mock()
        namespace = {"ROOT": PROJECT_ROOT, "REVIEW_ONLY": False, "run_action": run,
                     "RUN_REAL_STAGE4_GRADIENT_CHECKS": True, "RUN_DIAGNOSTIC_PILOTS": True,
                     "TEST_FOLD": 3, "AMP_MODE": "off", "asset_options": []}
        exec(cells["cf28b3ce"], namespace)
        exec(cells["b67b7435"], namespace)
        self.assertEqual(run.call_count, 5)
        for call in run.call_args_list:
            arguments = call.args[1]
            self.assertEqual(arguments[arguments.index("--test-fold") + 1], "3")
            self.assertEqual(arguments[arguments.index("--amp") + 1], "off")
            self.assertTrue(call.kwargs["enabled"])

    def test_local_audio_example_uses_configured_dataset_root(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame([dict(fold=3, slice_file_name="example.wav")])
            with patch("src.dataset.read_waveform", return_value=np.zeros(128000, dtype=np.float32)) as read, \
                 patch.object(plt, "show"):
                show_audio_example(root, frame, dataset_root=root / "custom-data")
            read.assert_called_once_with(root / "custom-data/audio/fold3/example.wav")


if __name__ == "__main__":
    unittest.main()
