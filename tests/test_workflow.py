"""Developer workflow regressions, using temporary data and no real model inference."""
import contextlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

import pandas as pd
import torch

from src.config import PROJECT_ROOT, SELECTED_SHA256, project_path
from src.arguments import training_parser, evaluation_parser, gradients_parser
from src.evidence import read_heldout_evidence, selected_checkpoint
from src.metadata import audit_dataset_paths, validate_metadata


class WorkflowTests(unittest.TestCase):
    def test_application_commands_have_one_package_entry_point(self):
        self.assertEqual(list(PROJECT_ROOT.glob("*.py")), [])
        self.assertEqual(list((PROJECT_ROOT / "tools").glob("*.py")), [])
        from src.__main__ import DELEGATES
        self.assertIn("check-loader", DELEGATES)

    def test_all_cli_help_and_review_work_without_site_packages(self):
        from src.__main__ import DELEGATES
        for args in (["--help"], ["check", "--help"], ["review"],
                     *([name, "--help"] for name in DELEGATES)):
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, "-S", "-m", "src", *args],
                                        cwd=PROJECT_ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_importing_environment_and_submission_does_not_run_actions_or_write_lock(self):
        code = "import src.environment, src.submission; print('import-only')"
        lock = PROJECT_ROOT / "requirements.lock.txt"
        before = lock.read_bytes()
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd=PROJECT_ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "import-only")
        self.assertEqual(lock.read_bytes(), before)

    def test_asset_overrides_and_training_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            for factory, prefix in ((training_parser, []), (evaluation_parser, ["--run-dir", tmp]),
                                    (gradients_parser, [])):
                args = factory().parse_args(prefix + ["--dataset-root", tmp, "--weights", tmp,
                                                       "--pann-source", tmp])
                self.assertEqual(args.dataset_root, Path(tmp).resolve())
                self.assertEqual(args.weights, Path(tmp).resolve())
                self.assertEqual(args.pann_source, Path(tmp).resolve())
        args = training_parser().parse_args([])
        self.assertEqual((args.epochs, args.patience, args.seed, args.accum_steps), (3, 3, 67, 4))
        with tempfile.TemporaryDirectory() as tmp:
            import os
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                self.assertEqual(project_path("data/example"), PROJECT_ROOT / "data/example")
            finally:
                os.chdir(previous)

    def test_training_passes_path_overrides_through_to_data_and_model(self):
        from src import finetune_train as train
        class StopBeforeTraining(Exception):
            pass
        frame = pd.DataFrame([dict(fold=2, classID=0, slice_file_name="example.wav")])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch.object(train, "seed_everything"), \
                 patch.object(train, "resolve_device", return_value=torch.device("cpu")), \
                 patch.object(train, "load_metadata", return_value=frame) as metadata, \
                 patch.object(train, "split_metadata", return_value=(frame, frame, frame)), \
                 patch.object(train, "make_loaders", return_value=([], [], [])) as loaders, \
                 patch.object(train, "load_model", side_effect=StopBeforeTraining) as model, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(StopBeforeTraining):
                    train.main(["--dataset-root", tmp, "--weights", str(root/'base.pth'),
                                "--pann-source", str(root/'panns')])
                metadata.assert_called_once_with(root)
                self.assertEqual(loaders.call_args.kwargs["dataset_root"], root)
                model.assert_called_once_with("block6", weights=root/'base.pth', source=root/'panns')

    def test_evaluation_passes_path_overrides_before_any_inference(self):
        from src import evaluate_selected as evaluate
        class StopBeforeInference(Exception):
            pass
        val, test = pd.DataFrame(index=range(888)), pd.DataFrame(index=range(873))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            selected = ({"seed": 67}, {"best_epoch": 4, "best_val_macro_f1": 0.9,
                        "test_samples_reserved_not_evaluated": 873, "val_samples": 888}, {})
            with patch.object(evaluate, "PROJECT_ROOT", root), \
                 patch.object(evaluate, "validate_selected_run", return_value=selected), \
                 patch.object(evaluate, "sha256_file", return_value="mock-checkpoint-hash"), \
                 patch.object(evaluate, "seed_everything"), \
                 patch.object(evaluate, "resolve_device", return_value=torch.device("cpu")), \
                 patch.object(evaluate, "load_metadata", return_value=val) as metadata, \
                 patch.object(evaluate, "split_metadata", return_value=(val, val, test)), \
                 patch.object(evaluate, "load_model", side_effect=StopBeforeInference) as model, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(StopBeforeInference):
                    evaluate.main(["--run-dir", str(root/'run'), "--out", str(root/'new-evaluation'),
                                   "--dataset-root", tmp, "--weights", str(root/'base.pth'),
                                   "--pann-source", str(root/'panns')])
                metadata.assert_called_once_with(root)
                model.assert_called_once_with("block6", weights=root/'base.pth', source=root/'panns')
                self.assertFalse((root/'new-evaluation').exists())

    def test_dataset_audit_runs_once_and_returns_failure_exit_code(self):
        from src.metadata import paths_main
        report = dict(metadata_rows=8732, fold_counts={1: 1}, wav_count=1,
                      missing_files=["missing.wav"], passed=False)
        with patch("src.metadata.audit_dataset_paths", return_value=report) as audit:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                rc = paths_main(["--dataset-root", str(PROJECT_ROOT)])
        self.assertEqual(rc, 1)
        audit.assert_called_once_with(PROJECT_ROOT)
        self.assertEqual(output.getvalue().count("DATASET AUDIT:"), 1)
        with patch("src.metadata.audit_dataset_paths", side_effect=ValueError("bad metadata")):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(paths_main([]), 1)

    def test_metadata_validation_preserves_order_and_detects_bad_mapping(self):
        from src.config import CLASS_NAMES
        frame = pd.DataFrame([dict(slice_file_name=f"clip{i}.wav", fold=i % 10 + 1,
                                   classID=(i // 10) % 10, **{"class": CLASS_NAMES[(i // 10) % 10]},
                                   fsID=i) for i in range(8732)])
        result = validate_metadata(frame)
        self.assertEqual(result.slice_file_name.tolist(), frame.slice_file_name.tolist())
        frame.loc[0, "class"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "classID"):
            validate_metadata(frame)

    def test_path_audit_checks_references_and_counts(self):
        frame = pd.DataFrame([dict(fold=1, slice_file_name="missing.wav")])
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.metadata.load_metadata", return_value=frame) as load:
                result = audit_dataset_paths(tmp)
            load.assert_called_once_with(Path(tmp).resolve())
            self.assertFalse(result["passed"])
            self.assertEqual(len(result["missing_files"]), 1)

    def test_cuda_is_optional_but_explicit_requirement_fails(self):
        from src import environment
        with patch.object(environment, "PACKAGES", [("torch", "torch")]), \
             patch.object(environment.importlib.metadata, "version", return_value="test"), \
             patch("torch.cuda.is_available", return_value=False), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(environment.check_environment(), [])
            self.assertTrue(environment.check_environment(require_cuda=True))

    def test_curated_evidence_works_without_local_results_and_checks_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "reports/faculty_evidence/heldout"
            shutil.copytree(PROJECT_ROOT / "reports/faculty_evidence/heldout", directory)
            saved = read_heldout_evidence(root)
            self.assertIn("historical", saved.origin)
            self.assertEqual(saved.summary["stage4_best_checkpoint_sha256"], SELECTED_SHA256)
            self.assertFalse((root / "results").exists())
            summary = saved.summary.copy()
            summary["stage4_best_checkpoint_sha256"] = "different"
            (directory / "evaluation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance"):
                read_heldout_evidence(root)

    def test_checkpoint_discovery_never_uses_latest_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unrelated = root / "results/stage4/99999999_block6_test1/best.pt"
            unrelated.parent.mkdir(parents=True)
            unrelated.write_bytes(b"unrelated")
            self.assertIsNone(selected_checkpoint(root))

    def test_partial_audio_audit_cannot_replace_full_output_paths(self):
        from src.audit_audio import audit_output_paths
        root = Path(tempfile.gettempdir()) / "audit-path-test"
        results, figures = root / "results", root / "figures"
        full = audit_output_paths(results, figures, partial=False)
        first = audit_output_paths(results, figures, partial=True)
        second = audit_output_paths(results, figures, partial=True)
        self.assertEqual(full, (results, figures))
        self.assertNotEqual(first, second)
        self.assertTrue(first[0].is_relative_to(results / "audio_audits"))
        self.assertNotEqual(first[0] / "dataset_integrity_report.json", full[0] / "dataset_integrity_report.json")

    def test_environment_check_never_refreshes_dependency_snapshot(self):
        from src import environment
        lock = PROJECT_ROOT / "requirements.lock.txt"
        before = lock.read_bytes()
        with patch.object(environment, "check_environment", return_value=[]), \
             patch.object(environment, "capture_versions") as capture, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(environment.environment_main([]), 0)
        capture.assert_not_called()
        self.assertEqual(lock.read_bytes(), before)

    def test_existing_submission_is_never_overwritten(self):
        from src.submission import make_archive
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "123456.zip"
            out.write_bytes(b"existing private archive")
            with self.assertRaises(FileExistsError):
                make_archive(root, out, "123456", "Test", "A", root / "best.pt")
            self.assertEqual(out.read_bytes(), b"existing private archive")

    def test_active_guides_and_notebook_markdown_link_to_existing_sources(self):
        """Check maintained navigation without treating historical output links as source."""
        guides = [PROJECT_ROOT / name for name in (
            "README.md", "docs/DEVELOPMENT.md", "docs/REPRODUCIBILITY.md", "docs/RESEARCH.md")]
        sources = []
        for path in guides:
            self.assertTrue(path.is_file(), f"Missing developer guide: {path}")
            sources.append((path, path.read_text(encoding="utf-8")))
        for path in (PROJECT_ROOT / "notebooks").glob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            sources.extend((path, "".join(cell["source"]))
                           for cell in notebook["cells"] if cell["cell_type"] == "markdown")
        for source, markdown in sources:
            markdown = re.sub(r"```.*?```", "", markdown, flags=re.S)
            for link in re.findall(r"\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", markdown):
                target = urlsplit(link.strip("<>"))
                if target.scheme or target.netloc or not target.path:
                    continue
                destination = (source.parent / unquote(target.path)).resolve()
                if not destination.is_relative_to(PROJECT_ROOT):
                    continue
                relative = destination.relative_to(PROJECT_ROOT)
                if relative.parts[0] in {"reports", "results", "figures", "external"}:
                    continue
                if destination.suffix.lower() not in {".md", ".ipynb", ".py", ".txt"}:
                    continue
                with self.subTest(source=source.name, link=link):
                    self.assertTrue(destination.is_file(), f"Broken source link in {source}: {link}")

    def test_notebooks_review_on_checkout_without_data_weights_results_or_zip(self):
        """Execute only default review paths; subprocess/action cells must remain idle."""
        import ast
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(PROJECT_ROOT / "reports", root / "reports")
            shutil.copytree(PROJECT_ROOT / "figures", root / "figures")
            (root / "src").mkdir()
            (root / "src/dataset.py").write_text("# root discovery marker\n", encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(plt, "show"), \
                     patch("subprocess.Popen", side_effect=AssertionError("Review launched an action")), \
                     patch("subprocess.run", side_effect=AssertionError("Review launched an action")), \
                     patch("subprocess.check_output", side_effect=AssertionError("Review launched an action")), \
                     contextlib.redirect_stdout(io.StringIO()):
                    for path in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
                        namespace = {"__name__": "__notebook_review__"}
                        notebook = json.loads(path.read_text(encoding="utf-8"))
                        for i, cell in enumerate(notebook["cells"]):
                            if cell["cell_type"] == "code":
                                source = "".join(cell["source"])
                                ast.parse(source)
                                exec(compile(source, f"{path.name}:cell{i}", "exec"), namespace)
                        plt.close("all")
                self.assertFalse((root / "results").exists())
                self.assertFalse((root / "README.md").exists())
                self.assertFalse(list(root.glob("*.zip")))
            finally:
                os.chdir(previous)
                plt.close("all")


if __name__ == "__main__":
    unittest.main()
