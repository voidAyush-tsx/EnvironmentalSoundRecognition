"""Read historical evidence without data, CUDA, checkpoints or third-party imports."""
from dataclasses import dataclass
from pathlib import Path

from .artifacts import read_json
from .config import EXPECTED_PANN_COMMIT, PROJECT_ROOT, SELECTED_RUN_RELATIVE, SELECTED_SHA256


@dataclass(frozen=True)
class HeldoutEvidence:
    directory: Path
    summary: dict
    origin: str


def read_heldout_evidence(root: Path = PROJECT_ROOT, *, evaluation_dir: Path | None = None) -> HeldoutEvidence:
    """Default to curated history. A local evaluation must be selected explicitly."""
    root = Path(root).resolve()
    if evaluation_dir is None:
        directory = root / "reports" / "faculty_evidence" / "heldout"
        origin = "Curated historical evidence (not a new execution)"
    else:
        directory = Path(evaluation_dir).expanduser()
        if not directory.is_absolute():
            directory = root / directory
        origin = "Explicitly selected local evaluation (saved results; not executed by this reader)"
    summary = read_json(directory / "evaluation_summary.json")
    expected = {"test_evaluated": True, "config": "block6", "test_fold": 1,
                "validation_fold": 2, "test_samples": 873, "checkpoint_epoch": 4,
                "model_source_commit": EXPECTED_PANN_COMMIT,
                "stage4_best_checkpoint_sha256": SELECTED_SHA256}
    for name, value in expected.items():
        if summary.get(name) != value:
            raise ValueError(f"Historical evidence provenance mismatch in {directory}: {name}")
    for name in ("per_class_metrics.csv", "confusion_matrix.png"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Missing evidence artifact: {directory / name}")
    return HeldoutEvidence(directory, summary, origin)


def selected_checkpoint(root: Path = PROJECT_ROOT) -> Path | None:
    """Known historical locations only; timestamp ordering never selects a model."""
    root = Path(root)
    for candidate in (root / SELECTED_RUN_RELATIVE / "best.pt",
                      root / "models" / "selected_block6_best.pt"):
        if candidate.is_file():
            return candidate
    return None


def print_review(root: Path = PROJECT_ROOT, *, evaluation_dir: Path | None = None) -> None:
    evidence = read_heldout_evidence(root, evaluation_dir=evaluation_dir)
    print(evidence.origin)
    print("Artifacts:", evidence.directory)
    result = evidence.summary
    print(f"block6 / epoch {result['checkpoint_epoch']} / held-out fold 1 / {result['test_samples']} clips")
    print(f"Accuracy: {result['accuracy']:.2%}; macro-F1: {result['macro_f1']:.4f}; "
          f"macro OvR ROC-AUC: {result['macro_roc_auc_ovr']:.4f}")
    print("One test fold, one seed; five source IDs cross official folds. No ten-fold mean.")
