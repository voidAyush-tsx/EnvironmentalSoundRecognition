"""Small presentation and command helpers shared by the three project notebooks.

Model, preprocessing and metric implementations remain in their existing modules.
Importing this module does not load PyTorch or inspect large model assets.
"""
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
import math
from pathlib import Path
import sys

import pandas as pd

from .artifacts import read_json
from .config import (
    CLASS_NAMES, CONFIGS, DATASET_ROOT, DEPENDENCIES, EXPECTED_PANN_COMMIT,
    PANN_SOURCE, PROJECT_ROOT, SAFE_WEIGHTS, SAMPLE_RATE,
)


def notebook_assets(root: Path, *, dataset_root=None, weights=None, pann_source=None) -> dict:
    """Resolve optional notebook overrides relative to the notebook's checkout."""
    def resolve(value, default):
        path = Path(value).expanduser() if value is not None else default.relative_to(PROJECT_ROOT)
        return (path if path.is_absolute() else root / path).resolve()
    return {"dataset_root": resolve(dataset_root, DATASET_ROOT),
            "weights": resolve(weights, SAFE_WEIGHTS),
            "pann_source": resolve(pann_source, PANN_SOURCE)}


def asset_arguments(assets: dict, *, include_dataset: bool = True) -> list[str]:
    options = ["--weights", str(assets["weights"]), "--pann-source", str(assets["pann_source"])]
    return (["--dataset-root", str(assets["dataset_root"])] if include_dataset else []) + options


def run_action(root: Path, arguments: list[str], *, enabled: bool = False) -> Path | None:
    """Run an explicitly enabled project CLI action, saving its complete output."""
    if not enabled:
        return None
    from .processes import run_logged
    logs = root / "results" / "notebook_evidence"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logfile = logs / f"{stamp}_{arguments[0]}.log"
    code = run_logged(arguments[0], [sys.executable, "-m", "src", *arguments],
                      cwd=root, log_path=logfile)
    if code:
        raise RuntimeError(f"Command failed with exit {code}. See {logfile}")
    return logfile


def environment_table() -> pd.DataFrame:
    """Read installed version metadata without importing packages or probing CUDA."""
    rows = []
    for package, _ in DEPENDENCIES:
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = "not installed"
        rows.append({"package": package, "version": installed})
    return pd.DataFrame(rows)


def read_training_evidence(root: Path) -> tuple[pd.DataFrame, dict]:
    """Read the documented full runs, never select a run by timestamp."""
    rows, histories = [], {}
    for mode in CONFIGS:
        directory = root / "reports" / "faculty_evidence" / "training_runs" / mode
        config = read_json(directory / "config.json")
        summary = read_json(directory / "summary.json")
        expected = {"config": mode, "test_fold": 1, "validation_fold": 2,
                    "seed": 67, "diagnostic_only": False,
                    "model_source_commit": EXPECTED_PANN_COMMIT}
        for key, value in expected.items():
            if config.get(key) != value or summary.get(key) != value:
                raise ValueError(f"Historical training evidence mismatch: {directory}: {key}")
        if (summary.get("test_evaluated") is not False or summary.get("best_epoch") != 4
                or config.get("max_train_batches") != 0 or config.get("max_val_batches") != 0):
            raise ValueError(f"Expected a completed, full-fold validation run: {directory}")
        history = pd.read_csv(directory / "history.csv")
        histories[mode] = history
        rows.append({
            "configuration": mode, "best_epoch": summary["best_epoch"],
            "best_val_accuracy_pct": 100 * summary["best_val_accuracy"],
            "best_val_macro_f1": summary["best_val_macro_f1"],
            "trainable_parameters": config["parameter_counts"]["trainable"],
            "total_parameters": config["parameter_counts"]["total"],
            "total_six_epoch_minutes": history["seconds"].sum() / 60,
            "peak_gpu_mib": history["peak_cuda_allocated_mib"].max(),
        })
    return pd.DataFrame(rows).set_index("configuration"), histories


def show_image(path: Path, *, width: int = 820) -> None:
    from IPython.display import Image, display
    display(Image(filename=str(path), width=width))


def show_class_distribution(root: Path, metadata=None) -> None:
    from IPython.display import display
    if metadata is not None:
        counts = metadata.groupby(["classID", "class"]).size().rename("clips").reset_index()
        display(counts)
        import matplotlib.pyplot as plt
        ax = counts.plot.bar(x="class", y="clips", legend=False, figsize=(10, 4))
        ax.set_title("Local metadata: class distribution")
        plt.tight_layout()
        plt.show()
        plt.close(ax.figure)
    else:
        show_image(root / "figures" / "class_distribution.png")


def read_per_class_metrics(evidence) -> pd.DataFrame:
    """Check class ordering and agreement with the saved aggregate metrics."""
    metrics = pd.read_csv(evidence.directory / "per_class_metrics.csv").sort_values("class_id")
    summary = evidence.summary
    expected_correct = round(summary["accuracy"] * summary["test_samples"])
    if (metrics["class_id"].tolist() != list(range(len(CLASS_NAMES)))
            or int(metrics["support"].sum()) != summary["test_samples"]
            or int(metrics["tp"].sum()) != expected_correct
            or not math.isclose(metrics["f1"].mean(), summary["macro_f1"], rel_tol=1e-5, abs_tol=1e-8)):
        raise ValueError(f"Per-class metrics disagree with the saved evaluation: {evidence.directory}")
    return metrics


def show_folds(root: Path, metadata=None, *, test_fold: int = 1) -> None:
    from IPython.display import display
    if metadata is not None:
        from .dataset import split_metadata
        train, validation, test = split_metadata(metadata, test_fold)
        display(metadata.groupby("fold").size().rename("clips").to_frame())
        counts = len(train), len(validation), len(test)
        folds = [f for f in range(1, 11) if f not in (test_fold, test_fold % 10 + 1)]
    else:
        # These counts belong to the recorded experiment, not a new dataset audit.
        config = read_json(root / "reports/faculty_evidence/training_runs/block6/config.json")
        test_fold = config["test_fold"]
        counts = config["train_samples"], config["val_samples"], config["test_samples_reserved_not_evaluated"]
        folds = [f for f in range(1, 11) if f not in (test_fold, config["validation_fold"])]
    display(pd.DataFrame({"subset": ["train", "validation", "test"],
                          "folds": [str(folds), str(test_fold % 10 + 1), str(test_fold)],
                          "clips": counts}))


def show_source_overlap(root: Path, metadata=None) -> None:
    from IPython.display import Markdown, display
    if metadata is not None:
        groups = metadata.groupby("fsID").agg(folds=("fold", "nunique"), clips=("fold", "size"))
        display(groups.query("folds > 1"))
    elif (root / "results/cross_fold_source_summary.csv").is_file():
        display(pd.read_csv(root / "results/cross_fold_source_summary.csv"))
    else:
        display(Markdown("The recorded audit found five shared source IDs; see the "
                         "[limitations](../docs/RESEARCH.md#freezing-depth-study)."))


def show_audio_example(root: Path, metadata=None, *, dataset_root: Path, mel: bool = False) -> None:
    """Show a saved figure in review, or preprocess one explicitly selected local clip."""
    if metadata is None:
        show_image(root / "figures" / ("logmel_example.png" if mel else "waveform_example.png"))
        return
    import numpy as np
    import matplotlib.pyplot as plt
    from .dataset import read_waveform
    row = metadata.iloc[0]
    waveform = read_waveform(dataset_root / "audio" / f"fold{int(row['fold'])}" / row["slice_file_name"])
    fig, ax = plt.subplots(figsize=(10, 3))
    if mel:
        import librosa
        import librosa.display
        power = librosa.feature.melspectrogram(y=waveform, sr=SAMPLE_RATE, n_fft=1024,
                                              hop_length=320, n_mels=64, fmin=50, fmax=14000)
        image = librosa.display.specshow(librosa.power_to_db(power, ref=np.max), sr=SAMPLE_RATE,
                                        hop_length=320, x_axis="time", y_axis="mel", ax=ax)
        fig.colorbar(image, ax=ax, label="dB")
        ax.set_title("Illustrative log-Mel; CNN14 computes its own frontend")
    else:
        ax.plot(np.arange(len(waveform)) / SAMPLE_RATE, waveform, linewidth=0.5)
        ax.set(xlabel="Time (s)", ylabel="Amplitude", title="Processed mono waveform")
    fig.tight_layout()
    plt.show()
    plt.close(fig)
