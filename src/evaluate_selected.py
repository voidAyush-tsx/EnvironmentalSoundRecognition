"""Stage 5: one-time held-out evaluation of a validation-selected CNN14 run.

This module does not train, tune, or evaluate alternative architectures. Use
only AFTER selecting the run/configuration using Stage 4 validation evidence.
From the project root:
    python -m src.evaluate_selected --run-dir results/stage4/<BLOCK6_RUN>

The Stage 4 'best.pt' contains ONLY selected trainable parameters, so this
script first loads the original verified AudioSet backbone, then applies that
validation-selected parameter subset before any held-out test inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .dataset import CLASS_NAMES, PROJECT_ROOT, UrbanSound8KDataset, load_metadata, split_metadata
from .finetune_model import EXPECTED_PANN_COMMIT, load_model, load_trainable_state
from .finetune_train import resolve_device, seed_everything
from .precision_policy import autocast_context, assert_finite_logits, choose_precision

ALL_LABELS = tuple(range(10))


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def validate_selected_run(run_dir: Path, *, expected_config: str = "block6", expected_test_fold: int = 1) -> tuple[dict, dict, dict]:
    """Reject pilots, mismatched checkpoints, and unintended test folds."""
    config = read_json(run_dir / "config.json")
    summary = read_json(run_dir / "summary.json")
    if not (run_dir / "history.csv").is_file() or not (run_dir / "best.pt").is_file():
        raise FileNotFoundError("Completed Stage 4 history.csv and best.pt are required")
    if not (config.get("diagnostic_only") is False and summary.get("diagnostic_only") is False):
        raise ValueError("Diagnostic pilot checkpoint cannot be evaluated as a final model")
    if config.get("max_train_batches") != 0 or config.get("max_val_batches") != 0:
        raise ValueError("Evaluation requires unrestricted train/validation folds")
    if any(config.get(k) != summary.get(k) for k in (
        "config", "test_fold", "validation_fold", "seed", "precision", "checkpoint_basis",
    )):
        raise ValueError("Stage 4 config.json and summary.json disagree")
    if config.get("config") != expected_config or summary.get("config") != expected_config:
        raise ValueError(f"Evaluation is locked to validation-selected configuration: {expected_config}")
    if config.get("test_fold") != expected_test_fold:
        raise ValueError(f"Wrong reserved test fold: {config.get('test_fold')} != {expected_test_fold}")
    if config.get("validation_fold") != expected_test_fold % 10 + 1:
        raise ValueError("Unexpected validation fold for the agreed protocol")
    if summary.get("test_evaluated") is not False:
        raise ValueError("Stage 4 summary must certify that test was reserved")
    if config.get("model_source_commit") != EXPECTED_PANN_COMMIT:
        raise ValueError("Stage 4 source commit mismatch")
    if not (0 < summary.get("best_epoch", 0) <= summary.get("epochs_completed", 0)):
        raise ValueError("No valid validation-selected epoch recorded")
    if not (isinstance(summary.get("best_val_macro_f1"), (float, int)) and
            0 <= summary["best_val_macro_f1"] <= 1):
        raise ValueError("Missing valid validation macro-F1")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint structure")
    for key, value in {
        "format": "cnn14_trainable_parameters_v1",
        "config": expected_config,
        "test_fold": expected_test_fold,
        "validation_fold": config["validation_fold"],
        "best_epoch": summary["best_epoch"],
        "diagnostic_only": False,
    }.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint {key} mismatch: {checkpoint.get(key)!r} != {value!r}")
    if abs(float(checkpoint.get("best_val_macro_f1", -1)) - summary["best_val_macro_f1"]) > 1e-9:
        raise ValueError("Best checkpoint macro-F1 mismatch")
    if abs(float(checkpoint.get("best_val_accuracy", -1)) - summary["best_val_accuracy"]) > 1e-9:
        raise ValueError("Best checkpoint validation accuracy mismatch")
    if not isinstance(checkpoint.get("trainable_parameters"), dict):
        raise ValueError("Selected checkpoint lacks trainable weights")
    return config, summary, checkpoint


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[dict, list[dict], np.ndarray]:
    """Ten-class single-label metrics; AUC from scores, not predicted labels."""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or scores.shape != (len(y_true), 10):
        raise ValueError("Expected targets [N] and probabilities [N,10]")
    if not len(y_true) or np.any((y_true < 0) | (y_true > 9)):
        raise ValueError("Empty or invalid class IDs")
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Invalid class probabilities")
    if not np.allclose(scores.sum(axis=1), 1, atol=5e-5):
        raise ValueError("Softmax probabilities must sum to one")
    preds = scores.argmax(axis=1).astype(np.int64)
    cm = confusion_matrix(y_true, preds, labels=ALL_LABELS)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, preds, labels=ALL_LABELS, zero_division=0,
    )
    w = support.astype(np.float64) / len(y_true)
    per_class = []
    aucs = []
    for i, class_name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(len(y_true) - tp - fp - fn)
        specificity = float(tn / (tn + fp)) if (tn + fp) else None
        binary_label = (y_true == i).astype(np.uint8)
        auc = float(roc_auc_score(binary_label, scores[:, i])) if len(np.unique(binary_label)) == 2 else None
        if auc is not None:
            aucs.append(auc)
        per_class.append({
            "class_id": i, "class_name": class_name, "support": int(support[i]),
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
            "specificity_ovr": specificity, "roc_auc_ovr": auc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    result = {
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f1)),
        "weighted_precision": float(np.dot(p, w)),
        "weighted_recall": float(np.dot(r, w)),
        "weighted_f1": float(np.dot(f1, w)),
        "macro_specificity_ovr": float(np.mean([q["specificity_ovr"] for q in per_class if q["specificity_ovr"] is not None])),
        "macro_roc_auc_ovr": float(np.mean(aucs)) if len(aucs) == 10 else None,
        "roc_auc_note": "One-vs-rest ROC-AUC computed from per-class softmax probabilities; undefined if any class absent",
    }
    return result, per_class, cm


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to save")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_figures(out: Path, cm: np.ndarray, rows: list[dict], y_true: np.ndarray, scores: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    names = [c.replace("_", " ") for c in CLASS_NAMES]
    for normalized in [False, True]:
        matrix = cm.astype(float)
        if normalized:
            matrix = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                               out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) != 0)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        ax.set_xticks(range(10), names, rotation=65, ha="right")
        ax.set_yticks(range(10), names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("CNN14 block6 — held-out test fold 1 confusion matrix" + (" (row normalized)" if normalized else ""))
        for i in range(10):
            for j in range(10):
                val = f"{matrix[i, j]:.2f}" if normalized else str(int(matrix[i, j]))
                ax.text(j, i, val, ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > (0.5 if normalized else (cm.max() * 0.5)) else "black")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        fig.tight_layout()
        fig.savefig(out / ("confusion_matrix_normalized.png" if normalized else "confusion_matrix.png"), dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(10), [r["f1"] for r in rows])
    ax.set_xticks(range(10), names, rotation=55, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title("Held-out fold 1 — per-class F1")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "per_class_f1.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for k in range(10):
        binary = (y_true == k).astype(int)
        if len(np.unique(binary)) == 2:
            RocCurveDisplay.from_predictions(binary, scores[:, k], ax=ax, name=names[k])
    ax.set_title("CNN14 — one-vs-rest ROC curves, held-out test fold 1")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "roc_curves_ovr.png", dpi=160)
    plt.close(fig)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="ONE held-out fold evaluation of the validation-selected block6 checkpoint")
    ap.add_argument("--run-dir", type=Path, required=True, help="Existing full Stage 4 block6 run folder with best.pt")
    ap.add_argument("--expected-test-fold", type=int, default=1)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--amp", choices=("auto", "off", "bf16", "on"), default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        ap.error("--batch-size must be positive and --num-workers nonnegative")
    if args.expected_test_fold != 1:
        ap.error("Stage 4 comparison selected the protocol with reserved test fold 1; do not switch folds")
    run_dir = args.run_dir.resolve()
    config, stage4_summary, ckpt = validate_selected_run(run_dir, expected_test_fold=args.expected_test_fold)
    out = (args.out or PROJECT_ROOT / "results" / "stage5" /
           (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_block6_test1")).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite or repeat output: {out}")
    # Refuse accidental repeated testing of the *same* validation-selected checkpoint.
    sha = sha256_file(run_dir / "best.pt")
    stage5root = PROJECT_ROOT / "results" / "stage5"
    if stage5root.is_dir():
        for previous in stage5root.glob("*/evaluation_summary.json"):
            try:
                prior = read_json(previous)
            except (OSError, ValueError):
                continue
            if prior.get("stage4_best_checkpoint_sha256") == sha and prior.get("test_fold") == 1:
                raise RuntimeError(f"This exact checkpoint was already evaluated: {previous.parent}. Reuse its results, not the test set")
    seed_everything(int(config["seed"]))
    device = resolve_device(args.device)
    precision = choose_precision(device, args.amp)
    df = load_metadata()
    _, validation_df, test_df = split_metadata(df, 1)
    if len(test_df) != stage4_summary["test_samples_reserved_not_evaluated"]:
        raise ValueError("Test fold size differs from Stage 4 configuration")
    if len(validation_df) != stage4_summary["val_samples"]:
        raise ValueError("Validation fold size differs from Stage 4 configuration")
    print(f"VALIDATION-SELECTED RUN: {run_dir}", flush=True)
    print(f"Configuration: block6, selected epoch={stage4_summary['best_epoch']}, val macro-F1={stage4_summary['best_val_macro_f1']:.6f}", flush=True)
    print(f"ONE-TIME TEST: fold=1, samples={len(test_df)}, device={device}, precision={precision}", flush=True)
    model = load_model("block6")
    load_trainable_state(model, ckpt["trainable_parameters"])
    del ckpt
    model.to(device)
    model.eval()
    loader = DataLoader(UrbanSound8KDataset(test_df), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    labels_collected = []
    score_batches = []
    test_loss_sum = 0.0
    with torch.inference_mode():
        for i, (wave, label) in enumerate(loader):
            wave = wave.to(device, non_blocking=True)
            with autocast_context(device, precision):
                logits = model(wave)
            assert_finite_logits(logits, phase="held-out test")
            if logits.shape != (label.shape[0], 10):
                raise AssertionError("Expected [batch, 10] logits")
            probabilities = torch.softmax(logits.float(), dim=1)
            test_loss_sum += float(torch.nn.functional.cross_entropy(
                logits.float(), label.to(device), reduction="sum"
            ).item())
            labels_collected.extend(label.numpy().tolist())
            score_batches.append(probabilities.cpu().numpy())
            if (i + 1) % 100 == 0:
                print(f"Test batches: {i+1}/{len(loader)}", flush=True)
    y_true = np.asarray(labels_collected, dtype=np.int64)
    scores = np.concatenate(score_batches, axis=0)
    expected = test_df["classID"].to_numpy(dtype=np.int64)
    if not np.array_equal(y_true, expected):
        raise AssertionError("Test labels do not align with ordered metadata")
    if len(y_true) != 873:
        raise AssertionError(f"Fold 1 expected 873 test samples, got {len(y_true)}")
    aggregate, per_class, cm = compute_metrics(y_true, scores)
    predictions = []
    for idx, (_, row) in enumerate(test_df.reset_index(drop=True).iterrows()):
        pred = int(scores[idx].argmax())
        record = {"file": str(row["slice_file_name"]), "fold": int(row["fold"]),
                  "true_class_id": int(y_true[idx]), "predicted_class_id": pred,
                  "confidence": float(scores[idx, pred]), "correct": bool(pred == y_true[idx])}
        record.update({f"prob_class_{k}": float(scores[idx, k]) for k in range(10)})
        predictions.append(record)
    result = {
        "test_cross_entropy": test_loss_sum / len(y_true),
        "evaluation_protocol": "Single held-out test fold 1; config chosen by validation fold 2 before testing",
        "config": "block6", "test_fold": 1, "validation_fold": 2,
        "stage4_run_directory": str(run_dir), "stage4_best_checkpoint_sha256": sha,
        "checkpoint_epoch": int(stage4_summary["best_epoch"]),
        "stage4_best_val_macro_f1": float(stage4_summary["best_val_macro_f1"]),
        "stage4_best_val_accuracy": float(stage4_summary["best_val_accuracy"]),
        "device": str(device), "precision": precision, "model_source_commit": EXPECTED_PANN_COMMIT,
        "test_evaluated": True, "dataset_source_isolation_caveat": "Five fsID values cross predefined folds; see results/cross_fold_source_summary.csv",
        **aggregate,
    }
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "per_class_metrics.csv", per_class)
    write_csv(out / "test_predictions.csv", predictions)
    np.savetxt(out / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    draw_figures(out, cm, per_class, y_true, scores)
    # Write the final marker only after all CSVs and figures succeeded.
    (out / "evaluation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"TEST ACCURACY: {aggregate['accuracy']:.4f}, macro-F1: {aggregate['macro_f1']:.4f}", flush=True)
    print(f"TEST macro ROC-AUC (OvR): {aggregate['macro_roc_auc_ovr']}", flush=True)
    print(f"RESULTS: {out}", flush=True)
    print("Evaluation complete. Do NOT tune using the held-out test metrics.", flush=True)


if __name__ == "__main__":
    main()
