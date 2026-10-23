"""Stage 4: CNN14 single-fold supervised training with validation-only selection.

This script NEVER evaluates the reserved test fold. Stage 5 will evaluate a
model/configuration selected using validation folds only. Pilot limits are
clearly flagged as DIAGNOSTIC_ONLY and must not be treated as project metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .dataset import PROJECT_ROOT, load_metadata, make_loaders, split_metadata
from .finetune_model import (
    CONFIGS, load_model, load_trainable_state, optimizer_for_model,
    parameter_summary, trainable_state, EXPECTED_PANN_COMMIT,
)


def resolve_device(device_str: str) -> torch.device:
    choice = ("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else device_str
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but not found")
    return torch.device(choice)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def accuracy_numerator(logits: torch.Tensor, labels: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == labels).sum().item())


def train_epoch(model, loader, optimizer, scaler, device, use_amp: bool, accum_steps: int,
                limit_batches: int) -> dict[str, float]:
    model.train()  # CNN14ForUrbanSound.train() preserves BN running statistics
    optimizer.zero_grad(set_to_none=True)
    planned = min(len(loader), limit_batches) if limit_batches > 0 else len(loader)
    if planned == 0:
        raise ValueError("Empty training DataLoader")
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for i, (waveforms, labels) in enumerate(loader):
        if i >= planned:
            break
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if not (labels.dtype == torch.long and waveforms.ndim == 2):
            raise ValueError("Unexpected batch dtype or shape")
        # Number of microbatches in the current accumulation window. Handles
        # shorter final windows without shrinking the last optimizer update.
        window_start = (i // accum_steps) * accum_steps
        window_len = min(accum_steps, planned - window_start)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(waveforms)
            assert logits.shape == (labels.numel(), 10)
            loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Training loss is nonfinite")
        scaler.scale(loss / window_len).backward()
        total_loss += loss.item() * labels.numel()
        total_correct += accuracy_numerator(logits, labels)
        total_samples += labels.numel()
        if ((i + 1) % accum_steps == 0) or (i + 1 == planned):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples,
            "samples": total_samples, "batches": planned}


@torch.inference_mode()
def validate_epoch(model, loader, device, use_amp: bool, limit_batches: int) -> dict[str, float]:
    model.eval()
    planned = min(len(loader), limit_batches) if limit_batches > 0 else len(loader)
    if planned == 0:
        raise ValueError("Empty validation DataLoader")
    loss_sum, count, correct = 0.0, 0, 0
    actual, predicted = [], []
    for i, (waveforms, labels) in enumerate(loader):
        if i >= planned:
            break
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(waveforms)
            loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Validation loss is nonfinite")
        loss_sum += loss.item() * labels.numel()
        count += labels.numel()
        correct += accuracy_numerator(logits, labels)
        actual.extend(labels.detach().cpu().tolist())
        predicted.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return {"loss": loss_sum / count, "accuracy": correct / count,
            "macro_f1": float(f1_score(actual, predicted, labels=list(range(10)),
                                        average="macro", zero_division=0)),
            "samples": count, "batches": planned}


def save_history_csv(rows: list[dict], file: Path) -> None:
    if not rows:
        return
    with file.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_learning_curves(rows: list[dict], target: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="Training")
    axes[0].plot(epochs, [r["val_loss"] for r in rows], label="Validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [r["train_accuracy"] for r in rows], label="Training (online)")
    axes[1].plot(epochs, [r["val_accuracy"] for r in rows], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("CNN14 UrbanSound8K: training and validation only")
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train official pretrained CNN14 on UrbanSound8K")
    parser.add_argument("--config", choices=CONFIGS, default="block6")
    parser.add_argument("--test-fold", type=int, choices=range(1, 11), metavar="1..10", default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="0=full training fold; positive=diagnostic-only batch limit")
    parser.add_argument("--max-val-batches", type=int, default=0,
                        help="0=full validation fold; positive=diagnostic-only batch limit")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional unique output directory; existing best.pt won't be overwritten")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.accum_steps < 1 or args.patience < 1:
        parser.error("epochs, batch-size, accum-steps, and patience must be >= 1")
    if args.max_train_batches < 0 or args.max_val_batches < 0:
        parser.error("batch limits must be >= 0")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda" if args.amp == "auto" else args.amp == "on"
    if use_amp and device.type != "cuda":
        parser.error("--amp on requires CUDA")
    diagnostic_only = args.max_train_batches > 0 or args.max_val_batches > 0
    fold_df = load_metadata()
    train_df, val_df, test_df = split_metadata(fold_df, args.test_fold)
    train_loader, val_loader, _ = make_loaders(
        fold_df, test_fold=args.test_fold, batch_size=args.batch_size,
        num_workers=0, seed=args.seed,
    )
    # NOTE: test DataLoader is never iterated and its recordings never decoded.
    print(f"CONFIG={args.config}, test-fold={args.test_fold}, val-fold={int(val_df.fold.iloc[0])}", flush=True)
    print(f"Train={len(train_df)}, validation={len(val_df)}, reserved test={len(test_df)}", flush=True)
    print(f"Device={device}, AMP={use_amp}, diagnostic_only={diagnostic_only}", flush=True)
    model = load_model(args.config).to(device)
    counts = parameter_summary(model)
    print("Parameter counts:", counts, flush=True)
    optimizer = optimizer_for_model(model, lr_head=args.lr_head,
                                    lr_backbone=args.lr_backbone,
                                    weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out = args.out or PROJECT_ROOT / "results" / "stage4" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{args.config}_test{args.test_fold}"
    )
    if (out / "best.pt").exists():
        raise FileExistsError(f"Refusing to overwrite an existing checkpoint: {out / 'best.pt'}")
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "config": args.config, "epochs_requested": args.epochs, "test_fold": args.test_fold,
        "validation_fold": int(val_df.fold.iloc[0]), "train_samples": len(train_df),
        "val_samples": len(val_df), "test_samples_reserved_not_evaluated": len(test_df),
        "batch_size": args.batch_size, "accum_steps": args.accum_steps,
        "lr_head": args.lr_head, "lr_backbone": args.lr_backbone,
        "weight_decay": args.weight_decay, "patience": args.patience,
        "seed": args.seed, "device": str(device), "torch_version": torch.__version__,
        "amp": use_amp, "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches, "diagnostic_only": diagnostic_only,
        "parameter_counts": counts,
        "checkpoint_basis": "checkpoints/Cnn14_AudioSet_state_dict_safe.pth",
        "model_source_commit": EXPECTED_PANN_COMMIT,
        "batchnorm_policy": "all pretrained BN running stats frozen; later block affine params train when unfrozen",
        "data_preprocessing": "mono; 32000 Hz; center crop/right pad to 128000 samples",
        "source_overlap_warning": "five fsID cross official folds; see results/cross_fold_source_summary.csv",
    }
    (out / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print("Outputs:", out, flush=True)
    rows = []
    best_f1, best_acc, best_epoch, no_improve = -1.0, -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        tr = train_epoch(model, train_loader, optimizer, scaler, device, use_amp,
                         args.accum_steps, args.max_train_batches)
        val = validate_epoch(model, val_loader, device, use_amp, args.max_val_batches)
        seconds = time.perf_counter() - started
        peak = round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device.type == "cuda" else None
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
               "val_loss": val["loss"], "val_accuracy": val["accuracy"],
               "val_macro_f1": val["macro_f1"], "train_samples_seen": tr["samples"],
               "val_samples_seen": val["samples"], "seconds": seconds,
               "peak_cuda_allocated_mib": peak}
        rows.append(row)
        save_history_csv(rows, out / "history.csv")
        print(f"Epoch {epoch}/{args.epochs}: train loss={tr['loss']:.4f} acc={tr['accuracy']:.4f} | "
              f"val loss={val['loss']:.4f} acc={val['accuracy']:.4f} "
              f"macro-F1={val['macro_f1']:.4f} | {seconds:.1f}s, peak GPU={peak} MiB", flush=True)
        improved = (val["macro_f1"] > best_f1 + 1e-12 or
                    (math.isclose(val["macro_f1"], best_f1, abs_tol=1e-12) and
                     val["accuracy"] > best_acc + 1e-12))
        if improved:
            best_epoch, best_f1, best_acc, no_improve = epoch, val["macro_f1"], val["accuracy"], 0
            payload = {"format": "cnn14_trainable_parameters_v1", "config": args.config,
                       "test_fold": args.test_fold, "validation_fold": settings["validation_fold"],
                       "best_epoch": epoch, "best_val_macro_f1": best_f1,
                       "best_val_accuracy": best_acc, "diagnostic_only": diagnostic_only,
                       "trainable_parameters": trainable_state(model)}
            torch.save(payload, out / "best.pt")
            print("  Validation-selected checkpoint saved (trainable parameters only)", flush=True)
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print("Early stopping: validation macro-F1 did not improve", flush=True)
            break
    save_learning_curves(rows, out / "learning_curves.png")
    summary = {**settings, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
               "best_val_accuracy": best_acc, "epochs_completed": len(rows),
               "test_evaluated": False}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Validate that this checkpoint can be safely read and has correct keys.
    safe = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    load_trainable_state(model, safe["trainable_parameters"])
    assert safe["config"] == args.config
    print("Selected checkpoint restricted reload: PASS", flush=True)
    print("Training run completed. TEST DATA WAS NOT EVALUATED.", flush=True)
    if diagnostic_only:
        print("DIAGNOSTIC_ONLY: limited batches, not publishable model performance.", flush=True)
    print("History:", out / "history.csv", flush=True)
    print("Best checkpoint:", out / "best.pt", flush=True)
    print("Summary:", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
