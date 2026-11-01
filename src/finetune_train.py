"""Stage 4: CNN14 single-fold supervised training with validation-only selection.

This script NEVER evaluates the reserved test fold. Stage 5 will evaluate a
model/configuration selected using validation folds only. Pilot limits are
clearly flagged as DIAGNOSTIC_ONLY and must not be treated as project metrics.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn

from .dataset import PROJECT_ROOT, load_metadata, make_loaders, split_metadata
from .finetune_model import (
    CONFIGS, load_model, load_trainable_state, optimizer_for_model,
    parameter_summary, trainable_state, EXPECTED_PANN_COMMIT,
)
from .runtime import AMP_CHOICES, choose_precision, autocast_context, assert_finite_logits
from .gradient_audit import gradient_measure


# Re-exported for existing notebook/import compatibility.
from .runtime import resolve_device, seed_everything
from .artifacts import write_csv


def accuracy_numerator(logits: torch.Tensor, labels: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == labels).sum().item())


def train_epoch(model, loader, optimizer, scaler, device, precision: str | bool, accum_steps: int,
                limit_batches: int) -> dict[str, float]:
    model.train()  # CNN14ForUrbanSound.train() preserves BN running statistics
    optimizer.zero_grad(set_to_none=True)
    planned = min(len(loader), limit_batches) if limit_batches > 0 else len(loader)
    if planned == 0:
        raise ValueError("Empty training DataLoader")
    total_loss, total_correct, total_samples = 0.0, 0, 0
    monitored = {}
    if hasattr(model, "classifier") and hasattr(model.classifier, "weight"):
        if model.classifier.weight.requires_grad:
            monitored["head"] = model.classifier.weight
    if hasattr(model, "backbone") and hasattr(model.backbone, "fc1"):
        if model.backbone.fc1.weight.requires_grad:
            monitored["fc1"] = model.backbone.fc1.weight
    if hasattr(model, "backbone") and hasattr(model.backbone, "conv_block6"):
        if model.backbone.conv_block6.conv1.weight.requires_grad:
            monitored["conv6"] = model.backbone.conv_block6.conv1.weight
    active_updates = {name: 0 for name in monitored}
    total_optimizer_steps = 0
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
        with autocast_context(device, precision):
            logits = model(waveforms)
            assert logits.shape == (labels.numel(), 10)
            assert_finite_logits(logits, phase="train")
            loss = nn.functional.cross_entropy(logits.float(), labels)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Training loss is nonfinite")
        scaler.scale(loss / window_len).backward()
        total_loss += loss.item() * labels.numel()
        total_correct += accuracy_numerator(logits, labels)
        total_samples += labels.numel()
        if ((i + 1) % accum_steps == 0) or (i + 1 == planned):
            scaler.unscale_(optimizer)
            for name, parameter in monitored.items():
                state, norm = gradient_measure(parameter)
                if state in ("missing", "nonfinite"):
                    raise FloatingPointError(f"{name} missing/nonfinite gradient at optimizer step {total_optimizer_steps+1}: {state}")
                if state == "nonzero":
                    active_updates[name] += 1
            nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_norm=1.0,
                error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            total_optimizer_steps += 1
    inactive = [name for name, count in active_updates.items() if count == 0]
    if inactive:
        raise RuntimeError(
            "No nonzero training gradients across the entire epoch for: " + ", ".join(inactive)
            + ". Finite logits/loss do not establish partial fine-tuning. "
            "Investigate the CNN14 activations, input audio and precision with "
            "python -m src check-gradients --config block6 --amp off."
        )
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples,
            "samples": total_samples, "batches": planned,
            "optimizer_steps": total_optimizer_steps, "active_gradient_steps": active_updates}


@torch.inference_mode()
def validate_epoch(model, loader, device, precision: str | bool, limit_batches: int) -> dict[str, float]:
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
        with autocast_context(device, precision):
            logits = model(waveforms)
            assert_finite_logits(logits, phase="validation")
            loss = nn.functional.cross_entropy(logits.float(), labels)
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
    write_csv(file, rows)


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


def main(argv=None):
    from .arguments import training_parser
    parser = training_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.accum_steps < 1 or args.patience < 1:
        parser.error("epochs, batch-size, accum-steps, and patience must be >= 1")
    if args.max_train_batches < 0 or args.max_val_batches < 0:
        parser.error("batch limits must be >= 0")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    precision = choose_precision(device, args.amp)
    diagnostic_only = args.max_train_batches > 0 or args.max_val_batches > 0
    fold_df = load_metadata(args.dataset_root)
    train_df, val_df, test_df = split_metadata(fold_df, args.test_fold)
    train_loader, val_loader, _ = make_loaders(
        fold_df, test_fold=args.test_fold, batch_size=args.batch_size,
        num_workers=0, seed=args.seed, dataset_root=args.dataset_root,
    )
    # NOTE: test DataLoader is never iterated and its recordings never decoded.
    print(f"CONFIG={args.config}, test-fold={args.test_fold}, val-fold={int(val_df.fold.iloc[0])}", flush=True)
    print(f"Train={len(train_df)}, validation={len(val_df)}, reserved test={len(test_df)}", flush=True)
    print(f"Device={device}, precision={precision} (requested={args.amp}), diagnostic_only={diagnostic_only}", flush=True)
    model = load_model(args.config, weights=args.weights, source=args.pann_source).to(device)
    counts = parameter_summary(model)
    print("Parameter counts:", counts, flush=True)
    optimizer = optimizer_for_model(model, lr_head=args.lr_head,
                                    lr_backbone=args.lr_backbone,
                                    weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # BF16/FP32 do not require scaling
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
        "amp": precision != "off", "precision": precision, "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches, "diagnostic_only": diagnostic_only,
        "parameter_counts": counts,
        "checkpoint_basis": (str(args.weights.relative_to(PROJECT_ROOT)).replace("\\", "/")
                             if args.weights.is_relative_to(PROJECT_ROOT) else str(args.weights)),
        "dataset_root": str(args.dataset_root), "pann_source": str(args.pann_source),
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
        tr = train_epoch(model, train_loader, optimizer, scaler, device, precision,
                         args.accum_steps, args.max_train_batches)
        val = validate_epoch(model, val_loader, device, precision, args.max_val_batches)
        seconds = time.perf_counter() - started
        peak = round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device.type == "cuda" else None
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
               "val_loss": val["loss"], "val_accuracy": val["accuracy"],
               "val_macro_f1": val["macro_f1"], "train_samples_seen": tr["samples"],
               "val_samples_seen": val["samples"], "seconds": seconds,
               "peak_cuda_allocated_mib": peak,
               "optimizer_steps": tr["optimizer_steps"],
               "head_nonzero_gradient_steps": tr["active_gradient_steps"].get("head"),
               "fc1_nonzero_gradient_steps": tr["active_gradient_steps"].get("fc1"),
               "conv6_nonzero_gradient_steps": tr["active_gradient_steps"].get("conv6")}
        print("Observed nonzero gradient steps:", tr["active_gradient_steps"], "/", tr["optimizer_steps"], flush=True)
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
