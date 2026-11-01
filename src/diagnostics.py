"""Explicit model, waveform-loader and train-only gradient diagnostics.

Run through `python -m src check-model`, `check-loader`, or `check-gradients`.
The legacy checkpoint converter is opt-in and verifies the official checksum.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from .artifacts import file_digest
from .config import (
    ORIGINAL_WEIGHTS as CHECKPOINT, SAFE_WEIGHTS as SAFE_CHECKPOINT,
    SAFE_WEIGHTS, PANN_SOURCE, EXPECTED_MD5, TARGET_SAMPLES,
)
from .dataset import load_metadata, make_loaders, split_metadata, UrbanSound8KDataset
from .finetune_model import load_model, parameter_summary, optimizer_for_model
from .gradient_audit import candidate_indices, gradient_measure, feature_statistics
from .pann_adapter import new_backbone, load_backbone, validate_backbone_state
from .runtime import resolve_device, seed_everything, choose_precision, autocast_context, assert_finite_logits


class UrbanSoundCNN14(nn.Module):
    """Official CNN14 preserved intact, plus a new untrained ten-logit head."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(2048, 10)

    def forward(self, waveform: torch.Tensor, return_details: bool = False):
        outputs = self.backbone(waveform)
        logits = self.classifier(outputs["embedding"])
        return (logits, outputs) if return_details else logits


def md5sum(path: Path) -> str:
    return file_digest(path, "md5")


def read_weights(args: argparse.Namespace) -> Mapping:
    if args.use_safe_weights:
        path = getattr(args, "weights", SAFE_CHECKPOINT)
        if not path.is_file():
            raise FileNotFoundError(
                f"Safe state dictionary missing: {path}\n"
                "First run with --trust-legacy-checkpoint --export-safe-weights."
            )
        print("Loading tensor-only exported state dictionary (restricted loader) ...")
        container = torch.load(path, map_location="cpu", weights_only=True)
    else:
        path = getattr(args, "original_weights", CHECKPOINT)
        if not path.is_file():
            raise FileNotFoundError(f"Original checkpoint missing: {path}")
        digest = md5sum(path)
        print("Original checkpoint MD5:", digest)
        if digest.lower() != EXPECTED_MD5:
            raise ValueError(
                "Original checkpoint MD5 differs from the official Zenodo record. "
                "Do not load this file; obtain a verified original checkpoint."
            )
        print("Original checkpoint MD5: MATCH")
        if args.trust_legacy_checkpoint:
            print("Loading verified official legacy checkpoint via unrestricted pickle ...")
            print("WARNING: Unrestricted pickle can execute code; use only with trusted sources.")
            container = torch.load(path, map_location="cpu", weights_only=False)
        else:
            print("Attempting restricted weights-only checkpoint loading ...")
            try:
                container = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as exc:
                if "Weights only load failed" not in str(exc):
                    raise
                raise RuntimeError(
                    "Original checkpoint contains NumPy/pickle objects not allowed by "
                    "weights_only=True. If you trust this verified official Zenodo "
                    "download, run:\n"
                    "  python -m src check-model --trust-legacy-checkpoint --export-safe-weights\n"
                    "After that, future checks can use:\n"
                    "  python -m src check-model --use-safe-weights"
                ) from exc

    state = validate_backbone_state(container)
    print("Model state_dict parameter/buffer keys:", len(state))
    return state


def model_main(argv=None) -> None:
    from .arguments import model_parser
    parser = model_parser()
    args = parser.parse_args(argv)
    if args.use_safe_weights and args.export_safe_weights:
        parser.error("--use-safe-weights and --export-safe-weights cannot be used together")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("PyTorch:", torch.__version__)
    print("Inference device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU total memory (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

    # Construct the ORIGINAL AudioSet model, including original 527-class layer.
    backbone = new_backbone(args.pann_source)
    state_dict = read_weights(args)
    load_result = backbone.load_state_dict(state_dict, strict=True)
    assert not load_result.missing_keys and not load_result.unexpected_keys
    print("Strict pretrained checkpoint loading: PASS (no missing or unexpected keys)")

    if args.export_safe_weights:
        args.weights.parent.mkdir(parents=True, exist_ok=True)
        print("Exporting tensor-only model state_dict:", args.weights)
        torch.save({"model": backbone.state_dict()}, args.weights)
        # Confirm it can be reloaded with restricted deserialization.
        exported = torch.load(args.weights, map_location="cpu", weights_only=True)
        if set(exported["model"]) != set(backbone.state_dict()):
            raise AssertionError("Exported state dictionary key validation failed")
        del exported
        print("Tensor-only export and restricted reload: PASS")

    del state_dict
    model = UrbanSoundCNN14(backbone).to(device).eval()
    total_parameters = sum(p.numel() for p in model.parameters())
    head_parameters = sum(p.numel() for p in model.classifier.parameters())
    print("Total parameters (backbone + new head):", f"{total_parameters:,}")
    print("New head parameters:", f"{head_parameters:,}")

    sr = 32000
    t = torch.arange(4 * sr, device=device, dtype=torch.float32) / sr
    waveform = (0.1 * torch.sin(2 * torch.pi * 440 * t)).unsqueeze(0)
    with torch.inference_mode():
        logits, outputs = model(waveform, return_details=True)

    assert waveform.shape == (1, 128000)
    assert outputs["clipwise_output"].shape == (1, 527), outputs["clipwise_output"].shape
    assert outputs["embedding"].shape == (1, 2048), outputs["embedding"].shape
    assert logits.shape == (1, 10), logits.shape
    assert torch.isfinite(logits).all().item()
    assert torch.isfinite(outputs["embedding"]).all().item()
    print("Input waveform shape:", tuple(waveform.shape))
    print("Original AudioSet predictions:", tuple(outputs["clipwise_output"].shape))
    print("Pretrained embedding shape:", tuple(outputs["embedding"].shape))
    print("UrbanSound8K logits shape:", tuple(logits.shape))
    print("CNN14 checkpoint + forward-pass verification: PASS")
    print("The new 10-class head is randomly initialized; no fine-tuning was performed.")


def test_real_audio_with_pretrained_backbone(waveforms: torch.Tensor, *,
                                            weights=SAFE_WEIGHTS, source=PANN_SOURCE) -> None:
    backbone = load_backbone(weights=weights, source=source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UrbanSoundCNN14(backbone).to(device).eval()
    with torch.inference_mode():
        logits = model(waveforms[:1].to(device))
    assert logits.shape == (1, 10), logits.shape
    assert torch.isfinite(logits).all().item()
    print("Real UrbanSound8K waveform -> official CNN14 -> [1, 10] logits: PASS")
    print("The 10-class head is UNTRAINED; predictions are not meaningful yet.")


def dataloader_main(argv=None) -> None:
    from .arguments import dataloader_parser
    parser = dataloader_parser()
    args = parser.parse_args(argv)
    df = load_metadata(args.dataset_root)
    train_loader, val_loader, test_loader = make_loaders(
        df, test_fold=args.test_fold, batch_size=args.batch_size, num_workers=0, dataset_root=args.dataset_root,
    )
    print("Train, validation, test counts:",
          len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    for name, loader in (("train", train_loader), ("validation", val_loader), ("test", test_loader)):
        waves, labels = next(iter(loader))
        assert waves.shape == (min(args.batch_size, len(loader.dataset)), TARGET_SAMPLES), waves.shape
        assert waves.dtype == torch.float32
        assert labels.dtype == torch.int64
        assert torch.isfinite(waves).all().item()
        assert ((labels >= 0) & (labels <= 9)).all().item()
        print(name, "waveforms", tuple(waves.shape), "labels", labels.tolist(), "PASS")
        if name == "train":
            first_batch = waves
    if args.model_forward:
        test_real_audio_with_pretrained_backbone(first_batch, weights=args.weights, source=args.pann_source)
    print("Waveform DataLoader checks: PASS")
    print("No audio augmentation, model training, or fine-tuning was performed.")


def gradients_main(argv=None):
    from .arguments import gradients_parser
    parser = gradients_parser()
    args = parser.parse_args(argv)
    seed_everything(67)
    device = resolve_device(args.device)
    precision = choose_precision(device, args.amp)
    print(f"Device={device}, config={args.config}, precision={precision} (requested={args.amp})", flush=True)
    metadata = load_metadata(args.dataset_root)
    train_df, val_df, test_df = split_metadata(metadata, args.test_fold)
    dataset = UrbanSound8KDataset(train_df, args.dataset_root)
    indices = candidate_indices(train_df, args.max_candidates)
    print(f"Probing {len(indices)} deterministic training-only clips; validation/test untouched", flush=True)

    model = load_model(config=args.config, weights=args.weights, source=args.pann_source).to(device)
    info = parameter_summary(model)
    print("Parameter counts:", info, flush=True)
    print("Trainable prefixes:", sorted({".".join(n.split(".")[:2])
                                       for n, p in model.named_parameters() if p.requires_grad}), flush=True)
    model.train()
    if args.feature_mode == "eval":
        # Diagnostic contrast ONLY: official eval-mode disables its internal
        # SpecAugment and dropout, but DOES NOT disable gradient computation.
        model.backbone.eval()
    print("Backbone feature mode:", "train (SpecAug/dropout enabled)" if model.backbone.training else "eval (SpecAug/dropout disabled)", flush=True)
    bn0 = model.backbone.bn0
    bn_before = {
        name: (module.running_mean.detach().cpu().clone(),
               module.running_var.detach().cpu().clone(),
               module.num_batches_tracked.detach().cpu().clone())
        for name, module in model.backbone.named_modules()
        if isinstance(module, _BatchNorm) and module.track_running_stats
    }
    frozen_before = model.backbone.conv_block1.conv1.weight.detach().flatten()[:512].cpu().clone()
    # Snapshot a small subset for update verification, checking gradients on entire tensors.
    head_before = model.classifier.weight.detach().cpu().clone()
    fc1_before = model.backbone.fc1.weight.detach().cpu().clone()
    conv6_before = model.backbone.conv_block6.conv1.weight.detach().cpu().clone() if args.config == "block6" else None
    assert not bn0.training
    assert not model.backbone.conv_block1.bn1.training
    assert not model.backbone.conv_block6.bn2.training

    optimizer = optimizer_for_model(model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # BF16/FP32 do not need scaling
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Observe activations at fc1; report the cause of zero weight gradients.
    observed = {}
    def fc1_input_hook(module, args):
        observed["fc1_input"] = feature_statistics(args[0])
    def fc1_output_hook(module, args, output):
        # The official backbone applies an inplace ReLU AFTER fc1.
        observed["fc1_pre_relu"] = feature_statistics(output)
    in_hook = model.backbone.fc1.register_forward_pre_hook(fc1_input_hook)
    out_hook = model.backbone.fc1.register_forward_hook(fc1_output_hook)
    chosen = None
    grad_summary = {}
    failed_cases = []
    try:
        for attempt, index in enumerate(indices, 1):
            waveform, class_id = dataset[index]  # NEVER validation/test
            optimizer.zero_grad(set_to_none=True)
            observed.clear()
            waves = waveform.unsqueeze(0).to(device)
            labels = torch.tensor([class_id], dtype=torch.long, device=device)
            if not torch.isfinite(waves).all():
                raise FloatingPointError(f"Nonfinite training waveform at index {index}")
            with autocast_context(device, precision):
                logits = model(waves)
                assert logits.shape == (1, 10)
                assert_finite_logits(logits, phase=f"single-step candidate {index}")
                loss = nn.functional.cross_entropy(logits.float(), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at index {index}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    assert p.grad is None, f"Frozen parameter has gradient: {name}"
            grad_summary = {}
            zero_parts = []
            for prefix, parameter in (
                ("head", model.classifier.weight),
                ("fc1", model.backbone.fc1.weight),
                ("conv6", model.backbone.conv_block6.conv1.weight),
            ):
                if parameter.requires_grad:
                    state, norm = gradient_measure(parameter)
                    if state in ("missing", "nonfinite"):
                        raise AssertionError(f"Invalid gradient for {prefix} (candidate={index}): {state}")
                    grad_summary[prefix] = norm
                    if state == "zero":
                        zero_parts.append(prefix)
            print(f"Candidate {attempt}/{len(indices)}: train-index={index}, classID={class_id}, "
                  f"peak={float(waves.abs().max()):.5f}; gradient norms={grad_summary}", flush=True)
            print("  fc1 input:", observed.get("fc1_input"), flush=True)
            print("  fc1 before ReLU:", observed.get("fc1_pre_relu"), flush=True)
            if zero_parts:
                print("  ZERO GRADIENT:", ", ".join(zero_parts), "— no optimizer update for this candidate", flush=True)
                failed_cases.append({"train_index": index, "class_id": class_id,
                                     "zero_gradients": zero_parts, "features": dict(observed)})
                continue
            chosen = (index, class_id)
            break
    finally:
        in_hook.remove()
        out_hook.remove()
    if chosen is None:
        raise RuntimeError(
            f"None of {len(indices)} training-only clips produced nonzero gradients for all "
            f"required layers ({args.config}). No optimizer step was applied. "
            "Compare --amp off and --feature-mode eval to distinguish precision "
            "from activation/dropout effects. Do not train yet."
        )
    if failed_cases:
        print(f"WARNING: {len(failed_cases)} candidate(s) produced zero gradients. "
              "This must be documented; passing on another clip does not clear that concern.", flush=True)
    print("SELECTED informative train-only candidate:", chosen, "Gradient norms:", grad_summary, flush=True)
    nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    for name, module in model.backbone.named_modules():
        if name in bn_before:
            prev_mean, prev_var, prev_steps = bn_before[name]
            assert torch.equal(module.running_mean.detach().cpu(), prev_mean), f"BN mean changed: {name}"
            assert torch.equal(module.running_var.detach().cpu(), prev_var), f"BN variance changed: {name}"
            assert torch.equal(module.num_batches_tracked.detach().cpu(), prev_steps), f"BN counter changed: {name}"
    assert torch.equal(model.backbone.conv_block1.conv1.weight.detach().flatten()[:512].cpu(), frozen_before), "Frozen conv changed"
    head_delta = float((model.classifier.weight.detach().cpu() - head_before).abs().max())
    assert head_delta > 0, "10-class head did not update"
    if args.config in ("fc1", "block6"):
        fc1_delta = float((model.backbone.fc1.weight.detach().cpu() - fc1_before).abs().max())
        assert fc1_delta > 0, "fc1 did not update"
        print("fc1 update max absolute difference:", fc1_delta, flush=True)
    if args.config == "block6":
        assert conv6_before is not None
        conv6_delta = float((model.backbone.conv_block6.conv1.weight.detach().cpu() - conv6_before).abs().max())
        assert conv6_delta > 0, "conv_block6 CONVOLUTION weights did not update"
        print("conv_block6 CONVOLUTION update max absolute difference:", conv6_delta, flush=True)
    print("New head update max absolute difference:", head_delta, flush=True)
    print("Frozen parameter gradient/update and BatchNorm running-stats checks: PASS", flush=True)
    if device.type == "cuda":
        print("Peak CUDA allocated MiB:", round(torch.cuda.max_memory_allocated()/1024**2, 1), flush=True)
        print("Peak CUDA reserved MiB:", round(torch.cuda.max_memory_reserved()/1024**2, 1), flush=True)
    print("ONE-STEP BACKWARD/UPDATE CHECK: PASS. This is NOT a trained model or evaluation.")
