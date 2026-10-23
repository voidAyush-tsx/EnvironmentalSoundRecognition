"""One real train-only waveform forward/backward/update + frozen BN verification.
Run: python -m src.finetune_check --config block6 --device cuda
Does not touch validation/test examples; does not save a trained checkpoint.
"""
from __future__ import annotations

import argparse
import math
import random
import numpy as np
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from .dataset import load_metadata, split_metadata, UrbanSound8KDataset
from .finetune_model import load_model, parameter_summary, optimizer_for_model, CONFIGS
from .precision_policy import AMP_CHOICES, choose_precision, autocast_context, assert_finite_logits
from .gradient_audit import candidate_indices, gradient_measure, feature_statistics


def choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(value)


def main():
    parser = argparse.ArgumentParser(description="Stage 4 gradient/freeze verification")
    parser.add_argument("--config", choices=CONFIGS, default="block6")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--test-fold", type=int, default=1)
    parser.add_argument("--amp", choices=AMP_CHOICES, default="auto",
                        help="auto: CUDA BF16 if supported, otherwise FP32; on/bf16: require BF16; off: FP32")
    parser.add_argument("--max-candidates", type=int, default=12,
                        help="Probe this many deterministic TRAIN-only clips before concluding gradients stay zero; 1 reproduces old check")
    parser.add_argument("--feature-mode", choices=("train", "eval"), default="train",
                        help="Diagnostic: train=current original PANNs training behavior; eval=disable pretrained SpecAug/dropout, while preserving gradients")
    args = parser.parse_args()
    random.seed(67)
    np.random.seed(67)
    torch.manual_seed(67)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(67)
    device = choose_device(args.device)
    precision = choose_precision(device, args.amp)
    print(f"Device={device}, config={args.config}, precision={precision} (requested={args.amp})", flush=True)
    metadata = load_metadata()
    train_df, val_df, test_df = split_metadata(metadata, args.test_fold)
    dataset = UrbanSound8KDataset(train_df)
    indices = candidate_indices(train_df, args.max_candidates)
    print(f"Probing {len(indices)} deterministic training-only clips; validation/test untouched", flush=True)

    model = load_model(config=args.config).to(device)
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
    print("STAGE 4 ONE-STEP BACKWARD/UPDATE CHECK: PASS. This is NOT a trained model or evaluation.")


if __name__ == "__main__":
    main()
