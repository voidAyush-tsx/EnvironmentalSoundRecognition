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
    parser.add_argument("--amp", choices=("auto", "on", "off"), default="auto")
    args = parser.parse_args()
    random.seed(67)
    np.random.seed(67)
    torch.manual_seed(67)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(67)
    device = choose_device(args.device)
    amp = (device.type == "cuda") if args.amp == "auto" else (args.amp == "on")
    if amp and device.type != "cuda":
        parser.error("AMP on is supported only with CUDA in this pilot")
    print(f"Device={device}, config={args.config}, AMP={amp}", flush=True)
    metadata = load_metadata()
    train_df, val_df, test_df = split_metadata(metadata, args.test_fold)
    waveform, class_id = UrbanSound8KDataset(train_df)[0]  # ONLY training fold
    print(f"Train-only input [1,{waveform.numel()}], classID={class_id}; validation/test untouched", flush=True)

    model = load_model(config=args.config).to(device)
    info = parameter_summary(model)
    print("Parameter counts:", info, flush=True)
    print("Trainable prefixes:", sorted({".".join(n.split(".")[:2])
                                       for n, p in model.named_parameters() if p.requires_grad}), flush=True)
    model.train()
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
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    optimizer.zero_grad(set_to_none=True)
    waves = waveform.unsqueeze(0).to(device)
    labels = torch.tensor([class_id], dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        logits = model(waves)
        assert logits.shape == (1, 10)
        assert torch.isfinite(logits).all(), "Nonfinite logits"
        loss = nn.functional.cross_entropy(logits, labels)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, f"Frozen parameter has gradient: {name}"
    grad_summary = {}
    for prefix, parameter in (
        ("head", model.classifier.weight),
        ("fc1", model.backbone.fc1.weight),
        ("conv6", model.backbone.conv_block6.conv1.weight),
    ):
        if parameter.requires_grad:
            assert parameter.grad is not None, f"Missing gradient for {prefix}"
            norm = float(parameter.grad.float().norm().item())
            if not math.isfinite(norm) or norm <= 0:
                raise AssertionError(f"Invalid/zero gradient for {prefix}: {norm}")
            grad_summary[prefix] = norm
    print("Gradient norms:", grad_summary, flush=True)
    nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
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
