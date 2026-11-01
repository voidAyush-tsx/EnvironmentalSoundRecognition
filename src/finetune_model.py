"""Stage 4: verified official PANNs Cnn14 + 10-class head and freeze policies.

The AudioSet architecture is instantiated unchanged, loaded STRICTLY from the
Stage-2 tensor-only checkpoint, and retained. Its 527-class layer stays frozen.
Frozen BatchNorm statistics (including bn0 and the selected conv blocks) are
kept in evaluation mode while learning trainable affine parameters if enabled.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from .config import PROJECT_ROOT, PANN_SOURCE, SAFE_WEIGHTS, CONFIGS, EXPECTED_PANN_COMMIT
from .pann_adapter import actual_source_commit, load_backbone

# Compatibility alias for callers that inspected this directory.
PANN_DIR = PANN_SOURCE / "pytorch"


class CNN14ForUrbanSound(nn.Module):
    def __init__(self, backbone: nn.Module, config: str = "block6") -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(2048, 10)
        self.config = ""
        self.configure(config)

    def configure(self, config: str) -> None:
        if config not in CONFIGS:
            raise ValueError(f"Invalid configuration: {config}; choose {CONFIGS}")
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.classifier.parameters():
            p.requires_grad_(True)
        if config in ("fc1", "block6"):
            for p in self.backbone.fc1.parameters():
                p.requires_grad_(True)
        if config == "block6":
            for p in self.backbone.conv_block6.parameters():
                p.requires_grad_(True)
        self.config = config
        self.train(self.training)  # reapply BatchNorm-running-stat policy

    def train(self, mode: bool = True):
        super().train(mode)
        # Small GPU batches make running-stat updates unreliable. Hold ALL
        # pretrained BN means/variances fixed, including trainable block6 BN.
        # Affine gamma/beta in an unfrozen block still get gradients.
        for module in self.backbone.modules():
            if isinstance(module, _BatchNorm):
                module.eval()
        return self

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # Official backbone accepts [B, 128000] and computes log-Mel internally.
        features = self.backbone(waveforms)["embedding"]
        logits = self.classifier(features)  # raw logits: NOT softmax
        return logits


def load_model(config: str = "block6", *, weights: str | Path = SAFE_WEIGHTS,
               source: str | Path = PANN_SOURCE) -> CNN14ForUrbanSound:
    if config not in CONFIGS:
        raise ValueError(f"Unknown fine-tuning config: {config}")
    return CNN14ForUrbanSound(load_backbone(weights=weights, source=source), config=config)


def parameter_summary(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def trainable_state(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    # Running BN statistics don't change under our frozen BN policy.
    return OrderedDict((name, p.detach().cpu().clone())
                       for name, p in model.named_parameters() if p.requires_grad)


def load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    target = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != set(target):
        missing = set(target) - set(state)
        extra = set(state) - set(target)
        raise ValueError(f"Selected-parameter checkpoint mismatch. Missing {missing}, extra {extra}")
    with torch.no_grad():
        for name, param in target.items():
            if param.shape != state[name].shape:
                raise ValueError(f"Shape mismatch in checkpoint: {name}")
            param.copy_(state[name].to(device=param.device, dtype=param.dtype))


def optimizer_for_model(model: CNN14ForUrbanSound, *, lr_head: float = 1e-3,
                        lr_backbone: float = 1e-5, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    if lr_head <= 0 or lr_backbone <= 0 or weight_decay < 0:
        raise ValueError("Invalid optimizer hyperparameters")
    groups = [{"params": list(model.classifier.parameters()), "lr": lr_head}]
    later = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    if later:
        groups.append({"params": later, "lr": lr_backbone})
    optim = torch.optim.AdamW(groups, weight_decay=weight_decay)
    optimized = {id(p) for group in optim.param_groups for p in group["params"]}
    intended = {id(p) for p in model.parameters() if p.requires_grad}
    if optimized != intended:
        raise AssertionError("Optimizer does not exactly match trainable parameters")
    return optim
