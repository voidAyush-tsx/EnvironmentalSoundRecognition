"""Stage 4: verified official PANNs Cnn14 + 10-class head and freeze policies.

The AudioSet architecture is instantiated unchanged, loaded STRICTLY from the
Stage-2 tensor-only checkpoint, and retained. Its 527-class layer stays frozen.
Frozen BatchNorm statistics (including bn0 and the selected conv blocks) are
kept in evaluation mode while learning trainable affine parameters if enabled.
"""
from __future__ import annotations

import sys
import subprocess
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from .dataset import PROJECT_ROOT

PANN_DIR = PROJECT_ROOT / "external" / "audioset_tagging_cnn" / "pytorch"
SAFE_WEIGHTS = PROJECT_ROOT / "checkpoints" / "Cnn14_AudioSet_state_dict_safe.pth"
CONFIGS = ("head", "fc1", "block6")
EXPECTED_PANN_COMMIT = "d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4"


def actual_source_commit() -> str:
    result = subprocess.check_output(
        ["git", "-C", str(PANN_DIR.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    return result



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


def load_model(config: str = "block6", *, weights: str | Path = SAFE_WEIGHTS) -> CNN14ForUrbanSound:
    if config not in CONFIGS:
        raise ValueError(f"Unknown fine-tuning config: {config}")
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Missing restricted-load AudioSet state dictionary: {weights}")
    for file in (PANN_DIR / "models.py", PANN_DIR / "pytorch_utils.py"):
        if not file.is_file():
            raise FileNotFoundError(f"Official PANNs implementation missing: {file}")
    found_commit = actual_source_commit()
    if found_commit != EXPECTED_PANN_COMMIT:
        raise RuntimeError(f"PANNs source commit mismatch: {found_commit} != {EXPECTED_PANN_COMMIT}")
    sys.path.insert(0, str(PANN_DIR))
    from models import Cnn14  # type: ignore[import-not-found]

    backbone = Cnn14(
        sample_rate=32000, window_size=1024, hop_size=320,
        mel_bins=64, fmin=50, fmax=14000, classes_num=527,
    )
    packaged = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(packaged, dict) or "model" not in packaged:
        raise ValueError("Expected tensor-only checkpoint with 'model' key")
    result = backbone.load_state_dict(packaged["model"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    del packaged
    return CNN14ForUrbanSound(backbone, config=config)


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
