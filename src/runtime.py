"""Shared device, reproducible seeding, and BF16/FP32 numerical policy."""
from contextlib import nullcontext
import random

import numpy as np
import torch

from .config import AMP_CHOICES


def resolve_device(device_str: str) -> torch.device:
    choice = ("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else device_str
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but not found; use --device cpu or --device auto")
    return torch.device(choice)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_precision(device: torch.device, requested: str = "auto") -> str:
    if requested not in AMP_CHOICES:
        raise ValueError(f"Invalid --amp={requested!r}; allowed: {AMP_CHOICES}")
    if device.type != "cuda":
        if requested in ("on", "bf16"):
            raise ValueError("--amp on/bf16 requires a CUDA device; select --amp off")
        return "off"
    if requested == "off":
        return "off"
    has_bf16 = bool(torch.cuda.is_bf16_supported())
    if has_bf16:
        return "bf16"
    if requested == "auto":
        # Prior float16 whole-model autocast produced nonfinite CNN14 logits.
        # Full FP32 is a safe functional fallback; it may need more VRAM.
        return "off"
    raise RuntimeError("CUDA bfloat16 is unsupported on this GPU; use --amp off")


def autocast_context(device: torch.device, mode: str | bool):
    """bool accepted for backward compatibility with existing CPU unit tests."""
    if isinstance(mode, bool):
        mode = "bf16" if mode else "off"
    if mode == "off":
        return nullcontext()
    if mode == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unsupported autocast mode/device: {mode}/{device.type}")


def assert_finite_logits(logits: torch.Tensor, *, phase: str) -> None:
    if not torch.isfinite(logits).all():
        nbad = int((~torch.isfinite(logits)).sum().item())
        raise FloatingPointError(
            f"{phase}: {nbad}/{logits.numel()} logits are nonfinite. "
            "Check --amp mode and waveform finiteness; never train on NaN/Inf. "
            "Try --amp off for a full-float32 diagnostic."
        )
