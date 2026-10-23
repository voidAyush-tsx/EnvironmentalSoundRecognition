"""Precision policy for official PANNs Cnn14 with torchlibrosa log-Mel frontend.

Avoid CUDA FP16 autocast around the whole model: torchlibrosa computes a
power spectrogram before log compression and intermediate values can overflow
in float16. Prefer bfloat16 (FP32 exponent range) on compatible GPUs; else FP32.
No changes are made to the official PANNs source code or pretrained weights.
"""
from __future__ import annotations

from contextlib import nullcontext
import torch

AMP_CHOICES = ("auto", "on", "off", "bf16")


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
