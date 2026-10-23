"""Transparent train-fold-only gradient diagnostics for original CNN14.

A zero fc1.weight gradient on one waveform can occur even if the downstream
head gets a nonzero gradient. This module NEVER turns a zero into a PASS.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import pandas as pd
import torch


def candidate_indices(train_df: pd.DataFrame, max_candidates: int = 12) -> list[int]:
    """Deterministic first clip, class representatives, then consecutive clips.

    Only positional indices *within the training subset* are returned.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if not len(train_df):
        raise ValueError("Training subset is empty")
    result: list[int] = []
    seen: set[int] = set()
    def add(i: int) -> None:
        if i not in seen and len(result) < max_candidates:
            result.append(i)
            seen.add(i)
    add(0)  # reproduce original single-clip failing case
    if "classID" in train_df.columns:
        # The first clip of each class gives a small diverse diagnostic sample.
        labels = train_df["classID"].to_numpy()
        for label in range(10):
            matches = (labels == label).nonzero()[0]
            if len(matches):
                add(int(matches[0]))
    for i in range(len(train_df)):
        add(i)
        if len(result) >= max_candidates:
            break
    return result


def gradient_measure(parameter: torch.nn.Parameter) -> tuple[str, float | None]:
    if parameter.grad is None:
        return "missing", None
    grad = parameter.grad.detach()
    if not torch.isfinite(grad).all():
        return "nonfinite", None
    norm = float(torch.linalg.vector_norm(grad.float()).item())
    if not math.isfinite(norm):
        return "nonfinite", None
    if norm == 0.0:
        return "zero", 0.0
    return "nonzero", norm


def feature_statistics(tensor: torch.Tensor) -> dict[str, float]:
    """Capture scalars inside a hook to avoid retaining a CUDA computation graph."""
    x = tensor.detach().float()
    return {"max_abs": float(x.abs().max().item()),
            "rms": float(x.square().mean().sqrt().item()),
            "nonzero_fraction": float((x != 0).float().mean().item()),
            "positive_fraction": float((x > 0).float().mean().item()),
            "finite": float(torch.isfinite(x).all().item())}
