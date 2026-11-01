"""The only boundary to the unchanged official PANNs source and backbone weights."""
from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import types

import torch

from .config import CNN14_FRONTEND, EXPECTED_PANN_COMMIT, PANN_SOURCE, SAFE_WEIGHTS, project_path


def actual_source_commit(source: str | Path = PANN_SOURCE) -> str:
    source = project_path(source)
    try:
        top = subprocess.check_output(["git", "-C", str(source), "rev-parse", "--show-toplevel"],
                                      text=True, stderr=subprocess.PIPE).strip()
        if Path(top).resolve() != source:
            raise ValueError("source must be its own Git checkout")
        return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(f"Cannot verify PANNs checkout at {source}. See docs/REPRODUCIBILITY.md") from exc


def _load_module(name: str, file: Path):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@contextmanager
def _upstream_utils_alias(module):
    """Satisfy upstream's absolute import without retaining an ambiguous global module."""
    previous = sys.modules.get("pytorch_utils")
    sys.modules["pytorch_utils"] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("pytorch_utils", None)
        else:
            sys.modules["pytorch_utils"] = previous


def new_backbone(source: str | Path = PANN_SOURCE):
    source = project_path(source)
    directory = source / "pytorch"
    for name in ("models.py", "pytorch_utils.py"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Official PANNs implementation missing: {directory / name}. "
                                    "Run git submodule update --init; see docs/REPRODUCIBILITY.md")
    found = actual_source_commit(source)
    if found != EXPECTED_PANN_COMMIT:
        raise RuntimeError(f"PANNs source commit mismatch: {found} != {EXPECTED_PANN_COMMIT}")
    package = "_esr_panns_" + hashlib.sha256(str(source).encode()).hexdigest()[:12]
    if package + ".models" not in sys.modules:
        namespace = types.ModuleType(package)
        namespace.__path__ = [str(directory)]
        sys.modules[package] = namespace
        utils = _load_module(package + ".pytorch_utils", directory / "pytorch_utils.py")
        with _upstream_utils_alias(utils):
            _load_module(package + ".models", directory / "models.py")
    return sys.modules[package + ".models"].Cnn14(**CNN14_FRONTEND)


def validate_backbone_state(container) -> Mapping:
    if not isinstance(container, Mapping) or "model" not in container:
        raise ValueError("Checkpoint format incorrect: expected a mapping with key 'model'")
    state = container["model"]
    if not isinstance(state, Mapping) or not state:
        raise TypeError("Checkpoint['model'] must be a non-empty state dictionary")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise TypeError("Checkpoint['model'] must contain only string keys and torch.Tensor values")
    return state


def load_backbone(*, weights: str | Path = SAFE_WEIGHTS, source: str | Path = PANN_SOURCE):
    weights = project_path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Missing restricted-load AudioSet state dictionary: {weights}. "
                                "See docs/REPRODUCIBILITY.md#checkpoint-reconstruction for conversion instructions")
    backbone = new_backbone(source)
    state = validate_backbone_state(torch.load(weights, map_location="cpu", weights_only=True))
    backbone.load_state_dict(state, strict=True)
    return backbone
