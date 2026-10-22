"""Stage 2: verify official AudioSet-pretrained PANNs CNN14 on UrbanSound8K.

Run from project root:
    python src/check_cnn14.py --trust-legacy-checkpoint --export-safe-weights
Then for subsequent checks:
    python src/check_cnn14.py --use-safe-weights

The original checkpoint may contain legacy NumPy objects. `--trust-legacy-checkpoint`
permits unrestricted pickle ONLY after the file MD5 matches the published
Zenodo checksum; this does not eliminate the inherent pickle security risk.
Do not use this option with checkpoints from unknown or untrusted sources.
No training or freezing takes place in this script.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from collections.abc import Mapping

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT / "external" / "audioset_tagging_cnn"
CHECKPOINT = ROOT / "checkpoints" / "Cnn14_mAP=0.431.pth"
SAFE_CHECKPOINT = ROOT / "checkpoints" / "Cnn14_AudioSet_state_dict_safe.pth"
# MD5 provided by the official Zenodo record for the specified original file.
EXPECTED_MD5 = "595633ac2d1cac7ef04ebf70e2fee4e4"


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
    digest = hashlib.md5()  # Integrity comparison to Zenodo's MD5, NOT a security guarantee.
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_weights(args: argparse.Namespace) -> Mapping:
    if args.use_safe_weights:
        path = SAFE_CHECKPOINT
        if not path.is_file():
            raise FileNotFoundError(
                f"Safe state dictionary missing: {path}\n"
                "First run with --trust-legacy-checkpoint --export-safe-weights."
            )
        print("Loading tensor-only exported state dictionary (restricted loader) ...")
        container = torch.load(path, map_location="cpu", weights_only=True)
    else:
        path = CHECKPOINT
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
                    "  python src/check_cnn14.py --trust-legacy-checkpoint --export-safe-weights\n"
                    "After that, future checks can use:\n"
                    "  python src/check_cnn14.py --use-safe-weights"
                ) from exc

    if not isinstance(container, Mapping) or "model" not in container:
        raise ValueError("Checkpoint format incorrect: expected a mapping with key 'model'")
    state = container["model"]
    if not isinstance(state, Mapping) or not state:
        raise TypeError("Checkpoint['model'] must be a non-empty state dictionary")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise TypeError("Checkpoint['model'] must contain only string keys and torch.Tensor values")
    print("Model state_dict parameter/buffer keys:", len(state))
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify AudioSet pretrained CNN14 and ten-logit wrapper")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for forward-pass verification")
    parser.add_argument(
        "--trust-legacy-checkpoint", action="store_true",
        help="Opt in to unrestricted pickle for an MD5-matched official download ONLY",
    )
    parser.add_argument(
        "--export-safe-weights", action="store_true",
        help="Save model state_dict only, for future restricted weights-only loading (~1.4 GB)",
    )
    parser.add_argument(
        "--use-safe-weights", action="store_true",
        help="Read previously exported tensor-only state_dict instead of original checkpoint",
    )
    args = parser.parse_args()
    if args.use_safe_weights and args.export_safe_weights:
        parser.error("--use-safe-weights and --export-safe-weights cannot be used together")

    pytorch_dir = REPOSITORY / "pytorch"
    required_files = [pytorch_dir / "models.py", pytorch_dir / "pytorch_utils.py"]
    for file in required_files:
        if not file.is_file():
            raise FileNotFoundError(f"Official PANNs source file missing: {file}")

    # The official models.py imports pytorch_utils from the same pytorch/ directory.
    sys.path.insert(0, str(pytorch_dir))
    from models import Cnn14  # type: ignore[import-not-found]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("PyTorch:", torch.__version__)
    print("Inference device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU total memory (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

    # Construct the ORIGINAL AudioSet model, including original 527-class layer.
    backbone = Cnn14(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
        fmin=50, fmax=14000, classes_num=527,
    )
    state_dict = read_weights(args)
    load_result = backbone.load_state_dict(state_dict, strict=True)
    assert not load_result.missing_keys and not load_result.unexpected_keys
    print("Strict pretrained checkpoint loading: PASS (no missing or unexpected keys)")

    if args.export_safe_weights:
        SAFE_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        print("Exporting tensor-only model state_dict:", SAFE_CHECKPOINT)
        torch.save({"model": backbone.state_dict()}, SAFE_CHECKPOINT)
        # Confirm it can be reloaded with restricted deserialization.
        exported = torch.load(SAFE_CHECKPOINT, map_location="cpu", weights_only=True)
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


if __name__ == "__main__":
    main()
