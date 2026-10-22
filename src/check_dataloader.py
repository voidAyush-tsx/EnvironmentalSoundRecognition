"""Stage 3 sanity-check raw-waveform DataLoaders on real UrbanSound8K audio.

Run from project root:
    python -m src.check_dataloader
    python -m src.check_dataloader --model-forward

--model-forward optionally tests one real WAV against the VERIFIED official
PANNs CNN14 safe state_dict, without training or modifying weights.
"""
from __future__ import annotations

import argparse
import sys
import torch

from .dataset import PROJECT_ROOT, load_metadata, make_loaders, TARGET_SAMPLES


def test_real_audio_with_pretrained_backbone(waveforms: torch.Tensor) -> None:
    source_dir = PROJECT_ROOT / "external" / "audioset_tagging_cnn" / "pytorch"
    checkpoint = PROJECT_ROOT / "checkpoints" / "Cnn14_AudioSet_state_dict_safe.pth"
    if not (source_dir / "models.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Official PANNs source or tensor-only safe checkpoint missing")
    sys.path.insert(0, str(source_dir))
    from models import Cnn14
    from .check_cnn14 import UrbanSoundCNN14
    backbone = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
                     mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    backbone.load_state_dict(stored["model"], strict=True)
    del stored
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UrbanSoundCNN14(backbone).to(device).eval()
    with torch.inference_mode():
        logits = model(waveforms[:1].to(device))
    assert logits.shape == (1, 10), logits.shape
    assert torch.isfinite(logits).all().item()
    print("Real UrbanSound8K waveform -> official CNN14 -> [1, 10] logits: PASS")
    print("The 10-class head is UNTRAINED; predictions are not meaningful yet.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify UrbanSound8K waveform DataLoaders")
    parser.add_argument("--test-fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model-forward", action="store_true")
    args = parser.parse_args()
    df = load_metadata()
    train_loader, val_loader, test_loader = make_loaders(
        df, test_fold=args.test_fold, batch_size=args.batch_size, num_workers=0,
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
        test_real_audio_with_pretrained_backbone(first_batch)
    print("Stage 3 DataLoader sanity checks: PASS")
    print("No audio augmentation, model training, or fine-tuning was performed.")


if __name__ == "__main__":
    main()
