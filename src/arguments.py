"""Dependency-free argument parsers for the unified project CLI."""
import argparse
from pathlib import Path
from .config import (
    CONFIGS, DATASET_ROOT, PROJECT_ROOT, ORIGINAL_WEIGHTS as CHECKPOINT,
    add_asset_arguments, project_path,
)
from .config import AMP_CHOICES
ROOT = PROJECT_ROOT
DEFAULT_METADATA = DATASET_ROOT / "metadata" / "UrbanSound8K.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "results"

def training_parser():
    parser = argparse.ArgumentParser(description="Train official pretrained CNN14 on UrbanSound8K")
    parser.add_argument("--config", choices=CONFIGS, default="block6")
    parser.add_argument("--test-fold", type=int, choices=range(1, 11), metavar="1..10", default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="0=full training fold; positive=diagnostic-only batch limit")
    parser.add_argument("--max-val-batches", type=int, default=0,
                        help="0=full validation fold; positive=diagnostic-only batch limit")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", choices=AMP_CHOICES, default="auto",
                        help="auto uses BF16 if supported; otherwise FP32. on/bf16 requires BF16")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--out", type=project_path, default=None,
                        help="Optional unique output directory; existing best.pt won't be overwritten")
    add_asset_arguments(parser)
    return parser


def evaluation_parser():
    ap = argparse.ArgumentParser(description="ONE held-out fold evaluation of the validation-selected block6 checkpoint")
    ap.add_argument("--run-dir", type=project_path, required=True, help="Completed block6 training run folder containing best.pt")
    ap.add_argument("--expected-test-fold", type=int, default=1)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--amp", choices=("auto", "off", "bf16", "on"), default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out", type=project_path, default=None)
    add_asset_arguments(ap)
    return ap


def gradients_parser():
    parser = argparse.ArgumentParser(description="Verify trainable-layer gradients and frozen statistics")
    parser.add_argument("--config", choices=CONFIGS, default="block6")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--test-fold", type=int, default=1)
    parser.add_argument("--amp", choices=AMP_CHOICES, default="auto",
                        help="auto: CUDA BF16 if supported, otherwise FP32; on/bf16: require BF16; off: FP32")
    parser.add_argument("--max-candidates", type=int, default=12,
                        help="Probe this many deterministic TRAIN-only clips before concluding gradients stay zero; 1 reproduces old check")
    parser.add_argument("--feature-mode", choices=("train", "eval"), default="train",
                        help="Diagnostic: train=current original PANNs training behavior; eval=disable pretrained SpecAug/dropout, while preserving gradients")
    add_asset_arguments(parser)
    return parser


def model_parser():
    parser = argparse.ArgumentParser(description="Verify AudioSet pretrained CNN14 and ten-logit wrapper")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for forward-pass verification")
    parser.add_argument(
        "--trust-legacy-checkpoint", action="store_true",
        help="Opt in to unrestricted pickle for an MD5-matched official download ONLY",
    )
    parser.add_argument(
        "--export-safe-weights", action="store_true",
        help="Save a tensor-only model state dict for subsequent restricted loading",
    )
    parser.add_argument(
        "--use-safe-weights", action="store_true",
        help="Read previously exported tensor-only state_dict instead of original checkpoint",
    )
    add_asset_arguments(parser, dataset=False)
    parser.add_argument("--original-weights", type=project_path, default=CHECKPOINT)
    return parser


def audio_parser():
    parser = argparse.ArgumentParser(description="Audit UrbanSound8K audio and generate figures")
    parser.add_argument("--dataset-root", type=project_path, default=DATASET_ROOT)
    parser.add_argument("--results-dir", type=project_path, default=PROJECT_ROOT / "results")
    parser.add_argument("--figures-dir", type=project_path, default=PROJECT_ROOT / "figures")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Scan only first N files for a quick smoke test; NOT a full integrity audit")
    return parser


def folds_parser():
    parser = argparse.ArgumentParser(description="Audit UrbanSound8K official folds")
    parser.add_argument("--metadata", type=project_path, default=DEFAULT_METADATA)
    parser.add_argument("--results-dir", type=project_path, default=DEFAULT_RESULTS)
    return parser


def dataloader_parser():
    parser = argparse.ArgumentParser(description="Verify UrbanSound8K waveform DataLoaders")
    parser.add_argument("--test-fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model-forward", action="store_true")
    add_asset_arguments(parser)
    return parser


def dataset_parser():
    parser = argparse.ArgumentParser(description='Validate metadata and WAV paths without decoding audio.')
    parser.add_argument("--dataset-root", type=project_path, default=DATASET_ROOT)
    return parser


def faculty_parser():
    p = argparse.ArgumentParser(description='Audit and package a faculty-ready UrbanSound8K/PANNs CNN14 project.')
    p.add_argument('command', choices=['audit', 'package'])
    p.add_argument('--project', type=project_path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--checkpoint', type=project_path, default=None, help='Optional path to the exact evaluated block6 best.pt')
    p.add_argument('--registration', default='', help='Registration number (digits only)')
    p.add_argument('--student-name', default='')
    p.add_argument('--section', default='')
    return p


def snapshot_parser():
    parser = argparse.ArgumentParser(description='Explicitly record direct dependency versions; this is not a transitive lock.')
    parser.add_argument("--out", type=project_path, default=PROJECT_ROOT / "requirements.lock.txt")
    return parser


PARSERS = {
    'train': training_parser,
    'evaluate': evaluation_parser,
    'check-gradients': gradients_parser,
    'check-model': model_parser,
    'check-loader': dataloader_parser,
    'audit-audio': audio_parser,
    'audit-folds': folds_parser,
    'audit-data': dataset_parser,
    'faculty': faculty_parser,
    'snapshot-env': snapshot_parser,
}
