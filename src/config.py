"""Project defaults and historical provenance. Importing this needs only Python."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "UrbanSound8K"
PANN_SOURCE = PROJECT_ROOT / "external" / "audioset_tagging_cnn"
SAFE_WEIGHTS = PROJECT_ROOT / "checkpoints" / "Cnn14_AudioSet_state_dict_safe.pth"
ORIGINAL_WEIGHTS = PROJECT_ROOT / "checkpoints" / "Cnn14_mAP=0.431.pth"
EXPECTED_PANN_COMMIT = "d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4"
EXPECTED_MD5 = "595633ac2d1cac7ef04ebf70e2fee4e4"
CLASS_NAMES = (
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
SAMPLE_RATE = 32_000
CLIP_SECONDS = 4
TARGET_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
CONFIGS = ("head", "fc1", "block6")
AMP_CHOICES = ("auto", "on", "off", "bf16")
CNN14_FRONTEND = dict(sample_rate=SAMPLE_RATE, window_size=1024, hop_size=320,
                      mel_bins=64, fmin=50, fmax=14000, classes_num=527)
DEPENDENCIES = (
    ("torch", "torch"), ("torchlibrosa", "torchlibrosa.stft"), ("numpy", "numpy"),
    ("scipy", "scipy"), ("librosa", "librosa"), ("soundfile", "soundfile"),
    ("pandas", "pandas"), ("scikit-learn", "sklearn"), ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"), ("tqdm", "tqdm"), ("ipykernel", "ipykernel"),
)

# Historical identity is independent of defaults for NEW training runs.
SELECTED_RUN_RELATIVE = Path("results/stage4/20261027_144004_603447_block6_test1")
SELECTED_SHA256 = "7c0e01a0078c225a18f4dc5eaa5110d051ef2c8effec7bcebb4e856e6659328c"
HISTORICAL_TRAINING = dict(epochs=6, patience=2, seed=67, batch_size=1, accum_steps=4)


def project_path(value: str | Path) -> Path:
    """Resolve relative CLI/config paths against this checkout, not the shell cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def add_asset_arguments(parser, *, dataset: bool = True) -> None:
    if dataset:
        parser.add_argument("--dataset-root", type=project_path, default=DATASET_ROOT,
                            help="UrbanSound8K root (relative paths use the project root)")
    parser.add_argument("--weights", type=project_path, default=SAFE_WEIGHTS,
                        help="Restricted-loadable official AudioSet backbone state dict")
    parser.add_argument("--pann-source", type=project_path, default=PANN_SOURCE,
                        help="Pinned official PANNs repository root")
