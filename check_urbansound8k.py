
from pathlib import Path
import csv

# Change this path to your actual extracted dataset folder.
DATASET_DIR = Path(r"D:\EnvironmentalSoundRecognition\data\UrbanSound8K")

metadata_file = DATASET_DIR / "metadata" / "UrbanSound8K.csv"
audio_dir = DATASET_DIR / "audio"

print("Dataset directory exists:", DATASET_DIR.exists())
print("Metadata file exists:", metadata_file.exists())
print("Audio directory exists:", audio_dir.exists())

if not metadata_file.exists() or not audio_dir.exists():
    raise FileNotFoundError(
        "Dataset structure not recognized. Check DATASET_DIR "
        "and the extracted folder names."
    )

with metadata_file.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("\nMetadata rows:", len(rows))
print("CSV columns:", list(rows[0].keys()) if rows else [])

# Count WAV files in the ten official fold directories.
fold_counts = {}
for fold_num in range(1, 11):
    fold_dir = audio_dir / f"fold{fold_num}"
    files = list(fold_dir.glob("*.wav")) if fold_dir.exists() else []
    fold_counts[fold_num] = len(files)

print("\nWAV files per fold:")
for fold_num, count in fold_counts.items():
    print(f"fold{fold_num}: {count}")

actual_wavs = sum(fold_counts.values())
print("\nTotal WAV files found:", actual_wavs)

# Validate metadata references and labels.
required_columns = {"slice_file_name", "fold", "classID", "class"}
missing_columns = required_columns - set(rows[0].keys()) if rows else required_columns

if missing_columns:
    raise ValueError(f"Missing required CSV columns: {missing_columns}")

missing_files = []
invalid_folds = []
invalid_classes = []

for row in rows:
    try:
        fold_num = int(row["fold"])
        class_id = int(row["classID"])
    except (ValueError, TypeError):
        invalid_folds.append(row)
        continue

    if fold_num not in range(1, 11):
        invalid_folds.append(row)
        continue

    if class_id not in range(10):
        invalid_classes.append(row)

    wav_path = audio_dir / f"fold{fold_num}" / row["slice_file_name"]
    if not wav_path.is_file():
        missing_files.append(str(wav_path))

print("\nUnique classes:", len({r["class"] for r in rows}))
print("Unique fold IDs:", sorted({r["fold"] for r in rows}))
print("Missing referenced files:", len(missing_files))
print("Invalid fold records:", len(invalid_folds))
print("Invalid class records:", len(invalid_classes))

if missing_files:
    print("\nFirst missing paths:")
    for path in missing_files[:10]:
        print(path)

passed = (
    len(rows) == 8732
    and actual_wavs == 8732
    and len({r["class"] for r in rows}) == 10
    and len({r["fold"] for r in rows}) == 10
    and not missing_files
    and not invalid_folds
    and not invalid_classes
)

print("\nDATASET AUDIT:", "PASS" if passed else "CHECK REQUIRED")

from pathlib import Path
import csv

# Change this path to your actual extracted dataset folder.
DATASET_DIR = Path(r"D:\EnvironmentalSoundRecognition\data\UrbanSound8K")

metadata_file = DATASET_DIR / "metadata" / "UrbanSound8K.csv"
audio_dir = DATASET_DIR / "audio"

print("Dataset directory exists:", DATASET_DIR.exists())
print("Metadata file exists:", metadata_file.exists())
print("Audio directory exists:", audio_dir.exists())

if not metadata_file.exists() or not audio_dir.exists():
    raise FileNotFoundError(
        "Dataset structure not recognized. Check DATASET_DIR "
        "and the extracted folder names."
    )

with metadata_file.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("\nMetadata rows:", len(rows))
print("CSV columns:", list(rows[0].keys()) if rows else [])

# Count WAV files in the ten official fold directories.
fold_counts = {}
for fold_num in range(1, 11):
    fold_dir = audio_dir / f"fold{fold_num}"
    files = list(fold_dir.glob("*.wav")) if fold_dir.exists() else []
    fold_counts[fold_num] = len(files)

print("\nWAV files per fold:")
for fold_num, count in fold_counts.items():
    print(f"fold{fold_num}: {count}")

actual_wavs = sum(fold_counts.values())
print("\nTotal WAV files found:", actual_wavs)

# Validate metadata references and labels.
required_columns = {"slice_file_name", "fold", "classID", "class"}
missing_columns = required_columns - set(rows[0].keys()) if rows else required_columns

if missing_columns:
    raise ValueError(f"Missing required CSV columns: {missing_columns}")

missing_files = []
invalid_folds = []
invalid_classes = []

for row in rows:
    try:
        fold_num = int(row["fold"])
        class_id = int(row["classID"])
    except (ValueError, TypeError):
        invalid_folds.append(row)
        continue

    if fold_num not in range(1, 11):
        invalid_folds.append(row)
        continue

    if class_id not in range(10):
        invalid_classes.append(row)

    wav_path = audio_dir / f"fold{fold_num}" / row["slice_file_name"]
    if not wav_path.is_file():
        missing_files.append(str(wav_path))

print("\nUnique classes:", len({r["class"] for r in rows}))
print("Unique fold IDs:", sorted({r["fold"] for r in rows}))
print("Missing referenced files:", len(missing_files))
print("Invalid fold records:", len(invalid_folds))
print("Invalid class records:", len(invalid_classes))

if missing_files:
    print("\nFirst missing paths:")
    for path in missing_files[:10]:
        print(path)

passed = (
    len(rows) == 8732
    and actual_wavs == 8732
    and len({r["class"] for r in rows}) == 10
    and len({r["fold"] for r in rows}) == 10
    and not missing_files
    and not invalid_folds
    and not invalid_classes
)

print("\nDATASET AUDIT:", "PASS" if passed else "CHECK REQUIRED")