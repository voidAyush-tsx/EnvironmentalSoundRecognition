"""Save exact local dependency versions without copying unrelated pip packages.

Run ONLY after the initial requirements were installed and checked:
    python finalize_requirements.py
Creates requirements.lock.txt beside this script.
"""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

names = [
    "torch", "torchlibrosa", "numpy", "scipy", "librosa", "soundfile",
    "pandas", "scikit-learn", "matplotlib", "seaborn", "tqdm", "ipykernel",
]

records, missing = [], []
for name in names:
    try:
        records.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        missing.append(name)

if missing:
    raise SystemExit("Cannot lock dependencies; missing: " + ", ".join(missing))

# The local torch version includes its CUDA build suffix, e.g. +cu128.
# Provide the correct official wheel source when rebuilding the environment.
torch_version = version("torch")
header = [
    "# Exact package versions observed on the local machine.",
    "# Generated only after local installation; use together with Python version",
    "# and a documented official PANNs source commit for reproducibility.",
]
if "+cu128" in torch_version:
    header += ["--extra-index-url https://download.pytorch.org/whl/cu128"]

out = Path(__file__).resolve().parent / "requirements.lock.txt"
out.write_text("\n".join(header + records) + "\n", encoding="utf-8")
print(f"Wrote {out}")
print("PyTorch version:", torch_version)
