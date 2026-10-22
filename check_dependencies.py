"""Stage 2 local dependency and GPU audit for PANNs / UrbanSound8K.

Run from the same Python interpreter that will execute the training notebook:
    python check_dependencies.py
"""

import importlib
import importlib.metadata
import platform
import sys

PACKAGES = [
    ("torch", "torch"),
    ("torchlibrosa", "torchlibrosa.stft"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("pandas", "pandas"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("tqdm", "tqdm"),
    ("ipykernel", "ipykernel"),
]

print("=== PYTHON ENVIRONMENT ===")
print("Executable:", sys.executable)
print("Version:", sys.version.split()[0])
print("Platform:", platform.platform())
print("\n=== PACKAGE IMPORT TEST ===")

failed = []
for package, module in PACKAGES:
    try:
        version = importlib.metadata.version(package)
        importlib.import_module(module)
        print(f"PASS  {package:16} {version}")
    except Exception as exc:
        print(f"FAIL  {package:16} {type(exc).__name__}: {exc}")
        failed.append(package)

print("\n=== PYTORCH / CUDA ===")
try:
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    cuda = torch.cuda.is_available()
    print("CUDA available:", cuda)
    if cuda:
        name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print("GPU:", name)
        print(f"GPU VRAM: {memory:.2f} GiB")
        # Check that a small GPU computation really executes successfully.
        a = torch.tensor([1.0, 2.0], device="cuda")
        b = torch.tensor([3.0, 4.0], device="cuda")
        assert torch.dot(a, b).item() == 11.0
        print("GPU tensor computation: PASS")
    else:
        failed.append("CUDA availability")
except Exception as exc:
    print("GPU test: FAIL", type(exc).__name__, str(exc))
    failed.append("GPU tensor computation")

print("\n=== FINAL STATUS ===")
if failed:
    print("CHECK REQUIRED:", ", ".join(failed))
    sys.exit(1)
print("PASS: all imports and CUDA test succeeded")
