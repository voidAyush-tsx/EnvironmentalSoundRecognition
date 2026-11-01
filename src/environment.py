"""Environment import checks and explicit direct-dependency snapshots.

Normal imports and version capture do not initialize a model or require CUDA.
Use the project CLI rather than standalone setup scripts.
"""
import argparse
import importlib
import importlib.metadata
from importlib.metadata import PackageNotFoundError, version
import platform
import sys

from .config import DEPENDENCIES

PACKAGES = DEPENDENCIES


def check_environment(*, require_cuda: bool = False) -> list[str]:
    print("Python:", sys.version.split()[0], sys.executable)
    print("Platform:", platform.platform())
    failed = []
    for package, module in PACKAGES:
        try:
            version = importlib.metadata.version(package)
            importlib.import_module(module)
            print(f"PASS  {package:16} {version}")
        except Exception as exc:
            print(f"FAIL  {package:16} {type(exc).__name__}: {exc}")
            failed.append(package)
    if "torch" not in failed:
        import torch
        available = torch.cuda.is_available()
        print("CUDA build:", torch.version.cuda, "| available:", available)
        if available:
            try:
                print("GPU:", torch.cuda.get_device_name(0))
                a = torch.tensor([1.0, 2.0], device="cuda")
                b = torch.tensor([3.0, 4.0], device="cuda")
                if torch.dot(a, b).item() != 11.0:
                    raise RuntimeError("Unexpected GPU computation result")
                print("GPU tensor computation: PASS")
            except Exception as exc:
                print("GPU computation failed:", exc)
                failed.append("GPU tensor computation")
        elif require_cuda:
            failed.append("CUDA required but unavailable")
        else:
            print("CPU development supported; GPU check skipped")
    return failed


def environment_main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check installed Python packages and optional CUDA capability")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    failed = check_environment(require_cuda=args.require_cuda)
    print("Environment:", "FAIL: " + ", ".join(failed) if failed else "PASS")
    return int(bool(failed))


def capture_versions(out) -> None:
    records, missing = [], []
    for name, _ in DEPENDENCIES:
        try:
            records.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            missing.append(name)
    if missing:
        raise ValueError("Cannot record dependencies; missing: " + ", ".join(missing))
    header = ["# Exact direct dependency versions observed locally (not a transitive lock).",
              "# Record Python/platform and the pinned PANNs revision separately."]
    if "+cu128" in version("torch"):
        header.append("--extra-index-url https://download.pytorch.org/whl/cu128")
    out.write_text("\n".join(header + records) + "\n", encoding="utf-8")
    print("Wrote", out)


def snapshot_main(argv=None) -> int:
    from .arguments import snapshot_parser
    parser = snapshot_parser()
    args = parser.parse_args(argv)
    try:
        capture_versions(args.out)
    except (OSError, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")
    return 0
