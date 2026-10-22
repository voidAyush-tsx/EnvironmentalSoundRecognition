#!/usr/bin/env python3
"""Run Stage 2 + Stage 3 verification for UrbanSound8K / official PANNs CNN14.

From the project root on Windows:
    python run_all_checks.py

Optional:
    python run_all_checks.py --quick       # skips the 8,732-file full audit
    python run_all_checks.py --setup       # OPTIONAL install requirements first
    python run_all_checks.py --offline     # never auto-clone missing official PANNs code
    python run_all_checks.py --skip-cpu    # omit the repeat CPU CNN14 forward pass
    python run_all_checks.py --dry-run     # inspect planned checks without executing

Auto-clones the small official PANNs source repository if missing and pins the
revision that was previously verified on the student's PC. Never overwrites an
existing clone or reinstalls packages unless --setup is explicitly passed.
Does NOT train, fine-tune, change UrbanSound8K's folds, or invoke unsafe pickle.
Requires an existing verified tensor-only safe CNN14 weights file for model tests.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "checkpoints" / "Cnn14_mAP=0.431.pth"
SAFE = ROOT / "checkpoints" / "Cnn14_AudioSet_state_dict_safe.pth"
PANN_REPO = ROOT / "external" / "audioset_tagging_cnn"
PANN_GIT = "https://github.com/qiuqiangkong/audioset_tagging_cnn.git"
PANN_COMMIT = "d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4"  # observed on the student PC
EXPECTED_MD5 = "595633ac2d1cac7ef04ebf70e2fee4e4"


def root_relative(relative: str) -> Path:
    return ROOT / relative


def missing_files() -> list[str]:
    required = [
        "requirements.txt", "check_dependencies.py", "check_urbansound8k.py",
        "finalize_requirements.py", "src/check_cnn14.py", "src/check_folds.py",
        "src/__init__.py", "src/dataset.py", "src/audit_audio.py",
        "src/check_dataloader.py", "tests/test_dataset.py",
        "data/UrbanSound8K/metadata/UrbanSound8K.csv",
        "external/audioset_tagging_cnn/pytorch/models.py",
        "external/audioset_tagging_cnn/pytorch/pytorch_utils.py",
        "checkpoints/Cnn14_mAP=0.431.pth",
        "checkpoints/Cnn14_AudioSet_state_dict_safe.pth",
    ]
    return [p for p in required if not root_relative(p).is_file()]


def calc_md5(path: Path) -> str:
    h = hashlib.md5()  # Compare published integrity hash; NOT a security guarantee.
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def planned_steps(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    py = sys.executable
    steps: list[tuple[str, list[str]]] = []
    # Reinstalling dependencies is deliberately opt-in, never a side effect of --quick.
    if args.setup:
        steps.append(("Install dependencies (opt-in)", [py, "-m", "pip", "install", "-r", "requirements.txt"]))

    # A fresh *project* checkout does not normally include the separate upstream repo.
    # Auto-repair only when the entire clone directory is absent; never overwrite it.
    if not PANN_REPO.exists() and not args.offline:
        steps.append(("Clone missing official PANNs source", ["git", "clone", PANN_GIT, str(PANN_REPO)]))
        steps.append(("Pin official PANNs to previously verified commit",
                      ["git", "-C", str(PANN_REPO), "checkout", "--detach", PANN_COMMIT]))

    steps += [
        ("Verify installed package requirements", [py, "-m", "pip", "check"]),
        ("Verify imports and CUDA computation", [py, "check_dependencies.py"]),
        ("Verify 8,732 dataset metadata paths", [py, "check_urbansound8k.py"]),
        ("Record official PANNs Git commit", ["git", "-C", str(PANN_REPO), "rev-parse", "HEAD"]),
        ("Strict CNN14 loading and CUDA inference", [py, "src/check_cnn14.py", "--use-safe-weights"]),
    ]
    if not args.skip_cpu:
        steps.append(("Strict CNN14 loading and CPU inference", [py, "src/check_cnn14.py", "--cpu", "--use-safe-weights"]))
    steps += [
        ("Validate all 10 official train/validation/test splits", [py, "src/check_folds.py"]),
        ("Run Stage 3 waveform-processing unit tests", [py, "-m", "unittest", "discover", "-s", "tests", "-v"]),
        ("Pilot audio decode audit (50 files)", [py, "-m", "src.audit_audio", "--max-files", "50"]),
    ]
    if not args.quick:
        steps.append(("Full audio decode audit (all 8,732 files)", [py, "-m", "src.audit_audio"]))
    steps += [
        ("Validate real audio DataLoaders", [py, "-m", "src.check_dataloader"]),
        ("Run real UrbanSound8K audio through CNN14", [py, "-m", "src.check_dataloader", "--model-forward"]),
        ("Refresh locally observed dependency lock", [py, "finalize_requirements.py"]),
    ]
    return steps


def run_logged(name: str, cmd: list[str], log_handle) -> int:
    print(f"\n{'=' * 72}\nCHECK: {name}\nCOMMAND: {subprocess.list2cmdline(cmd)}\n{'=' * 72}", flush=True)
    log_handle.write(f"\n=== {name} ===\n$ {subprocess.list2cmdline(cmd)}\n")
    log_handle.flush()
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        with subprocess.Popen(cmd, cwd=ROOT, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              encoding="utf-8", errors="replace") as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
            code = process.wait()
    except (FileNotFoundError, PermissionError, OSError) as exc:
        message = f"Could not start command: {type(exc).__name__}: {exc}\n"
        print(message, end="", flush=True)
        log_handle.write(message)
        code = 127
    log_handle.write(f"EXIT CODE: {code}\n")
    log_handle.flush()
    print(f"RESULT: {'PASS' if code == 0 else 'FAIL'} ({name})", flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="One-command Stage 2 + Stage 3 verification")
    parser.add_argument("--quick", action="store_true", help="Run only the 50-file audio pilot; skip full 8,732-file audit")
    parser.add_argument("--skip-cpu", action="store_true", help="Skip repeat CPU CNN14 inference")
    parser.add_argument("--setup", action="store_true", help="Opt in to pip install -r requirements.txt (only use if needed)")
    parser.add_argument("--offline", action="store_true", help="Do not auto-clone official PANNs source when absent")
    parser.add_argument("--dry-run", action="store_true", help="Print checks without running or writing results")
    args = parser.parse_args()
    plan = planned_steps(args)
    if args.dry_run:
        print("Project root:", ROOT)
        print("Mode:", "quick" if args.quick else "FULL")
        print("Only if missing, the small official PANNs source will be cloned and pinned; use --offline to disable.")
        print("No pip install, checkpoint downloads, legacy pickle, dataset changes, or training unless explicitly requested.")
        for i, (name, command) in enumerate(plan, 1):
            print(f"{i:02d}. {name}\n    {subprocess.list2cmdline(command)}")
        print("Plus: check expected file paths and official checkpoint MD5.")
        return 0

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir = ROOT / "results" / "verification_runs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "console.log"
    summary_path = log_dir / "summary.json"
    summary: dict = {
        "run_id": run_id,
        "project_root": str(ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "mode": "quick_pilot_only" if args.quick else "full_audit",
        "steps": [],
        "source_isolation_warning": "5 source fsIDs cross official folds (reported in Stage 2); NOT resolved",
        "no_training_performed_by_this_runner": True,
    }
    overall_ok = True
    try:
        with log_path.open("w", encoding="utf-8") as logfile:
            print("Project root:", ROOT)
            print("Python executable:", sys.executable)
            print("Mode:", summary["mode"])
            print("Saving full console transcript:", log_path)

            # Repair missing upstream source automatically (unless --offline).
            # Never overwrite an existing, partially-populated source directory.
            setup_names = (
                "Install dependencies", "Clone missing official PANNs source",
                "Pin official PANNs to previously verified commit",
            )
            for name, cmd in plan:
                if name.startswith(setup_names):
                    if name.startswith("Clone missing official PANNs source"):
                        PANN_REPO.parent.mkdir(parents=True, exist_ok=True)
                    rc = run_logged(name, cmd, logfile)
                    summary["steps"].append({"name": name, "exit_code": rc, "status": "PASS" if rc == 0 else "FAIL"})
                    if rc != 0:
                        overall_ok = False
                        raise RuntimeError(f"Setup step failed: {name}")

            missing = missing_files()
            if missing:
                print("\nMISSING PREREQUISITES:")
                for file in missing:
                    print("  -", file)
                print("\nExtract the Stage 3 ZIP into the project root and keep your original scripts/checkpoints.")
                if args.offline and not PANN_REPO.is_dir():
                    print("Official source is absent. Omit --offline to auto-clone, or clone it manually.")
                elif PANN_REPO.exists() and not (PANN_REPO / "pytorch" / "models.py").is_file():
                    print("The official source directory exists but is incomplete or misplaced.")
                    print("Check it manually; this script will not overwrite or delete it.")
                if "checkpoints/Cnn14_AudioSet_state_dict_safe.pth" in missing:
                    print("The safe weights file was previously created with:")
                    print("  python src/check_cnn14.py --trust-legacy-checkpoint --export-safe-weights")
                    print("Only use that option with a trusted, MD5-matched official checkpoint.")
                logfile.write("Missing prerequisites: " + json.dumps(missing) + "\n")
                summary["missing_prerequisites"] = missing
                overall_ok = False
                raise RuntimeError("Prerequisites missing; no checks executed beyond setup")

            digest = calc_md5(ORIGINAL)
            checksum_ok = digest.lower() == EXPECTED_MD5
            print("\nOfficial CNN14 original checksum:", digest)
            print("Expected checksum:", EXPECTED_MD5)
            print("MD5 comparison:", "PASS" if checksum_ok else "FAIL")
            logfile.write(f"Original checkpoint MD5 {digest}: {'PASS' if checksum_ok else 'FAIL'}\n")
            summary["original_checkpoint_md5"] = digest
            summary["checkpoint_md5_match"] = checksum_ok
            summary["steps"].append({"name": "Original checkpoint MD5", "status": "PASS" if checksum_ok else "FAIL"})
            if not checksum_ok:
                overall_ok = False
                raise RuntimeError("Original checkpoint MD5 does not match published Zenodo checksum")

            for name, cmd in plan:
                if name.startswith(setup_names):
                    continue
                code = run_logged(name, cmd, logfile)
                summary["steps"].append({"name": name, "exit_code": code, "status": "PASS" if code == 0 else "FAIL"})
                if code != 0:
                    overall_ok = False
                    # Stop on a failure; later checks depend on a known-good environment.
                    print("\nStopped after failure. Fix the first failing check, then rerun the same command.")
                    break
    except Exception as exc:
        overall_ok = False
        summary["run_error"] = f"{type(exc).__name__}: {exc}"
        print("\nCHECK RUN INTERRUPTED:", summary["run_error"])
    finally:
        summary["status"] = "PASS_WITH_KNOWN_SOURCE_ISOLATION_WARNING" if overall_ok else "FAIL_OR_INCOMPLETE"
        summary["full_audio_audit_was_requested"] = not args.quick
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\n" + "=" * 72)
        print("Run status:", summary["status"])
        print("Console log:", log_path)
        print("JSON summary:", summary_path)
        print("Results and figures: results/ and figures/")
        print("NOTE: A passing run does NOT prove source-recording isolation or model accuracy.")
        print("NOTE: No model training or partial fine-tuning occurs in this script.")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
