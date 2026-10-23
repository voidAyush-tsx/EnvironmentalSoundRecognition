"""Run Stage 4 unit and real-CNN14 backward/update checks on Windows.

From the existing project root:
    python run_stage4_checks.py
Optional diagnostic training (only 4 batches per subset):
    python run_stage4_checks.py --pilots

No pip installation, repository cloning, pretrained model changes, test-fold
scoring, or full training. Stops on the first failed check. Keeps logs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description="One-command CNN14 fine-tuning sanity checks")
    ap.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    ap.add_argument("--amp", choices=("auto", "on", "off", "bf16"), default="auto",
                    help="auto uses BF16 on capable CUDA GPUs; otherwise FP32")
    ap.add_argument("--pilots", action="store_true", help="Include two one-epoch limited-batch training diagnostics")
    ap.add_argument("--max-candidates", type=int, default=12,
                    help="Train-only clips to check for nonzero pretrained-layer gradients per config (default 12)")
    ap.add_argument("--unit-only", action="store_true", help="Only run CPU unit tests, without model/backward checks")
    ap.add_argument("--skip-block6", action="store_true", help="Skip convolutional update check if GPU VRAM is insufficient")
    args = ap.parse_args()
    if args.max_candidates < 1:
        ap.error("--max-candidates must be >= 1")
    commands: list[tuple[str, list[str]]] = [
        ("Stage 3 and Stage 4 unit tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    ]
    if not args.unit_only:
        for config in ("head", "fc1", "block6"):
            if config == "block6" and args.skip_block6:
                continue
            commands.append((f"One backward/update with {config}",
                             [sys.executable, "-m", "src.finetune_check", "--config", config,
                              "--device", args.device, "--amp", args.amp,
                              "--max-candidates", str(args.max_candidates)]))
        if args.pilots:
            for config in ("head", "block6"):
                if config == "block6" and args.skip_block6:
                    continue
                commands.append((f"LIMITED pilot training: {config}",
                                 [sys.executable, "-m", "src.finetune_train", "--config", config,
                                  "--device", args.device, "--amp", args.amp, "--test-fold", "1", "--epochs", "1",
                                  "--batch-size", "1", "--accum-steps", "4",
                                  "--max-train-batches", "4", "--max-val-batches", "4"]))
    # Display preconditions rather than silently downloading or altering the environment.
    if not args.unit_only:
        expected = [
            ROOT / "data/UrbanSound8K/metadata/UrbanSound8K.csv",
            ROOT / "checkpoints/Cnn14_AudioSet_state_dict_safe.pth",
            ROOT / "external/audioset_tagging_cnn/pytorch/models.py",
            ROOT / "external/audioset_tagging_cnn/pytorch/pytorch_utils.py",
        ]
        missing = [str(p.relative_to(ROOT)) for p in expected if not p.is_file()]
        if missing:
            print("Missing Stage 2/3 prerequisites (nothing was installed or downloaded):")
            print("\n".join("  - " + p for p in missing))
            return 2
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    outdir = ROOT / "results" / "stage4" / "checks" / stamp
    outdir.mkdir(parents=True, exist_ok=False)
    log_path = outdir / "console.log"
    status_path = outdir / "summary.json"
    history = []
    rc = 0
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        def echo(s):
            print(s, flush=True)
            log.write(s + "\n")
        echo("Stage 4 verification; device=" + args.device + "; precision request=" + args.amp + "; pilots=" + str(args.pilots))
        for i, (name, argv) in enumerate(commands, 1):
            echo("\n" + "=" * 70)
            echo(f"{i:02d}. {name}")
            echo("COMMAND: " + subprocess.list2cmdline(argv))
            echo("=" * 70)
            try:
                process = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, errors="replace",
                                           bufsize=1)
                assert process.stdout is not None
                for line in process.stdout:
                    echo(line.rstrip("\r\n"))
                rc = process.wait()
            except OSError as exc:
                echo(f"Could not run command: {exc}")
                rc = 1
            echo(f"RESULT: {'PASS' if rc == 0 else 'FAIL'}: {name}")
            history.append({"name": name, "command": argv, "exit_code": rc})
            if rc != 0:
                break
    summary = {
        "status": "PASS" if rc == 0 else "FAIL_INCOMPLETE",
        "device": args.device,
        "amp_requested": args.amp,
        "pilot_diagnostic_only": args.pilots,
        "conv6_gradient_check_executed": any("block6" in x["name"] and "backward" in x["name"]
                                             for x in history),
        "steps": history,
        "logs": str(log_path),
        "note": "This verifies implementation, not trained test accuracy; test fold never evaluated",
    }
    status_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nStage 4 check status:", summary["status"])
    print("Full log:", log_path)
    print("Summary:", status_path)
    print("No test-fold evaluation or full fine-tuning was run by this checker.")
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    sys.exit(main())
