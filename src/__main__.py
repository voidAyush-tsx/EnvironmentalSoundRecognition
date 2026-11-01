"""Developer CLI. Help and saved-result review use only the standard library."""
import argparse
import importlib
import sys

from .config import PROJECT_ROOT, project_path

DELEGATES = {
    "train": ("src.finetune_train", "main", "Train with validation-only checkpoint selection"),
    "evaluate": ("src.evaluate_selected", "main", "Explicit selected-model held-out evaluation"),
    "audit-data": ("src.metadata", "paths_main", "Validate metadata and WAV paths without decoding"),
    "audit-audio": ("src.audit_audio", "main", "Decode WAVs; --max-files gives an isolated pilot"),
    "audit-folds": ("src.metadata", "folds_main", "Audit official folds and source overlap"),
    "check-model": ("src.diagnostics", "model_main", "Check CNN14 or explicitly convert trusted weights"),
    "check-loader": ("src.diagnostics", "dataloader_main", "Check real-audio batches and optional model forward"),
    "check-gradients": ("src.diagnostics", "gradients_main", "Check a training-only backward/update step"),
    "faculty": ("src.submission", "main", "Audit or explicitly package the historical submission"),
    "snapshot-env": ("src.environment", "snapshot_main", "Explicitly save direct dependency versions"),
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Lightweight CPU unit tests or environment imports")
    group = check.add_mutually_exclusive_group(required=True)
    group.add_argument("--unit", action="store_true", help="Offline synthetic-data/toy-model test suite")
    group.add_argument("--environment", action="store_true", help="Import installed packages; GPU optional")
    check.add_argument("--require-cuda", action="store_true", help="With --environment: require a working GPU")
    review = commands.add_parser("review", help="Read committed historical evidence without ML dependencies")
    review.add_argument("--evaluation-dir", type=project_path, help="Explicit saved evaluation with matching provenance")
    for name, (_, _, help_text) in DELEGATES.items():
        commands.add_parser(name, help=help_text, add_help=False)
    return result


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cli = parser()
    args, remaining = cli.parse_known_args(argv)
    try:
        if args.command in DELEGATES:
            # Command parsers are shared with lightweight help, without loading ML modules.
            if "--help" in remaining or "-h" in remaining:
                from .arguments import PARSERS
                PARSERS[args.command]().print_help()
                return 0
            module_name, function_name, _ = DELEGATES[args.command]
            return getattr(importlib.import_module(module_name), function_name)(remaining) or 0
        if remaining:
            cli.error("unrecognized arguments: " + " ".join(remaining))
        if args.command == "review":
            from .evidence import print_review
            print_review(evaluation_dir=args.evaluation_dir)
            return 0
        if args.require_cuda and not args.environment:
            cli.error("--require-cuda requires --environment")
        if args.environment:
            from .environment import environment_main as check_environment
            return check_environment(["--require-cuda"] if args.require_cuda else [])
        import subprocess
        return subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=PROJECT_ROOT)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
