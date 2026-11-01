"""Shared UrbanSound8K metadata validation and physical-file audit (no PyTorch)."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASS_NAMES, DATASET_ROOT, project_path


def validate_metadata(df: pd.DataFrame, *, extra_columns=()) -> pd.DataFrame:
    """Validate official labels/folds without changing their order or membership."""
    df = df.copy()
    required = {"slice_file_name", "fold", "classID", "class", "fsID", *extra_columns}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if len(df) != 8_732:
        raise ValueError(f"Expected 8732 metadata rows; found {len(df)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Metadata contains null values in required columns")
    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype("int64")
    df["classID"] = pd.to_numeric(df["classID"], errors="raise").astype("int64")
    if set(df["fold"]) != set(range(1, 11)):
        raise ValueError("Fold IDs must be 1-10")
    if set(df["classID"]) != set(range(10)):
        raise ValueError("Class IDs must be 0-9")
    for class_id, name in enumerate(CLASS_NAMES):
        values = set(df.loc[df["classID"] == class_id, "class"])
        if values != {name}:
            raise ValueError(f"classID {class_id}: expected {name}; found {values}")
    keys = df["fold"].astype(str) + "/" + df["slice_file_name"].astype(str)
    if not keys.is_unique:
        raise ValueError("Duplicate (fold, slice_file_name) metadata entries")
    return df.reset_index(drop=True)


def load_metadata(dataset_root: str | Path = DATASET_ROOT) -> pd.DataFrame:
    csv_path = project_path(dataset_root) / "metadata" / "UrbanSound8K.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {csv_path}. See docs/REPRODUCIBILITY.md")
    return validate_metadata(pd.read_csv(csv_path))


def audit_dataset_paths(dataset_root: str | Path = DATASET_ROOT) -> dict:
    """Check all metadata references once, without decoding any audio."""
    root = project_path(dataset_root)
    df = load_metadata(root)
    counts = {i: len(list((root / "audio" / f"fold{i}").glob("*.wav"))) for i in range(1, 11)}
    missing = [str(root / "audio" / f"fold{r.fold}" / r.slice_file_name)
               for r in df.itertuples(index=False)
               if not (root / "audio" / f"fold{r.fold}" / r.slice_file_name).is_file()]
    return {"metadata_rows": len(df), "fold_counts": counts, "wav_count": sum(counts.values()),
            "missing_files": missing, "passed": not missing and sum(counts.values()) == 8732}


def paths_main(argv=None) -> int:
    from .arguments import dataset_parser
    parser = dataset_parser()
    args = parser.parse_args(argv)
    try:
        report = audit_dataset_paths(args.dataset_root)
    except (OSError, ValueError) as exc:
        print(f"DATASET AUDIT: FAIL: {exc}")
        return 1
    print("Dataset:", args.dataset_root)
    print("Metadata rows:", report["metadata_rows"])
    for fold, count in report["fold_counts"].items():
        print(f"fold{fold}: {count}")
    print("Total WAV files:", report["wav_count"])
    print("Missing referenced files:", len(report["missing_files"]))
    for path in report["missing_files"][:10]:
        print("  ", path)
    print("DATASET AUDIT:", "PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


def audit_source_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """List Freesound IDs present in >1 fold and possible cross-fold time overlaps.

    Having the same fsID in separate folds is a potential dependency, regardless
    of whether the time ranges overlap. An overlapping time range is stronger
    evidence, but even that must be interpreted with the source metadata.
    """
    detail_columns = [
        "fsID", "fold", "slice_file_name", "classID", "class", "start", "end",
    ]
    if "fsID" not in df.columns:
        print("WARNING: fsID metadata is absent; source-recording audit skipped.")
        return pd.DataFrame(columns=["fsID", "folds", "clip_count", "cross_fold_time_overlap_pairs"]), pd.DataFrame(columns=detail_columns)

    src = df.copy()
    src["fsID"] = pd.to_numeric(src["fsID"], errors="raise").astype("int64")
    counts = src.groupby("fsID", sort=True)["fold"].nunique()
    ids = counts.index[counts > 1]
    details = src[src["fsID"].isin(ids)].copy()
    for optional in ("start", "end"):
        if optional not in details.columns:
            details[optional] = np.nan
    details = details[detail_columns].sort_values(
        ["fsID", "fold", "start", "slice_file_name"], kind="stable"
    )

    summaries = []
    for source_id in ids:
        group = details.loc[details["fsID"] == source_id]
        folds = sorted(group["fold"].unique().tolist())
        overlap_pairs = 0
        can_measure = group[["start", "end"]].notna().all().all()
        if can_measure:
            intervals = {}
            for fold_id, clips in group.groupby("fold"):
                start = clips["start"].to_numpy(dtype=float)
                end = clips["end"].to_numpy(dtype=float)
                if not (np.isfinite(start).all() and np.isfinite(end).all()
                        and (end >= start).all()):
                    can_measure = False
                    break
                intervals[fold_id] = (start, end)
            if can_measure:
                for i, fold_a in enumerate(folds):
                    for fold_b in folds[i + 1:]:
                        a_start, a_end = intervals[fold_a]
                        b_start, b_end = intervals[fold_b]
                        # Strictly positive temporal overlap, not just touching endpoints.
                        overlapping = (
                            (a_start[:, None] < b_end[None, :])
                            & (b_start[None, :] < a_end[:, None])
                        )
                        overlap_pairs += int(overlapping.sum())
        summaries.append({
            "fsID": int(source_id),
            "folds": ",".join(map(str, folds)),
            "clip_count": len(group),
            "class_ids": ",".join(map(str, sorted(group["classID"].unique()))),
            "cross_fold_time_overlap_pairs": overlap_pairs if can_measure else "Not checked",
        })
    return pd.DataFrame(summaries, columns=[
        "fsID", "folds", "clip_count", "class_ids", "cross_fold_time_overlap_pairs"
    ]), details


def folds_main(argv=None) -> None:
    from .arguments import folds_parser
    parser = folds_parser()
    args = parser.parse_args(argv)

    metadata = args.metadata.resolve()
    if not metadata.is_file():
        raise FileNotFoundError(f"UrbanSound8K metadata not found: {metadata}")
    df = validate_metadata(pd.read_csv(metadata), extra_columns=("start", "end"))

    # The two-component key represents one physical clip in the official layout.
    df["record_key"] = df["fold"].astype(str) + "/" + df["slice_file_name"]
    if not df["record_key"].is_unique:
        raise AssertionError("Duplicate (fold, filename) metadata entries")
    all_keys = set(df["record_key"])
    all_count = len(df)

    source_summary, source_details = audit_source_groups(df)
    print("Metadata, classID labels, and unique clip paths: PASS")
    if not source_summary.empty:
        print(f"WARNING: {len(source_summary)} Freesound source IDs appear in multiple official folds.")
        print("This is a potential source-related evaluation dependence; do NOT change folds.")
        print("Review the recorded source IDs, folds and time intervals:")
        print(source_summary.to_string(index=False))
    else:
        print("No cross-fold fsID values detected.")

    print("\nTest | Val | Train | Validation | Test | Total | Source-ID overlaps (T/V, T/E, V/E)")
    print("-" * 91)
    summaries = []
    for test_fold in range(1, 11):
        val_fold = test_fold % 10 + 1
        train_folds = [f for f in range(1, 11) if f not in (test_fold, val_fold)]
        train = df[df["fold"].isin(train_folds)]
        validation = df[df["fold"] == val_fold]
        test = df[df["fold"] == test_fold]
        keys = [set(part["record_key"]) for part in (train, validation, test)]
        assert len(train_folds) == 8
        assert keys[0].isdisjoint(keys[1])
        assert keys[0].isdisjoint(keys[2])
        assert keys[1].isdisjoint(keys[2])
        assert set.union(*keys) == all_keys
        assert sum(len(part) for part in (train, validation, test)) == all_count

        source_sets = [set(part["fsID"]) for part in (train, validation, test)]
        overlap_tv = len(source_sets[0] & source_sets[1])
        overlap_te = len(source_sets[0] & source_sets[2])
        overlap_ve = len(source_sets[1] & source_sets[2])

        summaries.append({
            "test_fold": test_fold,
            "validation_fold": val_fold,
            "train_folds": ",".join(map(str, train_folds)),
            "train_count": len(train),
            "validation_count": len(validation),
            "test_count": len(test),
            "total_count": all_count,
            "train_percent": round(100 * len(train) / all_count, 2),
            "validation_percent": round(100 * len(validation) / all_count, 2),
            "test_percent": round(100 * len(test) / all_count, 2),
            "shared_fsids_train_val": overlap_tv,
            "shared_fsids_train_test": overlap_te,
            "shared_fsids_val_test": overlap_ve,
        })
        print(f"{test_fold:>4} | {val_fold:>3} | {len(train):>5} | "
              f"{len(validation):>10} | {len(test):>4} | {all_count:>5} | "
              f"{overlap_tv}/{overlap_te}/{overlap_ve}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / "fold_split_summary.csv"
    source_summary_path = args.results_dir / "cross_fold_source_summary.csv"
    source_details_path = args.results_dir / "cross_fold_source_details.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    source_summary.to_csv(source_summary_path, index=False)
    source_details.to_csv(source_details_path, index=False)

    print("\nOfficial ten-fold, 8-train/1-val/1-test clip partition: PASS")
    print("Summary:", summary_path.resolve())
    print("Source-ID audit:", source_summary_path.resolve())
    print("Affected source clips:", source_details_path.resolve())
    if not source_summary.empty:
        print("SOURCE-ISOLATION WARNING remains open; investigate before model training.")
    print("No clips were moved, removed, or relabelled; no model was trained.")
