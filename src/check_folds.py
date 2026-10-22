"""Stage 2: audit UrbanSound8K metadata and its fixed 8/1/1 ten-fold protocol.

Run from project root:
    python src/check_folds.py

Outputs:
    results/fold_split_summary.csv
    results/cross_fold_source_summary.csv
    results/cross_fold_source_details.csv

Never reallocates recordings or changes their prescribed folds.
It reports source-recording overlap as an AUDIT finding, not an automatic failure.
No audio is decoded and no model is trained.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "data" / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
DEFAULT_RESULTS = ROOT / "results"

CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music",
]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit UrbanSound8K official folds")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    metadata = args.metadata.resolve()
    if not metadata.is_file():
        raise FileNotFoundError(f"UrbanSound8K metadata not found: {metadata}")
    df = pd.read_csv(metadata)
    required = {"slice_file_name", "fsID", "fold", "classID", "class", "start", "end"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Required metadata columns missing: {sorted(missing)}")
    if len(df) != 8732:
        raise AssertionError(f"Expected 8,732 metadata rows, found {len(df)}")
    if df[list(required)].isna().any().any():
        raise AssertionError("Null values found in required metadata columns")

    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
    df["classID"] = pd.to_numeric(df["classID"], errors="raise").astype(int)
    assert set(df["fold"]) == set(range(1, 11)), "Fold IDs must be 1..10"
    assert set(df["classID"]) == set(range(10)), "Class IDs must be 0..9"
    for class_id, name in enumerate(CLASS_NAMES):
        actual = set(df.loc[df["classID"] == class_id, "class"])
        if actual != {name}:
            raise AssertionError(f"classID {class_id}: expected {name}, found {actual}")

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


if __name__ == "__main__":
    main()
