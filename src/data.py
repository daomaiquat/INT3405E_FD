"""Single data source and schema/ID checks for the S6E11 competition."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from .artifacts import read_json, sha256_file


FILENAMES = ("train.csv", "test.csv", "sample_submission.csv")


def verify_files(data_dir: Path, verify_checksums: bool = True) -> dict:
    manifest = read_json(data_dir / "manifest.json")
    checked = {}
    for name in FILENAMES:
        path = data_dir / name
        expected = manifest["input_files"][name]
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"{name}: file size differs from canonical data manifest")
        if verify_checksums and sha256_file(path) != expected["sha256"]:
            raise ValueError(f"{name}: SHA256 differs from canonical data manifest")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle))
        if len(columns) != len(set(columns)):
            raise ValueError(f"{name}: duplicate column names")
        checked[name] = {**expected, "checksum_verified": verify_checksums}
    return {**manifest, "input_files": checked}


def validate_frames(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, config: dict) -> dict:
    target = config["target"]
    features = config["numeric_columns"] + config["categorical_columns"] + config["flag_columns"]
    if len(features) != len(set(features)):
        raise ValueError("Feature groups overlap")
    if set(train.columns) != set(features + ["id", target]) or set(test.columns) != set(features + ["id"]):
        raise ValueError("Train/test columns differ from configured schema")
    if list(train.drop(columns=target).columns) != list(test.columns):
        raise ValueError("Train and test feature order differs")
    if list(sample.columns) != ["id", target]:
        raise ValueError("Sample submission needs exactly id and target")
    for name, frame in (("train", train), ("test", test), ("sample", sample)):
        if frame.isna().any().any():
            raise ValueError(f"{name}: missing values")
        if not is_integer_dtype(frame["id"].dtype) or frame["id"].duplicated().any():
            raise ValueError(f"{name}: id must contain unique integers")
        numeric = frame.select_dtypes(include="number")
        if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{name}: non-finite numbers")
    if set(train[target].unique()) != {0, 1}:
        raise ValueError("Target must contain both binary 0/1 classes")
    if train["id"].isin(test["id"]).any():
        raise ValueError("Train and test IDs overlap")
    if not test["id"].equals(sample["id"]):
        raise ValueError("Sample IDs/order differ from test")
    for name, frame in (("train", train), ("test", test)):
        if any(not is_numeric_dtype(frame[col]) for col in config["numeric_columns"]):
            raise ValueError(f"{name}: numeric feature has nonnumeric dtype")
        if frame[config["numeric_columns"]].lt(0).any().any():
            raise ValueError(f"{name}: negative numeric feature")
        for col in config["flag_columns"]:
            if not frame[col].isin([False, True, 0, 1]).all():
                raise ValueError(f"{name}: unexpected values in {col}")
    return {
        "train_rows": len(train), "test_rows": len(test), "features": len(features),
        "positive_rows": int(train[target].sum()), "positive_rate": float(train[target].mean()),
        "train_missing": 0, "test_missing": 0,
        "numeric_unique_values": {col: int(train[col].nunique()) for col in config["numeric_columns"]},
        "test_unseen_categories": {
            col: sorted(map(str, set(test[col]) - set(train[col]))) for col in config["categorical_columns"]
        },
    }


DECIMAL_TOLERANCE = 1e-6


def cast_target(train: pd.DataFrame, target: str) -> pd.DataFrame:
    values = train[target].to_numpy()
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError(f"{target}: label is not strictly binary 0/1")
    cast = train.copy()
    cast[target] = values.astype(np.int8)
    return cast


def _feature_hashes(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(frame[features], index=False)


def audit_data(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame,
               config: dict, manifest: dict) -> dict:
    """Full raw-data quality report; never mutates or drops rows."""
    target = config["target"]
    numeric = config["numeric_columns"]
    categorical = config["categorical_columns"]
    flags = config["flag_columns"]
    features = numeric + categorical + flags
    frames = {"train": train, "test": test, "sample": sample}

    schema = {}
    for name, frame in frames.items():
        missing = frame.isna().sum()
        finite = {}
        for col in numeric:
            if col in frame.columns:
                values = frame[col].to_numpy(dtype=np.float64)
                finite[col] = int((~np.isfinite(values)).sum())
        schema[name] = {
            "rows": len(frame),
            "columns": list(frame.columns),
            "column_count": len(frame.columns),
            "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
            "missing_cells": int(missing.sum()),
            "missing_by_column": {col: int(value) for col, value in missing.items() if value},
            "non_finite_by_column": {col: value for col, value in finite.items() if value},
            "non_finite_cells": int(sum(finite.values())),
        }

    label_values = np.sort(train[target].unique())
    label_counts = train[target].value_counts().sort_index()
    label = {
        "raw_dtype": str(train[target].dtype),
        "values": [float(v) for v in label_values],
        "class_counts": {str(k): int(v) for k, v in label_counts.items()},
        "positive_rows": int(train[target].sum()),
        "positive_rate": float(train[target].mean()),
        "both_classes_present": set(label_values.tolist()) == {0, 1},
        "cast_applied_by_runner": "int8",
    }

    id_checks = {
        "dtype": str(train["id"].dtype),
        "unique": {name: bool(frame["id"].is_unique) for name, frame in frames.items()},
        "train_test_overlap": int(train["id"].isin(test["id"]).sum()),
        "sample_matches_test_order": bool(test["id"].equals(sample["id"])),
    }

    feature_groups = {
        "numeric": list(numeric), "categorical": list(categorical), "flags": list(flags),
        "counts": {"numeric": len(numeric), "categorical": len(categorical), "flags": len(flags),
                   "total": len(features)},
        "id_excluded": "id" not in features,
        "target_excluded": target not in features,
        "nonoverlapping": len(features) == len(set(features)),
    }

    numeric_stats = {}
    for col in numeric:
        values = train[col].to_numpy(dtype=np.float64)
        low, high = config["numeric_expected_ranges"][col]
        numeric_stats[col] = {
            "dtype": str(train[col].dtype), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()),
            "distinct": int(train[col].nunique()), "negative_rows": int((values < 0).sum()),
            "zero_rows": int((values == 0).sum()), "zero_rows_test": int((test[col] == 0).sum()),
            "expected_range": [low, high],
            "out_of_range_rows_train": int(((values < low) | (values > high)).sum()),
            "out_of_range_rows_test": int(((test[col] < low) | (test[col] > high)).sum()),
        }

    categorical_stats = {}
    for col in categorical:
        train_categories = set(train[col])
        categorical_stats[col] = {
            "train_cardinality": int(train[col].nunique()),
            "test_cardinality": int(test[col].nunique()),
            "unseen_in_test": sorted(map(str, set(test[col]) - train_categories)),
            "unseen_rows_in_test": int((~test[col].isin(train_categories)).sum()),
        }

    train_hashes = _feature_hashes(train, features)
    counts = train_hashes.value_counts()
    duplicate_groups = counts[counts > 1]
    conflicting_groups = 0
    if len(duplicate_groups):
        by_hash = train.assign(_hash=train_hashes).groupby("_hash")[target].nunique()
        conflicting_groups = int((by_hash.loc[duplicate_groups.index] > 1).sum())
    test_hashes = _feature_hashes(test, features)
    cross_frame = int(test_hashes.isin(set(train_hashes)).sum())
    duplicates = {
        "key": "all 24 features, excluding id and label",
        "train_duplicate_rows": int(duplicate_groups.sum()),
        "train_duplicate_groups": int(len(duplicate_groups)),
        "train_conflicting_label_groups": conflicting_groups,
        "test_rows_identical_to_train": cross_frame,
        "deduplication_action": "none; reported only",
    }

    precision = {}
    for col in numeric:
        values = train[col].to_numpy(dtype=np.float64)
        errors = {}
        for decimals in range(5):
            errors[decimals] = float(np.abs(values * (10 ** decimals)
                                            - np.round(values * (10 ** decimals))).max())
        inferred = next((d for d in range(5) if errors[d] <= DECIMAL_TOLERANCE), None)
        cent_error = np.abs(values * 100 - np.round(values * 100))
        precision[col] = {
            "max_error_at_cent_scale": float(cent_error.max()),
            "rows_not_exact_at_cent_scale": int((cent_error > DECIMAL_TOLERANCE).sum()),
            "inferred_decimals": inferred,
            "cent_key_safe": inferred is not None and inferred <= 2,
        }

    return {
        "schema": schema, "label": label, "ids": id_checks,
        "feature_groups": feature_groups, "numeric": numeric_stats,
        "categorical": categorical_stats, "duplicates": duplicates,
        "decimal_precision": precision,
        "manifest": {"train_rows": manifest["train_rows"], "test_rows": manifest["test_rows"],
                     "input_files": manifest["input_files"]},
    }


def quality_table(report: dict) -> pd.DataFrame:
    rows = []

    def add(check: str, scope: str, value, ok: bool):
        rows.append({"check": check, "scope": scope, "value": str(value), "ok": bool(ok)})

    schema = report["schema"]
    manifest = report["manifest"]
    for name, expected in (("train", manifest["train_rows"]), ("test", manifest["test_rows"]),
                           ("sample", manifest["test_rows"])):
        add("row_count_matches_manifest", name, schema[name]["rows"], schema[name]["rows"] == expected)
        add("missing_cells", name, schema[name]["missing_cells"], schema[name]["missing_cells"] == 0)
        add("non_finite_cells", name, schema[name]["non_finite_cells"], schema[name]["non_finite_cells"] == 0)
        add("id_unique", name, schema[name]["rows"] if report["ids"]["unique"][name] else "duplicates",
            report["ids"]["unique"][name])

    label = report["label"]
    add("label_dtype", "train", label["raw_dtype"], True)
    add("label_classes", "train", label["values"], label["both_classes_present"])
    add("label_binary_cast", "train", label["cast_applied_by_runner"], label["both_classes_present"])
    add("train_test_id_overlap", "train/test", report["ids"]["train_test_overlap"],
        report["ids"]["train_test_overlap"] == 0)
    add("sample_id_order", "sample/test", report["ids"]["sample_matches_test_order"],
        report["ids"]["sample_matches_test_order"])

    groups = report["feature_groups"]
    add("feature_group_counts", "config", groups["counts"],
        groups["counts"] == {"numeric": 13, "categorical": 5, "flags": 6, "total": 24})
    add("id_and_target_excluded", "config", f"id={groups['id_excluded']}, target={groups['target_excluded']}",
        groups["id_excluded"] and groups["target_excluded"] and groups["nonoverlapping"])

    bad_negatives = sum(v["negative_rows"] for v in report["numeric"].values())
    bad_ranges = sum(v["out_of_range_rows_train"] + v["out_of_range_rows_test"]
                     for v in report["numeric"].values())
    add("negative_numeric_rows", "train", bad_negatives, bad_negatives == 0)
    add("out_of_expected_range_rows", "train+test", bad_ranges, bad_ranges == 0)
    for col, stats in report["categorical"].items():
        add("unseen_test_categories", col, len(stats["unseen_in_test"]), not stats["unseen_in_test"])

    dup = report["duplicates"]
    add("duplicate_feature_rows", "train", dup["train_duplicate_rows"], dup["train_duplicate_rows"] == 0)
    add("conflicting_label_groups", "train", dup["train_conflicting_label_groups"],
        dup["train_conflicting_label_groups"] == 0)
    add("identical_feature_rows", "test-vs-train", dup["test_rows_identical_to_train"],
        dup["test_rows_identical_to_train"] == 0)

    unsafe = [col for col, p in report["decimal_precision"].items() if not p["cent_key_safe"]]
    add("cent_key_safe_columns", "train", len(report["decimal_precision"]) - len(unsafe), not unsafe)
    for name, item in manifest["input_files"].items():
        add("checksum_verified", name, item["checksum_verified"], item["checksum_verified"])
    return pd.DataFrame(rows)


def load_data(data_dir: Path, config: dict):
    manifest = verify_files(data_dir, config.get("verify_checksums", True))
    train, test, sample = (pd.read_csv(data_dir / name) for name in FILENAMES)
    summary = validate_frames(train, test, sample, config)
    if len(train) != manifest["train_rows"] or len(test) != manifest["test_rows"]:
        raise ValueError("Row counts differ from canonical manifest")
    return train, test, sample, manifest, summary
