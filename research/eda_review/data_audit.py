"""Audit the official S6E11 CSVs and compare full train/test distributions."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parent
sys.path.insert(0, str(REPO))
from src.artifacts import read_json, sha256_file, write_json
from src.data import audit_data, validate_frames


def load_inputs():
    config = read_json(REPO / "configs" / "default.json")
    manifest = read_json(REPO / "data" / "manifest.json")
    for name, expected in manifest["input_files"].items():
        path = PROJECT.parent / "data" / name
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Unexpected official data bytes in {name}")
    train, test, sample = (pd.read_csv(PROJECT.parent / "data" / name) for name in ("train.csv", "test.csv", "sample_submission.csv"))
    validate_frames(train, test, sample, config)
    if len(train) != manifest["train_rows"] or len(test) != manifest["test_rows"]:
        raise ValueError("Row count does not match manifest")
    return train, test, sample, config, manifest


def key_support(train, test):
    counts = pd.Series(train).value_counts()
    test_support = pd.Series(test).map(counts).fillna(0).to_numpy(dtype=np.int64)
    return {
        "train_distinct": len(counts), "test_distinct": int(pd.Series(test).nunique()),
        "train_rows_singletons": int((counts == 1).sum()),
        "train_values_with_support_under_5": int((counts < 5).sum()),
        "test_distinct_unseen": int(pd.Series(test[test_support == 0]).nunique()),
        "test_rows_unseen": int((test_support == 0).sum()),
        "test_rows_seen_once": int((test_support == 1).sum()),
        "test_rows_with_train_support_under_5": int((test_support < 5).sum()),
        "test_train_support_median": float(np.median(test_support)),
    }


def numeric_audit(train, test, inferred_decimals):
    source = train.to_numpy(dtype=np.float64)
    query = test.to_numpy(dtype=np.float64)
    spread = np.sqrt((source.var() + query.var()) / 2)
    smd = (query.mean() - source.mean()) / spread if spread else 0.0
    keys = (source, query)
    precision = None
    if inferred_decimals is not None:
        scale = 10 ** inferred_decimals
        keys = tuple(np.rint(values * scale).astype(np.int64) for values in keys)
        grid_error = np.abs(query * scale - keys[1])
        precision = {
            "decimals": inferred_decimals,
            "test_cells_off_grid": int((grid_error > 1e-6).sum()),
            "max_test_distance_to_grid_in_raw_units": float(grid_error.max() / scale),
        }
        if inferred_decimals > 0:
            precision["integer_fraction_train"] = float((keys[0] % scale == 0).mean())
            precision["integer_fraction_test"] = float((keys[1] % scale == 0).mean())
            precision["most_common_fraction_train"] = [
                {"fraction_key": str(int(key)).zfill(inferred_decimals), "share": float(rate)}
                for key, rate in pd.Series(keys[0] % scale).value_counts(normalize=True).head(3).items()
            ]
    return {
        "ks": float(ks_2samp(source, query, method="asymp").statistic),
        "standardized_mean_difference_test_minus_train": float(smd),
        "train_mean": float(source.mean()), "test_mean": float(query.mean()),
        "train_min": float(source.min()), "train_median": float(np.median(source)),
        "train_max": float(source.max()), "test_min": float(query.min()), "test_max": float(query.max()),
        "train_zero_rows": int((source == 0).sum()), "test_zero_rows": int((query == 0).sum()),
        "key_support": key_support(*keys), "precision": precision,
    }


def categorical_audit(train, test):
    source = train.value_counts(normalize=True)
    query = test.value_counts(normalize=True)
    values = source.index.union(query.index, sort=False)
    source = source.reindex(values, fill_value=0)
    query = query.reindex(values, fill_value=0)
    return {
        "total_variation": 0.5 * float((source - query).abs().sum()),
        "key_support": key_support(train.to_numpy(), test.to_numpy()),
        "shares_by_value": {
            str(value): {"train": float(source[value]), "test": float(query[value])} for value in values
        },
    }


def main():
    train, test, sample, config, manifest = load_inputs()
    quality = audit_data(train, test, sample, config, manifest)
    features = quality["feature_groups"]
    counts = features["counts"]
    subset = features["numeric"] + features["categorical"] + features["flags"]
    duplicate_exact = {
        "train_rows_in_duplicate_groups": int(train.duplicated(subset=subset, keep=False).sum()),
        "test_rows_in_duplicate_groups": int(test.duplicated(subset=subset, keep=False).sum()),
        "train_groups_with_conflicting_labels": quality["duplicates"]["train_conflicting_label_groups"],
        "test_rows_identical_to_train": quality["duplicates"]["test_rows_identical_to_train"],
    }
    if duplicate_exact["test_rows_identical_to_train"]:
        matched = test[subset].merge(train[subset].drop_duplicates(), on=subset, how="inner")
        duplicate_exact["test_rows_identical_to_train"] = len(matched)

    numeric = {
        name: numeric_audit(train[name], test[name], quality["decimal_precision"][name]["inferred_decimals"])
        for name in features["numeric"]
    }
    categorical = {name: categorical_audit(train[name], test[name]) for name in features["categorical"]}
    flags = {name: categorical_audit(train[name], test[name]) for name in features["flags"]}
    audit = {
        "scope": {"train": "all rows and columns", "test": "all rows and columns", "sample_submission": "ID order and schema"},
        "distribution_methods": {
            "numeric_shift": "Two-sample KS statistic on all observed raw values",
            "categorical_shift": "Total variation on all observed categories, including flags",
            "numeric_support": "Project to the smallest decimal grid covering all train values. Report off-grid test values separately.",
        },
        "schema": {
            dataset: {key: value[key] for key in ("rows", "column_count", "missing_cells", "non_finite_cells")}
            for dataset, value in quality["schema"].items()
        },
        "feature_counts": counts, "test_has_label": config["target"] in test,
        "train_labels": quality["label"], "id_checks": quality["ids"], "duplicate_features": duplicate_exact,
        "numeric": numeric, "categorical": categorical, "flags": flags,
        "sha256": {name: values["sha256"] for name, values in manifest["input_files"].items()},
    }
    destination = HERE / "data_audit.json"
    staged = destination.with_suffix(".json.tmp")
    write_json(staged, audit)
    staged.replace(destination)
    print(f"Full-data audit saved to {destination}")


if __name__ == "__main__":
    main()
