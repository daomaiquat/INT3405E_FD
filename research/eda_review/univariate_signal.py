"""Measure the individual signal of each raw S6E11 feature on the locked folds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = PROJECT / "s6e11_gpu" / "plan_runs" / "results" / "full"
SMOOTHING = 20
DISCRETE_LEVEL_LIMIT = 25


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verified_inputs():
    train_file = PROJECT.parent / 'data' / "train.csv"
    paths = PROJECT.parent
    config = json_read(paths / "configs" / "default.json")
    data_manifest = json_read(paths / "data" / "manifest.json")
    run_metrics = json_read(RESULTS / "XGB-D03" / "metrics.json")
    with train_file.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != data_manifest["input_files"]["train.csv"]["sha256"] or digest != run_metrics["inputs"]["train.csv"]:
        raise ValueError("train.csv does not match the official data and experiment manifests")

    train = pd.read_csv(train_file)
    labels = train["is_fraud"].to_numpy(dtype=np.int8)
    if len(train) != data_manifest["train_rows"] or train["id"].duplicated().any():
        raise ValueError("Unexpected train rows or duplicate IDs")
    if not np.array_equal(np.unique(labels), [0, 1]) or train.isna().any().any():
        raise ValueError("Expected two classes and no missing values in the official train snapshot")

    splitter = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=config["seed"])
    splits = list(splitter.split(train, labels))
    folds = np.full(len(train), -1, dtype=np.int8)
    for fold, (_, validation) in enumerate(splits):
        folds[validation] = fold
    with np.load(RESULTS / "XGB-D03" / "predictions.npz", allow_pickle=False) as stored:
        if not np.array_equal(stored["ids"], train["id"].to_numpy()):
            raise ValueError("Stored OOF IDs do not match the official train order")
        if not np.array_equal(stored["fold"], folds):
            raise ValueError("Rebuilt CV split does not match the locked experiment folds")

    groups = {
        "numeric": config["numeric_columns"],
        "categorical": config["categorical_columns"],
        "flag": config["flag_columns"],
    }
    selected = [feature for names in groups.values() for feature in names]
    if len(selected) != 24 or set(selected) != set(train.columns) - {"id", "is_fraud"}:
        raise ValueError("Expected the 24 raw features exactly once")
    return train, labels, splits, groups, run_metrics, digest


def fold_keys(values, kind, train_index, valid_index):
    train, valid = values[train_index], values[valid_index]
    if kind == "numeric" and len(np.unique(train)) > DISCRETE_LEVEL_LIMIT:
        edges = np.unique(np.quantile(train, np.arange(1, 10) / 10))
        train_bins = np.searchsorted(edges, train, side="left")
        valid_bins = np.searchsorted(edges, valid, side="left")
        return train_bins, valid_bins, "train_deciles"
    return train, valid, "raw_levels"


def estimate_rates(train_keys, train_y, valid_keys):
    levels = pd.DataFrame({"key": train_keys, "label": train_y}).groupby("key", sort=False)["label"]
    rates = (levels.sum() + SMOOTHING * train_y.mean()) / (levels.size() + SMOOTHING)
    return pd.Series(valid_keys).map(rates).fillna(train_y.mean()).to_numpy(dtype=np.float64), len(rates)


def measure_feature(train, y, splits, feature, kind):
    values = train[feature].to_numpy()
    if kind == "numeric" and not np.isfinite(values).all():
        raise ValueError(f"Non-finite numeric values in {feature}")
    oof = np.full(len(train), np.nan, dtype=np.float64)
    per_fold = []
    for fold, (train_index, valid_index) in enumerate(splits, 1):
        train_keys, valid_keys, method = fold_keys(values, kind, train_index, valid_index)
        scores, levels = estimate_rates(train_keys, y[train_index], valid_keys)
        oof[valid_index] = scores
        per_fold.append({
            "fold": fold,
            "encoding": method,
            "train_key_count": levels,
            "auc": float(roc_auc_score(y[valid_index], scores)),
            "average_precision": float(average_precision_score(y[valid_index], scores)),
        })
    if not np.isfinite(oof).all():
        raise ValueError(f"Incomplete predictions for {feature}")
    return {
        "feature": feature,
        "kind": kind,
        "oof_auc": float(roc_auc_score(y, oof)),
        "average_precision": float(average_precision_score(y, oof)),
        "folds": per_fold,
    }


def main():
    train, labels, splits, groups, run_metrics, digest = verified_inputs()
    rows = [measure_feature(train, labels, splits, column, kind)
            for kind, names in groups.items() for column in names]
    rows.sort(key=lambda row: (-row["oof_auc"], row["feature"]))
    for rank, row in enumerate(rows, 1):
        row["auc_rank"] = rank
    payload = {
        "source": "train.csv",
        "train_sha256": digest,
        "rows": len(train),
        "fraud_rows": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "fold_hash": run_metrics["fold_hash"],
        "folds": len(splits),
        "smoothing": SMOOTHING,
        "numeric_discrete_level_limit": DISCRETE_LEVEL_LIMIT,
        "method": "Single-feature rates fitted on each outer train; decile boundaries also fitted on outer train; unseen levels use its prevalence.",
        "features": rows,
    }
    target = HERE / "univariate_signal.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    for row in rows[:8]:
        print(f"{row['auc_rank']:2d}  {row['feature']:28s}  AUC {row['oof_auc']:.6f}  AP {row['average_precision']:.6f}")


if __name__ == "__main__":
    main()
