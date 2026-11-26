"""LightGBM with fold-local, label-free occupancy of exact numeric pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
BASE_GROUPS = ("raw", "te13", "cents", "digits")
COLUMNS = {
    "amount_usd": (100, 500),
    "hours_since_last_txn": (100, 50),
    "distance_from_home_km": (100, 100),
    "velocity_score": (10, 30),
    "merchant_risk_score": (10, 30),
    "customer_age": (1, 5),
    "card_age_months": (1, 6),
}
PAIRS = (
    ("velocity_score", "merchant_risk_score"),
    ("customer_age", "merchant_risk_score"),
    ("customer_age", "velocity_score"),
    ("amount_usd", "velocity_score"),
    ("distance_from_home_km", "velocity_score"),
    ("hours_since_last_txn", "velocity_score"),
    ("customer_age", "card_age_months"),
)


def pair_occupancy(left, right, left_width, right_width, fitting, queries):
    """Return log occupancy and its excess over independence in each coarse cell."""
    left_bin = left // left_width
    right_bin = right // right_width
    stride = int(right.max()) + 1
    keys = (left * stride + right, left * stride + right_bin,
            left_bin * stride + right, left_bin * stride + right_bin)
    tables = tuple(pd.Series(key[fitting]).value_counts(sort=False) for key in keys)

    matrices = []
    for leave_one_out, rows in zip((True, *((False,) * len(queries))), (fitting, *queries), strict=True):
        exact, left_count, right_count, coarse = (
            table.reindex(key[rows], fill_value=0).to_numpy(dtype=np.float64) - leave_one_out
            for key, table in zip(keys, tables, strict=True)
        )
        expected = left_count * right_count / np.maximum(coarse, 1)
        log_count = np.log1p(exact)
        matrices.append(np.column_stack((log_count, log_count - np.log1p(expected))).astype(np.float32))
    return tuple(matrices)


def _metric(labels, scores):
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "log_loss": float(log_loss(labels, scores)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--control", action="store_true", help="Run a matched base-feature model on fold 0.")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT.parent / "s6e11_gpu" / "plan_runs"))
    from features import FeatureBuilder

    started = perf_counter()
    train = pd.read_csv(ROOT.parent / "data" / "train.csv")
    test = pd.read_csv(ROOT.parent / "data" / "test.csv")
    sample = pd.read_csv(ROOT.parent / "data" / "sample_submission.csv")
    ids = train["id"].to_numpy()
    test_ids = test["id"].to_numpy()
    labels = train["is_fraud"].to_numpy(dtype=np.uint8)
    test_rows = np.arange(len(train), len(train) + len(test))
    if not np.array_equal(test_ids, sample["id"].to_numpy()):
        raise ValueError("Test and sample submission IDs differ in identity or order.")

    builder = FeatureBuilder(train, test)
    ticks = {
        name: np.rint(np.concatenate((train[name].to_numpy(), test[name].to_numpy())) * units).astype(np.int64)
        for name, (units, _) in COLUMNS.items()
    }
    del train, test
    if any(int(values.min()) < 0 for values in ticks.values()):
        raise ValueError("Numeric pair keys require nonnegative values.")

    settings = dict(
        objective="binary", metric="auc", learning_rate=0.07, num_leaves=4,
        min_data_in_leaf=75, max_bin=255, feature_fraction=0.9, bagging_fraction=0.9,
        bagging_freq=1, lambda_l2=1.0, seed=42, num_threads=args.threads,
        force_col_wise=True, deterministic=True, verbosity=-1,
    )
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(len(labels), np.nan)
    fold_ids = np.full(len(labels), -1, dtype=np.int16)
    test_scores = np.zeros(len(test_rows))
    fold_metrics = []
    control = None
    importance = np.zeros(2 * len(PAIRS), dtype=np.float64)
    for fold, (fitting, held) in enumerate(splitter.split(ids, labels)):
        fold_started = perf_counter()
        base = list(builder.make(fitting, (held, test_rows), groups=BASE_GROUPS))
        base_width = base[0].shape[1]
        if fold == 0 and args.control:
            control_model = lgb.train(settings, lgb.Dataset(base[0], label=labels[fitting]), args.rounds)
            control = _metric(labels[held], control_model.predict(base[1], num_threads=args.threads))
            print(f"fold 0 matched control {control}", flush=True)
            del control_model

        extended = [np.empty((len(rows), base_width + 2 * len(PAIRS)), dtype=np.float32)
                    for rows in (fitting, held, test_rows)]
        for matrix in extended:
            matrix[:, :base_width] = base.pop(0)
        for index, (left, right) in enumerate(PAIRS):
            features = pair_occupancy(ticks[left], ticks[right], COLUMNS[left][1], COLUMNS[right][1],
                                      fitting, (held, test_rows))
            start = base_width + 2 * index
            for matrix, values in zip(extended, features, strict=True):
                matrix[:, start:start + 2] = values

        print(f"fold {fold} features ready at {perf_counter() - fold_started:.1f}s", flush=True)
        model = lgb.train(settings, lgb.Dataset(extended[0], label=labels[fitting]), args.rounds)
        oof[held] = model.predict(extended[1], num_threads=args.threads)
        fold_ids[held] = fold
        test_scores += model.predict(extended[2], num_threads=args.threads) / splitter.n_splits
        gains = model.feature_importance(importance_type="gain")
        importance += gains[base_width:] / max(gains.sum(), 1) / splitter.n_splits
        metrics = {"fold": fold, **_metric(labels[held], oof[held]),
                   "seconds": perf_counter() - fold_started}
        fold_metrics.append(metrics)
        print(metrics, flush=True)
        del extended, model

    for scores in (oof, test_scores):
        if not (np.isfinite(scores) & (0 <= scores) & (scores <= 1)).all():
            raise ValueError("Predictions must be finite probabilities in [0, 1].")
    if (fold_ids < 0).any() or np.unique(ids).size != len(ids) or np.unique(test_ids).size != len(test_ids):
        raise ValueError("Prediction IDs or fold coverage are invalid.")
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "occupancy_predictions.npz", oof=oof, test=test_scores,
                        ids=ids, test_ids=test_ids, fold=fold_ids)
    sample.assign(is_fraud=test_scores).to_csv(args.output / "submission_occupancy.csv", index=False)
    result = {
        **_metric(labels, oof), "fold_metrics": fold_metrics, "fold_0_matched_control": control,
        "settings": settings, "rounds": args.rounds, "fold_seed": 42, "feature_groups": BASE_GROUPS,
        "pair_widths_in_ticks": {name: width for name, (_, width) in COLUMNS.items()},
        "pair_feature_gain_share": {
            f"{left}__{right}": {"log_count": float(importance[2 * index]),
                                  "log_excess": float(importance[2 * index + 1])}
            for index, (left, right) in enumerate(PAIRS)
        },
        "elapsed_seconds": perf_counter() - started,
    }
    (args.output / "occupancy_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
