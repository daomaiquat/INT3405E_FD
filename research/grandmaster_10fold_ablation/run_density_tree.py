"""Train and compare a standalone class-conditional density XGBoost model."""

import argparse
import json
from hashlib import file_digest, sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "density_tree_validation"
COMPARATOR = ROOT / "validation"
BRANCHES = ("control_xgb", "density_xgb")
INNER_FOLDS = 4
XGB_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "n_estimators": 1100,
    "learning_rate": 0.035,
    "max_depth": 3,
    "max_bin": 256,
    "min_child_weight": 40,
    "reg_alpha": 2.0,
    "reg_lambda": 10.0,
    "subsample": 0.9,
    "colsample_bytree": 1.0,
    "n_jobs": 4,
}


def sha_file(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def cross_fit_inputs(frame, labels, scoring, factories, *, inner_folds, seed):
    """Build one block per factory, using inner OOF statistics on fitted rows."""
    labels = np.asarray(labels)
    if labels.shape != (len(frame),) or not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Expected a binary label for each fitting transaction.")
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(labels, labels))
    matrices = []
    for factory in factories:
        model = factory().fit(frame, labels)
        query = np.asarray(model.transform(scoring), dtype=np.float32, order="C")
        del model
        if query.ndim != 2 or len(query) != len(scoring) or not np.isfinite(query).all():
            raise ValueError("Encoder returned invalid scoring features.")
        training = np.full((len(frame), query.shape[1]), np.nan, dtype=np.float32)
        for fit, held in splits:
            inner = factory().fit(frame.iloc[fit], labels[fit])
            values = np.asarray(inner.transform(frame.iloc[held]))
            if values.shape != (len(held), query.shape[1]):
                raise ValueError("The feature shape changed across inner folds.")
            training[held] = values
        if not np.isfinite(training).all():
            raise ValueError("Inner out-of-fold features are incomplete or invalid.")
        matrices.append((training, query))
    return tuple(matrices)


def pair_influence(labels, scores):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    sorted_p, sorted_n = np.sort(positive), np.sort(negative)
    p_win = (np.searchsorted(sorted_n, positive, "left") + np.searchsorted(sorted_n, positive, "right")) / (2 * len(negative))
    n_win = 1 - (np.searchsorted(sorted_p, negative, "left") + np.searchsorted(sorted_p, negative, "right")) / (2 * len(positive))
    return p_win, n_win


def paired_auc_delta(labels, candidate, baseline):
    labels = np.asarray(labels)
    candidate, baseline = np.asarray(candidate), np.asarray(baseline)
    if (labels.shape != candidate.shape or labels.shape != baseline.shape or labels.ndim != 1
            or not np.isfinite(candidate).all() or not np.isfinite(baseline).all()
            or not np.array_equal(np.unique(labels), [0, 1]) or np.bincount(labels.astype(int)).min() < 2):
        raise ValueError("Paired AUC comparison needs aligned scores and two rows of each label.")
    left, right = pair_influence(labels, candidate), pair_influence(labels, baseline)
    d_positive, d_negative = (a - b for a, b in zip(left, right, strict=True))
    diff = d_positive.mean()
    var = np.var(d_positive, ddof=1) / len(d_positive) + np.var(d_negative, ddof=1) / len(d_negative)
    half_width = 1.959963984540054 * np.sqrt(var)
    return {"delta_roc_auc": float(diff), "ci95_low": float(diff - half_width), "ci95_high": float(diff + half_width)}


def fit_fold(frame, labels, scoring, factories, fold, device):
    from xgboost import XGBClassifier

    start = perf_counter()
    (common, common_query), (density, density_query) = cross_fit_inputs(
        frame, labels, scoring, factories, inner_folds=INNER_FOLDS, seed=7041 + fold,
    )
    common_width, density_width = common.shape[1], density.shape[1]
    print(f"  Encoded {common_width} common and {density_width} distribution features in {perf_counter() - start:.1f}s", flush=True)
    logits = []
    for name in BRANCHES:
        start = perf_counter()
        if name == "density_xgb":
            common = np.column_stack((common, density))
            common_query = np.column_stack((common_query, density_query))
            del density, density_query
        model = XGBClassifier(**XGB_PARAMS, device=device, random_state=2026 + fold)
        model.fit(common, labels, verbose=False)
        booster = model.get_booster()
        booster.set_param({"device": "cpu"})
        logits.append(booster.inplace_predict(common_query, predict_type="margin"))
        print(f"  {name} trained and scored in {perf_counter() - start:.1f}s", flush=True)
    split_counts = booster.get_score(importance_type="weight")
    density_counts = np.array([split_counts.get(f"f{idx}", 0) for idx in range(common_width, common_width + density_width)], dtype=np.int32)
    return np.column_stack(logits), density_counts


def metrics(labels, probabilities):
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities)),
    }


def dump_csv(path, frame):
    temp = path.with_suffix(".tmp.csv")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def main():
    from sklearn import __version__ as sklearn_version
    from xgboost import __version__ as xgb_version
    from class_conditional_density import ClassConditionalDensity
    from density_tree_inputs import DensityTreeInputs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    started = perf_counter()
    meta = json.loads((COMPARATOR / "metrics.json").read_text(encoding="utf-8"))
    train_path = ROOT.parents[1] / 'data' / "train.csv"
    train_hash = sha_file(train_path)
    if train_hash != meta["train_sha256"] or meta["cv_folds"] != 10:
        raise ValueError("Expected the official training data and verified 10-fold comparator.")

    columns = {column: "category" for column in ("merchant_category", "card_type", "auth_method", "channel", "device_type")}
    data = pd.read_csv(train_path, dtype=columns)
    ids = data.pop("id").to_numpy(dtype=np.int64)
    labels = data.pop("is_fraud").to_numpy(dtype=np.int8)
    if len(data) != 699_960 or len(np.unique(ids)) != len(ids) or not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Unexpected training shape, IDs or labels.")
    holdout_split = StratifiedShuffleSplit(n_splits=1, test_size=meta["holdout_fraction"], random_state=meta["holdout_seed"])
    dev_rows, holdout_rows = (np.sort(idx) for idx in next(holdout_split.split(ids, labels)))
    dev_ids, y_dev = ids[dev_rows], labels[dev_rows]
    holdout_ids, y_holdout = ids[holdout_rows], labels[holdout_rows]
    control_path = COMPARATOR / "holdout_predictions.csv"
    comparator = pd.read_csv(control_path, usecols=["id", "is_fraud", "hierarchical_amount"])
    saved = comparator["hierarchical_amount"].to_numpy()
    if (sha256(holdout_ids.tobytes()).hexdigest() != meta["holdout_ids_sha256"]
            or not np.array_equal(comparator["id"], holdout_ids)
            or not np.array_equal(comparator["is_fraud"], y_holdout)):
        raise ValueError("Comparator must score the exact verified holdout rows.")
    np.testing.assert_allclose(roc_auc_score(y_holdout, saved), meta["variants"]["hierarchical_amount"]["roc_auc"], rtol=0, atol=1e-12)
    development = data.iloc[dev_rows].reset_index(drop=True)
    holdout = data.iloc[holdout_rows].reset_index(drop=True)
    del data
    print(f"Development {len(development):,} rows | historical holdout {len(holdout):,} rows", flush=True)

    identity = {
        "train_sha256": train_hash,
        "holdout_ids_sha256": meta["holdout_ids_sha256"],
        "grandmaster_comparator_sha256": sha_file(control_path),
        "outer_folds": meta["cv_folds"],
        "outer_seed": meta["cv_seed"],
        "inner_folds": INNER_FOLDS,
        "device": args.device,
        "xgb_params": XGB_PARAMS,
        "versions": {"xgboost": xgb_version, "scikit-learn": sklearn_version, "numpy": np.__version__, "pandas": pd.__version__},
        "implementation_sha256": {
            name: sha_file(ROOT / name) for name in (Path(__file__).name, "class_conditional_density.py", "density_tree_inputs.py")
        },
    }
    signature = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    artifact_dir = RESULTS / "artifacts" / signature[:12]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=meta["cv_folds"], shuffle=True, random_state=meta["cv_seed"])
    oof = np.full((len(development), len(BRANCHES)), np.nan)
    holdout_sum = np.zeros((len(holdout), len(BRANCHES)), dtype=np.float64)
    fold_metrics = []
    density_feature_splits = np.zeros(len(ClassConditionalDensity.feature_names), dtype=np.int64)
    for fold, (fit, held) in enumerate(cv.split(y_dev, y_dev)):
        path = artifact_dir / f"fold_{fold + 1:02d}.npz"
        query_count = len(held) + len(holdout)
        if path.exists():
            with np.load(path, allow_pickle=False) as record:
                if record["signature"].item() != signature or not np.array_equal(record["held"], held):
                    raise ValueError(f"Incorrect fold checkpoint: {path}.")
                logits, density_counts = record["logits"], record["density_counts"]
            print(f"Fold {fold + 1}/{cv.n_splits}: restored", flush=True)
        else:
            print(f"Fold {fold + 1}/{cv.n_splits}: {len(fit):,} training rows", flush=True)
            query = pd.concat((development.iloc[held], holdout), ignore_index=True)
            logits, density_counts = fit_fold(
                development.iloc[fit], y_dev[fit], query, (DensityTreeInputs, ClassConditionalDensity), fold, args.device,
            )
            if logits.shape == (query_count, len(BRANCHES)) and np.isfinite(logits).all():
                temp = path.with_suffix(".tmp.npz")
                np.savez(temp, signature=signature, held=held, logits=logits, density_counts=density_counts)
                temp.replace(path)
        if logits.shape != (query_count, len(BRANCHES)) or not np.isfinite(logits).all():
            raise ValueError(f"Invalid logits for fold {fold + 1}.")
        oof[held] = logits[:len(held)]
        holdout_sum += logits[len(held):]
        density_feature_splits += density_counts
        aucs = {name: float(roc_auc_score(y_dev[held], col)) for name, col in zip(BRANCHES, logits[:len(held)].T, strict=True)}
        positives = int(y_dev[held].sum())
        fold_metrics.append({"fold": fold + 1, "rows": len(held), "pairs": positives * (len(held) - positives),
                             **aucs, "delta_roc_auc": aucs["density_xgb"] - aucs["control_xgb"]})
    if not np.isfinite(oof).all():
        raise ValueError("Missing development predictions.")

    dev_probs = dict(zip(BRANCHES, expit(oof).T, strict=True))
    hold_probs = {"current_grandmaster": saved, **dict(zip(BRANCHES, expit(holdout_sum / cv.n_splits).T, strict=True))}
    pair_counts = [row["pairs"] for row in fold_metrics]
    summary = {
        **identity,
        "checkpoint_signature": signature,
        "holdout_previously_used": True,
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "development_pooled": {name: metrics(y_dev, p) for name, p in dev_probs.items()},
        "holdout": {name: metrics(y_holdout, p) for name, p in hold_probs.items()},
        "holdout_density_vs_control": paired_auc_delta(y_holdout, hold_probs["density_xgb"], hold_probs["control_xgb"]),
        "holdout_density_vs_grandmaster": paired_auc_delta(y_holdout, hold_probs["density_xgb"], saved),
        "development_folds": fold_metrics,
        "development_pair_weighted_auc": {name: float(np.average([row[name] for row in fold_metrics], weights=pair_counts)) for name in BRANCHES},
        "development_pair_weighted_delta": float(np.average([row["delta_roc_auc"] for row in fold_metrics], weights=pair_counts)),
        "development_improving_folds": sum(row["delta_roc_auc"] > 0 for row in fold_metrics),
        "density_feature_splits": dict(zip(ClassConditionalDensity.feature_names, density_feature_splits.tolist(), strict=True)),
        "elapsed_seconds": perf_counter() - started,
    }
    dump_csv(RESULTS / "holdout_predictions.csv", pd.DataFrame({"id": holdout_ids, "is_fraud": y_holdout, **hold_probs}))
    dump_csv(RESULTS / "development_predictions.csv", pd.DataFrame({"id": dev_ids, "is_fraud": y_dev, **dev_probs}))
    temp = RESULTS / "metrics.tmp.json"
    temp.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(RESULTS / "metrics.json")
    print(f"Verified density-tree predictions and wrote results to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
