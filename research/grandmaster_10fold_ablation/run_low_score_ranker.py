"""Evaluate boosted pairwise corrections for the lowest-scored frauds."""

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from run_ablation import NOTEBOOK_SHA256, ROOT, file_hash, load_archive
from run_density_tree import dump_csv
from run_meta_improvements import VALIDATION, own_validation_inputs


RESULTS = ROOT / "low_score_rank_validation"
NAMES = ("current_grandmaster", "pointwise", "pairwise")
STACK_L2 = 1e-5
STACK_INNER_FOLDS = 3
MAX_CHANGE = 0.4
RANK_WEIGHT = 0.01
ROUNDS = 96
TREE_PARAMS = {
    "tree_method": "hist",
    "device": "cuda",
    "objective": "reg:squarederror",
    "base_score": 0.0,
    "max_depth": 3,
    "eta": 0.035,
    "min_child_weight": 15.0,
    "lambda": 15.0,
    "alpha": 1.0,
    "subsample": 0.85,
    "max_bin": 256,
    "nthread": 4,
}


def apply_correction(base, gate, correction):
    arrays = [np.asarray(x, dtype=np.float64) for x in (base, gate, correction)]
    if arrays[0].ndim != 1 or not all(x.shape == arrays[0].shape and np.isfinite(x).all() for x in arrays):
        raise ValueError("Expected aligned finite logit, gate and correction vectors.")
    base, gate, correction = arrays
    if ((gate < 0) | (gate > 1)).any():
        raise ValueError("Expected gate weights in [0, 1].")
    return base + np.clip(gate * correction, -MAX_CHANGE, MAX_CHANGE)


def anchor_logits(fit_stack, train_scores, labels, eval_scores, seed):
    """Train the meta model only on historical holdout rows outside the base fit."""
    labels = labels.astype(np.float64)
    splitter = StratifiedKFold(n_splits=STACK_INNER_FOLDS, shuffle=True, random_state=seed)
    train_logits = np.full(len(labels), np.nan)
    for fit, held in splitter.split(train_scores, labels):
        b, w = fit_stack(train_scores[fit], labels[fit], STACK_L2)
        train_logits[held] = b + train_scores[held] @ w
    b, w = fit_stack(train_scores, labels, STACK_L2)
    return train_logits, b + eval_scores @ w


def feature_matrices(raw, y, held, fit, historical, h_scores, v_scores, h_base, v_base, factories):
    upstream = raw.iloc[fit]
    validation = raw.iloc[held]
    fit_blocks, eval_blocks = [], []
    for factory in factories:
        model = factory().fit(upstream, y[fit])
        fit_blocks.append(model.transform(historical))
        eval_blocks.append(model.transform(validation))
    fit_blocks.extend((h_scores, h_base))
    eval_blocks.extend((v_scores, v_base))
    return [np.column_stack(blocks).astype(np.float32, copy=False) for blocks in (fit_blocks, eval_blocks)]


def fit_rank_trees(fitting, labels, h_base, scoring, v_base, seed):
    import xgboost as xgb
    from low_score_rank_objective import LowScoreObjective, low_score_gate

    matrix = xgb.DMatrix(fitting, label=labels, nthread=TREE_PARAMS["nthread"])
    result = [v_base]
    gate = low_score_gate(v_base)
    for rank_weight in (0.0, RANK_WEIGHT):
        objective = LowScoreObjective(labels, h_base, rank_weight=rank_weight, seed=seed)

        def training_objective(corrections, _):
            return objective.gradient_hessian(corrections)

        model = xgb.train({**TREE_PARAMS, "seed": seed}, matrix, ROUNDS, obj=training_objective)
        model.set_param({"device": "cpu"})
        correction = model.inplace_predict(scoring, predict_type="margin")
        result.append(apply_correction(v_base, gate, correction))
    return np.column_stack(result)


def main():
    from class_conditional_density import ClassConditionalDensity
    from density_tree_inputs import DensityTreeInputs

    began = perf_counter()
    meta = json.loads((VALIDATION / "metrics.json").read_text(encoding="utf-8"))
    train_path = ROOT.parents[1] / 'data' / "train.csv"
    data_hash = file_hash(train_path)
    if (data_hash != meta["train_sha256"] or NOTEBOOK_SHA256 != meta["notebook_sha256"] or meta["cv_folds"] != 10):
        raise ValueError("Expected the verified 10-fold development checkpoints.")
    nb, _ = load_archive(RESULTS, smoke=False)
    categories = ("merchant_category", "card_type", "auth_method", "channel", "device_type")
    full = pd.read_csv(train_path, dtype={name: "category" for name in categories})
    training, y, folds, h_scores, h_ids, h_labels = own_validation_inputs(nb, full, meta)
    splitter = nb["StratifiedShuffleSplit"](n_splits=1, test_size=meta["holdout_fraction"], random_state=meta["holdout_seed"])
    dev_rows, h_rows = (np.sort(idx) for idx in next(splitter.split(full["id"], full["is_fraud"])))
    development = full.iloc[dev_rows].reset_index(drop=True)
    historical = full.iloc[h_rows].reset_index(drop=True)
    del full
    ids = development.pop("id").to_numpy(dtype=np.int64)
    np.testing.assert_array_equal(development.pop("is_fraud"), y)
    np.testing.assert_array_equal(historical.pop("id"), h_ids)
    np.testing.assert_array_equal(historical.pop("is_fraud"), h_labels)

    code = (Path(__file__).name, "low_score_rank_objective.py", "class_conditional_density.py", "density_tree_inputs.py")
    identity = {
        "train_sha256": data_hash,
        "notebook_sha256": NOTEBOOK_SHA256,
        "historical_training_ids_sha256": meta["holdout_ids_sha256"],
        "development_validation_ids_sha256": sha256(ids.tobytes()).hexdigest(),
        "base_cache_sha256": {str(f + 1): file_hash(VALIDATION / "artifacts" / f"fold_{f + 1:02d}.npz") for f, _, _ in folds},
        "fit_implementation_sha256": {name: file_hash(ROOT / name) for name in code},
        "environment": nb["ENV_INFO"],
        "outer_folds": len(folds),
        "outer_seed": meta["cv_seed"],
        "stack_l2": STACK_L2,
        "stack_inner_folds": STACK_INNER_FOLDS,
        "tree_params": TREE_PARAMS,
        "boosting_rounds": ROUNDS,
        "pairwise_weight": RANK_WEIGHT,
        "correction_cap": MAX_CHANGE,
    }
    signature = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    artifacts = RESULTS / "artifacts" / signature[:12]
    artifacts.mkdir(parents=True, exist_ok=True)
    predictions = np.full((len(y), len(NAMES)), np.nan)
    fold_ids = np.zeros(len(y), dtype=np.int8)
    factories = (DensityTreeInputs, ClassConditionalDensity)
    print(f"Trees fit on {len(historical):,} historical rows; scoring {len(development):,} development rows across folds", flush=True)

    for fold, fit, held in folds:
        path = artifacts / f"fold_{fold + 1:02d}.npz"
        started = perf_counter()
        if path.exists():
            with np.load(path, allow_pickle=False) as record:
                if record["signature"].item() != signature or not np.array_equal(record["held"], held):
                    raise ValueError(f"Unexpected checkpoint: {path}.")
                logits = record["logits"]
        else:
            expected_train = h_scores[fold]
            expected_held = training[held]
            h_base, v_base = anchor_logits(nb["_fit_nonneg_l2_weights"], expected_train, h_labels, expected_held, 4679 + fold)
            X_H, X_V = feature_matrices(development, y, held, fit, historical, expected_train, expected_held, h_base, v_base, factories)
            logits = fit_rank_trees(X_H, h_labels, h_base, X_V, v_base, 8021 + fold)
            del X_H, X_V
            if logits.shape == (len(held), len(NAMES)) and np.isfinite(logits).all():
                pending = path.with_suffix(".tmp.npz")
                np.savez(pending, signature=signature, held=held, logits=logits)
                pending.replace(path)
        if logits.shape != (len(held), len(NAMES)) or not np.isfinite(logits).all():
            raise ValueError(f"Invalid predictions in fold {fold + 1}.")
        fold_ids[held] = fold + 1
        predictions[held] = logits
        print(f"Fold {fold + 1}/{len(folds)} verified ({perf_counter() - started:.1f}s)", flush=True)

    from low_score_rank_evaluation import evaluate_fold, summarize_folds

    if not np.isfinite(predictions).all() or (fold_ids == 0).any():
        raise ValueError("Incomplete fold predictions.")
    reports = [{"fold": fold + 1, "rows": len(held), **evaluate_fold(y[held], dict(zip(NAMES, predictions[held].T, strict=True)))}
               for fold, _, held in folds]
    summary = {
        **identity,
        "checkpoint_signature": signature,
        "evaluator_sha256": file_hash(ROOT / "low_score_rank_evaluation.py"),
        "validation_rows": len(y),
        "historical_training_rows": len(historical),
        "evaluation": summarize_folds(reports),
        "fold_reports": reports,
        "elapsed_seconds": perf_counter() - began,
    }
    probabilities = dict(zip(NAMES, expit(predictions).T, strict=True))
    dump_csv(RESULTS / "development_predictions.csv", pd.DataFrame({"id": ids, "is_fraud": y, "fold": fold_ids, **probabilities}))
    pending = RESULTS / "metrics.tmp.json"
    pending.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(RESULTS / "metrics.json")
    print(f"Verified low-score rank results in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
