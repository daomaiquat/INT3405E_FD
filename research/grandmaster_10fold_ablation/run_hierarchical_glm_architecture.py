"""Evaluate two new hierarchical GLM branches using verified Grandmaster fold caches."""

import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from run_ablation import NOTEBOOK_SHA256, ROOT, file_hash, load_archive
from run_meta_improvements import VALIDATION, assert_recorded_predictions, own_validation_inputs


RESULTS = ROOT / "hierarchical_glm_architecture"
BRANCHES = ("hierarchical_glm", "hierarchical_glm_interactions")
SOLVER = {"C": 1.0, "solver": "lbfgs", "max_iter": 500, "tol": 1e-5}


def save_fold(path, signature, held_rows, held_logits, holdout_logits):
    pending = path.with_suffix(".tmp.npz")
    np.savez(pending, signature=signature, held_rows=held_rows,
             held_logits=held_logits, holdout_logits=holdout_logits)
    pending.replace(path)


def load_fold(path, signature, held_rows, holdout_count):
    with np.load(path, allow_pickle=False) as saved:
        if saved["signature"].item() != signature or not np.array_equal(saved["held_rows"], held_rows):
            raise ValueError(f"Checkpoint does not match this fold: {path}.")
        held, holdout = saved["held_logits"], saved["holdout_logits"]
    if held.shape != (len(held_rows), len(BRANCHES)) or holdout.shape != (holdout_count, len(BRANCHES)):
        raise ValueError(f"Wrong prediction shape in {path}.")
    if not np.isfinite(held).all() or not np.isfinite(holdout).all():
        raise ValueError(f"Nonfinite branch logits in {path}.")
    return held, holdout


def fit_fold(design, development, labels, fit, held, holdout):
    fit_frame = development.iloc[fit]
    held_frame = development.iloc[held]
    feature_design = design().fit(fit_frame)
    held_columns, holdout_columns = [], []
    for name, interactions in zip(BRANCHES, (False, True), strict=True):
        started = perf_counter()
        training = feature_design.transform(fit_frame, interactions=interactions).astype(np.float64, copy=False)
        shape, nnz = training.shape, training.nnz
        model = LogisticRegression(**SOLVER).fit(training, labels[fit])
        if model.n_iter_.item() >= SOLVER["max_iter"]:
            raise RuntimeError(f"{name} reached the iteration limit before convergence.")
        del training
        held_columns.append(model.decision_function(feature_design.transform(held_frame, interactions=interactions)))
        holdout_columns.append(model.decision_function(feature_design.transform(holdout, interactions=interactions)))
        print(f"  {name}: {shape[1]:,} columns, {nnz:,} nonzeros, "
              f"{model.n_iter_.item()} iterations, {perf_counter() - started:.1f}s", flush=True)
    return np.column_stack(held_columns), np.column_stack(holdout_columns)


def main():
    from hierarchical_glm_design import HierarchicalGLMDesign

    began = perf_counter()
    meta = json.loads((VALIDATION / "metrics.json").read_text(encoding="utf-8"))
    train_path = ROOT.parents[1] / 'data' / "train.csv"
    train_hash = file_hash(train_path)
    if train_hash != meta["train_sha256"] or NOTEBOOK_SHA256 != meta["notebook_sha256"] or meta["cv_folds"] != 10:
        raise ValueError("The data or notebook differs from the verified 10-fold comparison.")
    nb, _ = load_archive(RESULTS, smoke=False)
    categories = ("merchant_category", "card_type", "auth_method", "channel", "device_type")
    train = pd.read_csv(train_path, dtype={name: "category" for name in categories})
    training, labels, folds, score_folds, ids, holdout_labels = own_validation_inputs(nb, train, meta)
    split = nb["StratifiedShuffleSplit"](n_splits=1, test_size=meta["holdout_fraction"], random_state=meta["holdout_seed"])
    development_rows, holdout_rows = next(split.split(train["id"], train["is_fraud"]))
    development = train.iloc[np.sort(development_rows)].reset_index(drop=True)
    holdout = train.iloc[np.sort(holdout_rows)].reset_index(drop=True)
    np.testing.assert_array_equal(development.pop("is_fraud"), labels)
    np.testing.assert_array_equal(holdout.pop("is_fraud"), holdout_labels)
    np.testing.assert_array_equal(holdout.pop("id"), ids)
    development.pop("id")
    del train

    fit_sources = ("hierarchical_glm_design.py", Path(__file__).name)
    identity = {
        "train_sha256": train_hash,
        "notebook_sha256": NOTEBOOK_SHA256,
        "parent_validation_sha256": file_hash(VALIDATION / "metrics.json"),
        "holdout_ids_sha256": meta["holdout_ids_sha256"],
        "folds": len(folds),
        "cv_seed": meta["cv_seed"],
        "environment": nb["ENV_INFO"],
        "branch_names": BRANCHES,
        "branch_solver": SOLVER,
        "fit_implementation_sha256": {source: file_hash(ROOT / source) for source in fit_sources},
    }
    signature = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    artifact_dir = RESULTS / "artifacts" / signature[:12]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    new_oof = np.full((len(labels), len(BRANCHES)), np.nan)
    new_holdout_folds = []
    for fold, fit, held in folds:
        path = artifact_dir / f"fold_{fold + 1:02d}.npz"
        if path.exists():
            print(f"Fold {fold + 1}/{len(folds)}: restoring verified branch logits", flush=True)
        else:
            print(f"Fold {fold + 1}/{len(folds)}: fitting branches on {len(fit):,} rows", flush=True)
            fold_held, fold_holdout = fit_fold(HierarchicalGLMDesign, development, labels, fit, held, holdout)
            save_fold(path, signature, held, fold_held, fold_holdout)
        fold_held, fold_holdout = load_fold(path, signature, held, len(ids))
        new_oof[held] = fold_held
        new_holdout_folds.append(fold_holdout)

    from hierarchical_glm_evaluation import evaluate_architecture

    evidence = evaluate_architecture(nb, meta, training, score_folds, labels, holdout_labels, folds,
                                     new_oof, new_holdout_folds)
    predictions = evidence.pop("predictions")
    assert_recorded_predictions(VALIDATION / "holdout_predictions.csv", ids, predictions["current_hierarchy"], "hierarchical_amount")
    branch_holdout = np.mean(new_holdout_folds, axis=0)
    standalone_holdout = {name: float(roc_auc_score(holdout_labels, branch_holdout[:, i])) for i, name in enumerate(BRANCHES)}
    summary = {
        **identity,
        "training_rows": len(labels),
        "holdout_rows": len(ids),
        "holdout_previously_used": True,
        "checkpoint_signature": signature,
        "evaluation_sha256": file_hash(ROOT / "hierarchical_glm_evaluation.py"),
        **evidence,
        "standalone_holdout_roc_auc": standalone_holdout,
        "elapsed_seconds": perf_counter() - began,
    }
    pending_predictions = RESULTS / "holdout_predictions.tmp.csv"
    pd.DataFrame({"id": ids, "is_fraud": holdout_labels, **predictions}).to_csv(pending_predictions, index=False)
    pending_predictions.replace(RESULTS / "holdout_predictions.csv")
    pending_report = RESULTS / "metrics.tmp.json"
    pending_report.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    pending_report.replace(RESULTS / "metrics.json")
    print(f"Verified baseline reproduction and wrote experiment results to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
