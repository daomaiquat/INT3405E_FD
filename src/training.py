"""Step-5 baseline training on the locked outer folds.

Each model writes id-keyed OOF/test predictions, fold metrics and a reloadable
bundle. Test labels are never read. Tree boosters choose n_trees on A/E and refit
on T; fixed models fit A once only for an E diagnostic and then refit on T.
"""

from __future__ import annotations

import logging
from time import perf_counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .artifacts import (fold_needs_run, load_bundle, read_fold_status, read_json,
                        save_bundle, write_fold_status, write_json)
from .features import feature_groups
from .metrics import auc_score, summarize_oof
from .models import BASELINES, fit_bundle, predict_bundle, select_iterations
from .predictions import join_oof, join_test, prediction_frame, shuffle_for_check
from .validation import partition_for_fold


def _fold_selection(model_id: str, part: dict, train: pd.DataFrame, config: dict,
                    fold: int) -> dict:
    if BASELINES[model_id].early_stop:
        return select_iterations(model_id, train.iloc[part["A"]], train.iloc[part["E"]],
                                 config, fold)
    probe = fit_bundle(model_id, train.iloc[part["A"]], config, fold=fold)
    evaluation = train.iloc[part["E"]]
    return {
        "n_trees": int(probe["n_trees"]),
        "best_iteration": None,
        "eval_auc": auc_score(evaluation[config["target"]], predict_bundle(probe, evaluation)),
        "warnings": list(probe["warnings"]),
    }


def _write_comparison(run_dir) -> pd.DataFrame:
    rows = []
    for model_id in BASELINES:
        path = run_dir / "baselines" / model_id / "metrics.json"
        if not path.is_file():
            continue
        metrics = read_json(path)
        rows.append({
            "model": model_id,
            "oof_auc": metrics["oof_auc"],
            "fold_auc": metrics["mean_auc"],
            "std_auc": metrics["std_auc"],
            "train_auc": metrics.get("train_auc", np.nan),
            "train_cv_gap": metrics.get("train_auc", np.nan) - metrics["oof_auc"],
            "eval_auc": metrics.get("eval_auc_mean", np.nan),
            "logloss": metrics["logloss"],
            "n_trees": metrics.get("n_trees_representative", 0),
        })
    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        comparison.to_csv(run_dir / "baseline_comparison.csv", index=False)
    return comparison


def _write_oof_correlations(run_dir) -> None:
    frames = {}
    for model_id in BASELINES:
        path = run_dir / "baselines" / model_id / "oof.csv"
        if path.is_file():
            frames[model_id] = pd.read_csv(path).set_index("id")["prediction"]
    if len(frames) < 2:
        return
    aligned = pd.concat(frames.values(), axis=1, keys=frames.keys())
    aligned.corr(method="spearman").to_csv(run_dir / "baseline_oof_spearman.csv")


def run_baseline(model_id: str, run_dir, train: pd.DataFrame, test: pd.DataFrame,
                 sample: pd.DataFrame, folds: pd.DataFrame, early: pd.DataFrame,
                 config: dict, n_folds: int, state: dict) -> dict:
    """Run one selected baseline with resume support and return a compact summary."""
    if model_id not in BASELINES:
        raise ValueError(f"unknown baseline: {model_id}")
    spec = BASELINES[model_id]
    target = config["target"]
    exp_dir = run_dir / "baselines" / model_id
    bundle_dir = exp_dir / "model_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    status_path = exp_dir / "fold_status.json"
    metrics_path = exp_dir / "fold_metrics.json"
    status = read_fold_status(status_path)
    persisted = read_json(metrics_path) if metrics_path.is_file() else {}
    oof_outputs, test_outputs, fold_metrics = {}, {}, {}

    write_json(exp_dir / "feature_groups.json", feature_groups(model_id, config, spec.mode))
    write_json(exp_dir / "params.json", {
        "model": spec.name, "mode": spec.mode, "family": spec.family,
        "early_stop": spec.early_stop, "fixed_trees": spec.fixed_trees,
        "model_seed": config["model_seed"], "patience": config["baselines"]["early_stopping"]["patience"],
        "params": config["baselines"].get({
            "B02": "logistic", "B03": "decision_tree", "B04": "random_forest",
            "B05": "lightgbm", "B06": "xgboost", "B07": "catboost"}.get(model_id, ""), {}),
    })

    for fold in range(n_folds):
        oof_path = exp_dir / f"oof_fold_{fold}.csv"
        test_path = exp_dir / f"test_fold_{fold}.csv"
        bundle_path = bundle_dir / f"fold_{fold}.joblib"
        required = [oof_path, test_path, bundle_path]
        reusable = (state.get("resumed") and not fold_needs_run(status, fold, required)
                    and "train_auc" in persisted.get(str(fold), {}))
        if reusable:
            oof_outputs[fold] = pd.read_csv(oof_path)
            test_outputs[fold] = pd.read_csv(test_path)
            fold_metrics[str(fold)] = persisted[str(fold)]
            logging.info("Reusing completed fold %s for %s", fold, model_id)
            continue
        status[str(fold)] = "started"
        write_fold_status(status_path, status)
        try:
            part = partition_for_fold(train, folds, early, config, fold)
            started = perf_counter()
            selection = _fold_selection(model_id, part, train, config, fold)
            selected_at = perf_counter()
            bundle = fit_bundle(model_id, train.iloc[part["T"]], config,
                                n_trees=selection["n_trees"], fold=fold)
            fitted_at = perf_counter()
            valid_prediction = predict_bundle(bundle, train.iloc[part["V"]])
            test_prediction = predict_bundle(bundle, test)
            save_bundle(bundle, bundle_path)
            reloaded_prediction = predict_bundle(load_bundle(bundle_path), train.iloc[part["V"]])
            roundtrip = float(np.max(np.abs(valid_prediction - reloaded_prediction)))
            if roundtrip > 1e-12:
                raise ValueError(f"{model_id} fold {fold}: bundle roundtrip changed predictions")
            train_prediction = predict_bundle(bundle, train.iloc[part["T"]])
            warnings = sorted(set(selection["warnings"]) | set(bundle["warnings"]))
            oof_frame = prediction_frame(train["id"].iloc[part["V"]].to_numpy(), valid_prediction, fold)
            test_frame = prediction_frame(test["id"].to_numpy(), test_prediction, fold)
            oof_frame.to_csv(oof_path, index=False)
            test_frame.to_csv(test_path, index=False)
            oof_outputs[fold] = oof_frame
            test_outputs[fold] = test_frame
            fold_metrics[str(fold)] = {
                "sizes": {name: len(values) for name, values in part.items()},
                "eval_auc": float(selection["eval_auc"]),
                "best_iteration": selection["best_iteration"],
                "n_trees": int(bundle["n_trees"]),
                "train_auc": auc_score(train[target].iloc[part["T"]], train_prediction),
                "convergence_warnings": warnings,
                "bundle_roundtrip_max_abs_diff": roundtrip,
                "reused": False,
                "selection_seconds": selected_at - started,
                "refit_seconds": fitted_at - selected_at,
                "total_seconds": perf_counter() - started,
            }
            write_json(metrics_path, fold_metrics)
            status[str(fold)] = "completed"
            write_fold_status(status_path, status)
            logging.info("%s fold %s fit; E AUC %.6f; trees %s", model_id, fold,
                         selection["eval_auc"], bundle["n_trees"])
        except Exception:
            status[str(fold)] = "failed"
            write_fold_status(status_path, status)
            raise

    oof = join_oof(oof_outputs, folds, config)
    test_by_fold = join_test(test_outputs, sample, n_folds)
    pd.testing.assert_frame_equal(
        oof, join_oof({k: shuffle_for_check(v, 300 + k) for k, v in oof_outputs.items()},
                      folds, config))
    pd.testing.assert_frame_equal(
        test_by_fold, join_test({k: shuffle_for_check(v, 400 + k) for k, v in test_outputs.items()},
                                sample, n_folds))

    all_warnings = sorted({item for values in fold_metrics.values()
                           for item in values.get("convergence_warnings", [])})
    representative = int(list(fold_metrics.values())[-1]["n_trees"])
    metrics = summarize_oof(
        oof,
        train_auc=float(np.mean([item["train_auc"] for item in fold_metrics.values()])),
        eval_auc_mean=float(np.mean([item["eval_auc"] for item in fold_metrics.values()])),
        n_trees_representative=representative,
        n_trees_by_fold={fold: item["n_trees"] for fold, item in fold_metrics.items()},
        convergence_warnings=all_warnings,
        model=model_id,
        smoke=bool(state.get("smoke")),
    )
    oof.to_csv(exp_dir / "oof.csv", index=False)
    test_by_fold.to_csv(exp_dir / "test_by_fold.csv", index=False)
    write_json(exp_dir / "metrics.json", metrics)
    write_json(metrics_path, fold_metrics)
    write_json(exp_dir / "baseline_checks.json", {
        "oof_rows": len(oof), "test_rows": len(test_by_fold),
        "oof_ids_unique": bool(oof["id"].is_unique),
        "each_train_row_scored_by_own_fold": True,
        "shuffled_input_joins_identical": True,
        "test_rows_used_for_labeled_fit": 0,
        "bundle_roundtrip_max_abs_diff": max(
            item["bundle_roundtrip_max_abs_diff"] for item in fold_metrics.values()),
        "fold_status": status,
    })
    comparison = _write_comparison(run_dir)
    _write_oof_correlations(run_dir)
    log_path = run_dir / "experiment_log.csv"
    pd.DataFrame([{
        "experiment_id": f"step5_{model_id}", "run_id": state["run_id"], "status": "completed",
        "event_time": datetime.now(timezone.utc).isoformat(),
        "resumed": bool(state.get("resumed")), "model": model_id,
        "family": spec.family, "oof_auc": metrics["oof_auc"],
        "mean_auc": metrics["mean_auc"], "std_auc": metrics["std_auc"],
        "train_auc": metrics["train_auc"],
        "hypothesis": f"step 5 baseline {spec.name}",
        "note": "development CV; smoke=true means sample only" if state.get("smoke") else
                "development CV; not a final selection",
    }]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    logging.info("%s OOF AUC %.6f (%s)", model_id, metrics["oof_auc"],
                 "smoke" if state.get("smoke") else "full")
    return {"model": model_id, "metrics": metrics, "fold_status": status,
            "comparison_rows": len(comparison)}
