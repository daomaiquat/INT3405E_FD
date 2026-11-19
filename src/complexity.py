"""Step 6: controlled tree complexity ablations and a one-fold learning curve."""

from __future__ import annotations

import copy
import logging
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .artifacts import read_json, save_bundle, write_json
from .metrics import auc_score
from .models import fit_bundle, predict_bundle, select_iterations
from .training import run_baseline
from .validation import partition_for_fold


CANDIDATES = {
    "S6-L15": ("B05", "lightgbm", "num_leaves", 15),
    "S6-L04": ("B05", "lightgbm", "num_leaves", 4),
    "S6-L08": ("B05", "lightgbm", "num_leaves", 8),
    "S6-L31": ("B05", "lightgbm", "num_leaves", 31),
    "S6-X03": ("B06", "xgboost", "max_depth", 3),
    "S6-X02": ("B06", "xgboost", "max_depth", 2),
}


def candidate_config(config, candidate, smoke=False):
    result = copy.deepcopy(config)
    _, family, parameter, value = CANDIDATES[candidate]
    result["baselines"][family][parameter] = value
    result["baselines"]["lightgbm"]["metric"] = "auc"
    result["baselines"]["early_stopping"]["first_metric_only"] = True
    if smoke:
        result["baselines"][family]["max_iterations"] = 30
        result["baselines"]["early_stopping"]["patience"] = 5
    return result


def _learning_curve(directory, train, folds, early, config, smoke, resumed):
    """V is fixed; every smaller T sample has its own stratified A/E split."""
    directory.mkdir(parents=True, exist_ok=True)
    cfg = candidate_config(config, "S6-L04", smoke)
    target = config["target"]
    part = partition_for_fold(train, folds, early, config, 0)
    rows = []
    for fraction in (0.1, 0.25, 0.5, 1.0):
        path = directory / f"fraction_{fraction:g}.json"
        if resumed and path.is_file():
            rows.append(read_json(path))
            continue
        positions = part["T"]
        if fraction < 1:
            positions, _ = train_test_split(positions, train_size=fraction,
                                            stratify=train[target].iloc[positions],
                                            random_state=config["seed"])
            a, e = train_test_split(positions,
                                   test_size=config["training_plan"]["early_stopping"]["fraction_of_outer_train"],
                                   stratify=train[target].iloc[positions],
                                   random_state=config["early_stop_seed"])
        else:
            a, e = part["A"], part["E"]
        started = perf_counter()
        selected = select_iterations("B05", train.iloc[a], train.iloc[e], cfg, 0)
        bundle = fit_bundle("B05", train.iloc[positions], cfg, selected["n_trees"], 0)
        valid_auc = auc_score(train[target].iloc[part["V"]], predict_bundle(bundle, train.iloc[part["V"]]))
        train_auc = auc_score(train[target].iloc[positions], predict_bundle(bundle, train.iloc[positions]))
        row = {"fraction": fraction, "train_rows": len(positions), "A_rows": len(a),
               "E_rows": len(e), "V_rows": len(part["V"]), "fold": 0,
               "model": "S6-L04", "valid_auc": valid_auc, "train_auc": train_auc,
               "n_trees": selected["n_trees"], "seconds": perf_counter() - started,
               "smoke": smoke, "note": "one-fold diagnostic, not five-fold CV"}
        write_json(path, row)
        rows.append(row)
        logging.info("Learning curve %.0f%%: V AUC %.6f (%s rows)", fraction * 100, valid_auc, len(positions))
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / "learning_curve.csv", index=False)
    return frame


def _plot(directory, comparison, learning, smoke):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for family, label in (("lightgbm", "LightGBM leaves"), ("xgboost", "XGBoost depth")):
        group = comparison.loc[comparison.family == family].sort_values("complexity")
        axes[0].plot(group.complexity, group.oof_auc, "o-", label=label)
        axes[1].plot(group.complexity, group.train_cv_gap, "o-", label=label)
    for axis, ylabel in zip(axes[:2], ("Pooled OOF ROC AUC", "Train AUC minus OOF AUC")):
        axis.set(xlabel="Leaves / depth (separate families)", ylabel=ylabel)
        axis.legend()
    axes[2].plot(learning.train_rows, learning.valid_auc, "o-", label="Validation fold 0")
    axes[2].plot(learning.train_rows, learning.train_auc, "o-", label="Train")
    axes[2].set(xlabel="Rows in sampled outer train", ylabel="ROC AUC", title="Learning curve: LGBM 4 leaves")
    axes[2].legend()
    fig.suptitle("Step 6: smoke pipeline check" if smoke else "Step 6: tree complexity and one-fold learning curve")
    fig.tight_layout()
    fig.savefig(directory / "complexity_learning_curve.png", dpi=150)
    plt.close(fig)


def run_complexity(run_dir, train, test, sample, folds, early, config, n_folds, state):
    directory = run_dir / "complexity"
    directory.mkdir(parents=True, exist_ok=True)
    smoke = bool(state["smoke"])
    rows = []
    fold_scores = {}
    for candidate, (model_id, family, parameter, value) in CANDIDATES.items():
        cfg = candidate_config(config, candidate, smoke)
        logging.info("STEP 6 %s: %s=%s; AUC-only early stopping", candidate, parameter, value)
        result = run_baseline(model_id, directory / candidate, train, test, sample,
                              folds, early, cfg, n_folds, state)
        metrics = result["metrics"]
        details = read_json(directory / candidate / "baselines" / model_id / "fold_metrics.json")
        counts = list(metrics["n_trees_by_fold"].values())
        fold_scores[candidate] = metrics["fold_auc"]
        rows.append({"candidate": candidate, "family": family, "complexity": value,
                     "oof_auc": metrics["oof_auc"], "mean_auc": metrics["mean_auc"],
                     "std_auc": metrics["std_auc"], "train_auc": metrics["train_auc"],
                     "train_cv_gap": metrics["train_auc"] - metrics["oof_auc"],
                     "trees_min": min(counts), "trees_median": float(np.median(counts)),
                     "trees_max": max(counts), "seconds": sum(v.get("total_seconds", 0) for v in details.values()),
                     "hit_tree_cap": max(counts) >= cfg["baselines"][family]["max_iterations"],
                     "smoke": smoke})
        pd.DataFrame(rows).to_csv(directory / "comparison.csv", index=False)
    comparison = pd.DataFrame(rows)
    for row in rows:
        control = "S6-L15" if row["family"] == "lightgbm" else "S6-X03"
        reference = comparison.set_index("candidate").loc[control, "oof_auc"]
        row["control"] = control
        row["delta_oof_auc"] = row["oof_auc"] - reference
        for fold in range(n_folds):
            row[f"delta_fold_{fold}"] = fold_scores[row["candidate"]][str(fold)] - fold_scores[control][str(fold)]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(directory / "comparison.csv", index=False)
    learning = _learning_curve(directory / "learning_curve", train, folds, early,
                               config, smoke, bool(state.get("resumed")))
    _plot(directory, comparison, learning, smoke)
    winners = {family: group.loc[group.oof_auc.idxmax(), "candidate"]
               for family, group in comparison.groupby("family")}
    write_json(directory / "summary.json", {"smoke": smoke, "winners": winners,
               "early_stopping_metric": "auc", "estimator_threads": 1,
               "learning_curve": "S6-L04, fold 0 only; not five-fold CV",
               "selection_status": "pipeline check only" if smoke else "development CV selection"})
    return {"comparison": str(directory / "comparison.csv"), "winners": winners,
            "smoke": smoke, "learning_curve_fold": 0}
