"""Check whether density features improve a tree fitted on Grandmaster's Blend50 offset."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from run_ablation import NOTEBOOK_SHA256, ROOT, file_hash, load_archive
from run_density_tree import cross_fit_inputs, dump_csv, metrics, paired_auc_delta
from run_meta_improvements import VALIDATION, assert_recorded_predictions, own_validation_inputs


RESULTS = ROOT / "density_offset_validation"
PRIMARY_VARIANT = "grandmaster_density_delta"
BOOSTER_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "device": "cuda",
    "grow_policy": "lossguide",
    "max_depth": 0,
    "max_leaves": 4,
    "learning_rate": 0.025,
    "min_child_weight": 50,
    "reg_alpha": 2.0,
    "reg_lambda": 10.0,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "max_bin": 1024,
    "n_jobs": 4,
}


def transfer_density_delta(anchor, control, density, weight):
    """Keep the parent ensemble and apply only the difference between matched trees."""
    anchor, control, density = (np.asarray(x, dtype=np.float64) for x in (anchor, control, density))
    if anchor.ndim != 1 or anchor.shape != control.shape or anchor.shape != density.shape:
        raise ValueError("Expected aligned one-dimensional margins.")
    if not all(np.isfinite(x).all() for x in (anchor, control, density, weight)):
        raise ValueError("Expected finite margins and weight.")
    return anchor + weight * (density - control)


def read_blend50(nb, partitions, development_rows, holdout_rows):
    names = nb["LOCKED_MODELS"]
    if names[:2] != ["02e_SpGLM", "02f_DeGLM"]:
        raise ValueError("Expected the archived Sparse/Dense GLM offsets first.")
    held_oof = np.full(development_rows, np.nan)
    holdout_sum = np.zeros(holdout_rows)
    for fold, _, held in partitions:
        with np.load(VALIDATION / "artifacts" / f"fold_{fold + 1:02d}.npz", allow_pickle=False) as artifact:
            np.testing.assert_array_equal(artifact["va_idx"], held)
            held_oof[held] = artifact["baseline_va"][:, :2].mean(axis=1)
            holdout_sum += artifact["baseline_hold"][:, :2].mean(axis=1)
    if not np.isfinite(held_oof).all() or not np.isfinite(holdout_sum).all():
        raise ValueError("Nonfinite Blend50 inputs.")
    return held_oof, holdout_sum / len(partitions)


def fit_offset_models(training, labels, train_offset, scoring, rounds, seeds):
    from xgboost import XGBClassifier

    zero_margin = np.zeros(len(scoring), dtype=np.float32)
    residual = np.zeros(len(scoring), dtype=np.float64)
    splits = {}
    for seed in seeds:
        tree = XGBClassifier(**BOOSTER_PARAMS, random_state=seed, n_estimators=rounds)
        tree.fit(training, labels, base_margin=train_offset, verbose=False)
        booster = tree.get_booster()
        booster.set_param({"device": "cpu"})
        residual += booster.inplace_predict(scoring, predict_type="margin", base_margin=zero_margin)
        for name, count in booster.get_score(importance_type="weight").items():
            splits[name] = splits.get(name, 0) + int(count)
    return residual / len(seeds), splits


def main():
    from class_conditional_density import ClassConditionalDensity
    from density_tree_inputs import DensityTreeInputs

    started = perf_counter()
    meta = json.loads((VALIDATION / "metrics.json").read_text(encoding="utf-8"))
    train_path = ROOT.parents[1] / 'data' / "train.csv"
    train_sha = file_hash(train_path)
    if train_sha != meta["train_sha256"] or NOTEBOOK_SHA256 != meta["notebook_sha256"] or meta["cv_folds"] != 10:
        raise ValueError("Expected the verified original development/holdout split.")
    nb, _ = load_archive(RESULTS, smoke=False)
    categories = ("merchant_category", "card_type", "auth_method", "channel", "device_type")
    train = pd.read_csv(train_path, dtype={name: "category" for name in categories})
    matrix, labels, folds, scoring, ids, hold_labels = own_validation_inputs(nb, train, meta)
    print(f"Recreating the frozen parent ensemble for {len(ids):,} holdout rows", flush=True)
    offsets, hold_offsets = read_blend50(nb, folds, len(labels), len(ids))
    params = [nb["_fit_nonneg_l2_weights"](matrix[fit], labels[fit].astype(float), l2)
              for (_, fit, _), l2 in zip(folds, meta["baseline_meta_l2_by_fold"], strict=True)]
    intercept = np.mean([b for b, _ in params])
    weights = np.mean([w for _, w in params], axis=0)
    summed = np.mean(scoring, axis=0)
    anchor = intercept + summed @ weights
    _, column_names = nb["build_resdecomp"]({name: np.zeros(1) for name in nb["LOCKED_MODELS"]})
    source_column = column_names.index("delta_03b_OffXGB")
    frozen_weight = float(weights[source_column])
    old_residual = summed[:, source_column]
    link = nb["from_logit_10"]
    assert_recorded_predictions(VALIDATION / "holdout_predictions.csv", ids, link(anchor), "hierarchical_amount")
    del matrix, scoring, summed
    print(f"Frozen OffXGB weight {frozen_weight:.6f}", flush=True)

    splitter = nb["StratifiedShuffleSplit"](n_splits=1, test_size=meta["holdout_fraction"], random_state=meta["holdout_seed"])
    fitted, held = (np.sort(index) for index in next(splitter.split(train["id"], train["is_fraud"])))
    development = train.iloc[fitted].reset_index(drop=True)
    holdout = train.iloc[held].reset_index(drop=True)
    np.testing.assert_array_equal(development.pop("is_fraud"), labels)
    np.testing.assert_array_equal(holdout.pop("is_fraud"), hold_labels)
    np.testing.assert_array_equal(holdout.pop("id"), ids)
    development.pop("id")
    del train

    inner_folds = 4
    factories = (DensityTreeInputs, ClassConditionalDensity)
    print(f"Cross-fitting inputs over {len(development):,} development rows", flush=True)
    (common, common_hold), (density, density_hold) = cross_fit_inputs(
        development, labels, holdout, factories, inner_folds=inner_folds, seed=7041,
    )
    del development, holdout
    width = common.shape[1]
    rounds, seeds = nb["XGB_N_EST"], nb["SEEDS"]
    print(f"Fitting the matched offset-tree control ({rounds} rounds × {len(seeds)} seeds)", flush=True)
    control, _ = fit_offset_models(common, labels, offsets, common_hold, rounds, seeds)
    print("Fitting the offset tree with density features", flush=True)
    common = np.column_stack((common, density))
    common_hold = np.column_stack((common_hold, density_hold))
    del density, density_hold
    treatment, splits = fit_offset_models(common, labels, offsets, common_hold, rounds, seeds)
    del common, common_hold

    scores = {
        "current_grandmaster": link(anchor),
        "offset_tree_control": link(hold_offsets + control),
        "offset_tree_density": link(hold_offsets + treatment),
        "grandmaster_density_replacement": link(transfer_density_delta(anchor, old_residual, treatment, frozen_weight)),
        PRIMARY_VARIANT: link(transfer_density_delta(anchor, control, treatment, frozen_weight)),
    }
    metrics_by_model = {name: metrics(hold_labels, values) for name, values in scores.items()}
    summary = {
        "primary_comparison": PRIMARY_VARIANT,
        "train_sha256": train_sha,
        "notebook_sha256": NOTEBOOK_SHA256,
        "holdout_ids_sha256": meta["holdout_ids_sha256"],
        "implementation_sha256": {name: file_hash(ROOT / name) for name in
                                  (Path(__file__).name, "run_density_tree.py", "class_conditional_density.py", "density_tree_inputs.py")},
        "development_rows": len(labels),
        "holdout_rows": len(ids),
        "holdout_previously_used": True,
        "offset_outer_folds": len(folds),
        "feature_inner_folds": inner_folds,
        "frozen_offxgb_weight": frozen_weight,
        "booster_params": {**BOOSTER_PARAMS, "n_estimators": rounds},
        "booster_seeds": seeds,
        "holdout": metrics_by_model,
        "offset_density_vs_control": paired_auc_delta(hold_labels, scores["offset_tree_density"], scores["offset_tree_control"]),
        "stack_comparison": {name: paired_auc_delta(hold_labels, scores[name], scores["current_grandmaster"])
                             for name in (PRIMARY_VARIANT, "grandmaster_density_replacement")},
        "density_feature_splits": {name: splits.get(f"f{width + i}", 0)
                                   for i, name in enumerate(ClassConditionalDensity.feature_names)},
        "elapsed_seconds": perf_counter() - started,
    }
    dump_csv(RESULTS / "holdout_predictions.csv", pd.DataFrame({"id": ids, "is_fraud": hold_labels, **scores}))
    pending = RESULTS / "metrics.tmp.json"
    pending.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(RESULTS / "metrics.json")
    print(f"Verified offset experiment outputs in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
