"""Evaluate a raw-feature ranking specialist for hard Grandmaster transactions."""

import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from export_hierarchical_submission import write_submission
from run_ablation import NOTEBOOK_SHA256, ROOT, file_hash, load_archive
from run_meta_improvements import PARENT_SUBMISSION, VALIDATION, own_full_inputs, own_validation_inputs
from run_post_stack_residual import original_oof_logits, read_recorded_probabilities


RESULTS = ROOT / "hard_group_ranker"
METHODS = ("pointwise", "uniform", "matched")
SCALES = (0.25, 0.5, 1.0)
DEVELOPMENT_CV_SEED = 1942
FIT_SEED = 2901
RANK_WEIGHT = 0.005
REGULARIZATION = 0.0005


def pair_auc(positive_scores, negative_scores):
    positive_scores = np.asarray(positive_scores)
    negatives = np.sort(negative_scores)
    if not len(positive_scores) or not len(negatives):
        raise ValueError("Expected positive and negative transactions for pair AUC.")
    smaller = np.searchsorted(negatives, positive_scores, side="left")
    smaller_or_equal = np.searchsorted(negatives, positive_scores, side="right")
    return float(np.mean(0.5 * (smaller + smaller_or_equal) / len(negatives)))


def specialist_scores(parent_probabilities, inside, deltas, scale, link):
    adjusted = np.asarray(parent_probabilities, dtype=np.float64).copy()
    inside = np.asarray(inside, dtype=bool)
    deltas = np.asarray(deltas, dtype=np.float64)
    if inside.shape != adjusted.shape or deltas.shape != adjusted[inside].shape:
        raise ValueError("The correction must contain one score for each target transaction.")
    adjusted[inside] = link(logit(adjusted[inside]) + scale * deltas)
    if not np.isfinite(adjusted).all() or not ((0 <= adjusted) & (adjusted <= 1)).all():
        raise ValueError("The specialist produced invalid probabilities.")
    return adjusted


def score_groups(y, scores, inside):
    positive = y == 1
    pos_in, neg_in = scores[positive & inside], scores[~positive & inside]
    pos_out, neg_out = scores[positive & ~inside], scores[~positive & ~inside]
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "log_loss": float(log_loss(y, scores)),
        "within_group_auc": pair_auc(pos_in, neg_in),
        "target_fraud_vs_other_normal_auc": pair_auc(pos_in, neg_out),
        "other_fraud_vs_target_normal_auc": pair_auc(pos_out, neg_in),
        "target_fraud_count": int(len(pos_in)),
        "target_expected_fraud": float(scores[inside].sum()),
    }


def choose_scales(frame, y, inside, logits, encoder_type, fit_specialist, methods):
    from sklearn.model_selection import StratifiedKFold

    stratification = 2 * y + inside.astype(np.int8)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=DEVELOPMENT_CV_SEED)
    oof = {method: np.full(len(y), np.nan) for method in methods}
    for fold, (fit, held) in enumerate(cv.split(y, stratification), start=1):
        print(f"Specialist scale selection, development fold {fold}/3", flush=True)
        in_fit, in_held = inside[fit], inside[held]
        fit_frame, held_frame = frame.iloc[fit[in_fit]], frame.iloc[held[in_held]]
        encoder = encoder_type().fit(fit_frame)
        feature_fit, feature_held = encoder.transform(fit_frame), encoder.transform(held_frame)
        for method in methods:
            weights = fit_specialist(
                feature_fit, in_fit, y[fit], logits[fit], method, FIT_SEED + fold,
                rank_weight=RANK_WEIGHT, l2=REGULARIZATION,
            )
            delta = np.zeros(len(held), dtype=np.float64)
            delta[in_held] = feature_held @ weights
            oof[method][held] = delta

    scores = {}
    for method, delta in oof.items():
        if not np.isfinite(delta).all() or not np.all(delta[~inside] == 0):
            raise ValueError("Development corrections are invalid or missing.")
        scores[method] = {scale: float(roc_auc_score(y, logits + scale * delta)) for scale in (0.0, *SCALES)}
    chosen = {method: max(SCALES, key=lambda scale: (choices[scale], -scale)) for method, choices in scores.items()}
    return chosen, scores


def main():
    from specialist_features import FEATURE_COLUMNS, GROUP_COLUMNS, SpecialistFeatures, hard_group_mask
    from specialist_ranker import fit_specialist

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced", action="store_true", help="Test all three affected pair blocks at their population weights.")
    args = parser.parse_args()
    methods = ("balanced",) if args.balanced else METHODS
    primary = "balanced" if args.balanced else "matched"
    results = RESULTS.with_name("hard_group_ranker_balanced") if args.balanced else RESULTS
    began = perf_counter()
    nb, _ = load_archive(results, smoke=False)
    validation = json.loads((VALIDATION / "metrics.json").read_text(encoding="utf-8"))
    train_path = ROOT.parents[1] / 'data' / "train.csv"
    train_hash = file_hash(train_path)
    if train_hash != validation["train_sha256"] or NOTEBOOK_SHA256 != validation["notebook_sha256"]:
        raise ValueError("The validation data or notebook has changed.")
    raw_fields = list(dict.fromkeys([*FEATURE_COLUMNS, *GROUP_COLUMNS]))
    train = pd.read_csv(train_path, usecols=["id", "is_fraud", *raw_fields])
    parent_oof, y_dev, parent_folds, scoring, ids, y_hold = own_validation_inputs(nb, train, validation)
    del scoring
    split = nb["StratifiedShuffleSplit"](
        n_splits=1, test_size=validation["holdout_fraction"], random_state=validation["holdout_seed"]
    )
    development_rows, holdout_rows = next(split.split(train["id"], train["is_fraud"]))
    dev = train.iloc[np.sort(development_rows)].reset_index(drop=True)
    hold = train.iloc[np.sort(holdout_rows)].reset_index(drop=True)
    np.testing.assert_array_equal(dev["is_fraud"], y_dev)
    np.testing.assert_array_equal(hold["id"], ids)
    np.testing.assert_array_equal(hold["is_fraud"], y_hold)
    dev_inside, hold_inside = hard_group_mask(dev), hard_group_mask(hold)
    reference = read_recorded_probabilities(VALIDATION / "holdout_predictions.csv", ids, "hierarchical_amount")
    np.testing.assert_allclose(roc_auc_score(y_hold, reference), validation["variants"]["hierarchical_amount"]["roc_auc"], atol=1e-12, rtol=0)

    print("Recomputing the current hierarchical stack's development OOF logits", flush=True)
    anchor = original_oof_logits(nb, parent_oof, y_dev, parent_folds, validation["baseline_meta_l2_by_fold"])
    del parent_oof
    scales, development_scores = choose_scales(dev[raw_fields], y_dev, dev_inside, anchor, SpecialistFeatures, fit_specialist, methods)
    dev_baseline_auc = float(roc_auc_score(y_dev, anchor))
    print(f"Development-selected scales {scales}", flush=True)
    fit_frame, hold_frame = dev.loc[dev_inside, raw_fields], hold.loc[hold_inside, raw_fields]
    encoder = SpecialistFeatures().fit(fit_frame)
    feat_fit, feat_hold = encoder.transform(fit_frame), encoder.transform(hold_frame)
    predictions = {"current_hierarchy": reference}
    for method in methods:
        print(f"Fitting {method} on all development rows", flush=True)
        weights = fit_specialist(feat_fit, dev_inside, y_dev, anchor, method, FIT_SEED,
                                 rank_weight=RANK_WEIGHT, l2=REGULARIZATION)
        predictions[method] = specialist_scores(reference, hold_inside, feat_hold @ weights, scales[method], nb["from_logit_10"])
        np.testing.assert_array_equal(predictions[method][~hold_inside], reference[~hold_inside])
    del feat_fit, feat_hold, fit_frame, hold_frame, dev, hold, anchor
    nb["gc"].collect()

    metrics = {name: score_groups(y_hold, values, hold_inside) for name, values in predictions.items()}
    for name in methods:
        interval = nb["delong_paired_test"](y_hold, predictions[name], reference)
        metrics[name].update({"delta_roc_auc": interval["diff"], "ci95_low": interval["ci_low"],
                              "ci95_high": interval["ci_high"], "p_value": interval["p_value"]})
    summary = {
        "train_sha256": train_hash,
        "notebook_sha256": NOTEBOOK_SHA256,
        "holdout_ids_sha256": validation["holdout_ids_sha256"],
        "development_rows": len(y_dev),
        "development_target_rows": int(dev_inside.sum()),
        "development_target_fraud": int(y_dev[dev_inside].sum()),
        "holdout_rows": len(y_hold),
        "holdout_target_rows": int(hold_inside.sum()),
        "primary_method": primary,
        "development_baseline_auc": dev_baseline_auc,
        "development_auc_by_scale": development_scores,
        "chosen_scales": scales,
        "development_preferred_scale_including_zero": {
            name: max(choices, key=lambda scale: (choices[scale], -scale))
            for name, choices in development_scores.items()
        },
        "rank_weight": RANK_WEIGHT,
        "l2": REGULARIZATION,
        "holdout": metrics,
    }
    path = results / "holdout_predictions.csv"
    temporary = path.with_suffix(".tmp.csv")
    pd.DataFrame({"id": ids, "is_fraud": y_hold, "hard_group": hold_inside, **predictions}).to_csv(temporary, index=False)
    temporary.replace(path)
    (results / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    primary_dev_auc = development_scores[primary][scales[primary]]
    if primary_dev_auc > dev_baseline_auc and metrics[primary]["delta_roc_auc"] > 1e-12:
        parent_manifest = json.loads((PARENT_SUBMISSION / "manifest.json").read_text(encoding="utf-8"))
        parent_file = PARENT_SUBMISSION / "submission.csv"
        test_path = ROOT.parents[1] / 'data' / "test.csv"
        if (parent_manifest["train_sha256"] != train_hash or parent_manifest["test_sha256"] != file_hash(test_path)
                or parent_manifest["submission_sha256"] != file_hash(parent_file)
                or parent_manifest["notebook_sha256"] != NOTEBOOK_SHA256):
            raise ValueError("Official train/test or the current submission changed.")
        test = pd.read_csv(test_path, usecols=["id", *raw_fields])
        test_ids = test["id"].to_numpy()
        official = pd.read_csv(ROOT.parents[1] / 'data' / "sample_submission.csv", usecols=["id"])["id"].to_numpy()
        if len(test_ids) != 299_984 or not np.array_equal(test_ids, official):
            raise ValueError("The full official test IDs must match the template.")
        test_reference = read_recorded_probabilities(parent_file, test_ids, "is_fraud")
        labels = train["is_fraud"].to_numpy(dtype=np.int8)
        train_inside, test_inside = hard_group_mask(train), hard_group_mask(test)
        print(f"Building the {primary} specialist for the official test set", flush=True)
        full_oof, full_folds, test_fold_matrices = own_full_inputs(nb, labels, parent_manifest)
        del test_fold_matrices
        full_anchor = original_oof_logits(nb, full_oof, labels, full_folds, parent_manifest["baseline_meta_l2_by_fold"])
        del full_oof
        full_features = SpecialistFeatures().fit(train.loc[train_inside, raw_fields])
        feat_train = full_features.transform(train.loc[train_inside, raw_fields])
        weights = fit_specialist(feat_train, train_inside, labels, full_anchor, primary, FIT_SEED,
                                 rank_weight=RANK_WEIGHT, l2=REGULARIZATION)
        del feat_train
        delta = full_features.transform(test.loc[test_inside, raw_fields]) @ weights
        scores = specialist_scores(test_reference, test_inside, delta, scales[primary], nb["from_logit_10"])
        np.testing.assert_array_equal(scores[~test_inside], test_reference[~test_inside])
        submission = results / "submission.csv"
        write_submission(submission, test_ids, scores, official)
        summary["submission"] = {"sha256": file_hash(submission), "rows": len(test_ids), "target_rows": int(test_inside.sum())}

    summary["elapsed_seconds"] = perf_counter() - began
    (results / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Verified specialist experiment outputs in {results}", flush=True)


if __name__ == "__main__":
    main()
