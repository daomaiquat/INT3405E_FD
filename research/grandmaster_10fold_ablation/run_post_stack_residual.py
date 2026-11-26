"""Test an interaction correction to the final Grandmaster logit."""

import json
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from export_hierarchical_submission import write_submission
from residual_postcalibration import fit_residual_calibrator
from run_ablation import NOTEBOOK_SHA256, ROOT, file_hash, load_archive
from run_meta_improvements import PARENT_SUBMISSION, VALIDATION, own_full_inputs, own_validation_inputs


RESULTS = ROOT / "post_stack_residual"
PENALTIES = (5.0, 25.0, 100.0)
FEATURES = ["velocity_score", "merchant_risk_score", "auth_method"]


def original_oof_logits(nb, matrix, labels, folds, l2s):
    raw = np.full(len(labels), np.nan)
    y = labels.astype(np.float64)
    for (_, fit, held), l2 in zip(folds, l2s, strict=True):
        intercept, weights = nb["_fit_nonneg_l2_weights"](matrix[fit], y[fit], l2)
        raw[held] = intercept + matrix[held] @ weights
    if not np.isfinite(raw).all():
        raise ValueError("Expected one finite OOF logit per development row.")
    return raw


def choose_penalty(frame, labels, offset, folds):
    tuning_scores = {penalty: np.full(len(labels), np.nan) for penalty in PENALTIES}
    for fold, fit, held in folds:
        print(f"Calibrator tuning fold {fold + 1}/{len(folds)}", flush=True)
        for penalty, output in tuning_scores.items():
            calibration = fit_residual_calibrator(frame.iloc[fit], offset[fit], labels[fit], penalty)
            output[held] = calibration.predict_logit(frame.iloc[held], offset[held])
    aucs = {penalty: float(roc_auc_score(labels, values)) for penalty, values in tuning_scores.items()}
    best = max(aucs, key=lambda penalty: (aucs[penalty], penalty))
    return best, aucs


def read_recorded_probabilities(path, ids, column):
    saved = pd.read_csv(path, usecols=["id", column])
    np.testing.assert_array_equal(saved["id"], ids)
    probabilities = saved[column].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or not ((0 < probabilities) & (probabilities < 1)).all():
        raise ValueError(f"Expected valid probabilities strictly between zero and one in {path}.")
    return probabilities


def main():
    started = perf_counter()
    nb, _ = load_archive(RESULTS, smoke=False)
    val_metrics = json.loads((VALIDATION / "metrics.json").read_text(encoding="utf-8"))
    train_path = ROOT.parents[1] / 'data' / "train.csv"
    train_hash = file_hash(train_path)
    if train_hash != val_metrics["train_sha256"] or NOTEBOOK_SHA256 != val_metrics["notebook_sha256"]:
        raise ValueError("Validated training data or archived notebook has changed.")
    train = pd.read_csv(train_path, usecols=["id", "is_fraud", *FEATURES])
    matrix, y_dev, folds, scoring_folds, hold_ids, y_hold = own_validation_inputs(nb, train, val_metrics)
    del scoring_folds
    splitter = nb["StratifiedShuffleSplit"](
        n_splits=1, test_size=val_metrics["holdout_fraction"], random_state=val_metrics["holdout_seed"]
    )
    dev_index, hold_index = next(splitter.split(train["id"], train["is_fraud"]))
    dev = train.iloc[np.sort(dev_index)].reset_index(drop=True)
    hold = train.iloc[np.sort(hold_index)].reset_index(drop=True)
    np.testing.assert_array_equal(dev["is_fraud"], y_dev)
    np.testing.assert_array_equal(hold["id"], hold_ids)
    np.testing.assert_array_equal(hold["is_fraud"], y_hold)
    reference = read_recorded_probabilities(VALIDATION / "holdout_predictions.csv", hold_ids, "hierarchical_amount")
    np.testing.assert_allclose(roc_auc_score(y_hold, reference), val_metrics["variants"]["hierarchical_amount"]["roc_auc"], atol=1e-12, rtol=0)

    print("Rebuilding development-only OOF logits of the current stack", flush=True)
    offset = original_oof_logits(nb, matrix, y_dev, folds, val_metrics["baseline_meta_l2_by_fold"])
    del matrix
    penalty, tuning = choose_penalty(dev[FEATURES], y_dev, offset, folds)
    print(f"Development-only selection chose L2 {penalty}", flush=True)
    fitted = fit_residual_calibrator(dev[FEATURES], offset, y_dev, penalty)
    probabilities = nb["from_logit_10"](fitted.predict_logit(hold[FEATURES], logit(reference)))
    comparison = nb["delong_paired_test"](y_hold, probabilities, reference)
    focus = hold["auth_method"].eq("OTP") & hold["velocity_score"].between(22.1, 24.0, inclusive="left")
    report = {
        "train_sha256": train_hash,
        "notebook_sha256": NOTEBOOK_SHA256,
        "holdout_ids_sha256": val_metrics["holdout_ids_sha256"],
        "holdout_rows": len(hold_ids),
        "development_rows": len(y_dev),
        "selected_l2": penalty,
        "development_baseline_roc_auc": float(roc_auc_score(y_dev, offset)),
        "development_l2_roc_auc": tuning,
        "holdout_baseline_roc_auc": float(roc_auc_score(y_hold, reference)),
        "holdout_roc_auc": float(roc_auc_score(y_hold, probabilities)),
        "holdout_baseline_average_precision": float(average_precision_score(y_hold, reference)),
        "holdout_average_precision": float(average_precision_score(y_hold, probabilities)),
        "holdout_baseline_log_loss": float(log_loss(y_hold, reference)),
        "holdout_log_loss": float(log_loss(y_hold, probabilities)),
        "delta_roc_auc": comparison["diff"],
        "ci95_low": comparison["ci_low"],
        "ci95_high": comparison["ci_high"],
        "p_value": comparison["p_value"],
        "otp_group": {"rows": int(focus.sum()), "fraud": int(y_hold[focus].sum()),
                      "baseline_predicted_fraud": float(reference[focus].sum()),
                      "corrected_predicted_fraud": float(probabilities[focus].sum())},
    }
    output = RESULTS / "holdout_predictions.csv"
    temporary = output.with_suffix(".tmp.csv")
    pd.DataFrame({"id": hold_ids, "is_fraud": y_hold, "current_hierarchy": reference,
                  "post_stack_residual": probabilities}).to_csv(temporary, index=False)
    temporary.replace(output)
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    del dev, hold, offset
    nb["gc"].collect()

    if comparison["diff"] > 1e-12:
        origin = json.loads((PARENT_SUBMISSION / "manifest.json").read_text(encoding="utf-8"))
        parent_path = PARENT_SUBMISSION / "submission.csv"
        test_path = ROOT.parents[1] / 'data' / "test.csv"
        if (origin["train_sha256"] != train_hash or origin["test_sha256"] != file_hash(test_path)
                or origin["notebook_sha256"] != NOTEBOOK_SHA256 or origin["submission_sha256"] != file_hash(parent_path)):
            raise ValueError("Official data or the current submission has changed.")
        test = pd.read_csv(test_path, usecols=["id", *FEATURES])
        ids = test["id"].to_numpy()
        template = pd.read_csv(ROOT.parents[1] / 'data' / "sample_submission.csv", usecols=["id"])["id"].to_numpy()
        if len(ids) != 299_984 or not np.array_equal(ids, template):
            raise ValueError("Expected the complete official test ID sequence.")
        original = read_recorded_probabilities(parent_path, ids, "is_fraud")
        full_y = train["is_fraud"].to_numpy(dtype=np.int8)
        print("Generating an experimental submission with the selected correction", flush=True)
        matrix, full_folds, test_folds = own_full_inputs(nb, full_y, origin)
        del test_folds
        full_offset = original_oof_logits(nb, matrix, full_y, full_folds, origin["baseline_meta_l2_by_fold"])
        correction = fit_residual_calibrator(train[FEATURES], full_offset, full_y, penalty)
        test_probabilities = nb["from_logit_10"](correction.predict_logit(test[FEATURES], logit(original)))
        destination = RESULTS / "submission.csv"
        write_submission(destination, ids, test_probabilities, template)
        report["submission"] = {"rows": len(ids), "sha256": file_hash(destination)}

    report["elapsed_seconds"] = perf_counter() - started
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Verified residual correction outputs in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
