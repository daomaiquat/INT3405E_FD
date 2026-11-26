"""Evaluate one-branch Grandmaster changes on a common labeled holdout."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def _score_holdout(labels, probabilities, otp_mask):
    otp_labels = labels[otp_mask]
    otp_probabilities = probabilities[otp_mask]
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "otp": {
            "n": int(len(otp_labels)),
            "fraud": int(otp_labels.sum()),
            "predicted_fraud": float(otp_probabilities.sum()),
            "log_loss": float(log_loss(otp_labels, otp_probabilities, labels=[0, 1]))
            if len(otp_labels) else None,
        },
    }


def evaluate_variants(
    namespace, baseline_oof, baseline_holdout, candidate_oof, candidate_holdout,
    y_dev, folds, y_holdout, otp_mask,
) -> tuple[dict, dict]:
    """Learn meta weights on development OOF scores and compare on holdout.

    The baseline receives raw per-branch logits and follows the notebook's
    stacker. Candidate matrices are already residual-decomposed, so unchanged
    branches retain their original scores and offsets.
    """
    y_dev = np.asarray(y_dev, dtype=np.float64)
    y_holdout = np.asarray(y_holdout, dtype=np.int8)
    otp_mask = np.asarray(otp_mask, dtype=bool)
    model_names = namespace["LOCKED_MODELS"]
    if y_dev.ndim != 1 or y_holdout.ndim != 1 or y_holdout.shape != otp_mask.shape:
        raise ValueError("Expected one-dimensional labels and a matching holdout subgroup mask.")
    if not folds:
        raise ValueError("Expected at least one development fold.")
    if candidate_oof.keys() != candidate_holdout.keys() or "baseline" in candidate_oof:
        raise ValueError("Development and holdout candidates must match and exclude the baseline.")
    for split, scores, labels in (
        ("development", baseline_oof, y_dev), ("holdout", baseline_holdout, y_holdout),
    ):
        for name in model_names:
            values = np.asarray(scores[name])
            if values.shape != labels.shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid baseline {split} logits for {name}.")

    prepared = {}
    for name, scores in candidate_oof.items():
        oof = np.asarray(scores, dtype=np.float64)
        holdout = np.asarray(candidate_holdout[name], dtype=np.float64)
        if oof.shape != (len(y_dev), len(model_names)) or not np.isfinite(oof).all():
            raise ValueError(f"Invalid candidate development scores for {name}.")
        if holdout.shape != (len(y_holdout), len(model_names)) or not np.isfinite(holdout).all():
            raise ValueError(f"Invalid candidate holdout scores for {name}.")
        prepared[name] = (oof, holdout)

    stack = namespace["run_stage4_stacking"](
        baseline_oof, baseline_holdout, y_dev, folds,
        meta_inner_folds=namespace["META_INNER_FOLDS"],
    )
    l2s = [float(l2) for l2 in stack["l2s"]]
    baseline_probabilities = stack["final_te"]
    probabilities = {"baseline": baseline_probabilities}
    metrics = {
        "baseline": _score_holdout(y_holdout, baseline_probabilities, otp_mask),
        "variants": {},
        "baseline_meta_l2_by_fold": l2s,
    }

    fit_weights = namespace["_fit_nonneg_l2_weights"]
    for name, (oof, holdout) in prepared.items():
        holdout_logits = np.zeros(len(y_holdout), dtype=np.float64)
        for (_, train_rows, _), l2 in zip(folds, l2s, strict=True):
            intercept, weights = fit_weights(oof[train_rows], y_dev[train_rows], l2)
            holdout_logits += (intercept + holdout @ weights) / len(folds)
        predictions = namespace["from_logit_10"](holdout_logits)
        comparison = namespace["delong_paired_test"](y_holdout, predictions, baseline_probabilities)
        result = _score_holdout(y_holdout, predictions, otp_mask)
        result.update({
            "delta_roc_auc": comparison["diff"],
            "ci95_low": comparison["ci_low"],
            "ci95_high": comparison["ci_high"],
            "p_value": comparison["p_value"],
        })
        metrics["variants"][name] = result
        probabilities[name] = predictions
    return metrics, probabilities
