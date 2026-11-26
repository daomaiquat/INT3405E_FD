"""Evaluate ranking changes on held folds and attribute them to baseline score strata."""

from math import fsum

import numpy as np
from scipy.special import log_expit, logit
from sklearn.metrics import average_precision_score

from run_density_tree import pair_influence


BASELINE = "current_grandmaster"
MODELS = (BASELINE, "pointwise", "pairwise")
CONTRASTS = (("pointwise", BASELINE), ("pairwise", BASELINE), ("pairwise", "pointwise"))
LOW_SCORE_LOGIT = logit(0.02)


def evaluate_fold(labels, logits):
    """Measure raw held logits, crediting ties as half a correctly ordered pair."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Expected one-dimensional labels containing both binary classes.")
    if set(logits) != set(MODELS):
        raise ValueError(f"Expected logits for {', '.join(MODELS)}.")

    positive = labels == 1
    positive_count = int(positive.sum())
    negative_count = len(labels) - positive_count
    pairs = positive_count * negative_count
    metrics = {}
    doubled_credits = {}
    scores = {name: np.asarray(logits[name], dtype=np.float64) for name in MODELS}
    for name, values in scores.items():
        if values.shape != labels.shape or not np.isfinite(values).all():
            raise ValueError(f"Expected finite one-dimensional {name} logits aligned to labels.")
        positive_influence, _ = pair_influence(labels, values)
        # Recover integer units so every tie counts as exactly half a pair.
        credits = np.rint(positive_influence * (2 * negative_count)).astype(np.int64)
        doubled_credits[name] = credits
        losses = -log_expit(np.where(positive, values, -values))
        scale = max(1.0, losses.max())
        metrics[name] = {
            "roc_auc": float(credits.sum()) / (2 * pairs),
            "average_precision": float(average_precision_score(labels, values)),
            "log_loss": float((losses / scale).mean() * scale),
        }

    low_score = scores[BASELINE][positive] < LOW_SCORE_LOGIT
    low_count = int(low_score.sum())
    low_pairs = low_count * negative_count
    contrasts = {}
    for candidate, comparator in CONTRASTS:
        delta = doubled_credits[candidate] - doubled_credits[comparator]
        total_corrected = float(delta.sum()) / 2
        low_corrected = float(delta[low_score].sum()) / 2
        contrasts[f"{candidate}_vs_{comparator}"] = {
            "delta_roc_auc": total_corrected / pairs,
            "low_score_net_corrected_pairs": low_corrected,
            "other_net_corrected_pairs": total_corrected - low_corrected,
        }
    return {
        "metrics": metrics,
        "positive_negative_pairs": pairs,
        "low_score_positive_count": low_count,
        "low_score_pairs": low_pairs,
        "low_score_baseline_misordered_pairs": low_pairs - float(doubled_credits[BASELINE][low_score].sum()) / 2,
        "contrasts": contrasts,
    }


def summarize_folds(reports):
    """Aggregate over comparable within-fold pairs and count strict fold improvements."""
    reports = list(reports)
    if not reports:
        raise ValueError("Expected at least one held fold report.")

    totals = {
        field: sum(report[field] for report in reports)
        for field in (
            "positive_negative_pairs", "low_score_positive_count", "low_score_pairs",
            "low_score_baseline_misordered_pairs",
        )
    }
    pairs = totals["positive_negative_pairs"]
    aucs = {
        name: fsum(report["metrics"][name]["roc_auc"] * report["positive_negative_pairs"]
                   for report in reports) / pairs
        for name in MODELS
    }
    contrasts = {}
    for candidate, comparator in CONTRASTS:
        name = f"{candidate}_vs_{comparator}"
        rows = [report["contrasts"][name] for report in reports]
        low_corrected = sum(row["low_score_net_corrected_pairs"] for row in rows)
        other_corrected = sum(row["other_net_corrected_pairs"] for row in rows)
        contrasts[name] = {
            "pair_weighted_delta_roc_auc": (low_corrected + other_corrected) / pairs,
            "positive_folds": sum(row["delta_roc_auc"] > 0 for row in rows),
            "low_score_net_corrected_pairs": low_corrected,
            "other_net_corrected_pairs": other_corrected,
        }

    return {"fold_count": len(reports), **totals, "pair_weighted_roc_auc": aucs, "contrasts": contrasts}
