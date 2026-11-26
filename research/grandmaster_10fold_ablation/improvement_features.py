"""Fold-local features for the Grandmaster 10-fold experiments."""

import numpy as np


def _group_indices(reference_keys, scoring_keys):
    keys, reference_indices = np.unique(reference_keys, return_inverse=True)
    scoring_indices = np.searchsorted(keys, scoring_keys)
    present_rows = np.flatnonzero(scoring_indices < keys.size)
    present_rows = present_rows[keys[scoring_indices[present_rows]] == scoring_keys[present_rows]]
    return reference_indices, present_rows, scoring_indices[present_rows]


def _labeled_counts(reference_keys, labels, scoring_keys):
    groups, rows, target = _group_indices(reference_keys, scoring_keys)
    counts = np.zeros(scoring_keys.size, dtype=np.int64)
    positives = np.zeros(scoring_keys.size, dtype=np.float64)
    counts[rows] = np.bincount(groups)[target]
    positives[rows] = np.bincount(groups, weights=labels)[target]
    return counts, positives


def amount_parent_feature(reference_keys, labels, scoring_keys):
    """Encode rare exact amounts against the other amounts in the same tenth dollar."""
    reference_keys = np.asarray(reference_keys)
    labels = np.asarray(labels, dtype=np.float64)
    scoring_keys = np.asarray(scoring_keys)
    if reference_keys.ndim != 1 or labels.shape != reference_keys.shape or scoring_keys.ndim != 1:
        raise ValueError("Expected one-dimensional amount keys and aligned reference labels.")

    feature = np.zeros(scoring_keys.size, dtype=np.float32)
    if not reference_keys.size:
        return feature

    child_count, child_positive = _labeled_counts(reference_keys, labels, scoring_keys)
    parent_count, parent_positive = _labeled_counts(reference_keys // 10, labels, scoring_keys // 10)
    parent_count -= child_count
    parent_positive -= child_positive
    eligible = (child_count < 4) & (parent_count >= 4)

    prior = np.clip(labels.mean(), 1e-6, 1.0 - 1e-6)
    parent_rate = (parent_positive[eligible] + 50 * prior) / (parent_count[eligible] + 50)
    child_rate = (child_positive[eligible] + 10 * parent_rate) / (child_count[eligible] + 10)
    child_rate = np.clip(child_rate, 1e-6, 1.0 - 1e-6)
    feature[eligible] = np.log(child_rate / (1.0 - child_rate)) - np.log(prior / (1.0 - prior))
    return feature


def residual_pair_features(reference_groups, labels, base_probabilities, scoring_groups):
    """Map each pair to its regularized logit adjustment relative to the base model."""
    reference_groups = np.asarray(reference_groups)
    scoring_groups = np.asarray(scoring_groups)
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(base_probabilities, dtype=np.float64)
    if (
        reference_groups.ndim != 2
        or scoring_groups.ndim != 2
        or reference_groups.shape[1] != scoring_groups.shape[1]
        or labels.shape != (reference_groups.shape[0],)
        or probabilities.shape != labels.shape
    ):
        raise ValueError("Expected matching group columns and aligned reference labels and probabilities.")

    features = np.zeros(scoring_groups.shape, dtype=np.float32)
    residuals = labels - probabilities
    curvature = probabilities * (1.0 - probabilities)
    for column in range(reference_groups.shape[1]):
        groups, rows, target = _group_indices(reference_groups[:, column], scoring_groups[:, column])
        numerator = np.bincount(groups, weights=residuals)
        denominator = np.bincount(groups, weights=curvature) + 5.0
        features[rows, column] = np.clip(numerator[target] / denominator[target], -1.0, 1.0)
    return features
