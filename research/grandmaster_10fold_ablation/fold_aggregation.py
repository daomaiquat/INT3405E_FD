"""Combine Grandmaster fold predictions after residual decomposition."""

import numpy as np


def aggregate_fold_scores(parameters, fold_matrices, link):
    """Return probabilities from three predeclared ways of combining folds."""
    count = len(parameters)
    if not count or count != len(fold_matrices):
        raise ValueError("Expected one parameter set per scoring fold.")

    intercepts = np.asarray([intercept for intercept, _ in parameters], dtype=np.float64)
    weights = np.asarray([coefficients for _, coefficients in parameters], dtype=np.float64)
    if intercepts.shape != (count,) or weights.ndim != 2:
        raise ValueError("Each fold requires a scalar intercept and a vector of feature weights.")
    shape = np.shape(fold_matrices[0])
    if len(shape) != 2 or shape[1] != weights.shape[1]:
        raise ValueError("The scoring matrices must match the number of feature weights.")

    mean_b = float(intercepts.mean())
    mean_w = weights.mean(axis=0)
    mean_matrix = np.zeros(shape, dtype=np.float64)
    paired_logits = np.zeros(shape[0], dtype=np.float64)
    mean_probabilities = np.zeros(shape[0], dtype=np.float64)
    for intercept, coefficients, fold_matrix in zip(intercepts, weights, fold_matrices, strict=True):
        matrix = np.asarray(fold_matrix)
        if matrix.shape != shape:
            raise ValueError("Every scoring fold must have the same rows and feature columns.")
        mean_matrix += matrix
        paired_logits += intercept + matrix @ coefficients
        mean_probabilities += link(mean_b + matrix @ mean_w)

    return {
        "mean_logits": link(mean_b + (mean_matrix / count) @ mean_w),
        "paired_logits": link(paired_logits / count),
        "mean_probabilities": mean_probabilities / count,
    }
