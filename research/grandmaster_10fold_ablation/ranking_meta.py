"""Fit a nonnegative stacker with logistic loss on sampled ranking pairs."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_rank_stacker(matrix, labels, l2, baseline_parameters, seed, *, rank_weight=0.03, pairs_per_positive=32):
    """Fit on development rows using a fixed sample of negatives per positive."""
    features = np.asarray(matrix, dtype=np.float64, order="C")
    targets = np.asarray(labels, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] == 0 or targets.shape != (features.shape[0],):
        raise ValueError("Expected a feature matrix and one label per row.")
    if not np.isfinite(features).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("Expected finite features and binary labels.")
    if not np.isfinite(l2) or l2 < 0 or not np.isfinite(rank_weight) or rank_weight < 0:
        raise ValueError("Loss weights must be finite and nonnegative.")
    if not isinstance(pairs_per_positive, (int, np.integer)) or pairs_per_positive < 1:
        raise ValueError("pairs_per_positive must be a positive integer.")

    positive = np.flatnonzero(targets == 1)
    negative = np.flatnonzero(targets == 0)
    if not positive.size or not negative.size:
        raise ValueError("Ranking requires both classes in the development labels.")

    intercept, baseline_weights = baseline_parameters
    baseline_weights = np.asarray(baseline_weights, dtype=np.float64)
    if (
        baseline_weights.shape != (features.shape[1],)
        or not np.isfinite(intercept)
        or not np.isfinite(baseline_weights).all()
        or (baseline_weights < 0).any()
    ):
        raise ValueError("Invalid starting intercept or nonnegative weights.")
    initial = np.concatenate(([intercept], baseline_weights))
    signed_targets = 2 * targets - 1

    pair_differences = None
    if rank_weight:
        random = np.random.default_rng(seed)
        sampled_negatives = random.choice(negative, size=(positive.size, pairs_per_positive))
        pairs = features[sampled_negatives]
        np.subtract(features[positive, None, :], pairs, out=pairs)
        pair_differences = pairs.reshape(-1, features.shape[1])

    def objective(parameters):
        weights = parameters[1:]
        negative_margins = -signed_targets * (parameters[0] + features @ weights)
        loss = np.logaddexp(0, negative_margins).mean() + 0.5 * l2 * (weights @ weights)
        residuals = -signed_targets * expit(negative_margins) / targets.size
        gradient = np.empty_like(parameters)
        gradient[0] = residuals.sum()
        gradient[1:] = features.T @ residuals + l2 * weights
        if pair_differences is not None:
            negative_pair_margins = -(pair_differences @ weights)
            loss += rank_weight * np.logaddexp(0, negative_pair_margins).mean()
            gradient[1:] -= (rank_weight / negative_pair_margins.size) * (
                pair_differences.T @ expit(negative_pair_margins)
            )
        return float(loss), gradient

    fitted = minimize(
        objective, initial, method="L-BFGS-B", jac=True,
        bounds=[(None, None)] + [(0.0, None)] * features.shape[1],
        options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-7, "maxls": 50},
    )
    if not np.isfinite(fitted.fun) or not np.isfinite(fitted.x).all() or not np.isfinite(fitted.jac).all():
        raise RuntimeError(f"Ranking stacker produced a nonfinite solution ({fitted.message}).")
    if not fitted.success:
        projected_gradient = fitted.jac.copy()
        at_lower_bound = fitted.x[1:] <= 1e-12
        projected_gradient[1:] = np.where(
            at_lower_bound, np.minimum(projected_gradient[1:], 0), projected_gradient[1:]
        )
        if np.max(np.abs(projected_gradient)) > 1e-6:
            raise RuntimeError(f"Ranking stacker did not converge ({fitted.message}).")
    return float(fitted.x[0]), fitted.x[1:]
