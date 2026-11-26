"""A two-regime logistic stacker for Grandmaster OOF logits."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit


WEIGHT_SHRINKAGE = 0.001


def _validate_matrix(matrix):
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite matrix with Sparse GLM and Dense GLM scores first.")
    return values


def _risk_anchor(matrix):
    return 0.5 * matrix[:, 0] + 0.5 * matrix[:, 1]


def _gate(anchor, lower, upper):
    if lower == upper:
        return np.full(anchor.shape, 0.5)
    bounded = np.clip(anchor, lower, upper)
    return (bounded - lower) / (upper - lower)


@dataclass(frozen=True)
class RiskGatedStacker:
    intercept: float
    low_weights: np.ndarray
    high_weights: np.ndarray
    lower: float
    upper: float

    def predict_logit(self, matrix):
        values = _validate_matrix(matrix)
        if values.shape[1] != self.low_weights.size:
            raise ValueError("Expected the same number of model inputs used to fit the stacker.")
        gate = _gate(_risk_anchor(values), self.lower, self.upper)
        return self.intercept + (1.0 - gate) * (values @ self.low_weights) + gate * (values @ self.high_weights)


def fit_gated_stacker(matrix, labels, l2, baseline_parameters):
    """Learn nonnegative weights with smooth interpolation between risk regimes."""
    values = _validate_matrix(matrix)
    labels = np.asarray(labels, dtype=np.float64)
    if labels.shape != (len(values),) or not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError("Expected one binary label per training row and both classes.")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("L2 regularization must be finite and nonnegative.")

    intercept, baseline_weights = baseline_parameters
    weights = np.asarray(baseline_weights, dtype=np.float64)
    features = values.shape[1]
    if weights.shape != (features,) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isfinite(intercept):
        raise ValueError("Baseline parameters must contain a finite intercept and aligned nonnegative weights.")

    anchor = _risk_anchor(values)
    lower, upper = np.quantile(anchor, [0.80, 0.97])
    gate = _gate(anchor, lower, upper)
    design = np.empty((len(values), 2 * features), dtype=np.float64)
    np.multiply(values, (1.0 - gate)[:, None], out=design[:, :features])
    np.multiply(values, gate[:, None], out=design[:, features:])
    signed_labels = 2.0 * labels - 1.0

    def loss_and_gradient(parameters):
        b, w = parameters[0], parameters[1:]
        difference = w[:features] - w[features:]
        logits = b + design @ w
        loss = -log_expit(signed_labels * logits).mean()
        loss += 0.25 * l2 * np.dot(w, w) + 0.5 * WEIGHT_SHRINKAGE * np.dot(difference, difference)

        residual = (expit(logits) - labels) / len(labels)
        gradient = np.empty_like(parameters)
        gradient[0] = residual.sum()
        gradient[1:] = design.T @ residual + 0.5 * l2 * w
        gradient[1:features + 1] += WEIGHT_SHRINKAGE * difference
        gradient[features + 1:] -= WEIGHT_SHRINKAGE * difference
        return float(loss), gradient

    fitted = minimize(
        loss_and_gradient,
        np.r_[intercept, weights, weights],
        jac=True,
        bounds=[(None, None)] + [(0.0, None)] * (2 * features),
        method="L-BFGS-B",
        options={"maxiter": 600, "maxls": 40, "ftol": 1e-11},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Risk-gated stacker fit failed: {fitted.message}")

    return RiskGatedStacker(
        float(fitted.x[0]), fitted.x[1:features + 1], fitted.x[features + 1:], float(lower), float(upper),
    )
