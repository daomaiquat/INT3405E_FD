"""Fit signed group corrections to frozen Grandmaster logits."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit


QUANTILES = np.arange(1, 8) / 8.0


def _read_inputs(frame, base_logits):
    velocity = frame["velocity_score"].to_numpy(dtype=np.float64)
    risk = frame["merchant_risk_score"].to_numpy(dtype=np.float64)
    auth_column = frame["auth_method"]
    auth = auth_column.astype("string").fillna("").to_numpy(dtype=str)
    has_auth = auth_column.notna().to_numpy()
    base = np.asarray(base_logits, dtype=np.float64)
    if base.shape != velocity.shape or not np.isfinite(base).all():
        raise ValueError("Expected one finite base logit per row.")
    if not np.isfinite(velocity).all() or not np.isfinite(risk).all():
        raise ValueError("Risk and velocity scores must be finite.")
    return velocity, risk, auth, has_auth, base


def _group_ids(velocity, risk, auth, has_auth, velocity_edges, risk_edges, auth_values):
    velocity_bins = np.searchsorted(velocity_edges, velocity, side="right")
    risk_bins = np.searchsorted(risk_edges, risk, side="right")
    risk_ids = risk_bins * (len(velocity_edges) + 1) + velocity_bins

    auth_ids = np.full(len(velocity), -1, dtype=np.intp)
    if len(auth_values):
        category = np.searchsorted(auth_values, auth)
        safe_category = np.minimum(category, len(auth_values) - 1)
        known = has_auth & (auth_values[safe_category] == auth)
        auth_ids[known] = velocity_bins[known] * len(auth_values) + safe_category[known]
    return auth_ids, risk_ids


@dataclass(frozen=True)
class ResidualCalibrator:
    velocity_edges: np.ndarray
    risk_edges: np.ndarray
    auth_values: np.ndarray
    auth_adjustments: np.ndarray
    risk_adjustments: np.ndarray

    def predict_logit(self, frame, base_logits):
        velocity, risk, auth, has_auth, base = _read_inputs(frame, base_logits)
        auth_ids, risk_ids = _group_ids(
            velocity, risk, auth, has_auth, self.velocity_edges, self.risk_edges, self.auth_values,
        )
        logits = base + self.risk_adjustments[risk_ids]
        known_auth = auth_ids >= 0
        logits[known_auth] += self.auth_adjustments[auth_ids[known_auth]]
        return logits


def fit_residual_calibrator(frame, base_logits, labels, l2):
    """Fit two categorical residual tables with a fixed base logit offset."""
    velocity, risk, auth, has_auth, base = _read_inputs(frame, base_logits)
    labels = np.asarray(labels, dtype=np.float64)
    if not len(base) or labels.shape != base.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected one binary label per training row and at least one row.")
    if not np.isfinite(l2) or l2 <= 0:
        raise ValueError("Expected a finite positive L2 penalty.")

    velocity_edges = np.unique(np.quantile(velocity, QUANTILES))
    risk_edges = np.unique(np.quantile(risk, QUANTILES))
    auth_values = np.unique(auth[has_auth])
    auth_ids, risk_ids = _group_ids(
        velocity, risk, auth, has_auth, velocity_edges, risk_edges, auth_values,
    )
    auth_active = auth_ids >= 0
    auth_ids = auth_ids[auth_active]
    n_auth_groups = (len(velocity_edges) + 1) * len(auth_values)
    n_risk_groups = (len(risk_edges) + 1) * (len(velocity_edges) + 1)
    signs = 2.0 * labels - 1.0

    def group_sums(weights):
        return np.r_[
            np.bincount(auth_ids, weights=weights[auth_active], minlength=n_auth_groups),
            np.bincount(risk_ids, weights=weights, minlength=n_risk_groups),
        ]

    # Diagonal scaling from curvature at the offset keeps group sizes with
    # different frequencies comparable for the optimizer.
    probability = expit(base)
    scale = 1.0 / np.sqrt(l2 + group_sums(probability * (1.0 - probability)))

    def objective(parameters):
        adjustments = scale * parameters
        logits = base + adjustments[n_auth_groups + risk_ids]
        logits[auth_active] += adjustments[auth_ids]
        loss = -log_expit(signs * logits).sum() + 0.5 * l2 * np.dot(adjustments, adjustments)
        gradient = group_sums(expit(logits) - labels) + l2 * adjustments
        return float(loss), scale * gradient

    fitted = minimize(
        objective, np.zeros(n_auth_groups + n_risk_groups), jac=True,
        method="L-BFGS-B", options={"maxiter": 600, "maxls": 40, "ftol": 1e-12},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Residual calibration fit failed: {fitted.message}")

    observed = np.r_[
        np.bincount(auth_ids, minlength=n_auth_groups),
        np.bincount(risk_ids, minlength=n_risk_groups),
    ] > 0
    adjustments = np.where(observed, scale * fitted.x, 0.0)
    return ResidualCalibrator(
        velocity_edges, risk_edges, auth_values,
        adjustments[:n_auth_groups], adjustments[n_auth_groups:],
    )
