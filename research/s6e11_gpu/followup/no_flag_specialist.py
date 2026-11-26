"""Regularized merchant-risk correction for transactions with six clear flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit, logit


FLAG_COLUMNS = (
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
)
PRIOR_SCALE = 0.10
CATEGORY_SLOPE_PRIOR_SCALE = 0.05


def _probabilities(values, rows):
    probabilities = np.asarray(values, dtype=np.float64)
    if (
        probabilities.shape != (rows,)
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("Expected one finite probability in [0, 1] per row")
    return probabilities


def _no_flags(frame):
    return frame.loc[:, FLAG_COLUMNS].eq(0).fillna(False).all(axis=1).to_numpy(dtype=bool)


def _merchant_features(frame, categories):
    risk = pd.to_numeric(frame["merchant_risk_score"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan,
    )
    risk = np.clip((np.where(np.isfinite(risk), risk, 50.0) - 50.0) / 25.0, -2.0, 2.0)
    codes = pd.Categorical(frame["merchant_category"], categories=categories).codes.astype(np.intp)
    rows = np.arange(len(frame))
    known = codes >= 0
    known_rows = rows[known]
    known_codes = codes[known]
    category_count = len(categories)
    return csr_matrix(
        (
            np.concatenate((np.ones(len(rows)), risk, np.ones(len(known_rows)), risk[known])),
            (
                np.concatenate((rows, rows, known_rows, known_rows)),
                np.concatenate((
                    np.zeros_like(rows), np.ones_like(rows),
                    2 + known_codes, 2 + category_count + known_codes,
                )),
            ),
        ),
        shape=(len(rows), 2 + 2 * category_count),
    )


def _offset(probabilities):
    eps = np.finfo(np.float64).eps
    return logit(np.clip(probabilities, eps, 1.0 - eps))


def fit_adjust_no_flags(
    train_frame: pd.DataFrame,
    train_probability: np.ndarray,
    query_frame: pd.DataFrame,
    query_probability: np.ndarray,
) -> np.ndarray:
    """Fit on inner OOF outer-train predictions and adjust the supplied queries.

    The baseline logit is a fixed offset. Ridge penalties shrink the global
    intercept, merchant risk slope, category intercepts and category slopes
    toward zero. Query categories absent from training use the global terms.
    """
    train_probability = _probabilities(train_probability, len(train_frame))
    adjusted = _probabilities(query_probability, len(query_frame)).copy()
    query_mask = _no_flags(query_frame)
    train_mask = _no_flags(train_frame)
    if not query_mask.any() or not train_mask.any():
        return adjusted

    train = train_frame.loc[train_mask]
    labels = train["is_fraud"].to_numpy(dtype=np.float64)
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("No-flag training labels must be binary")
    if np.all(labels == labels[0]):
        return adjusted

    categories = pd.Index(train["merchant_category"].dropna().unique())
    features = _merchant_features(train, categories)
    offset = _offset(train_probability[train_mask])
    precision = np.full(features.shape[1], PRIOR_SCALE ** -2)
    precision[2 + len(categories):] = CATEGORY_SLOPE_PRIOR_SCALE ** -2

    def objective(weights):
        corrected_logit = offset + features @ weights
        penalty = precision * weights
        loss = np.logaddexp(0.0, np.where(labels == 1, -corrected_logit, corrected_logit)).sum()
        gradient = features.T @ (expit(corrected_logit) - labels) + penalty
        return loss + 0.5 * (weights @ penalty), gradient

    fit = minimize(
        objective, np.zeros(features.shape[1]), jac=True, method="L-BFGS-B",
        options={"maxiter": 100, "ftol": 1e-12},
    )
    if not fit.success:
        raise RuntimeError(f"No-flag specialist optimizer failed: {fit.message}")

    query_features = _merchant_features(query_frame.loc[query_mask], categories)
    adjusted[query_mask] = expit(_offset(adjusted[query_mask]) + query_features @ fit.x)
    return adjusted
