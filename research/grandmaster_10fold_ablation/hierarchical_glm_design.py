"""Fold-local sparse design for the hierarchical logistic regression."""

import numpy as np
import pandas as pd
from scipy import sparse


_NUMERIC = (
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
)
_CATEGORIES = ("merchant_category", "card_type", "auth_method", "channel", "device_type")
_FLAGS = (
    "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
    "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
)
_LOG_SMOOTH = {"amount_usd", "account_balance_usd", "hours_since_last_txn", "distance_from_home_km"}
_MAIN_QUANTILES = (0.005, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 0.995)
_INTERACTION_QUANTILES = (0.02, 0.25, 0.50, 0.75, 0.98)
_INTERACTION_SMOOTH = ("amount_usd", "velocity_score", "merchant_risk_score")

# A scale s gives the raw effect L2 penalty proportional to 1/s**2. The
# amount prefixes are simultaneous terms, with stronger shrinkage deeper in
# the tree. Each level needs its own support from the fold's fitting rows.
_AMOUNT_LEVELS = (
    (1000, 2, 1.50),  # ten dollars
    (100, 3, 1.00),   # dollar
    (10, 5, 0.60),    # dime
    (1, 8, 0.35),     # cent
)
_OTHER_EXACT = {
    "velocity_score": (10, 5, 1.15),
    "merchant_risk_score": (10, 5, 1.00),
    "customer_age": (1, 5, 0.80),
    "hours_since_last_txn": (100, 8, 0.65),
    "account_balance_usd": (100, 8, 0.45),
    "txn_count_last_24h": (1, 5, 0.80),
    "cvv_retry_count": (1, 5, 0.80),
}


def _integer_keys(frame, column, units):
    return np.rint(frame[column].to_numpy(dtype=np.float64) * units).astype(np.int64)


def _supported_values(keys, minimum):
    levels, counts = np.unique(keys, return_counts=True)
    return pd.Index(levels[counts >= minimum])


def _indicator(keys, levels, scale):
    codes = levels.get_indexer(keys)
    known = codes >= 0
    return sparse.csr_matrix(
        (np.full(known.sum(), scale, dtype=np.float32), (np.flatnonzero(known), codes[known])),
        shape=(len(keys), len(levels)),
    )


def _smooth_values(frame, column):
    values = frame[column].to_numpy(dtype=np.float64)
    return np.log1p(values) if column in _LOG_SMOOTH else values


def _spline(values, knots):
    """Linear interpolation onto adjacent knots, constant beyond the end knots."""
    n_rows = len(values)
    if len(knots) == 1:
        return sparse.csr_matrix(np.ones((n_rows, 1), dtype=np.float32))

    clipped = np.clip(values, knots[0], knots[-1])
    right = np.clip(np.searchsorted(knots, clipped, side="right"), 1, len(knots) - 1)
    left = right - 1
    right_weight = (clipped - knots[left]) / (knots[right] - knots[left])
    data = np.column_stack((1 - right_weight, right_weight)).astype(np.float32)
    indices = np.column_stack((left, right)).astype(np.int32)
    indptr = np.arange(0, 2 * n_rows + 1, 2, dtype=np.int32)
    result = sparse.csr_matrix((data.ravel(), indices.ravel(), indptr), shape=(n_rows, len(knots)))
    result.eliminate_zeros()
    return result


def _category_contrasts(counts):
    """Adjacent contrasts have zero frequency-weighted mean and at most two entries per row."""
    columns = np.arange(len(counts) - 1, dtype=np.int32)
    lower = -counts[:-1] / counts[1:]
    weights = np.concatenate((np.ones(len(columns)), lower)).astype(np.float32)
    # The all-zero first row represents unseen categories.
    rows = np.concatenate((columns + 1, columns + 2))
    return sparse.csr_matrix(
        (weights, (rows, np.tile(columns, 2))), shape=(len(counts) + 1, len(columns)),
    )


def _row_products(left, right, scale):
    """Multiply only active same-row feature pairs, without a dense tensor product."""
    left_counts = np.diff(left.indptr)
    right_counts = np.diff(right.indptr)
    counts = left_counts * right_counts
    indptr = np.empty(len(counts) + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    rows = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    positions = np.arange(indptr[-1], dtype=np.int32) - indptr[rows]
    right_widths = right_counts[rows]
    left_entry = left.indptr[rows] + positions // right_widths
    right_entry = right.indptr[rows] + positions % right_widths
    indices = left.indices[left_entry] * right.shape[1] + right.indices[right_entry]
    data = scale * (left.data[left_entry] * right.data[right_entry])
    return sparse.csr_matrix(
        (data, indices, indptr), shape=(left.shape[0], left.shape[1] * right.shape[1]),
    )


class HierarchicalGLMDesign:
    """Fit raw-column vocabularies and knots, then build aligned float32 CSR matrices.

    Scoring an unsupported amount activates its supported parents and its
    global spline. Interactions append to the identical additive matrix.
    """

    def fit(self, frame):
        if frame.empty:
            raise ValueError("The hierarchical GLM design requires fitting rows.")

        amounts = _integer_keys(frame, "amount_usd", 100)
        self.amount_levels_ = [
            _supported_values(amounts // divisor, minimum)
            for divisor, minimum, _ in _AMOUNT_LEVELS
        ]
        self.exact_levels_ = {
            column: _supported_values(_integer_keys(frame, column, units), minimum)
            for column, (units, minimum, _) in _OTHER_EXACT.items()
        }
        self.category_levels_ = {}
        self.category_contrasts_ = {}
        for column in _CATEGORIES:
            levels, counts = np.unique(frame[column].to_numpy(), return_counts=True)
            order = np.argsort(counts, kind="stable")
            self.category_levels_[column] = pd.Index(levels[order])
            if column in ("merchant_category", "auth_method"):
                self.category_contrasts_[column] = _category_contrasts(counts[order])

        self.main_knots_ = {}
        self.interaction_knots_ = {}
        for column in _NUMERIC:
            values = _smooth_values(frame, column)
            self.main_knots_[column] = np.unique(np.quantile(values, _MAIN_QUANTILES))
            if column in _INTERACTION_SMOOTH:
                quantiles = np.quantile(values, _INTERACTION_QUANTILES)
                knots = np.unique(quantiles)
                center = np.searchsorted(knots, quantiles[2])
                self.interaction_knots_[column] = (knots, np.arange(len(knots)) != center)
        return self

    def transform(self, frame, *, interactions=False):
        amounts = _integer_keys(frame, "amount_usd", 100)
        blocks = [
            _indicator(amounts // divisor, levels, scale)
            for (divisor, _, scale), levels in zip(_AMOUNT_LEVELS, self.amount_levels_, strict=True)
        ]
        blocks.extend(
            _indicator(_integer_keys(frame, column, units), self.exact_levels_[column], scale)
            for column, (units, _, scale) in _OTHER_EXACT.items()
        )
        blocks.extend(
            _spline(_smooth_values(frame, column), self.main_knots_[column]) for column in _NUMERIC
        )
        blocks.extend(
            _indicator(frame[column], self.category_levels_[column], 1.0) for column in _CATEGORIES
        )
        blocks.append(sparse.csr_matrix(frame.loc[:, list(_FLAGS)].to_numpy(dtype=np.float32)))

        if interactions:
            centered = {}
            for column, (knots, keep) in self.interaction_knots_.items():
                # The spline basis sums to one. Removing the median column
                # after subtracting its unit vector leaves these other columns.
                centered[column] = _spline(_smooth_values(frame, column), knots)[:, keep]
            contrasts = {
                column: lookup[self.category_levels_[column].get_indexer(frame[column]) + 1]
                for column, lookup in self.category_contrasts_.items()
            }
            blocks.extend((
                _row_products(centered["amount_usd"], contrasts["merchant_category"], 0.45),
                _row_products(centered["velocity_score"], contrasts["auth_method"], 0.45),
                _row_products(centered["velocity_score"], centered["merchant_risk_score"], 0.30),
            ))

        return sparse.hstack(blocks, format="csr")
