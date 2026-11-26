"""Train-fitted features for the Grandmaster hard-group specialist."""

import numpy as np
import pandas as pd


GROUP_COLUMNS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
    "cvv_retry_count",
]
_CATEGORICAL_COLUMNS = ["auth_method", "channel", "merchant_category", "device_type"]
FEATURE_COLUMNS = [
    "velocity_score",
    "merchant_risk_score",
    "hours_since_last_txn",
    "amount_usd",
    "account_balance_usd",
    "txn_count_last_24h",
    "prior_disputes",
    *_CATEGORICAL_COLUMNS,
]
_KNOTS = np.array([0, 1 / 3, 2 / 3], dtype=np.float32)
# Positions in _numeric_values: velocity by merchant risk and by elapsed time.
_INTERACTIONS = ((0, 1), (0, 2))


def hard_group_mask(frame):
    """Select rows where each risk flag is false and CVV retries are zero."""
    no_flags = frame[GROUP_COLUMNS[:-1]].eq(False).fillna(False).all(axis=1)
    no_retries = frame["cvv_retry_count"].eq(0).fillna(False)
    return (no_flags & no_retries).to_numpy(dtype=bool)


def _numeric_values(frame):
    amount = frame["amount_usd"].to_numpy(dtype=np.float64)
    balance = frame["account_balance_usd"].to_numpy(dtype=np.float64)
    if not np.isfinite(balance).all() or (balance <= -1).any():
        raise ValueError("Account balances must be finite with a positive ratio denominator.")
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.column_stack((
            frame["velocity_score"].to_numpy(dtype=np.float64),
            frame["merchant_risk_score"].to_numpy(dtype=np.float64),
            np.log1p(frame["hours_since_last_txn"].to_numpy(dtype=np.float64)),
            np.log1p(amount / (1.0 + balance)),
            np.log1p(amount),
            frame["txn_count_last_24h"].to_numpy(dtype=np.float64),
            frame["prior_disputes"].to_numpy(dtype=np.float64),
        ))
    if not np.isfinite(values).all():
        raise ValueError("Specialist numeric features must be finite after the log transforms.")
    return values


class SpecialistFeatures:
    """Piecewise-linear and categorical features fitted on gated rows."""

    def fit(self, frame):
        if frame.empty:
            raise ValueError("At least one gated training row is required.")
        values = _numeric_values(frame)
        low, high = np.quantile(values, [0.01, 0.99], axis=0)
        # Preserve sparse count signals when both quantiles are tied.
        tied = low == high
        low[tied] = values[:, tied].min(axis=0)
        high[tied] = values[:, tied].max(axis=0)
        span = high - low
        self._lower = low
        self._inverse_span = np.divide(1.0, span, out=np.zeros_like(span), where=span > 0)
        self._categories = [pd.Index(sorted(frame[column].dropna().unique())) for column in _CATEGORICAL_COLUMNS]
        return self

    def transform(self, frame):
        if not hasattr(self, "_lower"):
            raise ValueError("Fit SpecialistFeatures before transforming rows.")
        numeric = _numeric_values(frame)
        numeric -= self._lower
        numeric *= self._inverse_span
        np.clip(numeric, 0.0, 1.0, out=numeric)
        scaled = numeric.astype(np.float32)
        del numeric

        basis_size = len(_KNOTS)
        categorical_size = sum(len(categories) for categories in self._categories)
        width = 1 + scaled.shape[1] * basis_size + len(_INTERACTIONS) * basis_size**2 + categorical_size
        output = np.empty((len(frame), width), dtype=np.float32)
        output[:, 0] = 1.0

        offset = 1
        for values in scaled.T:
            for knot in _KNOTS:
                feature = output[:, offset]
                np.subtract(values, knot, out=feature)
                np.maximum(feature, 0.0, out=feature)
                offset += 1

        for left, right in _INTERACTIONS:
            for i in range(basis_size):
                for j in range(basis_size):
                    np.multiply(output[:, 1 + left * basis_size + i],
                                output[:, 1 + right * basis_size + j], out=output[:, offset])
                    offset += 1

        output[:, offset:] = 0.0
        for column, categories in zip(_CATEGORICAL_COLUMNS, self._categories, strict=True):
            codes = categories.get_indexer(frame[column])
            known = np.flatnonzero(codes >= 0)
            output[known, offset + codes[known]] = 1.0
            offset += len(categories)
        return output
