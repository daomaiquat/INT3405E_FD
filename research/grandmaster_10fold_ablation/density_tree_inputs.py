"""Fold-local common inputs for the standalone XGBoost comparison."""

import numpy as np
import pandas as pd


RAW_COLUMNS = (
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
)
_RAW_POSITIONS = {name: i for i, name in enumerate(RAW_COLUMNS)}
_CATEGORY_COLUMNS = ("merchant_category", "card_type", "auth_method", "channel", "device_type")
_EXACT_COLUMNS = (
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
    *_CATEGORY_COLUMNS,
)
_LOG_COLUMNS = ("amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd")
_RATIOS = (
    ("log_amount_to_balance", "amount_usd", "account_balance_usd"),
    ("log_velocity_to_count", "velocity_score", "txn_count_last_24h"),
    ("log_count_to_elapsed", "txn_count_last_24h", "hours_since_last_txn"),
)
_DECIMALS = (
    ("amount_usd", 100),
    ("hours_since_last_txn", 100),
    ("distance_from_home_km", 100),
    ("account_balance_usd", 100),
    ("velocity_score", 10),
    ("merchant_risk_score", 100),
)
_SMOOTHING = 20.0
_FLOAT_LIMIT = np.finfo(np.float32).max


def _check_columns(frame):
    missing = [name for name in RAW_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"Missing official feature columns: {', '.join(missing)}")


def _numeric_values(series):
    values = series.to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(values).any():
        values = np.where(np.isfinite(values), values, np.nan)
    return values


def _exact_keys(frame, column):
    if column in _CATEGORY_COLUMNS:
        # Factorize labels so unused levels in a Categorical stay unobserved.
        return frame[column].to_numpy(dtype=object)
    return _numeric_values(frame[column])


def _fit_exact(keys, labels, prior):
    codes, vocabulary = pd.factorize(keys, sort=True)
    known = codes >= 0
    counts = np.bincount(codes[known], minlength=len(vocabulary))
    positives = np.bincount(codes[known], weights=labels[known], minlength=len(vocabulary))
    # get_indexer returns -1 for an unseen key, selecting the final row.
    table = np.empty((len(vocabulary) + 1, 3), dtype=np.float32)
    table[:-1, 0] = counts
    table[:-1, 1] = counts / len(labels)
    table[:-1, 2] = (positives + _SMOOTHING * prior) / (counts + _SMOOTHING)
    table[-1] = (0.0, 0.0, prior)
    return pd.Index(vocabulary), table


class DensityTreeInputs:
    """Encode official fields plus row-level transforms and supervised marginals.

    Fit each instance on reference rows and their labels. Cross-fit the
    training matrix by scoring held-out inner rows with separate instances.
    """

    feature_names = (
        *RAW_COLUMNS,
        *(f"log1p_{column}" for column in _LOG_COLUMNS),
        *(name for name, _, _ in _RATIOS),
        *(f"{column}_{feature}" for column, scale in _DECIMALS for feature in (
            ("fractional_units", "whole_grid", "tenth_grid") if scale == 100
            else ("fractional_units", "whole_grid")
        )),
        *(f"{column}_exact_{feature}" for column in _EXACT_COLUMNS
          for feature in ("count", "frequency", "target")),
    )

    def fit(self, frame, labels):
        _check_columns(frame)
        labels = np.asarray(labels, dtype=np.float64)
        if labels.shape != (len(frame),) or not labels.size or not np.isin(labels, [0, 1]).all():
            raise ValueError("Expected a nonempty fold with one binary label per row.")
        prior = labels.mean()
        self._exact = {
            column: _fit_exact(_exact_keys(frame, column), labels, prior)
            for column in _EXACT_COLUMNS
        }
        return self

    def transform(self, frame):
        if not hasattr(self, "_exact"):
            raise ValueError("Fit DensityTreeInputs before scoring rows.")
        _check_columns(frame)
        output = np.empty((len(frame), len(self.feature_names)), dtype=np.float32)
        for index, column in enumerate(RAW_COLUMNS):
            if column not in _CATEGORY_COLUMNS:
                values = _numeric_values(frame[column])
                output[:, index] = np.nan_to_num(np.clip(values, -_FLOAT_LIMIT, _FLOAT_LIMIT), nan=-1.0)

        offset = len(RAW_COLUMNS)
        for column in _LOG_COLUMNS:
            values = _numeric_values(frame[column])
            output[:, offset] = np.nan_to_num(np.sign(values) * np.log1p(np.abs(values)), nan=0.0)
            offset += 1

        for _, numerator, denominator in _RATIOS:
            upper = _numeric_values(frame[numerator])
            lower = _numeric_values(frame[denominator])
            log_ratio = np.sign(upper) * np.log1p(np.abs(upper) / (1.0 + np.abs(lower)))
            output[:, offset] = np.nan_to_num(log_ratio, nan=0.0)
            offset += 1

        for column, scale in _DECIMALS:
            values = _numeric_values(frame[column])
            fractional = np.round(np.remainder(np.abs(values), 1.0) * scale) % scale
            output[:, offset] = np.nan_to_num(fractional, nan=-1.0)
            output[:, offset + 1] = fractional == 0
            offset += 2
            if scale == 100:
                output[:, offset] = fractional % 10 == 0
                offset += 1

        for column, (vocabulary, table) in self._exact.items():
            codes = vocabulary.get_indexer(_exact_keys(frame, column))
            output[:, offset:offset + 3] = table[codes]
            if column in _CATEGORY_COLUMNS:
                output[:, _RAW_POSITIONS[column]] = codes
            offset += 3
        return output
