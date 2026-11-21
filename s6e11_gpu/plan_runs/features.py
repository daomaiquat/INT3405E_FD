"""Fold-scoped features for steps 8–10 of the S6E11 experiment plan."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.json"
LOG_COLUMNS = ("amount_usd", "account_balance_usd")
DIGIT_COLUMNS = (
    "amount_usd", "hours_since_last_txn", "distance_from_home_km",
    "account_balance_usd", "velocity_score", "merchant_risk_score",
)
DIGIT_NAMES = ("cents", "int", "int10", "tenths", "lead1", "lead2")
PAIR_GROUPS = {
    "p1": (
        ("merchant_category", "merchant_risk_score"),
        ("is_foreign_transaction", "ip_country_mismatch"),
        ("merchant_category", "decile:amount_usd"),
        ("txn_count_last_24h", "decile:velocity_score"),
    ),
    "p2": (
        ("merchant_category", "customer_age"),
        ("decile:amount_usd", "decile:merchant_risk_score"),
        ("decile:hours_since_last_txn", "decile:velocity_score"),
        ("customer_age", "decile:merchant_risk_score"),
    ),
    "p3": (
        ("is_new_merchant", "day_of_week"),
        ("is_ai_generated_scam_attempt", "decile:velocity_score"),
        ("customer_age", "card_age_months"),
        ("time_of_day_hour", "day_of_week"),
        ("txn_count_last_24h", "customer_age"),
    ),
}
GROUPS = frozenset(("raw", "glm_raw", "te13", "te24", "cents", "digits",
                    "p1", "p2", "p3", "count", "derived"))
MISSING_INT = np.iinfo(np.int64).min
POWERS_OF_TEN = 10 ** np.arange(18, dtype=np.int64)


def _numeric_units(values, scale):
    scaled = np.rint(values * scale)
    valid = np.isfinite(scaled) & (scaled > -(2.0 ** 63)) & (scaled < 2.0 ** 63)
    result = np.full(len(values), MISSING_INT, dtype=np.int64)
    result[valid] = scaled[valid].astype(np.int64)
    return result


def _digit_keys(units, kind):
    valid = units != MISSING_INT
    present = units[valid]
    if kind == "cents":
        keys = present % 100
    elif kind == "int":
        keys = present // 100
    elif kind == "int10":
        keys = present // 1000
    elif kind == "tenths":
        keys = present // 10
    else:
        integers = np.abs(present // 100)
        lengths = np.searchsorted(POWERS_OF_TEN[1:], integers, side="right") + 1
        prefix_width = 1 if kind == "lead1" else 2
        prefixes = integers // POWERS_OF_TEN[np.maximum(lengths - prefix_width, 0)]
        keys = 100 * lengths + prefixes
    result = np.full(len(units), MISSING_INT, dtype=np.int64)
    result[valid] = keys
    return result


def _decile_thresholds(values):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.empty(0, dtype=np.float64)
    return np.unique(np.quantile(finite, np.linspace(0, 1, 11)))[1:-1]


def _decile_codes(values, thresholds):
    codes = 1 + np.searchsorted(thresholds, values, side="left")
    return np.where(np.isfinite(values), codes, 0)


def _lookup(codes, labels, size, smoothing, counts_only, logit):
    counts = np.bincount(codes, minlength=size)
    if counts_only:
        return np.log1p(counts.astype(np.float32))
    prior = labels.mean()
    sum_y = np.bincount(codes, weights=labels, minlength=size)
    sum_y += smoothing * prior
    denominators = counts + smoothing
    rates = np.full(size, prior, dtype=np.float64)
    np.divide(sum_y, denominators, out=rates, where=denominators > 0)
    if logit:
        np.clip(rates, 1e-6, 1 - 1e-6, out=rates)
        rates = np.log(rates / (1 - rates))
    return rates.astype(np.float32)


class FeatureBuilder:
    """Indexes are positions in train followed by test, independently of DataFrame index.

    Integer codes assigned across X are only identifiers. Vocabularies for one-hot,
    decile boundaries, counts and label statistics use the designated reference rows.
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        self._numeric = tuple(config["numeric_columns"])
        self._flags = tuple(config["flag_columns"])
        self._categorical = tuple(config["categorical_columns"])
        self._columns = self._numeric + self._flags + self._categorical
        self._labels = train[config["target"]].to_numpy(copy=True)
        self._size = len(train) + len(test)
        self._scale = int(config["training_plan"]["target_encoding"]["key_scale"])
        self._values = {}
        for name in self._columns:
            options = {} if name in self._categorical else {"dtype": np.float64, "na_value": np.nan}
            self._values[name] = np.concatenate((
                train[name].to_numpy(**options), test[name].to_numpy(**options),
            ))
        self._cache = {}

    def _units(self, name):
        scale = 1 if name in self._flags else self._scale
        return _numeric_units(self._values[name], scale)

    def _codes(self, key):
        """Return compact stable IDs and the table size, one slot for missing."""
        if key not in self._cache:
            if isinstance(key, tuple):
                left, right = key
                if left in self._columns and right in self._columns:
                    left_codes, _ = self._codes(left)
                    right_codes, right_size = self._codes(right)
                    values = left_codes.astype(np.int64) * right_size + right_codes
                else:
                    name, kind = key
                    values = _digit_keys(self._units(name), kind)
                codes, unique = pd.factorize(values, sort=False)
            elif key in self._categorical:
                codes, unique = pd.factorize(self._values[key], sort=True)
            else:
                codes, unique = pd.factorize(self._units(key), sort=False)
            self._cache[key] = (codes.astype(np.int32) + 1, len(unique) + 1)
        return self._cache[key]

    def _dynamic_codes(self, key, fit, roles):
        operands = []
        sizes = []
        for name in key:
            if name.startswith("decile:"):
                scores = self._values[name.removeprefix("decile:")]
                thresholds = _decile_thresholds(scores[fit])
                operands.append([_decile_codes(scores[rows], thresholds) for rows in roles])
                sizes.append(len(thresholds) + 2)
            else:
                codes, size = self._codes(name)
                operands.append([codes[rows] for rows in roles])
                sizes.append(size)
        left, right = operands
        result = tuple(a.astype(np.int64, copy=False) * sizes[1] + b
                       for a, b in zip(left, right, strict=True))
        return result, sizes[0] * sizes[1]

    @staticmethod
    def _is_dynamic(key):
        return isinstance(key, tuple) and any(part.startswith("decile:") for part in key)

    def make(
        self, reference: np.ndarray, queries: tuple[np.ndarray, ...], *,
        groups: tuple[str, ...], smoothing: float = 20, seed: int = 2026,
        logit_te: bool = False,
    ) -> tuple[np.ndarray, ...]:
        selected = frozenset(groups)
        if not selected or selected - GROUPS:
            raise ValueError(f"Expected feature groups from {sorted(GROUPS)}, received {groups}")
        if not np.isfinite(smoothing) or smoothing < 0:
            raise ValueError("smoothing must be nonnegative and finite")
        reference = np.asarray(reference)
        queries = tuple(np.asarray(query) for query in queries)
        roles = (reference, *queries)
        for index, rows in enumerate(roles):
            limit = len(self._labels) if index == 0 else self._size
            if rows.ndim != 1 or rows.dtype.kind not in "iu" or (rows < 0).any() or (rows >= limit).any():
                raise ValueError("reference and queries must be valid one-dimensional row positions")
        if not len(reference) or len(np.unique(reference)) != len(reference):
            raise ValueError("reference must contain unique rows")
        member = np.zeros(self._size, dtype=bool)
        member[reference] = True
        if any(member[rows].any() for rows in queries):
            raise ValueError("query rows must be outside the fitting reference")

        raw = bool(selected & {"raw", "glm_raw"})
        numeric_raw = self._numeric + self._flags if raw else ()
        logs = LOG_COLUMNS if "glm_raw" in selected else ()
        category_specs = []
        for name in self._categorical if raw else ():
            codes, size = self._codes(name)
            categories = np.unique(codes[reference])
            mapper = np.full(size, -1, dtype=np.int32)
            mapper[categories] = np.arange(len(categories), dtype=np.int32)
            category_specs.append((codes, mapper, len(categories)))

        per_value = self._columns if "te24" in selected else self._numeric if "te13" in selected else ()
        cents = DIGIT_COLUMNS if "cents" in selected else ()
        digits = tuple((name, kind) for name in DIGIT_COLUMNS for kind in DIGIT_NAMES) if "digits" in selected else ()
        pairs = tuple(pair for group, keys in PAIR_GROUPS.items() if group in selected for pair in keys)
        counted = (*self._columns, *pairs) if "count" in selected else ()
        derived = 2 if "derived" in selected else 0
        encoded = (*per_value, *digits, *pairs)
        width = (len(numeric_raw) + len(logs) + sum(spec[2] for spec in category_specs)
                 + len(encoded) + len(cents) + len(counted) + derived)
        matrices = tuple(np.empty((len(rows), width), dtype=np.float32) for rows in roles)

        # Raw and deterministic transforms preserve source order and use no labels.
        offset = 0
        for name in (*numeric_raw, *logs):
            is_log = offset >= len(numeric_raw)
            values = self._values[name]
            for matrix, rows in zip(matrices, roles, strict=True):
                data = values[rows]
                data = np.where(np.isfinite(data), data, 0.0)
                matrix[:, offset] = np.log1p(np.maximum(data, 0.0)) if is_log else data
            offset += 1
        for codes, mapper, count in category_specs:
            for matrix, rows in zip(matrices, roles, strict=True):
                matrix[:, offset:offset + count] = 0
                local_codes = mapper[codes[rows]]
                observed = np.flatnonzero(local_codes >= 0)
                matrix[observed, offset + local_codes[observed]] = 1
            offset += count

        # The splitter and all supervised statistics see only labels of reference rows.
        labels = self._labels[reference]
        inner = []
        if encoded or counted:
            if not np.isin(labels, (0, 1)).all():
                raise ValueError("reference labels must be binary")
            labels = labels.astype(np.uint8, copy=False)
            if np.bincount(labels, minlength=2).min() < 5:
                raise ValueError("5 inner folds require 5 reference rows of each class")
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            inner = [(reference[fit], held, labels[fit])
                     for fit, held in splitter.split(reference, labels)]

        def write_keys(keys, offset, counts_only=False):
            for key in keys:
                dynamic = self._is_dynamic(key)
                if not dynamic:
                    codes, size = self._codes(key)
                for fit, held, fit_labels in inner:
                    if dynamic:
                        (fit_codes, held_codes), size = self._dynamic_codes(
                            key, fit, (fit, reference[held]),
                        )
                    else:
                        fit_codes, held_codes = codes[fit], codes[reference[held]]
                    table = _lookup(fit_codes, fit_labels, size, smoothing, counts_only, logit_te)
                    matrices[0][held, offset] = table[held_codes]
                if len(queries):
                    if dynamic:
                        (fit_codes, *query_codes), size = self._dynamic_codes(key, reference, roles)
                    else:
                        fit_codes, query_codes = codes[reference], (codes[rows] for rows in queries)
                    table = _lookup(fit_codes, labels, size, smoothing, counts_only, logit_te)
                    for matrix, values in zip(matrices[1:], query_codes, strict=True):
                        matrix[:, offset] = table[values]
                offset += 1
            return offset

        offset = write_keys(per_value, offset)
        for name in cents:
            values = _digit_keys(self._units(name), "cents")
            for matrix, rows in zip(matrices, roles, strict=True):
                data = values[rows]
                matrix[:, offset] = np.where(data == MISSING_INT, 0, data)
            offset += 1
        offset = write_keys((*digits, *pairs), offset)
        offset = write_keys(counted, offset, counts_only=True)
        if derived:
            for matrix, rows in zip(matrices, roles, strict=True):
                flags = sum(np.nan_to_num(self._values[name][rows], posinf=0, neginf=0)
                            for name in self._flags)
                amount, balance = (np.nan_to_num(self._values[name][rows], posinf=0, neginf=0)
                                   for name in LOG_COLUMNS)
                matrix[:, offset] = np.square(flags)
                matrix[:, offset + 1] = amount / np.maximum(balance, 1.0)
            offset += derived
        if offset != width:
            raise RuntimeError("feature layout did not fill every column")
        return matrices
