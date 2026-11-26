"""Sparse class-conditional likelihood ratios fit on a labeled training fold.

The caller builds cross-fitted training features and supplies only the fold's
training labels to fit. Quantization follows the precision of the official
features. Numeric missing values have their own code.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


# (column, ticks per unit, ticks per coarse bin, use auth x merchant-risk context)
_FIELDS = (
    ("hours_since_last_txn", 100, 25, True),
    ("velocity_score", 10, 20, True),
    ("merchant_risk_score", 10, 20, False),
    ("amount_usd", 100, 100, True),
)
_RISK_CUTS = (20, 30, 40, 50, 60, 70, 80)
_CONTEXT_MIN_CLASS_COUNT = 8
_LOCAL_RADIUS = 2

# A sparse joint key packs context in the high bits and value in the low bits.
# The low code 0 marks missing; observed ticks start at 1.
_KEY_RADIX = 1 << 32
_MAX_TICK = _KEY_RADIX - 2
_FEATURE_SUFFIXES = (
    "global_coarse_llr",
    "global_exact_llr",
    "context_coarse_llr",
    "context_exact_llr",
    "context_exact_weight",
)


class _ClassCounts:
    """Counts by key and class, stored only for keys observed at fit."""

    def __init__(self, key, labels):
        inverse, keys = pd.factorize(key, sort=True)
        self.keys = pd.Index(keys)
        frequencies = np.bincount(2 * inverse + labels, minlength=2 * len(keys))
        self.counts = np.zeros((len(keys) + 1, 2), dtype=np.int64)
        self.counts[:-1] = frequencies.reshape(-1, 2)

    def get(self, key):
        # pandas returns -1 for misses, which selects the last, all-zero row.
        return self.counts[self.keys.get_indexer(key)]

    def window(self, lo, hi):
        keys = self.keys.to_numpy()
        start = np.searchsorted(keys, lo)
        end = np.searchsorted(keys, hi, side="right")
        prefix = np.cumsum(np.vstack((self.counts[-1], self.counts[:-1])), axis=0)
        return prefix[end] - prefix[start]


def _numeric_codes(frame, name, scale, width):
    values = frame[name].to_numpy(dtype=np.float64, na_value=np.nan)
    valid = np.isfinite(values) & (values >= 0)
    exact = np.zeros(len(values), dtype=np.int64)
    bounded = np.minimum(values[valid], _MAX_TICK / scale)
    exact[valid] = np.rint(bounded * scale).astype(np.int64) + 1
    coarse = (exact - 1) // width + 1
    return exact, coarse


def _contexts(frame, auth_levels):
    auth = auth_levels.get_indexer(frame["auth_method"]) + 1
    risk = frame["merchant_risk_score"].to_numpy(dtype=np.float64, na_value=np.nan)
    band = np.searchsorted(_RISK_CUTS, risk, side="right")
    band[~np.isfinite(risk) | (risk < 0)] = len(_RISK_CUTS) + 1
    joint = 1 + (auth - 1) * (len(_RISK_CUTS) + 2) + band
    joint[auth == 0] = 0
    return {False: auth, True: joint}


def _mix_with_parent(counts, totals, parent, strength, minimum=1):
    """Class-normalize first, then blend both classes by the same weight."""
    support = totals.min(axis=-1, keepdims=True)
    weight = np.where(support >= minimum, support / (support + strength), 0)
    empirical = counts / np.maximum(totals, 1)
    return parent + weight * (empirical - parent), weight


def _log_ratio(probabilities):
    return np.log(probabilities[:, 1]) - np.log(probabilities[:, 0])


@dataclass
class _FieldDensity:
    global_coarse: _ClassCounts
    global_exact: _ClassCounts
    context_coarse: _ClassCounts
    context_exact: _ClassCounts

    @classmethod
    def fit(cls, exact, coarse, context, eligible, labels):
        return cls(
            _ClassCounts(coarse, labels),
            _ClassCounts(exact, labels),
            _ClassCounts(context[eligible] + coarse[eligible], labels[eligible]),
            _ClassCounts(context[eligible] + exact[eligible], labels[eligible]),
        )

    def transform(self, exact, coarse, width, context, global_total, context_total):
        global_coarse = self.global_coarse.get(coarse)
        global_exact = self.global_exact.get(exact)
        context_coarse = self.context_coarse.get(context + coarse)
        context_exact = self.context_exact.get(context + exact)

        # The shared coarse prior reserves positive mass for one unseen bucket.
        # Within a known bin, uniform mass is spread across all quantized ticks.
        pooled_coarse = (global_coarse.sum(axis=1, keepdims=True) + 1) / (
            global_total.sum() + len(self.global_coarse.keys) + 1
        )
        uniform_exact = np.where(
            (coarse == 0) | (global_coarse.sum(axis=1) == 0), 1.0, 1.0 / width,
        )[:, None]
        global_bin_prob, _ = _mix_with_parent(global_coarse, global_total, pooled_coarse, 8)
        global_tick_prob, _ = _mix_with_parent(global_exact, global_coarse, uniform_exact, 12)
        context_bin_prob, _ = _mix_with_parent(
            context_coarse, context_total, global_bin_prob, 32, _CONTEXT_MIN_CLASS_COUNT,
        )
        context_tick_prob, weight = _mix_with_parent(
            context_exact, context_coarse, global_tick_prob, 18, 3,
        )

        # The exact likelihood is the joint probability of the parent bin and
        # the tick within that bin, so its log ratio is their sum.
        global_bin_llr = _log_ratio(global_bin_prob)
        context_bin_llr = _log_ratio(context_bin_prob)
        return (
            global_bin_llr,
            global_bin_llr + _log_ratio(global_tick_prob),
            context_bin_llr,
            context_bin_llr + _log_ratio(context_tick_prob),
            weight[:, 0],
        )

    def elapsed_spike(self, exact, context, eligible):
        lo = np.maximum(exact - _LOCAL_RADIUS, 1)
        hi = np.minimum(exact + _LOCAL_RADIUS, _MAX_TICK + 1)
        uniform = (1.0 / (hi - lo + 1))[:, None]
        global_center = self.global_exact.get(exact)
        context_center = self.context_exact.get(context + exact)
        global_window = self.global_exact.window(lo, hi)
        context_window = self.context_exact.window(context + lo, context + hi)

        global_share, _ = _mix_with_parent(global_center, global_window, uniform, 16)
        context_share, weight = _mix_with_parent(
            context_center, context_window, global_share, 16, 3,
        )
        valid = exact != 0
        contrast = np.zeros(len(exact))
        # Compare the class-normalized odds of the central tick vs nearby ticks.
        contrast[valid] = _log_ratio(context_share[valid]) - _log_ratio(1 - context_share[valid])
        return contrast, np.where(valid & eligible, weight[:, 0], 0)


class ClassConditionalDensity:
    """Fit sparse global and contextual coarse-to-exact class distributions."""

    feature_names = tuple(
        f"{name}_{suffix}" for name, *_ in _FIELDS for suffix in _FEATURE_SUFFIXES
    ) + (
        "hours_since_last_txn_local_spike_llr",
        "hours_since_last_txn_local_neighbor_weight",
    )

    def fit(self, frame, labels):
        labels = np.asarray(labels)
        if not len(frame) or labels.shape != (len(frame),) or not np.isin(labels, [0, 1]).all():
            raise ValueError("fit requires a nonempty frame and one binary label per row")
        labels = labels.astype(np.int64, copy=False)

        auth_levels = pd.Index(frame["auth_method"].dropna().unique())
        contexts = _contexts(frame, auth_levels)
        populations = {kind: _ClassCounts(codes, labels) for kind, codes in contexts.items()}
        eligible = {
            kind: (codes > 0)
            & (populations[kind].get(codes).min(axis=1) >= _CONTEXT_MIN_CLASS_COUNT)
            for kind, codes in contexts.items()
        }
        fields = []
        for name, scale, width, use_risk in _FIELDS:
            exact, coarse = _numeric_codes(frame, name, scale, width)
            fields.append(_FieldDensity.fit(
                exact, coarse, contexts[use_risk] * _KEY_RADIX, eligible[use_risk], labels,
            ))

        self._fields = fields
        self._auth_levels = auth_levels
        self._populations = populations
        self._global_total = np.bincount(labels, minlength=2)
        return self

    def transform(self, frame):
        if not hasattr(self, "_fields"):
            raise RuntimeError("fit must be called before transform")
        result = np.empty((len(frame), len(self.feature_names)), dtype=np.float32)
        contexts = _contexts(frame, self._auth_levels)
        populations = {
            kind: np.where((codes > 0)[:, None], self._populations[kind].get(codes), 0)
            for kind, codes in contexts.items()
        }
        for i, (name, scale, width, use_risk) in enumerate(_FIELDS):
            exact, coarse = _numeric_codes(frame, name, scale, width)
            context = contexts[use_risk] * _KEY_RADIX
            context_total = populations[use_risk]
            field = self._fields[i]
            features = field.transform(exact, coarse, width, context, self._global_total, context_total)
            start = i * len(_FEATURE_SUFFIXES)
            for j, values in enumerate(features, start=start):
                result[:, j] = values
            if name == "hours_since_last_txn":
                eligible = context_total.min(axis=1) >= _CONTEXT_MIN_CLASS_COUNT
                result[:, -2], result[:, -1] = field.elapsed_spike(exact, context, eligible)
        return result
