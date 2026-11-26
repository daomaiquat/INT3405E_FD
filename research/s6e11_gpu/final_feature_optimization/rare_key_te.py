"""Cross-fitted target encoding for amount and balance with coarse-bin fallback."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


COLUMNS = ("amount_usd", "account_balance_usd")
QUANTILES = np.arange(1, 32) / 32
COARSE_STRENGTH = 50
EXACT_STRENGTH = 20
INNER_FOLDS = 5


class RareKeyEncoder:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self._train_size = len(train)
        self._size = len(train) + len(test)
        self._values = tuple(
            np.concatenate((train[column].to_numpy(dtype=np.float64), test[column].to_numpy(dtype=np.float64)))
            for column in COLUMNS
        )
        cent_limit = (2 ** 63) / 100
        if any(not np.isfinite(values).all() or (np.abs(values) >= cent_limit).any() for values in self._values):
            raise ValueError("Amount and balance must be finite and within integer-cent range")
        self._keys = tuple(np.rint(values * 100).astype(np.int64) for values in self._values)

    @staticmethod
    def _positions(rows, limit, role):
        rows = np.asarray(rows)
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or (rows < 0).any() or (rows >= limit).any():
            raise ValueError(f"{role} must be valid one-dimensional row positions")
        return rows

    @staticmethod
    def _encode(values, keys, fit, fit_labels, queries):
        boundaries = np.unique(np.quantile(values[fit], QUANTILES))
        fit_bins = np.searchsorted(boundaries, values[fit], side="right")
        bin_count = np.bincount(fit_bins, minlength=len(boundaries) + 1)
        bin_fraud = np.bincount(fit_bins, weights=fit_labels, minlength=len(boundaries) + 1)
        coarse = (bin_fraud + COARSE_STRENGTH * fit_labels.mean()) / (bin_count + COARSE_STRENGTH)
        exact, fit_exact = np.unique(keys[fit], return_inverse=True)
        exact_count = np.bincount(fit_exact)
        exact_fraud = np.bincount(fit_exact, weights=fit_labels)

        result = []
        for rows in queries:
            coarse_rate = coarse[np.searchsorted(boundaries, values[rows], side="right")]
            query_keys = keys[rows]
            position = np.searchsorted(exact, query_keys).clip(max=len(exact) - 1)
            known = exact[position] == query_keys
            matched = position[known]
            coarse_rate[known] = (
                (exact_fraud[matched] + EXACT_STRENGTH * coarse_rate[known])
                / (exact_count[matched] + EXACT_STRENGTH)
            )
            result.append(coarse_rate)
        return result

    def make(
        self, reference: np.ndarray, queries: tuple[np.ndarray, ...], labels: np.ndarray, seed: int,
    ) -> tuple[np.ndarray, ...]:
        """Fit each reference row out of fold; transform queries with the full reference."""
        reference = self._positions(reference, self._train_size, "Reference rows")
        queries = tuple(self._positions(rows, self._size, "Query rows") for rows in queries)
        if len(reference) == 0 or len(np.unique(reference)) != len(reference):
            raise ValueError("Reference rows must be nonempty and unique")
        membership = np.zeros(self._size, dtype=bool)
        membership[reference] = True
        if any(membership[rows].any() for rows in queries):
            raise ValueError("Query rows must be outside the reference")

        labels = np.asarray(labels)
        if labels.shape != (self._train_size,):
            raise ValueError("Labels must align with all train rows")
        reference_labels = labels[reference]
        if not np.isin(reference_labels, (0, 1)).all():
            raise ValueError("Reference labels must be binary")
        reference_labels = reference_labels.astype(np.uint8, copy=False)
        if np.bincount(reference_labels, minlength=2).min() < INNER_FOLDS:
            raise ValueError(f"Cross-fitting requires {INNER_FOLDS} reference rows of each class")

        splitter = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
        inner = tuple(splitter.split(reference, reference_labels))
        matrices = tuple(np.empty((len(rows), len(COLUMNS)), dtype=np.float32) for rows in (reference, *queries))
        for column, (values, keys) in enumerate(zip(self._values, self._keys, strict=True)):
            for fit, held in inner:
                (encoded,) = self._encode(values, keys, reference[fit], reference_labels[fit], (reference[held],))
                matrices[0][held, column] = encoded
            if queries:
                encoded = self._encode(values, keys, reference, reference_labels, queries)
                for matrix, result in zip(matrices[1:], encoded, strict=True):
                    matrix[:, column] = result
        return matrices
