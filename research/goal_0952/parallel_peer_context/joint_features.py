"""Measure unusual pairs relative to genuine transactions in the same context."""

import numpy as np
import pandas as pd
from scipy.special import ndtri


RELATIONS = (
    ('velocity_score', 'hours_since_last_txn', ('txn_count_last_24h', 'auth_method', 'channel')),
    ('amount_usd', 'merchant_risk_score', ('merchant_category', 'is_new_merchant', 'billing_shipping_mismatch')),
)
FEATURES_PER_RELATION = ('joint_distance', 'signed_residual', 'rank_product')
CDF_PARENT_ROWS = 80.
CORRELATION_PARENT_ROWS = 200.


def empirical_cdf(reference, values):
    result = np.empty_like(values, dtype=np.float64)
    for j in range(values.shape[1]):
        ordered = np.sort(reference[:, j])
        left = np.searchsorted(ordered, values[:, j], side='left')
        right = np.searchsorted(ordered, values[:, j], side='right')
        result[:, j] = (.5 * (left + right) + .5) / (len(ordered) + 1)
    return result


def conditional_normal_scores(reference, values, fit_keys, query_keys, sizes):
    distribution = empirical_cdf(reference, values)
    for a, b, size in zip(fit_keys, query_keys, sizes, strict=True):
        fit_order = np.argsort(a, kind='stable')
        query_order = np.argsort(b, kind='stable')
        fit_counts = np.bincount(a, minlength=size)
        query_counts = np.bincount(b, minlength=size)
        fit_ends, query_ends = np.cumsum(fit_counts), np.cumsum(query_counts)
        for group in np.flatnonzero((fit_counts > 0) & (query_counts > 0)):
            n = fit_counts[group]
            i = fit_order[fit_ends[group] - n:fit_ends[group]]
            j = query_order[query_ends[group] - query_counts[group]:query_ends[group]]
            weight = n / (n + CDF_PARENT_ROWS)
            distribution[j] += weight * (empirical_cdf(reference[i], values[j]) - distribution[j])
    return ndtri(np.clip(distribution, .0025, .9975))


def correlations(scores, keys, size):
    x, y = scores.T
    n = np.bincount(keys, minlength=size).astype(np.float64)
    safe_n = np.maximum(n, 1.)
    sx, sy = (np.bincount(keys, weights=v, minlength=size) for v in (x, y))
    sxx, syy, sxy = (np.bincount(keys, weights=v, minlength=size) for v in (x*x, y*y, x*y))
    scale = np.sqrt(np.maximum(sxx - sx*sx/safe_n, 0.) * np.maximum(syy - sy*sy/safe_n, 0.))
    covariance = sxy - sx*sy/safe_n
    rho = np.divide(covariance, scale, out=np.zeros_like(scale), where=scale > 1e-12)
    return n, rho


def pair_features(reference, query, fit_keys, query_keys, sizes):
    """Co-rank two variables, then measure departure from their usual dependence."""
    reference_size = len(reference)
    normal = conditional_normal_scores(
        reference, np.concatenate((reference, query)), fit_keys,
        tuple(np.concatenate((a, b)) for a, b in zip(fit_keys, query_keys, strict=True)), sizes,
    )
    training, scoring = normal[:reference_size], normal[reference_size:]
    _, pooled = correlations(training, np.zeros(reference_size, dtype=np.int8), 1)
    rho = np.full(len(query), pooled[0])
    for keys, query_key, size in zip(fit_keys, query_keys, sizes, strict=True):
        counts, own = correlations(training, keys, size)
        parent = pooled[np.arange(size) // (size // len(pooled))]
        weight = counts / (counts + CORRELATION_PARENT_ROWS)
        pooled = parent + weight * (own - parent)
        rho = pooled[query_key]
    rho = np.clip(rho, -.90, .90)
    x, y = scoring.T
    residual = (y - rho*x) / np.sqrt(1. - rho*rho)
    return np.column_stack((np.square(x) + np.square(residual), residual, x*y)).astype(np.float32)


def context_keys(reference, query, fields):
    fit_keys, query_keys, sizes = [], [], []
    fit = np.zeros(len(reference), dtype=np.int32)
    score = np.zeros(len(query), dtype=np.int32)
    size = 1
    for field in fields:
        a, b = reference[field], query[field]
        if field == 'txn_count_last_24h':
            # Keep 0 through 7 separate and pool the sparse 8+ tail.
            a, b = a.clip(upper=8), b.clip(upper=8)
        levels = pd.Index(a.unique()).sort_values()
        width = len(levels) + 1  # Zero is the unseen category.
        fit = fit * width + (levels.get_indexer(a) + 1).astype(np.int32)
        score = score * width + (levels.get_indexer(b) + 1).astype(np.int32)
        size *= width
        fit_keys.append(fit)
        query_keys.append(score)
        sizes.append(size)
    return tuple(fit_keys), tuple(query_keys), tuple(sizes)


def joint_features(fit, labels, query):
    genuine = fit.iloc[np.flatnonzero(labels == 0)]
    if genuine.empty:
        raise ValueError('At least one genuine transaction is required')
    parts = []
    for left, right, fields in RELATIONS:
        columns = [left, right]
        reference_values = genuine[columns].to_numpy(dtype=np.float64)
        query_values = query[columns].to_numpy(dtype=np.float64)
        parts.append(pair_features(reference_values, query_values, *context_keys(genuine, query, fields)))
    scores = np.column_stack(parts)
    if not np.isfinite(scores).all():
        raise ValueError('Joint context features must be finite')
    return scores
