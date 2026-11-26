"""Screen prespecified three-way fraud interactions beyond all two-way effects."""

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.special import expit
from scipy.stats import norm, spearmanr
from sklearn.model_selection import StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path[:0] = [str(PARENT), str(PARENT / 'followup')]
from eda_from_scratch import group_totals
from evaluate_and_export import assess
from grandmaster_inputs import write_json
from run_followup import build_inputs

TRIPLES = (
    ('channel', 'auth_method', 'cvv_retry_count'),
    ('card_type', 'auth_method', 'cvv_retry_count'),
    ('device_type', 'used_vpn', 'ip_country_mismatch'),
    ('channel', 'used_vpn', 'ip_country_mismatch'),
    ('is_foreign_transaction', 'ip_country_mismatch', 'used_vpn'),
    ('is_foreign_transaction', 'ip_country_mismatch', 'billing_shipping_mismatch'),
    ('is_ai_generated_scam_attempt', 'auth_method', 'channel'),
    ('is_ai_generated_scam_attempt', 'cvv_retry_count', 'is_new_merchant'),
    ('merchant_category', 'billing_shipping_mismatch', 'is_new_merchant'),
    ('is_new_merchant', 'merchant_risk_score', 'billing_shipping_mismatch'),
    ('velocity_score', 'hours_since_last_txn', 'ip_country_mismatch'),
    ('txn_count_last_24h', 'hours_since_last_txn', 'channel'),
    ('cvv_retry_count', 'prior_disputes', 'auth_method'),
    ('is_ai_generated_scam_attempt', 'is_foreign_transaction', 'used_vpn'),
    ('customer_age', 'card_age_months', 'auth_method'),
    ('day_of_week', 'channel', 'is_ai_generated_scam_attempt'),
)
QUANTILE_FIELDS = ('velocity_score', 'merchant_risk_score', 'card_age_months', 'customer_age')
FIXED_EDGES = {
    'cvv_retry_count': (.5, 1.5, 2.5),
    'prior_disputes': (.5, 1.5),
    'txn_count_last_24h': (.5, 2.5, 5.5, 9.5),
    'hours_since_last_txn': (1., 6., 24.),
}
STRENGTH = .5
SEED = 27481
MIN_ROWS, MIN_INFORMATION = 100, 6.


def fit_grid(fit, other, fields):
    encoded = {}
    for name in fields:
        if name in FIXED_EDGES or name in QUANTILE_FIELDS:
            edges = np.asarray(FIXED_EDGES[name]) if name in FIXED_EDGES else np.unique(np.quantile(fit[name], [.10, .30, .70, .90]))
            limits = [-np.inf, *edges, np.inf]
            levels = [f'[{a:.6g}, {b:.6g})' for a, b in zip(limits[:-1], limits[1:], strict=True)]
            codes = [np.searchsorted(edges, x[name], side='right') for x in (fit, other)]
        else:
            categories = pd.Index(fit[name].unique()).sort_values()
            levels = [str(c) for c in categories] + ['unseen']
            codes = [categories.get_indexer(x[name]) % len(levels) for x in (fit, other)]
        encoded[name] = (*codes, levels)
    return encoded


def keys_for(fields, encoded):
    axes = [encoded[name] for name in fields]
    levels = [a[2] for a in axes]
    shape = tuple(len(level) for level in levels)
    dev = np.ravel_multi_index(tuple(a[0] for a in axes), shape)
    destination = np.ravel_multi_index(tuple(a[1] for a in axes), shape)
    grid = np.column_stack(np.unravel_index(np.arange(np.prod(shape)), shape))
    return dev, destination, levels, grid


def estimate(grad, information, count, grid):
    keep = (count >= MIN_ROWS) & (information >= MIN_INFORMATION)
    effect = np.zeros_like(grad)
    if not keep.any():
        return effect, keep, 0
    nodes = grid[keep]
    pairs = []
    for i, j in combinations(range(3), 2):
        _, columns = np.unique(nodes[:, [i, j]], axis=0, return_inverse=True)
        pairs.append(np.eye(columns.max() + 1)[columns])
    fit = np.column_stack(pairs)
    w = np.sqrt(information[keep])
    design = fit * w[:, None]

    def remove_pairs(values):
        coefficients, _, rank, _ = lstsq(design, w * values, lapack_driver='gelsy')
        return values - fit @ coefficients, int(len(values) - rank)

    detail, dimension = remove_pairs(grad[keep] / information[keep])
    if dimension:
        regularized = detail * information[keep] / (information[keep] + 35.)
        centered, _ = remove_pairs(regularized)
        effect[keep] = centered * min(1., .18 / max(np.abs(centered).max(), 1e-12))
    return effect, keep, dimension


def group(keys, labels, offset, rows, size):
    return group_totals(keys[rows], labels[rows], expit(offset[rows]), size)


def predict_effect(table, keys):
    return STRENGTH * table[keys]


def gain(y, p, z):
    delta = np.logaddexp(0., p) - np.logaddexp(0., p + z) + y * z
    average = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    ci = [float((average - 1.96 * se) * 1e6), float((average + 1.96 * se) * 1e6)]
    return {'gain_logloss_x1m': average * 1e6, 'ci95_x1m': ci, 'conditional_p_one_sided': float(norm.sf(average / se)) if se else 1.,
            'rows': len(y), 'fraud': int(y.sum()), 'affected': int(np.count_nonzero(z))}


def size_shift(dev, test, size):
    train_freq = np.bincount(dev, minlength=size) / len(dev)
    test_freq = np.bincount(test, minlength=size) / len(test)
    return float(.5 * np.abs(train_freq - test_freq).sum())


def split(labels):
    split1 = StratifiedShuffleSplit(1, test_size=.3, random_state=SEED)
    screening, confirm = next(split1.split(labels, labels))
    split2 = StratifiedShuffleSplit(1, test_size=3/7, random_state=SEED + 1)
    first, second = next(split2.split(labels[screening], labels[screening]))
    result = screening[first], screening[second], confirm
    assert np.unique(np.concatenate(result)).size == len(labels)
    return result


def profile(cell, grid, levels, table, key, y, p):
    rows = key == cell
    return {'levels': [vals[i] for vals, i in zip(levels, grid[cell], strict=True)],
            'delta_logodds': STRENGTH * float(table[cell]), 'rows_c': int(rows.sum()),
            'fraud_c': int(y[rows].sum()), 'expected_fraud_c': float(expit(p[rows]).sum())}


def holm(records):
    running = 0.
    for number, row in enumerate(sorted(records, key=lambda r: r['confirm_c']['conditional_p_one_sided'])):
        effect = row['confirm_c']
        running = max(running, (len(records) - number) * effect['conditional_p_one_sided'])
        effect['holm_p_one_sided'] = min(1., running)


def run():
    ns, _, test, _, hashes, frame, y, dev, held, base, _, _, exact = build_inputs('development')
    development, holdout = frame.iloc[dev], frame.iloc[held]
    other = pd.concat([holdout, test], ignore_index=True)
    labels = y[dev]
    a, b, c = split(labels)
    ab = np.concatenate((a, b))
    encoded = fit_grid(development, other, {v for fields in TRIPLES for v in fields})
    screen = []
    candidates = []
    with threadpool_limits(limits=2):
        for fields in TRIPLES:
            key, external, levels, grid = keys_for(fields, encoded)
            size = len(grid)
            effects_a, support_a, rank_a = estimate(*group(key, labels, base, a, size), grid)
            effects_b, support_b, rank_b = estimate(*group(key, labels, base, b, size), grid)
            one = gain(labels[b], base[b], predict_effect(effects_a, key[b]))
            two = gain(labels[a], base[a], predict_effect(effects_b, key[a]))
            common = support_a & support_b
            correlation = None
            if common.sum() > 4 and np.ptp(effects_a[common]) > 0 and np.ptp(effects_b[common]) > 0:
                correlation = float(spearmanr(effects_a[common], effects_b[common]).statistic)
            row = {'fields': fields, 'cells': size, 'support_a': int(support_a.sum()), 'support_b': int(support_b.sum()),
                   'pure_rank_a': rank_a, 'pure_rank_b': rank_b, 'common_support': int(common.sum()),
                   'a_to_b': one, 'b_to_a': two, 'effect_spearman': correlation,
                   'screen_score': min(one['gain_logloss_x1m'], two['gain_logloss_x1m']),
                   'dev_test_total_variation': size_shift(key, external[len(held):], size)}
            screen.append(row)
            if row['screen_score'] > 0. and min(one['affected'], two['affected']) >= 3500 and (correlation or 0.) > 0:
                candidates.append((row, key, external, levels, grid))

        # Lock at most three candidates by both discovery directions before opening C.
        candidates.sort(key=lambda item: item[0]['screen_score'], reverse=True)
        locked = candidates[:3]
        for row, key, external, levels, grid in locked:
            effect, supported, dim = estimate(*group(key, labels, base, ab, len(grid)), grid)
            row['confirm_c'] = gain(labels[c], base[c], predict_effect(effect, key[c]))
            row['confirm_ab_pure_rank'] = dim
            row['test_supported_ab_fraction'] = float(np.mean(supported[external[len(held):]]))
            top = np.argsort(-np.abs(effect), kind='stable')[:5]
            row['strongest_ab_cells_scored_on_c'] = [profile(i, grid, levels, effect, key[c], labels[c], base[c])
                                                      for i in top if effect[i] != 0.]
        holm([k[0] for k in locked])
        for row, key, external, _, grid in locked:
            # An explicitly descriptive check on the historically reused holdout.
            effect, _, _ = estimate(*group(key, labels, base, np.arange(len(labels)), len(grid)), grid)
            shift = predict_effect(effect, external[:len(held)])
            row['frozen_ensemble_holdout'] = {**assess(ns, y[held], exact + shift, exact), **gain(y[held], exact, shift)}

    screen.sort(key=lambda row: row['screen_score'], reverse=True)
    metadata = {'input_sha256': hashes, 'holdout_previously_used': True, 'strength': STRENGTH, 'seed': SEED,
                'fold_roles': {key: {'rows': len(idx), 'fraud': int(labels[idx].sum())}
                               for key, idx in zip(('a', 'b', 'c'), (a, b, c), strict=True)},
                'holdout_rows': len(held), 'holdout_fraud': int(y[held].sum()),
                'discovery_gate': {'min_rows_cell': MIN_ROWS, 'min_information_cell': MIN_INFORMATION,
                                   'min_affected_rows_each_direction': 3500, 'max_locked': 3},
                'triple_results': screen}
    write_json(HERE / 'triple_eda_results.json', metadata)
    for row in screen:
        print(row['fields'], 'A->B', round(row['a_to_b']['gain_logloss_x1m'], 2),
              'B->A', round(row['b_to_a']['gain_logloss_x1m'], 2),
              'C', row.get('confirm_c', {}).get('gain_logloss_x1m'),
              'H', row.get('frozen_ensemble_holdout', {}).get('gain_logloss_x1m'), flush=True)


if __name__ == '__main__':
    run()
