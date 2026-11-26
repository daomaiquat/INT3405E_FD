"""Profile official S6E11 data and screen recurring residual structure on development only."""

from itertools import combinations
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

from grandmaster_inputs import (HERE, PROJECT, archive_functions, design, development_caches,
                                meta_predictions, official_data, read_json, write_json)


NUMERIC_GRIDS = {
    'amount_usd': 100, 'hours_since_last_txn': 100, 'distance_from_home_km': 100,
    'account_balance_usd': 100, 'velocity_score': 10, 'merchant_risk_score': 10,
}


def bucket_codes(series):
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) and series.nunique() > 25:
        edges = np.unique(np.quantile(series, [.02, .05, .10, .20, .35, .50, .65, .8, .9, .95, .98]))
        codes = np.searchsorted(edges, series, side='left')
        return codes, len(edges) + 1
    codes, levels = pd.factorize(series, sort=True)
    return codes, len(levels)


def group_totals(keys, labels, probabilities, size):
    residual = np.bincount(keys, weights=labels - probabilities, minlength=size)
    information = np.bincount(keys, weights=probabilities * (1 - probabilities), minlength=size)
    counts = np.bincount(keys, minlength=size)
    return residual, information, counts


def centered_pair(gradient, information):
    """Remove additive main effects using weighted least squares on a small table."""
    row_bias = np.zeros(gradient.shape[0])
    col_bias = np.zeros(gradient.shape[1])
    for _ in range(12):
        row_bias = (gradient.sum(axis=1) - information @ col_bias) / (information.sum(axis=1) + 2)
        col_bias = (gradient.sum(axis=0) - information.T @ row_bias) / (information.sum(axis=0) + 2)
    return gradient - information * (row_bias[:, None] + col_bias[None, :])


def repeatability(left, right, support=4):
    g0, h0, n0 = left
    g1, h1, n1 = right
    keep = (h0 >= support) & (h1 >= support) & (n0 >= 40) & (n1 >= 40)
    if keep.sum() < 5:
        return {'supported_cells': int(keep.sum()), 'spearman': None, 'cross_delta_logloss_x1m': None}
    z0, z1 = g0[keep] / np.sqrt(h0[keep]), g1[keep] / np.sqrt(h1[keep])
    a0, a1 = g0[keep] / (h0[keep] + 25), g1[keep] / (h1[keep] + 25)
    gain = (np.sum(g1[keep] * a0 - 0.5 * h1[keep] * a0**2)
            + np.sum(g0[keep] * a1 - 0.5 * h0[keep] * a1**2)) / (n0.sum() + n1.sum())
    return {'supported_cells': int(keep.sum()), 'spearman': float(spearmanr(z0, z1).statistic),
            'cross_delta_logloss_x1m': float(1e6 * gain)}


def split_tables(keys, size, splits, labels, probabilities):
    return tuple(group_totals(keys[rows], labels[rows], probabilities[rows], size) for rows in splits)


def scan_pairs(features, splits, labels, probabilities):
    coded = {name: bucket_codes(features[name]) for name in features}
    results = []
    for left, right in combinations(coded, 2):
        a, a_size = coded[left]
        b, b_size = coded[right]
        cell = a * b_size + b
        scores = []
        for g, h, n in split_tables(cell, a_size * b_size, splits, labels, probabilities):
            scores.append((centered_pair(g.reshape(a_size, b_size), h.reshape(a_size, b_size)).ravel(), h, n))
        results.append({'left': left, 'right': right, **repeatability(*scores)})
    return sorted(results, key=lambda row: -np.inf if row['cross_delta_logloss_x1m'] is None
                  else row['cross_delta_logloss_x1m'], reverse=True)


def scan_exact(features, splits, labels, probabilities):
    records = []
    for name, scale in NUMERIC_GRIDS.items():
        ticks = np.rint(scale * features[name].to_numpy()).astype(np.int64)
        for representation, key, parent in (
            ('exact', ticks, ticks // scale), ('last_digit', ticks % 10, None),
            ('fraction', ticks % scale, (ticks // scale) // 10),
        ):
            levels, codes = np.unique(key, return_inverse=True)
            halves = []
            for rows in splits:
                g, h, n = group_totals(codes[rows], labels[rows], probabilities[rows], len(levels))
                if parent is not None and representation == 'exact':
                    _, group = np.unique(levels // scale, return_inverse=True)
                    gh = np.bincount(group, weights=g)[group] - g
                    hh = np.bincount(group, weights=h)[group] - h
                    g = g - h * gh / (hh + 20)
                halves.append((g, h, n))
            records.append({'feature': name, 'key': representation, **repeatability(*halves)})
    return records


def risk_blocks(labels, probabilities):
    negative = np.sort(probabilities[labels == 0])
    bounds = np.quantile(probabilities, [0, .5, .75, .9, .95, .99, 1])
    out = []
    for i, (left, right) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        rows = (probabilities >= left) & ((probabilities < right) if i < 5 else (probabilities <= right))
        pos = probabilities[rows & (labels == 1)]
        missed_pairs = ((len(negative) - np.searchsorted(negative, pos, side='right')) / len(negative)).sum()
        out.append({'min_score': float(left), 'max_score': float(right), 'rows': int(rows.sum()),
                    'actual_fraud': int(pos.size), 'expected_fraud': float(probabilities[rows].sum()),
                    'fraction_of_all_pairs_misordered': float(missed_pairs / labels.sum())})
    return out


def plot_eda(univariate, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), layout='constrained')
    univariate = univariate[:8][::-1]
    axes[0].barh([r['feature'].replace('_', ' ') for r in univariate], [r['oof_auc'] for r in univariate],
                 color='#2b7773')
    axes[0].set_xlim(.5, .76)
    axes[0].set_title('Raw feature signal on 5 folds')
    axes[0].set_xlabel('ROC AUC')
    strongest = pairs[:8][::-1]
    axes[1].barh([f"{r['left']} x {r['right']}".replace('_', ' ') for r in strongest],
                 [r['cross_delta_logloss_x1m'] for r in strongest], color='#31558b')
    axes[1].set_title('Residual interaction discovery')
    axes[1].set_xlabel('Reciprocal half-split log-loss gain × 1,000,000')
    fig.savefig(HERE / 'eda_overview.png', dpi=150)
    plt.close(fig)


def run():
    sys.path.insert(0, str(PROJECT / 'eda_review'))
    import data_audit
    import univariate_signal

    # Keep the existing reports intact while recomputing both analyses from source CSVs.
    data_audit.HERE = HERE
    univariate_signal.HERE = HERE
    data_audit.main()
    univariate_signal.main()
    train, _, _, hashes = official_data()
    y, dev, _, folds, raw, held_raw, dense, held_dense, metadata = development_caches(train, hashes)
    ns = archive_functions()
    with threadpool_limits(limits=4):
        p, _, _ = meta_predictions(ns, y[dev], design(ns, raw, dense), design(ns, held_raw, held_dense), folds,
                                   metadata['baseline_meta_l2_by_fold'])
    probabilities = expit(p)
    labels = y[dev]
    features = train.iloc[dev].drop(columns=['id', 'is_fraud']).reset_index(drop=True)
    splits = next(StratifiedShuffleSplit(1, test_size=.5, random_state=731).split(labels, labels))
    pairs = scan_pairs(features, splits, labels, probabilities)
    exact = scan_exact(features, splits, labels, probabilities)
    ranges = np.array_split(np.arange(len(train)), 20)
    results = {'input_sha256': hashes, 'discovery_rows': len(dev), 'discovery_fraud': int(labels.sum()),
               'discovery_meta_oof_auc': float(roc_auc_score(labels, probabilities)),
               'raw_id_block_prevalence': [float(y[rows].mean()) for rows in ranges],
               'discovery_seed': 731, 'risk_blocks': risk_blocks(labels, probabilities),
               'pair_results': pairs, 'exact_value_results': exact}
    write_json(HERE / 'residual_eda.json', results)
    plot_eda(read_json(HERE / 'univariate_signal.json')['features'], pairs)
    for row in pairs[:12]:
        print(row, flush=True)


if __name__ == '__main__':
    run()
