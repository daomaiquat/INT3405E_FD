"""Audit fixed amount-hole groups without changing the selection or correction."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import logit

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
MEASUREMENT = PARENT / 'deep_eda_measurement'
sys.path.insert(0, str(MEASUREMENT))
from measurement_eda import changes, density_rules
from evaluate_and_export import auc_by_base_score_band, read_scores
from grandmaster_inputs import development_caches, official_data, read_json, write_json

SEED = 27482
RANDOM_DRAWS = 2999


def amount_ticks(frame):
    values = frame.amount_usd.to_numpy()
    ticks = np.rint(100 * values).astype(np.int64)
    return ticks, np.abs(100 * values - ticks) > 1e-6


def carryover(points, ticks, width):
    ordered = np.sort(ticks)
    exact = np.searchsorted(ordered, points, side='right') - np.searchsorted(ordered, points, side='left')
    span = np.searchsorted(ordered, points + width, side='right') - np.searchsorted(ordered, points - width, side='left')
    average = (span - exact) / (2 * width)
    return {'observed_counts': int(exact.sum()), 'neighbor_expected_counts': float(average.sum()),
            'ratio': float(exact.sum() / average.sum()), 'points_below_their_neighbor_mean': int((exact < average).sum())}


def matching(residuals, mask, score, amounts, match_last_digit=False):
    score_edges = np.quantile(score, np.arange(1, 10) / 10.)
    score_bands = np.searchsorted(score_edges, score, side='right')
    groups = ((amounts // 1000) * 10 + score_bands).astype(np.int64)
    if match_last_digit:
        groups = groups * 10 + amounts % 10
    rng = np.random.default_rng(SEED)
    null = np.zeros(RANDOM_DRAWS)
    selected = 0
    group_count = 0
    for group in np.unique(groups[mask]):
        rows = groups == group
        k = int(mask[rows].sum())
        values = residuals[rows]
        if k == len(values):
            null += values.sum()
            continue
        group_count += 1
        selected += k
        permuted = rng.random((RANDOM_DRAWS, len(values)))
        choices = np.argpartition(permuted, k - 1, axis=1)[:, :k]
        null += values[choices].sum(axis=1)
    actual = float(residuals[mask].sum())
    return {'seed': SEED, 'draws': RANDOM_DRAWS, 'match_last_digit': match_last_digit,
            'movable_strata': group_count, 'movable_rows': selected,
            'observed_residual_fraud': actual, 'null_residual_fraud_mean': float(null.mean()),
            'null_residual_fraud_ci95': np.quantile(null, [.025, .975]).tolist(),
            'conditional_one_sided_p': float((1 + np.sum(null >= actual)) / (len(null) + 1))}


def group_sensitivity(ticks, y, probability, offset, mask, shift):
    points, positions = np.unique(ticks[mask], return_inverse=True)
    residual_per_price = np.bincount(positions, weights=y[mask] - probability[mask])
    delta = -changes(y[mask], offset[mask], np.full(mask.sum(), shift))
    gain_per_price = np.bincount(positions, weights=delta)
    order = np.argsort(-gain_per_price)
    rng = np.random.default_rng(SEED + 1)
    boot = rng.integers(len(points), size=(9999, len(points)))
    bounds = np.quantile((gain_per_price[boot].sum(axis=1) / len(y)) * 1e6, [.025, .975])
    leave_one_out = (gain_per_price.sum() - gain_per_price) / len(y) * 1e6
    return {'observed_price_levels': len(points), 'bootstrap_draws': len(boot),
            'cluster_bootstrap_gain_logloss_ci95_x1m': bounds.tolist(),
            'leave_one_price_out_min_gain_logloss_x1m': float(leave_one_out.min()),
            'leave_one_price_out_max_gain_logloss_x1m': float(leave_one_out.max()),
            'most_influential_prices': [{
                'usd': float(points[i]) / 100, 'fraud_residual': float(residual_per_price[i]),
                'gain_logloss_x1m': float(gain_per_price[i] / len(y) * 1e6)} for i in order[:5]]}


def run():
    train, test, _, hashes = official_data()
    y, dev, held, *_ = development_caches(train, hashes)
    record = read_json(MEASUREMENT / 'measurement_metrics.json')
    assert record['input_sha256'] == hashes
    finalist = 'amount_usd_hole'
    assert finalist in record['finalists']
    frame = pd.concat([train[['amount_usd']], test[['amount_usd']]], ignore_index=True)
    ticks, off_grid = amount_ticks(frame)
    _, _, mask, definition = next(row for row in density_rules('amount_usd', {'ticks': ticks, 'off_grid': off_grid}, dev)
                                  if row[1] == finalist)
    frozen = record['groups'][finalist]
    assert int(mask[dev].sum()) == frozen['dev']['rows']
    selected = np.unique(ticks[dev][mask[dev]])
    split_rows = {'development': dev, 'holdout': held, 'test': len(train) + np.arange(len(test))}
    transfer = {}
    for split, rows in split_rows.items():
        aligned = ticks[rows][~off_grid[rows]]
        transfer[split] = {str(width): carryover(selected, aligned, width) for width in (5, 25)}
    hold = read_scores(PARENT / 'selected_predictions.npz', train.id.to_numpy()[held], 'holdout_ids', ('holdout',))
    probability = hold['holdout']
    target = mask[held]
    condition = record['holdout']['groups'][finalist]
    anchor = condition['anchor_group']
    assert int(target.sum()) == anchor['rows'] and int(y[held][target].sum()) == anchor['actual_fraud']
    np.testing.assert_allclose(probability[target].sum(), anchor['expected_fraud'], atol=1e-11, rtol=0.)
    sensitivity = group_sensitivity(ticks[held], y[held], probability, logit(probability), target, condition['dev_shift'])
    matching_result = matching(y[held] - probability, target, probability, ticks[held])
    digit_match = matching(y[held] - probability, target, probability, ticks[held], match_last_digit=True)
    base = logit(probability)
    bands = auc_by_base_score_band(y[held], base, base + target * condition['dev_shift'])
    np.testing.assert_allclose(sum(row['contribution_to_auc_delta'] for row in bands),
                              condition['exact_anchor_comparison']['delta_auc'], rtol=0, atol=1e-12)
    result = {'input_sha256': hashes, 'definition': definition, 'frozen_dev_shift': condition['dev_shift'],
              'neighbor_density_transfer': transfer, 'holdout_matched_price_and_risk_permutation': matching_result,
              'holdout_matched_price_risk_and_last_digit_permutation': digit_match,
              'holdout_auc_contribution_by_baseline_band': bands,
              'holdout_amount_level_sensitivity': sensitivity, 'holdout_previously_used': True}
    write_json(HERE / 'amount_hole_sensitivity.json', result)
    print(f'Frozen amount rule checked for {len(selected)} levels and {int(target.sum()):,} holdout rows')
    print(transfer)
    print(matching_result)
    print(digit_match)
    print(bands)
    print(sensitivity)


if __name__ == '__main__':
    run()
