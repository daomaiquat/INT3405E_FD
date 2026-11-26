"""Test prespecified transaction regimes against local out-of-fold Grandmaster scores."""

from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path[:0] = [str(PARENT)]
from evaluate_and_export import assess, read_scores
from grandmaster_inputs import archive_functions, design, development_caches, meta_predictions, official_data, write_json


RISK_EDGES = logit([.001, .005, .02, .10, .30])
RISK_WIDTH = len(RISK_EDGES) + 1
MIN_GROUP = 150
MIN_RISK_GROUP = 200
MIN_EXPECTED = 5


def numeric_bins(values, thresholds):
    endpoints = ['-inf', *map(str, thresholds), 'inf']
    labels = [f'[{lo},{hi})' for lo, hi in zip(endpoints[:-1], endpoints[1:], strict=True)]
    return np.searchsorted(thresholds, values, side='right'), labels


def categorical_bins(values, development):
    observed = sorted(values.iloc[development].unique())
    codes = pd.Categorical(values, categories=observed).codes.astype(np.int64)
    codes[codes < 0] = len(observed)
    return codes, [*map(str, observed), 'unseen']


def joint(*features):
    components, levels = zip(*features, strict=True)
    ids = np.ravel_multi_index(components, tuple(len(names) for names in levels))
    group_names = [' × '.join(names) for names in product(*levels)]
    return ids, group_names


def transaction_groups(frame, development):
    amount = frame.amount_usd.to_numpy()
    balance = frame.account_balance_usd.to_numpy()
    gap = frame.hours_since_last_txn.to_numpy()
    count = frame.txn_count_last_24h.to_numpy()
    hour = frame.time_of_day_hour.to_numpy()
    day = frame.day_of_week.to_numpy()
    if np.any(balance <= 0) or not set(np.unique(hour)) <= set(range(24)) or not set(np.unique(day)) <= set(range(7)):
        raise ValueError('Expected positive balance and integer clock/day codes in their documented ranges')

    # Assume the integer clock hour covers [hour,hour+1). Gap is rounded to cents of an hour.
    previous_early = hour - gap - .005
    previous_late = hour + 1 - gap + .005
    early_block = np.floor(previous_early / 4).astype(int) % 6
    late_block = np.floor(previous_late / 4).astype(int) % 6
    four_hour_names = [f'{start:02d}h-{start + 4:02d}h' for start in range(0, 24, 4)]
    previous_block = np.where(early_block == late_block, early_block, 6)

    def calendar_phase(previous):
        day_delta = np.floor(previous / 24).astype(int)
        return np.where(day_delta == 0, 0, np.where(day + day_delta >= 0, 1, 2))

    early_phase = calendar_phase(previous_early)
    later_phase = calendar_phase(previous_late)
    day_transition = np.where(early_phase == later_phase, early_phase, 3)
    # For current hours in [00,06), reaching the same night depends on crossing midnight.
    night_transition = np.where(hour >= 6, 0, np.where(previous_early >= 0., 1, np.where(previous_late < 0., 2, 3)))
    night_count, night_count_names = numeric_bins(count, [2, 4])
    night_count_group = np.where(night_transition == 0, 0, 1 + (night_transition - 1) * 3 + night_count)
    night_count_labels = ['current hour outside [00h,06h)'] + [
        f'{phase} × count {count_name}'
        for phase in ('same day', 'previous day', 'ambiguous midnight') for count_name in night_count_names
    ]

    authentication = categorical_bins(frame.auth_method, development)
    channel = categorical_bins(frame.channel, development)
    card_type = categorical_bins(frame.card_type, development)
    amount_share = amount / balance
    amount_share_bins = numeric_bins(amount_share, [.01, .05, .20, 1.])
    return {
        'clock_block_pair': joint((hour // 4, four_hour_names),
                                  (previous_block, [*four_hour_names, 'ambiguous previous block'])),
        'encoded_week_transition_channel': joint((day_transition, ['same day', 'prior day within encoded week',
                                                                'wrapped encoded week', 'ambiguous phase']), channel),
        'night_continuity_count': (night_count_group, night_count_labels),
        'rapid_amount_new_merchant': joint(numeric_bins(gap, [1., 6.]), numeric_bins(amount, [20., 100., 400.]),
                                          (frame.is_new_merchant.to_numpy(int), ['known merchant', 'new merchant'])),
        'amount_count_relative_to_balance': numeric_bins(amount_share * count, [.01, .05, .20, .50, 1.]),
        'remaining_balance_card_type': joint(numeric_bins(balance - amount, [0., 100., 500., 2000.]), card_type),
        'spend_share_auth_address': joint(amount_share_bins, authentication,
                                         (frame.billing_shipping_mismatch.to_numpy(int), ['address match', 'address mismatch'])),
        'short_gap_distance_channel': joint(numeric_bins(gap, [1., 6.]),
                                            numeric_bins(frame.distance_from_home_km.to_numpy(), [5., 20.]), channel),
        'spend_share_card_tenure': joint(amount_share_bins, numeric_bins(frame.card_age_months.to_numpy(), [40, 50, 60])),
    }


def totals(ids, values, width):
    return np.bincount(ids, weights=values, minlength=width)


def fit_offsets(base, y, keys, width, ridge, minimum_rows, minimum_expected):
    n = totals(keys, None, width)
    expected = totals(keys, expit(base), width)
    active = (n >= minimum_rows) & (expected >= minimum_expected)
    correction = np.zeros(width)
    for _ in range(8):
        p = expit(base + correction[keys])
        gradient = totals(keys, y - p, width) - ridge * correction
        curvature = totals(keys, p * (1 - p), width) + ridge
        correction = np.clip(correction + np.where(active, gradient / curvature, 0.), -.25, .25)
    return correction


def combined_keys(groups, bands):
    return groups * RISK_WIDTH + bands


def fit_regime(base, y, groups, bands, width):
    offset = fit_offsets(base, y, groups, width, 45, MIN_GROUP, MIN_EXPECTED)
    interactions = fit_offsets(base + offset[groups], y, combined_keys(groups, bands),
                               width * RISK_WIDTH, 80, MIN_RISK_GROUP, MIN_EXPECTED)
    return offset, interactions


def apply_regime(main, interaction, groups, bands):
    return main[groups] + interaction[combined_keys(groups, bands)]


def row_loss(y, logits):
    return np.logaddexp(0., logits) - y * logits


def metrics(y, base, improved):
    delta_loss = row_loss(y, base) - row_loss(y, improved)
    error = np.std(delta_loss, ddof=1) / np.sqrt(len(y))
    base_auc = roc_auc_score(y, base)
    new_auc = roc_auc_score(y, improved)
    return {
        'base_auc': float(base_auc), 'new_auc': float(new_auc), 'delta_auc': float(new_auc - base_auc),
        'delta_ap': float(average_precision_score(y, improved) - average_precision_score(y, base)),
        'logloss_gain_x1m': float(delta_loss.mean() * 1e6),
        'descriptive_logloss_gain_ci95_x1m': list(map(float, 1e6 * (delta_loss.mean() + np.array([-1.96, 1.96]) * error))),
    }


def base_residual_cells(groups, names, labels, control_scores, two_parts, test_groups):
    width = len(names)
    counts = [totals(groups[part], None, width) for part in two_parts]
    observed = [totals(groups[part], labels[part], width) for part in two_parts]
    p = expit(control_scores)
    expected = [totals(groups[part], p[part], width) for part in two_parts]
    variance = [totals(groups[part], p[part] * (1 - p[part]), width) for part in two_parts]
    z = [(actual - predicted) / np.sqrt(np.maximum(var, 1e-12))
         for actual, predicted, var in zip(observed, expected, variance, strict=True)]
    test_count = totals(test_groups, None, width)
    rows = []
    for i in range(width):
        if not (counts[0][i] >= MIN_GROUP and counts[1][i] >= MIN_GROUP
                and expected[0][i] >= MIN_EXPECTED and expected[1][i] >= MIN_EXPECTED):
            continue
        rows.append({'group': names[i], 'dev_half_rows': [int(n[i]) for n in counts],
                     'dev_half_fraud': [int(n[i]) for n in observed],
                     'dev_half_expected': [float(n[i]) for n in expected],
                     'dev_half_nominal_z': [float(n[i]) for n in z],
                     'same_direction': bool(z[0][i] * z[1][i] > 0),
                     'dev_share': float((counts[0][i] + counts[1][i]) / len(groups)),
                     'test_share': float(test_count[i] / len(test_groups))})
    rows.sort(key=lambda r: min(map(abs, r['dev_half_nominal_z'])) if r['same_direction'] else -1, reverse=True)
    return rows


def distribution(groups, test_groups, width):
    n, m = len(groups), len(test_groups)
    counts = totals(groups, None, width)
    test_counts = totals(test_groups, None, width)
    abs_gap = np.abs(counts / n - test_counts / m)
    return {'dev_groups': int(np.count_nonzero(counts)), 'test_groups': int(np.count_nonzero(test_counts)),
            'test_rows_unseen_share': float(test_counts[counts == 0].sum() / m),
            'test_rows_low_support_share': float(test_counts[counts < MIN_GROUP].sum() / m),
            'total_variation': float(.5 * abs_gap.sum()),
            'max_group_share_diff': float(abs_gap.max())}


def run():
    train, test, _, hashes = official_data()
    labels, dev, held, folds, raw_dev, raw_held, dense_dev, dense_held, metadata = development_caches(train, hashes)
    ns = archive_functions()
    with threadpool_limits(limits=3):
        dev_base, held_base, _ = meta_predictions(ns, labels[dev], design(ns, raw_dev, dense_dev),
                                                  design(ns, raw_held, dense_held), folds,
                                                  metadata['baseline_meta_l2_by_fold'])
    ids = train.id.to_numpy()[held]
    expected = read_scores(PARENT / 'residual_development.npz', ids, 'ids', ('grandmaster_hierarchical',))
    np.testing.assert_allclose(held_base, expected['grandmaster_hierarchical'], atol=2e-8, rtol=0)
    y = labels[dev]
    halves = tuple(np.sort(np.concatenate([fold[1] for fold in folds[start::2]])) for start in (0, 1))
    np.testing.assert_array_equal(np.sort(np.concatenate(halves)), np.arange(len(dev)))
    dev_bands = np.searchsorted(RISK_EDGES, dev_base)
    control_oof = np.empty(len(dev))
    directions = []
    for fitting, query in (halves, halves[::-1]):
        controls = fit_offsets(dev_base[fitting], y[fitting], dev_bands[fitting], RISK_WIDTH, 35, 1, 1)
        control_oof[query] = dev_base[query] + controls[dev_bands[query]]
        directions.append((fitting, query, dev_base[fitting] + controls[dev_bands[fitting]]))

    groups_all = transaction_groups(pd.concat([train.drop(columns='is_fraud'), test], ignore_index=True), dev)
    audit = {}
    frozen = []
    for name, (groups, group_names) in groups_all.items():
        dev_g, test_g, width = groups[dev], groups[len(train):], len(group_names)
        candidate_oof = control_oof.copy()
        for fitting, query, fbase in directions:
            main, interaction = fit_regime(fbase, y[fitting], dev_g[fitting], dev_bands[fitting], width)
            candidate_oof[query] += apply_regime(main, interaction, dev_g[query], dev_bands[query])
        per_half = [metrics(y[part], control_oof[part], candidate_oof[part]) for part in halves]
        pooled = metrics(y, control_oof, candidate_oof)
        shift = distribution(dev_g, test_g, width)
        supported = shift['test_rows_low_support_share'] < .005 and shift['total_variation'] < .02
        auc_pass = (supported and pooled['delta_auc'] >= .00001 and pooled['logloss_gain_x1m'] >= 0
                    and min(row['delta_auc'] for row in per_half) > 0)
        loss_pass = (supported and pooled['logloss_gain_x1m'] >= 1.
                     and min(row['logloss_gain_x1m'] for row in per_half) > 0)
        if auc_pass or loss_pass:
            frozen.append(name)
        audit[name] = {'distribution': shift, 'dev_halves': per_half, 'dev_pooled': pooled,
                       'calibration_gate_passes': loss_pass, 'auc_gate_passes': auc_pass,
                       'supported_cells': base_residual_cells(dev_g, group_names, y, control_oof, halves, test_g)}
        print(f'{name}: AUC {pooled["delta_auc"]:+.8f}, log loss gain {pooled["logloss_gain_x1m"]:+.3f}e-6, '
              f'development pass = {auc_pass or loss_pass}', flush=True)

    held_audit = {}
    if frozen:
        current = read_scores(PARENT / 'followup' / 'context_holdout.npz', ids, 'ids',
                              ('expanded_half_density',))['expanded_half_density']
        eda_probs = read_scores(PARENT / 'selected_predictions.npz', ids, 'holdout_ids', ('holdout',))['holdout']
        if np.any((eda_probs <= 0.) | (eda_probs >= 1.)):
            raise ValueError('Grandmaster EDA holdout probabilities must be in (0,1)')
        eda = logit(eda_probs)
        controls = fit_offsets(dev_base, y, dev_bands, RISK_WIDTH, 35, 1, 1)
        held_bands = np.searchsorted(RISK_EDGES, held_base)
        held_control = held_base + controls[held_bands]
        for name in frozen:
            groups, names = groups_all[name]
            main, interaction = fit_regime(dev_base + controls[dev_bands], y, groups[dev], dev_bands, len(names))
            correction = apply_regime(main, interaction, groups[held], held_bands)
            measures = {}
            for baseline, name_base in ((held_control, 'grandmaster_risk_band_control'),
                                         (held_base, 'grandmaster_raw'), (eda, 'grandmaster_eda'),
                                         (current, 'mixed_density')):
                augmented = baseline + correction
                measures[name_base] = {**metrics(labels[held], baseline, augmented),
                                       'paired_delong': assess(ns, labels[held], augmented, baseline)}
            strongest_dev_cell = audit[name]['supported_cells'][0]['group']
            in_group = groups[held] == names.index(strongest_dev_cell)
            measures['strongest_dev_cell_on_holdout'] = {
                'group': strongest_dev_cell, 'rows': int(in_group.sum()), 'fraud': int(labels[held][in_group].sum()),
                'expected_grandmaster_band_control': float(expit(held_control[in_group]).sum()),
                'expected_grandmaster_eda': float(eda_probs[in_group].sum()),
                'expected_mixed_density': float(expit(current[in_group]).sum()),
            }
            held_audit[name] = measures

    write_json(HERE / 'behavior_audit.json', {
        'input_sha256': hashes, 'dev_rows': len(dev), 'dev_fraud': int(y.sum()),
        'dev_halves_original_fold_numbers': [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]],
        'dev_half_rows': [len(part) for part in halves], 'dev_half_fraud': [int(y[part].sum()) for part in halves],
        'development_raw_baseline': metrics(y, dev_base, control_oof),
        'risk_probability_band_edges': list(map(float, expit(RISK_EDGES))),
        'gate': 'Test low support <0.5% and TV <2%; AUC gain >=1e-5 pooled, positive in each half '
                'and pooled log-loss gain >=0, or pooled log-loss gain >=1e-6 and positive in each half.',
        'test_rows': len(test), 'reused_holdout_rows': len(held), 'reused_holdout_fraud': int(labels[held].sum()),
        'holdout_candidates_frozen_after_development': frozen, 'hypotheses': audit, 'reused_holdout': held_audit,
    })


if __name__ == '__main__':
    run()
