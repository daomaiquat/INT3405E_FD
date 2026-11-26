"""Audit numeric grids, local point masses, and extreme regimes."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
GOAL = HERE.parent
sys.path.insert(0, str(GOAL))

from evaluate_and_export import assess, read_scores
from grandmaster_inputs import (
    archive_functions, design, development_caches, digest, meta_predictions,
    official_data, read_json, write_json,
)


SCALES = {
    'amount_usd': 100, 'hours_since_last_txn': 100, 'distance_from_home_km': 100,
    'account_balance_usd': 100, 'velocity_score': 10, 'merchant_risk_score': 10,
}
PRICES = ('amount_usd', 'account_balance_usd')
BOUNDARIES = {
    'amount_usd': ((1., 'floor', 1, 2),),
    'hours_since_last_txn': ((.01, 'floor', .01, .10),),
    'distance_from_home_km': ((0., 'floor', 0, .10),),
    'velocity_score': ((0., 'floor', 0, 1),),
    'merchant_risk_score': ((0., 'floor', 0, 1), (100., 'cap', 99, 100)),
    'customer_age': ((18., 'floor', None, None), (81., 'cap', None, None)),
}
RIDGE = 25.
CAP = .30


def decimal_data(frame, development):
    columns = {}
    any_off_grid, any_two_coarse, any_two_tail = (np.zeros(len(frame), dtype=np.uint8) for _ in range(3))
    for variable, scale in SCALES.items():
        values = frame[variable].to_numpy()
        ticks = np.rint(values * scale).astype(np.int64)
        off_grid = np.abs(values * scale - ticks) > 1e-6
        valid = (ticks > 0) & ~off_grid
        step, whole = (25, 100) if scale == 100 else (20, 50)
        coarse = valid & (ticks % step == 0)
        whole_grid = valid & (ticks % whole == 0)
        intermediate = coarse & ~whole_grid
        tails = np.quantile(values[development], [.005, .99, .999])
        columns[variable] = dict(
            values=values, ticks=ticks, off_grid=off_grid, intermediate=intermediate,
            whole=whole_grid, step=step / scale, whole_step=whole / scale, tails=tails,
        )
        any_off_grid += off_grid
        any_two_coarse += intermediate
        any_two_tail += values > tails[1]
    return columns, any_off_grid > 0, any_two_coarse >= 2, any_two_tail >= 2


def density_points(reference, width):
    """Means of all neighboring ticks, including those absent from the data."""
    points, counts = np.unique(reference, return_counts=True)
    cumulative = np.r_[0, np.cumsum(counts)]
    starts = np.searchsorted(points, points - width, side='left')
    stops = np.searchsorted(points, points + width, side='right')
    mean = (cumulative[stops] - cumulative[starts] - counts) / (2 * width)
    return points, counts, mean


def density_rules(variable, item, development):
    on_grid = ~item['off_grid']
    reference_rows = development[on_grid[development]]
    reference = item['ticks'][reference_rows]
    points, counts, short = density_points(reference, 5)
    _, _, wide = density_points(reference, 25)
    interior = (points > points[0] + 25) & (points < points[-1] - 25)
    short_ratio = (counts + 5.) / (short + 5.)
    wide_ratio = (counts + 5.) / (wide + 5.)
    peak = interior & (counts >= 30) & (short_ratio >= 3.) & (wide_ratio >= 3.)
    hole = (interior & (counts >= 10) & (short >= 40.) & (wide >= 40.)
            & (short_ratio <= .4) & (wide_ratio <= .4))
    for name, rule in (('peak', peak), ('hole', hole)):
        members = points[rule]
        examples = members[np.argsort(-counts[rule], kind='stable')[:6]]
        metadata = dict(
            selected_values=len(members),
            examples=(examples / SCALES[variable]).tolist(),
            neighborhood_tick_widths=[5, 25],
        )
        yield 'local_mass', f'{variable}_{name}', on_grid & np.isin(item['ticks'], members), metadata


def tail_rules(variable, item, development):
    values = item['values']
    lower, high, extreme = (float(x) for x in item['tails'])
    yield 'tail', f'{variable}_upper_shoulder', (values > high) & (values <= extreme), [high, extreme]
    yield 'tail', f'{variable}_upper_extreme', values > extreme, [extreme]
    yield 'tail', f'{variable}_lower_extreme', values <= lower, [lower]
    if variable in PRICES:
        points = np.unique(values[development])
        points = points[(points > high) & (points <= extreme)]
        gap = np.argmax(np.diff(np.log(points)))
        lo, hi = float(points[gap]), float(points[gap + 1])
        details = dict(quantiles=[high, extreme], gap=[lo, hi], relative_step=hi / lo - 1.)
        yield 'tail_gap', f'{variable}_below_largest_gap', (values > high) & (values <= lo), details
        yield 'tail_gap', f'{variable}_above_largest_gap', (values >= hi) & (values <= extreme), details


def rules(frame, development, columns, off_grid, two_coarse, two_tail):
    yield 'precision', 'any_off_nominal_grid', off_grid, {}
    for variable, item in columns.items():
        yield 'precision', f'{variable}_off_nominal_grid', item['off_grid'], {}
        yield 'grid', f'{variable}_intermediate_grid', item['intermediate'], [
            item['step'], item['whole_step'],
        ]
        yield 'grid', f'{variable}_whole_grid', item['whole'], [item['whole_step']]
        yield from density_rules(variable, item, development)
        yield from tail_rules(variable, item, development)

    yield 'co_grid', 'at_least_two_intermediate_grids', two_coarse, {}
    yield 'tail', 'at_least_two_upper_one_percent', two_tail, {}
    for left, right in (PRICES, ('hours_since_last_txn', 'distance_from_home_km'),
                        ('velocity_score', 'merchant_risk_score')):
        mask = columns[left]['intermediate'] & columns[right]['intermediate']
        yield 'co_grid', f'{left}_and_{right}', mask, {}

    amount = columns['amount_usd']
    amount_grid = amount['intermediate']
    median = float(np.median(amount['values'][development]))
    high_amount = amount['values'] > median
    for level, mask in (('lower', ~high_amount), ('upper', high_amount)):
        yield 'conditional_grid', f'amount_intermediate_grid_{level}_half', amount_grid & mask, [median]
    for channel in sorted(frame.iloc[development]['channel'].unique()):
        mask = frame['channel'].to_numpy() == channel
        yield 'conditional_grid', f'amount_intermediate_grid_channel_{channel}', amount_grid & mask, [channel]
    count_above_three = frame['txn_count_last_24h'].to_numpy() > 3
    velocity_grid = columns['velocity_score']['intermediate']
    for level, mask in (('le_3', ~count_above_three), ('gt_3', count_above_three)):
        yield 'conditional_grid', f'velocity_intermediate_grid_txn_count_{level}', velocity_grid & mask, [3]
    price_nines = ~amount['off_grid'] & (amount['ticks'] % 100 == 99) & (amount['ticks'] > 0)
    yield 'grid', 'amount_positive_ends_99_cents', price_nines, [.99]

    for variable, items in BOUNDARIES.items():
        values = frame[variable].to_numpy()
        for exact, side, lo, hi in items:
            yield 'boundary', f'{variable}_{side}', values == exact, [exact]
            if lo is not None:
                inner = ((values > lo) & (values <= hi)) if side == 'floor' else ((values >= lo) & (values < hi))
                yield 'boundary', f'{variable}_{side}_near', inner, [lo, hi]


def observed_spellings(project, files_and_frames, columns):
    """Check written digits without treating float parser differences as precision."""
    result = {}
    for name, row_slice in files_and_frames.items():
        counts = {var: Counter() for var in SCALES}
        exceptions = {var: [] for var in SCALES}
        start = row_slice.start or 0
        for chunk in pd.read_csv(project / name, usecols=list(SCALES), dtype=str, chunksize=100_000):
            for var in SCALES:
                lengths = chunk[var].str.rsplit('.', n=1).str[-1].str.len()
                counts[var].update({int(d): int(n) for d, n in lengths.value_counts().items()})
                unusual = columns[var]['off_grid'][start:start + len(chunk)]
                exceptions[var].extend(chunk[var][unusual].tolist())
            start += len(chunk)
        if start != row_slice.stop:
            raise ValueError(f'Unexpected row count scanning written digits in {name}')
        result[name] = {
            var: dict(decimal_place_counts=dict(sorted(counts[var].items())), off_grid_values=exceptions[var])
            for var in SCALES
        }
    return result


def changes(y, offset, delta):
    # Change in binary log-loss, computed directly on logits.
    return np.logaddexp(0., offset + delta) - np.logaddexp(0., offset) - y * delta


def mean_ci(values):
    if len(values) < 2:
        return [None, None]
    center = np.mean(values)
    radius = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
    return [float(center - radius), float(center + radius)]


def describe_group(labels, offset):
    probability = expit(offset)
    residual = labels - probability
    expected = float(probability.sum())
    variance = float((probability * (1. - probability)).sum())
    actual = int(labels.sum())
    rows = len(labels)
    return dict(
        rows=rows, actual_fraud=actual, expected_fraud=expected, bernoulli_variance=variance,
        actual_rate=actual / rows if rows else None,
        expected_rate=expected / rows if rows else None,
        rate_difference_ci95=mean_ci(residual),
        z=(actual - expected) / np.sqrt(variance) if variance > 0 else None,
        log_loss=float(np.mean(np.logaddexp(0., offset) - labels * offset)) if rows else None,
    )


def fit_shift(labels, offset):
    delta = 0.
    for _ in range(6):
        p = expit(offset + delta)
        gradient = (p - labels).sum() + delta * RIDGE
        information = (p * (1. - p)).sum() + RIDGE
        delta = float(np.clip(delta - gradient / information, -CAP, CAP))
    return delta


def screen(mask, development, halves, y, offset, train_n, test_n):
    observed = mask[development]
    fit_y = [y[observed & half] for half in halves]
    fit_offsets = [offset[observed & half] for half in halves]
    summary = [describe_group(a, b) for a, b in zip(fit_y, fit_offsets, strict=True)]
    train_count = int(mask[:train_n].sum())
    test_count = int(mask[train_n:].sum())
    dev_rate, test_rate = float(np.mean(observed)), test_count / test_n
    transport = test_rate / dev_rate if dev_rate > 0 else None
    eligible = bool(
        all(r['rows'] >= 300 and r['expected_fraud'] >= 15 and r['actual_fraud'] >= 10 for r in summary)
        and test_count >= 100 and .75 <= transport <= 1.25
    )
    shifts = [fit_shift(a, b) for a, b in zip(fit_y, fit_offsets, strict=True)]
    effects = [changes(a, b, delta) for a, b, delta in zip(
        fit_y, fit_offsets, shifts[::-1], strict=True,
    )]
    pooled = describe_group(y[observed], offset[observed])
    cross = [
        dict(delta=float(e.mean()), ci95=mean_ci(e)) if len(e) else dict(delta=None, ci95=[None, None])
        for e in effects
    ]
    return dict(
        train_rows=train_count, test_rows=test_count, train_share=train_count / train_n,
        test_share=test_rate, dev_share=dev_rate, test_over_dev_share=transport,
        dev=pooled, halves=summary, half_shifts=shifts, cross_log_loss=cross,
        pooled_two_sided_p=float(2 * norm.sf(abs(pooled['z']))) if pooled['z'] is not None else None,
        eligible=eligible,
    )


def confirms_development(record, eligible_count):
    if not record['eligible']:
        return False
    za, zb = (half['z'] for half in record['halves'])
    return bool(
        za * zb > 0 and min(abs(za), abs(zb)) >= 1.28
        and record['pooled_two_sided_p'] <= .05 / eligible_count
        and all(fold['delta'] < 0. for fold in record['cross_log_loss'])
    )


def confirm_holdout(records, finalists, frame, development, held, labels, columns, extra, ns):
    """Assess the locked development candidates against the saved ensemble."""
    if not finalists:
        return None
    anchor_path = GOAL / 'selected_predictions.npz'
    ids = frame['id'].to_numpy()[held]
    probability = read_scores(anchor_path, ids, 'holdout_ids', ('holdout',))['holdout']
    anchor = logit(probability)
    y = labels[held]
    anchor_auc = roc_auc_score(y, anchor)
    expected_row = read_json(GOAL / 'parallel_eda_resolution' / 'development_metrics.json')['variants']['grandmaster_eda']
    np.testing.assert_allclose(anchor_auc, expected_row['roc_auc'], atol=1e-12, rtol=0.)
    details = dict(
        anchor_file=anchor_path.name, anchor_sha256=digest(anchor_path), anchor_roc_auc=anchor_auc,
        reused_holdout=True, rows=len(held), fraud=int(y.sum()), groups={},
    )
    for _, name, mask, _ in rules(frame, development, columns, *extra):
        if name not in finalists:
            continue
        dev_mask, held_mask = mask[development], mask[held]
        shift = records[name]['locked_dev_shift']
        updated = np.where(held_mask, anchor + shift, anchor)
        paired = changes(y, anchor, held_mask * shift)
        result = assess(ns, y, updated, anchor)
        details['groups'][name] = dict(
            anchor_group=describe_group(y[held_mask], anchor[held_mask]),
            delta_log_loss=float(paired.mean()), delta_log_loss_ci95=mean_ci(paired),
            delta_log_loss_group=float(paired[held_mask].mean()) if held_mask.any() else None,
            delta_log_loss_group_ci95=mean_ci(paired[held_mask]),
            dev_shift=shift, exact_anchor_comparison=result,
        )
        assert dev_mask.sum() == records[name]['dev']['rows']
    return details


def main():
    train, test, _, hashes = official_data()
    labels, dev, held, folds, raw, raw_held, dense, dense_held, metadata = development_caches(train, hashes)
    ns = archive_functions()
    with threadpool_limits(limits=4):
        offset, query_offset, _ = meta_predictions(
            ns, labels[dev], design(ns, raw, dense), design(ns, raw_held, dense_held),
            folds, metadata['baseline_meta_l2_by_fold'],
        )
    ids = train['id'].to_numpy()[held]
    replay = read_scores(GOAL / 'residual_development.npz', ids, 'ids', ('grandmaster_hierarchical',))
    np.testing.assert_allclose(query_offset, replay['grandmaster_hierarchical'], atol=2e-8, rtol=0.)
    first = np.zeros(len(dev), dtype=bool)
    first[np.concatenate([valid for _, valid in folds[:5]])] = True
    halves = [first, ~first]

    needed = ['id', *SCALES, 'customer_age', 'channel', 'txn_count_last_24h']
    frame = pd.concat([train[needed], test[needed]], ignore_index=True)
    columns, *extra = decimal_data(frame, dev)
    full = len(frame)
    lexical = observed_spellings(HERE.parents[2] / "data", {
        'train.csv': slice(0, len(train)), 'test.csv': slice(len(train), full),
    }, columns)
    expected_decimal_fields = len(SCALES)
    assert all(len(fields) == expected_decimal_fields for fields in lexical.values())
    for file, field_info in lexical.items():
        length = len(train) if file == 'train.csv' else len(test)
        assert all(sum(data['decimal_place_counts'].values()) == length for data in field_info.values())

    results = {}
    for family, name, mask, rule in rules(frame, dev, columns, *extra):
        assert name not in results and mask.shape == (full,)
        results[name] = dict(
            rule=rule, family=family,
            **screen(mask, dev, halves, labels[dev], offset, len(train), len(test)),
        )
    eligible = sum(record['eligible'] for record in results.values())
    finalists = [name for name, record in results.items() if confirms_development(record, eligible)]
    if finalists:
        for _, name, mask, _ in rules(frame, dev, columns, *extra):
            if name in finalists:
                chosen = mask[dev]
                results[name]['locked_dev_shift'] = fit_shift(labels[dev][chosen], offset[chosen])
    holdout = confirm_holdout(results, finalists, frame, dev, held, labels, columns, extra, ns)
    report = dict(
        input_sha256=hashes, train_rows=len(train), test_rows=len(test), development_rows=len(dev),
        development_fraud=int(labels[dev].sum()), development_roc_auc=roc_auc_score(labels[dev], offset),
        development_base='Grandmaster hierarchical meta-OOF, frozen development banks',
        half_rows=[int(h.sum()) for h in halves],
        half_fraud=[int(labels[dev][h].sum()) for h in halves],
        eligible_group_count=eligible, two_sided_p_cutoff=.05 / eligible if eligible else None,
        group_count=len(results), finalists=finalists, groups=results, written_digits=lexical, holdout=holdout,
    )
    write_json(HERE / 'measurement_metrics.json', report)
    print(f'Official train rows: {len(train):,}; test rows: {len(test):,}; development rows: {len(dev):,}')
    print(f'Development Grandmaster hierarchical ROC AUC: {report["development_roc_auc"]:.12f}')
    print(f'{eligible} / {len(results)} eligible groups; finalists: {finalists}')


if __name__ == '__main__':
    main()
