"""Audit and test overlooked data relationships with a frozen tree recipe."""

import argparse
from itertools import combinations
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
FOLLOWUP = PARENT / 'followup'
sys.path[:0] = [str(PARENT), str(FOLLOWUP), str(PARENT / 'robust_validation')]
from context_expansion import density_and_context
from eda_from_scratch import group_totals
from evaluate_and_export import read_scores
from grandmaster_inputs import (ABLATION, PROJECT, archive_functions, design, development_caches,
                                digest, official_data, read_json, write_json)
from residual_models import auc_metrics
from robust_density import compare, isolated_meta
from run_followup import build_training_view, crossfit_extra, fit_xgb
from overlooked_features import (DIGITS, PEERS, digits_for, fold_fingerprints, peer_features,
                                 physical, physical_features)


SLOT_WEIGHT = .425
FAMILIES = ('logic_time', 'peer_profiles', 'numeric_fingerprints', 'combined')


def audit(validation_frame, labels, base, hashes):
    halves = next(StratifiedShuffleSplit(1, test_size=.5, random_state=26011).split(labels, labels))
    raw = physical(validation_frame)
    digits = digits_for(validation_frame)
    records = []
    examples = {}
    prediction = expit(base)
    for key in ('spend_exceeds_balance', 'short_interval_empty_count', 'long_interval_positive_count',
                'card_predates_adulthood', 'ip_flag_disagreement'):
        mask = raw[key] != 0
        examples[key] = {'rows': int(mask.sum()), 'fraud': int(labels[mask].sum()),
                         'grandmaster_expected_fraud': float(prediction[mask].sum())}

    def inspect(name, values, discrete_width=None):
        gains = []
        for ref, query in (halves, halves[::-1]):
            if discrete_width is None:
                edges = np.unique(np.quantile(values[ref], np.arange(1, 12) / 12))
                keys, width = np.searchsorted(edges, values), len(edges) + 1
            else:
                keys, width = values.astype(np.int64), discrete_width
            g, h, counts = group_totals(keys[ref], labels[ref], prediction[ref], width)
            effect = np.where(counts >= 100, np.clip(g / (h + 35), -.25, .25), 0)
            base_loss = log_loss(labels[query], prediction[query])
            trial_loss = log_loss(labels[query], expit(base[query] + effect[keys[query]]))
            gains.append(float(1e6 * (base_loss - trial_loss)))
        records.append({'feature': name, 'halves_logloss_gain_x1m': gains, 'both_positive': min(gains) > 0})

    for key, values in raw.items():
        inspect(key, values, 2 if key in examples else None)
    for a, b in combinations(digits, 2):
        inspect(f'terminal_digits:{a}×{b}', 10 * digits[a] + digits[b], 100)
    records.sort(key=lambda item: np.mean(item['halves_logloss_gain_x1m']), reverse=True)
    result = {'input_sha256': hashes, 'rows': len(labels), 'fraud': int(labels.sum()), 'discovery_seed': 26011,
              'examples': examples, 'screen': records,
              'context_profile_relationships': [[value, *conditions] for value, conditions in PEERS]}
    write_json(HERE / 'data_audit.json', result)
    for item in records[:8]:
        print('EDA', item, flush=True)


def extra_view(which, fit_frame, query_frame, labels, offsets):
    matrices = {}
    if which in ('logic_time', 'combined'):
        matrices['logic_time'] = (physical_features(fit_frame), physical_features(query_frame))
    if which in ('peer_profiles', 'combined'):
        matrices['peer_profiles'] = crossfit_extra(peer_features, fit_frame, labels, query_frame)
    if which in ('numeric_fingerprints', 'combined'):
        matrices['numeric_fingerprints'] = fold_fingerprints(fit_frame, labels, offsets, query_frame)
    return matrices


def append(base, other):
    return tuple(np.column_stack((a, b)) for a, b in zip(base, other, strict=True))


def train_one(base, features, name, labels, offsets, query_offsets, device):
    x_train, x_query = base
    if name == 'combined':
        full = tuple(np.column_stack([item[i] for item in features.values()]) for i in range(2))
        x_train, x_query = append(base, full)
    elif name != 'control':
        x_train, x_query = append(base, features[name])
    return fit_xgb(x_train, labels, x_query, offsets, query_offsets, device)


def forward(mode, device, candidate=None):
    context, base, (fitting, scoring), started = build_training_view(mode)
    ns, _, test, sample, hashes, frame, y, fit, query, offset, query_offset, original, incumbent = context
    ids = frame.id.to_numpy()[query]
    labels = y[fit]
    if mode == 'development':
        audit(fitting, labels, offset, hashes)
    base = append(base, crossfit_extra(density_and_context, fitting, labels, scoring))
    features = extra_view(candidate if candidate else 'combined', fitting, scoring, labels, offset)
    raw_control = train_one(base, features, 'control', labels, offset, query_offset, device)
    cases = (candidate,) if candidate else FAMILIES
    results = {}
    if mode == 'development':
        prior = read_scores(FOLLOWUP / 'context_holdout.npz', ids, 'ids', ('expanded', 'expanded_half_density'))
        np.testing.assert_allclose(incumbent + .85 * (raw_control - original), prior['expanded'], rtol=0, atol=2e-8)
        current = prior['expanded_half_density']
    else:
        p = read_scores(FOLLOWUP / 'context_test.npz', ids, 'ids', ('probabilities',))['probabilities']
        current = logit(np.clip(p, 1e-12, 1-1e-12))
    for name in cases:
        values = train_one(base, features, name, labels, offset, query_offset, device)
        results[name] = current + SLOT_WEIGHT * (values - raw_control)
        print(f'{mode}: {name} complete in {perf_counter()-started:.1f}s', flush=True)
    if mode == 'development':
        predictions = {'current': current, **results}
        scores = compare(ns, y[query], predictions, 'current')
        np.savez_compressed(HERE / 'holdout_predictions.npz', ids=ids, **predictions)
        write_json(HERE / 'holdout_metrics.json', {'input_sha256': hashes, 'candidates': scores,
                                                   'rows': len(query), 'seconds': perf_counter()-started})
        for name, row in scores.items():
            print(name, row, flush=True)
    else:
        probabilities = expit(results[candidate])
        path = HERE / f'submission_{candidate}.csv'
        temporary = path.with_suffix('.tmp')
        sample.assign(is_fraud=probabilities).to_csv(temporary, index=False)
        verified = pd.read_csv(temporary)
        if list(verified.columns) != ['id', 'is_fraud'] or not verified.id.equals(test.id):
            raise ValueError('Submission IDs or columns differ from competition input')
        np.testing.assert_allclose(verified.is_fraud.to_numpy(), probabilities, rtol=0, atol=1e-14)
        temporary.replace(path)
        np.savez_compressed(HERE / f'{candidate}_test.npz', ids=ids, probabilities=probabilities)
        write_json(HERE / f'{candidate}_submission.json', {'input_sha256': hashes, 'file': path.name,
                                                            'sha256': digest(path), 'test_rows': len(ids)})
        print(f'Verified {path.name}, {len(ids):,} rows', flush=True)


def reverse(device, candidate):
    started = perf_counter()
    train, test, _, hashes = official_data()
    y, dev, ref, folds, archive, _, dense, _, _ = development_caches(train, hashes)
    ns = archive_functions()
    sys.path.insert(0, str(PROJECT.parent / 's6e11_gpu' / 'plan_runs'))
    from features import FeatureBuilder

    labels = y[ref]
    ref_frame, dev_frame = train.iloc[ref], train.iloc[dev]
    base = FeatureBuilder(train, test).make(ref, (dev,), groups=('raw', 'te13', 'cents', 'digits'))
    base = append(base, crossfit_extra(density_and_context, ref_frame, labels, dev_frame))
    stable = {} if candidate in ('numeric_fingerprints', 'combined') else extra_view(candidate, ref_frame, dev_frame, labels, np.zeros(len(ref)))
    if candidate == 'combined':
        stable.update(extra_view('logic_time', ref_frame, dev_frame, labels, np.zeros(len(ref))))
        stable.update(extra_view('peer_profiles', ref_frame, dev_frame, labels, np.zeros(len(ref))))
    results = {key: np.full(len(dev), np.nan) for key in ('current', candidate)}
    fold_ids = np.zeros(len(dev), dtype=np.int8)
    per_fold = []
    for i, (_, valid) in enumerate(folds, start=1):
        if np.intersect1d(ref, dev[valid]).size:
            raise ValueError('Training labels overlap evaluation rows')
        path = ABLATION / 'validation' / 'artifacts' / f'fold_{i:02}.npz'
        with np.load(path, allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache['va_idx'], valid)
            ref_scores = design(ns, cache['baseline_hold'], cache['candidate_hold'][:, 1])
        held_scores = design(ns, archive[valid], dense[valid])
        offsets, held_offsets = isolated_meta(ns, ref_scores, held_scores, labels)
        matrices = {name: (a, b[valid]) for name, (a, b) in stable.items()}
        if candidate in ('numeric_fingerprints', 'combined'):
            matrices.update(extra_view('numeric_fingerprints', ref_frame, dev_frame.iloc[valid], labels, offsets))
        fold_base = (base[0], base[1][valid])
        control = train_one(fold_base, {}, 'control', labels, offsets, held_offsets, device)
        augmented = train_one(fold_base, matrices, candidate, labels, offsets, held_offsets, device)
        new = .5 * (control + augmented)
        evaluation = y[dev[valid]]
        scores = {key: auc_metrics(evaluation, s) for key, s in [('current', control), (candidate, new)]}
        pairs = int(evaluation.sum()) * int((evaluation == 0).sum())
        per_fold.append({'fold': i, 'rows': len(valid), 'pair_count': pairs, **scores})
        results['current'][valid] = control
        results[candidate][valid] = new
        fold_ids[valid] = i
        diff = scores[candidate]['roc_auc'] - scores['current']['roc_auc']
        print(f'Fold {i:2}/10, delta AUC {diff:+.8f}, {perf_counter()-started:.1f}s', flush=True)
    check = compare(ns, y[dev], results, 'current')
    wins_auc = sum(row[candidate]['roc_auc'] > row['current']['roc_auc'] for row in per_fold)
    wins_loss = sum(row[candidate]['log_loss'] < row['current']['log_loss'] for row in per_fold)
    pair_weighted = float(np.average([row[candidate]['roc_auc'] - row['current']['roc_auc'] for row in per_fold],
                                      weights=[row['pair_count'] for row in per_fold]))
    np.savez_compressed(HERE / 'reversed_predictions.npz', ids=train.id.to_numpy()[dev], fold=fold_ids, **results)
    write_json(HERE / 'reversed_metrics.json', {'input_sha256': hashes, 'candidate': candidate, 'rows': len(dev),
                                                'reference_rows': len(ref), 'metrics': check, 'folds': per_fold,
                                                'pair_weighted_delta_auc': pair_weighted,
                                                'auc_wins': wins_auc, 'logloss_wins': wins_loss})
    print(check, wins_auc, wins_loss, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'reverse', 'export'))
    parser.add_argument('--candidate', choices=FAMILIES)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.mode != 'development' and args.candidate is None:
        parser.error('--candidate is required for reverse or export')
    if args.mode == 'reverse':
        reverse(args.device, args.candidate)
    else:
        forward('full' if args.mode == 'export' else args.mode, args.device, args.candidate)
