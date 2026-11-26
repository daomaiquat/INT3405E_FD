"""Test risk-dependent residual interactions on the fixed Grandmaster EDA recipe."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path[:0] = [str(PARENT / 'grandmaster_eda_refinement')]
from refine_eda import (CONTROL, SPLIT_SEED, XGB_WEIGHT, compare, isolated_meta, select_pairs)
from eda_from_scratch import centered_pair
from evaluate_and_export import auc_by_base_score_band
from grandmaster_inputs import (ABLATION, PROJECT, archive_functions, design, development_caches,
                                digest, official_data, read_json, write_json)
from residual_models import auc_metrics, fit_codes, nested_residual_train, residual_features
from run_followup import build_training_view, fit_xgb

METHODS = ('risk_conditioned_tree', 'risk_conditioned_half', 'risk_conditioned_linear')
EXTRA_METHODS = ('extra_pairs_tree', 'extra_pairs_half', 'extra_pairs_linear')
RISK_PIVOT = logit(.03)
RISK_WIDTH = 1.5


def risk_coordinate(log_odds):
    return np.tanh((log_odds - RISK_PIVOT) / RISK_WIDTH)


def risk_slope_table(keys, shape, remainder, information, coordinate):
    size = shape[0] * shape[1]

    def totals(weights):
        return np.bincount(keys, weights=weights, minlength=size).reshape(shape)

    curvature = totals(information)
    moment = totals(information * coordinate)
    mean = np.divide(moment, curvature, out=np.zeros_like(moment), where=curvature > 0)
    curvature_slope = np.maximum(totals(information * coordinate**2) - moment * mean, 0.)
    gradient_slope = totals(remainder * coordinate) - mean * totals(remainder)
    contrast = centered_pair(gradient_slope, curvature_slope)
    reliability = curvature_slope / (curvature_slope + 6.)
    slope = np.clip(reliability * contrast / (curvature_slope + 25.), -.25, .25)
    return mean, slope


def risk_features(reference, labels, offsets, query, query_offsets, pairs):
    fields = {name for pair in pairs for name in pair}
    codes = {name: fit_codes(reference[name], query[name]) for name in fields}
    probability = expit(offsets)
    remainder = labels - probability
    information = probability * (1 - probability)
    coordinate, score_coordinate = risk_coordinate(offsets), risk_coordinate(query_offsets)
    columns = []
    for left, right in pairs:
        a, aq, ac = codes[left]
        b, bq, bc = codes[right]
        mean, slope = risk_slope_table(a * bc + b, (ac, bc), remainder, information, coordinate)
        columns.append(slope[aq, bq] * (score_coordinate - mean[aq, bq]))
    return np.column_stack(columns).astype(np.float32)


def extra_features(reference, labels, offsets, query, query_offsets, pairs):
    estimates = residual_features(reference, labels, offsets, query, pairs)[:, :-1]
    return estimates.sum(axis=1, keepdims=True)


def get_experiment(family, hashes):
    pairs = select_pairs(hashes)
    if family == 'risk':
        return METHODS, risk_features, pairs
    audit = read_json(PARENT / 'residual_eda.json')
    if audit['input_sha256'] != hashes:
        raise ValueError('Candidate EDA pairs refer to a different dataset')
    extra = [(row['left'], row['right']) for row in audit['pair_results']
             if (row['cross_delta_logloss_x1m'] or 0.) >= 1.
             and (row['left'], row['right']) not in pairs]
    if not extra:
        raise ValueError('No unused EDA pairs passed the complementary signal screen')
    return EXTRA_METHODS, extra_features, extra


def encoded_views(frame, query, y, offsets, q, pairs, encoder):
    grid = StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED)
    test = encoder(frame, y, offsets, query, q, pairs)
    train = np.empty((len(y), test.shape[1]), dtype=np.float32)
    for reference, held in grid.split(y, y):
        train[held] = encoder(frame.iloc[reference], y[reference], offsets[reference],
                              frame.iloc[held], offsets[held], pairs)
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError('Interaction features must be finite')
    return train, test


def additive_fit(train, query, y, p, bounds=None):
    # Fit a small ridge logistic correction over strictly cross-fitted columns.
    features = train.astype(np.float64)
    ridge = 12.

    def objective(weight):
        z = p + features @ weight
        loss = np.logaddexp(0, z).sum() - y @ z + .5 * ridge * (weight @ weight)
        gradient = features.T @ (expit(z) - y) + ridge * weight
        return loss, gradient

    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        result = minimize(objective, np.zeros(features.shape[1]), jac=True, method='L-BFGS-B', bounds=bounds,
                          options={'ftol': 1e-12, 'gtol': 1e-6})
        if not result.success:
            raise RuntimeError(result.message)
        correction = query @ result.x
    return correction, result.x


def fit_branches(train, query, extra, y, p, q, device, methods):
    control = fit_xgb(train, y, query, p, q, device)
    tree = fit_xgb(np.column_stack((train, extra[0])), y, np.column_stack((query, extra[1])), p, q, device)
    bounds = [(0., 1.)] if methods == EXTRA_METHODS else None
    correction, coefficients = additive_fit(*extra, y, p, bounds=bounds)
    return {CONTROL: control, methods[0]: tree, methods[1]: control + .5 * (tree - control),
            methods[2]: control + .5 * correction}, coefficients


def export(sample, ids, probabilities, hashes, methods):
    parent = PARENT / 'submission_grandmaster_eda.csv'
    parent_sha = read_json(PARENT / 'evaluation.json')['submission_sha256']
    if digest(parent) != parent_sha:
        raise ValueError('Grandmaster EDA original submission checksum differs from manifest')
    reference = pd.read_csv(parent)
    np.testing.assert_array_equal(ids, reference.id)
    np.testing.assert_allclose(probabilities[CONTROL], reference.is_fraud, rtol=0, atol=1e-14)
    for name in methods:
        destination = HERE / f'submission_grandmaster_eda_{name}.csv'
        temp = destination.with_suffix('.tmp')
        sample.assign(is_fraud=probabilities[name]).to_csv(temp, index=False)
        output = pd.read_csv(temp)
        if not output.id.equals(sample.id) or list(output.columns) != list(sample.columns) or not output.is_fraud.between(0, 1).all():
            raise ValueError(f'Invalid ID, probability or schema in {destination.name}')
        np.testing.assert_allclose(output.is_fraud, probabilities[name], rtol=0, atol=1e-14)
        temp.replace(destination)
        write_json(HERE / f'{name}_manifest.json', {'filename': destination.name, 'rows': len(output),
                   'sha256': digest(destination), 'parent_sha256': parent_sha, 'input_sha256': hashes})
        print(f'Export checked: {destination.name} ({len(output):,} rows)', flush=True)


def file_prefix(family, mode):
    return mode if family == 'risk' else f'extra_{mode}'


def forward(mode, device, family):
    context, views, (train, query), started = build_training_view(mode)
    ns, _, _, sample, hashes, frame, y, fit, score_rows, p, q, archived, incumbent = context
    methods, encoder, pairs = get_experiment(family, hashes)
    encoded = encoded_views(train, query, y[fit], p, q, pairs, encoder)
    models, coefficients = fit_branches(*views, encoded, y[fit], p, q, device, methods)
    replay = float(np.max(np.abs(models[CONTROL] - archived)))
    np.testing.assert_allclose(models[CONTROL], archived, rtol=0, atol=4e-6)
    scores = {key: incumbent + XGB_WEIGHT * (values - models[CONTROL]) for key, values in models.items()}
    ids = frame.id.to_numpy()[score_rows]
    names = pairs if family == 'risk' else ['sum_of_additional_pairs']
    diagnostics = [{'pair': name, 'coefficient': float(w), 'nonzero_train': float(np.mean(a != 0)),
                    'nonzero_query': float(np.mean(b != 0)), 'rms_train': float(np.sqrt(np.mean(a**2)))}
                   for name, w, a, b in zip(names, coefficients, encoded[0].T, encoded[1].T, strict=True)]
    metadata = {'input_sha256': hashes, 'fit_rows': len(fit), 'rows': len(ids), 'pairs': pairs,
                'maximum_control_logit_replay_error': replay, 'slope_diagnostics': diagnostics}
    if mode == 'development':
        metrics = compare(ns, y[score_rows], scores, CONTROL)
        for key, value in metrics.items():
            value['score_band_contribution'] = auc_by_base_score_band(y[score_rows], scores[CONTROL], scores[key])
            print(key, {k: v for k, v in value.items() if k != 'score_band_contribution'}, flush=True)
        metadata['variants'] = metrics
    else:
        export(sample, ids, {key: expit(values) for key, values in scores.items()}, hashes, methods)
    metadata['seconds'] = perf_counter() - started
    prefix = file_prefix(family, mode)
    np.savez_compressed(HERE / f'{prefix}_predictions.npz', ids=ids, **scores)
    write_json(HERE / f'{prefix}_metrics.json', metadata)


def reverse(device, family):
    train, test, _, hashes = official_data()
    y, dev, reference, folds, raw, _, dense, _, _ = development_caches(train, hashes)
    assert np.intersect1d(dev, reference).size == 0
    pairs = select_pairs(hashes)
    methods, encoder, enhancement_pairs = get_experiment(family, hashes)
    labels = y[reference]
    ns = archive_functions()
    sys.path.insert(0, str(PROJECT.parent / 's6e11_gpu' / 'plan_runs'))
    from features import FeatureBuilder

    raw_train, raw_query = FeatureBuilder(train, test).make(reference, (dev,), groups=('raw', 'te13', 'cents', 'digits'))
    original = {name: np.full(len(dev), np.nan) for name in (CONTROL, *methods)}
    fold_ids = np.zeros(len(dev), dtype=np.int8)
    fold_records = []
    for i, (_, held) in enumerate(folds, start=1):
        path = ABLATION / 'validation' / 'artifacts' / f'fold_{i:02}.npz'
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['va_idx'], held)
            source = design(ns, saved['baseline_hold'], saved['candidate_hold'][:, 1])
        p, q = isolated_meta(ns, source, design(ns, raw[held], dense[held]), labels)
        train_eda, query_eda = nested_residual_train(train, reference, dev[held], labels, p, pairs)
        encoded = encoded_views(train.iloc[reference], train.iloc[dev[held]], labels, p, q, enhancement_pairs, encoder)
        scores, _ = fit_branches(np.column_stack((raw_train, train_eda)),
                                 np.column_stack((raw_query[held], query_eda)), encoded, labels, p, q, device, methods)
        y_fold = y[dev[held]]
        fold_ids[held] = i
        measures = {name: auc_metrics(y_fold, score) for name, score in scores.items()}
        for name in scores:
            original[name][held] = scores[name]
        fold_records.append({'fold': i, 'rows': len(held), 'pairs': int(y_fold.sum()) * int((y_fold == 0).sum()),
                             'variants': measures})
        print(f'Fold {i}: ' + ', '.join(f'{name} {measures[name]["roc_auc"] - measures[CONTROL]["roc_auc"]:+.8f}'
                                        for name in methods), flush=True)

    results = compare(ns, y[dev], original, CONTROL)
    for name, item in results.items():
        delta = [f['variants'][name]['roc_auc'] - f['variants'][CONTROL]['roc_auc'] for f in fold_records]
        logloss = [f['variants'][name]['log_loss'] - f['variants'][CONTROL]['log_loss'] for f in fold_records]
        weights = [f['pairs'] for f in fold_records]
        item['fold_weighted_delta_auc'] = float(np.average(delta, weights=weights))
        item['fold_auc_wins'] = sum(d > 0 for d in delta)
        item['fold_mean_delta_logloss'] = float(np.average(logloss, weights=[f['rows'] for f in fold_records]))
        item['fold_logloss_wins'] = sum(d < 0 for d in logloss)
        print(name, item, flush=True)
    prefix = file_prefix(family, 'reverse')
    np.savez_compressed(HERE / f'{prefix}_predictions.npz', ids=train.id.to_numpy()[dev], fold=fold_ids, **original)
    write_json(HERE / f'{prefix}_metrics.json', {'input_sha256': hashes, 'reference_rows': len(reference),
                                                'rows': len(dev), 'variants': results, 'folds': fold_records})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'reverse', 'export'))
    parser.add_argument('--family', default='risk', choices=('risk', 'extra'))
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    args = parser.parse_args()
    if args.mode == 'reverse':
        reverse(args.device, args.family)
    else:
        forward('full' if args.mode == 'export' else 'development', args.device, args.family)
