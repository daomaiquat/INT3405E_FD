"""Test the information lost within the original EDA quantile cells."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT / 'grandmaster_eda_risk'))
from risk_eda import CONTROL, XGB_WEIGHT, additive_fit, encoded_views, select_pairs
from eda_from_scratch import group_totals
from grandmaster_inputs import (ABLATION, PROJECT, archive_functions, design, development_caches,
                                digest, official_data, read_json, write_json)
from residual_models import QUANTILES, auc_metrics, fit_codes, nested_residual_train
from robust_density import compare, isolated_meta
from run_followup import build_training_view, fit_xgb

METHODS = ('resolution_tree', 'resolution_half', 'resolution_linear')
BOUNDARIES = np.array([0., *QUANTILES, 1.])
MIDPOINTS = .5 * (BOUNDARIES[:-1] + BOUNDARIES[1:])


def ordinal(series):
    dtype = series.dtype
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype) and series.nunique() > 25


def fine_codes(reference, scoring):
    probabilities = np.sort(np.r_[QUANTILES, MIDPOINTS])
    cuts = np.quantile(reference, probabilities)
    coarse = np.unique(cuts[1::2])
    fine = np.unique(cuts)
    child_parent = np.searchsorted(coarse, np.r_[fine, np.inf])
    return np.searchsorted(fine, reference), np.searchsorted(fine, scoring), child_parent


def fine_detail(g, h, n, parent):
    """Fit the contrast between sibling bins after removing their common slope."""
    result = np.zeros_like(g)
    for family in np.unique(parent):
        siblings = np.flatnonzero(parent == family)
        if len(siblings) < 2:
            continue
        low, high = siblings
        total = h[low] + h[high]
        share = np.divide(h[high], total, out=np.zeros_like(total), where=total > 0)
        curvature = h[low] * share
        contrast = (1. - share) * g[high] - share * g[low]
        contrast = np.where(curvature > 0, contrast, 0.)
        common = contrast.sum() / max(curvature.sum(), 1e-12)
        support = curvature / (curvature + 8.)
        counts = np.minimum(n[low], n[high])
        support *= counts / (counts + 80.)
        effect = np.clip(support * (contrast - curvature * common) / (curvature + 25.), -.25, .25)
        result[low], result[high] = -share * effect, (1. - share) * effect
    return result


def eligible_pairs(frame, pairs):
    output = []
    for left, right in pairs:
        a, b = ordinal(frame[left]), ordinal(frame[right])
        if a != b:
            output.append((left, right) if a else (right, left))
    return output


def encode(reference, labels, offsets, query, _query_offsets, pairs):
    probability = expit(offsets)
    continuous = {x for x, _ in pairs}
    categorical = {x for _, x in pairs}
    children = {name: fine_codes(reference[name], query[name]) for name in continuous}
    categories = {name: fit_codes(reference[name], query[name]) for name in categorical}
    columns = []
    for left, right in pairs:
        a, aq, parent = children[left]
        b, bq, count = categories[right]
        shape = (len(parent), count)
        keys = a * count + b
        g, h, n = (part.reshape(shape) for part in group_totals(keys, labels, probability, np.prod(shape)))
        columns.append(fine_detail(g, h, n, parent)[aq, bq])
    return np.column_stack(columns).astype(np.float32)


def branches(train, query, views, y, p, q, device):
    a, b = views
    original = fit_xgb(train, y, query, p, q, device)
    pooled_train, pooled_query = a.sum(axis=1, keepdims=True), b.sum(axis=1, keepdims=True)
    tree = fit_xgb(np.column_stack((train, pooled_train)), y, np.column_stack((query, pooled_query)), p, q, device)
    correction, weights = additive_fit(a, b, y, p, bounds=[(0., 1.)] * a.shape[1])
    return {CONTROL: original, METHODS[0]: tree, METHODS[1]: .5 * tree + .5 * original,
            METHODS[2]: original + .5 * correction}, weights


def write_submissions(sample, ids, probabilities, hashes):
    original = PARENT / 'submission_grandmaster_eda.csv'
    metadata = read_json(PARENT / 'evaluation.json')
    assert metadata['submission_sha256'] == digest(original)
    control = pd.read_csv(original)
    np.testing.assert_array_equal(ids, control.id)
    np.testing.assert_allclose(probabilities[CONTROL], control.is_fraud, atol=1e-14, rtol=0)
    for method in METHODS:
        path = HERE / f'submission_grandmaster_eda_{method}.csv'
        temporary = path.with_suffix('.tmp')
        sample.assign(is_fraud=probabilities[method]).to_csv(temporary, index=False)
        table = pd.read_csv(temporary)
        assert table.id.equals(sample.id) and list(table.columns) == list(sample.columns)
        assert table.id.is_unique and table.is_fraud.between(0., 1.).all()
        np.testing.assert_allclose(table.is_fraud, probabilities[method], atol=1e-14, rtol=0)
        temporary.replace(path)
        write_json(HERE / f'{method}_manifest.json', {'file': path.name, 'sha256': digest(path), 'rows': len(ids),
                   'input_sha256': hashes, 'parent_sha256': metadata['submission_sha256']})
        print(f'{path.name}: verified {len(ids):,} rows', flush=True)


def forward(mode, device):
    context, design_matrices, (fitting, scoring), started = build_training_view(mode)
    ns, _, _, sample, hashes, frame, y, fit, query, p, q, archived, baseline = context
    pairs = eligible_pairs(fitting, select_pairs(hashes))
    views = encoded_views(fitting, scoring, y[fit], p, q, pairs, encode)
    predictions, weights = branches(*design_matrices, views, y[fit], p, q, device)
    replay_error = float(np.max(np.abs(predictions[CONTROL] - archived)))
    np.testing.assert_allclose(predictions[CONTROL], archived, atol=4e-6, rtol=0)
    ids = frame.id.to_numpy()[query]
    logits = {key: baseline + XGB_WEIGHT * (z - predictions[CONTROL]) for key, z in predictions.items()}
    report = {'input_sha256': hashes, 'rows': len(ids), 'fit_rows': len(fit), 'pairs': pairs,
              'control_replay_max_abs_logit': replay_error, 'linear_coefficients': weights.tolist(),
              'rms_oof_features': np.sqrt(np.mean(views[0]**2, axis=0)).tolist()}
    if mode == 'development':
        report['variants'] = compare(ns, y[query], logits, CONTROL)
        for method, result in report['variants'].items():
            print(method, result, flush=True)
    else:
        write_submissions(sample, ids, {k: expit(v) for k, v in logits.items()}, hashes)
    np.savez_compressed(HERE / f'{mode}_predictions.npz', ids=ids, **logits)
    report['seconds'] = perf_counter() - started
    write_json(HERE / f'{mode}_metrics.json', report)


def reverse(device):
    train, test, _, hashes = official_data()
    y, dev, ref, folds, raw, _, dense, _, _ = development_caches(train, hashes)
    assert not np.intersect1d(dev, ref).size
    pairs = select_pairs(hashes)
    eligible = eligible_pairs(train.iloc[ref], pairs)
    ns = archive_functions()
    sys.path.insert(0, str(PROJECT.parent / 's6e11_gpu' / 'plan_runs'))
    from features import FeatureBuilder

    x, z = FeatureBuilder(train, test).make(ref, (dev,), groups=('raw', 'te13', 'cents', 'digits'))
    levels = {name: np.full(len(dev), np.nan) for name in (CONTROL, *METHODS)}
    fold_ids = np.zeros(len(dev), dtype=np.int8)
    fold_records = []
    labels = y[ref]
    for i, (_, held) in enumerate(folds, start=1):
        with np.load(ABLATION / 'validation' / 'artifacts' / f'fold_{i:02}.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['va_idx'], held)
            frozen = design(ns, saved['baseline_hold'], saved['candidate_hold'][:, 1])
        p, q = isolated_meta(ns, frozen, design(ns, raw[held], dense[held]), labels)
        base_a, base_b = nested_residual_train(train, ref, dev[held], labels, p, pairs)
        views = encoded_views(train.iloc[ref], train.iloc[dev[held]], labels, p, q, eligible, encode)
        scores, coefficient = branches(np.column_stack((x, base_a)), np.column_stack((z[held], base_b)),
                                       views, labels, p, q, device)
        for method, values in scores.items():
            levels[method][held] = values
        fold_ids[held] = i
        fold_y = y[dev[held]]
        result = {k: auc_metrics(fold_y, v) for k, v in scores.items()}
        fold_records.append({'fold': i, 'rows': len(held), 'pairs': int(fold_y.sum()) * int((fold_y == 0).sum()),
                             'variants': result, 'linear_coefficients': coefficient.tolist()})
        print(f'Fold {i} delta AUC: ' + ', '.join(
            f'{method} {result[method]["roc_auc"] - result[CONTROL]["roc_auc"]:+.8f}' for method in METHODS), flush=True)
    results = compare(ns, y[dev], levels, CONTROL)
    for method, row in results.items():
        auc = np.array([record['variants'][method]['roc_auc'] - record['variants'][CONTROL]['roc_auc']
                        for record in fold_records])
        ll = np.array([record['variants'][method]['log_loss'] - record['variants'][CONTROL]['log_loss']
                       for record in fold_records])
        row['fold_weighted_delta_auc'] = float(np.average(auc, weights=[f['pairs'] for f in fold_records]))
        row['fold_auc_wins'] = int(np.sum(auc > 0))
        row['fold_logloss_wins'] = int(np.sum(ll < 0))
        print(method, row, flush=True)
    np.savez_compressed(HERE / 'reverse_predictions.npz', ids=train.id.to_numpy()[dev], fold=fold_ids, **levels)
    write_json(HERE / 'reverse_metrics.json', {'input_sha256': hashes, 'rows': len(dev), 'fit_rows': len(ref),
                                               'pairs': eligible, 'variants': results, 'folds': fold_records})


def verify():
    train, test, sample, hashes = official_data()
    y, dev, ref, folds, *_ = development_caches(train, hashes)
    source = PARENT / 'selected_predictions.npz'
    with np.load(source, allow_pickle=False) as saved:
        with np.load(HERE / 'development_predictions.npz', allow_pickle=False) as actual:
            np.testing.assert_array_equal(actual['ids'], saved['holdout_ids'])
            np.testing.assert_array_equal(expit(actual[CONTROL]), saved['holdout'])
        with np.load(HERE / 'full_predictions.npz', allow_pickle=False) as actual:
            np.testing.assert_array_equal(actual['ids'], test.id)
            np.testing.assert_array_equal(actual['ids'], saved['test_ids'])
            np.testing.assert_array_equal(expit(actual[CONTROL]), saved['test'])
            for method in METHODS:
                manifest = read_json(HERE / f'{method}_manifest.json')
                file = HERE / manifest['file']
                csv = pd.read_csv(file)
                assert manifest['sha256'] == digest(file) and manifest['input_sha256'] == hashes
                assert csv.id.is_unique and csv.id.equals(sample.id) and list(csv.columns) == list(sample.columns)
                assert csv.is_fraud.between(0., 1.).all()
                np.testing.assert_allclose(csv.is_fraud, expit(actual[method]), atol=1e-14, rtol=0)
                print(f'{file.name}: verified {len(csv):,} probabilities and SHA-256')
    for mode, rows in (('development', ref), ('reverse', dev)):
        report = read_json(HERE / f'{mode}_metrics.json')
        assert report['rows'] == len(rows) and report['input_sha256'] == hashes
        with np.load(HERE / f'{mode}_predictions.npz', allow_pickle=False) as actual:
            np.testing.assert_array_equal(actual['ids'], train.id.to_numpy()[rows])
            for name, record in report['variants'].items():
                z, labels = actual[name], y[rows]
                p = expit(z)
                scores = (roc_auc_score(labels, z), average_precision_score(labels, z), log_loss(labels, p),
                          brier_score_loss(labels, p))
                expected = [record[k] for k in ('roc_auc', 'average_precision', 'log_loss', 'brier')]
                np.testing.assert_allclose(scores, expected, atol=1e-12, rtol=0)
                if mode == 'reverse':
                    for i, (_, held) in enumerate(folds, start=1):
                        np.testing.assert_array_equal(actual['fold'][held], i)
                        expected_auc = report['folds'][i - 1]['variants'][name]['roc_auc']
                        np.testing.assert_allclose(roc_auc_score(labels[held], z[held]), expected_auc, atol=1e-12, rtol=0)
        print(f'{mode}: recalculated validation metrics on {len(rows):,} rows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'reverse', 'export', 'verify'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.mode == 'verify':
        verify()
    elif args.mode == 'reverse':
        reverse(args.device)
    else:
        forward('full' if args.mode == 'export' else 'development', args.device)
