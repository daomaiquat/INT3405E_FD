"""Refine only the EDA residual encodings in the original Grandmaster EDA recipe."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path[:0] = [str(PARENT), str(PARENT / 'followup'), str(PARENT / 'robust_validation')]
from eda_from_scratch import centered_pair, group_totals
from grandmaster_inputs import (ABLATION, PROJECT, archive_functions, design, development_caches,
                                digest, official_data, read_json, write_json)
from residual_models import auc_metrics, fit_codes, residual_features
from robust_density import compare, isolated_meta
from run_followup import ROUNDS, XGB_SETTINGS, build_training_view


CONTROL = 'grandmaster_eda'
METHODS = ('matched_encoding_size', 'smoothed_pairs', 'smoothed_pairs_half')
XGB_WEIGHT = .85
SPLIT_SEED = 62026


def borrow_neighbors(values, axis):
    """Pool 10% of each bin with each adjacent ordinal bin and preserve totals."""
    old = np.moveaxis(values, axis, -1)
    pooled = .8 * old
    pooled[..., 1:] += .1 * old[..., :-1]
    pooled[..., :-1] += .1 * old[..., 1:]
    pooled[..., 0] += .1 * old[..., 0]
    pooled[..., -1] += .1 * old[..., -1]
    return np.moveaxis(pooled, -1, axis)


def smoothed_table(g, h, n, axes):
    for a in axes:
        g, h, n = (borrow_neighbors(value, a) for value in (g, h, n))
    reliability = (h / (h + 4.)) * (n / (n + 60.))
    centered = centered_pair(g, h)
    return np.clip(reliability * centered / (h + 25.), -.35, .35)


def select_pairs(hashes):
    manifest = read_json(PARENT / 'residual_development_metrics.json')
    if manifest['input_sha256'] != hashes:
        raise ValueError('Parent EDA features refer to different source data')
    return [tuple(value) for value in manifest['selected_residual_pairs']]


def smooth_features(fit, labels, offsets, query, pairs):
    fields = {name for pair in pairs for name in pair}
    codes = {name: fit_codes(fit[name], query[name]) for name in fields}
    ordered = {name for name in fields if pd.api.types.is_numeric_dtype(fit[name])
               and not pd.api.types.is_bool_dtype(fit[name]) and fit[name].nunique() > 25}
    probability = expit(offsets)
    output = []
    for left, right in pairs:
        l_fit, l_score, l_size = codes[left]
        r_fit, r_score, r_size = codes[right]
        g, h, n = (part.reshape(l_size, r_size) for part in group_totals(
            l_fit * r_size + r_fit, labels, probability, l_size * r_size))
        axes = [axis for axis, name in enumerate((left, right)) if name in ordered]
        output.append(smoothed_table(g, h, n, axes)[l_score, r_score])
    distance_digit = residual_features(fit, labels, offsets, query, ())[:, 0]
    return np.column_stack((*output, distance_digit)).astype(np.float32)


def eda_matrices(fit_frame, query_frame, labels, offsets, pairs, original=None):
    width = len(pairs) + 1
    if original is None:
        orig_train = np.empty((len(fit_frame), width), dtype=np.float32)
        orig_query = residual_features(fit_frame, labels, offsets, query_frame, pairs)
    else:
        orig_train, orig_query = original
    smooth_train = np.empty((len(fit_frame), width), dtype=np.float32)
    smooth_query = smooth_features(fit_frame, labels, offsets, query_frame, pairs)
    scoring_views = []
    splitter = StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED)
    for reference, held in splitter.split(labels, labels):
        a, b = fit_frame.iloc[reference], fit_frame.iloc[held]
        partial_y, p = labels[reference], offsets[reference]
        if original is None:
            orig_train[held] = residual_features(a, partial_y, p, b, pairs)
        smooth_train[held] = smooth_features(a, partial_y, p, b, pairs)
        scoring_views.append(residual_features(a, partial_y, p, query_frame, pairs))
    return (orig_train, orig_query), (smooth_train, smooth_query), scoring_views


def train_and_score(base, original, smoothed, scoring_views, labels, offsets, score_offset, device):
    def fit(features):
        matrix = xgb.QuantileDMatrix(np.column_stack((base[0], features)), label=labels, base_margin=offsets,
                                     max_bin=XGB_SETTINGS['max_bin'], nthread=4)
        return xgb.train({**XGB_SETTINGS, 'device': device}, matrix, num_boost_round=ROUNDS)

    def score(model, features):
        matrix = xgb.DMatrix(np.column_stack((base[1], features)), base_margin=score_offset, nthread=4)
        return model.predict(matrix, output_margin=True, iteration_range=(0, ROUNDS)).astype(np.float64)

    baseline = fit(original[0])
    control = score(baseline, original[1])
    matched = np.mean(np.stack([score(baseline, view) for view in scoring_views]), axis=0)
    curved = score(fit(smoothed[0]), smoothed[1])
    return {CONTROL: control, 'matched_encoding_size': matched, 'smoothed_pairs': curved,
            'smoothed_pairs_half': .5 * control + .5 * curved}


def export_predictions(sample, hashes, ids, scores, methods, replay_error):
    baseline_info = read_json(PARENT / 'evaluation.json')
    parent_csv = PARENT / 'submission_grandmaster_eda.csv'
    if digest(parent_csv) != baseline_info['submission_sha256']:
        raise ValueError('Original Grandmaster EDA CSV no longer matches its manifest')
    archived_csv = pd.read_csv(parent_csv)
    np.testing.assert_array_equal(ids, archived_csv.id.to_numpy())
    np.testing.assert_allclose(expit(scores[CONTROL]), archived_csv.is_fraud.to_numpy(), rtol=0, atol=1e-14)
    for method in methods:
        pred = expit(scores[method])
        if not np.isfinite(pred).all():
            raise ValueError(f'Non-finite scores in {method}')
        path = HERE / f'submission_grandmaster_eda_{method}.csv'
        temporary = path.with_suffix('.tmp')
        sample.assign(is_fraud=pred).to_csv(temporary, index=False)
        new_csv = pd.read_csv(temporary)
        if list(new_csv.columns) != ['id', 'is_fraud'] or not new_csv.id.equals(sample.id):
            raise ValueError(f'Unexpected CSV IDs or columns in {method}')
        np.testing.assert_allclose(new_csv.is_fraud, pred, rtol=0, atol=1e-14)
        temporary.replace(path)
        write_json(HERE / f'{method}_manifest.json', {'input_sha256': hashes, 'method': method, 'file': path.name,
                                                       'rows': len(ids), 'sha256': digest(path),
                                                       'replay_max_margin_error': replay_error,
                                                       'parent_sha256': baseline_info['submission_sha256']})
        print(f'Verified {path.name}, {len(ids):,} rows', flush=True)
    np.savez_compressed(HERE / 'test_predictions.npz', ids=ids, **scores)


def forward(mode, device, methods=()):
    context, (training_view, test_view), (fit_frame, score_frame), started = build_training_view(mode)
    ns, _, _, sample, hashes, frame, y, fit, query, p, q, archived_xgb, incumbent = context
    pairs = select_pairs(hashes)
    n = len(pairs) + 1
    raw = (training_view[:, :-n], test_view[:, :-n])
    original = (training_view[:, -n:], test_view[:, -n:])
    views = eda_matrices(fit_frame, score_frame, y[fit], p, pairs, original=original)
    boosters = train_and_score(raw, *views, y[fit], p, q, device)
    # GPU histogram training and float32 margins can differ by a few ULPs on replay.
    replay_error = float(np.max(np.abs(boosters[CONTROL] - archived_xgb)))
    np.testing.assert_allclose(boosters[CONTROL], archived_xgb, rtol=0, atol=4e-6)
    print(f'XGBoost replay: maximum absolute margin drift {replay_error:.3g}', flush=True)
    ids = frame.id.to_numpy()[query]
    scores = {name: incumbent + XGB_WEIGHT * (prediction - boosters[CONTROL])
              for name, prediction in boosters.items()}
    if mode == 'development':
        report = compare(ns, y[query], scores, CONTROL)
        np.savez_compressed(HERE / 'holdout_predictions.npz', ids=ids, **scores)
        write_json(HERE / 'holdout_metrics.json', {'input_sha256': hashes, 'rows': len(ids), 'variants': report,
                                                   'selected_pairs': pairs, 'replay_max_margin_error': replay_error,
                                                   'seconds': perf_counter() - started})
        for name, result in report.items():
            print(name, result, flush=True)
    else:
        export_predictions(sample, hashes, ids, scores, methods, replay_error)


def reverse(device):
    started = perf_counter()
    train, test, _, hashes = official_data()
    y, dev, ref, folds, raw, _, dense, _, _ = development_caches(train, hashes)
    pairs = select_pairs(hashes)
    labels = y[ref]
    ns = archive_functions()
    sys.path.insert(0, str(PROJECT.parent / 's6e11_gpu' / 'plan_runs'))
    from features import FeatureBuilder

    raw_train, raw_query = FeatureBuilder(train, test).make(ref, (dev,), groups=('raw', 'te13', 'cents', 'digits'))
    ref_frame, dev_frame = train.iloc[ref], train.iloc[dev]
    oof = {name: np.full(len(dev), np.nan) for name in (CONTROL, *METHODS)}
    fold_ids = np.zeros(len(dev), dtype=np.int8)
    summaries = []
    for i, (_, held) in enumerate(folds, start=1):
        if np.intersect1d(ref, dev[held]).size:
            raise ValueError('Matched verification fold overlaps fitting rows')
        path = ABLATION / 'validation' / 'artifacts' / f'fold_{i:02}.npz'
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['va_idx'], held)
            start = design(ns, saved['baseline_hold'], saved['candidate_hold'][:, 1])
        scoring = design(ns, raw[held], dense[held])
        p, q = isolated_meta(ns, start, scoring, labels)
        views = eda_matrices(ref_frame, dev_frame.iloc[held], labels, p, pairs)
        scores = train_and_score((raw_train, raw_query[held]), *views, labels, p, q, device)
        for name, values in scores.items():
            oof[name][held] = values
        fold_ids[held] = i
        check = y[dev[held]]
        per_fold = {name: auc_metrics(check, score) for name, score in scores.items()}
        summaries.append({'fold': i, 'rows': len(held), 'pairs': int(check.sum())*int((check == 0).sum()), 'metrics': per_fold})
        print(f'Fold {i}/10, AUC changes: {[round(per_fold[k]["roc_auc"]-per_fold[CONTROL]["roc_auc"],8) for k in METHODS]}', flush=True)
    metrics = compare(ns, y[dev], oof, CONTROL)
    weights = [f['pairs'] for f in summaries]
    for name, item in metrics.items():
        diff = [f['metrics'][name]['roc_auc']-f['metrics'][CONTROL]['roc_auc'] for f in summaries]
        item['fold_weighted_delta_auc'] = float(np.average(diff, weights=weights))
        item['fold_auc_wins'] = sum(d > 0 for d in diff)
        item['fold_logloss_wins'] = sum(f['metrics'][name]['log_loss'] < f['metrics'][CONTROL]['log_loss'] for f in summaries)
        print(name, item, flush=True)
    np.savez_compressed(HERE / 'reverse_predictions.npz', ids=train.id.to_numpy()[dev], fold=fold_ids, **oof)
    write_json(HERE / 'reverse_metrics.json', {'input_sha256': hashes, 'rows': len(dev), 'fit_reference_rows': len(ref),
                                               'variants': metrics, 'folds': summaries, 'seconds': perf_counter()-started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'reverse', 'export'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--method', nargs='+', choices=METHODS, default=METHODS)
    args = parser.parse_args()
    if args.mode == 'reverse':
        reverse(args.device)
    elif args.mode == 'export':
        forward('full', args.device, args.method)
    else:
        forward('development', args.device)
