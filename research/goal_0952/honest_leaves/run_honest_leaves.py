"""Estimate XGBoost leaf corrections on disjoint calibration rows."""

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
REPO = HERE.parents[2]
SOURCE = HERE.parent
sys.path[:0] = [str(SOURCE), str(SOURCE / 'followup'), str(SOURCE / 'robust_validation'),
                str(REPO / 's6e11_gpu' / 'plan_runs')]

from evaluate_and_export import read_scores
from features import FeatureBuilder
from grandmaster_inputs import (ABLATION, GOAL_CACHE as GOAL, archive_functions, design, development_caches,
                                digest, official_data, read_json, write_json)
from residual_models import auc_metrics, nested_residual_train
from robust_density import compare, isolated_meta
from run_followup import ROUNDS, XGB_SETTINGS, build_inputs

CONTROL = 'grandmaster_eda'
METHODS = ('subforest_blend', 'honest_leaf_blend', 'honest_leaf_overlay')
GROUPS = ('raw', 'te13', 'cents', 'digits')
SEED = 18017
PARTS = 3
RIDGE = 80.
STEP = .5
CAP = .15
SUBFOREST_WEIGHT = .25
ENSEMBLE_WEIGHT = .85
BATCH = 12_000
NODE_SLOTS = 2 ** (XGB_SETTINGS['max_depth'] + 1) - 1


def inverse_leaf_indices(model, x):
    leaves = model.predict(xgb.DMatrix(x, nthread=4), pred_leaf=True).astype(np.int64)
    if leaves.shape[1] != ROUNDS or leaves.min() < 0 or leaves.max() >= NODE_SLOTS:
        raise ValueError('Unexpected XGBoost leaf shape or node identifier')
    return leaves + np.arange(ROUNDS) * NODE_SLOTS


def leaf_changes(grad, curvature, count, global_gradient, global_curvature):
    center = global_gradient / (global_curvature + RIDGE)
    targets = (grad - curvature * center) / (curvature + RIDGE)
    supported = (count >= 80) & (curvature >= 8.)
    return np.where(supported, np.clip(targets, -CAP, CAP), 0.)


def honest_leaf_table(model, x, labels, offset):
    margins = model.predict(xgb.DMatrix(x, base_margin=offset, nthread=4), output_margin=True)
    probability = expit(margins.astype(np.float64))
    residual = labels - probability
    information = probability * (1 - probability)
    width = ROUNDS * NODE_SLOTS
    grad, curvature, count = (np.zeros(width) for _ in range(3))
    for start in range(0, len(x), BATCH):
        end = min(start + BATCH, len(x))
        local = inverse_leaf_indices(model, x[start:end])
        flat = local.ravel()
        grad += np.bincount(flat, np.repeat(residual[start:end], ROUNDS), minlength=width)
        curvature += np.bincount(flat, np.repeat(information[start:end], ROUNDS), minlength=width)
        count += np.bincount(flat, minlength=width)
    return leaf_changes(grad, curvature, count, residual.sum(), information.sum())


def predict_components(model, x, offset, table):
    plain = model.predict(xgb.DMatrix(x, base_margin=offset, nthread=4), output_margin=True).astype(np.float64)
    update = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), BATCH):
        end = min(start + BATCH, len(x))
        indices = inverse_leaf_indices(model, x[start:end])
        update[start:end] = STEP * table[indices].mean(axis=1)
    return plain, update


def crossfit_structure(builder, frame, fit, query, labels, p, q, pairs, device):
    cv = StratifiedKFold(PARTS, shuffle=True, random_state=SEED)
    plain = np.zeros(len(query))
    update = np.zeros(len(query))
    for structure, calibration in cv.split(labels, labels):
        grow_fit, cal_fit = fit[structure], fit[calibration]
        destinations = np.r_[cal_fit, query]
        if np.intersect1d(grow_fit, destinations).size:
            raise ValueError('Structure and destination rows must be disjoint')
        grow, scoring = builder.make(grow_fit, (destinations,), groups=GROUPS)
        eda_grow, eda_scoring = nested_residual_train(
            frame, grow_fit, destinations, labels[structure], p[structure], pairs
        )
        matrix = xgb.QuantileDMatrix(
            np.column_stack((grow, eda_grow)), label=labels[structure], base_margin=p[structure],
            max_bin=XGB_SETTINGS['max_bin'], nthread=4
        )
        model = xgb.train({**XGB_SETTINGS, 'device': device}, matrix, num_boost_round=ROUNDS)
        combined = np.column_stack((scoring, eda_scoring))
        cal_count = len(calibration)
        table = honest_leaf_table(model, combined[:cal_count], labels[calibration], p[calibration])
        member, adjustment = predict_components(model, combined[cal_count:], q, table)
        plain += member / PARTS
        update += adjustment / PARTS
    return plain, update


def candidates(incumbent, original_xgb, subforest, update, weight):
    replace = weight * SUBFOREST_WEIGHT
    return {
        CONTROL: incumbent,
        METHODS[0]: incumbent + replace * (subforest - original_xgb),
        METHODS[1]: incumbent + replace * (subforest + update - original_xgb),
        METHODS[2]: incumbent + weight * update,
    }


def selected_pairs(hashes):
    setup = read_json(SOURCE / 'residual_development_metrics.json')
    if setup['input_sha256'] != hashes:
        raise ValueError('EDA recipe and official input checksums disagree')
    return [tuple(pair) for pair in setup['selected_residual_pairs']]


def compare_candidates(ns, labels, scores):
    result = compare(ns, labels, scores, CONTROL)
    paired = compare(ns, labels, {name: scores[name] for name in METHODS[:2]}, METHODS[0])
    return result, paired[METHODS[1]]


def forward(mode, device):
    started = perf_counter()
    ns, train, test, sample, hashes, frame, labels, fit, query, p, q, xgb_control, incumbent = build_inputs(mode)
    pairs = selected_pairs(hashes)
    members, update = crossfit_structure(FeatureBuilder(train, test), frame, fit, query, labels[fit], p, q, pairs, device)
    scores = candidates(incumbent, xgb_control, members, update, ENSEMBLE_WEIGHT)
    ids = frame.id.to_numpy()[query]
    metadata = {'input_sha256': hashes, 'structure_folds': PARTS, 'seed': SEED, 'ridge': RIDGE,
                'leaf_step': STEP, 'leaf_cap': CAP, 'subforest_weight': SUBFOREST_WEIGHT,
                'affected_logit_rms': float(np.sqrt(np.mean(update**2))), 'fit_rows': len(fit),
                'rows': len(query)}
    if mode == 'development':
        report, paired = compare_candidates(ns, labels[query], scores)
        metadata.update(variants=report, honest_vs_same_structure=paired)
        for key, result in report.items():
            print(key, result, flush=True)
        print('Honest leaf correction vs same structure:', paired, flush=True)
    else:
        export(sample, ids, scores, hashes)

    np.savez_compressed(HERE / f'{mode}_predictions.npz', ids=ids, **scores)
    metadata['seconds'] = perf_counter() - started
    write_json(HERE / f'{mode}_metrics.json', metadata)


def reverse(device):
    train, test, _, hashes = official_data()
    labels, development, reference, folds, raw, _, dense, _, _ = development_caches(train, hashes)
    if np.intersect1d(development, reference).size:
        raise ValueError('Reverse reference overlaps the development rows')
    baseline_manifest = read_json(SOURCE / 'grandmaster_eda_risk' / 'reverse_metrics.json')
    if baseline_manifest['input_sha256'] != hashes or baseline_manifest['reference_rows'] != len(reference):
        raise ValueError('Reverse score archive does not match this split')
    ids = train.id.to_numpy()[development]
    baseline = read_scores(GOAL / 'grandmaster_eda_risk' / 'reverse_predictions.npz',
                           ids, 'ids', (CONTROL, 'fold'))
    y = labels[reference]
    ns = archive_functions()
    frame = pd.concat((train.drop(columns='is_fraud'), test), ignore_index=True)
    builder = FeatureBuilder(train, test)
    pairs = selected_pairs(hashes)
    outputs = {CONTROL: baseline[CONTROL], **{k: np.full(len(development), np.nan) for k in METHODS}}
    fold_metrics = []
    for fold, (_, held) in enumerate(folds, start=1):
        with np.load(ABLATION / 'validation' / 'artifacts' / f'fold_{fold:02}.npz', allow_pickle=False) as values:
            np.testing.assert_array_equal(values['va_idx'], held)
            before = design(ns, values['baseline_hold'], values['candidate_hold'][:, 1])
        np.testing.assert_array_equal(baseline['fold'][held], fold)
        p, q = isolated_meta(ns, before, design(ns, raw[held], dense[held]), y)
        members, update = crossfit_structure(builder, frame, reference, development[held], y, p, q, pairs, device)
        original = baseline[CONTROL][held]
        scores = candidates(original, original, members, update, 1.)
        for key in METHODS:
            outputs[key][held] = scores[key]
        check = labels[development[held]]
        row = {key: auc_metrics(check, value) for key, value in scores.items()}
        fold_metrics.append({'fold': fold, 'pairs': int(check.sum()) * int((check == 0).sum()), 'variants': row})
        print('Fold', fold, [round(row[key]['roc_auc'] - row[CONTROL]['roc_auc'], 8) for key in METHODS], flush=True)
    results, paired = compare_candidates(ns, labels[development], outputs)
    pair_delta = [a['variants'][METHODS[1]]['roc_auc'] - a['variants'][METHODS[0]]['roc_auc']
                  for a in fold_metrics]
    paired['fold_weighted_delta_auc'] = float(np.average(pair_delta, weights=[a['pairs'] for a in fold_metrics]))
    paired['fold_auc_wins'] = sum(d > 0 for d in pair_delta)
    paired['fold_logloss_wins'] = sum(
        a['variants'][METHODS[1]]['log_loss'] < a['variants'][METHODS[0]]['log_loss'] for a in fold_metrics
    )
    for key, row in results.items():
        diffs = [a['variants'][key]['roc_auc'] - a['variants'][CONTROL]['roc_auc'] for a in fold_metrics]
        row['fold_weighted_delta_auc'] = float(np.average(diffs, weights=[a['pairs'] for a in fold_metrics]))
        row['fold_auc_wins'] = sum(d > 0 for d in diffs)
        row['fold_logloss_wins'] = sum(a['variants'][key]['log_loss'] < a['variants'][CONTROL]['log_loss']
                                      for a in fold_metrics)
        print(key, row, flush=True)
    np.savez_compressed(HERE / 'reverse_predictions.npz', ids=ids, fold=baseline['fold'], **outputs)
    write_json(HERE / 'reverse_metrics.json', {'input_sha256': hashes, 'reference_rows': len(reference),
              'rows': len(ids), 'variants': results, 'honest_vs_same_structure': paired, 'folds': fold_metrics})


def export(sample, ids, logits, hashes):
    manifest = read_json(SOURCE / 'evaluation.json')
    original_path = SOURCE / 'submission_grandmaster_eda.csv'
    if digest(original_path) != manifest['submission_sha256']:
        raise ValueError('Original EDA CSV checksum differs from the manifest')
    original = pd.read_csv(original_path)
    np.testing.assert_array_equal(ids, original.id)
    np.testing.assert_allclose(expit(logits[CONTROL]), original.is_fraud, rtol=0, atol=1e-14)
    for method in METHODS:
        path = HERE / f'submission_grandmaster_eda_{method}.csv'
        temporary = path.with_suffix('.tmp')
        predicted = expit(logits[method])
        sample.assign(is_fraud=predicted).to_csv(temporary, index=False)
        checked = pd.read_csv(temporary)
        if (list(checked.columns) != ['id', 'is_fraud'] or not checked.id.equals(sample.id)
                or not checked.id.is_unique or not checked.is_fraud.between(0., 1.).all()):
            raise ValueError(f'Invalid submission schema or probabilities: {path.name}')
        np.testing.assert_allclose(checked.is_fraud, predicted, rtol=0, atol=1e-14)
        temporary.replace(path)
        write_json(HERE / f'{method}_manifest.json', {'file': path.name, 'rows': len(ids), 'sha256': digest(path),
                   'parent_sha256': manifest['submission_sha256'], 'input_sha256': hashes})
        print(f'Verified {path.name}, {len(ids):,} rows', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'reverse', 'export'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.mode == 'reverse':
        reverse(args.device)
    else:
        forward('full' if args.mode == 'export' else 'development', args.device)
