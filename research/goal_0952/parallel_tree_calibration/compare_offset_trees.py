"""Compare two regularized LightGBM corrections over the exact Grandmaster EDA."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REFINEMENT = PARENT / 'grandmaster_eda_refinement'
RISK = PARENT / 'grandmaster_eda_risk'
sys.path[:0] = [str(PARENT), str(REFINEMENT)]
from refine_eda import CONTROL, XGB_WEIGHT, compare, isolated_meta, select_pairs
from evaluate_and_export import read_scores
from grandmaster_inputs import (ABLATION, PROJECT, archive_functions, design, development_caches,
                                digest, official_data, read_json, write_json)
from residual_models import auc_metrics, nested_residual_train
from run_followup import build_training_view, fit_lightgbm

REPLACEMENT = .15
METHODS = ('additive_stumps', 'five_leaf_trees')
STUMP_ROUNDS = 600
STUMP_SETTINGS = {
    'objective': 'binary', 'learning_rate': .025, 'num_leaves': 2, 'min_data_in_leaf': 180,
    'extra_trees': True, 'max_bin': 255, 'lambda_l2': 30, 'lambda_l1': 1.5,
    'bagging_fraction': .9, 'bagging_freq': 1, 'feature_fraction': .85,
    'seed': 8337, 'num_threads': 4, 'force_col_wise': True, 'deterministic': True, 'verbosity': -1,
}
META = {
    'replacement_of_xgb_log_odds': REPLACEMENT,
    'xgb_weight_in_full_control': XGB_WEIGHT,
    'lightgbm_weight_in_full_variant': REPLACEMENT * XGB_WEIGHT,
    'lightgbm_version': lgb.__version__,
    'methods': {
        'additive_stumps': {'rounds': STUMP_ROUNDS, 'settings': STUMP_SETTINGS},
        'five_leaf_trees': {'rounds': 224, 'implementation': 'followup/run_followup.py::fit_lightgbm'},
    },
}


def fit_stumps(fitting, labels, scoring, offsets, score_offsets):
    data = lgb.Dataset(fitting, label=labels, init_score=offsets)
    booster = lgb.train(STUMP_SETTINGS, data, num_boost_round=STUMP_ROUNDS)
    return score_offsets + booster.predict(scoring, raw_score=True, num_threads=4)


def predict_alternatives(fitting, labels, scoring, offsets, score_offsets, name, started):
    output = {}
    for method, learner in zip(METHODS, (fit_stumps, fit_lightgbm), strict=True):
        margins = learner(fitting, labels, scoring, offsets, score_offsets)
        if margins.shape != score_offsets.shape or not np.isfinite(margins).all():
            raise ValueError(f'Invalid {method} raw margins')
        output[method] = margins
        print(f'{name}: {method} finished after {perf_counter() - started:.1f}s', flush=True)
    return output


def combine(incumbent, control_tree, candidates, xgb_weight):
    return {CONTROL: incumbent, **{name: incumbent + xgb_weight * REPLACEMENT * (tree - control_tree)
                                  for name, tree in candidates.items()}}


def check_hashes(path, hashes):
    manifest = read_json(path)
    if manifest['input_sha256'] != hashes:
        raise ValueError(f'Competition inputs differ from {path}')
    return manifest


def print_measures(variants):
    for name, result in variants.items():
        print(f'{name:18} AUC {result["roc_auc"]:.10f} | delta {result["delta_auc"]:+.10f} | '
              f'AP {result["average_precision"]:.9f} | log loss {result["log_loss"]:.9f}', flush=True)


def forward(mode):
    context, (fitting, scoring), _, started = build_training_view(mode)
    ns, _, test, sample, hashes, frame, labels, fit, query, p, q, tree, incumbent = context
    check_hashes(PARENT / f'residual_{mode}_metrics.json', hashes)
    original = check_hashes(PARENT / 'evaluation.json', hashes)
    ids = frame.id.to_numpy()[query]
    if np.intersect1d(fit, query).size:
        raise ValueError('Fitting and scoring rows overlap')
    alternatives = predict_alternatives(fitting, labels[fit], scoring, p, q, mode, started)
    scores = combine(incumbent, tree, alternatives, XGB_WEIGHT)
    metadata = {**META, 'input_sha256': hashes, 'fit_rows': len(fit), 'rows': len(query),
                'feature_count': fitting.shape[1], 'control_sha256': original['submission_sha256'],
                'seconds': perf_counter() - started}

    if mode == 'development':
        variants = compare(ns, labels[query], scores, CONTROL)
        np.testing.assert_allclose(variants[CONTROL]['roc_auc'], original['best_holdout_auc'], rtol=0, atol=1e-12)
        previous = check_hashes(PARENT / 'followup' / 'holdout_metrics.json', hashes)
        old_five_leaf = read_scores(PARENT / 'followup' / 'holdout_predictions.npz', ids, 'ids',
                                    ('lightgbm_offset',))['lightgbm_offset']
        replay = incumbent + XGB_WEIGHT * (alternatives['five_leaf_trees'] - tree)
        np.testing.assert_allclose(replay, old_five_leaf, rtol=0, atol=1e-11)
        metadata.update(holdout_seed=original['holdout_seed'], holdout_previously_used=True,
                        reference_auc=previous['candidates']['incumbent']['roc_auc'], variants=variants,
                        five_leaf_replay_max_margin_error=float(np.max(np.abs(replay - old_five_leaf))))
        np.savez_compressed(HERE / 'holdout_predictions.npz', ids=ids, **scores)
        write_json(HERE / 'holdout_metrics.json', metadata)
        print_measures(variants)
    else:
        path = PARENT / 'submission_grandmaster_eda.csv'
        if digest(path) != original['submission_sha256']:
            raise ValueError('Original Grandmaster EDA CSV digest mismatch')
        archived = checked_csv(path, test.id)
        np.testing.assert_allclose(archived.is_fraud, expit(incumbent), rtol=0, atol=1e-14)
        metadata['outputs'] = {}
        for method in METHODS:
            path = HERE / f'submission_grandmaster_eda_{method}.csv'
            temp = path.with_suffix('.tmp')
            probability = expit(scores[method])
            sample.assign(is_fraud=probability).to_csv(temp, index=False)
            reread = checked_csv(temp, test.id)
            np.testing.assert_allclose(reread.is_fraud, probability, rtol=0, atol=1e-14)
            temp.replace(path)
            metadata['outputs'][method] = {'filename': path.name, 'sha256': digest(path), 'rows': len(reread)}
            print(f'Export checked: {path.name}, {len(reread):,} rows', flush=True)
        np.savez_compressed(HERE / 'test_predictions.npz', ids=ids, **scores)
        write_json(HERE / 'submission_manifest.json', metadata)


def reverse():
    started = perf_counter()
    train, test, _, hashes = official_data()
    y, dev, reference, folds, raw, _, dense, _, _ = development_caches(train, hashes)
    if np.intersect1d(dev, reference).size:
        raise ValueError('Reverse evaluation and fitting rows overlap')
    pairs = select_pairs(hashes)
    ref_labels = y[reference]
    ids = train.id.to_numpy()[dev]
    base_path = REFINEMENT / 'reverse_predictions.npz'
    check_hashes(REFINEMENT / 'reverse_metrics.json', hashes)
    base = read_scores(base_path, ids, 'ids', ('fold', CONTROL))
    check_hashes(RISK / 'reverse_metrics.json', hashes)
    independent_cache = read_scores(RISK / 'reverse_predictions.npz', ids, 'ids', ('fold', CONTROL))
    np.testing.assert_array_equal(base['fold'], independent_cache['fold'])
    np.testing.assert_allclose(base[CONTROL], independent_cache[CONTROL], rtol=0, atol=4e-6)
    ns = archive_functions()
    sys.path.insert(0, str(PROJECT.parent / 's6e11_gpu' / 'plan_runs'))
    from features import FeatureBuilder

    raw_train, raw_eval = FeatureBuilder(train, test).make(reference, (dev,), groups=('raw', 'te13', 'cents', 'digits'))
    oof = {name: np.full(len(dev), np.nan) for name in METHODS}
    fold_records = []
    for i, (_, held) in enumerate(folds, start=1):
        np.testing.assert_array_equal(base['fold'][held], np.full(len(held), i))
        path = ABLATION / 'validation' / 'artifacts' / f'fold_{i:02}.npz'
        with np.load(path, allow_pickle=False) as checkpoint:
            np.testing.assert_array_equal(checkpoint['va_idx'], held)
            source = design(ns, checkpoint['baseline_hold'], checkpoint['candidate_hold'][:, 1])
        p, q = isolated_meta(ns, source, design(ns, raw[held], dense[held]), ref_labels)
        eda_train, eda_eval = nested_residual_train(train, reference, dev[held], ref_labels, p, pairs)
        fitting = np.column_stack((raw_train, eda_train))
        scoring = np.column_stack((raw_eval[held], eda_eval))
        alternatives = predict_alternatives(fitting, ref_labels, scoring, p, q, f'fold {i}/10', started)
        control = base[CONTROL][held]
        blended = combine(control, control, alternatives, 1.)
        labels = y[dev[held]]
        metrics = {name: auc_metrics(labels, scores) for name, scores in blended.items()}
        for name in METHODS:
            oof[name][held] = blended[name]
        positives = int(labels.sum())
        fold_records.append({'fold': i, 'rows': len(held), 'fraud': positives,
                             'pairs': positives * (len(held) - positives), 'variants': metrics})

    if any(not np.isfinite(score).all() for score in oof.values()):
        raise ValueError('Missing reverse fold predictions')
    scores = {CONTROL: base[CONTROL], **oof}
    results = compare(ns, y[dev], scores, CONTROL)
    for name, values in results.items():
        deltas = [record['variants'][name]['roc_auc'] - record['variants'][CONTROL]['roc_auc']
                  for record in fold_records]
        deltas_loss = [record['variants'][name]['log_loss'] - record['variants'][CONTROL]['log_loss']
                       for record in fold_records]
        values.update(fold_weighted_delta_auc=float(np.average(deltas, weights=[r['pairs'] for r in fold_records])),
                      fold_auc_wins=sum(delta > 0 for delta in deltas),
                      fold_logloss_wins=sum(delta < 0 for delta in deltas_loss),
                      fold_deltas_auc=deltas)
    np.savez_compressed(HERE / 'reverse_predictions.npz', ids=ids, fold=base['fold'], **scores)
    write_json(HERE / 'reverse_metrics.json', {**META, 'input_sha256': hashes, 'fit_rows': len(reference),
               'rows': len(dev), 'variants': results, 'folds': fold_records, 'reverse_control_sha256': digest(base_path),
               'scope': 'isolated XGBoost-plus-Grandmaster branch, same 15% replacement in each fold',
               'seconds': perf_counter() - started})
    print_measures(results)


def checked_csv(path, ids):
    result = pd.read_csv(path)
    if list(result.columns) != ['id', 'is_fraud'] or not result.id.equals(ids) or not result.id.is_unique:
        raise ValueError(f'Schema or ID mismatch: {path}')
    values = result.is_fraud.to_numpy()
    if not np.isfinite(values).all() or not result.is_fraud.between(0, 1).all():
        raise ValueError(f'Probability outside [0, 1]: {path}')
    return result


def verify():
    train, test, _, hashes = official_data()
    y, dev, held, folds, *_ = development_caches(train, hashes)
    for prefix, rows, fit_rows in (('holdout', held, dev), ('reverse', dev, held)):
        report = check_hashes(HERE / f'{prefix}_metrics.json', hashes)
        ids = train.id.to_numpy()[rows]
        saved = read_scores(HERE / f'{prefix}_predictions.npz', ids, 'ids', (CONTROL, *METHODS))
        if report['rows'] != len(rows) or report['fit_rows'] != len(fit_rows):
            raise ValueError(f'Unexpected {prefix} row count')
        for name, logits in saved.items():
            for metric, value in auc_metrics(y[rows], logits).items():
                np.testing.assert_allclose(report['variants'][name][metric], value, rtol=0, atol=1e-12)
        if prefix == 'reverse':
            fold_ids = read_scores(HERE / 'reverse_predictions.npz', ids, 'ids', ('fold',))['fold']
            for i, (_, fold_rows) in enumerate(folds, start=1):
                np.testing.assert_array_equal(fold_ids[fold_rows], np.full(len(fold_rows), i))
                fold = report['folds'][i - 1]
                if fold['fold'] != i or fold['rows'] != len(fold_rows):
                    raise ValueError(f'Unexpected fold metadata for fold {i}')
                for name, logits in saved.items():
                    for metric, value in auc_metrics(y[dev[fold_rows]], logits[fold_rows]).items():
                        np.testing.assert_allclose(fold['variants'][name][metric], value, rtol=0, atol=1e-12)
        print(f'Verified {prefix} ID alignment and metrics ({len(rows):,} rows)', flush=True)

    report = check_hashes(HERE / 'submission_manifest.json', hashes)
    archived = PARENT / 'submission_grandmaster_eda.csv'
    if digest(archived) != report['control_sha256'] or set(report['outputs']) != set(METHODS):
        raise ValueError('Unexpected base checksum or set of submissions')
    saved = read_scores(HERE / 'test_predictions.npz', test.id.to_numpy(), 'ids', (CONTROL, *METHODS))
    previous = checked_csv(archived, test.id)
    np.testing.assert_allclose(previous.is_fraud, expit(saved[CONTROL]), rtol=0, atol=1e-14)
    for name, entry in report['outputs'].items():
        path = HERE / entry['filename']
        reread = checked_csv(path, test.id)
        if len(reread) != entry['rows'] or len(reread) != 299984 or digest(path) != entry['sha256']:
            raise ValueError(f'Unexpected row count or SHA-256 for {path}')
        np.testing.assert_allclose(reread.is_fraud, expit(saved[name]), rtol=0, atol=1e-14)
        print(f'Verified {path.name} {len(reread):,} rows SHA-256 {entry["sha256"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'reverse', 'export', 'verify'))
    mode = parser.parse_args().mode
    if mode == 'reverse':
        reverse()
    elif mode == 'verify':
        verify()
    else:
        forward('full' if mode == 'export' else mode)
