"""Replay the original FINAL stack and score new XGBoost branches on its fixed folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / 'plan_runs'
REFERENCE = PLAN / 'results' / 'full'
RESULTS = HERE / 'results'
NESTED = REFERENCE / 'nested'
VARIANTS = ('PAIR-TE', 'RARE-TE', 'PAIR-RARE-TE')
MEMBERS = ('GLM-NEWTON', 'XGB-D03')
PROBABILITY_TOLERANCE = 1e-9
METRIC_TOLERANCE = 1e-10

sys.path.insert(0, str(PLAN))
from ensemble import fit_meta_fold
from run_plan import CONTROL_PATH, PROJECT, REPO, TARGET, frame_hash, metric_row, read_json, sha256
from run_plan import write_json, write_npz
from src.validation import assign_folds, early_stop_assignments


def require(condition, description):
    if not condition:
        raise ValueError(description)


def score_vector(scores, size, label):
    valid = scores.shape == (size,) and np.isfinite(scores).all()
    require(valid and ((scores >= 0) & (scores <= 1)).all(), f'Invalid probabilities in {label}')
    return scores


def read_oof(directory, manifest, ids, folds, label, test_ids=None):
    metrics = read_json(directory / 'metrics.json')
    require(metrics['status'] == 'completed' and metrics['name'] == label, f'Incomplete result: {directory}')
    require(all(metrics[key] == value for key, value in manifest.items()), f'Data contract mismatch: {directory}')
    with np.load(directory / 'predictions.npz', allow_pickle=False) as archive:
        require(np.array_equal(archive['ids'], ids), f'Training IDs differ: {directory}')
        require(np.array_equal(archive['fold'], folds), f'Fold assignments differ: {directory}')
        if test_ids is not None:
            require(np.array_equal(archive['test_ids'], test_ids), f'Test IDs differ: {directory}')
        scores = score_vector(archive['oof'], len(ids), directory)
    return scores, metrics


def load_data_contract():
    manifest = read_json(REFERENCE / 'manifest.json')
    require(not manifest['smoke'], 'Reference must use the official full-data folds')
    control = read_json(CONTROL_PATH)['T01']
    expected_inputs = {name: item['sha256'] for name, item in control['input_files'].items()}
    require(manifest['inputs'] == expected_inputs, 'Reference CSV metadata changed')
    for name, expected in expected_inputs.items():
        require(sha256(PROJECT.parent / "data" / name) == expected, f'CSV hash mismatch: {name}')

    train = pd.read_csv(PROJECT.parent / 'data' / 'train.csv', usecols=['id', TARGET])
    test = pd.read_csv(PROJECT.parent / 'data' / 'test.csv', usecols=['id'])
    sample = pd.read_csv(PROJECT.parent / 'data' / 'sample_submission.csv', usecols=['id'])
    ids, test_ids = train['id'].to_numpy(), test['id'].to_numpy()
    require(len(ids) == manifest['rows'] and len(test_ids) == manifest['test_rows'], 'CSV row count differs')
    require(train['id'].is_unique and test['id'].is_unique and not np.intersect1d(ids, test_ids).size,
            'IDs must be distinct within and between train and test')
    require(np.array_equal(test_ids, sample['id'].to_numpy()), 'Submission IDs differ from test order')
    y = train[TARGET].to_numpy(dtype=np.int8)
    config = read_json(REPO / 'configs' / 'default.json')
    n_folds = config['n_folds']
    assignments = assign_folds(train, TARGET, n_folds, config['seed'])
    early = early_stop_assignments(train, assignments, config, n_folds)
    require(frame_hash(assignments) == manifest['fold_hash'] == control['fold_hash'], 'Outer fold hash mismatch')
    require(frame_hash(early) == manifest['early_stop_hash'] == control['early_stop_hash'], 'Early stop hash mismatch')
    return manifest, ids, test_ids, y, assignments['fold'].to_numpy(), n_folds


def nested_inner_scores(ids, y, outer_train, number, config, spec_hashes):
    seed = config['inner_seed']
    splitter = StratifiedKFold(n_splits=config['inner_folds'], shuffle=True, random_state=seed)
    scores = np.full((len(outer_train), len(MEMBERS)), np.nan)
    for inner, (_, holdout) in enumerate(splitter.split(outer_train, y[outer_train]), start=1):
        holdout_ids = ids[outer_train[holdout]]
        for column, (name, spec_hash) in enumerate(zip(MEMBERS, spec_hashes, strict=True)):
            path = NESTED / f'outer_{number}_inner_{inner}_{name}.npz'
            with np.load(path, allow_pickle=False) as archive:
                require(np.array_equal(archive['ids'], holdout_ids)
                        and archive['seed'].item() == seed and archive['spec_hash'].item() == spec_hash,
                        f'Inner IDs, seed or model configuration changed: {path.name}')
                scores[holdout, column] = score_vector(archive['prediction'], len(holdout), path.name)
    require(np.isfinite(scores).all(), f'Incomplete inner-fold predictions for outer fold {number}')
    return scores


def score_summary(y, scores, folds, n_folds):
    score_vector(scores, len(y), 'replayed FINAL')
    return {**metric_row(y, scores), 'folds': [
        {'fold': fold + 1, 'auc': float(roc_auc_score(y[folds == fold], scores[folds == fold]))}
        for fold in range(n_folds)
    ]}


def verify_tree_config(metrics, baseline_config, n_folds):
    config = metrics['config']
    for key in ('family', 'params', 'smoothing', 'encoder_seed', 'model_seed', 'boost', 'rounds_factor'):
        require(config[key] == baseline_config[key], f'{metrics["name"]}: XGB-D03 setting changed: {key}')
    details = metrics['folds']
    require([record['fold'] for record in details] == list(range(1, n_folds + 1)),
            f'{metrics["name"]}: incomplete fold records')
    require(all(row['device'].startswith('cuda:0:') and row['early']['device'].startswith('cuda:0:')
                for row in details), f'{metrics["name"]}: invalid CUDA training records')


def evaluate(variants, *, inputs_only=False):
    manifest, ids, test_ids, y, folds, n_folds = load_data_contract()
    config = read_json(NESTED / 'config.json')
    require(config['members'] == list(MEMBERS) and config['fold_hash'] == manifest['fold_hash']
            and config['outer_folds'] == config['inner_folds'] == n_folds and not config['smoke'],
            'Nested member order or validation configuration changed')
    stack = config['stack']
    selection = read_json(REFERENCE / 'final' / 'selection.json')
    require(stack['method'] == 'logistic_stack' and selection['selected'] == 'NESTED-STACK',
            'Unexpected original FINAL stacking specification')
    member_configs = [read_json(REFERENCE / name / 'config.json') for name in MEMBERS]
    spec_hashes = [hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
                   for item in member_configs]
    base = [read_oof(REFERENCE / name, manifest, ids, folds, name, test_ids)[0] for name in MEMBERS]
    require([hashlib.sha256(scores.tobytes()).hexdigest() for scores in base] == config['base_prediction_hashes'],
            'Original base-model OOF predictions changed')
    archive_final, archive_metrics = read_oof(REFERENCE / 'final', manifest, ids, folds, 'FINAL', test_ids)
    archived_summary = score_summary(y, archive_final, folds, n_folds)
    require(abs(archived_summary['oof_auc'] - archive_metrics['oof_auc']) <= METRIC_TOLERANCE,
            'Archived FINAL predictions no longer match the recorded AUC')

    alternatives = {}
    prediction_hashes = {}
    for variant in variants:
        name = f'XGB-D03-{variant}'
        directory = RESULTS / 'full' / name
        scores, metrics = read_oof(directory, manifest, ids, folds, name, test_ids)
        verify_tree_config(metrics, member_configs[1], n_folds)
        require(abs(roc_auc_score(y, scores) - metrics['oof_auc']) <= METRIC_TOLERANCE,
                f'{name}: saved predictions differ from the recorded AUC')
        alternatives[f'FINAL-{variant}'] = scores
        prediction_hashes[f'FINAL-{variant}'] = sha256(directory / 'predictions.npz')

    replay = {name: np.full(len(y), np.nan) for name in ('FINAL', *alternatives)}
    meta_folds = []
    for fold in range(n_folds):
        number = fold + 1
        outer_train = np.flatnonzero(folds != fold)
        outer_valid = np.flatnonzero(folds == fold)
        inner = nested_inner_scores(ids, y, outer_train, number, config, spec_hashes)
        with np.load(NESTED / f'outer_{number}.npz', allow_pickle=False) as archived_fold:
            require(np.array_equal(archived_fold['ids'], ids[outer_valid])
                    and np.array_equal(archived_fold['prediction'], archive_final[outer_valid]),
                    f'Archived nested output changed for outer fold {number}')
        if inputs_only:
            print(f'Outer fold {number}: official split and both inner prediction caches verified', flush=True)
            continue

        # One fit on ORIGINAL inner predictions. All outer arrays are inference-only.
        inputs = [np.column_stack((base[0][outer_valid], scores[outer_valid]))
                  for scores in (base[1], *alternatives.values())]
        predictions, details = fit_meta_fold(inner, y[outer_train], *inputs, c=stack['c'], return_details=True)
        require(details['device'].startswith('cuda:0:') and details['converged'],
                f'Meta GLM verification failed for fold {number}: {details}')
        for name, scores in zip(replay, predictions, strict=True):
            replay[name][outer_valid] = score_vector(scores, len(outer_valid), f'{name} fold {number}')
        meta_folds.append({
            'fold': number, 'train_rows': len(outer_train), 'validation_rows': len(outer_valid),
            'device': details['device'], 'iterations': details['iterations'], 'model_state': details['model_state'],
        })
        print(f'Outer fold {number}: replayed original meta fit and scored {len(alternatives)} alternatives', flush=True)
    if inputs_only:
        print(f'Archived FINAL OOF ROC AUC {archive_metrics["oof_auc"]:.9f}; CPU-only inputs checked.', flush=True)
        return

    reference = score_summary(y, replay['FINAL'], folds, n_folds)
    max_error = float(np.max(np.abs(archive_final - replay['FINAL'])))
    delta_auc = reference['oof_auc'] - archived_summary['oof_auc']
    require(max_error <= PROBABILITY_TOLERANCE and abs(delta_auc) <= METRIC_TOLERANCE,
            f'Original FINAL replay differs: max probability error={max_error:.3g}, AUC drift={delta_auc:+.3g}')
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / 'frozen_reference.json', {
        'name': 'FINAL', 'status': 'verified', **manifest, **reference,
        'archive_oof_auc': archived_summary['oof_auc'], 'max_abs_probability_difference': max_error,
        'auc_difference': delta_auc, 'nested_config': config, 'meta_folds': meta_folds,
    })
    print(f'Original FINAL reproduced: OOF ROC AUC {reference["oof_auc"]:.9f}, max probability error {max_error:.3g}',
          flush=True)

    for name, scores in replay.items():
        if name == 'FINAL':
            continue
        result = score_summary(y, scores, folds, n_folds)
        for row, baseline in zip(result['folds'], archived_summary['folds'], strict=True):
            row['delta_auc'] = row['auc'] - baseline['auc']
        deltas = {
            'delta_auc': result['oof_auc'] - archived_summary['oof_auc'],
            'delta_ap': result['average_precision'] - archived_summary['average_precision'],
            'delta_log_loss': result['log_loss'] - archived_summary['log_loss'],
            'fold_wins': sum(row['delta_auc'] > 0 for row in result['folds']),
        }
        target = RESULTS / 'full' / name
        target.mkdir(parents=True, exist_ok=True)
        write_npz(target / 'predictions.npz', ids=ids, fold=folds, oof=scores)
        write_json(target / 'metrics.json', {
            'name': name, 'status': 'completed', **manifest, **result, **deltas,
            'config': {
                'parent': 'FINAL', 'method': 'frozen_logistic_stack', 'c': stack['c'],
                'members': [MEMBERS[0], name.replace('FINAL-', 'XGB-D03-', 1)],
                'meta_trained_with': list(MEMBERS), 'xgb_prediction_sha256': prediction_hashes[name],
            },
        })
        print(f'{name}: OOF ROC AUC {result["oof_auc"]:.9f} ({deltas["delta_auc"]:+.9f}), '
              f'{deltas["fold_wins"]}/{n_folds} fold wins', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-reference', action='store_true', help='Replay just the original FINAL on GPU')
    mode.add_argument('--verify-inputs', action='store_true', help='Check reference inputs on CPU and do not fit')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, help='Select fewer candidate branches')
    args = parser.parse_args()
    if (args.verify_reference or args.verify_inputs) and args.variants:
        parser.error('--variants cannot be combined with a reference verification mode')
    variants = () if args.verify_reference or args.verify_inputs else tuple(dict.fromkeys(args.variants or VARIANTS))
    evaluate(variants, inputs_only=args.verify_inputs)


if __name__ == '__main__':
    main()
