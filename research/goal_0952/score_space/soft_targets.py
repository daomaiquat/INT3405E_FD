"""Test a single frozen teacher-label mixture against a simple shrinkage control."""

import argparse
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from score_space import FOLLOWUP, HERE
from grandmaster_inputs import (ABLATION, PROJECT, archive_functions, design, development_caches,
                                digest, official_data, write_json)
from evaluate_and_export import read_scores
from residual_models import auc_metrics
from robust_density import compare, isolated_meta

sys.path.insert(0, str(FOLLOWUP))
from context_expansion import density_and_context
from run_followup import build_training_view, crossfit_extra, fit_xgb


TEACHER_WEIGHT = .20
BRANCH_WEIGHT = .425


def targets(labels, oof_logits):
    if labels.shape != oof_logits.shape or not np.isin(labels, [0, 1]).all() or not np.isfinite(oof_logits).all():
        raise ValueError('Expected binary labels and aligned finite OOF score margins')
    return ((1 - TEACHER_WEIGHT) * labels + TEACHER_WEIGHT * expit(oof_logits)).astype(np.float32)


def train_alternatives(x_train, y, x_score, offset, score_offset, device):
    standard = fit_xgb(x_train, y, x_score, offset, score_offset, device)
    soft = fit_xgb(x_train, targets(y, offset), x_score, offset, score_offset, device)
    shrunk = score_offset + (1 - TEACHER_WEIGHT) * (standard - score_offset)
    return standard, soft, shrunk


def forward(device, mode):
    context, (train_view, query_view), (fit_frame, query_frame), started = build_training_view(mode)
    ns, _, test, sample, hashes, frame, y, fit, query, offset, query_offset, original, incumbent = context
    a, b = crossfit_extra(density_and_context, fit_frame, y[fit], query_frame)
    standard, soft, shrunk = train_alternatives(np.column_stack((train_view, a)), y[fit],
                                               np.column_stack((query_view, b)), offset, query_offset, device)
    ids = frame.id.to_numpy()[query]
    if mode == 'development':
        prior = read_scores(FOLLOWUP / 'context_holdout.npz', ids, 'ids', ('expanded', 'expanded_half_density'))
        np.testing.assert_allclose(incumbent + .85 * (standard-original), prior['expanded'], rtol=0, atol=2e-8)
        baseline = prior['expanded_half_density']
        scores = {'current': baseline, 'shrink_only': baseline + BRANCH_WEIGHT * (shrunk-standard),
                  'soft_targets': baseline + BRANCH_WEIGHT * (soft-standard)}
        results = compare(ns, y[query], scores, 'current')
        np.savez_compressed(HERE / 'soft_holdout_predictions.npz', ids=ids, **scores)
        write_json(HERE / 'soft_holdout_metrics.json', {'input_sha256': hashes, 'rows': len(ids), 'variants': results,
                                                        'teacher_weight': TEACHER_WEIGHT, 'seconds': perf_counter()-started})
        for name, row in results.items():
            print(name, row, flush=True)
    else:
        p = read_scores(FOLLOWUP / 'context_test.npz', ids, 'ids', ('probabilities',))['probabilities']
        baseline = logit(np.clip(p, 1e-12, 1-1e-12))
        updated = expit(baseline + BRANCH_WEIGHT * (soft-standard))
        path = HERE / 'submission_soft_targets.csv'
        temporary = path.with_suffix('.tmp')
        sample.assign(is_fraud=updated).to_csv(temporary, index=False)
        checked = pd.read_csv(temporary)
        if list(checked.columns) != ['id', 'is_fraud'] or not checked.id.equals(test.id) or not np.isfinite(updated).all():
            raise ValueError('Invalid output or ID order for soft label candidate')
        np.testing.assert_allclose(checked.is_fraud.to_numpy(), updated, rtol=0, atol=1e-14)
        temporary.replace(path)
        np.savez_compressed(HERE / 'soft_test_predictions.npz', ids=ids, probability=updated, baseline=baseline,
                            control=standard, learned=soft)
        write_json(HERE / 'soft_submission_manifest.json', {'input_sha256': hashes, 'file': path.name,
                                                             'rows': len(ids), 'sha256': digest(path)})
        print(f'Verified {path.name}, {len(ids):,} rows', flush=True)


def reverse(device):
    started = perf_counter()
    train, test, _, hashes = official_data()
    y, dev, ref, folds, raw, _, dense, _, _ = development_caches(train, hashes)
    ns = archive_functions()
    sys.path.insert(0, str(PROJECT.parent / 's6e11_gpu' / 'plan_runs'))
    from features import FeatureBuilder

    labels = y[ref]
    x, q = FeatureBuilder(train, test).make(ref, (dev,), groups=('raw', 'te13', 'cents', 'digits'))
    a, b = crossfit_extra(density_and_context, train.iloc[ref], labels, train.iloc[dev])
    x, q = np.column_stack((x, a)), np.column_stack((q, b))
    scores = {name: np.full(len(dev), np.nan) for name in ('current', 'shrink_only', 'soft_targets')}
    fold_id = np.zeros(len(dev), dtype=np.int8)
    records = []
    for i, (_, held) in enumerate(folds, start=1):
        if np.intersect1d(ref, dev[held]).size:
            raise ValueError('Reference overlaps held-out rows')
        with np.load(ABLATION / 'validation' / 'artifacts' / f'fold_{i:02}.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['va_idx'], held)
            ref_side = design(ns, saved['baseline_hold'], saved['candidate_hold'][:, 1])
        dest = design(ns, raw[held], dense[held])
        oof, hold = isolated_meta(ns, ref_side, dest, labels)
        standard, soft, shrunk = train_alternatives(x, labels, q[held], oof, hold, device)
        scores['current'][held] = standard
        scores['shrink_only'][held] = .5 * (standard+shrunk)
        scores['soft_targets'][held] = .5 * (standard+soft)
        fold_id[held] = i
        yy = y[dev[held]]
        metrics = {name: auc_metrics(yy, p[held]) for name, p in scores.items()}
        records.append({'fold': i, 'pairs': int(yy.sum()) * int((yy==0).sum()), 'metrics': metrics})
        print(f'Fold {i}/10, AUC delta {metrics["soft_targets"]["roc_auc"]-metrics["current"]["roc_auc"]:+.8f}', flush=True)
    result = compare(ns, y[dev], scores, 'current')
    for name, row in result.items():
        deltas = [part['metrics'][name]['roc_auc'] - part['metrics']['current']['roc_auc'] for part in records]
        row['pair_weighted_delta_auc'] = float(np.average(deltas, weights=[part['pairs'] for part in records]))
        row['auc_wins'] = sum(delta > 0 for delta in deltas)
        row['logloss_wins'] = sum(part['metrics'][name]['log_loss'] < part['metrics']['current']['log_loss'] for part in records)
        print(name, row, flush=True)
    np.savez_compressed(HERE / 'soft_reversed_predictions.npz', ids=train.id.to_numpy()[dev], fold=fold_id, **scores)
    write_json(HERE / 'soft_reversed_metrics.json', {'input_sha256': hashes, 'rows': len(dev), 'fit_reference_rows': len(ref),
                                                     'variants': result, 'folds': records, 'seconds': perf_counter()-started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'reverse', 'export'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.mode == 'reverse':
        reverse(args.device)
    else:
        forward(args.device, 'full' if args.mode == 'export' else 'development')
