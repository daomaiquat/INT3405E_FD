"""Compare a joint transaction context feature branch against Grandmaster EDA."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedShuffleSplit

from joint_features import FEATURES_PER_RELATION, RELATIONS, joint_features

HERE = Path(__file__).resolve().parent
GOAL = HERE.parent
sys.path[:0] = [str(GOAL / 'grandmaster_eda_refinement')]
from refine_eda import CONTROL, XGB_WEIGHT, compare, isolated_meta, select_pairs
from evaluate_and_export import auc_by_base_score_band, read_scores
from grandmaster_inputs import (ABLATION, PROJECT, archive_functions, design, development_caches,
                                digest, official_data, read_json, write_json)
from residual_models import auc_metrics, nested_residual_train
from run_followup import ROUNDS, XGB_SETTINGS, build_training_view, crossfit_extra, fit_xgb


METHODS = {'joint_context': 1., 'joint_context_half': .5}
REPLAY_TOLERANCE = 4e-6
PRIMARY_SPLIT_SEED = 6401


def save_metadata(path, hashes, **other):
    write_json(HERE / path, {'input_sha256': hashes, 'relations_in_tier_order': RELATIONS,
                             'features_per_relation': FEATURES_PER_RELATION,
                             'xgb_rounds': ROUNDS, 'xgb_settings': XGB_SETTINGS, **other})


def scores_with_replacement(ensemble, xgb_control, xgb_candidate, weight):
    change = weight * (xgb_candidate - xgb_control)
    return {CONTROL: ensemble, **{name: ensemble + fraction*change for name, fraction in METHODS.items()}}


def paired_metrics(ns, labels, values):
    metrics = compare(ns, labels, values, CONTROL)
    baseline = metrics[CONTROL]
    for result in metrics.values():
        result['delta_average_precision'] = result['average_precision'] - baseline['average_precision']
        result['delta_log_loss'] = result['log_loss'] - baseline['log_loss']
    return metrics


def print_metrics(name, row):
    print(f'{name:20} ROC AUC {row["roc_auc"]:.11f} ({row["delta_auc"]:+.11f}), '
          f'AP {row["average_precision"]:.11f} ({row["delta_average_precision"]:+.11f}), '
          f'log-loss {row["log_loss"]:.11f} ({row["delta_log_loss"]:+.11f})', flush=True)


def export_predictions(test, sample, ids, scores, hashes, replay_error):
    parent = GOAL / 'submission_grandmaster_eda.csv'
    expected = read_json(GOAL / 'evaluation.json')
    if digest(parent) != expected['submission_sha256']:
        raise ValueError('Grandmaster EDA parent CSV checksum changed')
    baseline = pd.read_csv(parent)
    np.testing.assert_array_equal(ids, test.id.to_numpy())
    np.testing.assert_array_equal(ids, baseline.id.to_numpy())
    np.testing.assert_allclose(expit(scores[CONTROL]), baseline.is_fraud, rtol=0., atol=1e-14)
    variants = {}
    for method, fraction in METHODS.items():
        path = HERE / f'submission_grandmaster_eda_{method}.csv'
        tmp = path.with_suffix('.tmp')
        probabilities = expit(scores[method])
        sample.assign(is_fraud=probabilities).to_csv(tmp, index=False)
        checked = pd.read_csv(tmp)
        if (list(checked.columns) != ['id', 'is_fraud'] or not checked.id.equals(test.id)
                or not checked.id.is_unique or not checked.is_fraud.between(0, 1).all()
                or len(checked) != expected['submission_rows']):
            raise ValueError(f'Unexpected IDs, schema or probabilities in {path.name}')
        np.testing.assert_allclose(checked.is_fraud, probabilities, atol=1e-14, rtol=0.)
        tmp.replace(path)
        variants[method] = {'filename': path.name, 'sha256': digest(path), 'rows': len(checked),
                            'ensemble_log_odds_weight_of_new_branch': XGB_WEIGHT * fraction}
        print(f'Verified {path.name} ({len(checked):,} rows, SHA-256 {variants[method]["sha256"]})', flush=True)
    save_metadata('submission_manifest.json', hashes, variants=variants,
                  parent_sha256=expected['submission_sha256'], replay_max_margin_error=replay_error)
    np.savez_compressed(HERE / 'test_predictions.npz', ids=ids, **scores)


def forward(mode, device):
    context, views, frames, started = build_training_view(mode)
    ns, _, test, sample, hashes, frame, labels, fit, query, p, q, archived_xgb, incumbent = context
    fit_labels = labels[fit]
    x_fit, x_query = views
    old = fit_xgb(x_fit, fit_labels, x_query, p, q, device)
    np.testing.assert_allclose(old, archived_xgb, rtol=0., atol=REPLAY_TOLERANCE)
    replay_error = float(np.max(np.abs(old - archived_xgb)))
    print(f'{mode}: XGBoost replay max error {replay_error:.6g}, {perf_counter() - started:.1f}s', flush=True)
    fit_extra, query_extra = crossfit_extra(joint_features, frames[0], fit_labels, frames[1])
    print(f'{mode}: {fit_extra.shape[1]} context features cross-fitted, {perf_counter() - started:.1f}s', flush=True)
    new = fit_xgb(np.column_stack((x_fit, fit_extra)), fit_labels, np.column_stack((x_query, query_extra)),
                  p, q, device)
    scores = scores_with_replacement(incumbent, old, new, XGB_WEIGHT)
    ids = frame.id.to_numpy()[query]
    if mode == 'full':
        export_predictions(test, sample, ids, scores, hashes, replay_error)
        return

    y = labels[query]
    metrics = paired_metrics(ns, y, scores)
    halves = next(StratifiedShuffleSplit(1, test_size=.5, random_state=PRIMARY_SPLIT_SEED).split(y, y))
    for name, result in metrics.items():
        result['half_delta_auc'] = [auc_metrics(y[rows], scores[name][rows])['roc_auc']
                                    - auc_metrics(y[rows], incumbent[rows])['roc_auc'] for rows in halves]
        result['base_score_band_contribution'] = auc_by_base_score_band(y, incumbent, scores[name])
        print_metrics(name, result)
    np.savez_compressed(HERE / 'holdout_predictions.npz', ids=ids, **scores)
    save_metadata('holdout_metrics.json', hashes, rows=len(query), fraud=int(y.sum()), fit_rows=len(fit),
                  weights={'xgb_total': XGB_WEIGHT, 'lightgbm': .1, 'spline': .05, 'xgb_fractions': METHODS},
                  holdout_previously_used=True, replay_max_margin_error=replay_error, variants=metrics,
                  seconds=perf_counter()-started)


def reverse(device):
    started = perf_counter()
    train, test, _, hashes = official_data()
    y, dev, reference, folds, raw, _, dense, _, _ = development_caches(train, hashes)
    if np.intersect1d(reference, dev).size:
        raise ValueError('Reverse fitting reference overlaps scoring rows')
    reference_y = y[reference]
    ns = archive_functions()
    pairs = select_pairs(hashes)
    sys.path.insert(0, str(PROJECT.parent / 's6e11_gpu' / 'plan_runs'))
    from features import FeatureBuilder

    ids = train.id.to_numpy()[dev]
    cached_path = GOAL / 'grandmaster_eda_risk' / 'reverse_predictions.npz'
    cached = read_scores(cached_path, ids, 'ids', ('fold', CONTROL))
    raw_train, raw_query = FeatureBuilder(train, test).make(reference, (dev,), groups=('raw', 'te13', 'cents', 'digits'))
    joint_train, joint_query = crossfit_extra(joint_features, train.iloc[reference], reference_y, train.iloc[dev])
    oof = {name: np.full(len(dev), np.nan) for name in (CONTROL, *METHODS)}
    fold_ids = np.zeros(len(dev), dtype=np.int8)
    fold_records = []
    for i, (_, held) in enumerate(folds, start=1):
        with np.load(ABLATION / 'validation' / 'artifacts' / f'fold_{i:02}.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['va_idx'], held)
            source = design(ns, saved['baseline_hold'], saved['candidate_hold'][:, 1])
        np.testing.assert_array_equal(cached['fold'][held], i)
        p, q = isolated_meta(ns, source, design(ns, raw[held], dense[held]), reference_y)
        eda_fit, eda_query = nested_residual_train(train, reference, dev[held], reference_y, p, pairs)
        a, b = np.column_stack((raw_train, eda_fit)), np.column_stack((raw_query[held], eda_query))
        old = fit_xgb(a, reference_y, b, p, q, device)
        np.testing.assert_allclose(old, cached[CONTROL][held], rtol=0., atol=REPLAY_TOLERANCE)
        new = fit_xgb(np.column_stack((a, joint_train)), reference_y,
                      np.column_stack((b, joint_query[held])), p, q, device)
        scores = scores_with_replacement(old, old, new, 1.)
        y_fold = y[dev[held]]
        fold_ids[held] = i
        measures = {name: auc_metrics(y_fold, values) for name, values in scores.items()}
        for name, values in scores.items():
            oof[name][held] = values
        deltas = {name: {metric: value - measures[CONTROL][metric] for metric, value in row.items()}
                  for name, row in measures.items()}
        fold_records.append({'fold': i, 'rows': len(held), 'pairs': int(y_fold.sum()) * int((y_fold == 0).sum()),
                             'variants': measures, 'deltas': deltas,
                             'replay_max_margin_error': float(np.max(np.abs(old - cached[CONTROL][held])))})
        print(f'Fold {i:02}/10, AUC deltas ' + ' '.join(f'{name} {deltas[name]["roc_auc"]:+.9f}'
                                                       for name in METHODS), flush=True)

    metrics = paired_metrics(ns, y[dev], oof)
    for name, item in metrics.items():
        fold_deltas = [f['deltas'][name] for f in fold_records]
        auc = np.array([row['roc_auc'] for row in fold_deltas])
        ap = np.array([row['average_precision'] for row in fold_deltas])
        loss = np.array([row['log_loss'] for row in fold_deltas])
        item.update(fold_weighted_delta_auc=float(np.average(auc, weights=[f['pairs'] for f in fold_records])),
                    fold_delta_auc_std=float(auc.std(ddof=1)), fold_delta_auc_min=float(auc.min()),
                    fold_delta_auc_max=float(auc.max()), fold_auc_wins=int((auc > 0).sum()),
                    fold_ap_wins=int((ap > 0).sum()), fold_logloss_wins=int((loss < 0).sum()))
        print_metrics(name, item)
    np.savez_compressed(HERE / 'reverse_predictions.npz', ids=ids, fold=fold_ids, **oof)
    save_metadata('reverse_metrics.json', hashes, rows=len(dev), fraud=int(y[dev].sum()), fit_rows=len(reference),
                  protocol='isolated XGBoost branch with fold-specific Grandmaster offsets', variants=metrics,
                  folds=fold_records, seconds=perf_counter()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'reverse', 'export'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.mode == 'reverse':
        reverse(args.device)
    else:
        forward('full' if args.mode == 'export' else 'development', args.device)
