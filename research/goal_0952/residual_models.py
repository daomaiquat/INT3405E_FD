"""Test repeatable residual interactions and shallow boosting over Grandmaster."""

import argparse
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import xgboost as xgb

from eda_from_scratch import centered_pair, group_totals
from grandmaster_inputs import (HERE, HIERARCHY, PROJECT, archive_functions, design, development_caches, full_caches,
                                meta_predictions, official_data, read_json, write_json)


QUANTILES = (.02, .05, .10, .20, .35, .50, .65, .80, .90, .95, .98)
ROUNDS = (96, 224)


def fit_codes(series, queries):
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series) and series.nunique() > 25:
        edges = np.unique(np.quantile(series, QUANTILES))
        return (np.searchsorted(edges, series), np.searchsorted(edges, queries), len(edges) + 1)
    levels = pd.Index(series.unique()).sort_values()
    return levels.get_indexer(series) + 1, levels.get_indexer(queries) + 1, len(levels) + 1


def pair_table(fit_left, fit_right, left_size, right_size, labels, probability):
    packed = fit_left * right_size + fit_right
    g, h, n = (block.reshape(left_size, right_size) for block in
               group_totals(packed, labels, probability, left_size * right_size))
    correction = np.clip(centered_pair(g, h) / (h + 25), -.35, .35)
    correction[(h < 8) | (n < 80)] = 0
    return correction


def residual_features(fit, labels, offsets, query, pairs):
    probability = expit(offsets)
    fields = {name for pair in pairs for name in pair}
    fields = {name: fit_codes(fit[name], query[name]) for name in fields}
    output = []
    for left, right in pairs:
        a_fit, a_query, ac = fields[left]
        b_fit, b_query, bc = fields[right]
        table = pair_table(a_fit, b_fit, ac, bc, labels, probability)
        output.append(table[a_query, b_query])

    distance = 'distance_from_home_km'
    fit_cents = np.rint(100 * fit[distance].to_numpy()).astype(np.int64) % 10
    query_cents = np.rint(100 * query[distance].to_numpy()).astype(np.int64) % 10
    g, h, n = group_totals(fit_cents, labels, probability, 10)
    effect = np.where((h >= 8) & (n >= 80), (g - h * g.sum() / (h.sum() + 25)) / (h + 25), 0)
    output.append(np.clip(effect, -.35, .35)[query_cents])
    return np.column_stack(output).astype(np.float32)


def nested_residual_train(frame, fit, query, labels, offsets, pairs):
    references = frame.iloc[fit]
    destination = residual_features(references, labels, offsets, frame.iloc[query], pairs)
    train = np.empty((len(fit), destination.shape[1]), dtype=np.float32)
    inner_cv = StratifiedKFold(5, shuffle=True, random_state=62026)
    for inner_fit, inner_valid in inner_cv.split(labels, labels):
        train[inner_valid] = residual_features(references.iloc[inner_fit], labels[inner_fit], offsets[inner_fit],
                                               references.iloc[inner_valid], pairs)
    return train, destination


def fit_boosting(fitting, labels, fit_offset, scoring, score_offset, device, rounds=ROUNDS, seeds=(8337,)):
    params = {'objective': 'binary:logistic', 'tree_method': 'hist', 'device': device, 'max_depth': 3,
              'min_child_weight': 35, 'max_bin': 512, 'eta': .025, 'reg_lambda': 40, 'reg_alpha': 1.5,
              'subsample': .90, 'colsample_bytree': .85, 'nthread': 4, 'seed': 8337}
    train = xgb.QuantileDMatrix(fitting, label=labels, base_margin=fit_offset, max_bin=512, nthread=4)
    query = xgb.DMatrix(scoring, base_margin=score_offset, nthread=4)
    results = {f'offset_xgb_{n}': np.zeros(len(scoring)) for n in rounds}
    for seed in seeds:
        model = xgb.train({**params, 'seed': seed}, train, num_boost_round=max(rounds))
        for n in rounds:
            values = model.predict(query, output_margin=True, iteration_range=(0, n)).astype(np.float64)
            results[f'offset_xgb_{n}'] += values / len(seeds)
    return results


def fit_ranked_boosting(fitting, labels, fit_offset, scoring, score_offset, device):
    from low_score_rank_objective import LowScoreObjective, low_score_gate

    rounds = (96, 224)
    params = {'tree_method': 'hist', 'device': device, 'max_depth': 3, 'base_score': 0,
              'min_child_weight': 25, 'max_bin': 512, 'eta': .035, 'reg_lambda': 40,
              'reg_alpha': 1.5, 'subsample': .90, 'colsample_bytree': .85, 'nthread': 4, 'seed': 8337}
    train = xgb.QuantileDMatrix(fitting, label=labels, max_bin=512, nthread=4)
    query = xgb.DMatrix(scoring, nthread=4)
    objective = LowScoreObjective(labels, fit_offset, rank_weight=.015, seed=8337)
    model = xgb.train(params, train, num_boost_round=max(rounds),
                      obj=lambda margins, data: objective.gradient_hessian(margins))
    gate = low_score_gate(score_offset)
    return {f'ranked_offset_{n}': score_offset + gate * np.clip(
        model.predict(query, output_margin=True, iteration_range=(0, n)), -.4, .4) for n in rounds}


def auc_metrics(labels, scores):
    return {'roc_auc': float(roc_auc_score(labels, scores)),
            'average_precision': float(average_precision_score(labels, scores)),
            'log_loss': float(log_loss(labels, expit(scores)))}


def run(mode, device, extended=False, ranked=False, bagged=False):
    started = perf_counter()
    train, test, _, hashes = official_data()
    y = train.is_fraud.to_numpy(dtype=np.uint8)
    frame = pd.concat([train.drop(columns=['is_fraud']), test], ignore_index=True)
    nb = archive_functions()
    if mode == 'development':
        _, fit, query, folds, raw, raw_query, dense, dense_query, metadata = development_caches(train, hashes)
        penalties = metadata['baseline_meta_l2_by_fold']
    else:
        folds, raw, raw_query, dense, dense_query, _ = full_caches(train, test, hashes)
        fit = np.arange(len(train))
        query = len(train) + np.arange(len(test))
        penalties = read_json(HIERARCHY / 'manifest.json')['baseline_meta_l2_by_fold']

    base_x, base_query = design(nb, raw, dense), design(nb, raw_query, dense_query)
    with threadpool_limits(limits=4):
        fit_offset, base, weights = meta_predictions(nb, y[fit], base_x, base_query, folds, penalties)
        _, unmodified, _ = meta_predictions(nb, y[fit], design(nb, raw, None), design(nb, raw_query, None),
                                            folds, penalties)
    if mode == 'development':
        reference = pd.read_csv(PROJECT / 'grandmaster_10fold_ablation' / 'validation' / 'holdout_predictions.csv')
        np.testing.assert_array_equal(reference.id.to_numpy(), frame.id.to_numpy()[query])
        np.testing.assert_allclose(expit(base), reference.hierarchical_amount, atol=3e-10, rtol=0)
        np.testing.assert_allclose(expit(unmodified), reference.baseline, atol=3e-10, rtol=0)
    else:
        reference = pd.read_csv(PROJECT / 'grandmaster_10fold_ablation' / 'submission_hierarchical_amount' / 'submission.csv')
        np.testing.assert_array_equal(reference.id.to_numpy(), frame.id.to_numpy()[query])
        np.testing.assert_allclose(expit(base), reference.is_fraud, atol=3e-10, rtol=0)
    print(f'{mode}: exact archive replay verified after {perf_counter() - started:.1f} seconds', flush=True)

    discovery = read_json(HERE / 'residual_eda.json')
    if discovery['input_sha256'] != hashes:
        raise ValueError('Invalid residual EDA source data')
    selected = [item for item in discovery['pair_results'] if (item['cross_delta_logloss_x1m'] or 0) > 2.0][:12]
    pairs = [(record['left'], record['right']) for record in selected]
    pair_train, pair_query = nested_residual_train(frame, fit, query, y[fit], fit_offset, pairs)
    residual = np.clip(pair_query.sum(axis=1, dtype=np.float64), -.50, .50)
    scores = {'grandmaster_original': unmodified, 'grandmaster_hierarchical': base,
              'screened_residual': base + .35 * residual}

    sys.path.insert(0, str(PROJECT.parent / 's6e11_gpu' / 'plan_runs'))
    from features import FeatureBuilder

    builder = FeatureBuilder(train, test)
    fitting, scoring = builder.make(fit, (query,), groups=('raw', 'te13', 'cents', 'digits'))
    del builder
    rounds = (448, 672) if extended else (224,) if bagged else ROUNDS
    extended_fit, extended_query = np.column_stack((fitting, pair_train)), np.column_stack((scoring, pair_query))
    if ranked:
        scores.update(fit_ranked_boosting(extended_fit, y[fit], fit_offset, extended_query, base, device))
    elif bagged:
        bagged_scores = fit_boosting(extended_fit, y[fit], fit_offset, extended_query, base, device, rounds,
                                     (8337, 42, 101, 202, 2032))
        scores['bagged_offset_224'] = bagged_scores['offset_xgb_224']
    else:
        scores.update(fit_boosting(extended_fit, y[fit], fit_offset, extended_query, base, device, rounds))
    if extended:
        no_extra = fit_boosting(fitting, y[fit], fit_offset, scoring, base, device, (224, 448))
        scores.update({key.replace('offset_', 'raw_offset_'): value for key, value in no_extra.items()})
    prefix = 'residual_ranked' if ranked else 'residual_extended' if extended else 'residual_bagged' if bagged else 'residual'
    np.savez_compressed(HERE / f'{prefix}_{mode}.npz', ids=frame.id.to_numpy()[query], **scores)
    metadata = {'input_sha256': hashes, 'selected_residual_pairs': pairs, 'boosting_rounds': list(rounds),
                'meta_weights': weights.tolist(), 'n_fit': len(fit), 'n_scoring': len(query),
                'grandmaster_reference_oof_auc': roc_auc_score(y[fit], fit_offset),
                'elapsed_seconds': perf_counter() - started}
    if bagged:
        metadata['seeds'] = [8337, 42, 101, 202, 2032]
    if mode == 'development':
        metadata['holdout_scores'] = {name: auc_metrics(y[query], val) for name, val in scores.items()}
        print(metadata['holdout_scores'], flush=True)
    write_json(HERE / f'{prefix}_{mode}_metrics.json', metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'full'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument('--extended', action='store_true', help='Fit longer paths with a matched raw-feature control.')
    variants.add_argument('--ranked', action='store_true', help='Test a low-risk gated pairwise ranking correction.')
    variants.add_argument('--bagged', action='store_true', help='Average five seeded fits of the 224-round booster.')
    arguments = parser.parse_args()
    run(arguments.mode, arguments.device, arguments.extended, arguments.ranked, arguments.bagged)
