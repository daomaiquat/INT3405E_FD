"""Compare held-out candidates and export the selected S6E11 test predictions."""

import argparse

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from grandmaster_inputs import (HERE, HIERARCHY, archive_functions, design, development_caches, digest,
                                full_caches, meta_predictions, official_data, read_json, write_json)
from residual_models import auc_metrics


TARGET_AUC = .952


def read_scores(path, ids, id_column, names):
    with np.load(path, allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache[id_column], ids)
        result = {name: cache[name] for name in names}
    if any(array.shape != ids.shape or not np.isfinite(array).all() for array in result.values()):
        raise ValueError(f'Wrong score shape or nonfinite score in {path.name}')
    return result


def combine(one, two, weight, mode='logit'):
    if mode == 'probability':
        return logit((1 - weight) * expit(one) + weight * expit(two))
    return (1 - weight) * one + weight * two


def base_candidates(mode, ids, extended, bagged=False):
    names = ('grandmaster_original', 'grandmaster_hierarchical', 'screened_residual',
             'offset_xgb_96', 'offset_xgb_224')
    scores = read_scores(HERE / f'residual_{mode}.npz', ids, 'ids', names)
    if extended:
        more = ('offset_xgb_448', 'offset_xgb_672', 'raw_offset_xgb_224', 'raw_offset_xgb_448')
        scores.update(read_scores(HERE / f'residual_extended_{mode}.npz', ids, 'ids', more))
        ranks = ('ranked_offset_96', 'ranked_offset_224')
        scores.update(read_scores(HERE / f'residual_ranked_{mode}.npz', ids, 'ids', ranks))

    branches = {name: read_scores(HERE / f'{name}_{mode}.npz', ids, 'query_ids', ('query_logits',))['query_logits']
                for name in ('mlp', 'spline', 'latent')}
    gm = scores['grandmaster_hierarchical']
    xgb = scores['offset_xgb_224']
    mlp, spline, latent = (branches[name] for name in ('mlp', 'spline', 'latent'))
    scores.update({
        'grandmaster_mlp10': combine(gm, mlp, .10, 'probability'),
        'grandmaster_spline10': combine(gm, spline, .10),
        'grandmaster_latent03': combine(gm, latent, .03),
        'grandmaster_mlp05_spline05': .90 * gm + .05 * mlp + .05 * spline,
        'offset224_mlp10': combine(xgb, mlp, .10, 'probability'),
        'offset224_spline05': combine(xgb, spline, .05),
        'offset224_mlp05_spline05': .90 * xgb + .05 * mlp + .05 * spline,
        'offset224_latent03': combine(xgb, latent, .03),
    })
    if extended:
        lgbm = read_scores(HERE / f'lightgbm_{mode}.npz', ids, 'query_ids', ('query_logits',))['query_logits']
        branches['lightgbm'] = lgbm
        scores['offset448_mlp05_spline05'] = .90 * scores['offset_xgb_448'] + .05 * mlp + .05 * spline
        scores['grandmaster_lgb20_mlp10'] = logit(.70 * expit(gm) + .20 * expit(lgbm) + .10 * expit(mlp))
        scores['offset224_lgb20_mlp10'] = logit(.70 * expit(xgb) + .20 * expit(lgbm) + .10 * expit(mlp))
        scores['offset224_lgb10_spline05'] = .85 * xgb + .10 * lgbm + .05 * spline
        scores['offset224_lgb10'] = combine(xgb, lgbm, .10, 'probability')
    if bagged:
        boosted = read_scores(HERE / f'residual_bagged_{mode}.npz', ids, 'ids', ('bagged_offset_224',))
        scores.update(boosted)
        scores['bagged224_lgb10_spline05'] = .85 * boosted['bagged_offset_224'] + .10 * branches['lightgbm'] + .05 * spline
    return scores, branches


def fit_mlp_meta(mode, labels, fit_ids, query_ids, folds, raw, raw_query, dense, dense_query, penalties, ns):
    mlp = read_scores(HERE / f'mlp_{mode}.npz', fit_ids, 'ids', ('oof_logits',))['oof_logits']
    mlp_query = read_scores(HERE / f'mlp_{mode}.npz', query_ids, 'query_ids', ('query_logits',))['query_logits']
    xfit = np.column_stack((design(ns, raw, dense), mlp))
    xquery = np.column_stack((design(ns, raw_query, dense_query), mlp_query))
    with threadpool_limits(limits=4):
        oof, output, weights = meta_predictions(ns, labels, xfit, xquery, folds, penalties)
    return output, float(weights[-1]), auc_metrics(labels, oof)['roc_auc']


def assess(ns, labels, scores, reference):
    values = auc_metrics(labels, scores)
    # The archive orders cases by -y, which wraps when y is uint8.
    result = ns['delong_paired_test'](labels.astype(np.int8, copy=False), scores, reference)
    expected = values['roc_auc'] - auc_metrics(labels, reference)['roc_auc']
    np.testing.assert_allclose(result['diff'], expected, rtol=0, atol=1e-12)
    values.update(delta_auc=float(result['diff']), ci95_low=float(result['ci_low']), ci95_high=float(result['ci_high']))
    return values


def auc_by_base_score_band(labels, reference, scores):
    positive = labels == 1

    def placements(logits):
        negatives = np.sort(logits[~positive])
        left = np.searchsorted(negatives, logits[positive], side='left')
        right = np.searchsorted(negatives, logits[positive], side='right')
        return (left + right) / (2 * len(negatives))

    differences = placements(scores) - placements(reference)
    quantiles = (0, .50, .75, .90, .95, .99, 1)
    edges = np.quantile(reference, quantiles)
    bins = np.searchsorted(edges[1:-1], reference, side='right')
    return [{'quantile_lower': low, 'quantile_upper': high, 'rows': int((bins == i).sum()),
             'fraud': int((bins[positive] == i).sum()),
             'contribution_to_auc_delta': float(differences[bins[positive] == i].sum() / positive.sum())}
            for i, (low, high) in enumerate(zip(quantiles[:-1], quantiles[1:], strict=True))]


def run(preview, extended, bagged=False):
    train, test, sample, hashes = official_data()
    y, dev, held, folds, raw, raw_held, dense, dense_held, metadata = development_caches(train, hashes)
    ns = archive_functions()
    train_ids, test_ids = train.id.to_numpy(), test.id.to_numpy()
    hold_scores, branches = base_candidates('development', train_ids[held], extended, bagged)
    hold_scores['grandmaster_learned_mlp'], weight, exploratory_auc = fit_mlp_meta(
        'development', y[dev], train_ids[dev], train_ids[held], folds, raw, raw_held, dense, dense_held,
        metadata['baseline_meta_l2_by_fold'], ns)
    ref = hold_scores['grandmaster_hierarchical']
    report = {name: assess(ns, y[held], scores, ref) for name, scores in hold_scores.items()}
    best = max(report, key=lambda name: report[name]['roc_auc'])
    identity = {
        'target_roc_auc': TARGET_AUC, 'holdout_seed': metadata['holdout_seed'], 'holdout_rows': len(held),
        'holdout_fraud': int(y[held].sum()), 'holdout_previously_used': True,
        'heldout_auc_is_selection_metric': True, 'input_sha256': hashes,
        'candidate_count': len(report),
        'candidates': report, 'branch_holdout_scores': {key: auc_metrics(y[held], s) for key, s in branches.items()},
        'selected': best, 'best_holdout_auc': report[best]['roc_auc'],
        'target_reached_on_holdout': bool(report[best]['roc_auc'] >= TARGET_AUC),
        'remaining_to_target': max(0, TARGET_AUC - report[best]['roc_auc']),
        'delta_auc_by_baseline_score_band': auc_by_base_score_band(y[held], ref, hold_scores[best]),
        'development_learned_mlp_logit_weight': weight,
        'development_learned_mlp_exploratory_meta_oof_auc': exploratory_auc,
        'full_original_exploratory_meta_oof_auc': read_json(HERE.parent / 'grandmaster_refit' / 'full' / 'metrics.json')[
            'baseline_oof']['oof_auc'],
    }
    for name, measurements in sorted(report.items(), key=lambda item: -item[1]['roc_auc']):
        print(f'{name:32s} AUC {measurements["roc_auc"]:.8f}  delta {measurements["delta_auc"]:+.8f}', flush=True)
    if preview:
        write_json(HERE / 'preview_metrics.json', identity)
        return

    full_scores, _ = base_candidates('full', test_ids, extended, best.startswith('bagged'))
    if best == 'grandmaster_learned_mlp':
        full_folds, full_raw, test_raw, full_dense, test_dense, _ = full_caches(train, test, hashes)
        p = read_json(HIERARCHY / 'manifest.json')['baseline_meta_l2_by_fold']
        full_scores[best], full_weight, stack_auc = fit_mlp_meta(
            'full', y, train_ids, test_ids, full_folds, full_raw, test_raw, full_dense, test_dense, p, ns)
        identity['full_learned_mlp_logit_weight'] = full_weight
        identity['full_learned_mlp_exploratory_meta_oof_auc'] = stack_auc
    identity['full_hierarchical_exploratory_meta_oof_auc'] = read_json(HERE / 'residual_full_metrics.json')[
        'grandmaster_reference_oof_auc']
    probabilities = expit(full_scores[best])
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError('Invalid test probabilities')
    output = HERE / 'submission_grandmaster_eda.csv'
    temp = output.with_suffix('.tmp')
    sample.assign(is_fraud=probabilities).to_csv(temp, index=False)
    check = pd.read_csv(temp)
    if list(check.columns) != ['id', 'is_fraud']:
        raise ValueError('Invalid CSV schema')
    np.testing.assert_array_equal(check.id.to_numpy(), test_ids)
    np.testing.assert_allclose(check.is_fraud.to_numpy(), probabilities, rtol=0, atol=1e-14)
    temp.replace(output)
    np.savez_compressed(HERE / 'selected_predictions.npz', holdout_ids=train_ids[held],
                        holdout=expit(hold_scores[best]), test_ids=test_ids, test=probabilities)
    identity.update(submission_rows=len(test_ids), submission_sha256=digest(output),
                    submission_min=float(probabilities.min()), submission_max=float(probabilities.max()),
                    submission_mean=float(probabilities.mean()))
    write_json(HERE / 'evaluation.json', identity)
    print(f'Selected {best}; verified {len(test_ids):,} test probabilities', flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--preview', action='store_true')
    cli.add_argument('--extended', action='store_true')
    cli.add_argument('--bagged', action='store_true')
    args = cli.parse_args()
    if args.bagged and not args.extended:
        cli.error('--bagged requires --extended for the LightGBM blend')
    run(args.preview, args.extended, args.bagged)
