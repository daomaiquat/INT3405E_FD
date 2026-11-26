"""Compare new residual boosters against the frozen S6E11 ensemble."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from threadpoolctl import threadpool_limits
import xgboost as xgb

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from grandmaster_inputs import (HIERARCHY, PROJECT, archive_functions, design, development_caches,
                                digest, full_caches, meta_predictions, official_data, read_json, write_json)
from residual_models import auc_metrics, nested_residual_train
from evaluate_and_export import assess, auc_by_base_score_band, read_scores


ROUNDS = 224
XGB_SETTINGS = {'objective': 'binary:logistic', 'tree_method': 'hist', 'max_depth': 3,
                'min_child_weight': 35, 'max_bin': 512, 'eta': .025, 'reg_lambda': 40, 'reg_alpha': 1.5,
                'subsample': .90, 'colsample_bytree': .85, 'nthread': 4, 'seed': 8337}
CHAMPION = 'incumbent'
PAIRS_MANIFEST = PARENT / 'residual_development_metrics.json'


def risk_weights(base_logits):
    """Reweight pointwise loss smoothly, with a positive floor at high risk."""
    weight = .45 + expit((logit(.03) - np.asarray(base_logits)) / .8)
    return weight / weight.mean()


def enhancement_features(fit_frame, labels, query_frame):
    from class_conditional_density import ClassConditionalDensity
    from multiscale_tree import neighborhood_features

    density = ClassConditionalDensity().fit(fit_frame, labels).transform(query_frame)
    neighborhood = neighborhood_features(fit_frame, labels, query_frame)
    return np.column_stack((density, neighborhood))


def crossfit_extra(encoder, fitting, labels, query):
    inferred = encoder(fitting, labels, query)
    train = np.empty((len(fitting), inferred.shape[1]), dtype=np.float32)
    cv = StratifiedKFold(5, shuffle=True, random_state=2026)
    for fit, held in cv.split(labels, labels):
        train[held] = encoder(fitting.iloc[fit], labels[fit], fitting.iloc[held])
    if not np.isfinite(train).all() or not np.isfinite(inferred).all():
        raise ValueError('Non-finite cross-fitted features')
    return train, inferred


def fit_xgb(x_fit, y_fit, x_query, offset, query_offset, device, weight=None):
    fitting = xgb.QuantileDMatrix(x_fit, label=y_fit, base_margin=offset, weight=weight,
                                 max_bin=512, nthread=4)
    query = xgb.DMatrix(x_query, base_margin=query_offset, nthread=4)
    model = xgb.train({**XGB_SETTINGS, 'device': device}, fitting, num_boost_round=ROUNDS)
    return model.predict(query, output_margin=True).astype(np.float64)


def fit_lightgbm(x_fit, labels, x_query, offset, query_offset):
    import lightgbm as lgb

    settings = {'objective': 'binary', 'learning_rate': .03, 'num_leaves': 5, 'min_data_in_leaf': 180,
                'extra_trees': True, 'max_bin': 255, 'lambda_l2': 30, 'lambda_l1': 1.5,
                'bagging_fraction': .9, 'bagging_freq': 1, 'feature_fraction': .85,
                'seed': 8337, 'num_threads': 6, 'force_col_wise': True, 'verbosity': -1}
    model = lgb.train(settings, lgb.Dataset(x_fit, label=labels, init_score=offset), num_boost_round=224)
    return query_offset + model.predict(x_query, raw_score=True, num_threads=6)


def fit_catboost(x_fit, labels, x_query, offset, query_offset, fit_frame, query_frame, device):
    from catboost import CatBoostClassifier, Pool

    cols = ('merchant_category', 'card_type', 'auth_method', 'channel', 'device_type')

    def input_frame(numeric, raw):
        return pd.concat((pd.DataFrame(numeric), raw[list(cols)].astype(str).reset_index(drop=True)), axis=1)

    data = Pool(input_frame(x_fit, fit_frame), label=labels, baseline=offset, cat_features=list(cols))
    query = Pool(input_frame(x_query, query_frame), cat_features=list(cols))
    model = CatBoostClassifier(task_type='GPU' if device == 'cuda' else 'CPU', iterations=224,
                               learning_rate=.035, depth=4, l2_leaf_reg=28, border_count=128,
                               random_seed=8337, thread_count=5, verbose=False, allow_writing_files=False,
                               one_hot_max_size=8, max_ctr_complexity=2)
    model.fit(data)
    return query_offset + model.predict(query, prediction_type='RawFormulaVal')


BOOSTERS = ('xgb_density', 'xgb_neighborhood', 'xgb_both', 'xgb_weighted', 'xgb_both_weighted',
            'lightgbm_offset', 'catboost_offset')


def recipes():
    base = {CHAMPION: {}}
    for name in BOOSTERS:
        base[name] = {name: 1.0}
        base[f'{name}_half'] = {name: .50}
    base['three_tree_families'] = {'xgb_both': .35, 'lightgbm_offset': .15, 'catboost_offset': .15}
    base['density_weighting_mix'] = {'xgb_density': .35, 'xgb_weighted': .35}
    return base


def combine(incumbent, original, boosters, recipe):
    score = incumbent.copy()
    for name, weight in recipe.items():
        score += .85 * weight * (boosters[name] - original)
    return score


def build_inputs(mode):
    train, test, sample, hashes = official_data()
    frame = pd.concat([train.drop(columns=['is_fraud']), test], ignore_index=True)
    labels = train.is_fraud.to_numpy(dtype=np.uint8)
    if mode == 'development':
        _, fit, query, folds, raw, raw_query, dense, dense_query, metadata = development_caches(train, hashes)
        penalties = metadata['baseline_meta_l2_by_fold']
    else:
        fit = np.arange(len(train))
        query = len(train) + np.arange(len(test))
        folds, raw, raw_query, dense, dense_query, _ = full_caches(train, test, hashes)
        penalties = read_json(HIERARCHY / 'manifest.json')['baseline_meta_l2_by_fold']

    ns = archive_functions()
    dest_ids = frame.id.to_numpy()[query]
    old = read_scores(PARENT / f'residual_{mode}.npz', dest_ids, 'ids', ('offset_xgb_224', 'grandmaster_hierarchical'))
    xfit, xquery = design(ns, raw, dense), design(ns, raw_query, dense_query)
    with threadpool_limits(limits=4):
        offset, query_offset, _ = meta_predictions(ns, labels[fit], xfit, xquery, folds, penalties)
    np.testing.assert_allclose(query_offset, old['grandmaster_hierarchical'], atol=2e-8, rtol=0)
    lgbm = read_scores(PARENT / f'lightgbm_{mode}.npz', dest_ids, 'query_ids', ('query_logits',))['query_logits']
    spline = read_scores(PARENT / f'spline_{mode}.npz', dest_ids, 'query_ids', ('query_logits',))['query_logits']
    original = old['offset_xgb_224']
    incumbent = .85 * original + .10 * lgbm + .05 * spline
    saved_name = 'holdout' if mode == 'development' else 'test'
    checked = read_scores(PARENT / 'selected_predictions.npz', dest_ids, f'{saved_name}_ids', (saved_name,))
    np.testing.assert_array_equal(expit(incumbent), checked[saved_name])
    return ns, train, test, sample, hashes, frame, labels, fit, query, offset, query_offset, original, incumbent


def build_training_view(mode):
    started = perf_counter()
    context = build_inputs(mode)
    _, train, test, _, hashes, frame, labels, fit, query, offset, query_offset, original, _ = context
    covariate_manifest = read_json(PAIRS_MANIFEST)
    if covariate_manifest['input_sha256'] != hashes:
        raise ValueError('EDA features refer to a different competition snapshot')
    pairs = covariate_manifest['selected_residual_pairs']
    print(f'{mode}: exact baseline replay, {len(fit):,} fit rows, {perf_counter() - started:.1f}s', flush=True)
    sys.path.insert(0, str(PROJECT.parent / 's6e11_gpu' / 'plan_runs'))
    from features import FeatureBuilder

    yfit = labels[fit]
    fit_frame, query_frame = frame.iloc[fit], frame.iloc[query]
    builder = FeatureBuilder(train, test)
    x_fit, x_query = builder.make(fit, (query,), groups=('raw', 'te13', 'cents', 'digits'))
    del builder
    pair_fit, pair_query = nested_residual_train(frame, fit, query, yfit, offset, pairs)
    x_fit, x_query = np.column_stack((x_fit, pair_fit)), np.column_stack((x_query, pair_query))
    return context, (x_fit, x_query), (fit_frame, query_frame), started


def build_boosters(mode, device, needed):
    context, (x_fit, x_query), (fit_frame, query_frame), started = build_training_view(mode)
    _, _, _, _, _, _, labels, fit, _, offset, query_offset, original, _ = context
    yfit = labels[fit]
    use_enhancement = set(needed) - {'xgb_weighted', 'lightgbm_offset'}
    if use_enhancement:
        sys.path.insert(0, str(PROJECT / 'novel_model_experiments'))
        from class_conditional_density import ClassConditionalDensity

        extra_fit, extra_query = crossfit_extra(enhancement_features, fit_frame, yfit, query_frame)
        boundary = len(ClassConditionalDensity.feature_names)
        print(f'{mode}: contextual and multiscale statistics cross-fitted, {perf_counter() - started:.1f}s', flush=True)

    columns = {'xgb_density': slice(None, boundary) if use_enhancement else None,
               'xgb_neighborhood': slice(boundary, None) if use_enhancement else None,
               'xgb_both': slice(None), 'xgb_both_weighted': slice(None), 'catboost_offset': slice(None)}
    predictions = {}
    for name in needed:
        if name in columns:
            extra = columns[name]
            inputs = np.column_stack((x_fit, extra_fit[:, extra])), np.column_stack((x_query, extra_query[:, extra]))
        else:
            inputs = x_fit, x_query
        a, b = inputs
        if name == 'lightgbm_offset':
            scores = fit_lightgbm(a, yfit, b, offset, query_offset)
        elif name == 'catboost_offset':
            scores = fit_catboost(a, yfit, b, offset, query_offset, fit_frame, query_frame, device)
        else:
            weight = risk_weights(offset) if name.endswith('weighted') else None
            scores = fit_xgb(a, yfit, b, offset, query_offset, device, weight)
        if scores.shape != original.shape or not np.isfinite(scores).all():
            raise ValueError(f'Invalid array from {name}')
        predictions[name] = scores.astype(np.float64)
        print(f'{mode}: {name} finished after {perf_counter() - started:.1f}s', flush=True)
    return context, predictions


def assess_holdout(ns, y, scores, incumbent):
    halves = next(StratifiedShuffleSplit(1, test_size=.5, random_state=6401).split(y, y))
    scores_by_model = {}
    for name, values in scores.items():
        result = assess(ns, y, values, incumbent)
        result['half_delta_auc'] = [auc_metrics(y[rows], values[rows])['roc_auc']
                                    - auc_metrics(y[rows], incumbent[rows])['roc_auc'] for rows in halves]
        scores_by_model[name] = result
    return scores_by_model


def run_development(device):
    context, boosters = build_boosters('development', device, BOOSTERS)
    ns, _, _, _, hashes, frame, labels, _, query, _, _, original, incumbent = context
    candidate_recipes = recipes()
    scores = {name: combine(incumbent, original, boosters, recipe) for name, recipe in candidate_recipes.items()}
    results = assess_holdout(ns, labels[query], scores, incumbent)
    winner = max(results, key=lambda name: results[name]['roc_auc'])
    # Keep a slightly higher score as a challenger if the improvement is small or unstable.
    uplift = results[winner]['delta_auc']
    promote = winner != CHAMPION and uplift >= .00010 and min(results[winner]['half_delta_auc']) > 0
    record = {'input_sha256': hashes, 'holdout_rows': len(query), 'holdout_fraud': int(labels[query].sum()),
              'holdout_previous_selection': True, 'best': winner, 'promote': bool(promote),
              'promotion_rule': 'delta >= .00010 with positive deltas on both descriptive halves',
              'best_score_band_auc_delta': auc_by_base_score_band(labels[query], incumbent, scores[winner]),
              'candidates': results, 'recipes': candidate_recipes}
    np.savez_compressed(HERE / 'holdout_predictions.npz', ids=frame.id.to_numpy()[query], **scores)
    write_json(HERE / 'holdout_metrics.json', record)
    for name, row in sorted(results.items(), key=lambda item: -item[1]['roc_auc']):
        print(f'{name:23s} {row["roc_auc"]:.8f}  {row["delta_auc"]:+.8f}', flush=True)


def export(device):
    record = read_json(HERE / 'holdout_metrics.json')
    winner = record['best']
    if winner == CHAMPION:
        print('Incumbent led the comparison; its existing submission remains the best result.', flush=True)
        return
    recipe = recipes()[winner]
    if recipe != record['recipes'][winner]:
        raise ValueError('Recipe differs from evaluated variant')
    context, boosters = build_boosters('full', device, tuple(recipe))
    _, _, test, sample, hashes, _, _, _, _, _, _, original, incumbent = context
    probabilities = expit(combine(incumbent, original, boosters, recipe))
    name = 'submission_followup.csv' if record['promote'] else 'submission_challenger.csv'
    path = HERE / name
    staged = path.with_suffix('.tmp')
    sample.assign(is_fraud=probabilities).to_csv(staged, index=False)
    reread = pd.read_csv(staged)
    if list(reread.columns) != ['id', 'is_fraud'] or not reread.id.equals(test.id):
        raise ValueError('Wrong Kaggle CSV schema or ID order')
    np.testing.assert_allclose(reread.is_fraud.to_numpy(), probabilities, atol=1e-14, rtol=0)
    staged.replace(path)
    np.savez_compressed(HERE / 'test_predictions.npz', ids=test.id.to_numpy(), predictions=probabilities,
                        original=incumbent, **boosters)
    write_json(HERE / 'submission_manifest.json', {'filename': name, 'sha256': digest(path), 'rows': len(probabilities),
                                                   'best': winner, 'recipe': recipe, 'input_sha256': hashes})
    print(f'Verified {name}, {len(probabilities):,} IDs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'export'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    (run_development if args.mode == 'development' else export)(args.device)
