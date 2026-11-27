"""Evaluate one frozen, regularized density alternative on two protocols."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import xgboost as xgb

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
FOLLOWUP = PARENT / 'followup'
sys.path[:0] = [str(PARENT), str(FOLLOWUP)]
from context_expansion import CONTEXTS, new_features
from grandmaster_inputs import (ABLATION, ABLATION_CODE, PROJECT, archive_functions, design, development_caches,
                                digest, official_data, read_json, write_json)
from run_followup import ROUNDS, XGB_SETTINGS, build_training_view, crossfit_extra, fit_xgb
from evaluate_and_export import assess, read_scores
from residual_models import auc_metrics, fit_codes

sys.path.insert(0, str(ABLATION_CODE))
from class_conditional_density import ClassConditionalDensity


CONSERVATIVE_TREE = {**XGB_SETTINGS, 'max_depth': 2, 'max_bin': 256, 'min_child_weight': 45, 'reg_lambda': 60}
DENSITY_MIX = .5
STACK_PENALTY = 1e-5


def pool_ordered_counts(counts):
    """Move 20% of each bin to each neighbor, retaining mass at the boundaries."""
    smoothed = .6 * counts
    smoothed[:, 1:] += .2 * counts[:, :-1]
    smoothed[:, :-1] += .2 * counts[:, 1:]
    smoothed[:, 0] += .2 * counts[:, 0]
    smoothed[:, -1] += .2 * counts[:, -1]
    return smoothed


def shrunken_context_contrast(value, context, width, contexts, y, score_value, score_context, ordered):
    keys = context * width + value
    counts = [np.bincount(keys[y == label], minlength=width * contexts).reshape(contexts, width)
              for label in (0, 1)]
    support = [table.sum(axis=1) for table in counts]
    observed_values = (counts[0] + counts[1]).sum(axis=0) > 0
    changes = []
    for table, strength in zip(counts, (180., 60.), strict=True):
        density = pool_ordered_counts(table) if ordered else table
        marginal = density.sum(axis=0) + .75
        marginal /= marginal.sum()
        conditional = (density + strength * marginal) / (density.sum(axis=1, keepdims=True) + strength)
        changes.append(np.log(conditional / marginal))
    effective_support = np.minimum(support[1], support[0] * .05)
    reliability = effective_support / (effective_support + 25.)
    score = (changes[1] - changes[0])[score_context, score_value] * reliability[score_context]
    return np.where(observed_values[score_value], np.clip(score, -2.5, 2.5), 0).astype(np.float32)


def smooth_context(fit, y, query):
    names = {name for variable, ctx in CONTEXTS for name in (variable, *ctx)}
    coded = {name: fit_codes(fit[name], query[name]) for name in names}
    features = []
    for name, ctx in CONTEXTS:
        a, b, width = coded[name]
        left, right, size = coded[ctx[0]]
        if len(ctx) > 1:
            c, d, more = coded[ctx[1]]
            left, right, size = left * more + c, right * more + d, size * more
        ordered = pd.api.types.is_numeric_dtype(fit[name]) and fit[name].nunique() > 25
        features.append(shrunken_context_contrast(a, left, width, size, y, b, right, ordered))
    return np.column_stack(features)


def conservative_features(fit, y, query):
    core = ClassConditionalDensity().fit(fit, y).transform(query)
    return np.column_stack((core, smooth_context(fit, y, query)))


def all_density_features(fit, y, query):
    core = ClassConditionalDensity().fit(fit, y).transform(query)
    return np.column_stack((core, new_features(fit, y, query), smooth_context(fit, y, query)))


def conservative_fit(fit, labels, query, base, query_base, device):
    matrix = xgb.QuantileDMatrix(fit, label=labels, base_margin=base, max_bin=256, nthread=4)
    scoring = xgb.DMatrix(query, base_margin=query_base, nthread=4)
    model = xgb.train({**CONSERVATIVE_TREE, 'device': device}, matrix, num_boost_round=ROUNDS)
    return model.predict(scoring, output_margin=True).astype(np.float64)


def compare(ns, y, predictions, reference):
    results = {}
    for name, scores in predictions.items():
        values = assess(ns, y, scores, predictions[reference])
        values['brier'] = float(brier_score_loss(y, expit(scores)))
        results[name] = values
    return results


def forward(device, mode, smoothed=False):
    context, (train_view, query_view), (fit_frame, query_frame), started = build_training_view(mode)
    ns, _, test, sample, hashes, frame, labels, fit, query, offset, query_offset, original, incumbent = context
    ref_fit, ref_query = crossfit_extra(conservative_features, fit_frame, labels[fit], query_frame)
    booster = fit_xgb if smoothed else conservative_fit
    conservative = booster(np.column_stack((train_view, ref_fit)), labels[fit],
                           np.column_stack((query_view, ref_query)), offset, query_offset, device)
    stand_alone = incumbent + .85 * (conservative - original)
    ids = frame.id.to_numpy()[query]
    if mode == 'development':
        previous = read_scores(FOLLOWUP / 'context_holdout.npz', ids, 'ids', ('expanded_half_density',))['expanded_half_density']
    else:
        previous = logit(np.clip(read_scores(FOLLOWUP / 'context_test.npz', ids, 'ids', ('probabilities',))['probabilities'],
                                1e-12, 1 - 1e-12))
    blended = (1 - DENSITY_MIX) * previous + DENSITY_MIX * stand_alone
    scores = {'without_density': incumbent, 'current_density': previous, 'conservative_only': stand_alone,
              'regularized_blend': blended}
    prefix = 'smoothed_' if smoothed else ''
    if mode == 'development':
        metrics = compare(ns, labels[query], scores, 'current_density')
        write_json(HERE / f'{prefix}forward_metrics.json', {'input_sha256': hashes, 'candidates': metrics,
                                                              'seconds': perf_counter() - started, 'mode': '4242_reused'})
        np.savez_compressed(HERE / f'{prefix}forward_predictions.npz', ids=ids, **scores)
        for name, result in metrics.items():
            print(f'{name:23} AUC {result["roc_auc"]:.9f}, delta {result["delta_auc"]:+.9f}, '
                  f'AP {result["average_precision"]:.9f}, logloss {result["log_loss"]:.9f}', flush=True)
    else:
        probabilities = expit(blended)
        output = HERE / ('submission_smoothed.csv' if smoothed else 'submission_regularized.csv')
        writing = output.with_suffix('.tmp')
        sample.assign(is_fraud=probabilities).to_csv(writing, index=False)
        checked = pd.read_csv(writing)
        if list(checked.columns) != ['id', 'is_fraud'] or not checked.id.equals(test.id) or not np.isfinite(probabilities).all():
            raise ValueError('Unexpected submission IDs or probabilities')
        np.testing.assert_allclose(checked.is_fraud.to_numpy(), probabilities, rtol=0, atol=1e-14)
        writing.replace(output)
        np.savez_compressed(HERE / f'{prefix}test_predictions.npz', ids=ids, prediction=probabilities, **scores)
        write_json(HERE / f'{prefix}submission_manifest.json', {'input_sha256': hashes, 'sha256': digest(output),
                                                              'rows': len(ids), 'weight_of_changed_branch': DENSITY_MIX})
        print(f'Verified {output.name} with {len(ids):,} rows', flush=True)


def isolated_meta(ns, ref, validation, labels):
    """Use only independent reference labels for the stack, cross-fit its input margins."""
    f = ns['_fit_nonneg_l2_weights']
    splitter = StratifiedKFold(3, shuffle=True, random_state=7301)
    training = np.full(len(labels), np.nan)
    with threadpool_limits(limits=3):
        for fit, held in splitter.split(labels, labels):
            b, weights = f(ref[fit], labels[fit], STACK_PENALTY)
            training[held] = b + ref[held] @ weights
        b, weights = f(ref, labels, STACK_PENALTY)
    return training, b + validation @ weights


def reverse(device, smoothed=False):
    started = perf_counter()
    train, test, _, hashes = official_data()
    labels, dev, reference, folds, raw_dev, _, dense_dev, _, metadata = development_caches(train, hashes)
    ns = archive_functions()
    sys.path.insert(0, str(PROJECT.parent / 's6e11_gpu' / 'plan_runs'))
    from features import FeatureBuilder

    ref_labels = labels[reference]
    fitting, scoring = FeatureBuilder(train, test).make(reference, (dev,), groups=('raw', 'te13', 'cents', 'digits'))
    if smoothed:
        cached = read_scores(HERE / 'reversed_predictions.npz', train.id.to_numpy()[dev], 'ids',
                             ('fold', 'base_only', 'current_density'))
        a, b = crossfit_extra(conservative_features, train.iloc[reference], ref_labels, train.iloc[dev])
        variants = {'conservative': (np.column_stack((fitting, a)), np.column_stack((scoring, b)))}
    else:
        cached = None
        a, b = crossfit_extra(all_density_features, train.iloc[reference], ref_labels, train.iloc[dev])
        base_count = len(ClassConditionalDensity.feature_names)
        extended_end = base_count + len(CONTEXTS)
        variants = {
            'base_only': (fitting, scoring),
            'old_density': (np.column_stack((fitting, a[:, :base_count])), np.column_stack((scoring, b[:, :base_count]))),
            'expanded': (np.column_stack((fitting, a[:, :extended_end])), np.column_stack((scoring, b[:, :extended_end]))),
            'conservative': (np.column_stack((fitting, a[:, :base_count], a[:, extended_end:])),
                             np.column_stack((scoring, b[:, :base_count], b[:, extended_end:]))),
        }
    ydev = labels[dev]
    names = ('base_only', 'current_density', 'conservative_only', 'regularized_blend')
    oof = {name: np.full(len(dev), np.nan) for name in names}
    fold_no = np.zeros(len(dev), dtype=np.int8)
    fold_results, cache_hashes = [], {}
    for i, (outer_fit, valid) in enumerate(folds, start=1):
        assert np.intersect1d(reference, dev[valid]).size == 0
        assert np.intersect1d(dev[outer_fit], dev[valid]).size == 0
        path = ABLATION / 'validation' / 'artifacts' / f'fold_{i:02}.npz'
        cache_hashes[path.name] = digest(path)
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['va_idx'], valid)
            ref_x = design(ns, saved['baseline_hold'], saved['candidate_hold'][:, 1])
        held_x = design(ns, raw_dev[valid], dense_dev[valid])
        offsets, held_offsets = isolated_meta(ns, ref_x, held_x, ref_labels)
        output = {}
        for name, (fit_x, all_x) in variants.items():
            booster = conservative_fit if not smoothed and name == 'conservative' else fit_xgb
            output[name] = booster(fit_x, ref_labels, all_x[valid], offsets, held_offsets, device)
        if cached is None:
            current = .5 * (output['old_density'] + output['expanded'])
            base_only = output['base_only']
        else:
            np.testing.assert_array_equal(cached['fold'][valid], np.full(len(valid), i))
            current, base_only = cached['current_density'][valid], cached['base_only'][valid]
        scores = {'base_only': base_only, 'current_density': current, 'conservative_only': output['conservative'],
                  'regularized_blend': (1 - DENSITY_MIX) * current + DENSITY_MIX * output['conservative']}
        fold_no[valid] = i
        for name, s in scores.items():
            oof[name][valid] = s
        values = {name: auc_metrics(ydev[valid], s) for name, s in scores.items()}
        weight = int(ydev[valid].sum()) * int((ydev[valid] == 0).sum())
        fold_results.append({'fold': i, 'rows': len(valid), 'pair_count': weight, 'candidates': values})
        print(f'fold {i:2}/10, baseline AUC {values["current_density"]["roc_auc"]:.8f}, '
              f'regularized {values["regularized_blend"]["roc_auc"]:.8f}, {perf_counter() - started:.1f}s', flush=True)

    metrics = compare(ns, ydev, oof, 'current_density')
    weights = np.array([fold['pair_count'] for fold in fold_results])
    for name, row in metrics.items():
        aucs = np.array([fold['candidates'][name]['roc_auc'] for fold in fold_results])
        control = np.array([fold['candidates']['current_density']['roc_auc'] for fold in fold_results])
        row['pair_weighted_fold_auc'] = np.average(aucs, weights=weights)
        row['pair_weighted_fold_delta'] = np.average(aucs - control, weights=weights)
        row['fold_auc_wins'] = int((aucs > control).sum())
        print(f'{name:23} pooled AUC {row["roc_auc"]:.9f}, fold-weighted delta {row["pair_weighted_fold_delta"]:+.9f}, '
              f'AP {row["average_precision"]:.9f}, logloss {row["log_loss"]:.9f}', flush=True)
    prefix = 'smoothed_' if smoothed else ''
    np.savez_compressed(HERE / f'{prefix}reversed_predictions.npz', ids=train.id.to_numpy()[dev], fold=fold_no, **oof)
    write_json(HERE / f'{prefix}reversed_metrics.json', {'input_sha256': hashes, 'cache_sha256': cache_hashes,
                                                         'outer_seed': metadata['cv_seed'], 'fit_reference_rows': len(reference),
                                                         'validation_rows': len(dev), 'candidates': metrics, 'folds': fold_results,
                                                         'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('forward', 'reverse', 'export'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--smooth-only', action='store_true', help='Use the existing tree settings and only smooth the density.')
    arguments = parser.parse_args()
    if arguments.mode == 'reverse':
        reverse(arguments.device, arguments.smooth_only)
    else:
        forward(arguments.device, 'full' if arguments.mode == 'export' else 'development', arguments.smooth_only)
