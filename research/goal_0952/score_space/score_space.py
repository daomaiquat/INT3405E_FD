"""Learn heavily shrunk residuals from disagreements between local model scores."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
FOLLOWUP = PARENT / 'followup'
sys.path[:0] = [str(PARENT), str(PARENT / 'robust_validation')]
from grandmaster_inputs import (ABLATION, FULL, HIERARCHY, archive_functions, design, development_caches,
                                digest, full_caches, meta_predictions, official_data, read_json, write_json)
from evaluate_and_export import read_scores
from robust_density import compare, isolated_meta
from residual_models import auc_metrics


CENTERS = 48
BLEND = .50
PENALTY = 55.
MAX_DELTA = .18
METHODS = ('quadratic', 'kernel_from_mean', 'kernel_fold_average')


def disagreement_inputs(matrix, anchor):
    center = (matrix[:, 0] + matrix[:, 1]) * .5
    return np.column_stack((anchor, matrix[:, 0] - matrix[:, 1], matrix[:, 2:5],
                            matrix[:, 5] - center, matrix[:, 6] - center))


def scales(values):
    center = np.median(values, axis=0)
    quantiles = np.quantile(values, [.1, .9], axis=0)
    spread = np.maximum((quantiles[1] - quantiles[0]) / 2.563, 1e-4)
    return center, spread


def transform(values, location):
    center, spread = location
    return np.clip((values - center) / spread, -4., 4.).astype(np.float32)


def squared_distances(a, b):
    s = np.sum(a*a, axis=1, keepdims=True) + np.sum(b*b, axis=1)[None, :] - 2 * (a @ b.T)
    return np.maximum(s, 0)


def quadratic(values):
    # Exclude the square of the ensemble's own score. The purpose is to learn disagreement.
    left, right = np.triu_indices(values.shape[1])
    return (values[:, left[1:]] * values[:, right[1:]]) * .25


class ResidualSurface:
    """Regularized Newton step on a nonlinear basis after removing linear effects."""

    def __init__(self, kind):
        self.kind = kind

    def nuisance_basis(self, matrix, normalized):
        own = normalized[:, 0]
        return np.column_stack((np.ones(len(matrix), dtype=np.float32), transform(matrix, self.linear_scale), own**2, own**3))

    def nonlinear(self, normalized):
        if self.kind == 'quadratic':
            return quadratic(normalized)
        kernel = -squared_distances(normalized, self.centers) / (2 * self.bandwidth_squared)
        return np.exp(kernel)

    def fit(self, matrix, offset, labels):
        if matrix.shape != (len(labels), 7) or offset.shape != labels.shape:
            raise ValueError('Expected seven model logit columns and aligned reference rows')
        features = disagreement_inputs(matrix, offset)
        self.input_scale = scales(features)
        self.linear_scale = scales(matrix)
        normalized = transform(features, self.input_scale)
        if self.kind == 'kernel':
            rng = np.random.default_rng(2931)
            sample = normalized[rng.choice(len(normalized), min(len(normalized), 40000), replace=False)]
            cluster = MiniBatchKMeans(n_clusters=CENTERS, random_state=2931, batch_size=1024, n_init=1, max_iter=40)
            self.centers = cluster.fit(sample).cluster_centers_.astype(np.float32)
            distances = squared_distances(self.centers, self.centers)
            pair = distances[np.triu_indices(CENTERS, 1)]
            self.bandwidth_squared = max(float(np.median(pair)), 1e-4)
        basis = self.nonlinear(normalized)
        linear = self.nuisance_basis(matrix, normalized)
        p = expit(offset)
        weight = (p * (1 - p)).astype(np.float32)
        residual = (labels - p).astype(np.float32)
        weighted = linear * weight[:, None]
        stable = np.eye(linear.shape[1]) * max(float(weight.sum()) * 1e-7, 1e-4)
        self.projection = np.linalg.solve(linear.T @ weighted + stable, weighted.T @ basis).astype(np.float32)
        basis -= linear @ self.projection
        normal = basis.T @ (weight[:, None] * basis) + PENALTY * np.eye(basis.shape[1])
        self.coefficients = np.linalg.solve(normal, basis.T @ residual).astype(np.float32)
        if not np.isfinite(self.coefficients).all():
            raise ValueError('Residual fit produced non-finite coefficients')
        return self

    def predict(self, matrix, offset):
        normalized = transform(disagreement_inputs(matrix, offset), self.input_scale)
        basis = self.nonlinear(normalized) - self.nuisance_basis(matrix, normalized) @ self.projection
        return np.clip(basis @ self.coefficients, -MAX_DELTA, MAX_DELTA).astype(np.float64)


def fit_surfaces(x, p, labels):
    with threadpool_limits(limits=4):
        return {name: ResidualSurface(name).fit(x, p, labels) for name in ('quadratic', 'kernel')}


def deltas_for_archive(model, mode, ns, folds, train_sizes, score_size, xscore, center, weights):
    expected_mean = np.zeros_like(xscore)
    result = np.zeros(score_size)
    for i, (fitting, held) in enumerate(folds, start=1):
        name = f'fold_{i:02}.npz'
        if mode == 'development':
            path = ABLATION / 'validation' / 'artifacts' / name
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['va_idx'], held)
                matrix = design(ns, saved['baseline_hold'], saved['candidate_hold'][:, 1])
        else:
            with np.load(FULL / 'artifacts' / name, allow_pickle=False) as saved, np.load(
                HIERARCHY / 'artifacts' / f'dense_{name}', allow_pickle=False
            ) as dense:
                np.testing.assert_array_equal(saved['held_rows'], held)
                np.testing.assert_array_equal(saved['fit_rows'], fitting)
                np.testing.assert_array_equal(dense['held_rows'], held)
                matrix = design(ns, saved['test'], dense['test'])
        if matrix.shape != (score_size, 7) or len(fitting) + len(held) != train_sizes:
            raise ValueError('Unexpected prediction bank fold shape')
        expected_mean += matrix / len(folds)
        result += model.predict(matrix, center + matrix @ weights) / len(folds)
    np.testing.assert_allclose(expected_mean, xscore, atol=2e-10, rtol=0)
    return result


def residual_by_disagreement(labels, matrix, offset):
    # Quantiles of spread, defined without fraud labels.
    values = disagreement_inputs(matrix, offset)[:, 1:]
    scale = scales(values)
    spread = np.sqrt(np.mean(transform(values, scale)**2, axis=1))
    cuts = np.unique(np.quantile(spread, [0, .5, .8, .95, 1.]))
    groups = np.searchsorted(cuts[1:-1], spread, side='right')
    probabilities = expit(offset)
    return [{'range': [float(cuts[i]), float(cuts[i+1])], 'rows': int((groups == i).sum()),
             'actual': int(labels[groups == i].sum()), 'expected': float(probabilities[groups == i].sum())}
            for i in range(len(cuts)-1)]


def forward(mode, method):
    started = perf_counter()
    train, test, sample, hashes = official_data()
    ns = archive_functions()
    if mode == 'development':
        y, development, held, folds, raw, raw_score, dense, dense_score, metadata = development_caches(train, hashes)
        penalties = metadata['baseline_meta_l2_by_fold']
        ids = train.id.to_numpy()[held]
        labels = y[development]
        existing = read_scores(FOLLOWUP / 'context_holdout.npz', ids, 'ids', ('expanded_half_density',))['expanded_half_density']
    else:
        folds, raw, raw_score, dense, dense_score, _ = full_caches(train, test, hashes)
        penalties = read_json(HIERARCHY / 'manifest.json')['baseline_meta_l2_by_fold']
        ids = test.id.to_numpy()
        labels = train.is_fraud.to_numpy(dtype=np.uint8)
        p = read_scores(FOLLOWUP / 'context_test.npz', ids, 'ids', ('probabilities',))['probabilities']
        existing = logit(np.clip(p, 1e-12, 1 - 1e-12))
    x, destination = design(ns, raw, dense), design(ns, raw_score, dense_score)
    with threadpool_limits(limits=4):
        oof, average, weights = meta_predictions(ns, labels, x, destination, folds, penalties)
        b = float(np.mean(average - destination @ weights))
        np.testing.assert_allclose(average, b + destination @ weights, rtol=0, atol=2e-10)
        if mode == 'development':
            original = pd.read_csv(ABLATION / 'validation' / 'holdout_predictions.csv')
            np.testing.assert_array_equal(ids, original.id)
            np.testing.assert_allclose(expit(average), original.hierarchical_amount, atol=3e-10, rtol=0)
        models = fit_surfaces(x, oof, labels) if method is None else {
            'quadratic' if method == 'quadratic' else 'kernel': ResidualSurface('quadratic' if method == 'quadratic' else 'kernel').fit(x, oof, labels)
        }
        corrections = {}
        if method in (None, 'quadratic'):
            corrections['quadratic'] = models['quadratic'].predict(destination, average)
        if method in (None, 'kernel_from_mean'):
            corrections['kernel_from_mean'] = models['kernel'].predict(destination, average)
        if method in (None, 'kernel_fold_average'):
            corrections['kernel_fold_average'] = deltas_for_archive(models['kernel'], mode, ns, folds, len(labels), len(ids),
                                                                     destination, b, weights)
    scores = {'current': existing, **{name: existing + BLEND * delta for name, delta in corrections.items()}}
    if mode == 'development':
        results = compare(ns, y[held], scores, 'current')
        np.savez_compressed(HERE / 'holdout_predictions.npz', ids=ids, **scores)
        write_json(HERE / 'holdout_metrics.json', {'input_sha256': hashes, 'rows': len(ids), 'variants': results,
                                                   'disagreement_dev': residual_by_disagreement(labels, x, oof),
                                                   'seconds': perf_counter() - started,
                                                   'max_half_logit_shift': {name: float(BLEND * np.abs(delta).max())
                                                                            for name, delta in corrections.items()}})
        for name, row in results.items():
            print(name, row, flush=True)
    else:
        probabilities = expit(scores[method])
        path = HERE / f'submission_{method}.csv'
        temp = path.with_suffix('.tmp')
        sample.assign(is_fraud=probabilities).to_csv(temp, index=False)
        checked = pd.read_csv(temp)
        if list(checked.columns) != ['id', 'is_fraud'] or not checked.id.equals(test.id) or not np.isfinite(probabilities).all():
            raise ValueError('Output CSV failed schema, IDs, or probability check')
        np.testing.assert_allclose(checked.is_fraud.to_numpy(), probabilities, rtol=0, atol=1e-14)
        temp.replace(path)
        np.savez_compressed(HERE / 'test_predictions.npz', ids=ids, predictions=probabilities,
                            baseline=existing, correction=corrections[method])
        write_json(HERE / 'submission_manifest.json', {'input_sha256': hashes, 'model': method, 'file': path.name,
                                                        'rows': len(ids), 'sha256': digest(path)})
        print(f'Verified {path.name} with {len(ids):,} rows', flush=True)


def reversed_check():
    started = perf_counter()
    train, _test, _sample, hashes = official_data()
    y, dev, ref, folds, raw, _, dense, _, _ = development_caches(train, hashes)
    ns = archive_functions()
    labels = y[ref]
    ids = train.id.to_numpy()[dev]
    cached = read_scores(PARENT / 'robust_validation' / 'reversed_predictions.npz', ids, 'ids', ('fold', 'current_density'))
    scores = {'current': cached['current_density'], 'quadratic': np.full(len(dev), np.nan), 'kernel': np.full(len(dev), np.nan)}
    reports = []
    for i, (_, held) in enumerate(folds, start=1):
        np.testing.assert_array_equal(np.flatnonzero(cached['fold'] == i), held)
        if np.intersect1d(ref, dev[held]).size:
            raise ValueError('Training and held rows overlap')
        with np.load(ABLATION / 'validation' / 'artifacts' / f'fold_{i:02}.npz', allow_pickle=False) as a:
            np.testing.assert_array_equal(a['va_idx'], held)
            fitter = design(ns, a['baseline_hold'], a['candidate_hold'][:, 1])
        scoring = design(ns, raw[held], dense[held])
        oof, anchor = isolated_meta(ns, fitter, scoring, labels)
        models = fit_surfaces(fitter, oof, labels)
        for model, fit in models.items():
            scores[model][held] = scores['current'][held] + BLEND * fit.predict(scoring, anchor)
        results = {name: auc_metrics(y[dev[held]], s[held]) for name, s in scores.items()}
        reports.append({'fold': i, 'pairs': int(y[dev[held]].sum()) * int((y[dev[held]] == 0).sum()), 'metrics': results})
        print(i, {key: round(row['roc_auc'] - results['current']['roc_auc'], 8) for key, row in results.items()}, flush=True)
    aggregate = compare(ns, y[dev], scores, 'current')
    for name, result in aggregate.items():
        changes = [r['metrics'][name]['roc_auc'] - r['metrics']['current']['roc_auc'] for r in reports]
        result['pair_weighted_delta_auc'] = float(np.average(changes, weights=[r['pairs'] for r in reports]))
        result['fold_auc_wins'] = sum(delta > 0 for delta in changes)
        result['fold_logloss_wins'] = sum(r['metrics'][name]['log_loss'] < r['metrics']['current']['log_loss'] for r in reports)
        print(name, result, flush=True)
    np.savez_compressed(HERE / 'reversed_predictions.npz', ids=ids, fold=cached['fold'], **scores)
    write_json(HERE / 'reversed_metrics.json', {'input_sha256': hashes, 'rows': len(ids), 'fit_reference_rows': len(ref),
                                                'variants': aggregate, 'folds': reports, 'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'reverse', 'export'))
    parser.add_argument('--method', choices=METHODS)
    args = parser.parse_args()
    if args.mode == 'reverse':
        reversed_check()
    elif args.mode == 'export':
        if args.method is None:
            parser.error('--method is required for export')
        forward('full', args.method)
    else:
        forward('development', None)
