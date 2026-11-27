"""Fit small offset-logistic corrections on out-of-fold Grandmaster EDA scores."""

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from build_oof import BRANCHES, HERE, SOURCE, WEIGHTS, partitions
from evaluate_and_export import read_scores
from grandmaster_inputs import (GOAL_CACHE, archive_functions, development_caches, digest, official_data,
                                read_json, write_json)
from residual_models import auc_metrics
from robust_density import compare

CONTROL = 'grandmaster_eda'
AMOUNT = 'amount_usd'
DISTANCE = 'distance_from_home_km'
REFERENCE_ROWS = 594_966
AMOUNT_RECIPE = HERE.parent / 'deep_eda_measurement' / 'amount_hole_submission.json'


@dataclass(frozen=True)
class Recipe:
    terms: tuple[str, ...]
    ridge: tuple[float, ...]
    step: float
    cap: float


RECIPES = {
    'refit_amount_hole': Recipe(('frozen_hole',), (25.,), 1., .30),
    'smooth_amount': Recipe((AMOUNT,), (40., 40.), 1., .30),
    'smooth_amount_distance': Recipe((AMOUNT, DISTANCE), (40., 40., 60., 60.), 1., .30),
    'linear_branch_adjustment': Recipe(('branch_linear',), (120.,) * 3, .50, .15),
    'risk_branch_adjustment': Recipe(('branch',), (120.,) * 6, .50, .15),
}


def ticks(values):
    scaled = values * 100
    integer = np.rint(scaled).astype(np.int64)
    return integer, np.abs(scaled - integer) <= 1e-6


def neighbor_deficit(reference, query):
    """Compare each cent tick with its neighbors at two different widths."""
    fit, valid = ticks(reference)
    dest, on_grid = ticks(query)
    points, counts = np.unique(fit[valid], return_counts=True)
    cumulative = np.r_[0, counts.cumsum()]
    first = np.searchsorted(points, dest)
    after = np.searchsorted(points, dest, side='right')
    count = cumulative[after] - cumulative[first]
    size_factor = REFERENCE_ROWS / len(reference)
    count = count * size_factor
    neighbor = []
    for width in (5, 25):
        left = np.searchsorted(points, dest - width)
        right = np.searchsorted(points, dest + width, side='right')
        neighbor.append(((cumulative[right] - cumulative[left]) * size_factor - count) / (2 * width))
    near, wide = neighbor
    ratio = (count + 5.) / (np.minimum(near, wide) + 5.)
    deficit = np.clip(-np.log(ratio) / np.log(2.5), 0., 1.)
    support = np.clip((np.minimum(near, wide) - 20.) / 20., 0., 1.) * np.minimum(count / 10., 1.)
    interior = on_grid & (dest > points[0] + 25) & (dest < points[-1] - 25)
    return np.column_stack((deficit, np.maximum(0., deficit - .6) / .4)) * (support * interior)[:, None]


def load_oof(mode, ids, hashes, y):
    path = HERE / f'{mode}_oof.npz'
    manifest = read_json(HERE / f'{mode}_oof_manifest.json')
    if (manifest['sha256'] != digest(path) or manifest['input_sha256'] != hashes
            or manifest['branch_order'] != list(BRANCHES) or manifest['weights'] != WEIGHTS.tolist()):
        raise ValueError('OOF bank does not match this experiment')
    with np.load(path, allow_pickle=False) as files:
        np.testing.assert_array_equal(files['ids'], ids)
        scores, fold = files['scores'], files['fold']
    if scores.shape != (len(ids), len(BRANCHES)) or not np.isfinite(scores).all():
        raise ValueError('Unexpected out-of-fold score bank')
    expected = np.zeros(len(y))
    for i, (_, valid) in enumerate(partitions(y), start=1):
        expected[valid] = i
    np.testing.assert_array_equal(fold, expected)
    return scores, fold


def final_bank(mode, ids):
    residual = read_scores(GOAL_CACHE / f'residual_{mode}.npz', ids, 'ids',
                           ('grandmaster_hierarchical', 'offset_xgb_224'))
    lgb = read_scores(GOAL_CACHE / f'lightgbm_{mode}.npz', ids, 'query_ids', ('query_logits',))['query_logits']
    spline = read_scores(GOAL_CACHE / f'spline_{mode}.npz', ids, 'query_ids', ('query_logits',))['query_logits']
    bank = np.column_stack((residual['grandmaster_hierarchical'], residual['offset_xgb_224'], lgb, spline))
    logits = blend(bank)
    key = 'holdout' if mode == 'development' else 'test'
    archive = read_scores(SOURCE / 'selected_predictions.npz', ids, f'{key}_ids', (key,))[key]
    np.testing.assert_array_equal(expit(logits), archive)
    return bank, logits


def blend(bank):
    # Match the arithmetic in the frozen model to preserve its original probabilities.
    return WEIGHTS[1] * bank[:, 1] + WEIGHTS[2] * bank[:, 2] + WEIGHTS[3] * bank[:, 3]


def feature_blocks(reference, query, ref_bank, query_bank, y, frozen_ticks):
    ref, score = {}, {}
    for col in (AMOUNT, DISTANCE):
        a, b = reference[col].to_numpy(), query[col].to_numpy()
        ref[col] = np.zeros((len(a), 2))
        for fit, held in partitions(y):
            ref[col][held] = neighbor_deficit(a[fit], a[held])
        score[col] = neighbor_deficit(a, b)

    for groups, bank, frame in ((ref, ref_bank, reference), (score, query_bank, query)):
        cents, valid = ticks(frame[AMOUNT].to_numpy())
        groups['frozen_hole'] = (valid & np.isin(cents, frozen_ticks))[:, None].astype(np.float64)
        diff = np.clip(bank[:, (0, 2, 3)] - bank[:, [1]], -2., 2.)
        gate = 2 * expit(blend(bank) + 3.) - 1.
        groups['branch_linear'] = diff
        groups['branch'] = np.column_stack((diff, diff * gate[:, None]))
    return ref, score


def fit_offset(labels, base, feature, recipe):
    """Fit a shared intercept and slope as nuisance terms in the residual model."""
    z = np.clip(base + 4., -5., 5.)
    terms = np.column_stack((np.ones(len(base)), z, feature))
    penalty = np.r_[1., 10., recipe.ridge]

    def objective(beta):
        logits = base + terms @ beta
        loss = np.logaddexp(0, logits) - labels * logits
        return loss.sum() + .5 * np.dot(beta**2, penalty), terms.T @ (expit(logits) - labels) + penalty * beta

    with threadpool_limits(limits=4):
        result = minimize(objective, np.zeros(len(penalty)), method='L-BFGS-B', jac=True,
                          options={'ftol': 1e-13, 'gtol': 1e-6, 'maxiter': 250})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f'Calibration fit failed: {result.message}')
    return result.x


def adjustment(x, beta, recipe):
    # Applying the feature effects alone avoids a population-wide recalibration.
    return np.clip(recipe.step * (x @ beta[2:]), -recipe.cap, recipe.cap)


def fit_recipes(y, ref, ref_blocks, query, query_blocks, folds, screen):
    logits = {CONTROL: query}
    details = {}
    for name, recipe in RECIPES.items():
        x = np.column_stack([ref_blocks[key] for key in recipe.terms])
        score_x = np.column_stack([query_blocks[key] for key in recipe.terms])
        if screen:
            checks = []
            for fold in np.unique(folds):
                held = folds == fold
                beta = fit_offset(y[~held], ref[~held], x[~held], recipe)
                contrasted = ref[held] + adjustment(x[held], beta, recipe)
                untreated = auc_metrics(y[held], ref[held])
                treated = auc_metrics(y[held], contrasted)
                checks.append({'fold': int(fold), 'rows': int(held.sum()),
                               'delta_log_loss': treated['log_loss'] - untreated['log_loss'],
                               'delta_auc': treated['roc_auc'] - untreated['roc_auc'],
                               'feature_coefficients': beta[2:].tolist()})
            mean_loss = np.average([item['delta_log_loss'] for item in checks], weights=[item['rows'] for item in checks])
            wins = sum(item['delta_log_loss'] < 0 for item in checks)
            details[name] = {'folds': checks, 'wins_log_loss': wins, 'weighted_delta_log_loss': float(mean_loss),
                             'passes_development_screen': bool(wins >= 2 and mean_loss < 0.)}
        beta = fit_offset(y, ref, x, recipe)
        delta = adjustment(score_x, beta, recipe)
        logits[name] = query + delta
        details.setdefault(name, {}).update(feature_coefficients=beta[2:].tolist(), step=recipe.step,
                                          ridge=list(recipe.ridge), cap=recipe.cap, nz_rows=int(np.count_nonzero(delta)),
                                          rms_adjustment=float(np.sqrt(np.mean(delta**2))))
    return logits, details


def export(sample, logits, hashes, mode):
    parent_manifest = read_json(SOURCE / 'evaluation.json')
    path = SOURCE / 'submission_grandmaster_eda.csv'
    if digest(path) != parent_manifest['submission_sha256']:
        raise ValueError('The original Grandmaster submission has changed')
    original = pd.read_csv(path)
    np.testing.assert_array_equal(original.id, sample.id)
    np.testing.assert_allclose(original.is_fraud, expit(logits[CONTROL]), atol=1e-14, rtol=0)
    output = {}
    for name in RECIPES:
        path = HERE / f'submission_grandmaster_eda_{name}.csv'
        staging = path.with_suffix('.tmp')
        probability = expit(logits[name])
        sample.assign(is_fraud=probability).to_csv(staging, index=False)
        readback = pd.read_csv(staging)
        if (list(readback.columns) != list(sample.columns) or not readback.id.equals(sample.id)
                or not readback.id.is_unique or not readback.is_fraud.between(0, 1).all()):
            raise ValueError(f'Invalid CSV {path.name}')
        np.testing.assert_allclose(readback.is_fraud, probability, atol=1e-14, rtol=0)
        staging.replace(path)
        output[name] = {'file': path.name, 'rows': len(readback), 'sha256': digest(path)}
    return {'files': output, 'input_sha256': hashes, 'calibration_reference': mode,
            'parent_sha256': parent_manifest['submission_sha256']}


def run(mode):
    train, test, sample, hashes = official_data()
    labels, dev, held, *_ = development_caches(train, hashes)
    if mode == 'export-full':
        ref_rows, ref_mode = np.arange(len(train)), 'full'
    else:
        ref_rows, ref_mode = dev, 'development'
    reference = train.iloc[ref_rows]
    y = labels[ref_rows]
    ref_bank, folds = load_oof(ref_mode, reference.id.to_numpy(), hashes, y)
    destination = train.iloc[held] if mode == 'development' else test
    score_bank, baseline = final_bank('development' if mode == 'development' else 'full', destination.id.to_numpy())
    prior = read_json(AMOUNT_RECIPE)
    if prior['input_sha256'] != hashes or len(set(prior['frozen_amount_ticks'])) != 404:
        raise ValueError('The frozen amount-hole recipe does not match the official experiment')
    ref, scoring = feature_blocks(reference, destination, ref_bank, score_bank, y, prior['frozen_amount_ticks'])
    predictions, details = fit_recipes(y, blend(ref_bank), ref, baseline, scoring, folds, mode == 'development')
    report = {'input_sha256': hashes, 'calibration_rows': len(reference), 'scored_rows': len(destination),
              'calibration_bank_sha256': read_json(HERE / f'{ref_mode}_oof_manifest.json')['sha256'], 'recipes': details}
    if mode == 'development':
        write_json(HERE / 'development_screen.json', report)
        frozen = scoring['frozen_hole'][:, 0]
        predictions['frozen_amount_hole'] = baseline + frozen * prior['dev_validation_logit_shift']
        report['holdout'] = compare(archive_functions(), labels[held], predictions, CONTROL)
    else:
        report.update(export(sample, predictions, hashes, ref_mode))
    np.savez_compressed(HERE / f'{mode}_predictions.npz', ids=destination.id.to_numpy(), **predictions)
    report_path = HERE / f'{mode}_metrics.json'
    write_json(report_path, report)
    if mode == 'development':
        for name, values in report['holdout'].items():
            print(f"{name}: AUC {values['roc_auc']:.8f}, log loss {values['log_loss']:.8f}", flush=True)
    else:
        for item in report['files'].values():
            print(f"{item['file']}: {item['rows']:,} rows", flush=True)
    print(report_path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'export-development', 'export-full'))
    run(parser.parse_args().mode)
