"""Cross-fit a monotone probability calibration of the frozen density ensemble."""

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from robust_density import FOLLOWUP, HERE, development_caches, digest, official_data, read_scores, write_json


PENALTY = np.array([40., 250.])


def fit_calibrator(logits, labels):
    identity = np.array([0., 1.])
    size = len(labels)

    def objective(parameters):
        intercept, slope = parameters
        z = intercept + slope * logits
        residual = expit(z) - labels
        distance = parameters - identity
        loss = np.sum(np.logaddexp(0., z) - labels * z) + .5 * np.dot(PENALTY, distance**2)
        gradient = np.array([residual.sum(), np.dot(residual, logits)]) + PENALTY * distance
        return loss / size, gradient / size

    result = minimize(objective, identity, jac=True, method='L-BFGS-B', bounds=[(None, None), (.20, 3.)],
                      options={'gtol': 1e-9, 'ftol': 1e-13})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f'Calibrator did not converge: {result.message}')
    return tuple(result.x)


def metrics(labels, scores):
    probabilities = expit(scores)
    return {'auc': float(roc_auc_score(labels, scores)), 'log_loss': float(log_loss(labels, probabilities)),
            'brier': float(brier_score_loss(labels, probabilities))}


def run(export):
    train, test, sample, hashes = official_data()
    labels, _, held, *_ = development_caches(train, hashes)
    y = labels[held]
    ids = train.id.to_numpy()[held]
    before = read_scores(FOLLOWUP / 'context_holdout.npz', ids, 'ids', ('expanded_half_density',))['expanded_half_density']
    after = np.full(len(held), np.nan)
    folds = []
    cv = StratifiedKFold(5, shuffle=True, random_state=7301)
    for fit, valid in cv.split(y, y):
        intercept, slope = fit_calibrator(before[fit], y[fit])
        after[valid] = intercept + slope * before[valid]
        original, calibrated = metrics(y[valid], before[valid]), metrics(y[valid], after[valid])
        np.testing.assert_allclose(original['auc'], calibrated['auc'], rtol=0, atol=1e-12)
        folds.append({'intercept': intercept, 'slope': slope, 'original': original, 'calibrated': calibrated})
    original, calibrated = metrics(y, before), metrics(y, after)
    won = sum(fold['calibrated']['log_loss'] < fold['original']['log_loss'] for fold in folds)
    stable = won >= 4 and calibrated['log_loss'] < original['log_loss'] and calibrated['brier'] < original['brier']
    report = {'input_sha256': hashes, 'predictions_precede_calibration_fit': True,
              'fold_auc_unchanged': True, 'logloss_wins': won, 'stable': bool(stable),
              'crossfit_original': original, 'crossfit_calibrated': calibrated, 'folds': folds}
    if export and stable:
        intercept, slope = fit_calibrator(before, y)
        ids = test.id.to_numpy()
        prior = read_scores(FOLLOWUP / 'context_test.npz', ids, 'ids', ('probabilities',))['probabilities']
        if not ((prior > 0) & (prior < 1)).all():
            raise ValueError('Calibration needs finite original logits')
        probabilities = expit(intercept + slope * logit(prior))
        by_original = np.argsort(prior, kind='stable')
        if (np.diff(probabilities[by_original]) < 0).any():
            raise ValueError('Monotone calibration changed probability order')
        path = HERE / 'submission_calibrated.csv'
        sample.assign(is_fraud=probabilities).to_csv(path, index=False)
        recovered = pd.read_csv(path)
        if list(recovered.columns) != ['id', 'is_fraud'] or not recovered.id.equals(test.id):
            raise ValueError('Invalid calibrated submission schema')
        np.testing.assert_allclose(recovered.is_fraud, probabilities, rtol=0, atol=1e-14)
        report['submission'] = {'sha256': digest(path), 'file': path.name, 'rows': len(ids),
                                'intercept': intercept, 'slope': slope}
    write_json(HERE / 'calibration_metrics.json', report)
    print(f'Original {original}\nCross-fitted calibration {calibrated}\nLog-loss wins {won}/5', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', action='store_true')
    run(parser.parse_args().export)
