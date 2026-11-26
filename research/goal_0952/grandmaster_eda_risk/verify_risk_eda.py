"""Independently check split identity, metrics, and all exported probabilities."""

import argparse

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from risk_eda import CONTROL, HERE, PARENT, file_prefix, get_experiment
from grandmaster_inputs import development_caches, digest, official_data, read_json


def measures(y, z):
    probability = expit(z)
    return {'roc_auc': roc_auc_score(y, z), 'average_precision': average_precision_score(y, z),
            'log_loss': log_loss(y, probability), 'brier': brier_score_loss(y, probability)}


def verify_metrics(prefix, methods, train, rows, y, hashes, folds=()):
    report = read_json(HERE / f'{prefix}_metrics.json')
    assert report['input_sha256'] == hashes and report['rows'] == len(rows)
    with np.load(HERE / f'{prefix}_predictions.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['ids'], train.id.to_numpy()[rows])
        baseline_auc = roc_auc_score(y[rows], saved[CONTROL])
        for name in (CONTROL, *methods):
            logits = saved[name]
            assert np.isfinite(logits).all()
            actual = measures(y[rows], logits)
            archived = report['variants'][name]
            for key, value in actual.items():
                np.testing.assert_allclose(value, archived[key], atol=1e-12, rtol=0)
            np.testing.assert_allclose(actual['roc_auc'] - baseline_auc, archived['delta_auc'], atol=1e-12, rtol=0)
            if folds:
                for i, (_, held) in enumerate(folds, start=1):
                    np.testing.assert_array_equal(saved['fold'][held], i)
                    auc = roc_auc_score(y[rows[held]], logits[held])
                    expected = report['folds'][i - 1]['variants'][name]['roc_auc']
                    np.testing.assert_allclose(auc, expected, atol=1e-12, rtol=0)
            else:
                delta = sum(b['contribution_to_auc_delta'] for b in archived['score_band_contribution'])
                np.testing.assert_allclose(delta, archived['delta_auc'], atol=1e-12, rtol=0)
    print(f'{prefix}: verified metrics on {len(rows):,} official rows')


def verify(family):
    train, test, sample, hashes = official_data()
    y, dev, held, folds, *_ = development_caches(train, hashes)
    methods, _, _ = get_experiment(family, hashes)
    development = file_prefix(family, 'development')
    full = file_prefix(family, 'full')
    verify_metrics(development, methods, train, held, y, hashes)
    verify_metrics(file_prefix(family, 'reverse'), methods, train, dev, y, hashes, folds)
    parent_csv = PARENT / 'submission_grandmaster_eda.csv'
    parent_sha = digest(parent_csv)
    assert parent_sha == read_json(PARENT / 'evaluation.json')['submission_sha256']
    with np.load(HERE / f'{development}_predictions.npz', allow_pickle=False) as scores, \
         np.load(PARENT / 'selected_predictions.npz', allow_pickle=False) as original:
        np.testing.assert_array_equal(scores['ids'], original['holdout_ids'])
        np.testing.assert_array_equal(expit(scores[CONTROL]), original['holdout'])
    with np.load(HERE / f'{full}_predictions.npz', allow_pickle=False) as scores, \
         np.load(PARENT / 'selected_predictions.npz', allow_pickle=False) as original:
        np.testing.assert_array_equal(scores['ids'], test.id)
        np.testing.assert_array_equal(scores['ids'], original['test_ids'])
        np.testing.assert_array_equal(expit(scores[CONTROL]), original['test'])
        for name in methods:
            manifest = read_json(HERE / f'{name}_manifest.json')
            destination = HERE / f'submission_grandmaster_eda_{name}.csv'
            assert manifest['filename'] == destination.name and manifest['rows'] == len(test)
            assert manifest['input_sha256'] == hashes and manifest['parent_sha256'] == parent_sha
            assert digest(destination) == manifest['sha256']
            csv = pd.read_csv(destination)
            assert list(csv.columns) == list(sample.columns)
            assert csv.id.is_unique and csv.id.equals(sample.id)
            assert csv.is_fraud.between(0., 1.).all()
            np.testing.assert_allclose(csv.is_fraud, expit(scores[name]), atol=1e-14, rtol=0)
            print(f'{destination.name}: {len(csv):,} rows, SHA-256 {manifest["sha256"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', default='risk', choices=('risk', 'extra'))
    verify(parser.parse_args().family)
