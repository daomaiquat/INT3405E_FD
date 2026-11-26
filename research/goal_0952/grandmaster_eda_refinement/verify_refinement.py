"""Recompute validation metrics and check the exported Kaggle CSV files."""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from refine_eda import CONTROL, HERE, METHODS, PARENT
from evaluate_and_export import auc_by_base_score_band
from grandmaster_inputs import development_caches, digest, official_data, read_json, write_json


def verify_metrics(labels, logits, report):
    probabilities = expit(logits)
    actual = [roc_auc_score(labels, logits), average_precision_score(labels, logits),
              log_loss(labels, probabilities), brier_score_loss(labels, probabilities)]
    names = ('roc_auc', 'average_precision', 'log_loss', 'brier')
    np.testing.assert_allclose(actual, [report[name] for name in names], rtol=0, atol=1e-12)


def verify():
    train, test, sample, hashes = official_data()
    y, dev, held, folds, *_ = development_caches(train, hashes)
    parent = read_json(PARENT / 'evaluation.json')
    original = PARENT / 'submission_grandmaster_eda.csv'
    parent_sha = digest(original)
    assert parent_sha == parent['submission_sha256']
    hold_report = read_json(HERE / 'holdout_metrics.json')
    rev_report = read_json(HERE / 'reverse_metrics.json')
    assert hold_report['input_sha256'] == rev_report['input_sha256'] == hashes
    assert hold_report['rows'] == len(held)
    assert rev_report['rows'] == len(dev) and rev_report['fit_reference_rows'] == len(held)
    assert not np.intersect1d(dev, held).size
    diagnostics = {}
    with np.load(HERE / 'holdout_predictions.npz', allow_pickle=False) as hold, \
         np.load(HERE / 'reverse_predictions.npz', allow_pickle=False) as rev, \
         np.load(PARENT / 'selected_predictions.npz', allow_pickle=False) as parent_scores:
        np.testing.assert_array_equal(hold['ids'], train.id.to_numpy()[held])
        np.testing.assert_array_equal(rev['ids'], train.id.to_numpy()[dev])
        np.testing.assert_array_equal(hold['ids'], parent_scores['holdout_ids'])
        np.testing.assert_array_equal(expit(hold[CONTROL]), parent_scores['holdout'])
        np.testing.assert_allclose(roc_auc_score(y[held], hold[CONTROL]), parent['best_holdout_auc'], atol=1e-12, rtol=0)
        for method in (CONTROL, *METHODS):
            a, b = hold[method], rev[method]
            assert np.isfinite(a).all() and np.isfinite(b).all()
            verify_metrics(y[held], a, hold_report['variants'][method])
            verify_metrics(y[dev], b, rev_report['variants'][method])
            for i, (_, indices) in enumerate(folds, start=1):
                np.testing.assert_array_equal(rev['fold'][indices], i)
                expected = rev_report['folds'][i - 1]['metrics'][method]['roc_auc']
                np.testing.assert_allclose(roc_auc_score(y[dev[indices]], b[indices]), expected, atol=1e-12, rtol=0)
            if method == CONTROL:
                continue
            bands = auc_by_base_score_band(y[held], hold[CONTROL], a)
            delta = hold_report['variants'][method]['delta_auc']
            np.testing.assert_allclose(sum(row['contribution_to_auc_delta'] for row in bands), delta, atol=1e-12, rtol=0)
            diagnostics[method] = {'auc_contribution_by_original_score_percentile': bands}
    with np.load(HERE / 'test_predictions.npz', allow_pickle=False) as scores, \
         np.load(PARENT / 'selected_predictions.npz', allow_pickle=False) as parent_scores:
        np.testing.assert_array_equal(scores['ids'], test.id)
        np.testing.assert_array_equal(scores['ids'], parent_scores['test_ids'])
        np.testing.assert_array_equal(expit(scores[CONTROL]), parent_scores['test'])
        for method in METHODS:
            manifest = read_json(HERE / f'{method}_manifest.json')
            path = HERE / f'submission_grandmaster_eda_{method}.csv'
            assert manifest['file'] == path.name and manifest['method'] == method
            assert manifest['input_sha256'] == hashes and manifest['parent_sha256'] == parent_sha
            assert manifest['rows'] == len(test) and manifest['sha256'] == digest(path)
            csv = pd.read_csv(path)
            assert list(csv.columns) == list(sample.columns) and csv.id.equals(sample.id)
            assert csv.id.is_unique and csv.is_fraud.between(0, 1).all()
            probabilities = expit(scores[method])
            np.testing.assert_allclose(csv.is_fraud, probabilities, atol=1e-14, rtol=0)
            delta = np.abs(probabilities - expit(scores[CONTROL]))
            diagnostics[method]['absolute_probability_delta_on_test'] = {
                str(q): float(np.quantile(delta, q)) for q in (.50, .90, .99)}
            print(f'{path.name}: {len(csv):,} valid rows; SHA-256 {manifest["sha256"]}')
    write_json(HERE / 'score_band_diagnostics.json', diagnostics)
    print('Both validation protocols and the original submission were independently checked.')


if __name__ == '__main__':
    verify()
