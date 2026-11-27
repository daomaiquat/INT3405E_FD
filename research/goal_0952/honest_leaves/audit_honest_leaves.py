"""Recheck the saved experiments and compare a matched shrinkage control."""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from run_honest_leaves import (ABLATION, CONTROL, ENSEMBLE_WEIGHT, GOAL, HERE, METHODS, SOURCE,
                               SUBFOREST_WEIGHT, archive_functions, compare, design,
                               development_caches, digest, isolated_meta, official_data,
                               read_json, read_scores, write_json)
from evaluate_and_export import auc_by_base_score_band

SHRUNK = 'shrink_toward_grandmaster'


def measure(labels, logits):
    p = expit(logits)
    return (roc_auc_score(labels, logits), average_precision_score(labels, logits),
            log_loss(labels, p), brier_score_loss(labels, p))


def verify_metrics(labels, logits, metrics):
    fields = ('roc_auc', 'average_precision', 'log_loss', 'brier')
    np.testing.assert_allclose(measure(labels, logits), [metrics[key] for key in fields], atol=1e-12, rtol=0)


def paired_summary(ns, labels, predictions, fold_records=()):
    summary = compare(ns, labels, predictions, CONTROL)
    if fold_records:
        for name, values in summary.items():
            d = [fold['auc'][name] - fold['auc'][CONTROL] for fold in fold_records]
            values['fold_weighted_delta_auc'] = float(np.average(d, weights=[row['pairs'] for row in fold_records]))
            values['fold_auc_wins'] = sum(x > 0 for x in d)
            values['fold_logloss_wins'] = sum(a['loss'][name] < a['loss'][CONTROL] for a in fold_records)
    return summary


def run():
    train, test, sample, hashes = official_data()
    labels, development, held, folds, raw, _, dense, _, _ = development_caches(train, hashes)
    ns = archive_functions()
    ids = train.id.to_numpy()
    dev_scores = read_scores(HERE / 'development_predictions.npz', ids[held], 'ids', (CONTROL, *METHODS))
    reverse = read_scores(HERE / 'reverse_predictions.npz', ids[development], 'ids', (CONTROL, *METHODS, 'fold'))
    test_scores = read_scores(HERE / 'full_predictions.npz', test.id.to_numpy(), 'ids', (CONTROL, *METHODS))
    dev_report = read_json(HERE / 'development_metrics.json')
    rev_report = read_json(HERE / 'reverse_metrics.json')
    assert dev_report['input_sha256'] == rev_report['input_sha256'] == hashes
    assert dev_report['rows'] == len(held) and rev_report['rows'] == len(development)
    with np.load(GOAL / 'selected_predictions.npz', allow_pickle=False) as original:
        np.testing.assert_array_equal(expit(dev_scores[CONTROL]), original['holdout'])
        np.testing.assert_array_equal(expit(test_scores[CONTROL]), original['test'])

    parent_sha = read_json(SOURCE / 'evaluation.json')['submission_sha256']
    assert digest(SOURCE / 'submission_grandmaster_eda.csv') == parent_sha
    for name in (CONTROL, *METHODS):
        verify_metrics(labels[held], dev_scores[name], dev_report['variants'][name])
        verify_metrics(labels[development], reverse[name], rev_report['variants'][name])
        if name != CONTROL:
            manifest = read_json(HERE / f'{name}_manifest.json')
            output = HERE / f'submission_grandmaster_eda_{name}.csv'
            assert manifest['parent_sha256'] == parent_sha and manifest['input_sha256'] == hashes
            assert manifest['file'] == output.name and manifest['sha256'] == digest(output)
            prediction = pd.read_csv(output)
            assert list(prediction.columns) == list(sample.columns) and prediction.id.equals(sample.id)
            assert len(prediction) == manifest['rows'] == 299_984
            assert prediction.id.is_unique and prediction.is_fraud.between(0, 1).all()
            np.testing.assert_allclose(prediction.is_fraud, expit(test_scores[name]), atol=1e-14, rtol=0)
            print(f'Verified {output.name} ({len(prediction):,} rows)')

    archived = read_scores(GOAL / 'residual_development.npz', ids[held], 'ids',
                           ('offset_xgb_224', 'grandmaster_hierarchical'))
    dev_scores[SHRUNK] = dev_scores[CONTROL] + ENSEMBLE_WEIGHT * SUBFOREST_WEIGHT * (
        archived['grandmaster_hierarchical'] - archived['offset_xgb_224']
    )
    original_folds = reverse.pop('fold')
    reverse[SHRUNK] = np.empty(len(development))
    fold_records = []
    for i, (_, fold) in enumerate(folds, start=1):
        with np.load(ABLATION / 'validation' / 'artifacts' / f'fold_{i:02}.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['va_idx'], fold)
            start = design(ns, saved['baseline_hold'], saved['candidate_hold'][:, 1])
        np.testing.assert_array_equal(original_folds[fold], i)
        _, logits = isolated_meta(ns, start, design(ns, raw[fold], dense[fold]), labels[held])
        reverse[SHRUNK][fold] = reverse[CONTROL][fold] + SUBFOREST_WEIGHT * (logits - reverse[CONTROL][fold])
        yy = labels[development[fold]]
        auc = {key: roc_auc_score(yy, value[fold]) for key, value in reverse.items()}
        loss = {key: log_loss(yy, expit(value[fold])) for key, value in reverse.items()}
        for name in (CONTROL, *METHODS):
            np.testing.assert_allclose(auc[name], rev_report['folds'][i - 1]['variants'][name]['roc_auc'],
                                       atol=1e-12, rtol=0)
        fold_records.append({'fold': i, 'pairs': int(yy.sum()) * int((yy == 0).sum()), 'auc': auc, 'loss': loss})

    dev_summary = paired_summary(ns, labels[held], dev_scores)
    rev_summary = paired_summary(ns, labels[development], reverse, fold_records)
    for description, value in (('holdout', dev_summary), ('reverse', rev_summary)):
        print(description, 'matched shrink', value[SHRUNK], flush=True)
    direct = {
        'holdout_subforest_vs_shrink_delta_auc': dev_summary[METHODS[0]]['roc_auc'] - dev_summary[SHRUNK]['roc_auc'],
        'reverse_subforest_vs_shrink_delta_weighted_auc': (
            rev_summary[METHODS[0]]['fold_weighted_delta_auc'] - rev_summary[SHRUNK]['fold_weighted_delta_auc']),
        'reverse_subforest_vs_shrink_fold_auc_wins': sum(
            item['auc'][METHODS[0]] > item['auc'][SHRUNK] for item in fold_records),
        'holdout_subforest_auc_contribution_by_original_score_band': auc_by_base_score_band(
            labels[held], dev_scores[CONTROL], dev_scores[METHODS[0]]
        ),
    }
    np.testing.assert_allclose(
        sum(row['contribution_to_auc_delta'] for row in direct['holdout_subforest_auc_contribution_by_original_score_band']),
        dev_summary[METHODS[0]]['delta_auc'], atol=1e-12, rtol=0)
    print(direct, flush=True)
    write_json(HERE / 'matched_shrink_audit.json', {'input_sha256': hashes, 'holdout': dev_summary,
               'reverse': rev_summary, 'folds': fold_records, **direct})


if __name__ == '__main__':
    run()
