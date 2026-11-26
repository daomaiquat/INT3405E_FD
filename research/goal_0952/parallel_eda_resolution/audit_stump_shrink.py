"""Check whether the stump blend gains more than a matched shrinkage control."""

import argparse

import numpy as np
import pandas as pd
from scipy.special import expit

from resolution_eda import CONTROL, HERE, PARENT, XGB_WEIGHT
from evaluate_and_export import assess, read_scores
from grandmaster_inputs import (ABLATION, archive_functions, design, development_caches, digest, official_data,
                                read_json, write_json)
from residual_models import auc_metrics
from robust_density import compare, isolated_meta

SHRINK = 'shrink_toward_grandmaster'
STUMPS = 'additive_stumps'
COMPARISON = PARENT / 'parallel_tree_calibration'


def audit():
    train, _, _, hashes = official_data()
    y, dev, held, folds, raw, _, dense, _, _ = development_caches(train, hashes)
    ids = train.id.to_numpy()
    ns = archive_functions()
    manifest = read_json(COMPARISON / 'holdout_metrics.json')
    if manifest['input_sha256'] != hashes:
        raise ValueError('Stump control data differs from the official hashes')
    weight = manifest['replacement_of_xgb_log_odds']

    baseline = read_scores(COMPARISON / 'holdout_predictions.npz', ids[held], 'ids', (CONTROL, STUMPS))
    old = read_scores(PARENT / 'residual_development.npz', ids[held], 'ids', ('grandmaster_hierarchical', 'offset_xgb_224'))
    baseline[SHRINK] = baseline[CONTROL] + XGB_WEIGHT * weight * (old['grandmaster_hierarchical'] - old['offset_xgb_224'])
    holdout = compare(ns, y[held], baseline, CONTROL)
    holdout_stumps_vs_shrink = assess(ns, y[held], baseline[STUMPS], baseline[SHRINK])

    reverse = read_scores(COMPARISON / 'reverse_predictions.npz', ids[dev], 'ids', (CONTROL, STUMPS, 'fold'))
    fold_ids = reverse.pop('fold')
    reverse[SHRINK] = reverse[CONTROL].copy()
    entries = []
    for i, (_, query) in enumerate(folds, start=1):
        path = ABLATION / 'validation' / 'artifacts' / f'fold_{i:02}.npz'
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['va_idx'], query)
            src = design(ns, saved['baseline_hold'], saved['candidate_hold'][:, 1])
        np.testing.assert_array_equal(fold_ids[query], i)
        _, q = isolated_meta(ns, src, design(ns, raw[query], dense[query]), y[held])
        reverse[SHRINK][query] += weight * (q - reverse[CONTROL][query])
        labels = y[dev[query]]
        entries.append({'fold': i, 'pairs': int(labels.sum()) * int((labels == 0).sum()),
                        'variants': {key: auc_metrics(labels, z[query]) for key, z in reverse.items()}})
    ratings = compare(ns, y[dev], reverse, CONTROL)
    weights = [item['pairs'] for item in entries]
    for name, row in ratings.items():
        auc_delta = [entry['variants'][name]['roc_auc'] - entry['variants'][CONTROL]['roc_auc'] for entry in entries]
        row['fold_weighted_delta_auc'] = float(np.average(auc_delta, weights=weights))
        row['fold_auc_wins'] = sum(v > 0 for v in auc_delta)
        row['fold_logloss_wins'] = sum(entry['variants'][name]['log_loss'] < entry['variants'][CONTROL]['log_loss']
                                      for entry in entries)
    reverse_stumps_vs_shrink = assess(ns, y[dev], reverse[STUMPS], reverse[SHRINK])
    deltas = [entry['variants'][STUMPS]['roc_auc'] - entry['variants'][SHRINK]['roc_auc'] for entry in entries]
    reverse_stumps_vs_shrink['fold_weighted_delta_auc'] = float(np.average(deltas, weights=weights))
    reverse_stumps_vs_shrink['fold_auc_wins'] = sum(v > 0 for v in deltas)
    write_json(HERE / 'stump_shrink_audit.json', {
        'input_sha256': hashes, 'replacement_weight': weight, 'holdout': holdout, 'reverse': ratings,
        'holdout_stumps_vs_shrink': holdout_stumps_vs_shrink, 'reverse_stumps_vs_shrink': reverse_stumps_vs_shrink,
        'folds': entries})
    print('Holdout shrink control:', holdout[SHRINK])
    print('Holdout stumps vs shrink:', holdout_stumps_vs_shrink)
    print('Reverse shrink control:', ratings[SHRINK])
    print('Reverse stumps vs shrink:', reverse_stumps_vs_shrink)


def export():
    _, test, sample, hashes = official_data()
    ids = test.id.to_numpy()
    parent_path = PARENT / 'submission_grandmaster_eda.csv'
    parent_sha = read_json(PARENT / 'evaluation.json')['submission_sha256']
    assert digest(parent_path) == parent_sha
    parent = pd.read_csv(parent_path)
    manifest = read_json(HERE / 'stump_shrink_audit.json')
    assert manifest['input_sha256'] == hashes
    residual_meta = read_json(PARENT / 'residual_full_metrics.json')
    assert residual_meta['input_sha256'] == hashes
    weight = manifest['replacement_weight']
    anchor = read_scores(COMPARISON / 'test_predictions.npz', ids, 'ids', (CONTROL,))[CONTROL]
    offsets = read_scores(PARENT / 'residual_full.npz', ids, 'ids', ('grandmaster_hierarchical', 'offset_xgb_224'))
    np.testing.assert_array_equal(parent.id, ids)
    np.testing.assert_allclose(expit(anchor), parent.is_fraud, atol=1e-14, rtol=0)
    z = anchor + XGB_WEIGHT * weight * (offsets['grandmaster_hierarchical'] - offsets['offset_xgb_224'])
    p = expit(z)
    target = HERE / f'submission_grandmaster_eda_{SHRINK}.csv'
    temp = target.with_suffix('.tmp')
    sample.assign(is_fraud=p).to_csv(temp, index=False)
    checked = pd.read_csv(temp)
    assert checked.id.is_unique and checked.id.equals(sample.id) and list(checked.columns) == list(sample.columns)
    assert checked.is_fraud.between(0., 1.).all()
    np.testing.assert_allclose(checked.is_fraud, p, atol=1e-14, rtol=0)
    temp.replace(target)
    write_json(HERE / 'stump_shrink_manifest.json', {'filename': target.name, 'rows': len(checked),
               'input_sha256': hashes, 'sha256': digest(target), 'parent_sha256': parent_sha,
               'replacement_weight': weight})
    print(f'{target.name}: verified {len(checked):,} rows and probabilities')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--export', action='store_true')
    export() if cli.parse_args().export else audit()
