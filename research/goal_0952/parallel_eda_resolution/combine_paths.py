"""Average three prespecified, conservative Grandmaster EDA adjustments."""

import argparse
from contextlib import ExitStack

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import log_loss, roc_auc_score

from resolution_eda import CONTROL, HERE, PARENT
from grandmaster_inputs import archive_functions, development_caches, digest, official_data, read_json, write_json
from robust_density import compare

NAME = 'parallel_three_way'
BASE_DIRS = (HERE, PARENT / 'parallel_peer_context', PARENT / 'parallel_tree_calibration')
RESOLUTION_FILES = ('development_predictions.npz', 'reverse_predictions.npz', 'full_predictions.npz')
STUMP_FILES = ('holdout_predictions.npz', 'reverse_predictions.npz', 'test_predictions.npz')
PEER_FILES = ('holdout_predictions.npz', 'reverse_predictions.npz', 'test_predictions.npz')
CANDIDATES = ('resolution_linear', 'joint_context_half', 'additive_stumps')


def collect(paths, candidates, ids, phased_folds=None):
    with ExitStack() as stack:
        archives = [stack.enter_context(np.load(f, allow_pickle=False)) for f in paths]
        base = archives[0][CONTROL]
        deltas = []
        for archive, path, method in zip(archives, paths, candidates, strict=True):
            np.testing.assert_array_equal(archive['ids'], ids, err_msg=str(path))
            np.testing.assert_allclose(archive[CONTROL], base, rtol=0,
                                       atol=0 if phased_folds is None else 4e-6, err_msg=str(path))
            if phased_folds is not None:
                np.testing.assert_array_equal(archive['fold'], phased_folds, err_msg=str(path))
            delta = archive[method] - archive[CONTROL]
            if not np.isfinite(delta).all():
                raise ValueError(f'Invalid adjustments in {path}')
            deltas.append(delta)
    return base, base + np.mean(deltas, axis=0)


def run():
    train, test, sample, hashes = official_data()
    y, dev, held, folds, *_ = development_caches(train, hashes)
    namespace = archive_functions()
    train_ids, test_ids = train.id.to_numpy(), test.id.to_numpy()
    fold_ids = np.empty(len(dev), dtype=np.int8)
    for i, (_, fold) in enumerate(folds, start=1):
        fold_ids[fold] = i
    records = {'input_sha256': hashes, 'weight_per_candidate_on_logit_delta': 1/3, 'phases': {}}
    aligned = {}
    for i, (phase, ids) in enumerate(zip(('holdout', 'reverse', 'test'),
                                        (train_ids[held], train_ids[dev], test_ids), strict=True)):
        names = (RESOLUTION_FILES[i], PEER_FILES[i], STUMP_FILES[i])
        paths = [directory / filename for directory, filename in zip(BASE_DIRS, names, strict=True)]
        run_folds = fold_ids if phase == 'reverse' else None
        base, scores = collect(paths, CANDIDATES, ids, run_folds)
        aligned[phase] = (base, scores)
        entry = {'rows': len(ids), 'sources': [
            {'file': str(path.relative_to(PARENT)), 'column': candidate, 'sha256': digest(path)}
            for path, candidate in zip(paths, CANDIDATES, strict=True)]}
        if phase != 'test':
            rows = dev if phase == 'reverse' else held
            entry['variants'] = compare(namespace, y[rows], {CONTROL: base, NAME: scores}, CONTROL)
            if phase == 'reverse':
                by_fold = []
                for i_fold, (_, idx) in enumerate(folds, start=1):
                    labels = y[dev[idx]]
                    a, b = base[idx], scores[idx]
                    by_fold.append({'fold': i_fold, 'rows': len(idx), 'pairs': int(labels.sum()) * int((labels == 0).sum()),
                                    'delta_auc': float(roc_auc_score(labels, b) - roc_auc_score(labels, a)),
                                    'delta_logloss': float(log_loss(labels, expit(b)) - log_loss(labels, expit(a)))})
                entry['folds'] = by_fold
                summary = entry['variants'][NAME]
                summary['fold_weighted_delta_auc'] = float(np.average([r['delta_auc'] for r in by_fold],
                                                                     weights=[r['pairs'] for r in by_fold]))
                summary['fold_auc_wins'] = sum(r['delta_auc'] > 0 for r in by_fold)
                summary['fold_logloss_wins'] = sum(r['delta_logloss'] < 0 for r in by_fold)
            print(phase, entry['variants'][NAME])
        records['phases'][phase] = entry

    official_submission = PARENT / 'submission_grandmaster_eda.csv'
    original_sha = read_json(PARENT / 'evaluation.json')['submission_sha256']
    if digest(official_submission) != original_sha:
        raise ValueError('Original Grandmaster EDA CSV failed manifest verification')
    original = pd.read_csv(official_submission)
    np.testing.assert_array_equal(original.id, test_ids)
    original_logits, logits = aligned['test']
    np.testing.assert_allclose(original.is_fraud, expit(original_logits), rtol=0, atol=1e-14)
    output = HERE / f'submission_grandmaster_eda_{NAME}.csv'
    temporary = output.with_suffix('.tmp')
    probabilities = expit(logits)
    sample.assign(is_fraud=probabilities).to_csv(temporary, index=False)
    check = pd.read_csv(temporary)
    assert check.id.is_unique and check.id.equals(sample.id) and list(check.columns) == list(sample.columns)
    assert check.is_fraud.between(0., 1.).all()
    np.testing.assert_allclose(check.is_fraud, probabilities, rtol=0, atol=1e-14)
    temporary.replace(output)
    records['phases']['test'].update(filename=output.name, sha256=digest(output), parent_sha256=original_sha)
    np.savez_compressed(HERE / f'{NAME}_predictions.npz', holdout_ids=train_ids[held], holdout=aligned['holdout'][1],
                        reverse_ids=train_ids[dev], reverse=aligned['reverse'][1], fold=fold_ids,
                        test_ids=test_ids, test_scores=logits)
    write_json(HERE / f'{NAME}_metrics.json', records)
    print(f'Verified {len(check):,} rows in {output.name}')


def verify():
    train, test, sample, hashes = official_data()
    y, dev, held, folds, *_ = development_caches(train, hashes)
    manifest = read_json(HERE / f'{NAME}_metrics.json')
    assert manifest['input_sha256'] == hashes
    records = manifest['phases']
    output = HERE / records['test']['filename']
    assert digest(output) == records['test']['sha256']
    table = pd.read_csv(output)
    assert table.id.is_unique and table.id.equals(sample.id) and list(table.columns) == list(sample.columns)
    assert table.is_fraud.between(0., 1.).all()
    with np.load(HERE / f'{NAME}_predictions.npz', allow_pickle=False) as saved:
        for phase, ids, rows in (('holdout', train.id.to_numpy()[held], held),
                                 ('reverse', train.id.to_numpy()[dev], dev), ('test', test.id.to_numpy(), None)):
            record = records[phase]
            assert len(ids) == record['rows']
            paths = [PARENT / source['file'] for source in record['sources']]
            for path, source in zip(paths, record['sources'], strict=True):
                assert digest(path) == source['sha256']
            _, expected = collect(paths, CANDIDATES, ids, saved['fold'] if phase == 'reverse' else None)
            np.testing.assert_array_equal(saved[f'{phase}_ids'], ids)
            np.testing.assert_array_equal(saved[phase if rows is not None else 'test_scores'], expected)
            if rows is not None:
                result = record['variants'][NAME]
                np.testing.assert_allclose(roc_auc_score(y[rows], expected), result['roc_auc'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(log_loss(y[rows], expit(expected)), result['log_loss'], atol=1e-12, rtol=0)
                if phase == 'reverse':
                    for i, (_, idx) in enumerate(folds, start=1):
                        np.testing.assert_array_equal(saved['fold'][idx], i)
            else:
                np.testing.assert_allclose(table.is_fraud, expit(expected), atol=1e-14, rtol=0)
    print(f'Checked source hashes, score formula, both validation protocols, and {len(table):,} CSV probabilities')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verify() if args.verify_only else run()
