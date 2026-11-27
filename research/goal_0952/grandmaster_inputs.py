"""Read official competition data and verify locally trained Grandmaster caches."""

from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA_DIR = PROJECT.parent / 'data'
CACHE_PROJECT = Path(os.environ.get('GRANDMASTER_CACHE_PROJECT', PROJECT)).resolve()
GOAL_CACHE = CACHE_PROJECT / 'goal_0952'
ABLATION_CODE = PROJECT / 'grandmaster_10fold_ablation'
ABLATION = CACHE_PROJECT / 'grandmaster_10fold_ablation'
FULL = CACHE_PROJECT / 'grandmaster_refit' / 'full'
HIERARCHY = ABLATION / 'submission_hierarchical_amount'
TARGET = 'is_fraud'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    with path.open('rb') as file:
        return file_digest(file, 'sha256').hexdigest()


def official_data():
    manifest = read_json(DATA_DIR / 'manifest.json')
    hashes = {name: digest(DATA_DIR / name) for name in manifest['input_files']}
    if any(hashes[name] != entry['sha256'] for name, entry in manifest['input_files'].items()):
        raise ValueError('Official competition data checksums do not match')
    train, test, sample = (pd.read_csv(DATA_DIR / name) for name in ('train.csv', 'test.csv', 'sample_submission.csv'))
    if (len(train) != manifest['train_rows'] or len(test) != manifest['test_rows']
            or not train.id.is_unique or not test.id.is_unique or np.intersect1d(train.id, test.id).size
            or list(sample.columns) != ['id', TARGET] or not sample.id.equals(test.id)):
        raise ValueError('Official competition data IDs, shape or schema do not match')
    return train, test, sample, hashes


def archive_functions():
    sys.path.insert(0, str(ABLATION_CODE))
    from run_ablation import load_archive

    namespace, _ = load_archive(HERE / 'notebook_runtime', False)
    return namespace


def splits_for(labels):
    return list(StratifiedKFold(10, shuffle=True, random_state=2026).split(labels, labels))


def development_caches(train, hashes):
    metadata = read_json(ABLATION / 'validation' / 'metrics.json')
    if metadata['train_sha256'] != hashes['train.csv']:
        raise ValueError('Development scores have a different training source')
    y = train[TARGET].to_numpy(dtype=np.uint8)
    split = StratifiedShuffleSplit(1, test_size=metadata['holdout_fraction'], random_state=metadata['holdout_seed'])
    dev, held = (np.sort(rows) for rows in next(split.split(y, y)))
    if sha256(train.id.to_numpy()[held].tobytes()).hexdigest() != metadata['holdout_ids_sha256']:
        raise ValueError('Development holdout identity does not match metadata')
    folds = splits_for(y[dev])
    raw_dev, raw_held = np.empty((len(dev), 7)), np.zeros((len(held), 7))
    dense_dev, dense_held = np.empty(len(dev)), np.zeros(len(held))
    signatures = set()
    for i, (_, valid) in enumerate(folds, start=1):
        with np.load(ABLATION / 'validation' / 'artifacts' / f'fold_{i:02}.npz', allow_pickle=False) as cache:
            signatures.add(cache['signature'].item())
            np.testing.assert_array_equal(cache['va_idx'], valid)
            raw_dev[valid] = cache['baseline_va']
            raw_held += cache['baseline_hold'] / len(folds)
            dense_dev[valid] = cache['candidate_va'][:, 1]
            dense_held += cache['candidate_hold'][:, 1] / len(folds)
    if len(signatures) != 1 or not all(np.isfinite(a).all() for a in (raw_dev, raw_held, dense_dev, dense_held)):
        raise ValueError('Development prediction bank is inconsistent')
    return y, dev, held, folds, raw_dev, raw_held, dense_dev, dense_held, metadata


def full_caches(train, test, hashes):
    old = read_json(FULL / 'metrics.json')
    newer = read_json(HIERARCHY / 'manifest.json')
    for manifest in (old, newer):
        if manifest['train_sha256'] != hashes['train.csv'] or manifest['test_sha256'] != hashes['test.csv']:
            raise ValueError('Grandmaster prediction bank came from different input data')
    labels = train[TARGET].to_numpy(dtype=np.uint8)
    folds = splits_for(labels)
    raw_oof, raw_test = np.empty((len(train), 7)), np.zeros((len(test), 7))
    dense_oof, dense_test = np.empty(len(train)), np.zeros(len(test))
    signatures = set()
    for i, (fit, valid) in enumerate(folds, start=1):
        name = f'fold_{i:02}.npz'
        path = FULL / 'artifacts' / name
        if digest(path) != newer['local_10fold_cache_sha256'][name]:
            raise ValueError(f'Checkpoint SHA-256 mismatch: {name}')
        with np.load(path, allow_pickle=False) as cache:
            signatures.add(cache['signature'].item())
            np.testing.assert_array_equal(cache['held_rows'], valid)
            np.testing.assert_array_equal(cache['fit_rows'], fit)
            raw_oof[valid] = cache['held']
            raw_test += cache['test'] / len(folds)
        with np.load(HIERARCHY / 'artifacts' / f'dense_{name}', allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache['held_rows'], valid)
            dense_oof[valid] = cache['held']
            dense_test += cache['test'] / len(folds)
    if len(signatures) != 1 or not all(np.isfinite(a).all() for a in (raw_oof, raw_test, dense_oof, dense_test)):
        raise ValueError('Grandmaster prediction bank is inconsistent')
    return folds, raw_oof, raw_test, dense_oof, dense_test, old


def design(namespace, raw, hierarchical):
    frame, columns = namespace['build_resdecomp'](dict(zip(namespace['LOCKED_MODELS'], raw.T, strict=True)))
    if hierarchical is not None:
        frame[:, columns.index('02f_DeGLM')] = hierarchical
    return frame


def meta_predictions(namespace, y, fit_x, score_x, folds, penalties):
    """Fit meta models on development rows and average scoring logits."""
    oof = np.full(len(y), np.nan)
    score = np.zeros(len(score_x))
    weight_values = []
    for (fit, valid), penalty in zip(folds, penalties, strict=True):
        intercept, weights = namespace['_fit_nonneg_l2_weights'](fit_x[fit], y[fit], penalty)
        oof[valid] = intercept + fit_x[valid] @ weights
        score += (intercept + score_x @ weights) / len(folds)
        weight_values.append(weights)
    np.testing.assert_equal(np.isfinite(oof).all(), True)
    return oof, score, np.mean(weight_values, axis=0)
