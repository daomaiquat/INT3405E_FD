"""Reproduce the three seed-42 follow-up submissions from their prediction banks."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from no_flag_specialist import FLAG_COLUMNS, fit_adjust_no_flags

REPO = Path(__file__).resolve().parents[3]
SCHEMA = ['id', 'is_fraud']
FOLD_HASH = '8947ed79a30033f822bb026ad6b556a180dbc649ab945d8587560473b044922a'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_probabilities(values, count):
    if values.shape != (count,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Expected one finite probability in [0, 1] per ID')


def checked_predictions(sources, full, hashes, train_ids, test_ids):
    scores = {}
    for name, source in sources.items():
        meta = json.loads((source / 'metrics.json').read_text(encoding='utf-8'))
        if (meta['name'] != name or meta['smoke'] or meta['models_use_llm']
                or meta['inputs'] != hashes or meta['fold_hash'] != FOLD_HASH
                or meta['rows'] != len(train_ids) or meta['test_rows'] != len(test_ids)):
            raise ValueError(f'Provenance mismatch for {name}')
        with np.load(source / 'predictions.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['ids'], train_ids)
            np.testing.assert_array_equal(archive['test_ids'], test_ids)
            scores[name] = archive['test']
            check_probabilities(scores[name], len(test_ids))
            if name == 'FINAL':
                final_oof, fold_ids = archive['oof'], archive['fold']
                check_probabilities(final_oof, len(train_ids))
            else:
                np.testing.assert_array_equal(archive['test_by_fold'].mean(axis=1), scores[name])
                np.testing.assert_array_equal(archive['fold'], fold_ids)

    selection = json.loads((sources['FINAL'] / 'selection.json').read_text(encoding='utf-8'))
    if selection['nested']['members'] != ['GLM-NEWTON', 'XGB-D03']:
        raise ValueError('Unexpected FINAL ensemble members')
    with np.load(full / 'nested' / 'predictions.npz', allow_pickle=False) as nested:
        np.testing.assert_array_equal(nested['ids'], train_ids)
        np.testing.assert_array_equal(nested['fold'], fold_ids)
        np.testing.assert_array_equal(nested['oof'], final_oof)
    baseline_csv = sources['FINAL'] / 'submission.csv'
    if digest(baseline_csv) != selection['submission']['sha256']:
        raise ValueError('Stored FINAL baseline CSV hash does not match selection')
    baseline = pd.read_csv(baseline_csv, float_precision='round_trip')
    np.testing.assert_array_equal(baseline.id, test_ids)
    np.testing.assert_allclose(baseline.is_fraud, scores['FINAL'], rtol=0, atol=2e-16)
    return scores, final_oof


def verify_csv(path, test_ids, expected):
    frame = pd.read_csv(path, float_precision='round_trip')
    if list(frame.columns) != SCHEMA or not frame.id.is_unique:
        raise ValueError(f'Invalid exported schema in {path}')
    np.testing.assert_array_equal(frame.id, test_ids)
    check_probabilities(frame.is_fraud.to_numpy(), len(test_ids))
    np.testing.assert_array_equal(frame.is_fraud, expected)


def run(data, gpu, out):
    feature_columns = ['id', *FLAG_COLUMNS, 'merchant_category', 'merchant_risk_score']
    train = pd.read_csv(data / 'train.csv', usecols=[*feature_columns, 'is_fraud'])
    test = pd.read_csv(data / 'test.csv', usecols=feature_columns)
    sample = pd.read_csv(data / 'sample_submission.csv')
    train_ids, test_ids = train.id.to_numpy(), test.id.to_numpy()
    if (list(sample.columns) != SCHEMA or len(test_ids) != 299_984 or not test.id.equals(sample.id)
            or not train.id.is_unique or not test.id.is_unique or np.intersect1d(train_ids, test_ids).size):
        raise ValueError('Official test and sample submission ID contract failed')

    full = gpu / 'plan_runs' / 'results' / 'full'
    sources = {'FINAL': full / 'final', 'D03': full / 'D03',
               'MLP-BCE': gpu / 'imbalance' / 'results' / 'mlp' / 'MLP-BCE'}
    hashes = {name: digest(data / name) for name in ('train.csv', 'test.csv', 'sample_submission.csv')}
    scores, final_oof = checked_predictions(sources, full, hashes, train_ids, test_ids)
    base = scores['FINAL']
    quiet = test.loc[:, FLAG_COLUMNS].eq(0).all(axis=1).to_numpy()
    adjusted = fit_adjust_no_flags(train, final_oof, test, base)
    np.testing.assert_array_equal(adjusted[~quiet], base[~quiet])
    if not (adjusted[quiet] != base[quiet]).all():
        raise ValueError('Some rows eligible for no-flags correction were not adjusted')

    exports = {'01_FINAL_LGBM_D03_20pct_submission.csv': .8 * base + .2 * scores['D03'],
               '02_FINAL_MLP_BCE_10pct_submission.csv': .9 * base + .1 * scores['MLP-BCE'],
               '03_FINAL_NO_FLAGS_submission.csv': adjusted}
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.staging-', dir=out) as temp:
        stage = Path(temp)
        for name, values in exports.items():
            check_probabilities(values, len(test_ids))
            if (out / name).exists():
                verify_csv(out / name, test_ids, values)
            else:
                pd.DataFrame({'id': test_ids, 'is_fraud': values}).to_csv(
                    stage / name, index=False, float_format='%.17g', lineterminator='\n')
                verify_csv(stage / name, test_ids, values)
        for path in stage.iterdir():
            path.rename(out / path.name)
    for name, values in exports.items():
        verify_csv(out / name, test_ids, values)
        print(f'{name}: {len(values):,} rows, sha256={digest(out / name)}')
    print(f"No-flags correction changed {np.count_nonzero(adjusted != base):,} rows.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=REPO / 'data')
    parser.add_argument('--gpu-results', type=Path, default=REPO / 's6e11_gpu')
    parser.add_argument('--output', type=Path, default=REPO / 'outputs' / 'followup')
    args = parser.parse_args()
    run(args.data_dir, args.gpu_results, args.output)
