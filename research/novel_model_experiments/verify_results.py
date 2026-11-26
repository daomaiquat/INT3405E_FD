"""Verify S6E11 CSVs and calculate paired model comparisons on official labels."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
MODEL_NAMES = ('neighborhood', 'copula', 'occupancy', 'spline_interactions',
               'fixed_blend', 'hours_residual', 'spline_blend')


def sha256(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def array_matches(current, expected, name):
    if not np.array_equal(current, expected):
        raise ValueError(f'{name} does not match official rows or the frozen split')


def probabilities(values, size, name):
    if values.shape != (size,) or not (np.isfinite(values) & (values >= 0) & (values <= 1)).all():
        raise ValueError(f'{name} has invalid probability values or row count')


def placements(scores, labels):
    positive, negative = scores[labels == 1], scores[labels == 0]

    def count_lower(values, reference):
        ordered = np.sort(reference)
        return 0.5 * (np.searchsorted(ordered, values, side='left')
                      + np.searchsorted(ordered, values, side='right')) / len(ordered)

    return count_lower(positive, negative), 1 - count_lower(negative, positive)


def compare(labels, folds, scores, baseline):
    diffs = [float(roc_auc_score(labels[folds == i], scores[folds == i])
                   - roc_auc_score(labels[folds == i], baseline[folds == i])) for i in range(5)]
    treatment = placements(scores, labels)
    control = placements(baseline, labels)
    paired = [a - b for a, b in zip(treatment, control, strict=True)]
    delta = float(paired[0].mean())
    margin = 1.959963984540054 * np.sqrt(sum(x.var(ddof=1) / len(x) for x in paired))
    return {
        'delta_auc': delta, 'fold_delta_auc': diffs,
        'folds_improved': sum(d > 0 for d in diffs),
        'paired_delong_ci95_conditional_on_predictions': [delta - margin, delta + margin],
    }


def read_scores(path, train_ids, test_ids, folds):
    with np.load(path, allow_pickle=False) as pred:
        id_key = 'ids' if 'ids' in pred else 'train_ids'
        array_matches(pred[id_key], train_ids, f'{path.name} train IDs')
        array_matches(pred['test_ids'], test_ids, f'{path.name} test IDs')
        if 'fold' in pred:
            array_matches(pred['fold'], folds, f'{path.name} folds')
        else:
            # The spline runner had stored fold assignments in its checkpoints.
            spline_oof = pred['oof']
            spline_baseline = pred['baseline_oof']
            run_id = json.loads(path.with_name('spline_interactions_metrics.json').read_text(
                encoding='utf-8'))['source_and_inputs_digest'][:12]
            for i in range(5):
                checkpoint = path.parent / '.spline_checkpoints' / run_id / f'fold_{i + 1}.npz'
                with np.load(checkpoint, allow_pickle=False) as fold:
                    rows = fold['valid']
                    array_matches(rows, np.flatnonzero(folds == i), checkpoint.name)
                    array_matches(spline_oof[rows], fold['oof'], f'{checkpoint.name} spline predictions')
                    array_matches(spline_baseline[rows], fold['baseline'], f'{checkpoint.name} additive predictions')
        oof, test = pred['oof'], pred['test']
        additive = pred['baseline_oof'] if 'baseline_oof' in pred else None
    probabilities(oof, len(train_ids), f'{path.name} OOF')
    probabilities(test, len(test_ids), f'{path.name} test')
    if additive is not None:
        probabilities(additive, len(train_ids), f'{path.name} additive OOF')
    return oof, test, additive


def locate(name):
    in_results = HERE / 'results' / name
    return in_results if in_results.exists() else HERE / name


def score(labels, predictions):
    return {
        'roc_auc': roc_auc_score(labels, predictions),
        'average_precision': average_precision_score(labels, predictions),
        'log_loss': log_loss(labels, predictions),
    }


def verify(names):
    manifest = json.loads((PROJECT.parent / 'data' / 'manifest.json').read_text(encoding='utf-8-sig'))
    official_hashes = {name: sha256(PROJECT.parent / "data" / name) for name in manifest['input_files']}
    if any(actual != manifest['input_files'][name]['sha256'] for name, actual in official_hashes.items()):
        raise ValueError('Competition CSV checksums differ from the verified local manifest')

    train = pd.read_csv(PROJECT.parent / 'data' / 'train.csv', usecols=['id', 'is_fraud'])
    sample = pd.read_csv(PROJECT.parent / 'data' / 'sample_submission.csv', usecols=['id', 'is_fraud'])
    test_ids = pd.read_csv(PROJECT.parent / 'data' / 'test.csv', usecols=['id']).id.to_numpy()
    array_matches(sample.id.to_numpy(), test_ids, 'sample IDs')
    labels, train_ids = train.is_fraud.to_numpy(dtype=np.uint8), train.id.to_numpy()
    if (len(np.unique(test_ids)) != manifest['test_rows'] or len(test_ids) != manifest['test_rows']
            or len(train_ids) != manifest['train_rows'] or len(np.unique(train_ids)) != len(train_ids)):
        raise ValueError('Official IDs are duplicated or row counts differ from the manifest')
    folds = np.empty(len(train), dtype=np.int8)
    for index, (_, hold) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train_ids, labels)):
        folds[hold] = index

    historical = PROJECT / 's6e11_gpu' / 'plan_runs' / 'results' / 'full'
    final, _, _ = read_scores(historical / 'final' / 'predictions.npz', train_ids, test_ids, folds)
    control_path = locate('control_predictions.npz')
    control = read_scores(control_path, train_ids, test_ids, folds)[0] if control_path.is_file() else None
    report = {'input_sha256': official_hashes, 'rows_train': len(train_ids), 'rows_test': len(test_ids),
              'fold_sizes': np.bincount(folds).tolist(), 'seed': 42, 'FINAL': score(labels, final), 'models': {}}
    if control is not None:
        report['xgboost_matched_control'] = score(labels, control)

    for name in names:
        pred_path = locate(f'{name}_predictions.npz')
        oof, test, additive = read_scores(pred_path, train_ids, test_ids, folds)
        csv_path = locate(f'submission_{name}.csv')
        submission = pd.read_csv(csv_path)
        if list(submission.columns) != ['id', 'is_fraud'] or len(submission) != len(test_ids):
            raise ValueError(f'{csv_path.name} has an invalid header or shape')
        array_matches(submission.id.to_numpy(), test_ids, f'{csv_path.name} IDs')
        csv_scores = submission.is_fraud.to_numpy()
        probabilities(csv_scores, len(test_ids), csv_path.name)
        max_diff = float(np.abs(csv_scores - test).max())
        # The spline CSV emits shortest round-trip decimals for float32 values.
        agrees = (np.array_equal(csv_scores.astype(test.dtype), test)
                  if test.dtype == np.float32 else max_diff <= 1e-12)
        if not agrees:
            raise ValueError(f'{csv_path.name} does not match its prediction array')

        result = {**score(labels, oof), 'vs_FINAL': compare(labels, folds, oof, final),
                  'submission_path': csv_path.relative_to(HERE).as_posix(), 'submission_sha256': sha256(csv_path),
                  'submission_rows': len(test), 'submission_min': float(test.min()),
                  'submission_max': float(test.max()), 'submission_mean': float(test.mean()),
                  'prediction_array_dtype': str(test.dtype),
                  'max_csv_vs_npz_abs_difference': max_diff}
        if name in {'neighborhood', 'copula'} and control is not None:
            result['vs_matched_xgboost'] = compare(labels, folds, oof, control)
        if additive is not None:
            result['additive_spline_control'] = score(labels, additive)
            result['vs_additive_spline'] = compare(labels, folds, oof, additive)
        report['models'][name] = result
        print(f'{name:23s} AUC {result["roc_auc"]:.8f}  vs FINAL {result["vs_FINAL"]["delta_auc"]:+.8f}  '
              f'{len(test):,} submission rows', flush=True)

    (HERE / 'results' / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=MODEL_NAMES)
    verify(parser.parse_args().models)
