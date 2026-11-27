"""Verify saved probabilities and inspect the 404-amount group across folds."""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from calibrate import AMOUNT, AMOUNT_RECIPE, CONTROL, HERE, RECIPES, SOURCE, blend, final_bank, load_oof, ticks
from evaluate_and_export import auc_by_base_score_band, read_scores
from grandmaster_inputs import archive_functions, development_caches, digest, official_data, read_json, write_json
from robust_density import compare


def amount_group(values, levels):
    key, valid = ticks(values.to_numpy())
    return valid & np.isin(key, levels)


def measure_group(mask, y, before, after):
    probability = expit(before[mask])
    target = y[mask]
    return {'rows': int(mask.sum()), 'fraud_count': int(target.sum()), 'expected': float(probability.sum()),
            'delta_log_loss': float(log_loss(target, expit(after[mask])) - log_loss(target, probability))}


def verify_metrics(y, z, expected):
    p = expit(z)
    measured = {'roc_auc': roc_auc_score(y, z), 'average_precision': average_precision_score(y, z),
                'brier': brier_score_loss(y, p), 'log_loss': log_loss(y, p)}
    for name, value in measured.items():
        np.testing.assert_allclose(value, expected[name], rtol=0, atol=1e-12)


def run(mode):
    train, test, sample, hashes = official_data()
    y, dev, held, *_ = development_caches(train, hashes)
    development = read_json(HERE / 'development_metrics.json')
    declared = read_json(HERE / f'{mode}_metrics.json')
    assert development['input_sha256'] == declared['input_sha256'] == hashes
    assert len(test) == declared['scored_rows'] == 299_984
    assert digest(SOURCE / 'submission_grandmaster_eda.csv') == declared['parent_sha256']
    holdout = read_scores(HERE / 'development_predictions.npz', train.id.to_numpy()[held], 'ids',
                          (CONTROL, *RECIPES, 'frozen_amount_hole'))
    for name, scores in holdout.items():
        verify_metrics(y[held], scores, development['holdout'][name])
    predictions = read_scores(HERE / f'{mode}_predictions.npz', test.id.to_numpy(), 'ids', (CONTROL, *RECIPES))
    _, original = final_bank('full', test.id.to_numpy())
    np.testing.assert_array_equal(predictions[CONTROL], original)
    levels = read_json(AMOUNT_RECIPE)['frozen_amount_ticks']
    mask = amount_group(test[AMOUNT], levels)
    assert mask.sum() == 4448
    np.testing.assert_array_equal(predictions['refit_amount_hole'][~mask], original[~mask])
    test_movement = {}
    for name, info in declared['files'].items():
        path = HERE / info['file']
        assert info['sha256'] == digest(path)
        table = pd.read_csv(path)
        assert list(table) == list(sample) and table.id.is_unique and table.id.equals(sample.id)
        assert len(table) == info['rows'] and table.is_fraud.between(0, 1).all()
        np.testing.assert_allclose(table.is_fraud, expit(predictions[name]), rtol=0, atol=1e-14)
        delta = predictions[name] - original
        assert np.max(np.abs(delta)) <= RECIPES[name].cap + 1e-12
        test_movement[name] = {
            'nonzero_rows': int(np.count_nonzero(delta)),
            'max_abs_logit_shift': float(np.abs(delta).max()),
            'rms_logit_shift': float(np.sqrt(np.mean(delta**2))),
        }
        print(f'{path.name}: {len(table):,} rows verified')

    held_group = amount_group(train.iloc[held][AMOUNT], levels)
    dev_group = amount_group(train.iloc[dev][AMOUNT], levels)
    branch_scores, folds = load_oof('development', train.id.to_numpy()[dev], hashes, y[dev])
    dev_logits = blend(branch_scores)
    calibration_checks = development['recipes']['refit_amount_hole']['folds']
    by_fold = []
    for check in calibration_checks:
        fold = check['fold']
        predicted = dev_logits + dev_group * np.clip(check['feature_coefficients'][0], -.30, .30)
        by_fold.append({'fold': fold, **measure_group(dev_group & (folds == fold), y[dev], dev_logits, predicted)})
    group_report = {'development_folds': by_fold, 'holdout': {}, 'test_rows': int(mask.sum())}
    for name, scores in holdout.items():
        group_report['holdout'][name] = measure_group(held_group, y[held], holdout[CONTROL], scores)
    ns = archive_functions()
    paired = compare(ns, y[held], holdout, 'frozen_amount_hole')
    contrasts = {}
    for before, after in (('smooth_amount', 'smooth_amount_distance'),
                          ('linear_branch_adjustment', 'risk_branch_adjustment')):
        variants = {name: holdout[name] for name in (before, after)}
        contrasts[f'{after}_vs_{before}'] = compare(ns, y[held], variants, before)[after]
    bands = {key: auc_by_base_score_band(y[held], holdout[CONTROL], holdout[key]) for key in RECIPES}
    write_json(HERE / 'release_verification.json', {
        'test_files': declared['files'], 'parent_sha256': declared['parent_sha256'],
        'calibration_reference': declared['calibration_reference'], 'frozen_amount_group': group_report,
        'test_movement': test_movement, 'holdout_vs_previous_amount_hole': paired,
        'holdout_nested_comparisons': contrasts, 'holdout_auc_bands': bands,
    })


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('export-development', 'export-full'))
    run(parser.parse_args().mode)
