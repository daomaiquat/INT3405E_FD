"""Apply the frozen amount-hole correction to the subforest blend."""

import numpy as np
import pandas as pd
from scipy.special import expit

from run_honest_leaves import (CONTROL, HERE, METHODS, archive_functions, compare,
                               development_caches, digest, official_data, read_json,
                               read_scores, write_json)

AMOUNT = HERE.parent / 'deep_eda_measurement'
OUTPUT = HERE / 'submission_grandmaster_eda_amount_hole_subforest.csv'


def amount_mask(values, frozen_ticks):
    scaled = values * 100
    ticks = np.rint(scaled).astype(np.int64)
    return (np.abs(scaled - ticks) <= 1e-6) & np.isin(ticks, frozen_ticks)


def run():
    train, test, sample, hashes = official_data()
    labels, _, holdout, *_ = development_caches(train, hashes)
    recipe = read_json(AMOUNT / 'amount_hole_submission.json')
    subforest = read_json(HERE / 'subforest_blend_manifest.json')
    if recipe['input_sha256'] != hashes or subforest['input_sha256'] != hashes:
        raise ValueError('Input checksums differ between the two experiments')
    if recipe['parent_submission_sha256'] != subforest['parent_sha256']:
        raise ValueError('The two experiments must use the same Grandmaster scores')

    blend = METHODS[0]
    ticks = recipe['frozen_amount_ticks']
    if len(set(ticks)) != 404:
        raise ValueError('Frozen amount-hole list must contain 404 distinct amounts')
    dev_scores = read_scores(HERE / 'development_predictions.npz',
                             train.id.to_numpy()[holdout], 'ids', (CONTROL, blend))
    holdout_mask = amount_mask(train.amount_usd.to_numpy()[holdout], ticks)
    holdout_shift = holdout_mask * recipe['dev_validation_logit_shift']
    dev_scores['amount_hole'] = dev_scores[CONTROL] + holdout_shift
    dev_scores['amount_hole_subforest'] = dev_scores[blend] + holdout_shift
    ns = archive_functions()
    metrics = compare(ns, labels[holdout], dev_scores, CONTROL)
    paired = compare(ns, labels[holdout], dev_scores, 'amount_hole')['amount_hole_subforest']

    test_scores = read_scores(HERE / 'full_predictions.npz', test.id.to_numpy(), 'ids', (CONTROL, blend))
    test_mask = amount_mask(test.amount_usd.to_numpy(), ticks)
    if test_mask.sum() != recipe['updated_test_rows']:
        raise ValueError('The mask must update the same test rows as the amount-hole release')
    test_shift = test_mask * recipe['full_refit_logit_shift']
    original_amount = AMOUNT / recipe['file']
    original_blend = HERE / subforest['file']
    for path, checksum, logits in (
        (original_amount, recipe['submission_sha256'], test_scores[CONTROL] + test_shift),
        (original_blend, subforest['sha256'], test_scores[blend]),
    ):
        if digest(path) != checksum:
            raise ValueError(f'Input CSV checksum differs from manifest: {path.name}')
        old = pd.read_csv(path)
        np.testing.assert_array_equal(old.id, sample.id)
        np.testing.assert_allclose(old.is_fraud, expit(logits), atol=1e-14, rtol=0)

    predictions = expit(test_scores[blend] + test_shift)
    temporary = OUTPUT.with_suffix('.tmp')
    sample.assign(is_fraud=predictions).to_csv(temporary, index=False)
    checked = pd.read_csv(temporary)
    if (list(checked.columns) != list(sample.columns) or not checked.id.equals(sample.id)
            or not checked.id.is_unique or not checked.is_fraud.between(0, 1).all()):
        raise ValueError('Invalid submission schema or probabilities')
    np.testing.assert_allclose(checked.is_fraud, predictions, atol=1e-14, rtol=0)
    temporary.replace(OUTPUT)
    write_json(HERE / 'amount_hole_subforest_metrics.json', {
        'input_sha256': hashes,
        'holdout_rows': len(holdout),
        'holdout_amount_hole_rows': int(holdout_mask.sum()),
        'holdout_variants': metrics,
        'holdout_combo_vs_amount_hole': paired,
        'holdout_status': 'Exploratory comparison on the reused development holdout',
        'file': OUTPUT.name,
        'test_rows': len(checked),
        'test_amount_hole_rows': int(test_mask.sum()),
        'sha256': digest(OUTPUT),
        'amount_hole_sha256': recipe['submission_sha256'],
        'subforest_sha256': subforest['sha256'],
    })
    print(f'Verified {OUTPUT.name} ({len(checked):,} rows)')
    print(paired)


if __name__ == '__main__':
    run()
