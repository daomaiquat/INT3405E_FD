"""Refit the screened Amount correction against the full Grandmaster OOF."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from measurement_eda import (
    GOAL, HERE, CAP, RIDGE, SCALES, describe_group, density_rules, fit_shift,
)
from evaluate_and_export import read_scores
from grandmaster_inputs import PROJECT, development_caches, digest, read_json, write_json

sys.path.insert(0, str(GOAL / 'followup'))
from run_followup import build_inputs


GROUP = 'amount_usd_hole'
OUTPUT = HERE / 'submission_grandmaster_eda_amount_hole.csv'
METADATA = HERE / 'amount_hole_submission.json'
SCREEN = HERE / 'measurement_metrics.json'
ANCHOR_CSV = GOAL / 'submission_grandmaster_eda.csv'
ANCHOR_SCORES = GOAL / 'selected_predictions.npz'
PARENT_METRICS = GOAL / 'evaluation.json'
AMOUNT = 'amount_usd'
TARGET = 'is_fraud'


def tick_data(values):
    scaled = values * SCALES[AMOUNT]
    ticks = np.rint(scaled).astype(np.int64)
    return dict(ticks=ticks, off_grid=np.abs(scaled - ticks) > 1e-6)


def read_predictions(path):
    return pd.read_csv(path, dtype={TARGET: str}, keep_default_na=False)


def check_anchor(ids):
    original = read_predictions(ANCHOR_CSV)
    parent_sha = digest(ANCHOR_CSV)
    if parent_sha != read_json(PARENT_METRICS)['submission_sha256']:
        raise ValueError('Unexpected parent submission checksum')
    if list(original.columns) != ['id', TARGET] or not original['id'].is_unique:
        raise ValueError('Unexpected parent submission schema')
    np.testing.assert_array_equal(original['id'], ids)
    probabilities = read_scores(ANCHOR_SCORES, ids, 'test_ids', ('test',))['test']
    np.testing.assert_allclose(original[TARGET].astype(float), probabilities, atol=1e-15, rtol=0)
    return original, probabilities, parent_sha


def verify():
    metadata = read_json(METADATA)
    metrics = read_json(SCREEN)
    if metadata['screen_sha256'] != digest(SCREEN) or metrics['finalists'] != [GROUP]:
        raise ValueError('Unexpected discovery metrics or finalist')
    for name, expected in metadata['input_sha256'].items():
        if digest(PROJECT.parent / 'data' / name) != expected:
            raise ValueError(f'Unexpected official data checksum for {name}')
    test = pd.read_csv(PROJECT.parent / 'data' / 'test.csv', usecols=['id', AMOUNT])
    sample = pd.read_csv(PROJECT.parent / 'data' / 'sample_submission.csv')
    ids = test['id'].to_numpy()
    parent, base, parent_sha = check_anchor(ids)
    candidate = read_predictions(OUTPUT)
    if (not candidate.columns.equals(sample.columns) or not candidate['id'].equals(sample['id'])
            or not candidate['id'].is_unique or not candidate[TARGET].astype(float).between(0., 1.).all()):
        raise ValueError('Invalid candidate submission schema or values')
    np.testing.assert_array_equal(candidate['id'], ids)
    amount = tick_data(test[AMOUNT].to_numpy())
    ticks = metadata['frozen_amount_ticks']
    if ticks != sorted(set(ticks)) or len(ticks) != metrics['groups'][GROUP]['rule']['selected_values']:
        raise ValueError('Invalid saved set of lower-density amount ticks')
    mask = np.isin(amount['ticks'], ticks) & ~amount['off_grid']
    assert mask.sum() == metadata['updated_test_rows'] == metrics['groups'][GROUP]['test_rows']
    np.testing.assert_array_equal(candidate.loc[~mask, TARGET], parent.loc[~mask, TARGET])
    actual = candidate.loc[mask, TARGET].astype(float)
    shift = metadata['full_refit_logit_shift']
    assert abs(shift) <= CAP and shift > 0
    np.testing.assert_allclose(actual, expit(logit(base[mask]) + shift), atol=1e-15, rtol=0.)
    assert (actual.to_numpy() > base[mask]).all()
    assert digest(OUTPUT) == metadata['submission_sha256']
    assert parent_sha == metadata['parent_submission_sha256']
    assert digest(ANCHOR_SCORES) == metadata['parent_scores_sha256']
    assert len(candidate) == metadata['test_rows']
    print(f'{OUTPUT.name}: {len(candidate):,} valid rows, {mask.sum():,} increased, SHA-256 verified')


def export():
    metrics = read_json(SCREEN)
    if metrics['finalists'] != [GROUP] or GROUP not in metrics['holdout']['groups']:
        raise ValueError('Screened validation of the fixed Amount rule is required')
    context = build_inputs('full')
    _, train, test, sample, hashes, frame, labels, fit, query, offset, _, _, incumbent = context
    assert hashes == metrics['input_sha256']
    archive = development_caches(train, hashes)
    development = archive[1]
    del archive
    item = tick_data(frame[AMOUNT].to_numpy())
    _, _, mask, definition = next(
        rule for rule in density_rules(AMOUNT, item, development) if rule[1] == GROUP
    )
    assert definition == metrics['groups'][GROUP]['rule']
    assert mask[development].sum() == metrics['groups'][GROUP]['dev']['rows']
    train_mask, test_mask = mask[fit], mask[query]
    ticks = np.unique(item['ticks'][mask])
    assert len(ticks) == definition['selected_values']
    assert train_mask.sum() == metrics['groups'][GROUP]['train_rows']
    assert test_mask.sum() == metrics['groups'][GROUP]['test_rows']
    shift = fit_shift(labels[fit][train_mask], offset[train_mask])
    ids = test['id'].to_numpy()
    original, base, parent_sha = check_anchor(ids)
    np.testing.assert_array_equal(expit(incumbent), base)
    np.testing.assert_array_equal(sample['id'], ids)
    output = original.copy()
    output.loc[test_mask, TARGET] = expit(incumbent[test_mask] + shift).astype(str)
    temporary = OUTPUT.with_suffix('.tmp')
    output.to_csv(temporary, index=False)
    temporary.replace(OUTPUT)
    metadata = dict(
        file=OUTPUT.name, submission_sha256=digest(OUTPUT), input_sha256=hashes,
        screen_sha256=digest(SCREEN), parent_submission_sha256=parent_sha,
        parent_scores_sha256=digest(ANCHOR_SCORES), test_rows=len(test),
        updated_test_rows=int(test_mask.sum()), updated_test_share=float(test_mask.mean()),
        frozen_amount_ticks=ticks.tolist(), full_refit_logit_shift=shift,
        full_refit_odds_multiplier=float(np.exp(shift)), ridge=RIDGE, shift_cap=CAP,
        full_fit_rows=len(train),
        full_fit_reference='Grandmaster hierarchical meta-OOF from full official training caches',
        full_fit_group=describe_group(labels[fit][train_mask], offset[train_mask]),
        dev_validation_logit_shift=metrics['groups'][GROUP]['locked_dev_shift'],
    )
    write_json(METADATA, metadata)
    print(f'Full refit logit shift: {shift:.12f}; odds multiplier: {np.exp(shift):.12f}')
    verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('export', 'verify'))
    mode = parser.parse_args().mode
    (export if mode == 'export' else verify)()
