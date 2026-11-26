"""Verify the frozen Grandmaster EDA release using the Python standard library."""

import argparse
import csv
from hashlib import file_digest
import json
import math
from pathlib import Path


GOAL = Path(__file__).resolve().parent / 'goal_0952'
RELEASE = GOAL / 'deep_eda_measurement'
PARENT_CSV = GOAL / 'submission_grandmaster_eda.csv'
CANDIDATE_CSV = RELEASE / 'submission_grandmaster_eda_amount_hole.csv'
EXPECTED_ROWS = 299_984
EXPECTED_UPDATED_ROWS = 4_448
EXPECTED_TICKS = 404


def verify_digest(path, expected):
    with path.open('rb') as stream:
        actual = file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError(f'SHA-256 mismatch for {path}')


def verify_submissions(affected_ids, shift):
    seen = set()
    ids = []
    multiplier = math.exp(shift)
    with (
        PARENT_CSV.open(encoding='utf-8', newline='') as parent,
        CANDIDATE_CSV.open(encoding='utf-8', newline='') as candidate,
    ):
        parent_rows = csv.reader(parent)
        candidate_rows = csv.reader(candidate)
        header = ['id', 'is_fraud']
        if next(parent_rows, None) != header or next(candidate_rows, None) != header:
            raise ValueError('Expected id and is_fraud columns in both submissions')

        for (parent_id, parent_text), (candidate_id, candidate_text) in zip(
                parent_rows, candidate_rows, strict=True):
            row_id = int(candidate_id)
            if row_id != int(parent_id) or row_id in seen:
                raise ValueError(f'Unaligned or duplicate transaction ID {row_id}')
            seen.add(row_id)
            ids.append(row_id)

            parent_probability = float(parent_text)
            candidate_probability = float(candidate_text)
            if not 0 <= parent_probability <= 1 or not 0 <= candidate_probability <= 1:
                raise ValueError(f'Probability outside [0, 1] for ID {row_id}')
            if row_id in affected_ids:
                denominator = 1 - parent_probability + parent_probability * multiplier
                expected = parent_probability * multiplier / denominator
                if candidate_probability <= parent_probability or not math.isclose(
                        candidate_probability, expected, rel_tol=0, abs_tol=1e-15):
                    raise ValueError(f'Incorrect probability shift for ID {row_id}')
            elif candidate_text != parent_text:
                raise ValueError(f'Unexpected probability change for ID {row_id}')

    if len(ids) != EXPECTED_ROWS or not affected_ids.issubset(seen):
        raise ValueError('Unexpected submission row count or missing affected IDs')
    return ids


def verify_official_test(path, expected_digest, ids, ticks, affected_ids):
    verify_digest(path, expected_digest)
    official_affected_ids = set()
    with path.open(encoding='utf-8', newline='') as stream:
        rows = csv.DictReader(stream)
        if not {'id', 'amount_usd'}.issubset(rows.fieldnames or []):
            raise ValueError('The official test CSV must contain id and amount_usd')
        for row_id, row in zip(ids, rows, strict=True):
            if int(row['id']) != row_id:
                raise ValueError(f'Official test ID order differs at ID {row_id}')
            cents = 100 * float(row['amount_usd'])
            tick = round(cents)
            if tick in ticks and abs(cents - tick) <= 1e-6:
                official_affected_ids.add(row_id)
    if official_affected_ids != affected_ids:
        raise ValueError('The amount ticks select a different set of official test IDs')


def verify(test_path):
    metadata = json.loads((RELEASE / 'amount_hole_submission.json').read_text(encoding='utf-8'))
    verification = json.loads((RELEASE / 'release_verification.json').read_text(encoding='utf-8'))
    saved_ids = verification['affected_test_ids']
    affected_ids = set(saved_ids)
    if (len(affected_ids) != EXPECTED_UPDATED_ROWS or len(saved_ids) != len(affected_ids)
            or not all(type(row_id) is int for row_id in affected_ids)
            or metadata['updated_test_rows'] != EXPECTED_UPDATED_ROWS
            or metadata['test_rows'] != EXPECTED_ROWS):
        raise ValueError('Unexpected row counts or affected test IDs in the release manifests')

    saved_ticks = metadata['frozen_amount_ticks']
    ticks = set(saved_ticks)
    if (len(ticks) != EXPECTED_TICKS or not all(type(tick) is int for tick in ticks)
            or saved_ticks != sorted(ticks)):
        raise ValueError('Expected 404 distinct, sorted integer amount ticks')
    shift = metadata['full_refit_logit_shift']
    if not 0 < shift <= 0.30:
        raise ValueError('The logit shift must be positive and at most 0.30')

    verify_digest(PARENT_CSV, metadata['parent_submission_sha256'])
    verify_digest(CANDIDATE_CSV, metadata['submission_sha256'])
    ids = verify_submissions(affected_ids, shift)
    if test_path is not None:
        verify_official_test(test_path, metadata['input_sha256']['test.csv'], ids, ticks, affected_ids)
        print(f'Verified official test SHA-256 and all {len(ticks)} frozen amount ticks.')
    print(f'Verified both submission SHA-256 hashes and {len(ids):,} aligned rows.')
    print(f'Verified {len(affected_ids):,} probability increases and all other probability strings.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', type=Path, metavar='PATH',
                        help='Verify the official test.csv and confirm the affected IDs from amount_usd.')
    args = parser.parse_args()
    try:
        verify(args.test)
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        parser.exit(1, f'Verification failed. {exc}\n')
