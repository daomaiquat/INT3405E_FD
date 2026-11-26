"""Recreate the byte-pinned amount-hole submission from frozen margins."""

import argparse
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from verify_release import EXPECTED_ROWS, EXPECTED_UPDATED_ROWS, PARENT_CSV, RELEASE, verify_digest


def replay(output: Path):
    metadata = json.loads((RELEASE / 'amount_hole_submission.json').read_text(encoding='utf-8'))
    release = json.loads((RELEASE / 'release_verification.json').read_text(encoding='utf-8'))
    archive = RELEASE / 'amount_hole_frozen_margin.npz'
    verify_digest(PARENT_CSV, metadata['parent_submission_sha256'])
    verify_digest(archive, release['frozen_margin_sha256'])
    with np.load(archive, allow_pickle=False) as frozen:
        ids, logits = frozen['ids'], frozen['logits']
    if ids.tolist() != release['affected_test_ids'] or logits.shape != (EXPECTED_UPDATED_ROWS,):
        raise ValueError('Frozen margins do not align with the affected IDs')
    edited = dict(zip(ids.tolist(), expit(logits + metadata['full_refit_logit_shift']).astype(str), strict=True))

    buffer = StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\r\n')
    with PARENT_CSV.open(encoding='utf-8', newline='') as parent:
        rows = csv.reader(parent)
        if next(rows) != ['id', 'is_fraud']:
            raise ValueError('Unexpected parent submission header')
        writer.writerow(['id', 'is_fraud'])
        count = 0
        for row_id, probability in rows:
            writer.writerow((row_id, edited.pop(int(row_id), probability)))
            count += 1

    data = buffer.getvalue().encode('utf-8')
    if edited or count != EXPECTED_ROWS or sha256(data).hexdigest() != metadata['submission_sha256']:
        raise ValueError('Replayed CSV does not match the frozen release')
    with output.open('xb') as stream:
        stream.write(data)
    print(f'Recreated {count:,} rows with the published SHA-256 at {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Path for the recreated CSV')
    args = parser.parse_args()
    replay(args.output)
