"""Account for the AUC gap by the score bands of both transactions in a pair."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from grandmaster_inputs import development_caches, official_data, read_json, write_json

QUANTILES = (0., .5, .75, .9, .95, .99, 1.)
NAMES = ('0–50', '50–75', '75–90', '90–95', '95–99', '99–100')


def lost_pairs(positives, negatives):
    ordered = np.sort(negatives)
    left = np.searchsorted(ordered, positives, side='left')
    right = np.searchsorted(ordered, positives, side='right')
    return float(np.sum(len(negatives) - .5 * (left + right)))


def analyze(y, probabilities, unlabeled):
    edges = np.quantile(probabilities, QUANTILES)[1:-1]
    bands = np.searchsorted(edges, probabilities, side='right')
    test_bands = np.searchsorted(edges, unlabeled, side='right')
    levels = range(len(NAMES))
    fraud = y == 1
    by_group = [{
        'percentile': label, 'rows': int((bands == i).sum()), 'fraud': int((fraud & (bands == i)).sum()),
        'legitimate': int((~fraud & (bands == i)).sum()), 'expected_fraud': float(probabilities[bands == i].sum()),
        'test_rows': int((test_bands == i).sum()), 'test_share': float(np.mean(test_bands == i))}
        for i, label in enumerate(NAMES)]
    matrix = np.array([[lost_pairs(probabilities[(bands == i) & fraud], probabilities[(bands == j) & ~fraud])
                        for j in levels] for i in levels])
    pairs = int(fraud.sum()) * int((~fraud).sum())
    auc = roc_auc_score(y, probabilities)
    np.testing.assert_allclose(matrix.sum() / pairs, 1. - auc, rtol=0., atol=1e-12)
    np.testing.assert_equal([sum(g[k] for g in by_group) for k in ('rows', 'fraud', 'legitimate', 'test_rows')],
                           [len(y), fraud.sum(), (~fraud).sum(), len(unlabeled)])
    for i, row in enumerate(by_group):
        row['lost_pair_share_for_fraud_in_band'] = float(matrix[i].sum() / matrix.sum())
        row['lost_pair_share_for_legitimate_in_band'] = float(matrix[:, i].sum() / matrix.sum())
    return {'auc': auc, 'target_auc': .952, 'positive_negative_pairs': pairs, 'lost_pairs': float(matrix.sum()),
            'net_pairs_to_correct_for_target_on_this_holdout': float((.952 - auc) * pairs),
            'score_edges': edges.tolist(), 'groups': by_group, 'lost_pairs_matrix': matrix.tolist()}


def render(result):
    matrix = np.asarray(result['lost_pairs_matrix'])
    shares = 100 * matrix / matrix.sum()
    fig, ax = plt.subplots(figsize=(9.3, 6.1), layout='constrained')
    fig.patch.set_facecolor('#f9fafb')
    ax.imshow(shares, cmap='Blues', vmin=0., vmax=float(shares.max()), aspect='auto')
    ax.set_xticks(range(len(NAMES)), NAMES)
    ax.set_yticks(range(len(NAMES)), NAMES)
    ax.tick_params(length=0, pad=8)
    ax.set_xlabel('Phân vị điểm của giao dịch hợp lệ', labelpad=12)
    ax.set_ylabel('Phân vị điểm của giao dịch gian lận', labelpad=12)
    fig.suptitle('Các cặp Grandmaster EDA xếp sai trên holdout', fontsize=15, fontweight='bold')
    ax.set_title('Tỷ trọng trong tổng số cặp xếp sai · Cặp bằng điểm tính nửa lỗi', fontsize=10, pad=12)
    for i in range(len(NAMES)):
        for j in range(len(NAMES)):
            v = shares[i, j]
            text = '—' if not v else f'{v:.1f}%'.replace('.', ',')
            ax.text(j, i, text, ha='center', va='center', color='white' if v > shares.max() * .65 else '#142638')
    for edge in ax.spines.values():
        edge.set_visible(False)
    fig.savefig(HERE / 'rank_errors.png', dpi=175)
    plt.close(fig)


def run():
    train, test, _, hashes = official_data()
    y, _, held, *_ = development_caches(train, hashes)
    archive = PARENT / 'selected_predictions.npz'
    with np.load(archive, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['holdout_ids'], train.id.to_numpy()[held])
        np.testing.assert_array_equal(saved['test_ids'], test.id)
        result = analyze(y[held], saved['holdout'], saved['test'])
    expected = read_json(PARENT / 'evaluation.json')['best_holdout_auc']
    np.testing.assert_allclose(result['auc'], expected, rtol=0., atol=1e-12)
    result['input_sha256'] = hashes
    write_json(HERE / 'rank_errors.json', result)
    render(result)
    print(f'Validated {result["positive_negative_pairs"]:,} pairs and exact archived AUC {expected:.8f}')


if __name__ == '__main__':
    run()
