"""Audit out-of-fold residuals, independent density maps, and score changes."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ks_2samp, norm
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from calibrate import (AMOUNT, AMOUNT_RECIPE, CONTROL, DISTANCE, SOURCE, blend, final_bank, load_oof,
                       neighbor_deficit, partitions, ticks)
from grandmaster_inputs import archive_functions, development_caches, digest, official_data, read_json, write_json
from evaluate_and_export import auc_by_base_score_band, read_scores
from robust_density import compare

SCALES = {AMOUNT: 100, DISTANCE: 100, 'hours_since_last_txn': 100, 'account_balance_usd': 100,
          'velocity_score': 10, 'merchant_risk_score': 10}
LEADS = (AMOUNT, DISTANCE)
TIME = 'hours_since_last_txn'
NEW = 'smooth_amount_distance'


def raw_profile(train, test):
    predictors = [c for c in test if c != 'id']
    output = {}
    for name in predictors:
        a, b = train[name], test[name]
        common = {'train_missing': int(a.isna().sum()), 'test_missing': int(b.isna().sum()),
                  'train_unique': int(a.nunique()), 'test_unique': int(b.nunique())}
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a) and a.nunique() > 10:
            s = np.sqrt((a.var() + b.var()) / 2.)
            common.update(kind='numeric', ks=float(ks_2samp(a.dropna(), b.dropna()).statistic),
                          standardized_mean_delta=float((b.mean() - a.mean()) / s) if s else 0.,
                          train_q=np.quantile(a.dropna(), [.005, .5, .995]).tolist(),
                          test_q=np.quantile(b.dropna(), [.005, .5, .995]).tolist())
        else:
            frequency = pd.concat((a.value_counts(normalize=True, dropna=False),
                                   b.value_counts(normalize=True, dropna=False)), axis=1).fillna(0)
            common.update(kind='discrete', total_variation=float(frequency.diff(axis=1).iloc[:, 1].abs().sum() / 2.))
        output[name] = common
    shared = pd.concat((train[predictors], test[predictors]), ignore_index=True)
    train_hash = pd.util.hash_pandas_object(shared.iloc[:len(train)], index=False)
    test_hash = pd.util.hash_pandas_object(shared.iloc[len(train):], index=False)
    return {'predictors': len(predictors), 'train_rows': len(train), 'test_rows': len(test),
            'train_fraud': int(train.is_fraud.sum()),
            'train_duplicate_predictors_excess_rows': int(shared.iloc[:len(train)].duplicated().sum()),
            'test_duplicate_predictors_excess_rows': int(shared.iloc[len(train):].duplicated().sum()),
            'test_rows_matching_train_hash': int(test_hash.isin(train_hash).sum()), 'fields': output}


def maps(values, dev, held, test, labels, scale):
    x = values * (scale / 100.)
    t = test * (scale / 100.)
    combined = np.empty((len(dev), 2))
    count = np.zeros(len(dev), dtype=np.int8)
    cv = list(partitions(labels))
    reference = [idx for _, idx in cv]
    for i, (fit, query) in enumerate(cv):
        combined[query] = neighbor_deficit(x[dev[fit]], x[dev[query]])
        for j, rows in enumerate(reference):
            if i != j:
                count[query] += neighbor_deficit(x[dev[rows]], x[dev[query]])[:, 0] > 0
    names = {'development': (combined, count)}
    for key, query in (('holdout', x[held]), ('test', t)):
        votes = sum((neighbor_deficit(x[dev[rows]], query)[:, 0] > 0).astype(np.int8) for rows in reference)
        names[key] = (neighbor_deficit(x[dev], query), votes)
    return names


def map_drift(ref, test, scale):
    a, b = ref * (scale / 100.), test * (scale / 100.)
    ta, va = ticks(a)
    tb, vb = ticks(b)
    union = np.union1d(ta[va], tb[vb])
    q = union / 100.
    left = neighbor_deficit(a, q)[:, 0]
    right = neighbor_deficit(b, q)[:, 0]
    out = {}
    for key, cut in (('any', 0.), ('strong', .6)):
        ref_hits, test_hits = left > cut, right > cut
        both = ref_hits & test_hits
        support = ref_hits | test_hits
        out[key] = {'dev_ticks': int(ref_hits.sum()), 'test_ticks': int(test_hits.sum()),
                    'shared_ticks': int(both.sum()), 'jaccard': float(both.sum() / max(1, support.sum())),
                    'test_rows_at_shared_ticks': int((vb & np.isin(tb, union[both])).sum()),
                    'test_rows_at_dev_ticks': int((vb & np.isin(tb, union[ref_hits])).sum()),
                    'test_rows_at_test_ticks': int((vb & np.isin(tb, union[test_hits])).sum())}
    return out


def residual(mask, y, z, strata, control=None):
    p = expit(z)
    e, h = y - p, p * (1 - p)
    n = int(mask.sum())
    data = {'rows': n, 'fraud': int(y[mask].sum()), 'expected': float(p[mask].sum()),
            'raw_excess': float(e[mask].sum())}
    if n == 0:
        return data
    contrast = ~mask if control is None else control
    size = int(strata.max()) + 1
    hc = np.bincount(strata[contrast], h[contrast], minlength=size)
    hm = np.bincount(strata[mask], h[mask], minlength=size)
    gc = np.bincount(strata[contrast], e[contrast], minlength=size)
    supported = hc >= 8.
    ratio = np.divide(hm, hc, out=np.zeros(size), where=supported)
    keep = mask & supported[strata]
    excess = e[keep].sum() - (ratio * gc).sum()
    information = h[keep].sum() + np.dot(ratio**2, hc)
    sd = np.sqrt(information)
    data.update(matched_rows=int(keep.sum()), matched_excess=float(excess), matched_per_1000=float(1000 * excess / max(1, keep.sum())),
                z=float(excess / sd) if sd else 0.,
                ci95_matched_excess=[float(excess - 1.96 * sd), float(excess + 1.96 * sd)],
                p_value=float(2 * norm.sf(abs(excess) / sd)) if sd else 1.)
    return data


def strata_scores(z, values, edges, information_weighted=False):
    cuts = np.arange(.1, 1., .1)
    if information_weighted:
        order = np.argsort(z)
        p = expit(z[order])
        weight = np.cumsum(p * (1 - p))
        breaks = z[order[np.searchsorted(weight, cuts * weight[-1])]]
    else:
        breaks = np.quantile(z, cuts)
    return 10 * np.searchsorted(edges, values) + np.searchsorted(breaks, z)


def holm(p_values):
    running = 0.
    result = {}
    for i, (name, value) in enumerate(sorted(p_values.items(), key=lambda x: x[1])):
        running = max(running, value * (len(p_values) - i))
        result[name] = min(running, 1.)
    return result


def point_audit(train, test, dev, held, labels, oof, baseline, folds):
    block = {}
    for field, scale in SCALES.items():
        values = train[field].to_numpy()
        computed = maps(values, dev, held, test[field].to_numpy(), labels[dev], scale)
        cutoff = np.unique(np.quantile(values[dev], [.2, .4, .6, .8]))
        rows = {}
        for key, (signals, votes) in computed.items():
            hit, strong = signals[:, 0] > 0, signals[:, 0] > .6
            row = {'rows': len(hit), 'active': int(hit.sum()), 'strong': int(strong.sum()),
                   'reference_votes': {str(i): int((votes == i).sum()) for i in range(3 if key == 'development' else 4)}}
            if key != 'test':
                ids, z = (dev, oof) if key == 'development' else (held, baseline)
                y = labels[ids]
                group = strata_scores(z, values[ids], cutoff)
                sensitivity_group = strata_scores(z, values[ids], cutoff, information_weighted=True)
                row['active_residual'] = residual(hit, y, z, group)
                row['information_balanced_residual'] = residual(hit, y, z, sensitivity_group)
                row['by_strength'] = {name: residual(mask, y, z, group, ~hit) for name, mask in (
                    ('weak_0_to_0.2', hit & (signals[:, 0] <= .2)),
                    ('mid_0.2_to_0.6', (signals[:, 0] > .2) & ~strong),
                    ('strong_over_0.6', strong))}
                row['by_replication'] = {str(i): residual(votes == i, y, z, group, votes == 0)
                                         for i in range(1, 3 if key == 'development' else 4)}
                row['information_balanced_replication'] = {
                    str(i): residual(votes == i, y, z, sensitivity_group, votes == 0)
                    for i in range(1, 3 if key == 'development' else 4)
                }
                if field == TIME:
                    high_count = train.txn_count_last_24h.to_numpy()[ids] > 3
                    needed = 2 if key == 'development' else 3
                    row['replicated_time_by_txn_count'] = {
                        name: residual((votes == needed) & mask, y, z, sensitivity_group, (votes == 0) & mask)
                        for name, mask in (('at_most_3', ~high_count), ('above_3', high_count))
                    }
                if key == 'development':
                    row['folds'] = [residual(hit[q], y[q], z[q], group[q]) for q in (folds == i for i in (1, 2, 3))]
            rows[key] = row
        block[field] = {'splits': rows, 'independent_train_test_maps': map_drift(values[dev], test[field].to_numpy(), scale)}
        if field in LEADS:
            block[field]['score_holdout'] = computed['holdout'][0]
    for section in ('development', 'holdout'):
        for comparison in ('active_residual', 'information_balanced_residual'):
            adjusted = holm({k: v['splits'][section][comparison]['p_value'] for k, v in block.items()})
            for name, value in adjusted.items():
                block[name]['splits'][section][comparison]['p_value_holm_6_columns'] = value
    a, b = (block[field].pop('score_holdout') for field in LEADS)
    return block, a, b


def score_audit(holdout, labels, a, b, coefficients):
    base = holdout[CONTROL]
    z_a, z_d = a @ coefficients[:2], b @ coefficients[2:]
    clips = lambda z: np.clip(z, -.30, .30)
    variants = {CONTROL: base, 'amount_only_joint_coefficients': base + clips(z_a),
                'distance_only_joint_coefficients': base + clips(z_d), NEW: base + clips(z_a + z_d)}
    np.testing.assert_allclose(variants[NEW], holdout[NEW], atol=1e-14, rtol=0)
    p0 = expit(base)
    n = len(labels)
    losses = {key: np.logaddexp(0, value) - labels * value for key, value in variants.items()}

    def change(group):
        delta = (losses[NEW] - losses[CONTROL])[group]
        avg = delta.mean()
        se = delta.std(ddof=1) / np.sqrt(len(delta))
        return {'rows': int(group.sum()), 'fraud': int(labels[group].sum()), 'expected': float(p0[group].sum()),
                'delta_log_loss': float(avg), 'ci95_delta_log_loss': [float(avg - 1.96 * se), float(avg + 1.96 * se)],
                'contribution_to_total_delta_loss': float(delta.sum() / n)}

    a_hit, d_hit = a[:, 0] > 0, b[:, 0] > 0
    parts = {key: change(mask) for key, mask in (('neither', ~a_hit & ~d_hit), ('amount_only', a_hit & ~d_hit),
                                               ('distance_only', ~a_hit & d_hit), ('both', a_hit & d_hit))}
    all_rows = change(np.ones(n, dtype=bool))
    np.testing.assert_allclose(sum(s['contribution_to_total_delta_loss'] for s in parts.values()),
                               all_rows['delta_log_loss'], atol=1e-14, rtol=0)
    detail = compare(archive_functions(), labels, variants, CONTROL)
    for name, score in variants.items():
        np.testing.assert_allclose(detail[name]['average_precision'], average_precision_score(labels, score))
        np.testing.assert_allclose(detail[name]['roc_auc'], roc_auc_score(labels, score))
        np.testing.assert_allclose(detail[name]['log_loss'], log_loss(labels, expit(score)))
    return {'components': detail, 'cooccurrence': parts, 'all_rows_loss': all_rows,
            'rank_bands': auc_by_base_score_band(labels, base, variants[NEW]),
            'rows_with_capped_combined_shift': int((np.abs(z_a + z_d) > .30).sum())}


def plot(summary):
    columns = list(SCALES)
    pretty = ['Số tiền', 'Khoảng cách', 'Giờ từ giao dịch trước', 'Số dư', 'Điểm vận tốc', 'Rủi ro người bán']
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True, sharey=True, layout='constrained')
    for axis, split, title in zip(axes, ('development', 'holdout'),
                                  ('Tập phát triển · ngoài fold', 'Tập giữ lại · tổ hợp gốc'), strict=True):
        stats = [summary[name]['splits'][split]['information_balanced_residual'] for name in columns]
        center = np.array([x['matched_per_1000'] for x in stats])
        interval = np.array([1000 * np.asarray(x['ci95_matched_excess']) / x['matched_rows'] for x in stats])
        axis.axvline(0, ls='--', color='#637083', lw=1)
        axis.errorbar(center, range(len(stats)), xerr=np.abs(interval.T - center), fmt='o',
                      color='#155e75', capsize=4, lw=1.5)
        axis.set_yticks(range(len(pretty)), pretty)
        axis.grid(axis='x', alpha=.22)
        axis.set_title(title)
        axis.set_xlabel('Ca gian lận dư trên 1.000 dòng')
    axes[0].invert_yaxis()
    fig.suptitle('Sai lệch sau khi ghép theo điểm rủi ro và độ lớn · khoảng tin cậy 95%')
    fig.savefig(HERE / 'matched_residuals.png', dpi=155)
    plt.close(fig)


def run():
    train, test, _, hashes = official_data()
    labels, dev, held, *_ = development_caches(train, hashes)
    score_bank, folds = load_oof('development', train.id.to_numpy()[dev], hashes, labels[dev])
    oof = blend(score_bank)
    ids = train.id.to_numpy()[held]
    bank = read_scores(HERE.parent / 'development_predictions.npz', ids, 'ids', (CONTROL, NEW))
    _, original = final_bank('development', ids)
    np.testing.assert_array_equal(bank[CONTROL], original)
    assert digest(SOURCE / 'submission_grandmaster_eda.csv') == read_json(SOURCE / 'evaluation.json')['submission_sha256']
    prior = read_json(AMOUNT_RECIPE)
    recipe = read_json(HERE.parent / 'development_metrics.json')
    assert prior['input_sha256'] == recipe['input_sha256'] == hashes
    fields, a, b = point_audit(train, test, dev, held, labels, oof, original, folds)
    decomposition = score_audit(bank, labels[held], a, b, np.asarray(recipe['recipes'][NEW]['feature_coefficients']))
    amounts = train[AMOUNT].to_numpy()
    edges = np.quantile(amounts[dev], [.2, .4, .6, .8])
    for name, ids, z in (('development', dev, oof), ('holdout', held, original)):
        points, valid = ticks(amounts[ids])
        frozen = valid & np.isin(points, prior['frozen_amount_ticks'])
        info = fields[AMOUNT]['splits'][name]
        info['frozen_404'] = residual(frozen, labels[ids], z, strata_scores(z, amounts[ids], edges))
        info['frozen_404_information_balanced'] = residual(
            frozen, labels[ids], z, strata_scores(z, amounts[ids], edges, information_weighted=True))
    output = {'input_sha256': hashes, 'raw_data': raw_profile(train, test), 'fields': fields,
              'joint_score_audit': decomposition}
    write_json(HERE / 'eda_metrics.json', output)
    plot(fields)
    for field in fields:
        for name in ('development', 'holdout'):
            x = fields[field]['splits'][name]['active_residual']
            print(field, name, x['rows'], round(x['matched_per_1000'], 3),
                  round(x['p_value_holm_6_columns'], 5), flush=True)


if __name__ == '__main__':
    run()
