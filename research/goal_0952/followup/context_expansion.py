"""Test additional likelihood ratios in EDA-supported transaction contexts."""

import argparse

import numpy as np
import pandas as pd
from scipy.special import expit

from run_followup import (HERE, assess_holdout, build_training_view, crossfit_extra, digest, fit_xgb,
                          read_scores, write_json)
from residual_models import fit_codes


# Contexts designed from development EDA and the known surviving interactions.
CONTEXTS = (
    ('txn_count_last_24h', ('card_type',)),
    ('cvv_retry_count', ('card_type', 'auth_method')),
    ('hours_since_last_txn', ('ip_country_mismatch', 'channel')),
    ('hours_since_last_txn', ('txn_count_last_24h',)),
    ('velocity_score', ('txn_count_last_24h', 'auth_method')),
    ('velocity_score', ('billing_shipping_mismatch',)),
    ('merchant_risk_score', ('merchant_category', 'billing_shipping_mismatch')),
    ('amount_usd', ('merchant_category', 'channel')),
    ('distance_from_home_km', ('channel', 'is_foreign_transaction')),
    ('account_balance_usd', ('billing_shipping_mismatch', 'is_foreign_transaction')),
)
CANDIDATES = ('targeted', 'expanded')


def conditional_contrast(value, context, n_value, n_context, labels, query_value, query_context):
    group = context * n_value + value
    neg = np.bincount(group[labels == 0], minlength=n_context * n_value).reshape(n_context, n_value)
    pos = np.bincount(group[labels == 1], minlength=n_context * n_value).reshape(n_context, n_value)

    def local_vs_global(counts, prior_strength):
        distribution = counts.sum(axis=0) + .75
        distribution /= distribution.sum()
        support = counts.sum(axis=1, keepdims=True)
        cpd = (counts + prior_strength * distribution) / (support + prior_strength)
        return np.log(cpd / distribution)

    excess = local_vs_global(pos, 35.) - local_vs_global(neg, 100.)
    enough = (pos.sum(axis=1) >= 8) & (neg.sum(axis=1) >= 120)
    excess[~enough] = 0
    return np.clip(excess[query_context, query_value], -2.5, 2.5).astype(np.float32)


def new_features(fitting, labels, query):
    names = {column for variable, context in CONTEXTS for column in (variable, *context)}
    coded = {name: fit_codes(fitting[name], query[name]) for name in names}
    out = []
    for variable, context_cols in CONTEXTS:
        a, b, n = coded[variable]
        first, second, size = coded[context_cols[0]]
        if len(context_cols) > 1:
            x, y, more = coded[context_cols[1]]
            first, second, size = first * more + x, second * more + y, size * more
        out.append(conditional_contrast(a, first, n, size, labels, b, second))
    return np.column_stack(out)


def density_and_context(fitting, labels, query):
    from class_conditional_density import ClassConditionalDensity

    original = ClassConditionalDensity().fit(fitting, labels).transform(query)
    return np.column_stack((original, new_features(fitting, labels, query)))


def run(mode, device, selected=None, mix_old_density=False):
    context, (x_fit, x_query), (fit_frame, query_frame), started = build_training_view(mode)
    ns, _, test, sample, hashes, frame, labels, fit, query, offset, query_offset, original, incumbent = context
    from class_conditional_density import ClassConditionalDensity

    a, b = crossfit_extra(density_and_context, fit_frame, labels[fit], query_frame)
    boundary = len(ClassConditionalDensity.feature_names)
    cases = (selected,) if selected is not None else CANDIDATES
    raw = {}
    for name in cases:
        subset = slice(boundary, None) if name == 'targeted' else slice(None)
        raw[name] = fit_xgb(np.column_stack((x_fit, a[:, subset])), labels[fit],
                            np.column_stack((x_query, b[:, subset])), offset, query_offset, device)
    ids = frame.id.to_numpy()[query]
    scores = {name: incumbent + .85 * (branch - original) for name, branch in raw.items()}
    if mode == 'development':
        prior = read_scores(HERE / 'holdout_predictions.npz', ids, 'ids', ('xgb_density',))['xgb_density']
        scores['expanded_half_density'] = .5 * scores['expanded'] + .5 * prior
        result = assess_holdout(ns, labels[query], scores, incumbent)
        np.savez_compressed(HERE / 'context_holdout.npz', ids=ids, incumbent=incumbent, **scores)
        write_json(HERE / 'context_metrics.json', {'input_sha256': hashes, 'metrics': result, 'contexts': CONTEXTS})
        for name, row in result.items():
            print(f'{name:24s} {row["roc_auc"]:.8f}  {row["delta_auc"]:+.8f}', flush=True)
    else:
        score = scores[selected]
        if mix_old_density:
            path = HERE / 'test_predictions.npz'
            archived = read_scores(path, ids, 'ids', ('xgb_density',))['xgb_density']
            score = .5 * score + .5 * (incumbent + .85 * (archived - original))
        p = expit(score)
        path = HERE / 'submission_context.csv'
        temporary = path.with_suffix('.tmp')
        sample.assign(is_fraud=p).to_csv(temporary, index=False)
        result = pd.read_csv(temporary)
        if list(result.columns) != ['id', 'is_fraud'] or not result.id.equals(test.id) or not np.isfinite(p).all():
            raise ValueError('Invalid context submission')
        np.testing.assert_allclose(result.is_fraud.to_numpy(), p, atol=1e-14, rtol=0)
        temporary.replace(path)
        np.savez_compressed(HERE / 'context_test.npz', ids=ids, probabilities=p)
        write_json(HERE / 'context_submission.json', {'candidate': 'expanded_half_density' if mix_old_density else selected,
                                                      'rows': len(p), 'filename': path.name, 'sha256': digest(path)})
        print(f'Verified {path.name}, {len(p):,} rows', flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    cli.add_argument('--export', choices=CANDIDATES)
    cli.add_argument('--mix-old-density', action='store_true')
    args = cli.parse_args()
    if args.mix_old_density and args.export != 'expanded':
        cli.error('--mix-old-density applies to --export expanded')
    run('full' if args.export else 'development', args.device, args.export, args.mix_old_density)
