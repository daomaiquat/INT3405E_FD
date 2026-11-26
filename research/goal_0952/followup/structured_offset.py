"""Fit a jointly regularized interaction correction, including a low-risk component."""

import argparse
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from run_followup import HERE, PARENT, assess_holdout, build_inputs, digest, read_json, write_json
from residual_models import fit_codes


STRENGTH = 95.
RISK_PIVOT = .023
MAX_PAIRS = 24
RECIPES = {'shared_half': ('shared', .50), 'gated_half': ('gated', .50), 'gated_full': ('gated', 1.)}


def gate(offset):
    return expit((logit(RISK_PIVOT) - offset) / .75)


def supported(left, right, width, information):
    counts = np.bincount(left, minlength=width)
    curvature = np.bincount(left, weights=information, minlength=width)
    good = (counts >= 80) & (curvature >= 8)
    return good[left].astype(np.float32), good[right].astype(np.float32)


def sparse_features(frame, query, base_fit, pairs):
    names = sorted({name for pair in pairs for name in pair})
    codes = {name: fit_codes(frame[name], query[name]) for name in names}
    probability = expit(base_fit)
    information = probability * (1 - probability)
    fit_blocks, query_blocks, weights_fit, weights_query = [], [], [], []
    cursor = 0

    def add_field(tr, te, size, strength):
        nonlocal cursor
        fit_blocks.append((cursor + tr).astype(np.int32))
        query_blocks.append((cursor + te).astype(np.int32))
        a, b = supported(tr, te, size, information)
        weights_fit.append(a * strength)
        weights_query.append(b * strength)
        cursor += size

    for name in names:
        add_field(*codes[name], .65)
    for left, right in pairs:
        a, b, n_left = codes[left]
        c, d, n_right = codes[right]
        add_field(a * n_right + c, b * n_right + d, n_left * n_right, 1.)
    distance = 'distance_from_home_km'
    fit_digit = np.rint(frame[distance].to_numpy() * 100).astype(np.int64) % 10
    query_digit = np.rint(query[distance].to_numpy() * 100).astype(np.int64) % 10
    add_field(fit_digit, query_digit, 10, .8)

    def get_csr(blocks, weights):
        m = len(blocks[0])
        matrix = csr_matrix((np.column_stack(weights).ravel(), np.column_stack(blocks).ravel(),
                             np.arange(m + 1, dtype=np.int32) * len(blocks)), shape=(m, cursor))
        matrix.eliminate_zeros()
        return matrix

    return get_csr(fit_blocks, weights_fit), get_csr(query_blocks, weights_query)


def joint_calibrator(fitting, labels, offset, scoring, query_offset, gated):
    width = fitting.shape[1]
    inv = 1 / len(labels)
    g = gate(offset)

    def objective(coef):
        adjusted = offset + fitting @ coef[:width]
        if gated:
            adjusted += g * (fitting @ coef[width:])
        residual = expit(adjusted) - labels
        gradient = fitting.T @ residual
        if gated:
            gradient = np.r_[gradient, fitting.T @ (g * residual)]
        loss = (np.logaddexp(0, adjusted) - labels * adjusted).sum()
        return inv * (loss + (STRENGTH / 2) * (coef @ coef)), inv * (gradient + STRENGTH * coef)

    with threadpool_limits(limits=4):
        model = minimize(objective, np.zeros(width * (1 + gated)), method='L-BFGS-B', jac=True,
                         options={'maxiter': 90, 'ftol': 1e-11, 'gtol': 3e-8})
    if not np.isfinite(model.x).all():
        raise ValueError('Nonfinite structured model coefficients')
    correction = scoring @ model.x[:width]
    if gated:
        correction += gate(query_offset) * (scoring @ model.x[width:])
    summary = {'iterations': int(model.nit), 'converged': bool(model.success), 'penalty': STRENGTH,
               'active_columns': int(np.count_nonzero(model.x)), 'reason': str(model.message)}
    return np.clip(correction, -.40, .40), summary


def proposed_corrections(mode, needed):
    t = perf_counter()
    ns, _train, test, sample, hashes, frame, y, fit, query, offset, query_offset, _, incumbent = build_inputs(mode)
    eda = read_json(PARENT / 'residual_eda.json')
    if eda['input_sha256'] != hashes:
        raise ValueError('EDA has a different source snapshot')
    pairs = [(row['left'], row['right']) for row in eda['pair_results']
             if (row['cross_delta_logloss_x1m'] or 0) > 0][:MAX_PAIRS]
    a, b = sparse_features(frame.iloc[fit], frame.iloc[query], offset, pairs)
    result, metrics = {}, {}
    for model in needed:
        result[model], metrics[model] = joint_calibrator(a, y[fit], offset, b, query_offset, model == 'gated')
        print(f'{mode}: {model} {metrics[model]}, {perf_counter() - t:.1f}s', flush=True)
    return ns, test, sample, frame.id.to_numpy()[query], y[query] if mode == 'development' else None, incumbent, result, metrics, hashes


def development():
    ns, _, _, ids, y, incumbent, effects, fit_record, hashes = proposed_corrections('development', ('shared', 'gated'))
    scores = {'incumbent': incumbent, **{name: incumbent + scale * effects[model] for name, (model, scale) in RECIPES.items()}}
    result = assess_holdout(ns, y, scores, incumbent)
    write_json(HERE / 'structured_metrics.json', {'input_sha256': hashes, 'metrics': result, 'fit': fit_record})
    np.savez_compressed(HERE / 'structured_holdout.npz', ids=ids, **scores)
    for name, row in result.items():
        print(f'{name:20s} {row["roc_auc"]:.8f}  {row["delta_auc"]:+.8f}', flush=True)


def export(name):
    model, scale = RECIPES[name]
    _, test, sample, ids, _, incumbent, effects, fit_record, hashes = proposed_corrections('full', (model,))
    p = expit(incumbent + scale * effects[model])
    path = HERE / 'submission_structured.csv'
    sample.assign(is_fraud=p).to_csv(path, index=False)
    result = pd.read_csv(path)
    if list(result.columns) != ['id', 'is_fraud'] or not result.id.equals(test.id) or not np.isfinite(p).all():
        raise ValueError('Invalid exported schema or probabilities')
    np.testing.assert_allclose(result.is_fraud.to_numpy(), p, atol=1e-14, rtol=0)
    np.savez_compressed(HERE / 'structured_test.npz', ids=ids, probabilities=p, correction=effects[model])
    write_json(HERE / 'structured_submission.json', {'candidate': name, 'fit': fit_record, 'sha256': digest(path),
                                                     'rows': len(test), 'input_sha256': hashes})
    print(f'Verified {path.name}, {len(test):,} rows', flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--export', choices=RECIPES)
    args = cli.parse_args()
    export(args.export) if args.export else development()
