"""Check whether one extra tree column helps beyond a within-risk placebo."""

import numpy as np

from risk_eda import CONTROL, HERE, XGB_WEIGHT, encoded_views, get_experiment
from evaluate_and_export import assess
from grandmaster_inputs import read_json, write_json
from run_followup import build_training_view, fit_xgb


def shuffle_in_bands(values, offset, edges, rng):
    result = np.empty_like(values)
    bands = np.searchsorted(edges, offset)
    for band in np.unique(bands):
        rows = np.flatnonzero(bands == band)
        result[rows] = values[rng.permutation(rows)]
    return result


def audit():
    context, (x, z), (fitting, scoring), _ = build_training_view('development')
    ns, _, _, _, hashes, frame, labels, fit, query, p, q, archived, baseline = context
    methods, encoder, pairs = get_experiment('extra', hashes)
    a, b = encoded_views(fitting, scoring, labels[fit], p, q, pairs, encoder)
    edges = np.quantile(p, np.arange(1, 20) / 20.)
    rng = np.random.default_rng(2026)
    shuffled_train = shuffle_in_bands(a, p, edges, rng)
    shuffled_query = shuffle_in_bands(b, q, edges, rng)
    control = fit_xgb(x, labels[fit], z, p, q, 'cuda')
    np.testing.assert_allclose(control, archived, atol=4e-6, rtol=0)
    placebo = fit_xgb(np.column_stack((x, shuffled_train)), labels[fit], np.column_stack((z, shuffled_query)),
                      p, q, 'cuda')
    placebo_score = baseline + XGB_WEIGHT * (placebo - control)
    with np.load(HERE / 'extra_development_predictions.npz', allow_pickle=False) as actual:
        np.testing.assert_array_equal(actual['ids'], frame.id.to_numpy()[query])
        np.testing.assert_array_equal(actual[CONTROL], baseline)
        real = actual[methods[0]]
    report = {'input_sha256': hashes, 'rows': len(query), 'shuffle_seed': 2026, 'risk_bands': 20,
              'placebo_vs_control': assess(ns, labels[query], placebo_score, baseline),
              'real_vs_placebo': assess(ns, labels[query], real, placebo_score),
              'real_vs_control': read_json(HERE / 'extra_development_metrics.json')['variants'][methods[0]]}
    write_json(HERE / 'extra_placebo_metrics.json', report)
    print(f'Placebo versus original: {report["placebo_vs_control"]}')
    print(f'Real versus placebo: {report["real_vs_placebo"]}')


if __name__ == '__main__':
    audit()
