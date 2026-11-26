"""Describe where the peer-profile candidate helps at fixed review budgets."""

import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss

from run_signals import HERE, PEERS, development_caches, official_data, peer_features, read_scores, write_json


def budget_summary(y, scores, fraction):
    size = int(np.ceil(len(scores) * fraction))
    ordered = np.argsort(-scores, kind='stable')
    hits = int(y[ordered[:size]].sum())
    return {'review_rows': size, 'fraud': hits, 'precision': hits / size, 'recall': hits / int(y.sum())}


def run():
    train, test, _, hashes = official_data()
    y, fit, held, *_ = development_caches(train, hashes)
    ids = train.id.to_numpy()[held]
    labels = y[held]
    scores = read_scores(HERE / 'holdout_predictions.npz', ids, 'ids', ('current', 'peer_profiles'))
    prob = {name: expit(values) for name, values in scores.items()}
    data = train.iloc[fit]
    control = peer_features(data, y[fit], data)[:, 3::4]
    validation = peer_features(data, y[fit], train.iloc[held])[:, 3::4]
    testing = peer_features(data, y[fit], test)[:, 3::4]
    thresholds = np.quantile(control, [.1, .9], axis=0)
    rows = []
    for i, (variable, context) in enumerate(PEERS):
        for tail, rule, limit in [('lower', np.less_equal, thresholds[0, i]), ('upper', np.greater_equal, thresholds[1, i])]:
            selected = rule(validation[:, i], limit)
            if not selected.any():
                continue
            original = log_loss(labels[selected], prob['current'][selected], labels=[0, 1])
            candidate = log_loss(labels[selected], prob['peer_profiles'][selected], labels=[0, 1])
            rows.append({'variable': variable, 'context': context, 'tail': tail, 'quantile_cut': float(limit),
                         'held_rows': int(selected.sum()), 'held_share': float(selected.mean()),
                         'test_share': float(rule(testing[:, i], limit).mean()),
                         'observed_fraud': int(labels[selected].sum()),
                         'expected_current': float(prob['current'][selected].sum()),
                         'expected_candidate': float(prob['peer_profiles'][selected].sum()),
                         'delta_logloss': float(candidate-original)})
    budgets = {str(fraction): {name: budget_summary(labels, values, fraction) for name, values in scores.items()}
               for fraction in (.01, .05, .10)}
    write_json(HERE / 'peer_diagnostics.json', {'input_sha256': hashes, 'threshold_fit_rows': len(fit),
                                                'held_rows': len(held), 'tails': rows, 'review_budgets': budgets})
    print(budgets, flush=True)
    for row in sorted(rows, key=lambda record: record['delta_logloss'])[:4]:
        print(row, flush=True)


if __name__ == '__main__':
    run()
