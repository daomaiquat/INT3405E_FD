"""Learn a centered exact-hours log-odds adjustment over the frozen local FINAL model."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
GPU_RUNS = PROJECT / 's6e11_gpu'
HOURS = 'hours_since_last_txn'
RADIUS = 20
STRENGTH = 12
SHRINKAGE = 0.5
MAX_STEP = 0.35


def fit_adjustments(ticks, labels, probabilities):
    """One ridge-shrunk Newton step relative to adjacent residual risk."""
    observed = ticks >= 0
    ticks, p = ticks[observed], probabilities[observed]
    gradient = np.bincount(ticks, weights=labels[observed] - p)
    information = np.bincount(ticks, weights=p * (1 - p), minlength=gradient.size)
    index = np.arange(gradient.size)
    low = np.maximum(index - RADIUS, 0)
    high = np.minimum(index + RADIUS + 1, len(index))

    def neighbor_sum(values):
        prefix = np.r_[0, np.cumsum(values)]
        return prefix[high] - prefix[low] - values

    parent = neighbor_sum(gradient) / (neighbor_sum(information) + 20)
    correction = (gradient - information * parent) / (information + STRENGTH)
    return SHRINKAGE * np.clip(correction, -MAX_STEP, MAX_STEP)


def lookup(table, ticks):
    eligible = (ticks >= 0) & (ticks < len(table))
    values = np.zeros(len(ticks))
    values[eligible] = table[ticks[eligible]]
    return values


def tick_values(path):
    values = pd.read_csv(path, usecols=[HOURS])[HOURS].to_numpy()
    return np.where(np.isfinite(values) & (values >= 0), np.rint(100 * values), -1).astype(np.int64)


def replay_stack(scores, state):
    scores = np.clip(np.asarray(scores, dtype=np.float32), 1e-6, 1 - 1e-6)
    log_odds = np.log(scores / (1 - scores))
    standardized = (log_odds - state['center']) / state['scale']
    return expit(standardized @ state['coefficients'] + state['intercept'])


def evaluate(labels, probabilities):
    return {'roc_auc': roc_auc_score(labels, probabilities),
            'average_precision': average_precision_score(labels, probabilities),
            'log_loss': log_loss(labels, probabilities)}


def run():
    # Reuse the earlier experiment's audited loader and nested archive contract.
    prior_experiment = GPU_RUNS / 'final_feature_optimization'
    sys.path.insert(0, str(prior_experiment))
    from evaluate_final import MEMBERS, NESTED, REFERENCE, load_data_contract, nested_inner_scores, read_oof

    manifest, ids, test_ids, labels, folds, n_folds = load_data_contract()
    frozen = json.loads((prior_experiment / 'results' / 'frozen_reference.json').read_text(encoding='utf-8'))
    config = json.loads((NESTED / 'config.json').read_text(encoding='utf-8'))
    if frozen['nested_config'] != config or config['members'] != list(MEMBERS):
        raise ValueError('Recorded nested stack differs from its archived configuration')
    base = []
    spec_hashes = []
    for name in MEMBERS:
        spec = json.loads((REFERENCE / name / 'config.json').read_text(encoding='utf-8'))
        spec_hashes.append(hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest())
        base.append(read_oof(REFERENCE / name, manifest, ids, folds, name, test_ids)[0])
    if [hashlib.sha256(member.tobytes()).hexdigest() for member in base] != config['base_prediction_hashes']:
        raise ValueError('Historical member predictions differ from the recorded configuration')

    final, _ = read_oof(REFERENCE / 'final', manifest, ids, folds, 'FINAL', test_ids)
    with np.load(REFERENCE / 'final' / 'predictions.npz', allow_pickle=False) as file:
        final_test = file['test']
    hours = tick_values(PROJECT.parent / 'data' / 'train.csv')
    oof = np.empty_like(final)
    replay_errors = []
    base = np.column_stack(base)
    for fold in range(n_folds):
        fit = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        record = frozen['meta_folds'][fold]
        if record['fold'] != fold + 1 or record['train_rows'] != len(fit):
            raise ValueError(f'Historical meta state differs from fold {fold + 1}')
        state = record['model_state']
        reference_prob = replay_stack(nested_inner_scores(ids, labels, fit, fold + 1, config, spec_hashes), state)
        replay_error = float(np.max(np.abs(replay_stack(base[held], state) - final[held])))
        if replay_error > 1e-9:
            raise ValueError(f'Could not replay archived FINAL fold {fold + 1}: {replay_error}')
        replay_errors.append(replay_error)
        effects = fit_adjustments(hours[fit], labels[fit], reference_prob)
        oof[held] = expit(logit(final[held]) + lookup(effects, hours[held]))

    effects = fit_adjustments(hours, labels, final)
    test = expit(logit(final_test) + lookup(effects, tick_values(PROJECT.parent / 'data' / 'test.csv')))
    sample = pd.read_csv(PROJECT.parent / 'data' / 'sample_submission.csv')
    if list(sample.columns) != ['id', 'is_fraud'] or not np.array_equal(test_ids, sample.id):
        raise ValueError('Test IDs or submission schema mismatch')
    if not np.isfinite(oof).all() or not np.isfinite(test).all():
        raise ValueError('Probabilities must be finite')
    results = HERE / 'results'
    results.mkdir(exist_ok=True)
    sample.assign(is_fraud=test).to_csv(results / 'submission_hours_residual.csv', index=False)
    np.savez_compressed(results / 'hours_residual_predictions.npz', ids=ids, fold=folds, oof=oof,
                        test_ids=test_ids, test=test)
    summary = {**evaluate(labels, oof), 'baseline': evaluate(labels, final),
               'radius_in_centihours': RADIUS, 'ridge_strength': STRENGTH, 'shrinkage': SHRINKAGE,
               'unshrunk_step_cap': MAX_STEP, 'max_final_replay_errors': replay_errors,
               'test_rows': len(test), 'max_test_adjustment': float(np.max(np.abs(test - final_test)))}
    (results / 'hours_residual_metrics.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
