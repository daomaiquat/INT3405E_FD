"""Fit the three Grandmaster EDA branches on disjoint development partitions."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import lightgbm as lgb
import numpy as np
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = HERE.parent
sys.path[:0] = [str(SOURCE), str(SOURCE / 'followup'), str(SOURCE / 'robust_validation'),
                str(REPO / 's6e11_gpu' / 'plan_runs'), str(REPO / 'research' / 'novel_model_experiments')]

from alternative_model import SplineDesign, fit_logistic
from features import FeatureBuilder
from grandmaster_inputs import (archive_functions, design, development_caches, digest,
                                full_caches, official_data, read_json, write_json)
from residual_models import nested_residual_train
from robust_density import isolated_meta
from run_followup import fit_xgb

BRANCHES = ('hierarchical', 'xgb', 'lightgbm', 'spline')
WEIGHTS = np.array([0., .85, .10, .05])
SEED = 6271
PARTS = 3


def partitions(y):
    return StratifiedKFold(PARTS, shuffle=True, random_state=SEED).split(y, y)


def xgb_branch(builder, train, fit, query, y, p, q, pairs, device):
    grow, score = builder.make(fit, (query,), groups=('raw', 'te13', 'cents', 'digits'))
    eda_grow, eda_score = nested_residual_train(train, fit, query, y, p, pairs)
    return fit_xgb(np.column_stack((grow, eda_grow)), y, np.column_stack((score, eda_score)), p, q, device)


def lightgbm_branch(builder, fit, query, y, config):
    grow, score = builder.make(fit, (query,), groups=('raw', 'te24', 'cents', 'digits'))
    booster = lgb.train(config['params'], lgb.Dataset(grow, label=y), num_boost_round=config['rounds'])
    return logit(np.clip(booster.predict(score, num_threads=config['params']['num_threads']), 1e-12, 1 - 1e-12))


def spline_branch(fit, query, y):
    with threadpool_limits(limits=4):
        features = SplineDesign().fit(fit, y)
        additive = features.additive(fit)
        control = fit_logistic(additive, y)
        pairs = features.screen(fit, y, control.predict_proba(additive)[:, 1])
        with_pairs = features.include_pairs(fit, additive, pairs)
        del additive
        model = fit_logistic(with_pairs, y, control)
        del with_pairs
        query_design = features.include_pairs(query, features.additive(query), pairs)
        probability = model.predict_proba(query_design)[:, 1]
    return logit(np.clip(probability, 1e-12, 1 - 1e-12))


def run(mode, device):
    started = perf_counter()
    train, test, _, hashes = official_data()
    if mode == 'development':
        y, rows, _, _, raw, _, dense, _, _ = development_caches(train, hashes)
    else:
        y = train.is_fraud.to_numpy(dtype=np.uint8)
        rows = np.arange(len(y))
        _, raw, _, dense, _, _ = full_caches(train, test, hashes)
    ns = archive_functions()
    stack = design(ns, raw, dense)
    config = read_json(SOURCE / 'lightgbm_metrics.json')
    screen = read_json(SOURCE / 'residual_development_metrics.json')
    if config['source'] != hashes or screen['input_sha256'] != hashes:
        raise ValueError('Training recipes must match official inputs')
    pairs = [tuple(x) for x in screen['selected_residual_pairs']]
    builder = FeatureBuilder(train, test)
    frame = train.drop(columns='is_fraud')
    labels = y[rows]
    all_scores = np.full((len(rows), len(BRANCHES)), np.nan)
    all_folds = np.zeros(len(rows), dtype=np.uint8)
    details = []
    for number, (fit, query) in enumerate(partitions(labels), start=1):
        start = perf_counter()
        p, q = isolated_meta(ns, stack[fit], stack[query], labels[fit])
        train_rows, query_rows = rows[fit], rows[query]
        xgb = xgb_branch(builder, frame, train_rows, query_rows, labels[fit], p, q, pairs, device)
        print(f'Fold {number} XGBoost done at {perf_counter() - start:.1f}s', flush=True)
        lightgbm = lightgbm_branch(builder, train_rows, query_rows, labels[fit], config)
        print(f'Fold {number} LightGBM done at {perf_counter() - start:.1f}s', flush=True)
        spline = spline_branch(frame.iloc[train_rows], frame.iloc[query_rows], labels[fit])
        scores = np.column_stack((q, xgb, lightgbm, spline))
        if not np.isfinite(scores).all() or np.any(all_folds[query]):
            raise ValueError('Nonfinite predictions or duplicate validation rows')
        all_scores[query], all_folds[query] = scores, number
        details.append({'fold': number, 'fit_rows': len(fit), 'scored_rows': len(query),
                        'seconds': perf_counter() - start})
        print(f'Fold {number} done', details[-1], flush=True)

    if not np.isfinite(all_scores).all() or np.any(all_folds == 0):
        raise ValueError('Incomplete out-of-fold scores')
    path = HERE / f'{mode}_oof.npz'
    staging = path.with_suffix('.tmp.npz')
    np.savez_compressed(staging, ids=train.id.to_numpy()[rows], fold=all_folds, scores=all_scores)
    staging.replace(path)
    write_json(HERE / f'{mode}_oof_manifest.json', {
        'input_sha256': hashes, 'sha256': digest(path), 'branch_order': list(BRANCHES), 'weights': WEIGHTS.tolist(),
        'seed': SEED, 'folds': details, 'rows': len(rows), 'seconds': perf_counter() - started,
        'scope': 'Second-stage branch predictions, using cached Grandmaster base inputs and fixed EDA pairs',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'full'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    options = parser.parse_args()
    run(options.mode, options.device)
