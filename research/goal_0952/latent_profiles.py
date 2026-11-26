"""Fit a regularized mixture of discrete transaction profiles per class."""

import argparse
from time import perf_counter

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
import torch

from grandmaster_inputs import HERE, development_caches, official_data, write_json
from residual_models import fit_codes


COMPONENTS = 12
MAX_NEGATIVE = 160_000
ITERATIONS = 26
BATCH_ROWS = 32_768


def prepare(fitting, scoring):
    blocks_fit, blocks_query, widths = [], [], []
    fields = [name for name in fitting if name not in {'id', 'is_fraud'}]
    for name in fields:
        train, query, count = fit_codes(fitting[name], scoring[name])
        blocks_fit.append(train)
        blocks_query.append(query)
        widths.append(count)
    widths = np.array(widths)
    offsets = np.r_[0, np.cumsum(widths)[:-1]]
    return tuple(np.column_stack(parts) + offsets for parts in (blocks_fit, blocks_query)), widths


def component_log_likelihood(keys, weights, emission):
    scores = weights.expand(len(keys), -1).clone()
    for field in range(keys.shape[1]):
        scores += emission[keys[:, field]]
    return scores


def fit_profiles(keys, widths, strength, device, seed):
    rng = np.random.default_rng(seed)
    all_bins = int(widths.sum())
    keys = torch.as_tensor(keys, dtype=torch.long, device=device)
    mass = torch.bincount(keys.flatten(), minlength=all_bins).float() + .5
    widths_tensor = torch.as_tensor(widths, device=device)
    group = torch.repeat_interleave(torch.arange(len(widths), device=device), widths_tensor)
    base = mass / torch.zeros(len(widths), device=device).index_add_(0, group, mass)[group]
    priors = strength * base[:, None]
    emission = (base[:, None] * 0.40).repeat(1, COMPONENTS)
    center = keys[rng.choice(len(keys), size=COMPONENTS, replace=False)]
    emission.scatter_add_(0, center.T, torch.full(center.T.shape, .60, device=device))
    log_weights = torch.full((1, COMPONENTS), -np.log(COMPONENTS), device=device)
    for _ in range(ITERATIONS):
        counts = torch.zeros_like(emission) + priors
        component_mass = torch.full((COMPONENTS,), 5.0, device=device)
        log_emission = emission.log()
        for batch in keys.split(BATCH_ROWS):
            soft = torch.softmax(component_log_likelihood(batch, log_weights, log_emission), dim=1)
            component_mass += soft.sum(dim=0)
            for field in range(batch.shape[1]):
                counts.index_add_(0, batch[:, field], soft)
        normalizers = torch.zeros((len(widths), COMPONENTS), device=device).index_add_(0, group, counts)
        emission = counts / normalizers[group]
        log_weights = (component_mass / component_mass.sum()).log()[None, :]
    return log_weights, emission.log()


def score_profiles(keys, model, device):
    scores = np.empty(len(keys))
    for start in range(0, len(keys), BATCH_ROWS):
        batch = torch.as_tensor(keys[start:start + BATCH_ROWS], dtype=torch.long, device=device)
        likelihood = torch.logsumexp(component_log_likelihood(batch, *model), dim=1)
        scores[start:start + len(batch)] = likelihood.cpu().numpy()
    return scores


def raw_margin(fit, labels, query, device, seed):
    (x_train, x_query), widths = prepare(fit, query)
    rr = np.random.default_rng(seed)
    raw = np.zeros(len(query))
    for target, strength in ((0, 140), (1, 40)):
        rows = np.flatnonzero(labels == target)
        if target == 0 and len(rows) > MAX_NEGATIVE:
            rows = rr.choice(rows, MAX_NEGATIVE, replace=False)
        model = fit_profiles(x_train[rows], widths, strength, device, seed + target)
        values = score_profiles(x_query, model, device)
        raw += values if target == 1 else -values
    return raw + logit(labels.mean())


def run(mode):
    started = perf_counter()
    train, test, _, hashes = official_data()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the mixture fit')
    device = torch.device('cuda')
    y = train.is_fraud.to_numpy(dtype=np.uint8)
    if mode == 'development':
        _, source, target, *_ = development_caches(train, hashes)
        query = train.iloc[target]
    else:
        source = np.arange(len(train))
        query = test
    fit = train.iloc[source]
    labels = y[source]
    calibration_fit, calibration_valid = next(StratifiedShuffleSplit(1, test_size=.25, random_state=19413)
                                              .split(labels, labels))
    calibration = raw_margin(fit.iloc[calibration_fit], labels[calibration_fit], fit.iloc[calibration_valid],
                             device, 19413)
    calibrator = LogisticRegression(C=1).fit(calibration[:, None], labels[calibration_valid])
    margin = raw_margin(fit, labels, query, device, 19413)
    scores = calibrator.decision_function(margin[:, None])
    if not np.isfinite(scores).all():
        raise ValueError('Invalid latent profile scores')
    np.savez_compressed(HERE / f'latent_{mode}.npz', query_ids=query.id.to_numpy(), query_logits=scores)
    metrics = {'input_sha256': hashes, 'components': COMPONENTS, 'iterations': ITERATIONS,
               'negative_subsample_cap': MAX_NEGATIVE, 'raw_calibration_auc': roc_auc_score(labels[calibration_valid], calibration),
               'calibration_slope': float(calibrator.coef_[0, 0]), 'elapsed_seconds': perf_counter() - started}
    if mode == 'development':
        metrics['holdout_auc'] = roc_auc_score(y[target], scores)
    write_json(HERE / f'latent_{mode}_metrics.json', metrics)
    print(metrics, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'full'))
    run(parser.parse_args().mode)
