"""Rebuild the project's BCE MLP on partitions compatible with Grandmaster."""

import argparse
import sys
from time import perf_counter

import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from grandmaster_inputs import HERE, PROJECT, development_caches, digest, official_data, splits_for, write_json


def run(mode):
    sys.path.insert(0, str(PROJECT / 's6e11_gpu' / 'imbalance'))
    from run_mlp import EPOCHS, cuda_device, fit_fold
    import torch

    sys.path.insert(0, str(PROJECT.parent / 's6e11_gpu' / 'plan_runs'))
    from features import FeatureBuilder

    started = perf_counter()
    train, test, _, hashes = official_data()
    y = train.is_fraud.to_numpy(dtype=np.uint8)
    if mode == 'development':
        _, selection, held, folds, *_ = development_caches(train, hashes)
        query = held
        query_ids = train.id.to_numpy()[held]
    else:
        selection = np.arange(len(train))
        folds = splits_for(y)
        query = len(train) + np.arange(len(test))
        query_ids = test.id.to_numpy()
    destination = HERE / f'mlp_{mode}.npz'
    device, device_name = cuda_device()
    torch.set_float32_matmul_precision('high')
    builder = FeatureBuilder(train, test)
    settings = {'objective': 'bce', 'focal_gamma': None, 'model_seed': 42, 'epochs': EPOCHS, 'name': 'MLP-BCE'}
    out = np.empty(len(selection))
    score = np.zeros(len(query))
    fit_records = []
    with threadpool_limits(limits=4):
        for i, (fitting, validation) in enumerate(folds, start=1):
            fit_rows = selection[fitting]
            valid_rows = selection[validation]
            xfit, xvalid, xquery = builder.make(fit_rows, (valid_rows, query),
                                               groups=('raw', 'te13', 'cents', 'digits'))
            scaler = StandardScaler().fit(xfit)
            xfit, xvalid, xquery = (
                torch.as_tensor(scaler.transform(values), dtype=torch.float32, device=device)
                for values in (xfit, xvalid, xquery)
            )
            (p_valid, p_query), fit_details = fit_fold(settings, i, device, xfit, y[fit_rows], xvalid, xquery,
                                                       device_name)
            out[validation] = logit(np.clip(p_valid, 1e-12, 1 - 1e-12))
            score += logit(np.clip(p_query, 1e-12, 1 - 1e-12)) / len(folds)
            fit_records.append(fit_details)
            print(f'{mode} fold {i}/10 AUC {roc_auc_score(y[valid_rows], p_valid):.8f}', flush=True)
    if not np.isfinite(out).all() or not np.isfinite(score).all():
        raise ValueError('Non-finite MLP scores')
    with destination.with_suffix('.tmp').open('wb') as file:
        np.savez_compressed(file, ids=train.id.to_numpy()[selection], query_ids=query_ids, oof_logits=out,
                            query_logits=score)
    destination.with_suffix('.tmp').replace(destination)
    write_json(destination.with_suffix('.json'), {'input_sha256': hashes, 'model': 'MLP-BCE',
                                                   'source_sha256': digest(PROJECT / 's6e11_gpu' / 'imbalance' / 'run_mlp.py'),
                                                   'logits_averaged': True, 'roc_auc_oof': roc_auc_score(y[selection], out),
                                                   'elapsed_seconds': perf_counter() - started, 'fits': fit_records})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('development', 'full'))
    run(parser.parse_args().mode)
