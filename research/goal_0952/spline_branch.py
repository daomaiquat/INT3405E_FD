"""Score the existing smooth tensor spline on the Grandmaster holdout and test."""

import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import logit
from threadpoolctl import threadpool_limits

from grandmaster_inputs import HERE, PROJECT, development_caches, official_data, write_json


def run():
    sys.path.insert(0, str(PROJECT / 'novel_model_experiments'))
    from alternative_model import SplineDesign, fit_logistic

    train, test, _, hashes = official_data()
    y, dev, held, *_ = development_caches(train, hashes)
    details = {}
    with threadpool_limits(limits=4):
        for name, rows, destination in (('development', dev, train.iloc[held]),
                                        ('full', np.arange(len(train)), test)):
            started = perf_counter()
            fit_rows = train.iloc[rows]
            labels = y[rows]
            features = SplineDesign().fit(fit_rows, labels)
            design = features.additive(fit_rows)
            control = fit_logistic(design, labels)
            pairs = features.screen(fit_rows, labels, control.predict_proba(design)[:, 1])
            fit_design = features.include_pairs(fit_rows, design, pairs)
            del design
            model = fit_logistic(fit_design, labels, control)
            del fit_design
            query_design = features.include_pairs(destination, features.additive(destination), pairs)
            predicted = model.predict_proba(query_design)[:, 1].astype(np.float64)
            if not np.isfinite(predicted).all():
                raise RuntimeError('Nonfinite spline predictions')
            np.savez_compressed(HERE / f'spline_{name}.npz', query_ids=destination.id.to_numpy(),
                                query_logits=logit(np.clip(predicted, 1e-12, 1 - 1e-12)))
            details[name] = {'rows_fit': len(rows), 'iterations': int(model.n_iter_[0]),
                             'control_iterations': int(control.n_iter_[0]),
                             'seconds': perf_counter() - started,
                             'interactions': [[float(value), left, right] for value, left, right in pairs]}
            print(name, details[name], flush=True)
    write_json(HERE / 'spline_metrics.json', {'input_sha256': hashes, **details})


if __name__ == '__main__':
    run()
