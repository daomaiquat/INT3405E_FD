"""Re-fit the shallow D03 LightGBM branch for the Grandmaster comparison."""

import sys
from time import perf_counter

import lightgbm as lgb
import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from grandmaster_inputs import HERE, PROJECT, development_caches, official_data, write_json


def run():
    sys.path.insert(0, str(PROJECT.parent / 's6e11_gpu' / 'plan_runs'))
    from features import FeatureBuilder

    train, test, _, hashes = official_data()
    y, dev, held, *_ = development_caches(train, hashes)
    builder = FeatureBuilder(train, test)
    params = {'objective': 'binary', 'learning_rate': .05, 'num_leaves': 4, 'min_data_in_leaf': 50,
              'bagging_fraction': .90, 'bagging_freq': 1, 'feature_fraction': .90,
              'lambda_l2': 1.0, 'max_bin': 255, 'force_col_wise': True, 'deterministic': True,
              'seed': 42, 'num_threads': 8, 'verbosity': -1}
    rounds = 1600
    modes = (('development', dev, held, train.id.to_numpy()[held]),
             ('full', np.arange(len(train)), len(train) + np.arange(len(test)), test.id.to_numpy()))
    summaries = {}
    for name, fit, query, ids in modes:
        start = perf_counter()
        x_fit, x_query = builder.make(fit, (query,), groups=('raw', 'te24', 'cents', 'digits'))
        model = lgb.train(params, lgb.Dataset(x_fit, label=y[fit]), num_boost_round=rounds)
        scores = logit(np.clip(model.predict(x_query, num_threads=8), 1e-12, 1 - 1e-12))
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite LightGBM scores')
        np.savez_compressed(HERE / f'lightgbm_{name}.npz', query_ids=ids, query_logits=scores)
        summary = {'training_rows': len(fit), 'elapsed_seconds': perf_counter() - start}
        if name == 'development':
            summary['holdout_auc'] = roc_auc_score(y[query], scores)
        summaries[name] = summary
        print(name, summary, flush=True)
    write_json(HERE / 'lightgbm_metrics.json', {'params': params, 'rounds': rounds, 'source': hashes, **summaries})


if __name__ == '__main__':
    run()
