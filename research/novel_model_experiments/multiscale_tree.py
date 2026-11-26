"""Compare two fold-local feature families on the official S6E11 data."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUTER_SEED = 42
FEATURE_MODULE = PROJECT.parent / 's6e11_gpu' / 'plan_runs'
BASELINE_GROUPS = ('raw', 'te13', 'cents', 'digits')
TARGET = 'is_fraud'

# Units (ticks per whole unit), near radius and wider radius in ticks.
NEIGHBORHOODS = {
    'amount_usd': (100, 20, 200),
    'hours_since_last_txn': (100, 4, 25),
    'distance_from_home_km': (100, 10, 100),
    'account_balance_usd': (100, 100, 1200),
    'velocity_score': (100, 10, 150),
    'merchant_risk_score': (10, 4, 30),
}

QUANTILES = (0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80,
             0.90, 0.95, 0.975, 0.99)

PAIR_GROUPS = {
    'amount_usd': ('merchant_risk_score', 'velocity_score', 'hours_since_last_txn',
                   'channel', 'auth_method', 'merchant_category', 'account_balance_usd'),
    'velocity_score': ('merchant_risk_score', 'hours_since_last_txn',
                       'txn_count_last_24h', 'auth_method', 'cvv_retry_count'),
    'distance_from_home_km': ('is_foreign_transaction', 'ip_country_mismatch', 'used_vpn',
                              'channel', 'auth_method', 'account_balance_usd'),
    'merchant_risk_score': ('auth_method', 'channel', 'merchant_category', 'prior_disputes'),
    'hours_since_last_txn': ('txn_count_last_24h', 'auth_method'),
    'account_balance_usd': ('card_age_months',),
}


def numeric_ticks(series, scale):
    values = series.to_numpy(dtype=np.float64, na_value=np.nan)
    return np.where(np.isfinite(values), np.rint(values * scale), -10**12).astype(np.int64)


def local_contrasts(fit, labels, query, near, wide):
    keys, codes = np.unique(fit, return_inverse=True)
    counts = np.bincount(codes)
    positives = np.bincount(codes, weights=labels)
    n_prefix = np.r_[0, counts.cumsum()]
    y_prefix = np.r_[0, positives.cumsum()]

    left, right = np.searchsorted(keys, query), np.searchsorted(keys, query, side='right')
    exact_n, exact_y = n_prefix[right] - n_prefix[left], y_prefix[right] - y_prefix[left]

    def nearby_counts(radius):
        lower = np.searchsorted(keys, query - radius)
        upper = np.searchsorted(keys, query + radius, side='right')
        return n_prefix[upper] - n_prefix[lower] - exact_n, y_prefix[upper] - y_prefix[lower] - exact_y

    near_n, near_y = nearby_counts(near)
    wide_n, wide_y = nearby_counts(wide)
    prior = (labels.sum() + 0.5) / (len(labels) + 1)
    broad = (wide_y + 100 * prior) / (wide_n + 100)
    outside = (wide_y - near_y + 100 * prior) / (wide_n - near_n + 100)
    narrow = (near_y + 50 * outside) / (near_n + 50)
    exact = (exact_y + 20 * narrow) / (exact_n + 20)
    return np.column_stack((logit(narrow), logit(broad), logit(exact) - logit(narrow),
                            np.log1p(exact_n))).astype(np.float32)


def neighborhood_features(reference, labels, query):
    return np.column_stack([
        local_contrasts(numeric_ticks(reference[name], scale), labels,
                        numeric_ticks(query[name], scale), near, wide)
        for name, (scale, near, wide) in NEIGHBORHOODS.items()
    ])


def copula_table(negative, positive):
    """Log likelihood ratio for dependence alone, with an independence prior."""
    def log_dependence(counts, strength):
        row, col = counts.sum(axis=1) + 0.5, counts.sum(axis=0) + 0.5
        independent = np.outer(row / row.sum(), col / col.sum())
        joint = (counts + strength * independent) / (counts.sum() + strength)
        return np.log(joint) - np.log(independent)

    return np.clip(log_dependence(positive, 100) - log_dependence(negative, 200), -3, 3)


def copula_features(reference, labels, query):
    fields = set(PAIR_GROUPS).union(*PAIR_GROUPS.values())
    codes = {}
    for name in fields:
        if name in NEIGHBORHOODS or name == 'card_age_months':
            fit_values = reference[name].to_numpy(dtype=np.float64, na_value=np.nan)
            test_values = query[name].to_numpy(dtype=np.float64, na_value=np.nan)
            finite = fit_values[np.isfinite(fit_values)]
            cuts = np.unique(np.quantile(finite, QUANTILES)) if finite.size else []
            codes[name] = (
                np.where(np.isfinite(fit_values), np.searchsorted(cuts, fit_values) + 1, 0),
                np.where(np.isfinite(test_values), np.searchsorted(cuts, test_values) + 1, 0),
                len(cuts) + 2,
            )
        else:
            levels = pd.Index(reference[name].dropna().unique())
            codes[name] = (levels.get_indexer(reference[name]) + 1,
                           levels.get_indexer(query[name]) + 1, len(levels) + 1)

    features = []
    for left, partners in PAIR_GROUPS.items():
        l_fit, l_query, l_size = codes[left]
        for right in partners:
            r_fit, r_query, r_size = codes[right]
            cells = l_fit * r_size + r_fit
            neg = np.bincount(cells[labels == 0], minlength=l_size * r_size).reshape(l_size, r_size)
            pos = np.bincount(cells[labels == 1], minlength=l_size * r_size).reshape(l_size, r_size)
            table = copula_table(neg, pos)
            table[neg + pos < 5] = 0
            features.append(table[l_query, r_query])
    return np.column_stack(features).astype(np.float32)


def cross_fit(encoder, frame, labels, fit, queries):
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    output = encoder(frame.iloc[fit], labels[fit], frame.iloc[queries])
    training = np.empty((len(fit), output.shape[1]), dtype=np.float32)
    for ref, hold in inner.split(fit, labels[fit]):
        rows = fit[ref]
        training[hold] = encoder(frame.iloc[rows], labels[rows], frame.iloc[fit[hold]])
    return training, output


def prepare_builder(train, test):
    sys.path.insert(0, str(FEATURE_MODULE))
    from features import FeatureBuilder

    return FeatureBuilder(train, test)


def fit_tree(train, labels, query, rounds, device, fold):
    settings = {
        'objective': 'binary:logistic', 'tree_method': 'hist', 'device': device,
        'max_depth': 2, 'min_child_weight': 5, 'learning_rate': 0.04,
        'subsample': 0.90, 'colsample_bytree': 0.90, 'reg_lambda': 1,
        'max_bin': 256, 'nthread': 4, 'seed': 42 + fold,
    }
    matrix = xgb.QuantileDMatrix(train, label=labels, max_bin=256, nthread=4)
    model = xgb.train(settings, matrix, num_boost_round=rounds)
    actual = json.loads(model.save_config())['learner']['generic_param']['device']
    if device.startswith('cuda') and not actual.startswith('cuda'):
        raise RuntimeError(f'XGBoost fell back to {actual}')
    if device.startswith('cuda'):
        # DMatrix prediction avoids inplace CPU/GPU device mismatch and keeps exact splits.
        return model.predict(xgb.DMatrix(query, nthread=4))
    return model.inplace_predict(query)


def metrics(labels, predictions):
    return {
        'roc_auc': float(roc_auc_score(labels, predictions)),
        'average_precision': float(average_precision_score(labels, predictions)),
        'log_loss': float(log_loss(labels, predictions)),
    }


def read_inputs():
    train = pd.read_csv(PROJECT.parent / 'data' / 'train.csv')
    test = pd.read_csv(PROJECT.parent / 'data' / 'test.csv')
    sample = pd.read_csv(PROJECT.parent / 'data' / 'sample_submission.csv')
    if list(sample.columns) != ['id', TARGET] or not np.array_equal(sample.id, test.id):
        raise ValueError('test IDs or sample submission header do not match')
    return train, test, sample


def save_result(output, name, ids, labels, folds, sample, oof, test_folds, folds_ran, rounds, elapsed, device):
    test = np.column_stack(test_folds)
    predictions = test.mean(axis=1, dtype=np.float64)
    complete = len(folds_ran) == 5
    valid = np.isfinite(oof)
    result = {
        'model': name, 'complete_5_fold_oof': complete, 'outer_seed': OUTER_SEED,
        'inner_seed': 2026, 'rounds': rounds, 'device': device, 'elapsed_seconds': elapsed,
        'folds_ran': folds_ran, 'fold_metrics': [metrics(labels[folds == f], oof[folds == f]) for f in folds_ran],
        'oof': metrics(labels[valid], oof[valid]), 'oof_rows': int(valid.sum()),
    }
    if complete:
        if not valid.all() or not np.isfinite(test).all() or ((test < 0) | (test > 1)).any():
            raise ValueError('nonfinite or out-of-range predictions')
        sample.assign(**{TARGET: predictions}).to_csv(output / f'submission_{name}.csv', index=False)
        np.savez_compressed(output / f'{name}_predictions.npz', ids=ids, fold=folds,
                            oof=oof, test_ids=sample.id.to_numpy(), test=predictions, test_by_fold=test)
    (output / f'{name}_metrics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


def run(output, names, folds_to_run, rounds, device):
    encoders = {'control': None, 'neighborhood': neighborhood_features, 'copula': copula_features}
    train, test, sample = read_inputs()
    labels = train[TARGET].to_numpy(dtype=np.uint8)
    ids = train.id.to_numpy()
    frame = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    builder = prepare_builder(train, test)
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED).split(ids, labels))
    folds = np.empty(len(train), dtype=np.int8)
    for f, (_, hold) in enumerate(splits):
        folds[hold] = f
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    oof = {name: np.full(len(train), np.nan) for name in names}
    test_folds = {name: [] for name in names}
    for fold in folds_to_run:
        fold_started = perf_counter()
        fit, hold = splits[fold]
        query = np.r_[hold, len(train) + np.arange(len(test))]
        fit_base, query_base = builder.make(fit, (query,), groups=BASELINE_GROUPS)
        print(f'fold {fold + 1}: D03 features {perf_counter() - fold_started:.1f}s', flush=True)
        for name in names:
            start = perf_counter()
            encoder = encoders[name]
            if encoder:
                fit_extra, query_extra = cross_fit(encoder, frame, labels, fit, query)
                fit_x = np.column_stack((fit_base, fit_extra))
                query_x = np.column_stack((query_base, query_extra))
            else:
                fit_x, query_x = fit_base, query_base
            prediction = fit_tree(fit_x, labels[fit], query_x, rounds, device, fold)
            oof[name][hold] = prediction[:len(hold)]
            test_folds[name].append(prediction[len(hold):])
            print(f'fold {fold + 1}: {name} {roc_auc_score(labels[hold], oof[name][hold]):.8f} '
                  f'({perf_counter() - start:.1f}s)', flush=True)
            del fit_x, query_x
    for name in names:
        save_result(output, name, ids, labels, folds, sample, oof[name], test_folds[name],
                    folds_to_run, rounds, perf_counter() - started, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--models', nargs='+', choices=('control', 'neighborhood', 'copula'),
                        default=['control', 'neighborhood', 'copula'])
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1, 6), default=list(range(1, 6)))
    parser.add_argument('--rounds', type=int, default=1650)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or len(set(args.models)) != len(args.models) or args.rounds < 1:
        parser.error('provide unique folds and models and a positive number of rounds')
    run(args.output, args.models, [f - 1 for f in sorted(args.folds)], args.rounds, args.device)
