"""Train an additive ridge GLM with data-selected tensor-product spline effects."""

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from grandmaster_10fold_ablation.hierarchical_glm_design import (
    HierarchicalGLMDesign,
    _CATEGORIES,
    _FLAGS,
    _NUMERIC,
    _indicator,
    _integer_keys,
    _row_products,
    _smooth_values,
    _spline,
)

MODEL = "spline_interactions"
TARGET = "is_fraud"
CV_SEED = 42
N_FOLDS = 5
MAX_INTERACTIONS = 14
PAIR_SCALE = 0.36
RIDGE_C = 0.70
MAX_ITERATIONS = 220
PAIR_QUANTILES = (0.005, 0.08, 0.25, 0.5, 0.75, 0.92, 0.995)
TAIL_QUANTILES = (0.0005, 0.003, 0.012, 0.025, 0.975, 0.988, 0.997, 0.9995)
FRAUD_QUANTILES = (0.10, 0.30, 0.50, 0.70, 0.85, 0.95, 0.99)
DIGITS = (
    ("amount_usd", 100, 1, 100, 0.65),
    ("amount_usd", 100, 100, 100, 0.65),
    ("hours_since_last_txn", 100, 1, 10, 0.65),
    ("account_balance_usd", 100, 1, 100, 0.55),
    ("account_balance_usd", 100, 100, 100, 0.45),
    ("velocity_score", 10, 1, 10, 0.65),
    ("merchant_risk_score", 100, 1, 100, 0.65),
)


def pair_score(left, right, n_left, n_right, residual, information):
    """Score a coarse interaction after projecting out both marginal effects."""
    width = n_left * n_right
    cells = left * n_right + right
    gradient = np.bincount(cells, weights=residual, minlength=width).reshape(n_left, n_right)
    curvature = np.bincount(cells, weights=information, minlength=width).reshape(n_left, n_right)
    active = curvature > 0

    # In the logistic model this projects the Newton updates onto the
    # interaction subspace under the curvature-weighted inner product.
    adjustment = np.divide(gradient, curvature, out=np.zeros_like(gradient), where=active)
    row_totals = curvature.sum(axis=1, keepdims=True)
    column_totals = curvature.sum(axis=0, keepdims=True)
    for _ in range(8):
        row_mean = (curvature * adjustment).sum(axis=1, keepdims=True)
        row_mean /= np.maximum(row_totals, 1e-12)
        adjustment -= row_mean
        column_mean = (curvature * adjustment).sum(axis=0, keepdims=True)
        column_mean /= np.maximum(column_totals, 1e-12)
        adjustment -= column_mean

    degrees = max(1, int(active.sum()) - n_left - n_right + 1)
    # Low-curvature cells receive empirical-Bayes shrinkage towards zero.
    shrinkage = curvature / (curvature + 3.0)
    statistic = np.sum(curvature * shrinkage * adjustment**2)
    prior_mean = max(1.0, degrees * shrinkage[active].mean()) if active.any() else 1.0
    return float((statistic - prior_mean) / np.sqrt(2.0 * prior_mean))


class SplineDesign(HierarchicalGLMDesign):
    def fit(self, frame, labels):
        super().fit(frame)
        self.pair_specs = {}
        for column in _NUMERIC:
            raw = _smooth_values(frame, column)
            support = np.unique(raw)
            if len(support) <= 12:
                self.pair_specs[column] = ("levels", pd.Index(support))
                continue
            self.main_knots_[column] = np.unique(np.concatenate((
                self.main_knots_[column],
                np.quantile(raw, TAIL_QUANTILES),
                np.quantile(raw[labels == 1], FRAUD_QUANTILES),
            )))
            knots = np.unique(np.quantile(raw, PAIR_QUANTILES))
            self.pair_specs[column] = ("spline", knots)

        self.pair_specs.update(
            (column, ("levels", levels)) for column, levels in self.category_levels_.items()
        )
        for column in _FLAGS:
            self.pair_specs[column] = ("levels", pd.Index([False, True]))
        return self

    def additive(self, frame):
        blocks = [super().transform(frame)]
        for column, unit, divide, modulus, scale in DIGITS:
            codes = (_integer_keys(frame, column, unit) // divide) % modulus
            blocks.append(_indicator(codes, pd.RangeIndex(modulus), scale))
        flag_count = frame.loc[:, list(_FLAGS)].sum(axis=1).to_numpy(dtype=np.int64)
        blocks.append(_indicator(flag_count, pd.RangeIndex(len(_FLAGS) + 1), 0.55))
        return sparse.hstack(blocks, format="csr")

    def encode(self, frame, column):
        mode, support = self.pair_specs[column]
        if mode == "spline":
            values = _smooth_values(frame, column)
            midpoints = (support[:-1] + support[1:]) / 2
            return np.searchsorted(midpoints, values).astype(np.int32), len(support)
        values = _smooth_values(frame, column) if column in _NUMERIC else frame[column]
        codes = support.get_indexer(values).astype(np.int32)
        # Seen during fitting and safe in the screening data. Unknown levels
        # on validation or test receive all-zero interpolation in pair_basis.
        return codes, len(support)

    def pair_basis(self, frame, column):
        mode, support = self.pair_specs[column]
        if mode == "spline":
            return _spline(_smooth_values(frame, column), support)
        values = _smooth_values(frame, column) if column in _NUMERIC else frame[column]
        return _indicator(values, support, 1.0)

    def screen(self, fitting_frame, labels, baseline_probability):
        residual = labels - baseline_probability
        information = baseline_probability * (1.0 - baseline_probability)
        encoded = {column: self.encode(fitting_frame, column) for column in self.pair_specs}
        scored = []
        for left_name, right_name in itertools.combinations(encoded, 2):
            left, n_left = encoded[left_name]
            right, n_right = encoded[right_name]
            if min(n_left, n_right) <= 1:
                continue
            score = pair_score(left, right, n_left, n_right, residual, information)
            if score > 1.0:
                scored.append((score, left_name, right_name))
        return sorted(scored, reverse=True)[:MAX_INTERACTIONS]

    def include_pairs(self, frame, additive_matrix, pairs):
        bases = {
            column: self.pair_basis(frame, column)
            for column in dict.fromkeys(column for _, left, right in pairs for column in (left, right))
        }
        blocks = [_row_products(bases[left], bases[right], PAIR_SCALE) for _, left, right in pairs]
        return sparse.hstack([additive_matrix, *blocks], format="csr")


def score(labels, values):
    return {
        "roc_auc": roc_auc_score(labels, values),
        "average_precision": average_precision_score(labels, values),
        "log_loss": log_loss(labels, values),
    }


def fit_logistic(matrix, labels, baseline=None):
    model = LogisticRegression(
        C=RIDGE_C,
        solver="lbfgs",
        max_iter=MAX_ITERATIONS,
        tol=5e-7,
        warm_start=baseline is not None,
    )
    if baseline is not None:
        extra = matrix.shape[1] - baseline.coef_.shape[1]
        model.coef_ = np.pad(baseline.coef_, [(0, 0), (0, extra)])
        model.intercept_ = baseline.intercept_.copy()
    model.fit(matrix, labels)
    return model


def run_fold(train, test, labels, fit, valid):
    started = time.perf_counter()
    fitting_frame = train.iloc[fit]
    valid_frame = train.iloc[valid]
    fit_labels = labels[fit]
    design = SplineDesign().fit(fitting_frame, fit_labels)
    fit_main = design.additive(fitting_frame)
    baseline = fit_logistic(fit_main, fit_labels)
    baseline_fit = baseline.predict_proba(fit_main)[:, 1]
    valid_main = design.additive(valid_frame)
    baseline_valid = baseline.predict_proba(valid_main)[:, 1]
    pairs = design.screen(fitting_frame, fit_labels, baseline_fit)

    if pairs:
        fit_matrix = design.include_pairs(fitting_frame, fit_main, pairs)
        del fit_main
        model = fit_logistic(fit_matrix, fit_labels, baseline)
        del fit_matrix
        valid_matrix = design.include_pairs(valid_frame, valid_main, pairs)
        del valid_main
        valid_predictions = model.predict_proba(valid_matrix)[:, 1]
        del valid_matrix
        test_matrix = design.include_pairs(test, design.additive(test), pairs)
    else:
        del fit_main, valid_main
        model = baseline
        valid_predictions = baseline_valid
        test_matrix = design.additive(test)
    test_predictions = model.predict_proba(test_matrix)[:, 1]
    return baseline_valid, valid_predictions, test_predictions, {
        "baseline": score(labels[valid], baseline_valid),
        "spline_interactions": score(labels[valid], valid_predictions),
        "interaction_scores": [{"left": left, "right": right, "score": value} for value, left, right in pairs],
        "baseline_iterations": int(baseline.n_iter_[0]),
        "model_iterations": int(model.n_iter_[0]),
        "seconds": time.perf_counter() - started,
    }


def run(data_dir, output_dir, selected_folds):
    data_paths = [data_dir / name for name in ("train.csv", "test.csv", "sample_submission.csv")]
    train, test, sample = (pd.read_csv(path) for path in data_paths)
    if list(sample.columns) != ["id", TARGET] or not sample["id"].equals(test["id"]):
        raise ValueError("Submission template must have exactly id, is_fraud and match test row order.")
    labels = train[TARGET].to_numpy(dtype=np.int8)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    splits = list(splitter.split(train, labels))
    fingerprint = hashlib.sha256(Path(__file__).read_bytes())
    for path in data_paths:
        with path.open("rb") as source:
            fingerprint.update(hashlib.file_digest(source, "sha256").digest())
    run_id = fingerprint.hexdigest()[:12]
    checkpoints = output_dir / ".spline_checkpoints" / run_id
    checkpoints.mkdir(parents=True, exist_ok=True)
    for fold, (fit, valid) in enumerate(splits, start=1):
        path = checkpoints / f"fold_{fold}.npz"
        if fold in selected_folds and not path.exists():
            baseline, oof, predictions, metrics = run_fold(train, test, labels, fit, valid)
            metrics["fold"] = fold
            writing = path.with_suffix(".tmp")
            with writing.open("wb") as temporary:
                np.savez(temporary, valid=valid, baseline=baseline, oof=oof, test=predictions,
                         metrics=json.dumps(metrics))
            writing.replace(path)
            print(f"Fold {fold}: {json.dumps(metrics)}", flush=True)

    oof = np.full(len(train), np.nan)
    baseline_oof = np.full(len(train), np.nan)
    test_predictions = []
    fold_metrics = []
    for fold, (_, valid) in enumerate(splits, start=1):
        fold_path = checkpoints / f"fold_{fold}.npz"
        if not fold_path.exists():
            continue
        with np.load(fold_path, allow_pickle=False) as result:
            if not np.array_equal(valid, result["valid"]):
                raise ValueError(f"Checkpoint split mismatch for fold {fold}.")
            oof[valid] = result["oof"]
            baseline_oof[valid] = result["baseline"]
            test_predictions.append(result["test"])
            fold_metrics.append(json.loads(result["metrics"].item()))

    if len(fold_metrics) < N_FOLDS:
        print(f"Screening complete for folds {[row['fold'] for row in fold_metrics]}.", flush=True)
        return
    test_average = np.mean(test_predictions, axis=0)
    for probabilities in (oof, baseline_oof, test_average):
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Invalid predictions, check the fold results.")

    summary = {
        "name": MODEL,
        "source_and_inputs_digest": fingerprint.hexdigest(),
        "cv_seed": CV_SEED,
        "n_folds": N_FOLDS,
        "fit_iterations_limit": MAX_ITERATIONS,
        "ridge_C": RIDGE_C,
        "pair_scale": PAIR_SCALE,
        "train_rows": len(train),
        "test_rows": len(test),
        "baseline": score(labels, baseline_oof),
        "spline_interactions": score(labels, oof),
        "folds": fold_metrics,
    }
    sample[TARGET] = test_average
    sample.to_csv(output_dir / f"submission_{MODEL}.csv", index=False)
    np.savez_compressed(output_dir / f"{MODEL}_predictions.npz", oof=oof, test=test_average,
                        baseline_oof=baseline_oof, train_ids=train["id"].to_numpy(),
                        test_ids=test["id"].to_numpy())
    (output_dir / f"{MODEL}_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "folds"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=PROJECT.parent / "data", type=Path)
    parser.add_argument("--output-dir", default=Path(__file__).resolve().parent, type=Path)
    parser.add_argument("--folds", type=int, nargs="+", choices=range(1, N_FOLDS + 1),
                        default=list(range(1, N_FOLDS + 1)), help="Run selected folds and keep checkpoints.")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        run(args.data_dir, args.output_dir, args.folds)


if __name__ == "__main__":
    main()
