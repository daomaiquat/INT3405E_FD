"""Ensemble diagnostics for aligned development predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, log_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _labels(y):
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("Expected a one-dimensional binary target with both classes")
    return y.astype(np.int8, copy=False)


def _scores(values, size):
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("Expected one finite score per aligned row")
    return result


def _ordered_auc(scores, y):
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    positive = y[order]
    return order, starts, positive


def _weighted_auc(prepared, weights, denominator):
    order, starts, positive = prepared
    row_weights = weights[order]
    total = np.add.reduceat(row_weights, starts)
    pos = np.add.reduceat(row_weights * positive, starts)
    neg = total - pos
    return float(np.sum(pos * (np.cumsum(neg) - .5 * neg)) / denominator)


def bootstrap_delta(y, a, b, repeats=500, seed=42):
    """Paired stratified percentile interval for ROC AUC(a) minus ROC AUC(b)."""
    y = _labels(y)
    if repeats < 1:
        raise ValueError("Bootstrap needs at least one repeat")
    a, b = (_scores(values, len(y)) for values in (a, b))
    negative, positive = (np.flatnonzero(y == value) for value in (0, 1))
    denominator = len(negative) * len(positive)
    ordered_a, ordered_b = (_ordered_auc(values, y) for values in (a, b))
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for index in range(repeats):
        rows = np.concatenate((rng.choice(negative, len(negative)),
                               rng.choice(positive, len(positive))))
        weights = np.bincount(rows, minlength=len(y))
        samples[index] = (_weighted_auc(ordered_a, weights, denominator)
                          - _weighted_auc(ordered_b, weights, denominator))
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"delta": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "ci_low": float(low), "ci_high": float(high),
            "repeats": repeats, "seed": seed, "method": "paired_stratified_percentile",
            "samples": samples.tolist()}


def _rank_logit_arrays(reference, others):
    transformed = [np.empty(pair.shape, dtype=np.float32) for pair in (reference, *others)]
    for column in range(reference.shape[1]):
        ordered = np.sort(reference[:, column])
        for values, output in zip((reference, *others), transformed):
            x = values[:, column]
            left = np.searchsorted(ordered, x, side="left")
            right = np.searchsorted(ordered, x, side="right")
            p = (left + right + 1) / (2 * (len(ordered) + 1))
            output[:, column] = np.log(p / (1 - p))
    return transformed


def _logit(values):
    probabilities = np.clip(np.asarray(values, dtype=np.float32), 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities))


def fit_meta_fold(x_train, y_train, *prediction_sets, c=1., rank_logit=False,
                  fit_backend=None, return_details=False):
    """Train meta GLM through the GPU backend and return aligned probabilities."""
    if fit_backend is None:
        from models import fit_predict
        fit_backend = fit_predict
    if rank_logit:
        x_train, *prediction_sets = _rank_logit_arrays(x_train, prediction_sets)
    else:
        x_train, *prediction_sets = [_logit(x) for x in (x_train, *prediction_sets)]
    predictions, details = fit_backend(
        "glm", x_train, y_train, tuple(prediction_sets),
        params={"C": c, "solver": "newton-cholesky", "include_model_state": return_details},
    )
    aligned = tuple(_scores(prediction, len(rows)) for prediction, rows in
                    zip(predictions, prediction_sets, strict=True))
    return (aligned, details) if return_details else aligned


def _meta_cv(x, y, test, splits, c, rank_logit, fit_backend):
    oof = np.empty(len(y))
    test_mean = np.zeros(len(test))
    fold_details = []
    for train, valid in splits:
        (valid_scores, test_scores), details = fit_meta_fold(
            x[train], y[train], x[valid], test, c=c, rank_logit=rank_logit,
            fit_backend=fit_backend, return_details=True,
        )
        oof[valid] = valid_scores
        test_mean += test_scores / len(splits)
        fold_details.append(details)
    return (oof, test_mean), fold_details


def _metric_row(name, method, y, p, folds, model_names, **metadata):
    return {
        "candidate": name, "method": method, "models": " ".join(model_names),
        "auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": None if method == "rank_mean" else float(log_loss(y, p)),
        **{f"auc_fold_{fold}": float(roc_auc_score(y[folds == fold], p[folds == fold]))
           for fold in np.unique(folds)},
        **metadata,
    }


def _error_report(y, scores, ids):
    errors = pd.DataFrame({"id": ids, "label": y, "score": scores})
    positives = errors.loc[errors.label == 1].nsmallest(100, "score").assign(error="low_scoring_positive")
    negatives = errors.loc[errors.label == 0].nlargest(100, "score").assign(error="high_scoring_negative")
    return pd.concat((positives, negatives), ignore_index=True)


def _budget_report(y, ids, candidates):
    rows = []
    for name, scores in candidates:
        order = np.lexsort((ids, -scores))
        for fraction in (.005, .01, .02):
            limit = int(np.ceil(len(y) * fraction))
            hits = int(y[order[:limit]].sum())
            rows.append({"candidate": name, "fraction": fraction, "reviews": limit,
                         "positives": hits, "precision": hits / limit, "recall": hits / int(y.sum())})
    return pd.DataFrame(rows)


def error_groups(y, ids, groups, candidates, fraction=.01):
    """Measure errors per group at the same fixed overall review budget."""
    y = _labels(y)
    if not 0 < fraction <= 1:
        raise ValueError("Review fraction must be in (0,1]")
    limit = int(np.ceil(len(y) * fraction))
    tables = []
    for name, scores in candidates:
        order = np.lexsort((ids, -_scores(scores, len(y))))
        reviewed = np.zeros(len(y), dtype=bool)
        reviewed[order[:limit]] = True
        for column, values in groups.items():
            aligned = np.asarray(values)
            if aligned.shape != y.shape:
                raise ValueError(f"Group {column!r} is not aligned with labeled rows")
            rows = pd.DataFrame({"group": aligned, "positive": y, "reviewed": reviewed,
                                 "true_positive": y * reviewed})
            grouped = rows.groupby("group", dropna=False).agg(
                rows=("positive", "size"), positives=("positive", "sum"),
                reviewed=("reviewed", "sum"), true_positives=("true_positive", "sum"),
            ).reset_index()
            grouped["missed_positives"] = grouped.positives - grouped.true_positives
            grouped["reviewed_negatives"] = grouped.reviewed - grouped.true_positives
            grouped["precision"] = grouped.true_positives / grouped.reviewed.where(grouped.reviewed > 0)
            grouped["recall"] = grouped.true_positives / grouped.positives.where(grouped.positives > 0)
            tables.append(grouped.assign(candidate=name, variable=column, review_fraction=fraction))
    return pd.concat(tables, ignore_index=True)


def validate_submission(path, sample, *, run_id, score_kind):
    """Validate a caller-created submission against the supplied competition template."""
    path = Path(path)
    frame = pd.read_csv(path)
    expected = ["id", "is_fraud"]
    if list(frame.columns) != expected or list(sample.columns) != expected:
        raise ValueError(f"Expected exactly {expected}")
    if (frame.id.isna().any() or not frame.id.is_unique
            or not np.array_equal(frame.id.to_numpy(), sample.id.to_numpy())):
        raise ValueError("Submission IDs must be unique and match the sample order")
    scores = _scores(frame.is_fraud, len(sample))
    if not len(scores) or scores.min() < 0 or scores.max() > 1:
        raise ValueError("Submission scores must be nonempty and in [0,1]")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"run_id": run_id, "path": str(path.resolve()), "sha256": digest,
            "rows": len(frame), "score_kind": score_kind}


def evaluate_ensembles(y, oof, test, folds, *, output, meta_seeds=(7, 8),
                       c_values=(.01, .1, 1.), repeats=500, seed=42,
                       rank_logit=False, ids=None, fit_backend=None, groups=None):
    """Compare base and ensemble predictions with development-only meta CV."""
    y = _labels(y)
    folds = np.asarray(folds)
    if folds.shape != y.shape or pd.isna(folds).any():
        raise ValueError("Expected one base fold per labeled row")
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2 or any(np.unique(y[folds == fold]).size != 2 for fold in unique_folds):
        raise ValueError("Each of at least two base folds must contain both classes")
    if not oof or set(oof) != set(test):
        raise ValueError("OOF and test must have the same nonempty set of model keys")
    if not meta_seeds or not c_values or any(c <= 0 for c in c_values):
        raise ValueError("Provide meta seeds and positive C values")
    names = list(oof)
    n_test = len(test[names[0]])
    ids = np.arange(len(y)) if ids is None else np.asarray(ids)
    if not n_test or ids.shape != y.shape or not pd.Index(ids).is_unique or pd.isna(ids).any():
        raise ValueError("Expected nonempty test and one unique ID per labeled row")
    x = np.column_stack([_scores(oof[name], len(y)) for name in names])
    xt = np.column_stack([_scores(test[name], n_test) for name in names])
    if min(x.min(), xt.min()) < 0 or max(x.max(), xt.max()) > 1:
        raise ValueError("Base prediction inputs must be probabilities")
    metrics = [_metric_row(name, "base", y, x[:, i], folds, [name])
               for i, name in enumerate(names)]
    winner_index = max(range(len(names)), key=lambda i: metrics[i]["auc"])
    best_base = names[winner_index]
    baseline = x[:, winner_index]
    winner = best_base
    winner_auc = metrics[winner_index]["auc"]
    winner_oof, winner_test = baseline, xt[:, winner_index]
    winner_refit = None
    full_stacks = []
    coefficient_rows = []

    def record(name, method, pair, used, *, eligible=True, **metadata):
        nonlocal winner, winner_auc, winner_oof, winner_test, winner_refit
        scores, test_scores = pair
        row = _metric_row(name, method, y, scores, folds, used, **metadata)
        metrics.append(row)
        if eligible and row["auc"] > winner_auc:
            winner, winner_auc, winner_oof, winner_test = name, row["auc"], scores, test_scores
            winner_refit = ((metadata["c"], method == "rank_logit_stack")
                            if method in {"logistic_stack", "rank_logit_stack"} else None)
        return row["auc"]

    def record_coefficients(name, used, details):
        for fold, fit in enumerate(details):
            state = fit.get("model_state")
            if state is None:
                continue
            for model, coefficient in zip(used, state["coefficients"], strict=True):
                coefficient_rows.append({"candidate": name, "meta_fold": fold, "member": model,
                                         "coefficient": coefficient, "intercept": state["intercept"],
                                         "device": fit.get("device"), "converged": fit.get("converged")})

    if len(names) > 1:
        record("probability_mean", "probability_mean", (x.mean(1), xt.mean(1)), names)
        record("rank_mean", "rank_mean",
               (rankdata(x, axis=0).mean(1) / (len(x) + 1),
                rankdata(xt, axis=0).mean(1) / (len(xt) + 1)), names)
        for meta_seed in meta_seeds:
            splitter = StratifiedKFold(n_splits=len(unique_folds), shuffle=True, random_state=meta_seed)
            splits = list(splitter.split(x, y))
            for use_rank in (False, True) if rank_logit else (False,):
                method = "rank_logit_stack" if use_rank else "logistic_stack"
                for c in c_values:
                    name = f"{method}_seed{meta_seed}_c{c:g}"
                    pair, details = _meta_cv(x, y, xt, splits, c, use_rank, fit_backend)
                    auc = record(name, method, pair, names, meta_seed=meta_seed, c=c)
                    record_coefficients(name, names, details)
                    full_stacks.append((auc, meta_seed, c, use_rank, name))
        best_stack_auc, meta_seed, c, use_rank, best_stack = max(full_stacks, key=lambda item: item[0])
        splitter = StratifiedKFold(n_splits=len(unique_folds), shuffle=True, random_state=meta_seed)
        splits = list(splitter.split(x, y))
        for i, omitted in enumerate(names):
            columns = np.arange(len(names)) != i
            used = [name for name in names if name != omitted]
            pair, details = _meta_cv(x[:, columns], y, xt[:, columns], splits, c, use_rank, fit_backend)
            record(f"leave_out_{omitted}", "leave_one_out", pair, used, eligible=False,
                   omitted=omitted, meta_seed=meta_seed, c=c,
                   full_stack=best_stack, full_stack_auc=best_stack_auc)
            record_coefficients(f"leave_out_{omitted}", used, details)

    if winner_refit is not None:
        c, use_rank = winner_refit
        (winner_test,), final_details = fit_meta_fold(
            x, y, xt, c=c, rank_logit=use_rank, fit_backend=fit_backend, return_details=True,
        )
    else:
        final_details = None

    for row in metrics:
        row["delta_auc_vs_best_base"] = row["auc"] - metrics[winner_index]["auc"]
        deltas = []
        for fold in unique_folds:
            fold_key = f"auc_fold_{fold}"
            delta = row[fold_key] - metrics[winner_index][fold_key]
            row[f"delta_fold_{fold}_vs_best_base"] = delta
            deltas.append(delta)
        row["positive_delta_folds"] = sum(delta > 0 for delta in deltas)
        row["passes_practical_screen"] = (
            row["delta_auc_vs_best_base"] >= .0001
            and row["positive_delta_folds"] >= int(np.ceil(.8 * len(unique_folds)))
        )

    confidence = bootstrap_delta(y, winner_oof, baseline, repeats=repeats, seed=seed)
    outcome = {"winner": winner, "best_base": best_base, "winner_auc": winner_auc,
               "bootstrap": confidence, "metrics": metrics,
               "validation": "development_meta_cv_on_base_oof",
               "bootstrap_scope": "fixed_selected_predictions",
               "test_refit": "all_base_oof" if winner_refit is not None else "base_test_inputs",
               "test_refit_models": names if winner_refit is not None else None,
               "test_refit_details": final_details,
               "meta_backend": "injected" if fit_backend is not None else "models.fit_predict_gpu"}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output / "ensemble_metrics.csv", index=False)
    pd.DataFrame(coefficient_rows, columns=["candidate", "meta_fold", "member", "coefficient",
                                             "intercept", "device", "converged"]).to_csv(
        output / "ensemble_coefficients.csv", index=False,
    )
    _error_report(y, winner_oof, ids).to_csv(output / "ensemble_errors.csv", index=False)
    candidates = [(best_base, baseline)]
    if winner != best_base:
        candidates.append((winner, winner_oof))
    _budget_report(y, ids, candidates).to_csv(output / "ensemble_review_budget.csv", index=False)
    if groups:
        error_groups(y, ids, groups, candidates).to_csv(output / "ensemble_error_groups.csv", index=False)
    curves = []
    for name, scores in candidates:
        precision, recall, threshold = precision_recall_curve(y, scores)
        curves.append(pd.DataFrame({"candidate": name, "precision": precision, "recall": recall,
                                    "threshold": np.r_[threshold, np.nan]}))
    pd.concat(curves, ignore_index=True).to_csv(output / "ensemble_pr_curve.csv", index=False)
    (output / "ensemble_summary.json").write_text(json.dumps(outcome, indent=2, allow_nan=False),
                                                 encoding="utf-8")
    return {**outcome, "oof": winner_oof, "test": winner_test}
