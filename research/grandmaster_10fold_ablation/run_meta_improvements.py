"""Test four changes to the 10-fold hierarchical Grandmaster stack."""

from hashlib import sha256
import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from export_hierarchical_submission import BASELINE, OUT_DIR as PARENT_SUBMISSION, replace_dense_meta_input, write_submission
from fold_aggregation import aggregate_fold_scores
from ranking_meta import fit_rank_stacker
from risk_gated_meta import fit_gated_stacker
from run_ablation import NOTEBOOK_SHA256, ROOT, file_hash, load_archive


VALIDATION = ROOT / "validation"
RESULTS = ROOT / "meta_improvements"
CANDIDATES = ("pairwise_loss", "risk_gated", "paired_logits", "mean_probabilities")
RANK_WEIGHT = 0.03
RANK_PAIRS = 32


def meta_matrix(nb, original, new_dense):
    raw = dict(zip(nb["LOCKED_MODELS"], original.T, strict=True))
    return replace_dense_meta_input(nb["build_resdecomp"], raw, new_dense)


def get_folds(nb, labels, count, seed):
    splitter = nb["StratifiedKFold"](n_splits=count, shuffle=True, random_state=seed)
    return [(i, train, held) for i, (train, held) in enumerate(splitter.split(labels, labels))]


def stack_predictions(nb, training, labels, folds, l2_values, scoring, methods):
    fit_logistic = nb["_fit_nonneg_l2_weights"]
    link = nb["from_logit_10"]
    y = labels.astype(np.float64)
    common_input = np.zeros_like(scoring[0])
    for fold_input in scoring:
        common_input += fold_input / len(scoring)

    parameters = []
    ranking_logits = np.zeros(len(common_input))
    gated_logits = np.zeros(len(common_input))
    for (fold, train, _), l2 in zip(folds, l2_values, strict=True):
        print(f"Meta fold {fold + 1}/{len(folds)}", flush=True)
        x_fit, y_fit = training[train], y[train]
        baseline = fit_logistic(x_fit, y_fit, l2)
        parameters.append(baseline)
        if "pairwise_loss" in methods:
            intercept, weights = fit_rank_stacker(
                x_fit, y_fit, l2, baseline, seed=7601 + fold,
                rank_weight=RANK_WEIGHT, pairs_per_positive=RANK_PAIRS,
            )
            ranking_logits += (intercept + common_input @ weights) / len(folds)
        if "risk_gated" in methods:
            model = fit_gated_stacker(x_fit, y_fit, l2, baseline)
            gated_logits += model.predict_logit(common_input) / len(folds)

    combined = aggregate_fold_scores(parameters, scoring, link)
    baseline = combined.pop("mean_logits")
    combined["pairwise_loss"] = link(ranking_logits)
    combined["risk_gated"] = link(gated_logits)
    result = {"current_hierarchy": baseline, **{name: combined[name] for name in methods}}
    for name, values in result.items():
        if values.shape != baseline.shape or not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
            raise ValueError(f"Invalid probabilities for {name}.")
    return result


def assert_recorded_predictions(path, expected_ids, probabilities, column):
    recorded = pd.read_csv(path, usecols=["id", column])
    np.testing.assert_array_equal(recorded["id"], expected_ids)
    np.testing.assert_allclose(recorded[column], probabilities, atol=1e-10, rtol=0)


def own_validation_inputs(nb, train, meta):
    for name, expected in meta["experiment_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"The verified validation code differs at {name}.")
    splitter = nb["StratifiedShuffleSplit"](
        n_splits=1, test_size=meta["holdout_fraction"], random_state=meta["holdout_seed"]
    )
    develop_rows, evaluation_rows = next(splitter.split(train["id"], train["is_fraud"]))
    y = train["is_fraud"].to_numpy(dtype=np.int8)
    develop_labels = y[np.sort(develop_rows)]
    evaluation_rows = np.sort(evaluation_rows)
    evaluation_ids = train["id"].to_numpy()[evaluation_rows]
    if (len(develop_rows), len(evaluation_rows)) != (meta["develop_rows"], meta["holdout_rows"]):
        raise ValueError("Unexpected development and holdout sizes.")
    if sha256(evaluation_ids.tobytes()).hexdigest() != meta["holdout_ids_sha256"]:
        raise ValueError("Holdout IDs differ from the validated experiment.")

    folds = get_folds(nb, develop_labels, meta["cv_folds"], meta["cv_seed"])
    training = np.full((len(develop_labels), len(nb["LOCKED_MODELS"])), np.nan)
    scoring = []
    original_signature = None
    for fold, _, held in folds:
        path = VALIDATION / "artifacts" / f"fold_{fold + 1:02d}.npz"
        with np.load(path, allow_pickle=False) as data:
            signature = data["signature"].item()
            if (original_signature is not None and signature != original_signature) or not np.array_equal(data["va_idx"], held):
                raise ValueError(f"Validation checkpoint differs at {path}.")
            original_signature = signature
            training[held] = meta_matrix(nb, data["baseline_va"], data["candidate_va"][:, 1])
            scoring.append(meta_matrix(nb, data["baseline_hold"], data["candidate_hold"][:, 1]))
    if not np.isfinite(training).all() or any(len(part) != len(evaluation_ids) or not np.isfinite(part).all() for part in scoring):
        raise ValueError("Validation OOF or holdout predictions are incomplete or invalid.")
    return training, develop_labels, folds, scoring, evaluation_ids, y[evaluation_rows]


def own_full_inputs(nb, labels, full_meta):
    for name, expected in full_meta["implementation_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Prior submission code differs at {name}.")
    folds = get_folds(nb, labels, full_meta["folds"], full_meta["cv_seed"])
    training = np.full((len(labels), len(nb["LOCKED_MODELS"])), np.nan)
    scoring = []
    dense_identity_fields = (
        "variant", "train_sha256", "test_sha256", "notebook_sha256", "folds", "cv_seed",
        "train_rows", "test_rows", "local_10fold_cache_sha256", "implementation_sha256", "environment",
    )
    dense_identity = {name: full_meta[name] for name in dense_identity_fields}
    signature = sha256(json.dumps(dense_identity, sort_keys=True).encode()).hexdigest()
    for fold, fit, held in folds:
        path = BASELINE / "artifacts" / f"fold_{fold + 1:02d}.npz"
        if file_hash(path) != full_meta["local_10fold_cache_sha256"][path.name]:
            raise ValueError(f"Local baseline checkpoint differs at {path}.")
        dense_path = PARENT_SUBMISSION / "artifacts" / f"dense_fold_{fold + 1:02d}.npz"
        with np.load(path, allow_pickle=False) as base, np.load(dense_path, allow_pickle=False) as dense:
            if (dense["signature"].item() != signature or not np.array_equal(dense["held_rows"], held)
                    or not np.array_equal(base["fit_rows"], fit) or not np.array_equal(base["held_rows"], held)):
                raise ValueError(f"Full-training checkpoint does not match fold {fold + 1}.")
            training[held] = meta_matrix(nb, base["held"], dense["held"])
            scoring.append(meta_matrix(nb, base["test"], dense["test"]))
    if not np.isfinite(training).all() or any(len(part) != full_meta["test_rows"] or not np.isfinite(part).all() for part in scoring):
        raise ValueError("Full-training OOF or test predictions are incomplete or invalid.")
    return training, folds, scoring


def score_variants(nb, labels, predictions):
    baseline = predictions["current_hierarchy"]
    metrics = {}
    for name, values in predictions.items():
        entry = {
            "roc_auc": float(roc_auc_score(labels, values)),
            "average_precision": float(average_precision_score(labels, values)),
            "log_loss": float(log_loss(labels, values)),
            "delta_roc_auc": 0.0,
        }
        if name != "current_hierarchy":
            compared = nb["delong_paired_test"](labels, values, baseline)
            entry.update({
                "delta_roc_auc": compared["diff"], "ci95_low": compared["ci_low"],
                "ci95_high": compared["ci_high"], "p_value": compared["p_value"],
            })
        metrics[name] = entry
    return metrics


def main():
    began = perf_counter()
    nb, _ = load_archive(RESULTS, smoke=False)
    metadata = json.loads((VALIDATION / "metrics.json").read_text(encoding="utf-8"))
    train_path = ROOT.parents[1] / 'data' / "train.csv"
    train_sha = file_hash(train_path)
    if (train_sha != metadata["train_sha256"] or NOTEBOOK_SHA256 != metadata["notebook_sha256"]
            or metadata["cv_folds"] != 10 or metadata["mode"] != "validation"):
        raise ValueError("Expected the official train data and the verified 10-fold validation run.")
    train = pd.read_csv(train_path, usecols=["id", "is_fraud"])
    matrices, y_dev, folds, score_folds, ids, y_hold = own_validation_inputs(nb, train, metadata)
    print(f"Testing the meta changes on the same {len(ids):,} held-out rows", flush=True)
    estimates = stack_predictions(nb, matrices, y_dev, folds, metadata["baseline_meta_l2_by_fold"], score_folds, CANDIDATES)
    assert_recorded_predictions(VALIDATION / "holdout_predictions.csv", ids, estimates["current_hierarchy"], "hierarchical_amount")
    metrics = score_variants(nb, y_hold, estimates)
    best = max(CANDIDATES, key=lambda name: metrics[name]["roc_auc"])
    winner = best if metrics[best]["delta_roc_auc"] > 1e-12 else None
    scored = RESULTS / "holdout_predictions.csv"
    pending = scored.with_suffix(".tmp.csv")
    pd.DataFrame({"id": ids, "is_fraud": y_hold, **estimates}).to_csv(pending, index=False)
    pending.replace(scored)
    summary = {
        "train_sha256": train_sha,
        "notebook_sha256": NOTEBOOK_SHA256,
        "holdout_ids_sha256": metadata["holdout_ids_sha256"],
        "holdout_rows": len(ids),
        "development_rows": len(y_dev),
        "folds": len(folds),
        "rank_weight": RANK_WEIGHT,
        "pairs_per_positive": RANK_PAIRS,
        "variants": metrics,
        "best_positive_holdout_candidate": winner,
    }
    # Save the comparison before the separate submission stage so the scores
    # remain available if the export is interrupted.
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    del matrices, score_folds, estimates
    nb["gc"].collect()

    if winner is not None:
        full = json.loads((PARENT_SUBMISSION / "manifest.json").read_text(encoding="utf-8"))
        test_path = ROOT.parents[1] / 'data' / "test.csv"
        if (full["train_sha256"] != train_sha or full["test_sha256"] != file_hash(test_path)
                or full["notebook_sha256"] != NOTEBOOK_SHA256 or full["variant"] != "hierarchical_amount"
                or file_hash(PARENT_SUBMISSION / "submission.csv") != full["submission_sha256"]):
            raise ValueError("Official train/test or parent submission data has changed.")
        test_ids = pd.read_csv(test_path, usecols=["id"])["id"].to_numpy()
        template = pd.read_csv(ROOT.parents[1] / 'data' / "sample_submission.csv", usecols=["id"])["id"].to_numpy()
        if len(test_ids) != len(template) or len(template) != 299_984 or not np.array_equal(test_ids, template):
            raise ValueError("Expected all official test IDs in the template order.")
        labels = train["is_fraud"].to_numpy(dtype=np.int8)
        print(f"Generating the official test predictions for {winner}", flush=True)
        matrices, folds, score_folds = own_full_inputs(nb, labels, full)
        estimates = stack_predictions(nb, matrices, labels, folds, full["baseline_meta_l2_by_fold"], score_folds, (winner,))
        assert_recorded_predictions(PARENT_SUBMISSION / "submission.csv", test_ids, estimates["current_hierarchy"], "is_fraud")
        submission = RESULTS / "submission.csv"
        write_submission(submission, test_ids, estimates[winner], template)
        summary["submission"] = {
            "variant": winner,
            "rows": len(test_ids),
            "sha256": file_hash(submission),
            "mean_probability": float(estimates[winner].mean()),
            "mean_absolute_probability_change": float(np.abs(estimates[winner] - estimates["current_hierarchy"]).mean()),
        }
    summary["elapsed_seconds"] = perf_counter() - began
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Verified meta experiment outputs in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
