"""Refit selected individual models on all competition training rows."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
GPU = HERE.parent
PLAN = GPU / "plan_runs"
BASE = PLAN / "results" / "full"
SEED = GPU / "seed3_improvement" / "results"
IMBALANCE = GPU / "imbalance" / "results" / "xgb"
OUTPUT = HERE / "results"
BASELINE = "XGB-D03"
CANDIDATES = {
    "XGB-D03-S71-E2028": SEED,
    BASELINE: BASE,
    "XGB-RANDOM": IMBALANCE,
    "XGB-D03-R125": BASE,
    "XGB-HARD": IMBALANCE,
    "D03": BASE,
}

sys.path[:0] = [str(PLAN), str(GPU / "imbalance")]
from models import fit_predict, predict_from_state
from run_plan import Experiment, PlanRun, read_json, sha256, write_json, write_npz
from run_xgb import NEGATIVE_FRACTION, NEGATIVE_WEIGHT, RANDOM_SEED, _digest_array, _fit_cuda, _negative_indices


def probabilities(values, count):
    scores = np.asarray(values, dtype=np.float64)
    if scores.shape != (count,) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid prediction vector")
    return scores


def load_reference(run, name, source):
    path = source / name
    metrics = read_json(path / "metrics.json")
    if metrics["name"] != name or metrics["status"] != "completed" or any(
        metrics.get(key) != value for key, value in run.manifest.items()
    ):
        raise ValueError(f"Incompatible reference for {name}")
    with np.load(path / "predictions.npz", allow_pickle=False) as saved:
        for key, array in (("ids", run.ids), ("test_ids", run.test_ids), ("fold", run.folds)):
            np.testing.assert_array_equal(saved[key], array, err_msg=f"{name} {key}")
        oof = probabilities(saved["oof"], len(run.ids))
        test = probabilities(saved["test"], len(run.test_ids))
    if not np.isclose(roc_auc_score(run.y, oof), metrics["oof_auc"], rtol=0, atol=1e-12):
        raise ValueError(f"OOF AUC mismatch for {name}")
    folds = metrics["folds"]
    rounds = [fold["rounds"] for fold in folds]
    valid_rounds = all(count == 0 for count in rounds) if metrics["config"]["family"] == "glm" else min(rounds, default=0) > 0
    if len(folds) != 5 or not valid_rounds or any("RTX 4060" not in fold["device"] for fold in folds):
        raise ValueError(f"Missing GPU fold evidence for {name}")
    return metrics, oof, test, path


def select_weights(run, name, config, base_oof):
    scores = base_oof if name == "XGB-HARD" else None
    if config["negative_fraction"] != NEGATIVE_FRACTION or config["negative_weight"] != NEGATIVE_WEIGHT:
        raise ValueError(f"Weighted model policy changed for {name}")
    if name == "XGB-RANDOM" and config["random_negative_seed"] != RANDOM_SEED:
        raise ValueError("Random negative seed changed")
    # Zero is the dedicated full-data run index for the existing seed-sequence rule.
    picked = _negative_indices(run.y, run.ids, scores, 0)[name]
    weights = np.ones(len(run.y), dtype=np.float32)
    weights[picked] = NEGATIVE_WEIGHT
    return weights, {
        "rows_upweighted": len(picked),
        "weight": NEGATIVE_WEIGHT,
        "upweighted_id_sha256": _digest_array(np.sort(run.ids[picked])),
        "criterion": "XGB-D03 OOF risk" if scores is not None else "seed 42, run index 0",
    }


def save_submission(path, ids, scores):
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(("id", "is_fraud"))
        writer.writerows(zip(ids.tolist(), scores.tolist(), strict=True))
    temp.replace(path)


def fit_full(run, spec, rounds, matrices, weights, history, reference):
    x_train, x_test = matrices
    if weights is None:
        params = run.model_params(spec, matrices, reference)
        params["include_model_state"] = True
        return fit_predict(spec.family, x_train, run.y, (x_test,),
                           params=params, seed=spec.model_seed, rounds=rounds or None)
    settings = history[0]["settings"]
    if any(fold["settings"] != settings for fold in history):
        raise ValueError(f"Inconsistent training settings for {spec.name}")
    return _fit_cuda(settings, rounds, x_train, run.y, (x_test,),
                     weights=weights, include_model_state=True)


def run_models(run, candidates=CANDIDATES):
    base_metrics, base_oof, _, base_path = load_reference(run, BASELINE, BASE)
    full_rows = np.arange(len(run.ids), dtype=np.int64)
    reports = []
    matrices, last_key = None, None
    for name, source in candidates.items():
        metrics, _, fold_average, path = load_reference(run, name, source)
        config = metrics["config"]
        is_weighted = source == IMBALANCE
        spec_args = base_metrics["config"] if is_weighted else config
        spec = Experiment(**{**spec_args, "groups": tuple(spec_args["groups"])})
        folds = metrics["folds"]
        fold_rounds = [fold["rounds"] for fold in folds]
        rounds = int(np.median(fold_rounds))
        weights, weighting = None, None
        if is_weighted:
            base_digest = sha256(base_path / "predictions.npz")
            if config["source"]["predictions_sha256"] != base_digest or config["scale_pos_weight"] != 1:
                raise ValueError(f"Weighted prediction provenance changed for {name}")
            weights, weighting = select_weights(run, name, config, base_oof)
        record = {
            "fit_protocol": "full_train_once_using_cv_median_rounds",
            "input_manifest": run.manifest,
            "name": name,
            "spec": json_safe(asdict(spec)),
            "reference_metrics_sha256": sha256(path / "metrics.json"),
            "reference_predictions_sha256": sha256(path / "predictions.npz"),
            "reference_oof_auc": metrics["oof_auc"],
            "rounds_from_folds": fold_rounds,
            "rounds": rounds,
            "weighting": weighting,
        }
        directory = run.output / name
        directory.mkdir(exist_ok=True)
        report_path = directory / "metrics.json"
        prediction_path = directory / "predictions.npz"
        model_path = directory / ("model.txt" if spec.family == "lgb" else "model.json")
        if report_path.is_file():
            report = read_json(report_path)
            if (
                report["protocol"] != record
                or report["model_sha256"] != sha256(model_path)
                or report["predictions_sha256"] != sha256(prediction_path)
            ):
                raise ValueError(f"Checkpoint provenance changed for {name}")
            with np.load(prediction_path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["id"], run.test_ids)
                scores = probabilities(saved["full"], len(run.test_ids))
                np.testing.assert_array_equal(saved["fold_mean"], fold_average)
            print(f"{name}: verified saved full-train result", flush=True)
        else:
            started = perf_counter()
            print(f"{name}: fitting {len(run.ids):,} training rows, {rounds:,} rounds on GPU", flush=True)
            key = (spec.family, spec.groups, spec.smoothing, spec.encoder_seed)
            if key != last_key:
                matrices = run.matrices(spec, full_rows, (run.test_indices,))
                last_key = key
            (prediction,), details = fit_full(run, spec, rounds, matrices, weights, folds, full_rows)
            if "RTX 4060" not in details["device"] or rounds != details["rounds"]:
                raise RuntimeError(f"Unexpected fit device or rounds for {name}: {details}")
            scores = probabilities(prediction, len(run.test_ids))
            state = details.pop("model_state")
            temp = model_path.with_suffix(model_path.suffix + ".tmp")
            raw_state = json.dumps(state, allow_nan=False) if spec.family == "glm" else state
            temp.write_bytes(raw_state.encode("utf-8"))
            temp.replace(model_path)
            saved_state = read_json(model_path) if spec.family == "glm" else model_path.read_text(encoding="utf-8")
            (replayed,) = predict_from_state(spec.family, saved_state, (matrices[1],))
            replay_delta = float(np.abs(replayed - scores).max())
            tolerance = 2e-5 if spec.family == "xgb" else 1e-6
            if not np.isfinite(replay_delta) or replay_delta > tolerance:
                raise RuntimeError(f"Saved {name} replay exceeded tolerance: {replay_delta}")
            write_npz(prediction_path, id=run.test_ids, full=scores, fold_mean=fold_average)
            report = {
                "protocol": record,
                "fit": details,
                "total_seconds": perf_counter() - started,
                "model_sha256": sha256(model_path),
                "predictions_sha256": sha256(prediction_path),
                "replay_max_abs_delta": replay_delta,
                "test_spearman_vs_fold_mean": float(spearmanr(scores, fold_average).statistic),
                "test_mean_abs_delta_vs_fold_mean": float(np.mean(np.abs(scores - fold_average))),
            }
            write_json(report_path, report)
            print(f"{name}: done on {details['device']} | {report['total_seconds']:.1f}s", flush=True)
        submission_path = directory / f"{name}_FULLREFIT_submission.csv"
        save_submission(submission_path, run.test_ids, scores)
        reports.append({**report, "submission": str(submission_path.relative_to(run.output.parent))})
    write_json(run.output / "summary.json", reports)
    return reports


def json_safe(spec):
    return {**spec, "groups": list(spec["groups"])}


if __name__ == "__main__":
    run_models(PlanRun(OUTPUT))
