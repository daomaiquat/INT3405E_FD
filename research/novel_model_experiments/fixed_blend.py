"""Evaluate a fixed probability blend of three locally trained S6E11 models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
GPU = PROJECT / "s6e11_gpu"
PLAN = GPU / "plan_runs"
BASE = PLAN / "results" / "full"
FOLLOWUP = GPU / "followup"
MLP = GPU / "imbalance" / "results" / "mlp" / "MLP-BCE"
LABEL = "is_fraud"
SEEDS = (42, 1337, 2718)
WEIGHTS = {
    "FINAL": (1.0, 0.0, 0.0),
    "FINAL_PLUS_LGBM20": (0.8, 0.2, 0.0),
    "FINAL_PLUS_MLP10": (0.9, 0.0, 0.1),
    "FIXED_BLEND": (0.7, 0.2, 0.1),
}

sys.path.insert(0, str(PLAN))
from run_plan import frame_hash, metric_row, read_json, sha256, write_json, write_npz
from src.validation import assign_folds


def blend(final, lightgbm, mlp, weights=WEIGHTS["FIXED_BLEND"]):
    return weights[0] * final + weights[1] * lightgbm + weights[2] * mlp


def probabilities(values, rows, origin):
    valid = values.shape == (rows,) and np.isfinite(values).all()
    if not valid or (values < 0).any() or (values > 1).any():
        raise ValueError(f"Invalid probability vector in {origin}")
    return values


def verified_manifest(path, inputs, fold_hash):
    manifest = read_json(path)
    if (manifest["inputs"] != inputs or manifest["fold_hash"] != fold_hash
            or manifest["models_use_llm"] is not False or manifest["smoke"] is not False):
        raise ValueError(f"Incompatible input, fold, or pipeline provenance in {path}")
    return manifest


def load_seed42(train, test_ids, folds, fold_hash, inputs, provenance):
    plan_manifest = BASE / "manifest.json"
    mlp_manifest = MLP.parent / "manifest.json"
    for path in (plan_manifest, mlp_manifest):
        verified_manifest(path, inputs, fold_hash)
        provenance[path.relative_to(PROJECT).as_posix()] = sha256(path)

    oof, test = [], []
    for directory in (BASE / "final", BASE / "D03", MLP):
        metrics_path = directory / "metrics.json"
        metrics = verified_manifest(metrics_path, inputs, fold_hash)
        path = directory / "predictions.npz"
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["ids"], train["id"].to_numpy())
            np.testing.assert_array_equal(data["fold"], folds)
            np.testing.assert_array_equal(data["test_ids"], test_ids)
            oof.append(probabilities(data["oof"], len(train), path))
            test.append(probabilities(data["test"], len(test_ids), path))
        np.testing.assert_allclose(roc_auc_score(train[LABEL], oof[-1]), metrics["oof_auc"], rtol=0, atol=1e-12)
        for source in (metrics_path, path):
            provenance[source.relative_to(PROJECT).as_posix()] = sha256(source)

    for name, file in (
        ("FINAL_PLUS_LGBM20", "01_FINAL_LGBM_D03_20pct_submission.csv"),
        ("FINAL_PLUS_MLP10", "02_FINAL_MLP_BCE_10pct_submission.csv"),
    ):
        path = FOLLOWUP / "submissions" / file
        prior = pd.read_csv(path)
        np.testing.assert_array_equal(prior["id"].to_numpy(), test_ids)
        np.testing.assert_allclose(prior[LABEL], blend(*test, weights=WEIGHTS[name]), rtol=0, atol=1e-14)
        provenance[path.relative_to(PROJECT).as_posix()] = sha256(path)
    return oof, test


def load_followup(seed, ids, folds, fold_hash, inputs, provenance):
    directory = FOLLOWUP / "results" / "full" / f"seed_{seed}"
    manifest_path = directory / "manifest.json"
    manifest = verified_manifest(manifest_path, inputs, fold_hash)
    configs = {name: sha256(BASE / name / "config.json") for name in ("GLM-NEWTON", "XGB-D03", "D03")}
    expected_blends = {name: [weights[0], weights[1] + weights[2]] for name, weights in WEIGHTS.items()
                       if name.startswith("FINAL_PLUS_")}
    if (manifest["seed"] != seed or manifest["rows"] != len(ids) or manifest["configs"] != configs
            or manifest["runner"] != sha256(FOLLOWUP / "run_followup.py")
            or manifest["mlp_epochs"] != 12 or manifest["blends"] != expected_blends):
        raise ValueError(f"Incompatible follow-up configuration in {manifest_path}")

    predictions_path = directory / "predictions.npz"
    with np.load(predictions_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["ids"], ids)
        np.testing.assert_array_equal(data["fold"], folds)
        archived = {name: probabilities(data[name], len(ids), predictions_path)
                    for name in WEIGHTS if name != "FIXED_BLEND"}
    for path in (manifest_path, predictions_path):
        provenance[path.relative_to(PROJECT).as_posix()] = sha256(path)

    lightgbm, mlp = np.empty(len(ids)), np.empty(len(ids))
    for fold in np.unique(folds):
        mask = folds == fold
        fold_dir = directory / f"fold_{fold + 1}"
        for name, destination in (("D03", lightgbm), ("MLP-BCE", mlp)):
            path = fold_dir / f"outer_{name}.npz"
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["ids"], ids[mask])
                destination[mask] = probabilities(data["prediction"], int(mask.sum()), path)
            provenance[path.relative_to(PROJECT).as_posix()] = sha256(path)
    components = (archived["FINAL"], lightgbm, mlp)
    for name, expected in archived.items():
        np.testing.assert_allclose(blend(*components, weights=WEIGHTS[name]), expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(blend(*components), archived["FINAL_PLUS_LGBM20"]
                               + archived["FINAL_PLUS_MLP10"] - archived["FINAL"], rtol=0, atol=1e-14)
    return components


def prepare_auc(labels, values):
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    tie_starts = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0) + 1]
    is_positive = labels[order].astype(bool)
    return order, tie_starts, is_positive


def auc_for_weights(prepared, weights):
    order, tie_starts, is_positive = prepared
    sorted_weights = weights[order]
    positive = np.add.reduceat(sorted_weights * is_positive, tie_starts)
    negative = np.add.reduceat(sorted_weights * ~is_positive, tie_starts)
    negatives_before_or_tied = np.cumsum(negative) - 0.5 * negative
    pairs = positive.sum() * negative.sum()
    return float(np.sum(positive * negatives_before_or_tied) / pairs)


def score_seed(labels, folds, predictions, repetitions, seed):
    metrics = {name: {**metric_row(labels, values), "fold_auc": [
        float(roc_auc_score(labels[folds == fold], values[folds == fold])) for fold in np.unique(folds)
    ]} for name, values in predictions.items()}
    baseline = metrics["FINAL"]
    for values in metrics.values():
        differences = np.asarray(values["fold_auc"]) - baseline["fold_auc"]
        values["delta_auc_vs_final"] = values["oof_auc"] - baseline["oof_auc"]
        values["delta_by_fold_vs_final"] = differences.tolist()
        values["fold_wins_vs_final"] = int(np.count_nonzero(differences > 0))

    comparisons = {
        "LGBM20_vs_FINAL": ("FINAL_PLUS_LGBM20", "FINAL"),
        "MLP10_vs_FINAL": ("FINAL_PLUS_MLP10", "FINAL"),
        "FIXED_BLEND_vs_FINAL": ("FIXED_BLEND", "FINAL"),
        "FIXED_BLEND_vs_LGBM20": ("FIXED_BLEND", "FINAL_PLUS_LGBM20"),
        "FIXED_BLEND_vs_MLP10": ("FIXED_BLEND", "FINAL_PLUS_MLP10"),
    }
    prepared = [prepare_auc(labels, predictions[name]) for name in predictions]
    bootstrap = np.empty((repetitions, len(prepared)))
    rng = np.random.default_rng(9101 + seed)
    class_rows = [np.flatnonzero(labels == label) for label in (0, 1)]
    weights = np.empty(len(labels), dtype=np.int64)
    for rep in range(repetitions):
        for rows in class_rows:
            weights[rows] = np.bincount(rng.integers(len(rows), size=len(rows)), minlength=len(rows))
        bootstrap[rep] = [auc_for_weights(order, weights) for order in prepared]

    columns = {name: index for index, name in enumerate(predictions)}
    contrasts = {}
    for name, (candidate, control) in comparisons.items():
        left, right = metrics[candidate], metrics[control]
        deltas = np.asarray(left["fold_auc"]) - right["fold_auc"]
        interval = np.quantile(bootstrap[:, columns[candidate]] - bootstrap[:, columns[control]], [0.025, 0.975])
        contrasts[name] = {
            "delta_auc": left["oof_auc"] - right["oof_auc"], "delta_by_fold": deltas.tolist(),
            "fold_wins": int(np.count_nonzero(deltas > 0)), "bootstrap_ci95": interval.tolist(),
        }
    return {"metrics": metrics, "contrasts": contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--bootstrap-reps", type=int, default=120)
    args = parser.parse_args()
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be positive")
    filenames = ["submission_fixed_blend.csv", "fixed_blend_predictions.npz", "fixed_blend_metrics.json"]
    output_files = [args.output / name for name in filenames]
    if any(path.exists() for path in output_files):
        parser.error("An output already exists. Choose a fresh --output directory for a repeat run.")

    inputs = {name: sha256(PROJECT.parent / "data" / name) for name in ("train.csv", "test.csv", "sample_submission.csv")}
    train = pd.read_csv(PROJECT.parent / 'data' / "train.csv", usecols=["id", LABEL])
    ids, labels = train["id"].to_numpy(), train[LABEL].to_numpy(dtype=np.int8)
    test_ids = pd.read_csv(PROJECT.parent / 'data' / "test.csv", usecols=["id"])["id"].to_numpy()
    sample = pd.read_csv(PROJECT.parent / 'data' / "sample_submission.csv")
    if list(sample) != ["id", LABEL] or not np.array_equal(sample["id"].to_numpy(), test_ids):
        raise ValueError("Official sample columns or test ID order changed")
    provenance = {}
    reports = {}
    for seed in SEEDS:
        assignment = assign_folds(train, LABEL, 5, seed)
        folds, fold_hash = assignment["fold"].to_numpy(), frame_hash(assignment)
        if seed == 42:
            components, test = load_seed42(train, test_ids, folds, fold_hash, inputs, provenance)
            test = probabilities(blend(*test), len(test_ids), "submission")
            archive = {"ids": ids, "test_ids": test_ids, "fold": folds, "test": test}
        else:
            components = load_followup(seed, ids, folds, fold_hash, inputs, provenance)
        scores = {name: probabilities(blend(*components, weights=weights), len(ids), name)
                  for name, weights in WEIGHTS.items()}
        if seed == 42:
            archive["oof"] = scores["FIXED_BLEND"]
        report = score_seed(labels, folds, scores, args.bootstrap_reps, seed)
        reports[str(seed)] = {"fold_hash": fold_hash, **report}
        for name, result in report["metrics"].items():
            print(f"Seed {seed} | {name:19} | AUC {result['oof_auc']:.9f} | "
                  f"delta {result['delta_auc_vs_final']:+.9f} | wins {result['fold_wins_vs_final']}/5", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    submission, predictions_path, metrics_path = output_files
    staged = submission.with_suffix(".csv.tmp")
    pd.DataFrame({"id": test_ids, LABEL: archive["test"]}).to_csv(staged, index=False)
    checked = pd.read_csv(staged)
    np.testing.assert_array_equal(checked["id"].to_numpy(), test_ids)
    np.testing.assert_allclose(checked[LABEL], archive["test"], rtol=0, atol=1e-14)
    staged.replace(submission)
    write_npz(predictions_path, **archive)
    write_json(metrics_path, {
        "method": "fixed_probability_blend", "model_order": ["FINAL", "D03", "MLP-BCE"],
        "weights": WEIGHTS, "models_use_llm": False, "inputs": inputs, "provenance": provenance,
        "script_sha256": sha256(Path(__file__)),
        "bootstrap": {"repetitions": args.bootstrap_reps, "seed_offset": 9101,
                      "design": "paired and class-stratified, conditional on fixed OOF scores"},
        "seeds": reports, "submission_seed": 42, "submission_rows": len(test_ids),
        "submission_sha256": sha256(submission), "predictions_sha256": sha256(predictions_path),
    })


if __name__ == "__main__":
    main()
