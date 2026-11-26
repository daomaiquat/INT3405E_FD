"""Evaluate a fixed 90/10 log-odds blend of locally trained FINAL and spline predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold

from alternative_model import CV_SEED, N_FOLDS, TARGET, score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
HISTORICAL = PROJECT / "s6e11_gpu" / "plan_runs" / "results" / "full" / "final"
SPLINE_WEIGHT = 0.10
CLIP = 1e-12


def mix_probabilities(final, spline):
    left = logit(np.clip(final.astype(np.float64, copy=False), CLIP, 1.0 - CLIP))
    right = logit(np.clip(spline.astype(np.float64, copy=False), CLIP, 1.0 - CLIP))
    return expit((1.0 - SPLINE_WEIGHT) * left + SPLINE_WEIGHT * right)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def assert_probabilities(vector, size, description):
    if vector.shape != (size,) or not np.isfinite(vector).all() or np.any((vector < 0) | (vector > 1)):
        raise ValueError(f"Invalid probabilities for {description}.")


def run(data_dir, output_dir, historical_dir, spline_dir):
    dataset = {name: data_dir / name for name in ("train.csv", "test.csv", "sample_submission.csv")}
    input_digests = {name: digest(path) for name, path in dataset.items()}
    train = pd.read_csv(dataset["train.csv"], usecols=["id", TARGET])
    test_ids = pd.read_csv(dataset["test.csv"], usecols=["id"])["id"].to_numpy()
    sample = pd.read_csv(dataset["sample_submission.csv"])
    train_ids = train["id"].to_numpy()
    labels = train[TARGET].to_numpy(dtype=np.int8)
    if list(sample) != ["id", TARGET] or not np.array_equal(sample["id"], test_ids):
        raise ValueError("Submission template columns or ID order differ from the official test data.")
    if not train["id"].is_unique or not sample["id"].is_unique:
        raise ValueError("Train and test IDs must be unique.")

    final_metrics_file = historical_dir / "metrics.json"
    spline_metrics_file = spline_dir / "spline_interactions_metrics.json"
    final_metrics = json.loads(final_metrics_file.read_text(encoding="utf-8"))
    spline_metrics = json.loads(spline_metrics_file.read_text(encoding="utf-8"))
    if (final_metrics["name"] != "FINAL" or final_metrics["status"] != "completed"
            or final_metrics["smoke"] is not False or final_metrics["models_use_llm"] is not False
            or final_metrics["inputs"] != input_digests):
        raise ValueError("Historical FINAL was not built from the expected competition data.")

    spline_fingerprint = hashlib.sha256((HERE / "alternative_model.py").read_bytes())
    for value in input_digests.values():
        spline_fingerprint.update(bytes.fromhex(value))
    if (spline_metrics["source_and_inputs_digest"] != spline_fingerprint.hexdigest()
            or spline_metrics["cv_seed"] != CV_SEED or spline_metrics["n_folds"] != N_FOLDS):
        raise ValueError("Spline model source, data, or split differs from the trained model.")

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    fold = np.full(len(train), -1, dtype=np.int8)
    for fold_number, (_, validation) in enumerate(splitter.split(train_ids, labels)):
        fold[validation] = fold_number

    historical_path = historical_dir / "predictions.npz"
    spline_path = spline_dir / "spline_interactions_predictions.npz"
    with np.load(historical_path, allow_pickle=False) as historical, np.load(spline_path, allow_pickle=False) as spline:
        np.testing.assert_array_equal(historical["ids"], train_ids)
        np.testing.assert_array_equal(spline["train_ids"], train_ids)
        np.testing.assert_array_equal(historical["test_ids"], test_ids)
        np.testing.assert_array_equal(spline["test_ids"], test_ids)
        np.testing.assert_array_equal(historical["fold"], fold)
        final_oof, final_test = historical["oof"], historical["test"]
        spline_oof, spline_test = spline["oof"], spline["test"]
    for name, values, rows in (
        ("FINAL OOF", final_oof, len(train)), ("FINAL test", final_test, len(test_ids)),
        ("Spline OOF", spline_oof, len(train)), ("Spline test", spline_test, len(test_ids)),
    ):
        assert_probabilities(values, rows, name)

    base_metrics = score(labels, final_oof)
    np.testing.assert_allclose(base_metrics["roc_auc"], final_metrics["oof_auc"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        score(labels, spline_oof)["roc_auc"], spline_metrics["spline_interactions"]["roc_auc"], rtol=0, atol=1e-12,
    )
    oof = mix_probabilities(final_oof, spline_oof)
    test = mix_probabilities(final_test, spline_test)
    combined_metrics = score(labels, oof)
    fold_metrics = []
    for number in range(N_FOLDS):
        rows = fold == number
        final_score = score(labels[rows], final_oof[rows])
        combined = score(labels[rows], oof[rows])
        fold_metrics.append({
            "fold": number + 1,
            "historical_final": final_score,
            "spline_blend": combined,
            "delta_roc_auc": combined["roc_auc"] - final_score["roc_auc"],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    submission = output_dir / "submission_spline_blend.csv"
    staged_submission = submission.with_suffix(".csv.tmp")
    sample[TARGET] = test
    sample.to_csv(staged_submission, index=False)
    written = pd.read_csv(staged_submission)
    np.testing.assert_array_equal(written["id"], test_ids)
    np.testing.assert_allclose(written[TARGET], test, rtol=0, atol=1e-14)
    staged_submission.replace(submission)

    predictions = output_dir / "spline_blend_predictions.npz"
    np.savez_compressed(predictions, ids=train_ids, fold=fold, oof=oof, test_ids=test_ids, test=test)
    summary = {
        "name": "spline_blend",
        "method": "fixed_log_odds_blend",
        "exploratory": True,
        "selection_note": "The 10% weight was selected after looking at fold 1. Treat the results as exploratory.",
        "weights": {"historical_final": 1.0 - SPLINE_WEIGHT, "spline_interactions": SPLINE_WEIGHT},
        "logit_clipping_epsilon": CLIP,
        "cv_seed": CV_SEED,
        "n_folds": N_FOLDS,
        "models_use_llm": False,
        "inputs": input_digests,
        "source_prediction_digests": {"historical_final": digest(historical_path), "spline": digest(spline_path)},
        "historical_final": base_metrics,
        "spline_blend": combined_metrics,
        "delta_roc_auc": combined_metrics["roc_auc"] - base_metrics["roc_auc"],
        "fold_wins": sum(result["delta_roc_auc"] > 0 for result in fold_metrics),
        "folds": fold_metrics,
        "submission_sha256": digest(submission),
        "predictions_sha256": digest(predictions),
    }
    (output_dir / "spline_blend_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("spline_blend", "delta_roc_auc", "fold_wins")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=PROJECT.parent / "data", type=Path)
    parser.add_argument("--output", default=HERE / "results", type=Path)
    parser.add_argument("--historical-dir", default=HISTORICAL, type=Path)
    parser.add_argument("--spline-dir", default=HERE, type=Path)
    args = parser.parse_args()
    run(args.data_dir, args.output, args.historical_dir, args.spline_dir)


if __name__ == "__main__":
    main()
