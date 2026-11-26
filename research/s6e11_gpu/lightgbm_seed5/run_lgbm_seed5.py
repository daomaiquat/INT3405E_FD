"""Refit a five-seed LightGBM D03 mean using all competition training rows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
GPU = HERE.parent
RESULTS = HERE / "results"
NAME = "LGBM-SEED5-MEAN"
SEEDS = ((42, 2026), (71, 2026), (42, 2028), (71, 2028), (103, 2032))

sys.path.insert(0, str(GPU / "blend_full_refit"))
from run_blend_refits import BASE, SINGLE, load_member
from run_full_refit import Experiment, PlanRun, probabilities, save_submission
from models import fit_predict, predict_from_state
from run_plan import read_json, sha256, write_json, write_npz


def fit_member(run, spec, rounds, matrices):
    x_train, x_test = matrices
    started = perf_counter()
    (raw,), fit = fit_predict(
        spec.family, x_train, run.y, (x_test,),
        params={**spec.params, "include_model_state": True}, seed=spec.model_seed, rounds=rounds,
    )
    if (
        fit["rounds"] != rounds
        or "RTX 4060" not in fit["device"]
        or fit["settings"]["device_type"] != "gpu"
        or fit["settings"]["seed"] != spec.model_seed
    ):
        raise RuntimeError(f"Unexpected GPU or fit settings for {spec.name}")
    prediction = probabilities(raw, len(run.test_ids))
    state = fit.pop("model_state")
    directory = RESULTS / spec.name
    directory.mkdir(exist_ok=True)
    model = directory / "model.txt"
    temporary = model.with_suffix(".txt.tmp")
    temporary.write_bytes(state.encode("utf-8"))
    temporary.replace(model)
    (replayed,) = predict_from_state("lgb", model.read_text(encoding="utf-8"), (x_test,))
    error = float(np.abs(prediction - replayed).max())
    if not np.isfinite(error) or error > 1e-6:
        raise RuntimeError(f"Could not replay {spec.name}: absolute error {error}")
    data = directory / "predictions.npz"
    write_npz(data, id=run.test_ids, full=prediction)
    report = {
        "name": spec.name,
        "model_seed": spec.model_seed,
        "encoder_seed": spec.encoder_seed,
        "training_rows": len(x_train),
        "fit": fit,
        "replay_max_abs_delta": error,
        "model_sha256": sha256(model),
        "predictions_sha256": sha256(data),
        "predictions": str(data.relative_to(GPU)),
        "seconds": perf_counter() - started,
    }
    write_json(directory / "metrics.json", report)
    print(f"{spec.name}: {len(x_train):,} rows, {fit['device']} | {report['seconds']:.1f}s", flush=True)
    return prediction, report


def compare(test, reference):
    return {
        "spearman": float(spearmanr(test, reference).statistic),
        "mean_abs_probability_delta": float(np.mean(np.abs(test - reference))),
    }


def main():
    run = PlanRun(RESULTS)
    config = read_json(BASE / "D03" / "config.json")
    source = Experiment(**{**config, "groups": tuple(config["groups"])})
    if (source.model_seed, source.encoder_seed) != SEEDS[0] or source.params["num_leaves"] != 4:
        raise ValueError("D03 reference configuration changed")
    (anchor, _), anchor_info = load_member(run, "D03", BASE, SINGLE / "results")
    rounds = anchor_info["rounds"]
    members = [{
        **anchor_info,
        "name": "D03",
        "model_seed": source.model_seed,
        "encoder_seed": source.encoder_seed,
        "training_rows": len(run.y),
        "reused_full_refit": True,
    }]
    predictions = [anchor]
    reference = np.arange(len(run.y), dtype=np.int64)
    last_encoder, matrices = None, None
    for model_seed, encoder_seed in SEEDS[1:]:
        name = f"LGB-D03-S{model_seed}-E{encoder_seed}"
        spec = replace(source, name=name, model_seed=model_seed, encoder_seed=encoder_seed)
        print(f"{name}: fitting {len(reference):,} rows with {rounds:,} boosting rounds", flush=True)
        if encoder_seed != last_encoder:
            matrices = run.matrices(spec, reference, (run.test_indices,))
            last_encoder = encoder_seed
        scores, report = fit_member(run, spec, rounds, matrices)
        members.append(report)
        predictions.append(scores)
    if any(np.array_equal(left, right) for i, left in enumerate(predictions) for right in predictions[i + 1:]):
        raise RuntimeError("Two seed pairs yielded identical full-train predictions")
    final = probabilities(np.mean(predictions, axis=0), len(run.test_ids))
    directory = RESULTS / NAME
    directory.mkdir(exist_ok=True)
    submission = directory / f"{NAME}_FULLREFIT_submission.csv"
    save_submission(submission, run.test_ids, final)
    write_npz(directory / "predictions.npz", id=run.test_ids, full=final)
    xgb_path = GPU / "blend_full_refit" / "results" / "SEED5-MEAN" / "predictions.npz"
    with np.load(xgb_path, allow_pickle=False) as xgb:
        np.testing.assert_array_equal(xgb["id"], run.test_ids)
        xgb_scores = probabilities(xgb["full"], len(run.test_ids))
    report = {
        "name": NAME,
        "input_manifest": run.manifest,
        "base_config": config,
        "num_boost_rounds_per_member": rounds,
        "rounds_source": "median of D03's original five fold refits",
        "members": members,
        "weights": [0.2] * len(members),
        "test_vs_D03_full": compare(final, anchor),
        "test_vs_XGB_SEED5_full": compare(final, xgb_scores),
        "submission": str(submission.relative_to(HERE)),
        "submission_sha256": sha256(submission),
    }
    write_json(directory / "metrics.json", report)
    print(f"{NAME}: {len(final):,} test predictions | {submission}", flush=True)


if __name__ == "__main__":
    main()
