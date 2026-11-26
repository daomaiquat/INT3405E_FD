"""Replay the selected blends using individually fitted full-train base models."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
GPU_ROOT = HERE.parent
RESULTS = HERE / "results"
SINGLE = GPU_ROOT / "single_full_refit"
NEW_BASE = RESULTS / "base_models"
sys.path.insert(0, str(SINGLE))

from run_full_refit import BASE, SEED, PlanRun, load_reference, probabilities, run_models, save_submission
from ensemble import _logit
from models import predict_from_state
from run_plan import read_json, sha256, write_json, write_npz

ANCHOR = "XGB-D03"
GLM = "GLM-NEWTON"
LIGHTGBM = "D03"
SEED3 = (ANCHOR, "XGB-D03-S71", "XGB-D03-S93")
SEED5 = (*SEED3, "XGB-D03-S71-E2028", "XGB-D03-S103-E2032")
FINAL = (GLM, ANCHOR)
FINAL3 = (GLM, LIGHTGBM, ANCHOR)
NEW_MEMBERS = {SEED3[1]: BASE, SEED3[2]: BASE, SEED5[4]: SEED, GLM: BASE}
ARCHIVED_REFITS = {ANCHOR: BASE, SEED5[3]: SEED, LIGHTGBM: BASE}


def load_member(run, name, source, refits):
    _, _, old, source_path = load_reference(run, name, source)
    directory = refits / name
    checkpoint = directory / "predictions.npz"
    record = read_json(directory / "metrics.json")
    protocol = record["protocol"]
    model = directory / ("model.txt" if protocol["spec"]["family"] == "lgb" else "model.json")
    if (
        protocol["name"] != name
        or protocol["input_manifest"] != run.manifest
        or protocol["reference_metrics_sha256"] != sha256(source_path / "metrics.json")
        or protocol["reference_predictions_sha256"] != sha256(source_path / "predictions.npz")
        or protocol["rounds"] != record["fit"]["rounds"]
        or "RTX 4060" not in record["fit"]["device"]
        or record["predictions_sha256"] != sha256(checkpoint)
        or record["model_sha256"] != sha256(model)
    ):
        raise ValueError(f"Incompatible full refit for {name}")
    with np.load(checkpoint, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["id"], run.test_ids)
        np.testing.assert_array_equal(saved["fold_mean"], old)
        full = probabilities(saved["full"], len(old))
    source_info = {
        "predictions": str(checkpoint.relative_to(GPU_ROOT)),
        "predictions_sha256": record["predictions_sha256"],
        "model_sha256": record["model_sha256"],
        "device": record["fit"]["device"],
        "rounds": record["fit"]["rounds"],
    }
    return (full, old), source_info


def stack(members, vectors, state):
    x = _logit(np.column_stack([vectors[name] for name in members]))
    return predict_from_state("glm", state, (x,))[0]


def mean(members, vectors):
    return np.mean([vectors[name] for name in members], axis=0)


def archived_vector(run, directory):
    with np.load(directory / "predictions.npz", allow_pickle=False) as source:
        np.testing.assert_array_equal(source["test_ids"], run.test_ids)
        return probabilities(source["test"], len(run.test_ids))


def archived_csv(run, path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        if next(reader) != ["id", "is_fraud"]:
            raise ValueError(f"Unexpected headers in {path}")
        values = []
        for row, expected in zip(reader, run.test_ids, strict=True):
            if len(row) != 2 or int(row[0]) != expected:
                raise ValueError(f"Unexpected ID in {path}")
            values.append(float(row[1]))
    return probabilities(values, len(run.test_ids))


def export_blends(run):
    new, old, provenance = {}, {}, {}
    for members, refits in ((ARCHIVED_REFITS, SINGLE / "results"), (NEW_MEMBERS, NEW_BASE)):
        for name, source in members.items():
            (new[name], old[name]), provenance[name] = load_member(run, name, source, refits)

    selection = read_json(BASE / "final" / "selection.json")
    development = read_json(BASE / "ensemble" / "ensemble_summary.json")
    if (
        selection["selected"] != "NESTED-STACK"
        or selection["nested"]["members"] != list(FINAL)
        or selection["nested"]["stack"]["method"] != "logistic_stack"
        or development["test_refit_models"] != list(FINAL3)
        or development["winner"] != "logistic_stack_seed8_c0.1"
        or development["test_refit"] != "all_base_oof"
    ):
        raise ValueError("Stored stack definitions differ from the approved blends")
    state_final = selection["stack_fit_on_all_base_oof"]["model_state"]
    state_final3 = development["test_refit_details"]["model_state"]
    seed5_dir = SEED / "SEED5-MEAN"
    seed3_dir = BASE / "SEED3-MEAN"
    final3_csv = GPU_ROOT / "final_three_stack" / "FINAL3_GLM-NEWTON_LGBM-D03_XGB-D03_submission.csv"
    plus_csv = GPU_ROOT / "followup" / "submissions" / "01_FINAL_LGBM_D03_20pct_submission.csv"
    archived = {
        "SEED5-MEAN": archived_vector(run, seed5_dir),
        "FINAL": archived_vector(run, BASE / "final"),
        "FINAL_PLUS_LGBM20": archived_csv(run, plus_csv),
        "FINAL3": archived_csv(run, final3_csv),
        "SEED3-MEAN": archived_vector(run, seed3_dir),
    }
    np.testing.assert_allclose(mean(SEED5, old), archived["SEED5-MEAN"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(mean(SEED3, old), archived["SEED3-MEAN"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(stack(FINAL3, old, state_final3), archived["FINAL3"], rtol=0, atol=1e-14)
    drift = selection["model_bundle"]["max_delta_vs_original_test_prediction"]
    np.testing.assert_allclose(stack(FINAL, old, state_final), archived["FINAL"], rtol=0, atol=drift + 1e-12)
    np.testing.assert_allclose(0.8 * archived["FINAL"] + 0.2 * old[LIGHTGBM],
                               archived["FINAL_PLUS_LGBM20"], rtol=0, atol=1e-14)

    refit_final = stack(FINAL, new, state_final)
    scores = {
        "SEED5-MEAN": (SEED5, mean(SEED5, new), {"member_weights": [0.2] * 5}),
        "FINAL": (FINAL, refit_final, {"meta_model_state": state_final}),
        "FINAL_PLUS_LGBM20": (FINAL3, 0.8 * refit_final + 0.2 * new[LIGHTGBM],
                             {"weights": {"FINAL": 0.8, LIGHTGBM: 0.2}, "final_meta_model_state": state_final}),
        "FINAL3": (FINAL3, stack(FINAL3, new, state_final3), {"meta_model_state": state_final3}),
        "SEED3-MEAN": (SEED3, mean(SEED3, new), {"member_weights": [1 / 3] * 3}),
    }
    oof_reference = {
        "SEED5-MEAN": read_json(seed5_dir / "metrics.json")["oof_auc"],
        "FINAL": selection["oof_auc"],
        "FINAL3": development["winner_auc"],
        "SEED3-MEAN": read_json(seed3_dir / "metrics.json")["oof_auc"],
    }
    reports = []
    for name, (members, predicted, definition) in scores.items():
        prediction = probabilities(predicted, len(run.test_ids))
        prior = archived[name]
        directory = RESULTS / name
        directory.mkdir(exist_ok=True)
        submission = directory / f"{name}_FULLREFIT_submission.csv"
        save_submission(submission, run.test_ids, prediction)
        write_npz(directory / "predictions.npz", id=run.test_ids, full=prediction, fold_mean=prior)
        report = {
            "name": name,
            "input_manifest": run.manifest,
            "members": [dict(provenance[member], name=member) for member in members],
            "blend": definition,
            "reference_oof_auc": oof_reference.get(name),
            "test_spearman_vs_fold_mean": float(spearmanr(prediction, prior).statistic),
            "test_mean_abs_delta_vs_fold_mean": float(np.abs(prediction - prior).mean()),
            "submission": str(submission.relative_to(HERE)),
            "submission_sha256": sha256(submission),
        }
        write_json(directory / "metrics.json", report)
        reports.append(report)
        print(f"{name}: {len(prediction):,} scores | Spearman {report['test_spearman_vs_fold_mean']:.6f}", flush=True)
    write_json(RESULTS / "summary.json", reports)


if __name__ == "__main__":
    run = PlanRun(NEW_BASE)
    run_models(run, NEW_MEMBERS)
    export_blends(run)
