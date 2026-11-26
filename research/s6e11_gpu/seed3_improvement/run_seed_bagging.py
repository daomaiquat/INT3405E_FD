"""Compare fixed seed bags on the original five-fold XGB-D03 protocol."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / "plan_runs"
REFERENCE = PLAN / "results" / "full"
OUTPUT = HERE / "results"
BASELINE = "SEED3-MEAN"
ORIGINAL = ("XGB-D03", "XGB-D03-S71", "XGB-D03-S93")
CROSS = "XGB-D03-S71-E2028"
INDEPENDENT = "XGB-D03-S103-E2032"
BAGS = {
    "SEED4-CROSS-MEAN": (*ORIGINAL, CROSS),
    "SEED4-NEW-MEAN": (*ORIGINAL, INDEPENDENT),
    "SEED5-MEAN": (*ORIGINAL, CROSS, INDEPENDENT),
}

sys.path.insert(0, str(PLAN))
from ensemble import bootstrap_delta
from run_plan import Experiment, PlanRun, metric_row, read_json, write_json


def load_reference(run, name):
    directory = REFERENCE / name
    metrics = read_json(directory / "metrics.json")
    if metrics["name"] != name or metrics["status"] != "completed":
        raise ValueError(f"Invalid reference metadata for {name}")
    for key, value in run.manifest.items():
        if metrics.get(key) != value:
            raise ValueError(f"{name} has a different {key}")
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        for key, values in (("ids", run.ids), ("test_ids", run.test_ids), ("fold", run.folds)):
            if not np.array_equal(saved[key], values):
                raise ValueError(f"Reference {name} has different {key}")
        oof, test = saved["oof"], saved["test"]
    for scores, size in ((oof, len(run.ids)), (test, len(run.test_ids))):
        if scores.shape != (size,) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"Invalid prediction probabilities for {name}")
    return (oof, test), metrics


def load_experiment():
    config = read_json(REFERENCE / ORIGINAL[0] / "config.json")
    config["groups"] = tuple(config["groups"])
    anchor = Experiment(**config)
    return (
        replace(anchor, name=CROSS, parent=BASELINE, model_seed=71, encoder_seed=2028),
        replace(anchor, name=INDEPENDENT, parent=BASELINE, model_seed=103, encoder_seed=2032),
    )


def load_baseline(run):
    original = {name: load_reference(run, name)[0] for name in ORIGINAL}
    baseline, metrics = load_reference(run, BASELINE)
    config = metrics["config"]
    if config["members"] != list(ORIGINAL) or config["rank"] or config["weights"] != [1 / 3] * 3:
        raise ValueError("The archived baseline does not match the experiment design")
    for column, saved in enumerate(baseline):
        recomputed = np.average([scores[column] for scores in original.values()], axis=0, weights=config["weights"])
        np.testing.assert_allclose(saved, recomputed, rtol=0, atol=1e-15)
    np.testing.assert_allclose(metric_row(run.y, baseline[0])["oof_auc"], metrics["oof_auc"], rtol=0, atol=1e-12)
    return original, baseline, metrics


def score_bags(run, original, baseline, base_metrics):
    predictions = {**original, CROSS: run.load(CROSS), INDEPENDENT: run.load(INDEPENDENT)}
    rows = []
    for name, members in BAGS.items():
        oof, test = (np.mean([predictions[member][column] for member in members], axis=0) for column in (0, 1))
        metrics = run.save_predictions(name, oof, test, {
            "members": list(members),
            "weights": [1 / len(members)] * len(members),
            "parent": BASELINE,
            "rank": False,
        })
        gains = [new["auc"] - old["auc"] for new, old in zip(metrics["folds"], base_metrics["folds"], strict=True)]
        print(f"{name}: paired bootstrap in progress", flush=True)
        interval = bootstrap_delta(run.y, oof, baseline[0], repeats=500)
        interval.pop("samples")
        row = {
            "name": name, "members": list(members),
            **{key: metrics[key] for key in ("oof_auc", "average_precision", "log_loss")},
            "delta_oof_auc": metrics["oof_auc"] - base_metrics["oof_auc"],
            "delta_by_fold": gains,
            "folds_up": sum(gain > 0 for gain in gains),
            "bootstrap": interval,
        }
        row["preferred"] = row["delta_oof_auc"] >= 0.0001 and row["folds_up"] >= 4 and interval["ci_low"] > 0
        rows.append(row)
    preferred = max((row for row in rows if row["preferred"]), key=lambda row: row["oof_auc"], default=None)
    summary = {
        "baseline": base_metrics,
        "new_members": [read_json(run.output / name / "metrics.json") for name in (CROSS, INDEPENDENT)],
        "candidates": rows,
        "primary_candidate": "SEED5-MEAN",
        "decision_rule": "OOF AUC gain of at least 0.0001, at least four folds up and positive paired 95% interval",
        "selected": preferred["name"] if preferred else BASELINE,
    }
    write_json(run.output / "evaluation.json", summary)
    print(f"Selected {summary['selected']}", flush=True)
    return summary


def main():
    run = PlanRun(OUTPUT)
    original, baseline, metrics = load_baseline(run)
    for spec in load_experiment():
        run.run(spec)
    score_bags(run, original, baseline, metrics)


if __name__ == "__main__":
    main()
