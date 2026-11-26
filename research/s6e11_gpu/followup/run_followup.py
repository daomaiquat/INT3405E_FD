"""Compare the frozen FINAL recipe, two probability blends and a no-flags adjustment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PLAN = REPO / "s6e11_gpu" / "plan_runs"
IMBALANCE = PLAN.parent / "imbalance"
sys.path[:0] = [str(PLAN), str(IMBALANCE)]

from ensemble import fit_meta_fold
from features import FeatureBuilder
from mlp_bce_focal_and_pairwise import EPOCHS, config_for, cuda_device, fit_fold as fit_mlp_fold
from run_plan import (
    CONTROL_PATH, Experiment, PlanRun, TARGET, assign_folds, frame_hash, locate_data,
    metric_row, read_json, sha256, write_json, write_npz,
)

SEEDS = (1337, 2718)
INNER_SEED = 2026
META_C = 0.1
SCORES = ("FINAL", "FINAL_PLUS_LGBM20", "FINAL_PLUS_MLP10", "FINAL_NO_FLAGS")


class FollowupRun(PlanRun):
    """Use the existing fitting methods with a fresh, validation-only partition."""

    def __init__(self, train, config, builder, seed, smoke):
        self.train, self.config, self.builder, self.smoke = train, config, builder, smoke
        self.y = train[TARGET].to_numpy(dtype=np.int8)
        self.ids = train["id"].to_numpy()
        assignments = assign_folds(train, TARGET, 2 if smoke else 5, seed)
        self.fold_hash = frame_hash(assignments)
        self.folds = assignments["fold"].to_numpy()
        self.partitions = [
            self.make_partition(np.flatnonzero(self.folds != fold), np.flatnonzero(self.folds == fold))
            for fold in np.unique(self.folds)
        ]


def cached_base(run, part, spec, path, *, mlp_config=None, device=None):
    valid = part["V"]
    if path.is_file():
        with np.load(path) as data:
            if not np.array_equal(data["ids"], run.ids[valid]):
                raise ValueError(f"Cached validation IDs changed in {path}")
            prediction = data["prediction"]
    else:
        print(f"  {path.stem} | training", flush=True)
        if mlp_config is None:
            (prediction,), details = run.fit_fold(spec, part, (valid,))
        else:
            fit, predict = run.matrices(spec, part["T"], (valid,))
            scaler = StandardScaler(copy=False).fit(fit)
            torch_device, device_name = device
            fit_gpu = torch.as_tensor(scaler.transform(fit), dtype=torch.float32, device=torch_device)
            valid_gpu = torch.as_tensor(scaler.transform(predict), dtype=torch.float32, device=torch_device)
            (prediction, _), details = fit_mlp_fold(
                mlp_config, int(run.folds[valid[0]]) + 1, torch_device, fit_gpu,
                run.y[part["T"]], valid_gpu, valid_gpu[:1], device_name,
            )
        write_npz(path, ids=run.ids[valid], prediction=prediction)
        print(f"  {path.stem} | {details['seconds']:.1f}s | {details['device']}", flush=True)
    if (prediction.shape != (len(valid),) or not np.isfinite(prediction).all()
            or ((prediction < 0) | (prediction > 1)).any()):
        raise ValueError(f"Invalid predictions in {path}")
    return prediction


def score_fold(run, part, specs, directory, mlp_config, device):
    from no_flag_specialist import fit_adjust_no_flags

    glm, xgb, lgb = specs
    train, valid = part["T"], part["V"]
    outer = np.column_stack([
        cached_base(run, part, spec, directory / f"outer_{spec.name}.npz") for spec in (glm, xgb)
    ])
    inner = np.full((len(train), 2), np.nan)
    cv = StratifiedKFold(n_splits=2 if run.smoke else 3, shuffle=True, random_state=INNER_SEED)
    splits = list(cv.split(train, run.y[train]))
    for number, (fit, holdout) in enumerate(splits, start=1):
        inner_part = run.make_partition(train[fit], train[holdout])
        for column, spec in enumerate((glm, xgb)):
            inner[holdout, column] = cached_base(
                run, inner_part, spec, directory / f"inner_{number}_{spec.name}.npz",
            )

    (final,) = fit_meta_fold(inner, run.y[train], outer, c=META_C)
    # Both the base models and the meta fit exclude the current outer validation fold.
    # For training the adjustment, meta fits also exclude the scored inner rows' labels.
    meta_oof = np.full(len(train), np.nan)
    for fit, holdout in splits:
        (meta_oof[holdout],) = fit_meta_fold(inner[fit], run.y[train[fit]], inner[holdout], c=META_C)
    adjusted = fit_adjust_no_flags(
        run.train.iloc[train], meta_oof, run.train.iloc[valid].drop(columns=TARGET), final,
    )
    lgb_prediction = cached_base(run, part, lgb, directory / "outer_D03.npz")
    mlp = cached_base(run, part, xgb, directory / "outer_MLP-BCE.npz",
                      mlp_config=mlp_config, device=device)
    return (final, 0.8 * final + 0.2 * lgb_prediction, 0.9 * final + 0.1 * mlp, adjusted)


def summarize(run, values):
    quiet = run.train[run.config["flag_columns"]].eq(0).all(axis=1).to_numpy()
    labels = run.y
    result = {}
    for name, prediction in zip(SCORES, values, strict=True):
        metrics = metric_row(labels, prediction)
        metrics["no_flags_auc"] = float(roc_auc_score(labels[quiet], prediction[quiet]))
        metrics["fold_auc"] = [float(roc_auc_score(labels[m], prediction[m]))
                               for fold in np.unique(run.folds) for m in [run.folds == fold]]
        if name != "FINAL":
            baseline = result["FINAL"]
            metrics["delta_auc"] = metrics["oof_auc"] - baseline["oof_auc"]
            metrics["delta_by_fold"] = [score - ref for score, ref in
                                        zip(metrics["fold_auc"], baseline["fold_auc"], strict=True)]
            metrics["fold_wins"] = sum(delta > 0 for delta in metrics["delta_by_fold"])
        result[name] = metrics
    return result


def run_seed(run, seed, specs, output, provenance, mlp_config, device):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {**provenance, "seed": seed, "fold_hash": run.fold_hash, "smoke": run.smoke,
                "rows": len(run.y), "outer_folds": len(run.partitions),
                "inner_folds": 2 if run.smoke else 3, "inner_seed": INNER_SEED, "meta_c": META_C}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and manifest != read_json(manifest_path):
        raise ValueError(f"Input provenance changed for {output}")
    write_json(manifest_path, manifest)
    oof = np.full((len(SCORES), len(run.y)), np.nan)
    for fold, part in enumerate(run.partitions, start=1):
        directory = output / f"fold_{fold}"
        directory.mkdir(exist_ok=True)
        valid = part["V"]
        print(f"Seed {seed}, fold {fold}/{len(run.partitions)}", flush=True)
        predictions_file = directory / "predictions.npz"
        if predictions_file.exists():
            with np.load(predictions_file) as data:
                if not np.array_equal(data["ids"], run.ids[valid]):
                    raise ValueError(f"Cached fold IDs changed in {predictions_file}")
                values = [data[name] for name in SCORES]
        else:
            values = score_fold(run, part, specs, directory, mlp_config, device)
            write_npz(predictions_file, ids=run.ids[valid], **dict(zip(SCORES, values, strict=True)))
        oof[:, valid] = values
        base = roc_auc_score(run.y[valid], values[0])
        print("  AUC " + ", ".join(f"{name}={roc_auc_score(run.y[valid], value):.6f}"
                                   for name, value in zip(SCORES, values, strict=True))
              + f" | FINAL reference {base:.6f}", flush=True)
    if not np.isfinite(oof).all() or ((oof < 0) | (oof > 1)).any():
        raise ValueError(f"Incomplete OOF predictions for seed {seed}")
    write_npz(output / "predictions.npz", ids=run.ids, fold=run.folds,
              **dict(zip(SCORES, oof, strict=True)))
    report = summarize(run, oof)
    write_json(output / "metrics.json", report)
    for name, metrics in report.items():
        print(f"Seed {seed} | {name} | AUC {metrics['oof_auc']:.8f}"
              f" | delta {metrics.get('delta_auc', 0):+.8f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    data_dir = locate_data()
    snapshot = read_json(CONTROL_PATH)["T01"]["input_files"]
    input_hashes = {name: sha256(data_dir / name) for name in snapshot}
    if any(input_hashes[name] != row["sha256"] for name, row in snapshot.items()):
        raise ValueError("Official input files differ from the original model snapshot")
    train = pd.read_csv(data_dir / "train.csv")
    config = read_json(REPO / "configs" / "default.json")
    if args.smoke:
        train = train.groupby(TARGET, group_keys=False).sample(frac=0.03, random_state=42).sort_index()
    builder = FeatureBuilder(train, train.iloc[:0].drop(columns=TARGET))
    config_paths = [PLAN / "published" / "full" / name / "config.json"
                    for name in ("GLM-NEWTON", "XGB-D03", "D03")]
    specs = [Experiment(**read_json(path)) for path in config_paths]
    device = cuda_device()
    torch.set_float32_matmul_precision("high")
    mlp_config = config_for("MLP-BCE", specs[1], 2 if args.smoke else EPOCHS, sha256(config_paths[1]))
    provenance = {
        "inputs": input_hashes, "models_use_llm": False, "configs": {path.parent.name: sha256(path)
                                                                    for path in config_paths},
        "runner": sha256(Path(__file__)), "specialist": sha256(HERE / "no_flag_specialist.py"),
        "torch_version": str(torch.__version__), "mlp_epochs": mlp_config["epochs"],
        "blends": {"FINAL_PLUS_LGBM20": [0.8, 0.2], "FINAL_PLUS_MLP10": [0.9, 0.1]},
        "gpu": device[1],
    }
    for seed in args.seed:
        run = FollowupRun(train, config, builder, seed, args.smoke)
        output = HERE / "results" / ("smoke" if args.smoke else "full") / f"seed_{seed}"
        run_seed(run, seed, specs, output, provenance, mlp_config, device)


if __name__ == "__main__":
    main()
