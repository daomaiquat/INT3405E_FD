"""Run fixed-round XGBoost imbalance experiments on the local RTX 4060."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / "plan_runs"
sys.path.insert(0, str(PLAN))

from finalize import INNER_VALIDATION_SEED
from models import _cuda_name
from run_plan import PlanRun, get_specs, locate_data, read_json, sha256, write_json, write_npz


BASELINE = "XGB-D03"
SOURCE = PLAN / "results" / "full"
OUTPUT = HERE / "results" / "xgb"
VARIANTS = {"XGB-W2": 2, "XGB-W4": 4, "XGB-HARD": 1, "XGB-RANDOM": 1}
NEGATIVE_FRACTION = 0.10
NEGATIVE_WEIGHT = 3.0
RANDOM_SEED = 42
SMOKE_ROUNDS = 12


def _digest_array(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _probabilities(values, length, description):
    values = np.asarray(values, dtype=np.float64)
    if (values.shape != (length,) or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))):
        raise ValueError(f"Invalid probabilities for {description}")
    return values


def _inner_splits(run, outer_train):
    splitter = StratifiedKFold(
        n_splits=len(run.partitions), shuffle=True, random_state=INNER_VALIDATION_SEED
    )
    return splitter.split(outer_train, run.y[outer_train])


def _read_inner_scores(run, spec, number, outer_train, source_root):
    fingerprints = []
    recovery = np.full(len(outer_train), np.nan)
    spec_digest = hashlib.sha256(json.dumps(asdict(spec), sort_keys=True).encode()).hexdigest()
    for inner, (_, valid) in enumerate(_inner_splits(run, outer_train), start=1):
        source = source_root / "nested" / f"outer_{number}_inner_{inner}_{spec.name}.npz"
        with np.load(source) as saved:
            if (not np.array_equal(saved["ids"], run.ids[outer_train[valid]])
                    or saved["seed"].item() != INNER_VALIDATION_SEED
                    or saved["spec_hash"].item() != spec_digest
                    or int(saved["rounds"]) < 1
                    or "RTX 4060" not in saved["device"].item()):
                raise ValueError(f"Unexpected cross-fit checkpoint provenance: {source}")
            if np.isfinite(recovery[valid]).any():
                raise ValueError(f"Repeated inner holdout rows: {source}")
            recovery[valid] = _probabilities(saved["prediction"], len(valid), source.name)
        fingerprints.append(sha256(source))
    _probabilities(recovery, len(outer_train), f"outer {number} inner holdout coverage")
    return recovery, fingerprints


def verify_baseline(run, spec, source_root):
    if run.smoke:
        settings = {key: val for key, val in spec.params.items() if key != "max_iterations"}
        settings.update(
            max_bin=256, nthread=4, objective="binary:logistic", eval_metric="auc",
            device="cuda:0", seed=spec.model_seed,
        )
        return (
            [{"settings": settings, "rounds": SMOKE_ROUNDS, "scores": None, "inner_hashes": []}
             for _ in run.partitions],
            {"baseline": BASELINE, "inner_seed": INNER_VALIDATION_SEED,
             "baseline_oof_auc": None, "fixed_rounds_by_fold": [SMOKE_ROUNDS] * len(run.partitions)},
        )
    baseline = source_root / BASELINE
    config = read_json(baseline / "config.json")
    metrics = read_json(baseline / "metrics.json")
    manifest_path = source_root / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest != run.manifest or metrics["status"] != "completed":
        raise ValueError("The baseline and this run do not share the same full input manifest.")
    if (metrics["name"] != BASELINE or metrics["config"] != config
            or config != json.loads(json.dumps(asdict(spec)))
            or any(metrics.get(key) != value for key, value in manifest.items())):
        raise ValueError("XGB-D03 metadata does not match its experiment or source run.")
    result_path = baseline / "predictions.npz"
    folds = []
    with np.load(result_path) as saved:
        if (not np.array_equal(saved["ids"], run.ids)
                or not np.array_equal(saved["test_ids"], run.test_ids)
                or not np.array_equal(saved["fold"], run.folds)):
            raise ValueError("XGB-D03 aggregate checkpoint row IDs or fold assignments changed.")
        oof = _probabilities(saved["oof"], len(run.ids), BASELINE + " OOF")
        test = _probabilities(saved["test"], len(run.test_ids), BASELINE + " test")
        fold_test = saved["test_by_fold"]
        if fold_test.shape != (len(test), len(run.partitions)) or not np.array_equal(fold_test.mean(axis=1), test):
            raise ValueError("XGB-D03 test fold mean does not match the published predictions.")
    nested = read_json(source_root / "nested" / "config.json")
    if (nested["smoke"] or nested["inner_seed"] != INNER_VALIDATION_SEED
            or nested["fold_hash"] != run.fold_hash
            or nested["outer_folds"] != len(run.partitions)
            or nested["inner_folds"] != len(run.partitions)
            or BASELINE not in nested["members"]
            or nested["base_prediction_hashes"][nested["members"].index(BASELINE)] != _digest_array(oof)):
        raise ValueError("Nested validation metadata does not match the XGB-D03 baseline.")
    if not np.isclose(roc_auc_score(run.y, oof), metrics["oof_auc"], rtol=0, atol=1e-14):
        raise ValueError("Stored XGB-D03 OOF AUC does not agree with its predictions.")
    for number, part in enumerate(run.partitions, start=1):
        fold_path = baseline / f"fold_{number}.npz"
        details = read_json(baseline / f"fold_{number}.json")
        rounds = details["rounds"]
        settings = details["settings"]
        early = details["early"]
        if (details != metrics["folds"][number - 1]
                or not 1 <= rounds <= spec.params["max_iterations"]
                or rounds != early["rounds"] or settings != early["settings"]
                or settings["device"] != "cuda:0" or settings["seed"] != spec.model_seed
                or settings["tree_method"] != "hist" or settings["objective"] != "binary:logistic"
                or "scale_pos_weight" in settings
                or "RTX 4060" not in details["device"] or "RTX 4060" not in early["device"]
                or any(settings.get(k) != value for k, value in spec.params.items() if k != "max_iterations")):
            raise ValueError(f"XGB-D03 fixed rounds or parameters changed in fold {number}.")
        with np.load(fold_path) as saved:
            if (not np.array_equal(saved["ids"], run.ids[part["V"]])
                    or not np.array_equal(saved["valid"], oof[part["V"]])
                    or not np.array_equal(saved["test"], fold_test[:, number - 1])):
                raise ValueError(f"XGB-D03 individual and aggregate checkpoint disagree in fold {number}.")
        scores, hashes = _read_inner_scores(run, spec, number, part["T"], source_root)
        folds.append({"settings": settings, "rounds": rounds, "scores": scores, "inner_hashes": hashes})
    provenance = {
        "baseline": BASELINE,
        "baseline_path": str(baseline),
        "config_sha256": sha256(baseline / "config.json"),
        "metrics_sha256": sha256(baseline / "metrics.json"),
        "predictions_sha256": sha256(result_path),
        "source_manifest_sha256": sha256(manifest_path),
        "inner_seed": INNER_VALIDATION_SEED,
        "inner_checkpoint_sha256_by_fold": [record["inner_hashes"] for record in folds],
        "baseline_oof_auc": metrics["oof_auc"],
        "fixed_rounds_by_fold": [record["rounds"] for record in folds],
    }
    return folds, provenance


def _fit_cuda(settings, rounds, x_fit, y_fit, queries, *, weights=None):
    import xgboost as xgb

    gpu = _cuda_name()
    if not xgb.build_info().get("USE_CUDA") or settings["device"] != "cuda:0":
        raise RuntimeError("XGBoost requires the verified CUDA build and cuda:0.")

    class RequireCuda(xgb.callback.TrainingCallback):
        def before_training(self, model):
            self.check(model)
            return model

        def after_training(self, model):
            self.check(model)
            return model

        @staticmethod
        def check(model):
            device = json.loads(model.save_config())["learner"]["generic_param"]["device"]
            if device != "cuda:0":
                raise RuntimeError(f"XGBoost actually selected {device!r}.")

    started = perf_counter()
    train = xgb.QuantileDMatrix(
        x_fit, label=y_fit, weight=weights, max_bin=settings["max_bin"], nthread=settings["nthread"]
    )
    model = xgb.train(settings, train, num_boost_round=rounds, callbacks=[RequireCuda()])
    if model.num_boosted_rounds() != rounds:
        raise RuntimeError("XGBoost fitted a different number of rounds from the fixed baseline budget.")
    fit_seconds = perf_counter() - started
    predictions = tuple(
        _probabilities(model.predict(xgb.DMatrix(matrix, nthread=settings["nthread"])), len(matrix), "CUDA fit")
        for matrix in queries
    )
    return predictions, {
        "device": f"cuda:0: {gpu}",
        "rounds": rounds,
        "fit_seconds": fit_seconds,
        "settings": settings,
        "version": xgb.__version__,
    }


def _smoke_scores(run, spec, outer_train, controls):
    oof = np.full(len(outer_train), np.nan)
    for fit, valid in _inner_splits(run, outer_train):
        fit_rows, valid_rows = outer_train[fit], outer_train[valid]
        x_fit, x_valid = run.matrices(spec, fit_rows, (valid_rows,))
        (oof[valid],), _ = _fit_cuda(controls["settings"], controls["rounds"], x_fit, run.y[fit_rows], (x_valid,))
    return _probabilities(oof, len(outer_train), "smoke cross-fit")


def _negative_indices(y, ids, scores, number):
    negative = np.flatnonzero(y == 0)
    count = int(len(negative) * NEGATIVE_FRACTION)
    random = np.random.default_rng(np.random.SeedSequence([RANDOM_SEED, number]))
    selected = {"XGB-RANDOM": random.choice(negative, size=count, replace=False)}
    if scores is not None:
        order = np.lexsort((ids[negative], -scores[negative]))
        selected["XGB-HARD"] = negative[order[:count]]
    return selected


def _variant_config(spec, variant, provenance):
    config = asdict(spec)
    config.update(
        name=variant,
        parent=BASELINE,
        scale_pos_weight=VARIANTS[variant],
        source=provenance,
    )
    if variant in {"XGB-HARD", "XGB-RANDOM"}:
        config.update(negative_fraction=NEGATIVE_FRACTION, negative_weight=NEGATIVE_WEIGHT)
    if variant == "XGB-RANDOM":
        config["random_negative_seed"] = RANDOM_SEED
    return json.loads(json.dumps(config))


def run_variants(run, spec, names, controls, provenance):
    configs = {name: _variant_config(spec, name, provenance) for name in names}
    oof = {name: np.full(len(run.ids), np.nan) for name in names}
    test_by_fold = {name: np.empty((len(run.test_ids), len(run.partitions))) for name in names}
    report = {name: [] for name in names}
    for name in names:
        directory = run.output / name
        directory.mkdir(exist_ok=True)
        config_path = directory / "config.json"
        if config_path.is_file() and read_json(config_path) != configs[name]:
            raise ValueError(f"Existing {name} results use a different configuration.")
        write_json(config_path, configs[name])

    for number, (part, control) in enumerate(zip(run.partitions, controls, strict=True), start=1):
        train, valid = part["T"], part["V"]
        y_fit = run.y[train]
        paths = {
            name: (run.output / name / f"fold_{number}.npz", run.output / name / f"fold_{number}.json")
            for name in names
        }
        pending = [name for name, pair in paths.items() if not all(path.is_file() for path in pair)]
        if pending:
            scores = control["scores"]
            if run.smoke and "XGB-HARD" in pending:
                scores = _smoke_scores(run, spec, train, control)
            selected = _negative_indices(y_fit, run.ids[train], scores, number)
            x_fit, x_valid, x_test = run.matrices(spec, train, (valid, run.test_indices))
            for name in pending:
                started = perf_counter()
                print(f"{name} fold {number}: {control['rounds']} fixed rounds", flush=True)
                settings = dict(control["settings"])
                if VARIANTS[name] != 1:
                    settings["scale_pos_weight"] = VARIANTS[name]
                rows = selected.get(name)
                sample_weight = None
                if rows is not None:
                    sample_weight = np.ones(len(train), dtype=np.float32)
                    sample_weight[rows] = NEGATIVE_WEIGHT
                (p_valid, p_test), details = _fit_cuda(
                    settings, control["rounds"], x_fit, y_fit, (x_valid, x_test),
                    weights=sample_weight,
                )
                details.update(
                    fold=number,
                    auc=float(roc_auc_score(run.y[valid], p_valid)),
                    seconds=perf_counter() - started,
                    train_rows=len(train),
                    valid_rows=len(valid),
                    negatives_upweighted=0 if rows is None else len(rows),
                    upweighted_id_sha256=None if rows is None else _digest_array(np.sort(run.ids[train[rows]])),
                    source_baseline=BASELINE,
                )
                data_path, metrics_path = paths[name]
                write_npz(data_path, ids=run.ids[valid], valid=p_valid, test=p_test)
                write_json(metrics_path, details)
                print(f"{name} fold {number}: AUC {details['auc']:.6f} | {details['device']}", flush=True)
            del x_fit, x_valid, x_test

        for name in names:
            data_path, metrics_path = paths[name]
            details = read_json(metrics_path)
            if (details["rounds"] != control["rounds"] or details["fold"] != number
                    or "RTX 4060" not in details["device"]):
                raise ValueError(f"Invalid {name} fold {number} checkpoint metadata.")
            with np.load(data_path) as saved:
                if not np.array_equal(saved["ids"], run.ids[valid]):
                    raise ValueError(f"{name} fold {number} checkpoint IDs changed.")
                oof[name][valid] = _probabilities(saved["valid"], len(valid), name + " validation")
                test_by_fold[name][:, number - 1] = _probabilities(saved["test"], len(run.test_ids), name + " test")
            report[name].append(details)

    for name in names:
        test = test_by_fold[name]
        run.save_predictions(name, oof[name], test.mean(axis=1), configs[name], report[name], test_by_fold=test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS))
    parser.add_argument("--data-dir", type=Path, help="Directory holding the three official CSVs.")
    parser.add_argument("--plan-results", type=Path, default=SOURCE, help="Full baseline and nested results.")
    parser.add_argument("--output", type=Path, help="Output directory for the imbalance results.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Use the fixed small competition subset.")
    mode.add_argument("--validate-only", action="store_true", help="Verify the full cached controls on CPU.")
    args = parser.parse_args()
    names = tuple(dict.fromkeys(args.only))
    if args.data_dir is not None:
        os.environ["S6E11_DATA_DIR"] = str(locate_data(args.data_dir))
    run = PlanRun(args.output or (OUTPUT / "smoke" if args.smoke else OUTPUT), smoke=args.smoke)
    t03 = next(candidate for _, candidate in get_specs() if candidate.name == "T03")
    spec = replace(t03, name=BASELINE, groups=(*t03.groups, "cents", "digits"), parent=t03.name)
    controls, provenance = verify_baseline(run, spec, args.plan_results.resolve())
    print(
        f"{BASELINE}: baseline verified; rounds {provenance['fixed_rounds_by_fold']}; "
        f"{sum(len(record['inner_hashes']) for record in controls)} inner checkpoints verified",
        flush=True,
    )
    if not args.validate_only:
        run_variants(run, spec, names, controls, provenance)


if __name__ == "__main__":
    main()
