"""Phase 8: reuse step-7 cross-fitted features; fit GLM and shallow trees."""

from __future__ import annotations

import copy
import hashlib
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .artifacts import load_bundle, read_json, save_bundle, write_json
from .features import feature_groups
from .metrics import auc_score, summarize_oof
from .models import fit_bundle, predict_bundle, select_iterations
from .predictions import join_oof, join_test, prediction_frame
from .target_encoding import MISSING_KEY, canonical_hash
from .validation import partition_for_fold


TE_MODELS = {"T01": ("B02", "logistic", "glm"),
             "T02": ("B05", "lightgbm", "tree_onehot"),
             "T03": ("B06", "xgboost", "tree_onehot")}
ROLES = {"T_crossfit": "T", "A_crossfit": "A", "E_from_A": "E", "V_from_T": "V"}


def canonical_csv_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def phase8_config(config, name, smoke=False):
    cfg = copy.deepcopy(config)
    cfg["phase8_use_te"] = True
    if name == "T02":
        cfg["baselines"]["lightgbm"].update(num_leaves=4, metric="auc")
    if name == "T03":
        cfg["baselines"]["xgboost"]["max_depth"] = 2
    cfg["baselines"]["early_stopping"]["first_metric_only"] = True
    if smoke and name != "T01":
        cfg["baselines"][TE_MODELS[name][1]]["max_iterations"] = 30
        cfg["baselines"]["early_stopping"]["patience"] = 5
    return cfg


def align_te(raw, encoded, columns):
    expected = ["id"] + [f"te_{c}" for c in columns]
    if list(encoded.columns) != expected or encoded.id.isna().any() or not encoded.id.is_unique:
        raise ValueError("TE cache: unexpected columns or duplicate/missing IDs")
    if len(encoded) != len(raw) or set(encoded.id) != set(raw.id):
        raise ValueError("TE cache: IDs differ from the required fold role")
    aligned = encoded.set_index("id").loc[raw.id]
    values = aligned.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("TE cache: non-finite or out-of-range probabilities")
    result = raw.copy()
    result[expected[1:]] = values
    return result


def _compatible_cache(directory, config, folds, early, manifest, smoke):
    settings = read_json(directory / "config.json")
    summary = read_json(directory / "summary.json")
    if not summary.get("gate_passed") or bool(summary["smoke"]) != smoke:
        raise ValueError("TE cache is not a valid matching full/smoke run")
    te = config["training_plan"]["target_encoding"]
    if (settings["columns"] != config["numeric_columns"]
            or settings["m"] != te["smooth"] or settings["inner_folds"] != te["inner_folds"]
            or settings["key_scale"] != te["key_scale"] or settings["encoder_seed"] != config["encoder_seed"]):
        raise ValueError("TE cache parameters differ from phase 8")
    source = directory.parent
    for name, actual in (("fold_assignments.csv", folds), ("early_stop_assignments.csv", early)):
        pd.testing.assert_frame_equal(pd.read_csv(source / name), actual, check_dtype=False)
    saved_manifest = read_json(source / "data_manifest.json")
    for name, record in manifest["input_files"].items():
        if saved_manifest["input_files"][name]["sha256"] != record["sha256"]:
            raise ValueError("TE cache uses different raw data")
    return settings


def find_encoding_cache(root, requested, config, folds, early, manifest, smoke):
    if requested:
        path = requested.resolve()
        directory = path if path.name == "encoding" else path / "encoding"
        _compatible_cache(directory, config, folds, early, manifest, smoke)
        return directory
    candidates = sorted((root / "Step7").glob("*/encoding"), reverse=True)
    candidates += sorted((root / "outputs").glob("*/encoding"), reverse=True)
    if Path("/kaggle/input").is_dir():
        candidates += [p.parent for p in sorted(Path("/kaggle/input").rglob("encoding/summary.json"))]
    reasons = []
    for directory in dict.fromkeys(candidates):
        try:
            _compatible_cache(directory, config, folds, early, manifest, smoke)
            return directory
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            reasons.append(f"{directory}: {str(exc)[:140]}")
    raise ValueError("No matching step-7 cache; attach Step7 input or set --encoding-dir. " + "; ".join(reasons))


def _tables(path):
    frame = pd.read_csv(path, dtype={"key": str})
    return {column: {"table": {(MISSING_KEY if row.key == MISSING_KEY else int(row.key)): float(row.te)
                               for row in group.itertuples()},
                     "prior": float(group.iloc[0].prior), "m": float(group.iloc[0].m)}
            for column, group in frame.groupby("column", sort=False)}


def load_fold_cache(directory, train, test, part, config):
    columns = config["numeric_columns"]
    frames = {name: align_te(train.iloc[part[role]], pd.read_csv(directory / f"{name}.csv"), columns)
              for name, role in ROLES.items()}
    frames["test_from_T"] = align_te(test, pd.read_csv(directory / "test_from_T.csv"), columns)
    return frames, _tables(directory / "tables_T_full.csv")


def run_te_models(root, run_dir, train, test, sample, folds, early, config, n_folds,
                  state, manifest, encoding_dir=None, models=None, check_cache_only=False):
    selected = list(models or TE_MODELS)
    smoke = bool(state["smoke"])
    cache = find_encoding_cache(root, encoding_dir, config, folds, early, manifest, smoke)
    directory = run_dir / "te_models"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "cache_source.json", {"directory": str(cache), "smoke": smoke,
               "note": "Only role-specific cross-fit features are used; leak_demo is never read"})
    controls = read_json(root / "configs/phase8_controls.json")
    write_json(directory / "controls_used.json", {name: controls[name] for name in selected})
    fold_hash, early_hash = canonical_csv_hash(folds), canonical_csv_hash(early)
    for name in selected:
        cfg = phase8_config(config, name, smoke)
        if not smoke:
            ref = controls[name]
            if ref["fold_hash"] != fold_hash or ref["early_stop_hash"] != early_hash:
                raise ValueError("Phase 8 control uses different folds; provide a matching control")
            if any(ref["input_files"][file]["sha256"] != record["sha256"]
                   for file, record in manifest["input_files"].items()):
                raise ValueError("Phase 8 control uses different raw data")
            if cfg["baselines"][TE_MODELS[name][1]] != ref["params"]:
                raise ValueError(f"{name}: raw model parameters differ from its control")
    outputs = {name: ({}, {}) for name in selected}
    metadata = {name: {} for name in selected}
    for fold in range(n_folds):
        part = partition_for_fold(train, folds, early, config, fold)
        logging.info("Phase 8 loading TE cache fold %s from %s", fold, cache)
        frames, tables = load_fold_cache(cache / f"fold_{fold}", train, test, part, config)
        if check_cache_only:
            continue
        for name in selected:
            model_id, family, mode = TE_MODELS[name]
            cfg = phase8_config(config, name, smoke)
            exp = directory / name
            exp.mkdir(parents=True, exist_ok=True)
            cfg_key = canonical_hash(cfg)
            signature = {"config": cfg_key, "cache": str(cache), "fold": fold}
            meta_path = exp / f"fold_{fold}.json"
            required = [exp / f"oof_fold_{fold}.csv", exp / f"test_fold_{fold}.csv",
                        exp / "model_bundle" / f"fold_{fold}.joblib"]
            if state.get("resumed") and meta_path.is_file() and all(p.is_file() for p in required):
                previous = read_json(meta_path)
                if previous.get("signature") == signature and previous.get("status") == "completed":
                    outputs[name][0][fold], outputs[name][1][fold] = map(pd.read_csv, required[:2])
                    metadata[name][str(fold)] = previous
                    logging.info("Phase 8 %s reusing fold %s", name, fold)
                    continue
            write_json(meta_path, {"status": "started", "signature": signature})
            try:
                started = perf_counter()
                if name == "T01":
                    selection = {"n_trees": 0, "warnings": [], "eval_auc": None}
                else:
                    selection = select_iterations(model_id, frames["A_crossfit"], frames["E_from_A"], cfg, fold)
                selected_at = perf_counter()
                bundle = fit_bundle(model_id, frames["T_crossfit"], cfg, selection["n_trees"], fold)
                bundle.update(te_tables=tables, te_scale=config["training_plan"]["target_encoding"]["key_scale"])
                fitted_at = perf_counter()
                p_valid = predict_bundle(bundle, frames["V_from_T"])
                p_test = predict_bundle(bundle, frames["test_from_T"])
                # Training diagnostic must retain cross-fitted features.
                train_auc = auc_score(frames["T_crossfit"][config["target"]], predict_bundle(bundle, frames["T_crossfit"]))
                save_bundle(bundle, required[2])
                if name == "T01":
                    pd.DataFrame({"feature": bundle["preprocessor"].get_feature_names_out(),
                                  "coefficient": bundle["estimator"].coef_[0]}).to_csv(
                        exp / f"coefficients_fold_{fold}.csv", index=False)
                loaded = load_bundle(required[2])
                # Raw inference must reproduce the cached V transform/model path.
                difference = float(np.max(np.abs(p_valid - predict_bundle(loaded, train.iloc[part["V"]]))))
                if difference > 1e-10:
                    raise ValueError("Saved raw-inference model does not reproduce cached predictions")
                oof = prediction_frame(train.id.iloc[part["V"]], p_valid, fold)
                test_frame = prediction_frame(test.id, p_test, fold)
                oof.to_csv(required[0], index=False)
                test_frame.to_csv(required[1], index=False)
                detail = {"status": "completed", "signature": signature,
                          "n_trees": int(bundle["n_trees"]), "eval_auc": selection["eval_auc"],
                          "train_auc": train_auc, "raw_inference_max_diff": difference,
                          "selection_seconds": selected_at-started, "refit_seconds": fitted_at-selected_at,
                          "total_seconds": perf_counter()-started,
                          "warnings": sorted(set(selection["warnings"] + bundle["warnings"]))}
                write_json(meta_path, detail)
                outputs[name][0][fold], outputs[name][1][fold] = oof, test_frame
                metadata[name][str(fold)] = detail
                write_json(exp / "params.json", {"model_id": model_id, "config": cfg,
                           "feature_groups": feature_groups(model_id, cfg, mode)})
                logging.info("Phase 8 %s fold %s completed; trees=%s", name, fold, bundle["n_trees"])
            except Exception as exc:
                write_json(meta_path, {"status": "failed", "signature": signature, "error": str(exc)})
                raise
    if check_cache_only:
        result = {"status": "cache_verified", "cache": str(cache), "folds": n_folds, "smoke": smoke}
        write_json(directory / "cache_check.json", result)
        return result
    rows = []
    for name in selected:
        exp = directory / name
        oof = join_oof(outputs[name][0], folds, config)
        test_frame = join_test(outputs[name][1], sample, n_folds)
        metrics = summarize_oof(oof, train_auc=float(np.mean([v["train_auc"] for v in metadata[name].values()])),
                                smoke=smoke, model=name)
        write_json(exp / "metrics.json", metrics)
        write_json(exp / "fold_metrics.json", metadata[name])
        oof.to_csv(exp / "oof.csv", index=False)
        test_frame.to_csv(exp / "test_by_fold.csv", index=False)
        ref = controls[name]["metrics"] if not smoke else None
        row = {"model": name, "base_model": TE_MODELS[name][0], "oof_auc": metrics["oof_auc"],
               "mean_auc": metrics["mean_auc"], "logloss": metrics["logloss"],
               "train_auc": metrics["train_auc"], "control_auc": ref["oof_auc"] if ref else None,
               "delta_auc": metrics["oof_auc"]-ref["oof_auc"] if ref else None,
               "seconds": sum(v["total_seconds"] for v in metadata[name].values()), "smoke": smoke}
        if ref:
            for k in range(n_folds):
                row[f"delta_fold_{k}"] = metrics["fold_auc"][str(k)]-ref["fold_auc"][str(k)]
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(directory / "comparison.csv", index=False)
    comparison.assign(run_id=state["run_id"], stage="phase8", status="completed").to_csv(
        run_dir / "experiment_log.csv", index=False)
    write_json(directory / "summary.json", {"status": "completed", "smoke": smoke,
               "cache": str(cache), "models": selected, "target_encoding": "numeric13, m=20, inner=5",
               "note": "development CV only; no submission or stack is selected"})
    logging.info("Phase 8 completed\n%s", comparison.to_string(index=False))
    return {"models": selected, "cache": str(cache), "smoke": smoke}
