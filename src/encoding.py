"""Step 7 stage: cross-fitted target-encoding artifacts and leakage guards."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .artifacts import environment, sha256_file, write_json
from .metrics import auc_score
from .target_encoding import (canonical_hash, encode_partition, encoded_frame,
                              fit_table, full_matrix, full_table_rows, integer_keys,
                              inner_summary_rows, key_matrix, te_feature_names,
                              transform_matrix)
from .validation import partition_for_fold


def _te_matrix(frame: pd.DataFrame, config: dict, columns) -> np.ndarray:
    return frame[te_feature_names(config, columns)].to_numpy(dtype=np.float64)


def _fit_predict(features: np.ndarray, labels, evaluation: np.ndarray, config: dict) -> np.ndarray:
    scaler = StandardScaler()
    train_values = scaler.fit_transform(np.asarray(features, dtype=np.float64))
    evaluation_values = scaler.transform(np.asarray(evaluation, dtype=np.float64))
    params = dict(config["baselines"]["logistic"])
    model = LogisticRegression(**params, random_state=int(config["model_seed"]))
    model.fit(train_values, np.asarray(labels, dtype=np.int8))
    return model.predict_proba(evaluation_values)[:, 1]


def _write_frames(directory, result: dict, config: dict, columns) -> None:
    signatures = {}
    for name, frame in result["frames"].items():
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False)
        signature = dict(result["signatures"][name])
        signature["file_sha256"] = sha256_file(path)
        signatures[name] = signature
    write_json(directory / "cache_index.json", signatures)


def _write_tables(directory, result: dict, train: pd.DataFrame, part: dict,
                  config: dict, columns) -> None:
    full_table_rows(result["tables"]["T_full"], "T").to_csv(
        directory / "tables_T_full.csv", index=False)
    full_table_rows(result["tables"]["A_full"], "A").to_csv(
        directory / "tables_A_full.csv", index=False)
    inner_summary_rows(result["summaries"], "all").to_csv(
        directory / "tables_inner_summary.csv", index=False)
    for scope, positions in (("T", part["T"]), ("A", part["A"])):
        pd.DataFrame({
            "id": train["id"].iloc[np.asarray(positions, dtype=int)].to_numpy(),
            "inner_fold": np.asarray(result["assignments"][scope], dtype=np.int16),
        }).to_csv(directory / f"inner_folds_{scope}.csv", index=False)


def _check_inner_folds(train: pd.DataFrame, folds: pd.DataFrame, early: pd.DataFrame,
                       config: dict, n_folds: int, columns, results: dict) -> dict:
    target = config["target"]
    te_cfg = config["training_plan"]["target_encoding"]
    scale = int(te_cfg.get("key_scale", 100))
    m = float(te_cfg.get("smooth", te_cfg.get("m", 20.0)))
    n_splits = int(te_cfg.get("inner_folds", 5))
    seed = int(config["encoder_seed"])
    labels = train[target].to_numpy(dtype=np.int8)
    failures = []
    ok = True
    for fold in range(n_folds):
        part = partition_for_fold(train, folds, early, config, fold)
        for scope in ("T", "A"):
            positions = np.asarray(part[scope], dtype=int)
            scope_labels = labels[positions]
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            expected_splits = list(splitter.split(np.zeros(len(positions)), scope_labels))
            expected_assignments = np.full(len(positions), -1, dtype=np.int16)
            for inner, (reference, holdout) in enumerate(expected_splits):
                expected_assignments[holdout] = inner
            if not np.array_equal(results[fold]["assignments"][scope], expected_assignments):
                ok = False
                failures.append({"fold": fold, "scope": scope, "ok": False,
                                 "reason": "inner fold assignments differ from saved splitter"})
            for column in columns:
                keys = integer_keys(train[column], scale)[positions]
                for inner, (reference, holdout) in enumerate(expected_splits):
                    expected = fit_table(keys[reference], scope_labels[reference], m)
                    if not np.isclose(expected["prior"], float(scope_labels[reference].mean()),
                                      atol=1e-12):
                        ok = False
                        failures.append({"fold": fold, "scope": scope, "column": column,
                                         "inner_fold": inner, "ok": False,
                                         "reason": "reference prior mismatch"})
                    if len(holdout) == 0 or len(reference) == 0:
                        ok = False
                        failures.append({"fold": fold, "scope": scope, "column": column,
                                         "inner_fold": inner, "ok": False,
                                         "reason": "empty inner partition"})
    return {"ok": bool(ok), "checked_folds": n_folds, "checked_scopes": 2,
            "checked_columns": list(columns), "inner_folds": n_splits,
            "details": failures}


def _check_outer_v_labels(train: pd.DataFrame, test: pd.DataFrame, part: dict,
                          result: dict, config: dict, data_hash: str,
                          code_hash: str, columns) -> dict:
    changed = train.copy()
    labels = changed[config["target"]].to_numpy(dtype=np.int8)
    valid_positions = np.asarray(part["V"], dtype=int)
    changed.iloc[valid_positions, changed.columns.get_loc(config["target"])] = (
        1 - labels[valid_positions])
    changed_result = encode_partition(changed, test, part, config, data_hash, code_hash, columns)
    differences = {}
    for name, frame in result["frames"].items():
        before = _te_matrix(frame, config, columns)
        after = _te_matrix(changed_result["frames"][name], config, columns)
        differences[name] = float(np.max(np.abs(before - after)))
    train_labels = train[config["target"]].to_numpy(dtype=np.int8)
    fit_features = _te_matrix(result["frames"]["T_crossfit"], config, columns)
    valid_features = _te_matrix(result["frames"]["V_from_T"], config, columns)
    original_prediction = _fit_predict(fit_features, train_labels[np.asarray(part["T"], dtype=int)],
                                       valid_features, config)
    changed_prediction = _fit_predict(
        _te_matrix(changed_result["frames"]["T_crossfit"], config, columns),
        train_labels[np.asarray(part["T"], dtype=int)],
        _te_matrix(changed_result["frames"]["V_from_T"], config, columns), config)
    model_difference = float(np.max(np.abs(original_prediction - changed_prediction)))
    ok = max(differences.values()) < 1e-12 and model_difference < 1e-12
    return {"ok": bool(ok), "fold": 0, "max_feature_diff": differences,
            "max_model_prediction_diff": model_difference,
            "note": "outer V labels are never used for T/A encoders or model fitting"}


def _check_shuffled_labels(train: pd.DataFrame, test: pd.DataFrame, part: dict,
                           config: dict, data_hash: str, code_hash: str, columns) -> dict:
    target = config["target"]
    original = train[target].to_numpy(dtype=np.int8)
    shuffled = np.random.default_rng(int(config["encoder_seed"])).permutation(original)
    changed = train.copy()
    changed[target] = shuffled
    result = encode_partition(changed, test, part, config, data_hash, code_hash, columns)
    features = _te_matrix(result["frames"]["T_crossfit"], config, columns)
    evaluation = _te_matrix(result["frames"]["V_from_T"], config, columns)
    prediction = _fit_predict(features, shuffled[np.asarray(part["T"], dtype=int)],
                              evaluation, config)
    score = float(auc_score(shuffled[np.asarray(part["V"], dtype=int)], prediction))
    return {"ok": bool(0.35 <= score <= 0.65), "fold": 0, "shuffled_label_auc": score,
            "acceptance_range": [0.35, 0.65],
            "note": "wide sanity range; no forced 0.48-0.50 window"}


def _leak_demo(directory, train: pd.DataFrame, part: dict, result: dict,
               config: dict, columns) -> dict:
    target = config["target"]
    te_cfg = config["training_plan"]["target_encoding"]
    scale = int(te_cfg.get("key_scale", 100))
    m = float(te_cfg.get("smooth", te_cfg.get("m", 20.0)))
    labels = train[target].to_numpy(dtype=np.int8)
    train_keys = key_matrix(train, columns, scale)
    leak_tables = full_matrix(train_keys, np.arange(len(train)), labels, m)
    leak_values = transform_matrix(train_keys, np.asarray(part["V"], dtype=int), leak_tables)
    leak_frame = encoded_frame(train["id"].iloc[np.asarray(part["V"], dtype=int)], leak_values)
    fit_features = _te_matrix(result["frames"]["T_crossfit"], config, columns)
    fit_labels = labels[np.asarray(part["T"], dtype=int)]
    clean_prediction = _fit_predict(
        fit_features, fit_labels, _te_matrix(result["frames"]["V_from_T"], config, columns), config)
    leak_prediction = _fit_predict(fit_features, fit_labels, _te_matrix(leak_frame, config, columns), config)
    valid_labels = labels[np.asarray(part["V"], dtype=int)]
    summary = {
        "valid_registry": False,
        "reason": "intentional leakage: V labels were used to build the encoding table",
        "fold": 0,
        "clean_v_auc": float(auc_score(valid_labels, clean_prediction)),
        "leaky_v_auc": float(auc_score(valid_labels, leak_prediction)),
        "note": "diagnostic only; never registered as a valid model or feature source",
    }
    write_json(directory / "leak_demo" / "summary.json", summary)
    return summary


def run_encoding(run_dir, train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame,
                 folds: pd.DataFrame, early: pd.DataFrame, config: dict, n_folds: int,
                 state: dict, manifest: dict) -> dict:
    """Write step-7 cross-fitted TE artifacts and run the mandatory guards."""
    directory = run_dir / "encoding"
    directory.mkdir(parents=True, exist_ok=True)
    columns = list(config["numeric_columns"])
    te_cfg = config["training_plan"]["target_encoding"]
    data_hash = canonical_hash(manifest)
    code_hash = canonical_hash(state["source_signature"])
    settings = {
        "columns": columns, "m": float(te_cfg.get("smooth", te_cfg.get("m", 20.0))),
        "inner_folds": int(te_cfg.get("inner_folds", 5)),
        "key_scale": int(te_cfg.get("key_scale", 100)),
        "encoder_seed": int(config["encoder_seed"]),
        "scope_T": "cross-fit within T; full-T table transforms V and test",
        "scope_A": "cross-fit within A; full-A table transforms E",
        "early_stopping": "A/E split is locked before any TE is fitted",
        "test_labels_used": 0,
    }
    write_json(directory / "config.json", settings)
    fold_summaries = []
    partition_results = {}
    for fold in range(n_folds):
        started = perf_counter()
        part = partition_for_fold(train, folds, early, config, fold)
        result = encode_partition(train, test, part, config, data_hash, code_hash, columns)
        partition_results[fold] = result
        fold_dir = directory / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_frames(fold_dir, result, config, columns)
        _write_tables(fold_dir, result, train, part, config, columns)
        write_json(fold_dir / "parameters.json", result["parameters"])
        fold_summaries.append({
            "fold": fold,
            "rows": {name: len(frame) for name, frame in result["frames"].items()},
            "seconds": perf_counter() - started,
        })
        logging.info("Step 7 fold %s encoded %s columns with %s inner folds",
                     fold, len(columns), settings["inner_folds"])

    first_part = partition_for_fold(train, folds, early, config, 0)
    first_result = encode_partition(train, test, first_part, config, data_hash, code_hash, columns)
    checks = {
        "inner_holdout_excluded": _check_inner_folds(train, folds, early, config, n_folds,
                                                     columns, partition_results),
        "outer_v_label_change": _check_outer_v_labels(train, test, first_part, first_result,
                                                      config, data_hash, code_hash, columns),
        "shuffled_label_model": _check_shuffled_labels(train, test, first_part, config,
                                                       data_hash, code_hash, columns),
        "leak_demo": _leak_demo(directory, train, first_part, first_result, config, columns),
    }
    all_ok = all(item["ok"] for name, item in checks.items() if name != "leak_demo")
    write_json(directory / "checks_summary.json", {"all_valid_checks_passed": bool(all_ok),
                                                   "checks": checks})
    write_json(directory / "summary.json", {
        "smoke": bool(state.get("smoke")), "folds": n_folds, "columns": columns,
        "data_hash": data_hash, "code_hash": code_hash, "gate_passed": bool(all_ok),
        "valid_registry": "cross-fitted TE only; leak_demo is excluded",
        "fold_artifacts": fold_summaries, "environment": environment()["packages"],
    })
    log_path = run_dir / "experiment_log.csv"
    pd.DataFrame([{
        "experiment_id": "step7_te_contract", "run_id": state["run_id"],
        "status": "completed" if all_ok else "failed",
        "event_time": datetime.now(timezone.utc).isoformat(),
        "resumed": bool(state.get("resumed")), "model": "m_estimate_encoder",
        "oof_auc": checks["shuffled_label_model"]["shuffled_label_auc"],
        "note": "cross-fitted target encoding guards; shuffled-label AUC is not a model result",
    }]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    if not all_ok:
        raise ValueError("step 7 target-encoding guards failed; see encoding/checks_summary.json")
    return {"directory": str(directory), "folds": n_folds, "columns": columns,
            "gate_passed": bool(all_ok),
            "shuffled_label_auc": checks["shuffled_label_model"]["shuffled_label_auc"]}
