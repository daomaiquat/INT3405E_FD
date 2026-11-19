"""Step 7 m-estimate target encoding with strict inner cross-fitting."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from sklearn.model_selection import StratifiedKFold

MISSING_KEY = "__MISSING__"


def canonical_hash(payload) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def te_feature_names(config: dict, columns=None) -> list[str]:
    return [f"te_{column}" for column in (columns or config["numeric_columns"])]


def integer_keys(values, scale: int = 100) -> np.ndarray:
    series = pd.Series(values)
    keys = np.empty(len(series), dtype=object)
    if is_bool_dtype(series.dtype):
        for position, value in enumerate(series.to_numpy(dtype=object)):
            keys[position] = MISSING_KEY if pd.isna(value) else int(bool(value))
        return keys
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    missing = np.isnan(numeric)
    scaled = np.zeros(len(numeric), dtype=np.int64)
    scaled[~missing] = np.rint(numeric[~missing] * int(scale)).astype(np.int64)
    keys[missing] = MISSING_KEY
    keys[~missing] = scaled[~missing]
    return keys


def key_matrix(frame: pd.DataFrame, columns, scale: int = 100) -> dict[str, np.ndarray]:
    return {column: integer_keys(frame[column], scale) for column in columns}


def fit_table(keys, labels, m: float, prior: float | None = None) -> dict:
    y = np.asarray(labels, dtype=np.float64)
    key_series = pd.Series(np.asarray(keys, dtype=object))
    if len(y) == 0 or len(y) != len(key_series):
        raise ValueError("target encoding needs non-empty aligned keys and labels")
    reference_prior = float(y.mean()) if prior is None else float(prior)
    stats = pd.DataFrame({"key": key_series, "y": y}).groupby("key", sort=False)["y"].agg(["sum", "count"])
    values = (stats["sum"] + float(m) * reference_prior) / (stats["count"] + float(m))
    return {"prior": reference_prior, "m": float(m), "table": values.to_dict(), "stats": stats}


def transform_keys(keys, table: dict) -> np.ndarray:
    mapping = table["table"]
    prior = float(table["prior"])
    values = pd.Series(np.asarray(keys, dtype=object)).map(mapping)
    return values.fillna(prior).to_numpy(dtype=np.float64)


def crossfit_keys(keys, labels, m: float, n_splits: int, seed: int) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    key_array = np.asarray(keys, dtype=object)
    if len(y) != len(key_array):
        raise ValueError("keys and labels must have the same length")
    if min(np.bincount(y, minlength=2)) < n_splits:
        raise ValueError("each inner fold needs both target classes")
    values = np.full(len(y), np.nan, dtype=np.float64)
    assignments = np.full(len(y), -1, dtype=np.int16)
    summaries = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (reference, holdout) in enumerate(splitter.split(np.zeros(len(y)), y)):
        table = fit_table(key_array[reference], y[reference], m)
        values[holdout] = transform_keys(key_array[holdout], table)
        assignments[holdout] = fold
        summaries.append({"fold": fold, "prior": table["prior"],
                          "reference_rows": len(reference), "holdout_rows": len(holdout)})
    if (assignments < 0).any() or np.isnan(values).any():
        raise RuntimeError("inner cross-fitting left unencoded rows")
    return {"values": values, "assignments": assignments, "summaries": summaries}


def crossfit_matrix(keys: dict[str, np.ndarray], positions: np.ndarray, labels, m: float,
                    n_splits: int, seed: int) -> dict:
    values, assignments, summaries = {}, {}, {}
    reference_assignments = None
    for column, key_values in keys.items():
        result = crossfit_keys(key_values[positions], labels, m, n_splits, seed)
        values[column] = result["values"]
        assignments[column] = result["assignments"]
        summaries[column] = result["summaries"]
        if reference_assignments is None:
            reference_assignments = result["assignments"]
        elif not np.array_equal(reference_assignments, result["assignments"]):
            raise RuntimeError("inner splits differ between encoded columns")
    return {"values": values, "assignments": reference_assignments, "summaries": summaries}


def full_matrix(keys: dict[str, np.ndarray], positions: np.ndarray, labels, m: float) -> dict:
    return {column: fit_table(key_values[positions], labels, m)
            for column, key_values in keys.items()}


def transform_matrix(keys: dict[str, np.ndarray], positions: np.ndarray, tables: dict) -> dict[str, np.ndarray]:
    return {column: transform_keys(key_values[positions], tables[column])
            for column, key_values in keys.items()}


def encoded_frame(ids, values: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame({"id": np.asarray(ids)})
    for column, encoded in values.items():
        frame[f"te_{column}"] = np.asarray(encoded, dtype=np.float64)
    return frame


def full_table_rows(tables: dict, scope: str) -> pd.DataFrame:
    rows = []
    for column, table in tables.items():
        stats = table["stats"]
        for key, counts in stats.iterrows():
            rows.append({"column": column, "scope": scope, "key": str(key),
                         "count": int(counts["count"]), "sum": float(counts["sum"]),
                         "te": float(table["table"][key]), "prior": float(table["prior"]),
                         "m": float(table["m"])})
    return pd.DataFrame(rows)


def inner_summary_rows(summaries: dict, scope: str = "all") -> pd.DataFrame:
    rows = []
    for group, payload in summaries.items():
        if isinstance(payload, dict):
            for column, entries in payload.items():
                for entry in entries:
                    rows.append({"column": column, "scope": group,
                                 "inner_fold": int(entry["fold"]),
                                 "prior": float(entry["prior"]),
                                 "reference_rows": int(entry["reference_rows"]),
                                 "holdout_rows": int(entry["holdout_rows"])})
        else:
            for entry in payload:
                rows.append({"column": group, "scope": scope,
                             "inner_fold": int(entry["fold"]),
                             "prior": float(entry["prior"]),
                             "reference_rows": int(entry["reference_rows"]),
                             "holdout_rows": int(entry["holdout_rows"])})
    return pd.DataFrame(rows)


def artifact_signature(row_ids, reference_ids, config: dict, scope: str,
                       data_hash: str, code_hash: str, columns=None) -> dict:
    te_cfg = config["training_plan"]["target_encoding"]
    columns = list(columns or config["numeric_columns"])
    payload = {
        "data": data_hash,
        "row_ids": canonical_hash([str(value) for value in np.asarray(row_ids)]),
        "reference_ids": canonical_hash([str(value) for value in np.asarray(reference_ids)]),
        "scope": scope,
        "columns": columns,
        "m": float(te_cfg.get("smooth", te_cfg.get("m", 20.0))),
        "scale": int(te_cfg.get("key_scale", 100)),
        "inner_folds": int(te_cfg.get("inner_folds", 5)),
        "seed": int(config["encoder_seed"]),
        "splitter": "StratifiedKFold(shuffle=True, random_state=encoder_seed)",
        "code": code_hash,
        "target": config["target"],
    }
    return {"signature": canonical_hash(payload), **payload}


def encode_partition(train: pd.DataFrame, test: pd.DataFrame, part: dict, config: dict,
                     data_hash: str, code_hash: str, columns=None) -> dict:
    te_cfg = config["training_plan"]["target_encoding"]
    columns = list(columns or config["numeric_columns"])
    m = float(te_cfg.get("smooth", te_cfg.get("m", 20.0)))
    n_splits = int(te_cfg.get("inner_folds", 5))
    scale = int(te_cfg.get("key_scale", 100))
    seed = int(config["encoder_seed"])
    target = config["target"]
    labels = train[target].to_numpy(dtype=np.int8)
    train_keys = key_matrix(train, columns, scale)
    test_keys = key_matrix(test, columns, scale)
    t_positions = np.asarray(part["T"], dtype=int)
    a_positions = np.asarray(part["A"], dtype=int)
    v_positions = np.asarray(part["V"], dtype=int)
    e_positions = np.asarray(part["E"], dtype=int)

    t_cross = crossfit_matrix(train_keys, t_positions, labels[t_positions], m, n_splits, seed)
    t_full = full_matrix(train_keys, t_positions, labels[t_positions], m)
    v_from_t = transform_matrix(train_keys, v_positions, t_full)
    test_from_t = transform_matrix(test_keys, np.arange(len(test)), t_full)

    a_cross = crossfit_matrix(train_keys, a_positions, labels[a_positions], m, n_splits, seed)
    a_full = full_matrix(train_keys, a_positions, labels[a_positions], m)
    e_from_a = transform_matrix(train_keys, e_positions, a_full)

    frames = {
        "T_crossfit": encoded_frame(train["id"].iloc[t_positions], t_cross["values"]),
        "V_from_T": encoded_frame(train["id"].iloc[v_positions], v_from_t),
        "test_from_T": encoded_frame(test["id"], test_from_t),
        "A_crossfit": encoded_frame(train["id"].iloc[a_positions], a_cross["values"]),
        "E_from_A": encoded_frame(train["id"].iloc[e_positions], e_from_a),
    }
    assignments = {"T": t_cross["assignments"], "A": a_cross["assignments"]}
    tables = {"T_full": t_full, "A_full": a_full}
    summaries = {"T": t_cross["summaries"], "A": a_cross["summaries"]}
    references = {"T": t_positions, "A": a_positions}
    signatures = {}
    for name, frame in frames.items():
        scope = "A" if name in ("A_crossfit", "E_from_A") else "T"
        signatures[name] = artifact_signature(frame["id"], train["id"].iloc[references[scope]],
                                              config, scope, data_hash, code_hash, columns)
    return {"frames": frames, "assignments": assignments, "tables": tables,
            "summaries": summaries, "signatures": signatures,
            "parameters": {"m": m, "inner_folds": n_splits, "scale": scale,
                           "seed": seed, "columns": columns, "target": target}}
