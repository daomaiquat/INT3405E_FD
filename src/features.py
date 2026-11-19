"""Feature matrices for raw baselines and optional phase-8 TE inputs.

Raw columns are retained. Phase 8 additionally consumes role-specific TE columns;
GLM uses their clipped logits and trees use the encoded probabilities. Learned
scalers and one-hot categories are fitted by the owning training fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOG_COLUMNS = ("amount_usd", "account_balance_usd")
MODES = ("none", "numeric", "glm", "tree_onehot", "catboost")


def numeric_columns(config: dict, mode: str) -> list[str]:
    columns = list(config["numeric_columns"])
    if mode == "glm":
        columns.extend(f"log_{name}" for name in LOG_COLUMNS)
    if config.get("phase8_use_te"):
        columns.extend(f"te_{name}" for name in config["numeric_columns"])
    return columns


def feature_matrix(frame: pd.DataFrame, config: dict, mode: str) -> pd.DataFrame:
    """Return model input columns in a stable order."""
    if mode not in MODES:
        raise ValueError(f"unknown feature mode: {mode}")
    if mode == "none":
        return pd.DataFrame(index=frame.index)

    columns = list(config["numeric_columns"]) + list(config["flag_columns"])
    data = frame[columns].copy()
    if mode == "glm":
        for name in LOG_COLUMNS:
            data[f"log_{name}"] = np.log1p(data[name].clip(lower=0.0))
    if mode in ("glm", "tree_onehot", "catboost"):
        for name in config["categorical_columns"]:
            data[name] = frame[name].astype(str).to_numpy()
    if config.get("phase8_use_te"):
        for column in config["numeric_columns"]:
            name = f"te_{column}"
            values = frame[name].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"{name}: invalid target-encoded probability")
            if mode == "glm":
                values = np.clip(values, 1e-6, 1 - 1e-6)
                values = np.log(values / (1 - values))
            data[name] = values
    return data


def feature_groups(model_id: str, config: dict, mode: str) -> dict:
    return {
        "model_id": model_id,
        "mode": mode,
        "numeric": numeric_columns(config, mode),
        "flags": list(config["flag_columns"]),
        "categorical": list(config["categorical_columns"]) if mode in
        ("glm", "tree_onehot", "catboost") else [],
        "logs": list(LOG_COLUMNS) if mode == "glm" else [],
        "target_encoding": {"enabled": bool(config.get("phase8_use_te")),
                            "transform": "logit" if mode == "glm" else "probability"},
    }
