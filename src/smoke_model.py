"""Infrastructure-only contract model used by the step 3 smoke check.

This is NOT a baseline or a reported result; baselines begin at step 5 of
plan.md. It exists to prove fit / OOF / test / bundle roundtrip boundaries.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import load_bundle, save_bundle  # noqa: F401


def fit_contract_model(train: pd.DataFrame, config: dict, features=None) -> dict:
    columns = list(features or config["numeric_columns"])
    x = train[columns].to_numpy(dtype=np.float64)
    y = train[config["target"]].to_numpy(dtype=np.int8)
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, tol=1e-4, random_state=config["model_seed"]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(x, y)
    return {"kind": "contract_logreg_smoke", "features": columns, "pipeline": pipeline,
            "n_trees": 0, "warnings": [str(item.message) for item in caught]}


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    x = frame[bundle["features"]].to_numpy(dtype=np.float64)
    return bundle["pipeline"].predict_proba(x)[:, 1]


