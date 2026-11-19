"""Step-5 baseline estimators with fold-local preprocessing and refit rules.

Tree boosters select the number of trees on A/E and refit on T with that count.
Baseline models never see test labels and are not reported as final results.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import feature_matrix, numeric_columns
from .metrics import auc_score


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    mode: str
    family: str
    early_stop: bool = False
    fixed_trees: int = 0


BASELINES = {
    "B00": BaselineSpec("prior_constant", "none", "constant", fixed_trees=0),
    "B01": BaselineSpec("independent_random", "none", "random", fixed_trees=0),
    "B02": BaselineSpec("logistic_l2", "glm", "linear", fixed_trees=0),
    "B03": BaselineSpec("decision_tree_depth4", "numeric", "tree", fixed_trees=1),
    "B04": BaselineSpec("random_forest", "numeric", "tree", fixed_trees=300),
    "B05": BaselineSpec("lightgbm", "tree_onehot", "boosting", early_stop=True),
    "B06": BaselineSpec("xgboost", "tree_onehot", "boosting", early_stop=True),
    "B07": BaselineSpec("catboost", "catboost", "boosting", early_stop=True),
}
BASELINE_IDS = tuple(BASELINES)


def _params(config: dict, key: str) -> dict:
    return dict(config["baselines"][key])


def _preprocessor(model_id: str, config: dict):
    mode = BASELINES[model_id].mode
    if mode == "glm":
        return ColumnTransformer([
            ("numeric", StandardScaler(), numeric_columns(config, mode)),
            ("flags", "passthrough", list(config["flag_columns"])),
            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             list(config["categorical_columns"])),
        ], sparse_threshold=0.0)
    if mode == "tree_onehot":
        return ColumnTransformer([
            ("features", "passthrough",
             numeric_columns(config, mode) + list(config["flag_columns"])),
            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             list(config["categorical_columns"])),
        ], sparse_threshold=0.0)
    return None


def _estimator(model_id: str, config: dict, n_trees: int, early_stop: bool):
    seed = int(config["model_seed"])
    patience = int(config["baselines"]["early_stopping"]["patience"])
    if model_id == "B02":
        return LogisticRegression(**_params(config, "logistic"), random_state=seed)
    if model_id == "B03":
        return DecisionTreeClassifier(**_params(config, "decision_tree"), random_state=seed)
    if model_id == "B04":
        return RandomForestClassifier(**_params(config, "random_forest"), random_state=seed,
                                      n_jobs=1)
    if model_id == "B05":
        import lightgbm as lgb
        params = _params(config, "lightgbm")
        params.pop("max_iterations", None)
        params.update(n_estimators=n_trees, random_state=seed, n_jobs=1, verbosity=-1)
        return lgb.LGBMClassifier(**params)
    if model_id == "B06":
        from xgboost import XGBClassifier
        params = _params(config, "xgboost")
        params.pop("max_iterations", None)
        params.update(n_estimators=n_trees, random_state=seed, n_jobs=1, verbosity=0,
                      eval_metric="auc",
                      early_stopping_rounds=patience if early_stop else None)
        return XGBClassifier(**params)
    if model_id == "B07":
        from catboost import CatBoostClassifier
        params = _params(config, "catboost")
        params.pop("max_iterations", None)
        params.update(iterations=n_trees, random_seed=seed, thread_count=1, verbose=False,
                      loss_function="Logloss", eval_metric="AUC")
        return CatBoostClassifier(**params)
    raise ValueError(f"unknown baseline: {model_id}")


def _predict_matrix(bundle: dict, matrix: pd.DataFrame) -> np.ndarray:
    preprocessor = bundle["preprocessor"]
    values = preprocessor.transform(matrix) if preprocessor is not None else matrix
    return bundle["estimator"].predict_proba(values)[:, 1]


def _fit_estimator(bundle: dict, matrix: pd.DataFrame, labels: np.ndarray,
                   eval_pair=None, early_stop: bool = False) -> None:
    if bundle["preprocessor"] is not None:
        features = bundle["preprocessor"].fit_transform(matrix)
        eval_features = (bundle["preprocessor"].transform(eval_pair[0])
                         if eval_pair is not None else None)
    else:
        features = matrix
        eval_features = eval_pair[0] if eval_pair is not None else None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model_id = bundle["model_id"]
        estimator = bundle["estimator"]
        if model_id == "B05":
            import lightgbm as lgb
            kwargs = {}
            if eval_pair is not None:
                kwargs.update(eval_set=[(eval_features, eval_pair[1])], eval_metric="auc",
                              callbacks=[lgb.early_stopping(
                                  int(bundle["config"]["baselines"]["early_stopping"]["patience"]),
                                  first_metric_only=bool(bundle["config"]["baselines"]["early_stopping"].get("first_metric_only", False)),
                                  verbose=False)])
            estimator.fit(features, labels, **kwargs)
            best = getattr(estimator, "best_iteration_", 0) or 0
            bundle["best_iteration"] = int(best) if best > 0 else None
        elif model_id == "B06":
            kwargs = {"verbose": False}
            if eval_pair is not None:
                kwargs["eval_set"] = [(eval_features, eval_pair[1])]
            estimator.fit(features, labels, **kwargs)
            best = getattr(estimator, "best_iteration", None)
            bundle["best_iteration"] = int(best) + 1 if best is not None else None
        elif model_id == "B07":
            kwargs = {}
            if eval_pair is not None:
                kwargs.update(eval_set=[(eval_features, eval_pair[1])],
                              early_stopping_rounds=int(
                                  bundle["config"]["baselines"]["early_stopping"]["patience"]),
                              use_best_model=True)
            estimator.fit(features, labels, cat_features=list(
                bundle["config"]["categorical_columns"]), **kwargs)
            best = getattr(estimator, "best_iteration_", None)
            bundle["best_iteration"] = int(best) + 1 if best is not None else None
        else:
            estimator.fit(features, labels)
            bundle["best_iteration"] = None
    bundle["warnings"] = [str(item.message) for item in caught]


def fit_bundle(model_id: str, fit_frame: pd.DataFrame, config: dict,
               n_trees: int | None = None, fold: int = 0) -> dict:
    """Fit the final model for one fold. No early stopping happens here."""
    if model_id not in BASELINES:
        raise ValueError(f"unknown baseline: {model_id}")
    spec = BASELINES[model_id]
    labels = fit_frame[config["target"]].to_numpy(dtype=np.int8)
    bundle = {"model_id": model_id, "name": spec.name, "mode": spec.mode,
              "family": spec.family, "config": config, "fold": int(fold),
              "preprocessor": None, "estimator": None, "warnings": [],
              "best_iteration": None, "eval_auc": None}

    if model_id == "B00":
        bundle.update(prior=float(labels.mean()), n_trees=0)
        return bundle
    if model_id == "B01":
        bundle.update(seed=int(config["model_seed"]) + int(fold), n_trees=0)
        return bundle

    if n_trees is None:
        if spec.early_stop:
            n_trees = int(config["baselines"][{
                "B05": "lightgbm", "B06": "xgboost", "B07": "catboost"}[model_id]]["max_iterations"])
        else:
            n_trees = int(spec.fixed_trees)
    matrix = feature_matrix(fit_frame, config, spec.mode)
    bundle["preprocessor"] = _preprocessor(model_id, config)
    bundle["estimator"] = _estimator(model_id, config, int(n_trees), early_stop=False)
    _fit_estimator(bundle, matrix, labels)
    bundle["n_trees"] = int(n_trees) if spec.family == "boosting" else int(spec.fixed_trees)
    if model_id == "B02":
        estimator = bundle["estimator"]
        bundle["n_iter"] = int(np.asarray(getattr(estimator, "n_iter_", [0])).ravel()[0])
    return bundle


def select_iterations(model_id: str, fit_frame: pd.DataFrame, eval_frame: pd.DataFrame,
                      config: dict, fold: int = 0) -> dict:
    """Pick n_trees on A/E only; the returned count is reused for the T refit."""
    spec = BASELINES.get(model_id)
    if spec is None or not spec.early_stop:
        raise ValueError(f"{model_id} does not use early stopping")
    labels = fit_frame[config["target"]].to_numpy(dtype=np.int8)
    eval_labels = eval_frame[config["target"]].to_numpy(dtype=np.int8)
    matrix = feature_matrix(fit_frame, config, spec.mode)
    eval_matrix = feature_matrix(eval_frame, config, spec.mode)
    key = {"B05": "lightgbm", "B06": "xgboost", "B07": "catboost"}[model_id]
    max_iterations = int(config["baselines"][key]["max_iterations"])

    bundle = {"model_id": model_id, "name": spec.name, "mode": spec.mode,
              "family": spec.family, "config": config, "fold": int(fold),
              "preprocessor": _preprocessor(model_id, config),
              "estimator": _estimator(model_id, config, max_iterations, early_stop=True),
              "warnings": [], "best_iteration": None, "eval_auc": None, "n_trees": max_iterations}
    _fit_estimator(bundle, matrix, labels, eval_pair=(eval_matrix, eval_labels), early_stop=True)
    eval_prediction = _predict_matrix(bundle, eval_matrix)
    return {"n_trees": int(bundle["best_iteration"] or max_iterations),
            "best_iteration": bundle["best_iteration"],
            "eval_auc": auc_score(eval_labels, eval_prediction),
            "warnings": list(bundle["warnings"])}


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Return probabilities in the same row order as `frame`."""
    if bundle["model_id"] == "B00":
        return np.full(len(frame), float(bundle["prior"]), dtype=np.float64)
    if bundle["model_id"] == "B01":
        return np.random.default_rng(int(bundle["seed"])).random(len(frame))
    if bundle.get("te_tables"):
        from .target_encoding import integer_keys, transform_keys
        frame = frame.copy()
        for column, table in bundle["te_tables"].items():
            name = f"te_{column}"
            # Already attached cross-fitted train values must not be replaced by
            # full-train encodings. Raw inference is transformed by the saved table.
            if name not in frame:
                frame[name] = transform_keys(integer_keys(frame[column], bundle["te_scale"]), table)
    matrix = feature_matrix(frame, bundle["config"], bundle["mode"])
    return np.asarray(_predict_matrix(bundle, matrix), dtype=np.float64)
