"""Fold assignments and deterministic smoke sampling; no model fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def assign_folds(train: pd.DataFrame, target: str, n_folds: int, seed: int) -> pd.DataFrame:
    y = train[target].to_numpy(dtype=np.int8)
    if n_folds < 2 or min(np.bincount(y, minlength=2)) < n_folds:
        raise ValueError("Each target class needs at least n_folds rows")
    fold = np.full(len(train), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, valid) in enumerate(splitter.split(train, y)):
        fold[valid] = k
    if (fold < 0).any():
        raise RuntimeError("Unassigned training rows")
    return pd.DataFrame({"id": train["id"].to_numpy(), target: y, "fold": fold})


def early_stop_assignments(train: pd.DataFrame, folds: pd.DataFrame, config: dict,
                           n_folds: int) -> pd.DataFrame:
    """Store only E rows; A = outer train minus E, V = the outer fold itself."""
    target = config["target"]
    fraction = config["training_plan"]["early_stopping"]["fraction_of_outer_train"]
    if not 0 < fraction < 1:
        raise ValueError("early_stopping fraction must be between 0 and 1")
    labels = train[target].to_numpy()
    fold_values = folds["fold"].to_numpy()
    records = []
    for k in range(n_folds):
        train_positions = np.flatnonzero(fold_values != k)
        _, early_positions = train_test_split(
            train_positions, test_size=fraction, stratify=labels[train_positions],
            random_state=config["early_stop_seed"])
        records.extend((int(train["id"].iloc[p]), k) for p in early_positions)
    result = pd.DataFrame(records, columns=["id", "fold"])
    result["split"] = "E"
    return result[["id", "fold", "split"]]


def partition_for_fold(train: pd.DataFrame, folds: pd.DataFrame, early_stop: pd.DataFrame,
                       config: dict, fold: int) -> dict:
    """Positional A/E/V/T indices for one outer fold; E comes from the saved table."""
    fold_values = folds["fold"].to_numpy()
    ids = train["id"].to_numpy()
    train_positions = np.flatnonzero(fold_values != fold)
    valid_positions = np.flatnonzero(fold_values == fold)
    early_ids = early_stop.loc[early_stop["fold"] == fold, "id"].to_numpy()
    early_positions = train_positions[np.isin(ids[train_positions], early_ids)]
    fit_positions = np.setdiff1d(train_positions, early_positions)
    if len(early_positions) != len(early_ids) or len(fit_positions) + len(early_positions) != len(train_positions):
        raise ValueError(f"fold {fold}: early-stop table does not partition the outer train")
    return {"T": train_positions, "A": fit_positions, "E": early_positions, "V": valid_positions}


def validate_partitions(train: pd.DataFrame, folds: pd.DataFrame, early_stop: pd.DataFrame,
                        config: dict, n_folds: int) -> dict:
    target = config["target"]
    labels = train[target].to_numpy()
    total = len(train)
    summary = {}
    for k in range(n_folds):
        part = partition_for_fold(train, folds, early_stop, config, k)
        sets = {name: set(values.tolist()) for name, values in part.items()}
        if not sets["A"].isdisjoint(sets["E"]) or sets["A"] | sets["E"] != sets["T"]:
            raise ValueError(f"fold {k}: A/E must partition T")
        if not sets["T"].isdisjoint(sets["V"]) or sets["T"] | sets["V"] != set(range(total)):
            raise ValueError(f"fold {k}: T/V must partition all training rows")
        for name in ("A", "E", "V"):
            if set(np.unique(labels[part[name]])) != {0, 1}:
                raise ValueError(f"fold {k}: {name} is missing a target class")
        summary[k] = {name: len(values) for name, values in part.items()}
        summary[k]["positive_rates"] = {name: float(labels[values].mean())
                                        for name, values in part.items()}
    return summary


def smoke_sample(train, test, sample, config):
    target = config["target"]
    n_train = min(config["smoke"]["train_rows"], len(train))
    if n_train < len(train):
        selected, _ = train_test_split(np.arange(len(train)), train_size=n_train,
                                       stratify=train[target], random_state=config["seed"])
        train = train.iloc[np.sort(selected)].reset_index(drop=True)
    count = min(config["smoke"]["test_rows"], len(test))
    return train, test.iloc[:count].copy(), sample.iloc[:count].copy()
