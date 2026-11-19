"""Id-keyed assembly of OOF and per-fold test predictions (plan.md step 3).

No positional merging: every frame is reindexed on `id` and checked one-to-one
against the canonical fold/sample tables before predictions are combined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assert_one_to_one(ids, name: str) -> None:
    series = pd.Series(ids)
    if series.isna().any() or not series.is_unique:
        raise ValueError(f"{name}: ids must be unique and non-null")


def prediction_frame(ids, predictions, fold: int) -> pd.DataFrame:
    return pd.DataFrame({"id": np.asarray(ids), "prediction": np.asarray(predictions, dtype=np.float64),
                         "fold": int(fold)})


def shuffle_for_check(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)


def join_oof(fold_outputs: dict, folds: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Each training row appears once, from the fold whose validation set contains it."""
    target = config["target"]
    reference = folds.set_index("id")
    expected_ids = set(reference.index)
    frames = []
    for fold, frame in sorted(fold_outputs.items()):
        if not {"id", "prediction", "fold"}.issubset(frame.columns):
            raise ValueError(f"fold {fold}: OOF frame needs id, prediction and fold")
        if set(frame["fold"]) != {fold}:
            raise ValueError(f"fold {fold}: producer fold column disagrees")
        assert_one_to_one(frame["id"], f"OOF fold {fold}")
        frames.append(frame[["id", "prediction", "fold"]])
    oof = pd.concat(frames, ignore_index=True).sort_values("id").reset_index(drop=True)
    assert_one_to_one(oof["id"], "OOF")
    if set(oof["id"]) != expected_ids:
        raise ValueError("OOF ids differ from the training id set")
    aligned = reference.loc[oof["id"], ["fold", target]].reset_index(drop=True)
    if not np.array_equal(aligned["fold"].to_numpy(), oof["fold"].to_numpy()):
        raise ValueError("OOF row was not produced by its own outer fold")
    if not np.isfinite(oof["prediction"].to_numpy(dtype=np.float64)).all():
        raise ValueError("OOF predictions contain non-finite values")
    oof[target] = aligned[target].to_numpy(dtype=np.int8)
    return oof[["id", target, "fold", "prediction"]]


def join_test(fold_outputs: dict, sample: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Align every fold to sample IDs/order, then average fold predictions."""
    expected_ids = sample["id"]
    assert_one_to_one(expected_ids, "sample")
    result = pd.DataFrame({"id": expected_ids.to_numpy()})
    fold_columns = []
    for fold in range(n_folds):
        if fold not in fold_outputs:
            raise ValueError(f"missing test predictions for fold {fold}")
        frame = fold_outputs[fold]
        if not {"id", "prediction"}.issubset(frame.columns):
            raise ValueError(f"fold {fold}: test frame needs id and prediction")
        assert_one_to_one(frame["id"], f"test fold {fold}")
        if set(frame["id"]) != set(expected_ids):
            raise ValueError(f"fold {fold}: test ids differ from sample")
        aligned = frame.set_index("id")["prediction"].reindex(expected_ids)
        if aligned.isna().any():
            raise ValueError(f"fold {fold}: test ids could not be aligned")
        name = f"fold_{fold}"
        result[name] = aligned.to_numpy(dtype=np.float64)
        fold_columns.append(name)
    result["prediction"] = result[fold_columns].mean(axis=1)
    if not np.isfinite(result["prediction"].to_numpy(dtype=np.float64)).all():
        raise ValueError("test predictions contain non-finite values")
    return result
