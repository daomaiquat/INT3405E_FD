"""Unified metric helpers locked in plan.md step 2.

Primary metric is pooled OOF ROC AUC; per-fold AUC, mean/std, log-loss and an
optional train AUC are always reported alongside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def auc_score(y, prediction) -> float:
    labels = np.asarray(y)
    scores = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("predictions contain non-finite values")
    if len(np.unique(labels)) < 2:
        raise ValueError("ROC AUC needs both target classes")
    return float(roc_auc_score(labels, scores))


def logloss_score(y, prediction) -> float:
    labels = np.asarray(y)
    scores = np.clip(np.asarray(prediction, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(log_loss(labels, scores, labels=[0, 1]))


def summarize_oof(oof: pd.DataFrame, target: str = "is_fraud", prediction: str = "prediction",
                  fold: str = "fold", train_predictions=None, **extras) -> dict:
    if not oof["id"].is_unique:
        raise ValueError("OOF ids must be unique")
    fold_auc = {}
    for number, group in oof.groupby(fold, sort=True):
        fold_auc[int(number)] = auc_score(group[target], group[prediction])
    if len(fold_auc) < 2:
        raise ValueError("OOF summary needs at least two folds")
    values = np.asarray(list(fold_auc.values()), dtype=np.float64)
    summary = {
        "oof_auc": auc_score(oof[target], oof[prediction]),
        "fold_auc": {str(k): v for k, v in fold_auc.items()},
        "mean_auc": float(values.mean()),
        "std_auc": float(values.std(ddof=1)),
        "logloss": logloss_score(oof[target], oof[prediction]),
        "rows": len(oof),
    }
    if train_predictions is not None:
        train_y, train_p = train_predictions
        summary["train_auc"] = auc_score(train_y, train_p)
    summary.update(extras)
    return summary
