"""Diagnose and screen S6E11 feature groups with saved nested OOF scores."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = PROJECT / "s6e11_gpu" / "plan_runs" / "results" / "full"
CONFIG = PROJECT.parent / "configs" / "default.json"
RIDGE = 50.0
MIN_ROWS = 500
MIN_EXPECTED = 5.0
DELTA_CAP = 0.5
TARGET = "is_fraud"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_base(train_ids, test_ids, folder):
    with np.load(folder / "predictions.npz", allow_pickle=False) as values:
        np.testing.assert_array_equal(values["ids"], train_ids)
        np.testing.assert_array_equal(values["test_ids"], test_ids)
        scores, folds = values["oof"], values["fold"]
    if scores.shape != train_ids.shape or not np.all(np.isfinite(scores) & (scores >= 0) & (scores <= 1)):
        raise ValueError(f"Unexpected OOF scores in {folder}")
    return scores, folds


def inner_scores(ids, y, train_index, outer, n_folds, seed, nested):
    scores = np.full(len(train_index), np.nan)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for inner, (_, valid) in enumerate(splitter.split(train_index, y[train_index]), 1):
        cache = nested / f"outer_{outer}_inner_{inner}_XGB-D03.npz"
        with np.load(cache, allow_pickle=False) as values:
            np.testing.assert_array_equal(values["ids"], ids[train_index[valid]])
            if values["seed"].item() != seed:
                raise ValueError(f"Seed mismatch in {cache.name}")
            scores[valid] = values["prediction"]
    if not np.all(np.isfinite(scores) & (scores >= 0) & (scores <= 1)):
        raise ValueError(f"Incomplete inner scores for fold {outer}")
    return scores


def key(*columns):
    value = pd.Series(columns[0], dtype="str").astype("str")
    for column in columns[1:]:
        value = value.str.cat(pd.Series(column).astype("str"), sep=" / ")
    return value


def quantile_key(values, fit, groups=10):
    edges = np.unique(np.quantile(values[fit], np.arange(1, groups) / groups))
    return np.searchsorted(edges, values, side="right") + 1


def centered(values, context, fit):
    grouped = pd.DataFrame({"value": values[fit], "context": context.iloc[fit].to_numpy()})
    stats = grouped.groupby("context", sort=False)["value"].agg(["median", "size"])
    medians = stats.loc[stats["size"] >= 250, "median"]
    by_row = context.map(medians).fillna(np.median(values[fit])).to_numpy()
    return values - by_row


class CandidateKeys:
    def __init__(self, frame, flags):
        number = lambda column: frame[column].to_numpy()
        self.flags = frame[flags].sum(axis=1).to_numpy(dtype=np.int8)
        cvv = np.minimum(number("cvv_retry_count"), 2)
        foreign = number("is_foreign_transaction").astype(int)
        ip = number("ip_country_mismatch").astype(int)
        new = number("is_new_merchant").astype(int)
        hour = number("time_of_day_hour") // 6
        channel = frame["channel"]
        self.amount = np.log1p(number("amount_usd"))
        self.ratio = np.log1p(number("amount_usd") / np.maximum(number("account_balance_usd"), 1.0))
        self.velocity = number("velocity_score")
        self.distance = np.log1p(number("distance_from_home_km"))
        self.network_count = np.clip(foreign + ip + number("used_vpn") + number("billing_shipping_mismatch"), 0, 2)
        self.amount_context = key(frame["merchant_category"], channel)
        self.distance_context = key(channel, foreign)
        self.velocity_context = key(
            np.digitize(number("txn_count_last_24h"), [3, 6, 10]),
            np.digitize(number("hours_since_last_txn"), [0.25, 1, 6, 24]),
        )
        amount = number("amount_usd")
        amount_band = np.searchsorted([25, 100, 250, 1000], amount, side="left")
        cents = np.rint(amount * 100).astype(np.int64) % 100
        self.fixed = {
            "cvv_auth": key(cvv, frame["auth_method"]),
            "cvv_channel": key(cvv, channel),
            "network_new": key(foreign, ip, new),
            "merchant_new": key(frame["merchant_category"], new),
            "channel_device": key(channel, frame["device_type"]),
            "channel_hour": key(channel, hour),
            "cvv_flag_count": key(cvv, np.minimum(self.flags, 3)),
            "cvv_amount": key(cvv, amount_band),
            "amount_cents_scale": key(amount_band, (cents % 10 == 0).astype(int)),
        }

    def with_reference(self, fit):
        yield from self.fixed.items()
        yield "amount_balance_ratio", key(quantile_key(self.ratio, fit))
        yield "amount_context", key(quantile_key(centered(self.amount, self.amount_context, fit), fit))
        yield "velocity_context", key(quantile_key(centered(self.velocity, self.velocity_context, fit), fit))
        yield "distance_context", key(quantile_key(centered(self.distance, self.distance_context, fit), fit))
        yield "velocity_network", key(quantile_key(self.velocity, fit, 5), self.network_count)


def correction(groups, size, train_index, y, inner):
    training_groups = groups[train_index]
    rows = np.bincount(training_groups, minlength=size)
    expected = np.bincount(training_groups, weights=inner, minlength=size)
    residual = np.bincount(training_groups, weights=y[train_index] - inner, minlength=size)
    curvature = np.bincount(training_groups, weights=inner * (1 - inner), minlength=size)
    offset = np.clip(residual / (curvature + RIDGE), -DELTA_CAP, DELTA_CAP)
    return np.where((rows >= MIN_ROWS) & (expected >= MIN_EXPECTED), offset, 0.0)


def rank_summary(y, predictions):
    top_count = int(np.ceil(0.01 * len(y)))
    review = np.zeros(len(y), dtype=bool)
    review[np.argsort(-predictions, kind="stable")[:top_count]] = True
    found = int(y[review].sum())
    return {
        "auc": roc_auc_score(y, predictions), "ap": average_precision_score(y, predictions),
        "log_loss": log_loss(y, predictions), "reviewed": top_count,
        "fraud_found": found, "precision": found / top_count, "recall": found / int(y.sum()),
        "expected_fraud": float(predictions.sum()),
    }, review


def group_result(mask, y, base, final):
    return {
        "rows": int(mask.sum()), "fraud": int(y[mask].sum()),
        "xgb_expected": float(base[mask].sum()), "final_expected": float(final[mask].sum()),
    }


def describe_groups(codes, labels, y, base, final, folds):
    train_codes = codes[:len(y)]
    discovery = folds < 3
    counts = np.bincount(codes[len(y):], minlength=len(labels))
    groups = []
    for code, label in enumerate(labels):
        included = train_codes == code
        first = group_result(included & discovery, y, base, final)
        if first["rows"] < 500 or first["fraud"] < 40 or counts[code] < 100:
            continue
        groups.append({
            "group": str(label), "test_rows": int(counts[code]), "discovery": first,
            "confirmation": [group_result(included & (folds == fold), y, base, final) for fold in (3, 4)],
            "all_train": group_result(included, y, base, final),
        })
    return sorted(groups, key=lambda row: abs(row["discovery"]["fraud"] - row["discovery"]["xgb_expected"]), reverse=True)


def verify_inputs():
    manifest = read_json(RESULTS / "manifest.json")
    if manifest["smoke"]:
        raise ValueError("Expected full official results")
    for name, expected in manifest["inputs"].items():
        with (PROJECT.parent / "data" / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Official input changed: {name}")


def main():
    verify_inputs()
    train = pd.read_csv(PROJECT.parent / 'data' / "train.csv")
    test = pd.read_csv(PROJECT.parent / 'data' / "test.csv")
    ids, test_ids = train["id"].to_numpy(), test["id"].to_numpy()
    y = train[TARGET].to_numpy(dtype=np.int8)
    xgb, folds = read_base(ids, test_ids, RESULTS / "XGB-D03")
    final, final_folds = read_base(ids, test_ids, RESULTS / "final")
    np.testing.assert_array_equal(folds, final_folds)
    config = read_json(CONFIG)
    nested = RESULTS / "nested"
    nested_config = read_json(nested / "config.json")
    xgb_index = nested_config["members"].index("XGB-D03")
    if hashlib.sha256(xgb.tobytes()).hexdigest() != nested_config["base_prediction_hashes"][xgb_index]:
        raise ValueError("Nested XGB prediction snapshot changed")
    seed = nested_config["inner_seed"]
    n_folds = nested_config["outer_folds"]
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config["seed"])
    frame = pd.concat((train.drop(columns=["id", TARGET]), test.drop(columns="id")), ignore_index=True)
    candidate_keys = CandidateKeys(frame, config["flag_columns"])
    adjusted, coverage = {}, {}
    xgb_logit = logit(np.clip(xgb, 1e-9, 1 - 1e-9))

    for fold, (fit, valid) in enumerate(splitter.split(ids, y)):
        np.testing.assert_array_equal(valid, np.flatnonzero(folds == fold))
        inner = inner_scores(ids, y, fit, fold + 1, n_folds, seed, nested)
        for name, values in candidate_keys.with_reference(fit):
            group, labels = pd.factorize(values, sort=True)
            offset = correction(group, len(labels), fit, y, inner)[group[valid]]
            if name not in adjusted:
                adjusted[name] = np.full(len(y), np.nan)
                coverage[name] = []
            adjusted[name][valid] = expit(xgb_logit[valid] + offset)
            coverage[name].append(float(np.count_nonzero(offset) / len(valid)))
        print(f"Fold {fold + 1}: score correction evaluated per candidate", flush=True)

    xgb_metrics, xgb_review = rank_summary(y, xgb)
    final_metrics, final_review = rank_summary(y, final)
    cohorts = {}
    cvv = frame["cvv_retry_count"].to_numpy()[:len(y)]
    for name, mask in {
        "no_flags": candidate_keys.flags[:len(y)] == 0,
        "some_flags": candidate_keys.flags[:len(y)] > 0,
        "cvv0": cvv == 0,
        "cvv1": cvv == 1,
        "cvv2plus": cvv >= 2,
    }.items():
        cohorts[name] = {
            **group_result(mask, y, xgb, final),
            "xgb_found": int(y[mask & xgb_review].sum()),
            "final_found": int(y[mask & final_review].sum()),
        }
    xgb_by_fold = [roc_auc_score(y[folds == fold], xgb[folds == fold]) for fold in range(n_folds)]
    candidates = {}
    for name, prediction in adjusted.items():
        metrics, _ = rank_summary(y, prediction)
        fold_auc = [roc_auc_score(y[folds == fold], prediction[folds == fold]) for fold in range(n_folds)]
        delta = [value - original for value, original in zip(fold_auc, xgb_by_fold, strict=True)]
        candidates[name] = {
            **metrics, "delta_auc": metrics["auc"] - xgb_metrics["auc"],
            "delta_ap": metrics["ap"] - xgb_metrics["ap"],
            "delta_log_loss": metrics["log_loss"] - xgb_metrics["log_loss"],
            "fold_delta_auc": delta, "fold_wins": sum(value > 0 for value in delta),
            "affected_share": coverage[name],
        }
    descriptive = {}
    for name, values in candidate_keys.with_reference(np.flatnonzero(folds < 3)):
        group, labels = pd.factorize(values, sort=True)
        descriptive[name] = describe_groups(group, labels, y, xgb, final, folds)
    payload = {
        "train_rows": len(y), "test_rows": len(test_ids), "positive_rows": int(y.sum()),
        "parameters": {"ridge": RIDGE, "min_rows": MIN_ROWS, "min_expected": MIN_EXPECTED,
                       "max_abs_offset": DELTA_CAP, "inner_oof_files_verified": n_folds ** 2},
        "xgb": xgb_metrics, "final": final_metrics, "cohorts": cohorts,
        "candidates": candidates, "groups": descriptive,
    }
    path = HERE / "feature_screen.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    for name, result in sorted(candidates.items(), key=lambda item: -item[1]["delta_auc"]):
        print(f"{name:24} {result['delta_auc']:+.8f} AUC | {result['fold_wins']}/{n_folds} folds")


if __name__ == "__main__":
    main()
