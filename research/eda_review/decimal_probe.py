"""Screen decimal interactions for residual structure in the stored XGB-D03 OOF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RUN = PROJECT / "s6e11_gpu" / "plan_runs" / "results" / "full"
COLUMNS = ["id", "amount_usd", "account_balance_usd", "merchant_category"]
AMOUNT_BANDS = np.array([25, 100, 250, 500, 1000])
BAND_NAMES = ("<25", "25–<100", "100–<250", "250–<500", "500–<1000", ">=1000")
DISCOVERY = (0, 1, 2)
CONFIRMATION = (3, 4)


def load_data():
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    for name in ("train.csv", "test.csv"):
        with (PROJECT.parent / "data" / name).open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != manifest["inputs"][name]:
                raise ValueError(f"Unexpected official data snapshot for {name}")
    train = pd.read_csv(PROJECT.parent / 'data' / "train.csv", usecols=[*COLUMNS, "is_fraud"])
    test = pd.read_csv(PROJECT.parent / 'data' / "test.csv", usecols=COLUMNS)
    labels = train["is_fraud"].to_numpy(dtype=np.int8)
    with np.load(RUN / "XGB-D03" / "predictions.npz", allow_pickle=False) as saved:
        for key, table in (("ids", train), ("test_ids", test)):
            if not np.array_equal(saved[key], table["id"].to_numpy()):
                raise ValueError(f"Saved {key} and official CSV order differ")
        oof, fold = saved["oof"], saved["fold"]
    actual_fold = np.zeros(len(labels), dtype=np.int8)
    for index, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(labels, labels)):
        actual_fold[valid] = index
    if not np.array_equal(fold, actual_fold):
        raise ValueError("OOF assignment differs from the locked folds")
    if np.any(~np.isfinite(oof)) or np.any((oof < 0) | (oof > 1)):
        raise ValueError("OOF predictions must be finite probabilities")
    expected_auc = json.loads((RUN / "XGB-D03" / "metrics.json").read_text(encoding="utf-8"))["oof_auc"]
    if abs(roc_auc_score(labels, oof) - expected_auc) > 1e-9:
        raise ValueError("OOF AUC could not be reproduced")
    return train, test, labels, oof, fold, expected_auc


def notation(df):
    amount = df["amount_usd"].to_numpy()
    balance = df["account_balance_usd"].to_numpy()
    amount_cents = np.rint(amount * 100).astype(np.int64)
    balance_cents = np.rint(balance * 100).astype(np.int64)
    return {
        "amount": amount_cents,
        "balance": balance_cents,
        "amount_band": np.searchsorted(AMOUNT_BANDS, amount, side="right"),
        "merchant": df["merchant_category"].to_numpy(),
        "rounding": {
            "amount_subcent_rows": int(np.count_nonzero(np.abs(amount * 100 - amount_cents) > 1e-7)),
            "balance_subcent_rows": int(np.count_nonzero(np.abs(balance * 100 - balance_cents) > 1e-7)),
        },
    }


def groups(data, merchants):
    amount, balance = data["amount"], data["balance"]
    amount_suffix, balance_suffix = amount % 100, balance % 100
    all_rows = np.ones(len(amount), dtype=bool)
    for name, cents in (("amount", amount), ("balance", balance)):
        for dollars in (1, 5, 10, 25, 50, 100):
            if dollars == 1:
                family = "raw_cents_check"
            else:
                family = "dollar_grid"
            yield family, f"{name} multiple of ${dollars}", (cents % (100 * dollars) == 0), all_rows
        yield "dollar_grid", f"{name} whole $ with last dollar digit 0 or 5", (cents % 500 == 0), cents % 100 == 0

    amount_ends = (
        ("whole dollar", amount_suffix == 0),
        (".50/.99", np.isin(amount_suffix, (50, 99))),
        ("nonzero multiple of 5 cents", (amount_suffix % 5 == 0) & (amount_suffix > 0)),
        ("nonzero multiple of 10 cents", (amount_suffix % 10 == 0) & (amount_suffix > 0)),
        (".25/.50/.75", np.isin(amount_suffix, (25, 50, 75))),
    )
    for name, mask in amount_ends:
        yield "raw_cents_check", f"amount ending {name}", mask, all_rows
        for band, display_name in enumerate(BAND_NAMES):
            in_band = data["amount_band"] == band
            yield "amount_band_x_suffix", f"amount {display_name} ending {name}", mask & in_band, in_band

    for name, mask in amount_ends[:3]:
        for merchant in merchants:
            in_merchant = data["merchant"] == merchant
            yield "merchant_x_amount_suffix", f"{merchant}: amount {name}", mask & in_merchant, in_merchant

    for same, comparison in (
        ("same amount and balance cent suffix", amount_suffix == balance_suffix),
        ("same terminal cent digit", amount_suffix % 10 == balance_suffix % 10),
        ("same amount and balance final dollar+cent suffix", amount % 1000 == balance % 1000),
    ):
        yield "amount_x_balance_suffix", same, comparison, all_rows
    for label, amount_mask, balance_mask in (
        ("whole dollar", amount_suffix == 0, balance_suffix == 0),
        ("nonzero multiple of 5 cents", (amount_suffix % 5 == 0) & (amount_suffix > 0),
         (balance_suffix % 5 == 0) & (balance_suffix > 0)),
        (".25/.50/.75", np.isin(amount_suffix, (25, 50, 75)), np.isin(balance_suffix, (25, 50, 75))),
    ):
        yield "amount_x_balance_suffix", f"both endings {label}", amount_mask & balance_mask, amount_mask


def totals(mask, labels, probability, residual, fold):
    selected_folds = fold[mask]
    return np.stack([
        np.bincount(selected_folds, weights=weights, minlength=5)
        for weights in (None, labels[mask], probability[mask], residual[mask], residual[mask] ** 2)
    ])


def describe(sums, against):
    size, fraud, expected, residual, squared = sums
    count, _, _, against_residual, against_squared = against
    observed_rate = float(fraud / size) if size else None
    expected_rate = float(expected / size) if size else None
    gap = z = None
    if size > 1 and count > 1:
        gap = float(residual / size - against_residual / count)
        contrast_variance = (squared - residual**2 / size) / (size * (size - 1))
        contrast_variance += (against_squared - against_residual**2 / count) / (count * (count - 1))
        if contrast_variance > 0:
            z = float(gap / np.sqrt(contrast_variance))
    return {
        "n": int(size), "fraud": int(fraud), "expected": float(expected),
        "fraud_rate": observed_rate, "expected_rate": expected_rate,
        "observed_expected": float(fraud / expected) if expected else None,
        "residual_pp": float(100 * residual / size) if size else None,
        "comparison_n": int(count), "residual_contrast_pp": 100 * gap if gap is not None else None,
        "contrast_z_descriptive": z,
    }


def evaluate(train, test, labels, probability, fold):
    train_notation, test_notation = notation(train), notation(test)
    merchants = sorted(train["merchant_category"].unique().tolist())
    test_counts = {name: int(mask.sum()) for _, name, mask, _ in groups(test_notation, merchants)}
    summaries = []
    residual = labels - probability
    for family, name, mask, context in groups(train_notation, merchants):
        values = totals(mask, labels, probability, residual, fold)
        other = totals(context & ~mask, labels, probability, residual, fold)
        disc = describe(values[:, DISCOVERY].sum(axis=1), other[:, DISCOVERY].sum(axis=1))
        held = [describe(values[:, number], other[:, number]) for number in CONFIRMATION]
        joint = describe(values[:, CONFIRMATION].sum(axis=1), other[:, CONFIRMATION].sum(axis=1))
        train_count, test_count = int(values[0].sum()), test_counts[name]
        eligible = disc["n"] >= 500 and disc["comparison_n"] >= 500 and disc["fraud"] >= 20 and disc["expected"] >= 12
        checkable = all(item["n"] >= 200 and item["fraud"] >= 5 and item["comparison_n"] >= 200 for item in held) and test_count >= 200
        direction = np.sign(disc["residual_contrast_pp"] or 0)
        repeated_direction = checkable and direction != 0 and all(direction * (item["residual_contrast_pp"] or 0) > 0 for item in held)
        summaries.append({
            "family": family, "group": name, "discovery": disc, "confirm_fold4": held[0],
            "confirm_fold5": held[1], "confirm_pooled": joint,
            "train_rows": train_count, "test_rows": test_count,
            "train_pct": 100 * train_count / len(train), "test_pct": 100 * test_count / len(test),
            "eligible_discovery": eligible, "adequate_confirmation": checkable,
            "repeated_direction": repeated_direction,
        })
    novel = [item for item in summaries if item["eligible_discovery"] and item["family"] != "raw_cents_check"]
    novel.sort(key=lambda row: abs(row["discovery"]["contrast_z_descriptive"] or 0), reverse=True)
    return train_notation["rounding"], test_notation["rounding"], summaries, novel[:15]


def main():
    train, test, labels, probability, fold, expected_auc = load_data()
    precision_train, precision_test, summaries, shortlist = evaluate(train, test, labels, probability, fold)
    report = {
        "rows": {"train": len(train), "test": len(test), "train_fraud": int(labels.sum())},
        "train_fraud_rate": float(labels.mean()), "verified_oof_auc": expected_auc,
        "baseline_oof_mean": float(probability.mean()), "numeric_precision": {"train": precision_train, "test": precision_test},
        "method": {
            "discovery": [1, 2, 3], "confirmation": [4, 5], "shortlist": "absolute descriptive residual contrast z in discovery only",
            "residual": "y - verified XGB-D03 OOF probability",
            "comparison": "rest of the same context if defined; otherwise all remaining rows",
            "scope": "Exploratory residual screen on previously used folds; overlapping groups and model training sets limit independence. Test counts only.",
        },
        "scanned_groups": len(summaries), "discovery_shortlist": shortlist, "all_groups": summaries,
    }
    target = HERE / "decimal_probe.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Checked {len(summaries)} groups, {len(shortlist)} shortlisted by discovery residual contrasts.")
    for row in shortlist:
        disc = row["discovery"]
        gap4 = row["confirm_fold4"]["residual_contrast_pp"]
        gap5 = row["confirm_fold5"]["residual_contrast_pp"]
        fold4 = f"{gap4:+.4f}" if gap4 is not None else "—"
        fold5 = f"{gap5:+.4f}" if gap5 is not None else "—"
        print(f"{row['group']}: n={disc['n']}, fraud={disc['fraud']}, z={disc['contrast_z_descriptive']:+.2f}, "
              f"gap folds 4/5 {fold4}/{fold5} pp, test={row['test_rows']}, repeat={row['repeated_direction']}")


if __name__ == "__main__":
    main()
