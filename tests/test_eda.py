"""Step 4 checks: EDA tables carry counts and discovery/confirmation stay separate."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import read_json
from src.eda import _digit_labels, run_eda, split_halves

CONFIG = read_json(Path(__file__).resolve().parents[1] / "configs/default.json")
TARGET = CONFIG["target"]


def make_fixture(rows: int = 240):
    rng = np.random.default_rng(11)
    index = np.arange(rows)
    labels = (index % 5 == 0).astype(np.int8)
    frame = {"id": index.astype(np.int64)}
    for offset, column in enumerate(CONFIG["numeric_columns"]):
        if column == "cvv_retry_count":
            frame[column] = (index % 4).astype(np.float64)
        else:
            frame[column] = 1.0 + (index % (20 + offset)) / (3.0 + offset) \
                + rng.normal(0.0, 0.01, rows)
    frame["merchant_category"] = [f"M{value % 30}" for value in index]
    frame["card_type"] = [("A", "B", "C")[value % 3] for value in index]
    frame["auth_method"] = np.where(labels == 1, "No Authentication", "Biometric")
    frame["channel"] = [("web", "app")[value % 2] for value in index]
    frame["device_type"] = [("mobile", "desktop")[value % 2] for value in index]
    flags = {
        "is_foreign_transaction": labels,
        "is_new_merchant": (index % 7 == 0).astype(np.int8),
        "used_vpn": (index % 11 == 0).astype(np.int8),
        "ip_country_mismatch": labels,
        "billing_shipping_mismatch": (index % 13 == 0).astype(np.int8),
        "is_ai_generated_scam_attempt": (index % 17 == 0).astype(np.int8),
    }
    for column, values in flags.items():
        frame[column] = values.astype(np.int8)
    train = pd.DataFrame(frame)
    train[TARGET] = labels

    test = train.drop(columns=TARGET).iloc[:120].copy()
    test["id"] = np.arange(1000, 1120, dtype=np.int64)
    for column in CONFIG["numeric_columns"]:
        test[column] = test[column] + 0.01
    folds = pd.DataFrame({"id": train["id"], "fold": index % CONFIG["n_folds"]})

    config = copy.deepcopy(CONFIG)
    config["eda"].update({"min_digit_group": 1, "min_pair_group": 1,
                          "adversarial_rows_per_side": 60, "ks_rows_per_side": 60})
    return train, test, folds, config


class EdaTests(unittest.TestCase):
    def setUp(self):
        self.train, self.test, self.folds, self.config = make_fixture()

    def test_digit_keys_match_the_documented_example(self):
        values = np.asarray([1371.37])
        expected = {"cents": "37", "int": "1371", "int10": "137", "tenths": "13713",
                    "lead1": "4|1", "lead2": "4|13"}
        for key_type, value in expected.items():
            self.assertEqual(_digit_labels(values, key_type).iloc[0], value)

    def test_discovery_and_confirmation_are_disjoint_and_stratified(self):
        discovery, confirmation = split_halves(self.train, self.config)
        self.assertTrue(set(discovery["id"]).isdisjoint(set(confirmation["id"])))
        self.assertEqual(len(discovery) + len(confirmation), len(self.train))
        self.assertAlmostEqual(discovery[TARGET].mean(), self.train[TARGET].mean(), places=6)
        self.assertAlmostEqual(confirmation[TARGET].mean(), self.train[TARGET].mean(), places=6)

    def test_run_eda_writes_tables_and_eight_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            summary = run_eda(output, self.train, self.test, self.folds, self.config)
            self.assertEqual(summary["findings"], 8)
            for name in ("findings.json", "imbalance.json", "adversarial.json", "eda_report.md",
                         "univariate_numeric_summary.csv", "flags_n_flags.csv",
                         "flag_effects.csv", "cardinality_unseen.csv", "digit_stability.csv",
                         "pair_stability.csv", "train_test_shift.csv"):
                self.assertTrue((output / name).exists(), name)

            findings = read_json(output / "findings.json")
            self.assertEqual([item["id"] for item in findings], [f"F{i}" for i in range(1, 9)])
            for item in findings:
                self.assertTrue(item["hypothesis"])
                self.assertIn(item["priority"], {"high", "medium", "low"})
                self.assertTrue("unverified" in item["status"] or
                                item["status"] == "confirmed_from_data")
            self.assertEqual(read_json(output / "imbalance.json")["positive_rows"],
                             int(self.train[TARGET].sum()))

    def test_every_rate_table_has_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_eda(output, self.train, self.test, self.folds, self.config)
            tables = ("univariate_numeric_cvv_retry_count.csv",
                      "univariate_categorical_auth_method.csv", "univariate_flag_used_vpn.csv",
                      "flags_n_flags.csv", "digit_discovery.csv", "digit_confirmation.csv",
                      "pair_discovery.csv", "pair_confirmation.csv")
            for name in tables:
                table = pd.read_csv(output / name)
                for column in ("n_rows", "n_fraud", "fraud_rate"):
                    self.assertIn(column, table.columns, name)
                self.assertTrue((table["n_fraud"] <= table["n_rows"]).all(), name)
                self.assertTrue((table["fraud_rate"] <= 1.0).all(), name)

    def test_report_keeps_shift_caveat_and_adversarial_folds(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_eda(output, self.train, self.test, self.folds, self.config)
            report = (output / "eda_report.md").read_text(encoding="utf-8")
            self.assertIn("discovery/confirmation", report)
            self.assertIn("không chứng minh", report)
            adversarial = read_json(output / "adversarial.json")
            self.assertEqual(set(adversarial["fold_auc"]), {"0", "1", "2"})
            self.assertAlmostEqual(adversarial["mean_auc"],
                                   float(np.mean(list(adversarial["fold_auc"].values()))))
            stability = pd.read_csv(output / "pair_stability.csv")
            stable = stability.loc[stability["groups_in_both"] >= 20]
            self.assertFalse(stable.empty)


if __name__ == "__main__":
    unittest.main()
