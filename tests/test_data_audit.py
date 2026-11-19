"""Step 0-1 checks: data quality table, label cast and submission naming."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import locked_requirements, submission_path
from src.data import FILENAMES, audit_data, cast_target, quality_table
from src.artifacts import read_json

CONFIG = read_json(Path(__file__).resolve().parents[1] / "configs/default.json")
TARGET = CONFIG["target"]


def fixture(rows: int = 20):
    frame = {"id": np.arange(rows, dtype=np.int64)}
    for col in CONFIG["numeric_columns"]:
        frame[col] = np.arange(rows, dtype=np.float64) / 10.0
    for col in CONFIG["categorical_columns"]:
        frame[col] = ["A"] * rows
    for col in CONFIG["flag_columns"]:
        frame[col] = [False] * rows
    train = pd.DataFrame(frame)
    train[TARGET] = [0, 1] * (rows // 2)
    test = train.drop(columns=TARGET).iloc[:3].copy()
    test["id"] = np.arange(100, 103)
    for col in CONFIG["numeric_columns"]:
        test[col] = 5.0 + np.arange(len(test), dtype=np.float64) / 10.0
    sample = pd.DataFrame({"id": test["id"], TARGET: 0.0})
    manifest = {"train_rows": len(train), "test_rows": len(test),
                "input_files": {name: {"bytes": 1, "sha256": "x", "checksum_verified": True}
                                for name in FILENAMES}}
    return train, test, sample, manifest


class DataAuditTests(unittest.TestCase):
    def test_clean_fixture_passes_all_quality_rows(self):
        train, test, sample, manifest = fixture()
        table = quality_table(audit_data(train, test, sample, CONFIG, manifest))
        self.assertTrue(table["ok"].all(), table.loc[~table["ok"]].to_dict("records"))

    def test_label_cast_to_int8_and_binary_guard(self):
        train, test, sample, manifest = fixture()
        train[TARGET] = train[TARGET].astype(float)
        cast = cast_target(train, TARGET)
        self.assertEqual(str(cast[TARGET].dtype), "int8")
        self.assertEqual(set(cast[TARGET].unique()), {0, 1})
        train.loc[0, TARGET] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            cast_target(train, TARGET)

    def test_duplicate_and_conflicting_label_groups(self):
        train, test, sample, manifest = fixture()
        features = [c for c in train.columns if c not in ("id", TARGET)]
        train.loc[1, features] = train.loc[0, features]
        audit = audit_data(train, test, sample, CONFIG, manifest)
        self.assertEqual(audit["duplicates"]["train_duplicate_rows"], 2)
        self.assertEqual(audit["duplicates"]["train_conflicting_label_groups"], 1)
        test.loc[0, features] = train.loc[2, features].values
        audit = audit_data(train, test, sample, CONFIG, manifest)
        self.assertEqual(audit["duplicates"]["test_rows_identical_to_train"], 1)

    def test_decimal_precision_flags_non_cent_values(self):
        train, test, sample, manifest = fixture()
        train["amount_usd"] = 0.12345
        audit = audit_data(train, test, sample, CONFIG, manifest)
        self.assertFalse(audit["decimal_precision"]["amount_usd"]["cent_key_safe"])
        self.assertTrue(audit["decimal_precision"]["hours_since_last_txn"]["cent_key_safe"])

    def test_out_of_range_row_fails_quality_table(self):
        train, test, sample, manifest = fixture()
        train.loc[0, "amount_usd"] = 1e9
        table = quality_table(audit_data(train, test, sample, CONFIG, manifest))
        bad = table.loc[~table["ok"], "check"].tolist()
        self.assertIn("out_of_expected_range_rows", bad)

    def test_smoke_never_writes_official_submission_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self.assertEqual(submission_path(run_dir, True).name, "submission_smoke.csv")
            (run_dir / "submission.csv").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "smoke run"):
                submission_path(run_dir, True)
            self.assertEqual(submission_path(run_dir, False).name, "submission.csv")

    def test_locked_requirements_contains_pinned_versions(self):
        env = {"python": "x", "platform": "y",
               "packages": {"numpy": "2.5.4", "catboost": None}}
        text = locked_requirements(env)
        self.assertIn("numpy==2.5.4", text)
        self.assertNotIn("catboost==", text)


if __name__ == "__main__":
    unittest.main()
