"""Check schema/ID boundaries and deterministic folds on synthetic fixtures."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import read_json
from src.data import validate_frames
from src.validation import assign_folds


CONFIG = read_json(Path(__file__).resolve().parents[1] / "configs/default.json")


def fixture():
    frame = {"id": np.arange(20)}
    frame.update({c: np.arange(20, dtype=float) for c in CONFIG["numeric_columns"]})
    frame.update({c: ["A"] * 20 for c in CONFIG["categorical_columns"]})
    frame.update({c: [False] * 20 for c in CONFIG["flag_columns"]})
    train = pd.DataFrame(frame)
    train[CONFIG["target"]] = [0, 1] * 10
    test = train.drop(columns=CONFIG["target"]).iloc[:3].copy()
    test["id"] = np.arange(100, 103)
    sample = pd.DataFrame({"id": test["id"], CONFIG["target"]: 0.0})
    return train, test, sample


class FoundationTests(unittest.TestCase):
    def test_valid_fixture(self):
        self.assertEqual(validate_frames(*fixture(), CONFIG)["positive_rows"], 10)

    def test_duplicate_id_rejected(self):
        train, test, sample = fixture()
        train.loc[1, "id"] = train.loc[0, "id"]
        with self.assertRaisesRegex(ValueError, "unique integers"):
            validate_frames(train, test, sample, CONFIG)

    def test_submission_order_rejected(self):
        train, test, sample = fixture()
        with self.assertRaisesRegex(ValueError, "IDs/order"):
            validate_frames(train, test, sample.iloc[::-1], CONFIG)

    def test_overlapping_ids_rejected(self):
        train, test, sample = fixture()
        test.loc[test.index[0], "id"] = 0
        sample.loc[sample.index[0], "id"] = 0
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_frames(train, test, sample, CONFIG)

    def test_folds_cover_rows_and_repeat(self):
        train, _, _ = fixture()
        a = assign_folds(train, CONFIG["target"], 5, 42)
        pd.testing.assert_frame_equal(a, assign_folds(train, CONFIG["target"], 5, 42))
        self.assertTrue(a["id"].is_unique)
        self.assertEqual(set(a["fold"]), set(range(5)))
        self.assertTrue((a.groupby("fold")[CONFIG["target"]].nunique() == 2).all())

    def test_too_few_positives_rejected(self):
        train, _, _ = fixture()
        train[CONFIG["target"]] = [1] + [0] * 19
        with self.assertRaisesRegex(ValueError, "at least n_folds"):
            assign_folds(train, CONFIG["target"], 5, 42)


if __name__ == "__main__":
    unittest.main()
