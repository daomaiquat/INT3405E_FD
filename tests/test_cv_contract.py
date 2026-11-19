"""Step 2 checks: locked A/E/V splits and the unified metric contract."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.artifacts import read_json
from src.metrics import auc_score, logloss_score, summarize_oof
from src.validation import (assign_folds, early_stop_assignments, partition_for_fold,
                            validate_partitions)

CONFIG = read_json(Path(__file__).resolve().parents[1] / "configs/default.json")
TARGET = CONFIG["target"]
N_FOLDS = CONFIG["n_folds"]


def make_train(rows: int = 200) -> pd.DataFrame:
    frame = {"id": np.arange(rows, dtype=np.int64)}
    for col in CONFIG["numeric_columns"]:
        frame[col] = np.arange(rows, dtype=np.float64) / 10.0
    for col in CONFIG["categorical_columns"]:
        frame[col] = ["A"] * rows
    for col in CONFIG["flag_columns"]:
        frame[col] = [False] * rows
    train = pd.DataFrame(frame)
    train[TARGET] = [0, 1] * (rows // 2)
    return train


class CvContractTests(unittest.TestCase):
    def setUp(self):
        self.train = make_train()
        self.folds = assign_folds(self.train, TARGET, N_FOLDS, CONFIG["seed"])
        self.early = early_stop_assignments(self.train, self.folds, CONFIG, N_FOLDS)

    def test_three_seeds_are_separate_config_fields(self):
        for key in ("model_seed", "encoder_seed", "early_stop_seed"):
            self.assertIn(key, CONFIG)
        self.assertEqual(CONFIG["seed"], 42)

    def test_partitions_are_disjoint_complete_and_stratified(self):
        summary = validate_partitions(self.train, self.folds, self.early, CONFIG, N_FOLDS)
        for fold, sizes in summary.items():
            self.assertEqual(sizes["T"] + sizes["V"], len(self.train))
            self.assertEqual(sizes["A"] + sizes["E"], sizes["T"])
            for name in ("A", "E", "V"):
                self.assertGreater(sizes["positive_rates"][name], 0)
                self.assertLess(sizes["positive_rates"][name], 1)
        first = partition_for_fold(self.train, self.folds, self.early, CONFIG, 0)
        self.assertEqual(len(first["V"]), 40)
        self.assertEqual(len(first["E"]), 16)
        self.assertEqual(len(first["A"]), 144)
        self.assertTrue(set(first["A"]).isdisjoint(first["E"]))
        self.assertTrue(set(first["T"]).isdisjoint(first["V"]))

    def test_early_stop_split_is_reproducible_and_seed_sensitive(self):
        again = early_stop_assignments(self.train, self.folds, CONFIG, N_FOLDS)
        pd.testing.assert_frame_equal(self.early, again)
        changed = dict(CONFIG)
        changed["early_stop_seed"] = CONFIG["early_stop_seed"] + 1
        other = early_stop_assignments(self.train, self.folds, changed, N_FOLDS)
        self.assertFalse(self.early.equals(other))

    def test_early_stop_rows_never_come_from_validation_or_test(self):
        test_ids = set(self.train["id"]) | {9999}
        for fold in range(N_FOLDS):
            part = partition_for_fold(self.train, self.folds, self.early, CONFIG, fold)
            fit_ids = set(self.train["id"].iloc[part["T"]])
            self.assertTrue(fit_ids.isdisjoint(set(self.train["id"].iloc[part["V"]])))
            self.assertTrue(fit_ids.issubset(test_ids))
            self.assertNotIn(9999, fit_ids)

    def test_oof_summary_matches_sklearn_and_rejects_bad_ids(self):
        rng = np.random.default_rng(7)
        oof = pd.DataFrame({"id": np.arange(200), TARGET: [0, 1] * 100,
                            "fold": np.repeat(np.arange(5), 40),
                            "prediction": rng.random(200)})
        result = summarize_oof(oof)
        self.assertAlmostEqual(result["oof_auc"], roc_auc_score(oof[TARGET], oof["prediction"]))
        self.assertEqual(len(result["fold_auc"]), 5)
        self.assertGreater(result["std_auc"], 0)
        self.assertTrue(np.isfinite(result["logloss"]))
        oof.loc[0, "id"] = oof.loc[1, "id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            summarize_oof(oof)

    def test_constant_prediction_auc_is_half(self):
        labels = [0, 1] * 50
        self.assertAlmostEqual(auc_score(labels, np.ones(100)), 0.5)
        self.assertTrue(0 < logloss_score(labels, np.full(100, 0.5)) < 1)


if __name__ == "__main__":
    unittest.main()
