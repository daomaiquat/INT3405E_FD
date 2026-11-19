"""Step 3 checks: id-keyed OOF/test assembly and fold status/resume helpers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import fold_needs_run, read_fold_status, read_json, write_fold_status
from src.predictions import (assert_one_to_one, join_oof, join_test, prediction_frame,
                             shuffle_for_check)
from src.smoke_model import fit_contract_model, load_bundle, predict_bundle, save_bundle
from src.validation import assign_folds

CONFIG = read_json(Path(__file__).resolve().parents[1] / "configs/default.json")
TARGET = CONFIG["target"]


def fixture():
    rows = 60
    frame = {"id": np.arange(rows, dtype=np.int64)}
    for col in CONFIG["numeric_columns"]:
        frame[col] = np.arange(rows, dtype=np.float64) / 10.0
    for col in CONFIG["categorical_columns"]:
        frame[col] = ["A"] * rows
    for col in CONFIG["flag_columns"]:
        frame[col] = [False] * rows
    train = pd.DataFrame(frame)
    train[TARGET] = [0, 1] * (rows // 2)
    folds = assign_folds(train, TARGET, 3, CONFIG["seed"])
    test = train.drop(columns=TARGET).iloc[:10].copy()
    test["id"] = np.arange(1000, 1010)
    sample = pd.DataFrame({"id": test["id"], TARGET: 0.0})
    rng = np.random.default_rng(3)
    oof_outputs = {}
    test_outputs = {}
    for fold in range(3):
        valid_ids = train.loc[folds["fold"] == fold, "id"].to_numpy()
        oof_outputs[fold] = prediction_frame(valid_ids, rng.random(len(valid_ids)), fold)
        test_outputs[fold] = prediction_frame(test["id"].to_numpy(), rng.random(len(test)), fold)
    return train, test, sample, folds, oof_outputs, test_outputs


class PredictionContractTests(unittest.TestCase):
    def setUp(self):
        (self.train, self.test, self.sample, self.folds,
         self.oof_outputs, self.test_outputs) = fixture()

    def test_oof_is_one_to_one_and_fold_aligned(self):
        oof = join_oof(self.oof_outputs, self.folds, CONFIG)
        self.assertEqual(list(oof.columns), ["id", TARGET, "fold", "prediction"])
        self.assertTrue(set(oof[TARGET].unique()) == {0, 1})
        self.assertTrue(oof["id"].is_unique)
        self.assertEqual(len(oof), len(self.train))
        broken = {k: v.copy() for k, v in self.oof_outputs.items()}
        broken[0].loc[broken[0].index[0], "id"] = broken[1]["id"].iloc[0]
        with self.assertRaisesRegex(ValueError, "unique"):
            join_oof(broken, self.folds, CONFIG)
        wrong = {k: v.copy() for k, v in self.oof_outputs.items()}
        wrong[0]["fold"] = 1
        with self.assertRaisesRegex(ValueError, "producer fold"):
            join_oof(wrong, self.folds, CONFIG)

    def test_shuffled_inputs_join_identically_by_id(self):
        shuffled_oof = {k: shuffle_for_check(v, 11 + k) for k, v in self.oof_outputs.items()}
        shuffled_test = {k: shuffle_for_check(v, 21 + k) for k, v in self.test_outputs.items()}
        pd.testing.assert_frame_equal(join_oof(self.oof_outputs, self.folds, CONFIG),
                                      join_oof(shuffled_oof, self.folds, CONFIG))
        pd.testing.assert_frame_equal(join_test(self.test_outputs, self.sample, 3),
                                      join_test(shuffled_test, self.sample, 3))

    def test_test_prediction_is_fold_average_in_sample_order(self):
        result = join_test(self.test_outputs, self.sample, 3)
        self.assertEqual(list(result.columns), ["id", "fold_0", "fold_1", "fold_2", "prediction"])
        self.assertTrue(result["id"].equals(self.sample["id"]))
        expected = result[["fold_0", "fold_1", "fold_2"]].mean(axis=1).rename("prediction")
        pd.testing.assert_series_equal(result["prediction"], expected)
        broken = {k: v.copy() for k, v in self.test_outputs.items()}
        broken[0].loc[broken[0].index[0], "id"] = 9999
        with self.assertRaisesRegex(ValueError, "differ from sample"):
            join_test(broken, self.sample, 3)
        with self.assertRaisesRegex(ValueError, "unique"):
            assert_one_to_one([1, 1], "ids")

    def test_bundle_roundtrip_reproduces_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fit_contract_model(self.train, CONFIG)
            path = Path(tmp) / "model.joblib"
            save_bundle(bundle, path)
            first = predict_bundle(bundle, self.test)
            second = predict_bundle(load_bundle(path), self.test)
            np.testing.assert_allclose(first, second, atol=1e-12)
            self.assertEqual(load_bundle(path)["n_trees"], 0)

    def test_fold_status_resume_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fold_status.json"
            write_fold_status(path, {"0": "failed", "1": "completed"})
            self.assertEqual(read_fold_status(path)["0"], "failed")
            existing = [Path(tmp) / "model.joblib"]
            existing[0].write_text("x", encoding="utf-8")
            self.assertTrue(fold_needs_run(read_fold_status(path), 0, existing))
            self.assertFalse(fold_needs_run(read_fold_status(path), 1, existing))


if __name__ == "__main__":
    unittest.main()
