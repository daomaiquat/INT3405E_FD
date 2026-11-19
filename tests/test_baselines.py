"""Step 5 checks: baseline registry, fold-local preprocessing and artifact contract.

The fixtures are tiny; no full-data baseline is trained by these tests.
"""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import read_json
from src.features import feature_matrix
from src.models import BASELINE_IDS, BASELINES, fit_bundle, predict_bundle, select_iterations
from src.training import run_baseline
from src.validation import assign_folds, early_stop_assignments

CONFIG = read_json(Path(__file__).resolve().parents[1] / "configs/default.json")
TARGET = CONFIG["target"]


def small_config():
    config = copy.deepcopy(CONFIG)
    config["baselines"]["logistic"]["max_iter"] = 200
    config["baselines"]["decision_tree"]["min_samples_leaf"] = 5
    config["baselines"]["random_forest"].update(n_estimators=10, min_samples_leaf=2)
    for key in ("lightgbm", "xgboost"):
        config["baselines"][key].update(max_iterations=12, learning_rate=0.1)
    config["baselines"]["lightgbm"].update(num_leaves=4, min_child_samples=5)
    config["baselines"]["xgboost"].update(max_depth=2, min_child_weight=1)
    config["baselines"]["catboost"].update(max_iterations=12, learning_rate=0.1,
                                           depth=2, l2_leaf_reg=1.0)
    config["baselines"]["early_stopping"]["patience"] = 3
    return config


def make_fixture(rows: int = 180):
    rng = np.random.default_rng(3)
    index = np.arange(rows)
    labels = (index % 4 == 0).astype(np.int8)
    frame = {"id": index.astype(np.int64)}
    for offset, column in enumerate(CONFIG["numeric_columns"]):
        if column == "cvv_retry_count":
            frame[column] = (index % 4).astype(np.float64)
        else:
            frame[column] = 1.0 + (index % (13 + offset)) / (2.0 + offset) \
                + rng.normal(0.0, 0.01, rows)
    frame["merchant_category"] = [f"M{value % 5}" for value in index]
    frame["card_type"] = [("A", "B", "C")[value % 3] for value in index]
    frame["auth_method"] = np.where(labels == 1, "No Authentication", "Biometric")
    frame["channel"] = [("web", "app")[value % 2] for value in index]
    frame["device_type"] = [("mobile", "desktop")[value % 2] for value in index]
    for column in CONFIG["flag_columns"]:
        frame[column] = np.zeros(rows, dtype=np.int8)
    frame["is_foreign_transaction"] = labels.copy()
    frame["ip_country_mismatch"] = labels.copy()
    train = pd.DataFrame(frame)
    train[TARGET] = labels

    size = 80
    test = train.drop(columns=TARGET).iloc[:size].copy()
    test["id"] = np.arange(1000, 1000 + size, dtype=np.int64)
    for column in CONFIG["numeric_columns"]:
        test[column] = test[column] + 0.01
    sample = pd.DataFrame({"id": test["id"], TARGET: 0.0})
    config = small_config()
    folds = assign_folds(train, TARGET, 2, config["seed"])
    early = early_stop_assignments(train, folds, config, 2)
    return train, test, sample, folds, early, config


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.train, self.test, self.sample, self.folds, self.early, self.config = make_fixture()

    def test_registry_has_eight_models_and_distinct_flags(self):
        self.assertEqual(BASELINE_IDS, ("B00", "B01", "B02", "B03", "B04", "B05", "B06", "B07"))
        flags = {f"--{model_id.lower()}" for model_id in BASELINE_IDS}
        self.assertEqual(len(flags), 8)
        self.assertTrue(BASELINES["B05"].early_stop)
        self.assertTrue(BASELINES["B07"].early_stop)
        self.assertFalse(BASELINES["B02"].early_stop)

    def test_feature_modes_are_stable_and_do_not_mutate_input(self):
        before = self.train.copy(deep=True)
        glm = feature_matrix(self.train, self.config, "glm")
        numeric = feature_matrix(self.train, self.config, "numeric")
        onehot = feature_matrix(self.train, self.config, "tree_onehot")
        self.assertIn("log_amount_usd", glm.columns)
        self.assertIn("merchant_category", glm.columns)
        self.assertNotIn("merchant_category", numeric.columns)
        self.assertIn("merchant_category", onehot.columns)
        pd.testing.assert_frame_equal(self.train, before)

    def test_constant_and_random_baselines_are_reproducible(self):
        prior = fit_bundle("B00", self.train, self.config)
        self.assertAlmostEqual(float(predict_bundle(prior, self.test)[0]),
                               float(self.train[TARGET].mean()))
        random = fit_bundle("B01", self.train, self.config, fold=1)
        first = predict_bundle(random, self.test)
        second = predict_bundle(random, self.test)
        np.testing.assert_array_equal(first, second)
        self.assertTrue((first >= 0).all() and (first <= 1).all())

    def test_sklearn_baselines_predict_finite_with_unseen_category(self):
        altered = self.test.copy()
        altered.loc[altered.index[0], "merchant_category"] = "UNSEEN"
        for model_id in ("B02", "B03", "B04"):
            bundle = fit_bundle(model_id, self.train, self.config)
            for frame in (self.test, altered):
                prediction = predict_bundle(bundle, frame)
                self.assertEqual(len(prediction), len(frame), model_id)
                self.assertTrue(np.isfinite(prediction).all(), model_id)
                self.assertTrue(((prediction >= 0) & (prediction <= 1)).all(), model_id)

    def test_boosters_select_trees_on_e_and_refit_inside_limits(self):
        fit_part, eval_part = self.train.iloc[:140], self.train.iloc[140:]
        for model_id in ("B05", "B06", "B07"):
            selection = select_iterations(model_id, fit_part, eval_part, self.config)
            self.assertGreaterEqual(selection["n_trees"], 1, model_id)
            self.assertLessEqual(selection["n_trees"], 12, model_id)
            self.assertTrue(0.0 <= selection["eval_auc"] <= 1.0, model_id)
            bundle = fit_bundle(model_id, self.train, self.config,
                                n_trees=selection["n_trees"])
            prediction = predict_bundle(bundle, self.test)
            self.assertTrue(np.isfinite(prediction).all(), model_id)
            self.assertEqual(bundle["n_trees"], selection["n_trees"])

    def test_run_baseline_writes_contract_artifacts_without_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            state = {"run_id": "unit", "resumed": False, "smoke": True}
            for model_id in ("B00", "B01"):
                summary = run_baseline(model_id, run_dir, self.train, self.test, self.sample,
                                       self.folds, self.early, self.config, 2, state)
                self.assertEqual(summary["model"], model_id)
                experiment = run_dir / "baselines" / model_id
                for name in ("oof.csv", "test_by_fold.csv", "metrics.json", "fold_metrics.json",
                             "baseline_checks.json", "feature_groups.json", "params.json",
                             "fold_status.json", "model_bundle/fold_0.joblib"):
                    self.assertTrue((experiment / name).exists(), name)
                metrics = read_json(experiment / "metrics.json")
                self.assertEqual(metrics["rows"], len(self.train))
                self.assertIn("train_auc", metrics)
                self.assertIn("0", metrics["fold_auc"])
                oof = pd.read_csv(experiment / "oof.csv")
                self.assertTrue(oof["id"].is_unique)
                self.assertEqual(set(oof[TARGET]), {0, 1})
            self.assertTrue((run_dir / "baseline_comparison.csv").is_file())
            self.assertTrue((run_dir / "baseline_oof_spearman.csv").is_file())
            self.assertEqual(len(pd.read_csv(run_dir / "baseline_comparison.csv")), 2)
            self.assertFalse((run_dir / "submission.csv").exists())
            self.assertFalse((run_dir / "submission_smoke.csv").exists())


if __name__ == "__main__":
    unittest.main()
