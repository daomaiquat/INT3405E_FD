"""Verify both comparisons for candidate base logits."""

import json
from pathlib import Path
import unittest

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from hierarchical_glm_evaluation import evaluate_architecture


class HierarchicalGlmEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive = Path(__file__).resolve().parents[1] / "Model tham khảo" / "Grandmaster" / "notebook.ipynb"
        source = json.loads(archive.read_text(encoding="utf-8"))["cells"][7]["source"]
        cls.namespace = {
            "np": np, "stats": stats, "minimize": minimize, "expit": expit, "log_expit": log_expit,
            "LOGIT_EPS": 1e-10,
        }
        exec(compile("".join(source), str(archive), "exec"), cls.namespace)

    def setUp(self):
        self.labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.holdout_labels = np.array([0, 0, 1, 1])
        small, large = np.arange(4), np.arange(4, 10)
        self.folds = [(0, large, small), (1, small, large)]
        self.meta = {"baseline_meta_l2_by_fold": [0.001, 0.01]}
        self.training = np.zeros((10, 1))
        self.score_folds = [np.zeros((4, 1)), np.zeros((4, 1))]
        signal = 2 * self.labels - 1
        signal[4:] *= -1
        self.new_oof = np.column_stack((signal, -signal))
        self.new_holdout_folds = [np.array([[-2., -1.], [-1., -2.], [1., 2.], [2., 1.]])] * 2

    def evaluate(self, **overrides):
        inputs = {
            "nb": self.namespace,
            "meta": self.meta,
            "training": self.training,
            "score_folds": self.score_folds,
            "labels": self.labels,
            "holdout_labels": self.holdout_labels,
            "folds": self.folds,
            "new_oof": self.new_oof,
            "new_holdout_folds": self.new_holdout_folds,
        }
        return evaluate_architecture(**(inputs | overrides))

    def test_ensemble_adds_each_candidate_separately_and_ignores_holdout_labels_for_fit(self):
        result = self.evaluate()
        changed = self.evaluate(holdout_labels=1 - self.holdout_labels)
        np.testing.assert_allclose(result["predictions"]["current_hierarchy"], np.full(4, 0.5))
        for name, scores in result["predictions"].items():
            np.testing.assert_array_equal(scores, changed["predictions"][name])
        self.assertEqual(result["metrics"]["hierarchical_glm"]["roc_auc"], 1.0)
        self.assertEqual(result["metrics"]["hierarchical_glm_interactions"]["roc_auc"], 1.0)
        self.assertEqual(result["metrics"]["hierarchical_glm"]["delta_roc_auc"], 0.5)
        self.assertEqual(changed["metrics"]["hierarchical_glm"]["roc_auc"], 0.0)

    def test_screen_fits_on_same_fold_holdout_and_weights_delta_by_comparable_pairs(self):
        result = self.evaluate()["fold_screening"]
        folds = result["folds"]
        self.assertEqual([fold["positive_negative_pairs"] for fold in folds], [4, 9])
        self.assertEqual([fold["roc_auc"]["hierarchical_glm"] for fold in folds], [1.0, 0.0])
        self.assertEqual([fold["roc_auc"]["hierarchical_glm_interactions"] for fold in folds], [0.0, 1.0])
        self.assertEqual([fold["delta_roc_auc"]["hierarchical_glm"] for fold in folds], [0.5, -0.5])
        self.assertAlmostEqual(result["pair_weighted_delta_roc_auc"]["hierarchical_glm"], -2.5 / 13)
        self.assertAlmostEqual(result["pair_weighted_delta_roc_auc"]["hierarchical_glm_interactions"], 2.5 / 13)
        self.assertAlmostEqual(result["pair_weighted_standalone_roc_auc"]["hierarchical_glm"], 4 / 13)

    def test_screen_fixed_penalty_is_independent_of_historical_penalties(self):
        first = self.evaluate()
        second = self.evaluate(meta={"baseline_meta_l2_by_fold": [100.0, 1000.0]})
        self.assertEqual(first["fold_screening"]["screen_l2"], 1e-5)
        self.assertEqual(first["fold_screening"], second["fold_screening"])
        self.assertNotEqual(first["mean_added_logit_weights"], second["mean_added_logit_weights"])

    def test_interaction_contrast_compares_additive_and_interactions_directly(self):
        reversed_interactions = self.new_holdout_folds[0].copy()
        reversed_interactions[:, 1] *= -1
        result = self.evaluate(new_holdout_folds=[reversed_interactions] * 2)
        contrast = result["interactions_vs_additive"]
        self.assertEqual(contrast["delta_roc_auc"], -1.0)
        self.assertLess(contrast["ci95_high"], 0.0)

    def test_reports_mean_weight_assigned_to_the_added_logit(self):
        result = self.evaluate(new_oof=np.column_stack((self.new_oof[:, 0], np.zeros(10))))
        weights = result["mean_added_logit_weights"]
        self.assertGreater(weights["hierarchical_glm"], 0.0)
        self.assertEqual(weights["hierarchical_glm_interactions"], 0.0)

    def evaluate_with_inactive_fit_drift(self, **overrides):
        fit = self.namespace["_fit_nonneg_l2_weights"]

        def fit_with_drift(matrix, labels, l2):
            intercept, weights = fit(matrix, labels, l2)
            if matrix.shape[1] == 2 and weights[-1] == 0.0:
                intercept += 1e-7
            return intercept, weights

        return self.evaluate(nb=self.namespace | {"_fit_nonneg_l2_weights": fit_with_drift}, **overrides)

    def test_inactive_added_logit_preserves_exact_baseline_predictions(self):
        result = self.evaluate_with_inactive_fit_drift(
            new_oof=np.column_stack((self.new_oof[:, 0], np.zeros(10))),
            new_holdout_folds=[np.column_stack((scores[:, 0], np.zeros(4))) for scores in self.new_holdout_folds],
        )
        baseline = result["predictions"]["current_hierarchy"]
        np.testing.assert_array_equal(result["predictions"]["hierarchical_glm_interactions"], baseline)
        for fold in result["fold_screening"]["folds"]:
            self.assertEqual(fold["added_logit_weights"]["hierarchical_glm_interactions"], 0.0)
            self.assertEqual(fold["delta_roc_auc"]["hierarchical_glm_interactions"], 0.0)

    def test_inactive_ensemble_fold_uses_its_baseline_fit_when_another_fold_is_active(self):
        original = self.evaluate()
        changed = self.evaluate_with_inactive_fit_drift()
        name = "hierarchical_glm"
        self.assertGreater(changed["mean_added_logit_weights"][name], 0.0)
        np.testing.assert_array_equal(changed["predictions"][name], original["predictions"][name])

    def test_rejects_wrong_number_of_candidate_columns(self):
        with self.assertRaises(ValueError):
            self.evaluate(new_oof=np.zeros((10, 1)))

    def test_rejects_nonfinite_candidate_holdout(self):
        invalid_scores = self.new_holdout_folds[0].copy()
        invalid_scores[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.evaluate(new_holdout_folds=[invalid_scores, self.new_holdout_folds[1]])


if __name__ == "__main__":
    unittest.main()
