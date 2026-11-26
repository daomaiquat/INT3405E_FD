"""Tests for fold-local ranking and attribution of low-score fraud pairs."""

import unittest

import numpy as np
from scipy.special import logit

from low_score_rank_evaluation import evaluate_fold, summarize_folds


class LowScoreRankEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.labels = [0, 0, 1, 1, 1]
        self.logits = {
            "current_grandmaster": [-6, -4, -5, logit(0.02), 2],
            "pointwise": [-6, -4, -4, -5, 2],
            "pairwise": [-6, -4, -3, -6, 2],
        }

    def test_attributes_half_credit_ties_to_fixed_baseline_strata(self):
        report = evaluate_fold(self.labels, self.logits)

        self.assertEqual(report["positive_negative_pairs"], 6)
        self.assertEqual(report["low_score_positive_count"], 1)
        self.assertEqual(report["low_score_pairs"], 2)
        self.assertEqual(report["low_score_baseline_misordered_pairs"], 1)
        expected = {
            "pointwise_vs_current_grandmaster": (0.5, -1.0),
            "pairwise_vs_current_grandmaster": (1.0, -1.5),
            "pairwise_vs_pointwise": (0.5, -0.5),
        }
        for name, (low, other) in expected.items():
            with self.subTest(contrast=name):
                contrast = report["contrasts"][name]
                self.assertEqual(contrast["low_score_net_corrected_pairs"], low)
                self.assertEqual(contrast["other_net_corrected_pairs"], other)
                self.assertAlmostEqual(6 * contrast["delta_roc_auc"], low + other)
        self.assertAlmostEqual(report["metrics"]["current_grandmaster"]["roc_auc"], 5 / 6)
        self.assertEqual(report["metrics"]["pairwise"]["roc_auc"], 0.75)

    def test_uses_raw_logits_for_ranking_and_stable_log_loss(self):
        report = evaluate_fold([0, 1], {
            "current_grandmaster": [100, 101],
            "pointwise": [101, 100],
            "pairwise": [100, 100],
        })

        expected = {
            "current_grandmaster": (1.0, 1.0, 50.0),
            "pointwise": (0.0, 0.5, 50.5),
            "pairwise": (0.5, 0.5, 50.0),
        }
        for name, (auc, average_precision, loss) in expected.items():
            with self.subTest(model=name):
                metrics = report["metrics"][name]
                self.assertEqual(metrics["roc_auc"], auc)
                self.assertEqual(metrics["average_precision"], average_precision)
                self.assertAlmostEqual(metrics["log_loss"], loss)

    def test_summarizes_only_comparable_within_fold_pairs(self):
        first = evaluate_fold(self.labels, self.logits)
        second = evaluate_fold([0, 0, 1], {
            "current_grandmaster": [-4, -5, -6],
            "pointwise": [-4, -5, -4],
            "pairwise": [-4, -5, -3],
        })

        summary = summarize_folds([first, second])

        self.assertEqual(summary["fold_count"], 2)
        self.assertEqual(summary["positive_negative_pairs"], 8)
        self.assertEqual(summary["low_score_positive_count"], 2)
        self.assertEqual(summary["low_score_pairs"], 4)
        self.assertEqual(summary["low_score_baseline_misordered_pairs"], 3)
        self.assertEqual(summary["pair_weighted_roc_auc"], {
            "current_grandmaster": 5 / 8,
            "pointwise": 6 / 8,
            "pairwise": 6.5 / 8,
        })
        contrast = summary["contrasts"]["pairwise_vs_pointwise"]
        self.assertEqual(contrast["pair_weighted_delta_roc_auc"], 0.5 / 8)
        self.assertEqual(contrast["positive_folds"], 1)
        self.assertEqual(contrast["low_score_net_corrected_pairs"], 1.0)
        self.assertEqual(contrast["other_net_corrected_pairs"], -0.5)

    def test_rejects_nonfinite_logits_and_single_class_folds(self):
        with self.assertRaises(ValueError):
            evaluate_fold(self.labels, self.logits | {"pointwise": np.full(5, np.inf)})
        with self.assertRaises(ValueError):
            evaluate_fold([0] * 5, self.logits)

    def test_rejects_empty_fold_summary(self):
        with self.assertRaises(ValueError):
            summarize_folds([])


if __name__ == "__main__":
    unittest.main()
