import unittest

import numpy as np
from scipy.special import expit

from fold_aggregation import aggregate_fold_scores


class FoldAggregationTests(unittest.TestCase):
    def test_cross_fold_averaging_applies_each_formula(self):
        parameters = [(1.0, np.array([2.0, -1.0])), (-3.0, np.array([4.0, 1.0]))]
        predictions = [np.array([[1.0, 3.0], [-1.0, 0.0]]), np.array([[3.0, 2.0], [2.0, 2.0]])]

        actual = aggregate_fold_scores(parameters, predictions, expit)

        np.testing.assert_allclose(actual["mean_logits"], expit([5.0, 0.5]))
        np.testing.assert_allclose(actual["paired_logits"], expit([5.5, 3.0]))
        np.testing.assert_allclose(actual["mean_probabilities"], (expit([2.0, -4.0]) + expit([8.0, 5.0])) / 2.0)

    def test_identical_models_produce_identical_probabilities(self):
        parameters = [(0.5, np.array([1.0, -1.0]))] * 3
        predictions = [np.array([[2.0, 1.0], [0.0, 1.0]])] * 3

        actual = aggregate_fold_scores(parameters, predictions, expit)

        for result in actual.values():
            np.testing.assert_allclose(result, expit([1.5, -0.5]))

    def test_rejects_misaligned_folds_and_feature_shapes(self):
        parameters = [(0.0, np.ones(2)), (0.0, np.ones(2))]
        with self.assertRaises(ValueError):
            aggregate_fold_scores(parameters, [np.zeros((3, 2))], expit)
        with self.assertRaises(ValueError):
            aggregate_fold_scores(parameters, [np.zeros((3, 2)), np.zeros((4, 2))], expit)
        with self.assertRaises(ValueError):
            aggregate_fold_scores(parameters, [np.zeros((3, 1)), np.zeros((3, 1))], expit)


if __name__ == "__main__":
    unittest.main()
