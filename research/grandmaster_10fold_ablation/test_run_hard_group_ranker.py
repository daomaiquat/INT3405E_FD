import unittest

import numpy as np
from scipy.special import expit

from run_hard_group_ranker import pair_auc, specialist_scores


class SpecialistEvaluationTests(unittest.TestCase):
    def test_pair_auc_counts_ties_as_half_for_cross_group_comparisons(self):
        self.assertAlmostEqual(pair_auc(np.array([0.2, 0.8]), np.array([0.2, 0.4, 0.9])), 2.5 / 6)

    def test_correction_changes_only_the_target_transactions(self):
        parent = np.array([0.20, 0.70, 0.90])
        mask = np.array([True, False, True])
        result = specialist_scores(parent, mask, np.array([1.0, -0.5]), 0.5, expit)
        self.assertEqual(result[1], parent[1])
        self.assertGreater(result[0], parent[0])
        self.assertLess(result[2], parent[2])
        np.testing.assert_array_equal(parent, [0.20, 0.70, 0.90])


if __name__ == "__main__":
    unittest.main()
