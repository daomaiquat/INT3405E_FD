import unittest

import numpy as np
from scipy.special import expit

from ranking_meta import fit_rank_stacker


class RankingMetaTests(unittest.TestCase):
    def test_pairwise_term_increases_signal_weight_and_preserves_nonnegativity(self):
        labels = np.array([0, 0, 1, 1])
        matrix = np.zeros((4, 7))
        matrix[:, 0] = [-1, -1, 1, 1]
        matrix[:, 1] = -matrix[:, 0]

        intercept, weights = fit_rank_stacker(
            matrix, labels, 0.25, (-0.3, np.full(7, 0.2)),
            seed=17, rank_weight=0.4, pairs_per_positive=3,
        )

        # Symmetry fixes the intercept at zero. The first weight solves
        # 0.25*w = sigmoid(-w) + 0.8*sigmoid(-2*w).
        self.assertAlmostEqual(intercept, 0.0, places=6)
        np.testing.assert_allclose(weights, [1.1963908558, 0, 0, 0, 0, 0, 0], atol=1e-6)
        probabilities = expit(intercept + matrix @ weights)
        self.assertLess(probabilities[1], probabilities[2])

    def test_intercept_recovers_imbalanced_prior_with_no_signal(self):
        intercept, weights = fit_rank_stacker(
            np.zeros((5, 7)), np.array([0, 0, 0, 0, 1]),
            0.25, (0.0, np.ones(7)), seed=42,
        )

        self.assertAlmostEqual(expit(intercept), 0.2, places=6)
        np.testing.assert_allclose(weights, np.zeros(7), atol=1e-6)

    def test_seed_repeats_sampled_pair_fit(self):
        labels = np.array([0, 0, 0, 0, 1, 1])
        matrix = np.zeros((6, 7))
        matrix[:, 0] = [-2, -1, 0, 1, 0.5, 2]
        start = (0.0, np.zeros(7))
        kwargs = {"rank_weight": 0.5, "pairs_per_positive": 1}
        first_b, first_w = fit_rank_stacker(matrix, labels, 0.1, start, seed=4, **kwargs)
        same_b, same_w = fit_rank_stacker(matrix, labels, 0.1, start, seed=4, **kwargs)
        _, other_w = fit_rank_stacker(matrix, labels, 0.1, start, seed=12, **kwargs)

        self.assertEqual(first_b, same_b)
        np.testing.assert_array_equal(first_w, same_w)
        self.assertGreater(abs(first_w[0] - other_w[0]), 0.01)

    def test_single_class_has_no_valid_pairs(self):
        with self.assertRaisesRegex(ValueError, "both classes"):
            fit_rank_stacker(np.ones((4, 7)), np.zeros(4), 0.25,
                             (0.0, np.zeros(7)), seed=7)


if __name__ == "__main__":
    unittest.main()
