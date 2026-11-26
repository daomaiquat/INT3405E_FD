import unittest

import numpy as np

from alternative_model import pair_score
from blend_spline import mix_probabilities


class PairScreenTest(unittest.TestCase):
    def test_pair_score_detects_interactions_after_removing_main_effects(self):
        left = np.array([0, 0, 1, 1], dtype=np.int32).repeat(400)
        right = np.array([0, 1, 0, 1], dtype=np.int32).repeat(400)
        information = np.full(left.size, 0.25)
        additive = np.array([0.025, 0.075, -0.075, -0.025]).repeat(400)
        interaction = np.array([0.075, -0.075, -0.075, 0.075]).repeat(400)

        self.assertLess(pair_score(left, right, 2, 2, additive, information), 0)
        self.assertGreater(pair_score(left, right, 2, 2, interaction, information), 20)

    def test_pair_score_stays_finite_for_unobserved_cells(self):
        left = np.array([0, 0, 1], dtype=np.int32)
        right = np.array([0, 1, 1], dtype=np.int32)
        self.assertTrue(np.isfinite(pair_score(left, right, 2, 2, np.zeros(3), np.ones(3))))


class SplineBlendTest(unittest.TestCase):
    def test_mixes_geometric_odds_at_fixed_90_10_weights(self):
        # FINAL p=1/2 has odds 1. Spline p=4/5 has odds 4.
        expected = 1.0 / (1.0 + 4.0**-0.1)
        actual = mix_probabilities(np.array([0.5]), np.array([0.8]))
        np.testing.assert_allclose(actual, [expected], rtol=1e-13)


if __name__ == "__main__":
    unittest.main()
