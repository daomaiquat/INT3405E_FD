import unittest

import numpy as np

from specialist_ranker import _balanced_negative_rows, _balanced_pair_design, _sample_negative_rows, fit_specialist


class SpecialistRankerTests(unittest.TestCase):
    def test_matched_pairs_use_nearby_rows_and_fall_back_to_the_nearest(self):
        logits = np.array([0.0, 3.0, -5.0, 0.125, 0.9, -0.375])
        negatives = np.array([2, 3, 4, 5])
        pairs = _sample_negative_rows(
            logits, np.array([0, 1]), negatives, np.random.default_rng(4), "matched"
        )

        self.assertEqual(pairs.shape, (2, 12))
        np.testing.assert_array_equal(pairs[:, :8], [[3] * 8, [4] * 8])
        self.assertTrue(np.isin(pairs[:, 8:], negatives).all())

    def test_pointwise_sampling_preserves_the_population_odds_and_frozen_offset(self):
        features = np.ones((82, 1))
        inside = np.ones(82, dtype=bool)
        labels = np.r_[1, 1, np.zeros(80)]
        coefficients = fit_specialist(features, inside, labels, np.full(82, 0.7), "pointwise", 11, l2=0)

        # Full-population log-odds = log(2 / 80). The frozen offset is +0.7.
        np.testing.assert_allclose(coefficients, [-4.388879454], rtol=0, atol=1e-5)
        self.assertEqual(coefficients.dtype, np.float64)

    def test_higher_outside_negative_logit_requires_a_larger_inside_correction(self):
        features = np.ones((2, 1))
        inside = np.array([True, True, False])
        labels = np.array([1, 0, 0])
        low = fit_specialist(features, inside, labels, [0, 0, -3], "uniform", 11, rank_weight=1, l2=0.1)
        high = fit_specialist(features, inside, labels, [0, 0, 3], "uniform", 11, rank_weight=1, l2=0.1)

        self.assertGreater(float(low[0]), 0)
        self.assertGreater(float(high[0] - low[0]), 0.5)

    def test_rank_fit_uses_the_available_negative_pool_when_one_stratum_is_empty(self):
        coefficients = fit_specialist(
            np.array([[1.0], [-1.0]]), np.array([True, True]), [1, 0], [0, 0], "matched", 7,
        )
        self.assertGreater(float(coefficients[0]), 0)

    def test_matched_fit_repeats_exactly_with_the_same_seed(self):
        rng = np.random.default_rng(150)
        inside = np.arange(180) < 140
        features = rng.normal(size=(140, 3))
        labels = (np.arange(180) % 49 == 0).astype(int)
        logits = rng.normal(size=180)
        first = fit_specialist(features, inside, labels, logits, "matched", 14)
        repeated = fit_specialist(features, inside, labels, logits, "matched", 14)

        np.testing.assert_array_equal(first, repeated)

    def test_balanced_sampling_corrects_for_the_actual_nearby_proposal(self):
        logits = np.array([0.0, 2.0, -0.125, 0.125, 1.0, 4.0])
        negatives, weights = _balanced_negative_rows(
            logits, np.array([0, 1]), np.array([2, 3, 4, 5]), np.random.default_rng(4),
        )

        self.assertTrue(np.isin(negatives[0, :6], [2, 3]).all())
        np.testing.assert_array_equal(negatives[1, :6], [4] * 6)
        # The target is 1/4. Nearby proposal mass is 7/16 with two
        # candidates and 13/16 with a single nearest candidate.
        np.testing.assert_allclose(weights[0], np.where(np.isin(negatives[0], [2, 3]), 4 / 7, 4))
        np.testing.assert_allclose(weights[1], np.where(negatives[1] == 4, 4 / 13, 4))

    def test_balanced_design_uses_signed_features_and_original_offsets_for_all_affected_pairs(self):
        design, offsets, mass = _balanced_pair_design(
            np.array([[3.0], [5.0]]), np.array([True, True, False, False]),
            np.array([1, 0, 1, 0]), np.array([1.0, 0.0, 4.0, 2.0]), np.random.default_rng(7),
        )

        np.testing.assert_array_equal(design[:, 0], [-2] * 8 + [3] * 8 + [-5] * 8)
        np.testing.assert_array_equal(offsets, [1] * 8 + [-1] * 8 + [4] * 8)
        np.testing.assert_allclose(mass, 1 / 24)

    def test_balanced_design_weights_each_stratum_by_its_population_pair_count(self):
        _, _, mass = _balanced_pair_design(
            np.ones((3, 1)), np.array([True, True, True, False, False, False]),
            np.array([1, 0, 0, 1, 1, 0]), np.zeros(6), np.random.default_rng(8),
        )

        # The full stratum pair counts are 1*2, 1*1, and 2*2.
        np.testing.assert_allclose([mass[:8].sum(), mass[8:16].sum(), mass[16:].sum()], [2 / 7, 1 / 7, 4 / 7])

    def test_balanced_fit_accounts_for_fraud_outside_the_cohort(self):
        features = np.ones((2, 1))
        inside = np.array([True, True, False, False])
        labels = np.array([1, 0, 1, 0])
        logits = np.array([0.0, 0.0, -3.0, 3.0])
        balanced = fit_specialist(features, inside, labels, logits, "balanced", 1, rank_weight=1, l2=0.1)

        # In this symmetric fixture the two crossing strata exert equal and
        # opposite forces at zero. The inside-only term is also minimized at zero.
        np.testing.assert_allclose(balanced, [0.0], atol=1e-8)
