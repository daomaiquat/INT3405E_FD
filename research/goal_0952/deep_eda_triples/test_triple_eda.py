import unittest

import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss

from triple_eda import estimate, gain, holm, split


class TripleTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.column_stack(np.unravel_index(np.arange(8), (2, 2, 2)))
        self.info = np.array([10., 25., 75., 31., 42., 32., 11., 17.])

    def test_every_pair_marginal_is_removed_under_uneven_support(self):
        a, b, c = self.grid.T
        only_pairs = .1 * (a * b) - .15 * (a * c) + .4 * (b * c) + .3 * a
        estimated, supported, remainder = estimate(only_pairs * self.info, self.info, np.full(8, 300), self.grid)
        self.assertEqual(remainder, 1)
        self.assertTrue(supported.all())
        np.testing.assert_allclose(estimated, 0., atol=1e-13)

    def test_pure_three_way_effect_retains_sign_and_remains_orthogonal(self):
        a, b, c = self.grid.T
        sign = (-1.)**(a + b + c)
        estimated, _, _ = estimate(.8 * sign * self.info, self.info, np.full(8, 300), self.grid)
        np.testing.assert_array_equal(np.sign(estimated), sign)
        self.assertLessEqual(np.max(np.abs(estimated)), .18)
        for x, y in ((a, b), (a, c), (b, c)):
            np.testing.assert_allclose(np.bincount(2*x + y, weights=self.info * estimated), 0., atol=1e-12)

    def test_sparse_support_does_not_produce_spurious_effects(self):
        counts = np.array([20, *([300]*7)])
        result, supported, freedom = estimate(np.linspace(-1., 1., 8), self.info, counts, self.grid)
        self.assertFalse(supported[0])
        self.assertEqual(freedom, 0)
        np.testing.assert_array_equal(result, 0.)

    def test_fold_split_covers_each_id_and_logloss_gain_has_correct_orientation(self):
        labels = np.tile([0, 1], 100)
        rows = split(labels)
        np.testing.assert_array_equal(np.sort(np.concatenate(rows)), np.arange(len(labels)))
        baseline = np.full(len(labels), -.5)
        correction = .15 * (2*labels - 1)
        result = gain(labels, baseline, correction)
        expected = log_loss(labels, expit(baseline)) - log_loss(labels, expit(baseline + correction))
        self.assertAlmostEqual(result['gain_logloss_x1m'], expected * 1e6, places=7)

    def test_holm_adjustment_retains_monotonic_p_value_order(self):
        records = [{'confirm_c': {'conditional_p_one_sided': p}} for p in [.035, .01, .02]]
        holm(records)
        np.testing.assert_allclose([r['confirm_c']['holm_p_one_sided'] for r in records], [.04, .03, .04])


if __name__ == '__main__':
    unittest.main()
