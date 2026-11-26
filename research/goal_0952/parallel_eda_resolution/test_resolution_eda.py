import unittest

import numpy as np

from resolution_eda import QUANTILES, fine_codes, fine_detail


class CellContrastTests(unittest.TestCase):
    def test_fine_codes_nest_inside_original_bins_with_quantile_ties(self):
        reference = np.r_[np.zeros(100), np.repeat(np.arange(1, 52), 7), np.full(50, 52)]
        scoring = np.r_[-10, reference, 55.]
        fine_train, fine_test, parent = fine_codes(reference, scoring)
        coarse = np.unique(np.quantile(reference, QUANTILES))
        np.testing.assert_array_equal(parent[fine_train], np.searchsorted(coarse, reference))
        np.testing.assert_array_equal(parent[fine_test], np.searchsorted(coarse, scoring))
        self.assertLessEqual(np.bincount(parent).max(), 2)

    def test_cell_constants_and_category_independent_trend_cancel(self):
        information = np.array([[50., 40., 15.], [60., 30., 40.], [30., 100., 55.], [60., 40., 20.]])
        changes = np.array([[.2, -.4, .8], [.3, -.3, .9], [.02, .07, .1], [.2, .25, .28]])
        counts = information * 10.
        result = fine_detail(information * changes, information, counts, np.array([0, 0, 1, 1]))
        np.testing.assert_allclose(result, 0., atol=1e-15)

    def test_opposite_sibling_effects_are_centered_within_each_category(self):
        h = np.array([[40., 30., 0.], [60., 50., 21.]])
        gradient = h * np.array([[-.15, .15, .7], [.15, -.15, .8]])
        result = fine_detail(gradient, h, h * 20., np.array([0, 0]))
        self.assertGreater(result[1, 0], 0.)
        self.assertLess(result[1, 1], 0.)
        np.testing.assert_array_equal(result[:, 2], 0.)
        np.testing.assert_allclose((h * result).sum(axis=0), 0., atol=1e-14)


if __name__ == '__main__':
    unittest.main()
