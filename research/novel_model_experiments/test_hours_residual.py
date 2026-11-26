import unittest

import numpy as np

from hours_residual import fit_adjustments, lookup


class HoursAdjustmentTests(unittest.TestCase):
    def test_flat_risk_has_no_correction_and_unknown_keys_stay_neutral(self):
        keys = np.repeat([99, 100, 101], 20)
        labels = np.tile([1] + [0] * 19, 3)
        table = fit_adjustments(keys, labels, np.full(60, 0.05))
        np.testing.assert_allclose(lookup(table, np.array([-1, 99, 100, 101, 200])), 0, atol=1e-14)

    def test_isolated_excess_increases_score_with_bounded_adjustment(self):
        keys = np.repeat([99, 100, 101], 20)
        labels = np.r_[np.zeros(20), np.ones(20), np.zeros(20)]
        table = fit_adjustments(keys, labels, np.full(60, 0.05))
        values = lookup(table, np.array([99, 100, 101]))
        self.assertGreater(values[1], 0)
        self.assertLessEqual(values[1], 0.175)
        self.assertLess(values[0], values[1])


if __name__ == '__main__':
    unittest.main()
