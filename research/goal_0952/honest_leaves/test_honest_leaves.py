import unittest

import numpy as np

from run_honest_leaves import CAP, CONTROL, METHODS, RIDGE, candidates, leaf_changes


class LeafCorrectionTests(unittest.TestCase):
    def test_common_logit_shift_is_removed_from_every_supported_leaf(self):
        hessian = np.array([10., 100., 30.])
        counts = np.array([300, 5000, 750])
        common = .02
        result = leaf_changes(hessian * common, hessian, counts, (hessian.sum() + RIDGE) * common,
                              hessian.sum())
        np.testing.assert_allclose(result, 0., atol=1e-15)

    def test_unsupported_leaves_are_neutral_and_supported_steps_are_bounded(self):
        result = leaf_changes(np.array([10., 40., 100., -200.]), np.array([10., 7., 50., 100.]),
                              np.array([79, 400, 400, 600]), 0., 200.)
        np.testing.assert_allclose(result, [0, 0, CAP, -CAP])

    def test_leaf_correction_comparison_uses_the_same_subforest(self):
        ensemble = np.array([-5., -2., -1.])
        original = np.array([-5.3, -1.9, -1.2])
        small = np.array([-5.1, -2.2, -1.1])
        correction = np.array([.04, -.06, .02])
        outputs = candidates(ensemble, original, small, correction, .85)
        np.testing.assert_array_equal(outputs[CONTROL], ensemble)
        np.testing.assert_allclose(outputs[METHODS[1]] - outputs[METHODS[0]], .85 * .25 * correction)
        np.testing.assert_allclose(outputs[METHODS[2]] - ensemble, .85 * correction)


if __name__ == '__main__':
    unittest.main()
