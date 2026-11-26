import unittest

import numpy as np
import pandas as pd

from context_expansion import conditional_contrast
from run_followup import combine, crossfit_extra, risk_weights
from structured_offset import gate, supported


class FollowupTests(unittest.TestCase):
    def test_risk_weights_have_positive_floor_and_unit_mean(self):
        weights = risk_weights([-20, -5, -3, 1, 10])
        self.assertAlmostEqual(weights.mean(), 1)
        self.assertTrue((weights > 0).all() and (np.diff(weights) < 0).all())
        self.assertTrue((np.diff(gate(np.array([-20., -5., -3., 1., 10.]))) < 0).all())

    def test_crossfit_does_not_read_held_or_query_labels(self):
        fit = pd.DataFrame({'position': np.arange(30)})
        query = pd.DataFrame({'position': np.arange(30, 38)})
        y = np.resize([0, 1], len(fit))

        def encoder(ref, labels, target):
            self.assertEqual(len(np.intersect1d(ref['position'], target['position'])), 0)
            np.testing.assert_array_equal(y[ref['position']], labels)
            return target[['position']].to_numpy(dtype=np.float32)

        a, b = crossfit_extra(encoder, fit, y, query)
        np.testing.assert_array_equal(a[:, 0], fit['position'])
        np.testing.assert_array_equal(b[:, 0], query['position'])

    def test_combiner_replaces_only_the_requested_share_of_the_parent_tree(self):
        incumbent = np.array([0.1, -2.5])
        original = np.array([0.2, -2.0])
        candidate = np.array([0.4, -1.0])
        result = combine(incumbent, original, {'new': candidate}, {'new': .5})
        np.testing.assert_allclose(result, incumbent + .425 * (candidate - original))

    def test_conditional_contrast_detects_class_specific_association_and_defaults_to_neutral(self):
        contexts = np.repeat([0, 1], 220)
        labels = np.tile(np.r_[np.zeros(180), np.ones(40)], 2)
        values = np.r_[np.zeros(90), np.ones(90), np.zeros(40),
                       np.zeros(90), np.ones(90), np.ones(40)].astype(np.int64)
        result = conditional_contrast(values, contexts, 2, 3, labels,
                                      np.array([0, 0, 0]), np.array([0, 1, 2]))
        self.assertGreater(result[0], 0)
        self.assertLess(result[1], 0)
        self.assertEqual(result[2], 0)

    def test_sparse_cells_require_count_and_expected_curvature(self):
        rows = np.repeat([0, 1, 2], [100, 81, 50])
        hessian = np.repeat([.20, .04, .25], [100, 81, 50])
        training, scoring = supported(rows, np.array([0, 1, 2, 3]), 4, hessian)
        self.assertEqual(int(training.sum()), 100)
        np.testing.assert_array_equal(scoring, [1, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
