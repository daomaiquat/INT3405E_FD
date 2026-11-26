import unittest

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold

from risk_eda import SPLIT_SEED, encoded_views, risk_coordinate, risk_slope_table


class RiskInteractionTests(unittest.TestCase):
    def test_cell_intercept_is_not_a_risk_interaction(self):
        keys = np.repeat(np.arange(4), 600)
        coordinate = np.tile(np.linspace(-1, 1, 600), 4)
        information = np.full(len(keys), .2)
        intercept = np.array([.2, -.2, -.2, .2])[keys]
        _, slope = risk_slope_table(keys, (2, 2), intercept * information, information, coordinate)
        np.testing.assert_allclose(slope, 0., atol=1e-14)

    def test_supported_risk_interactions_preserve_sign_and_unseen_cells_are_neutral(self):
        keys = np.repeat([0, 1, 3, 4], 600)
        coordinate = np.tile(np.r_[np.full(300, -.8), np.full(300, .8)], 4)
        information = np.full(len(keys), .2)
        sign = np.repeat([1, -1, -1, 1], 600)
        _, slope = risk_slope_table(keys, (3, 3), sign * .2 * information * coordinate, information, coordinate)
        self.assertGreater(slope[0, 0], 0.)
        self.assertGreater(slope[1, 1], 0.)
        self.assertLess(slope[0, 1], 0.)
        self.assertLess(slope[1, 0], 0.)
        np.testing.assert_array_equal(slope[2, :], 0.)
        np.testing.assert_array_equal(slope[:, 2], 0.)

    def test_risk_coordinate_is_bounded_and_monotone(self):
        z = risk_coordinate(logit(np.array([1e-8, .01, .03, .12, 1 - 1e-8])))
        self.assertEqual(z[2], 0.)
        self.assertTrue(np.all(np.diff(z) > 0.))
        self.assertTrue(np.all(np.abs(z) <= 1.))

    def test_shared_encoding_path_excludes_each_training_row(self):
        labels = np.tile([0, 1], 20)
        fitting = pd.DataFrame({'id': np.arange(len(labels))})
        scoring = pd.DataFrame({'id': [42, 43]})
        offsets = fitting.id.to_numpy() / 10.
        score_offsets = scoring.id.to_numpy() / 10.

        def encoder(a, y, p, b, q, _pairs):
            self.assertTrue(set(a.id).isdisjoint(b.id))
            np.testing.assert_array_equal(labels[a.id], y)
            np.testing.assert_array_equal(a.id.to_numpy() / 10., p)
            np.testing.assert_array_equal(b.id.to_numpy() / 10., q)
            return np.column_stack((b.id, np.full(len(b), len(a))))

        encoded, score = encoded_views(fitting, scoring, labels, offsets, score_offsets, (), encoder)
        for reference, held in StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED).split(labels, labels):
            np.testing.assert_array_equal(encoded[held], np.column_stack((held, np.full(len(held), len(reference)))))
        np.testing.assert_array_equal(score, [[42, len(fitting)], [43, len(fitting)]])


if __name__ == '__main__':
    unittest.main()
