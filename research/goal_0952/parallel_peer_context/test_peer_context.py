import unittest

import numpy as np
from scipy.special import ndtr

from joint_features import conditional_normal_scores, pair_features


class PeerContextTest(unittest.TestCase):
    def test_empirical_midrank_and_parent_fallback(self):
        values = np.array([[1., 1.], [2., 2.], [2., 2.], [3., 3.]])
        query = np.array([[2., 2.], [3., 3.], [3., 3.]])
        fit_keys = (np.array([1, 1, 1, 1]), np.array([1, 1, 2, 2]))
        query_keys = (np.array([0, 1, 1]), np.array([0, 0, 1]))
        scores = conditional_normal_scores(values, query, fit_keys, query_keys, (2, 4))
        np.testing.assert_allclose(ndtr(scores[0]), .5)
        np.testing.assert_allclose(ndtr(scores[1]), .8)
        assert np.all(scores[2] > scores[1])

    def test_disagreement_is_unusual_in_a_correlated_reference(self):
        x = np.linspace(-2, 2, 801)
        values = np.column_stack((x, x))
        query = np.array([[1., 1.], [1., -1.], [4., 4.]])
        features = pair_features(values, query, (), (), ())
        self.assertEqual(features.shape, (3, 3))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(features[1, 0], features[0, 0])
        self.assertLess(features[1, 1], features[0, 1])
