import unittest

import numpy as np
import pandas as pd

from multiscale_tree import copula_table, local_contrasts, cross_fit


class FeatureTests(unittest.TestCase):
    def test_neighborhood_excludes_the_center_and_handles_unseen_ticks(self):
        keys = np.array([100] * 70 + [99] * 15 + [101] * 15 + [500] * 50)
        labels = np.array([1] * 40 + [0] * 58 + [1] * 2 + [0] * 50)
        result = local_contrasts(keys, labels, np.array([100, 101, 300]), 2, 10)
        self.assertEqual(result.shape, (3, 4))
        self.assertTrue(np.isfinite(result).all())
        self.assertLess(result[0, 0], result[1, 0])
        self.assertGreater(result[0, 2], 0)
        self.assertEqual(result[2, 2], 0)

    def test_copula_removes_marginals_and_recovers_dependency(self):
        negative = np.array([[81, 9], [9, 1]])
        independent = np.array([[1, 9], [9, 81]])
        dependent = np.array([[50, 0], [0, 50]])
        np.testing.assert_allclose(copula_table(negative, independent), 0, atol=0.09)
        contrasts = copula_table(negative, dependent)
        self.assertGreater(contrasts[0, 0], contrasts[0, 1])
        self.assertGreater(contrasts[1, 1], contrasts[1, 0])

    def test_crossfit_never_passes_validation_or_test_labels_to_the_encoder(self):
        frame = pd.DataFrame({'row': np.arange(48)})
        labels = np.resize([0, 1], 40)
        fit = np.arange(32)
        queries = np.arange(32, 48)

        def check_scope(reference, fold_labels, query):
            self.assertFalse(np.intersect1d(reference['row'], query['row']).size)
            self.assertTrue(np.isin(reference['row'], fit).all())
            np.testing.assert_array_equal(fold_labels, labels[reference['row']])
            return np.asarray(query[['row']], dtype=np.float32)

        built_train, built_query = cross_fit(check_scope, frame, labels, fit, queries)
        np.testing.assert_array_equal(built_train[:, 0], fit)
        np.testing.assert_array_equal(built_query[:, 0], queries)


if __name__ == '__main__':
    unittest.main()
