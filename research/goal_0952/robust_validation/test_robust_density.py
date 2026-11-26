import unittest

import numpy as np

from robust_density import isolated_meta, pool_ordered_counts, shrunken_context_contrast


class RobustDensityTests(unittest.TestCase):
    def test_pooling_conserves_mass_and_each_neighbor_gets_only_one_hop(self):
        counts = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        pooled = pool_ordered_counts(counts)
        np.testing.assert_allclose(pooled, [[8, 2, 0], [2, 6, 2], [0, 2, 8]])
        np.testing.assert_allclose(pooled.sum(axis=1), counts.sum(axis=1))
        np.testing.assert_allclose(pool_ordered_counts(np.array([[3]])), [[3]])

    def test_sparse_or_unseen_contexts_revert_to_the_global_prior(self):
        y = np.tile(np.r_[np.zeros(160), np.ones(40)], 2)
        values = np.r_[np.zeros(80), np.ones(80), np.zeros(40),
                       np.zeros(80), np.ones(80), np.ones(40)].astype(np.int64)
        contexts = np.repeat([0, 1], 200)
        result = shrunken_context_contrast(values, contexts, 3, 3, y,
                                          np.array([0, 0, 0, 2]), np.array([0, 1, 2, 0]), False)
        self.assertGreater(result[0], 0)
        self.assertLess(result[1], 0)
        np.testing.assert_array_equal(result[2:], [0, 0])

    def test_meta_fit_only_uses_reference_rows_and_crossfits_training_offsets(self):
        ref = np.arange(30, dtype=np.float64)[:, None]
        query = 100. + np.arange(6, dtype=np.float64)[:, None]
        y = np.resize([0, 1], len(ref))
        seen = []

        def fit(x, labels, _):
            rows = x[:, 0].astype(int)
            self.assertTrue((rows < len(ref)).all())
            np.testing.assert_array_equal(y[rows], labels)
            seen.append(set(rows))
            return float(len(seen)), np.zeros(1)

        train, held = isolated_meta({'_fit_nonneg_l2_weights': fit}, ref, query, y)
        for row, study in enumerate(train):
            self.assertNotIn(row, seen[int(study) - 1])
        np.testing.assert_array_equal(held, np.full(len(query), 4.))


if __name__ == '__main__':
    unittest.main()
