import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from refine_eda import SPLIT_SEED, borrow_neighbors, eda_matrices, smoothed_table


class ResidualTableTests(unittest.TestCase):
    def test_adjacent_bins_share_counts_without_mixing_categories(self):
        counts = np.array([[10, 0], [0, 0], [0, 30]], dtype=float)
        pooled = borrow_neighbors(counts, axis=0)
        np.testing.assert_allclose(pooled, [[9, 0], [1, 3], [0, 27]])
        np.testing.assert_allclose(pooled.sum(axis=0), counts.sum(axis=0))
        np.testing.assert_allclose(borrow_neighbors(counts[:1], axis=0), counts[:1])

    def test_unseen_category_remains_neutral_and_supported_contrast_keeps_its_sign(self):
        gradients = np.array([[0, 0, 0], [0, 30, -30], [0, -30, 30]], dtype=float)
        counts = np.array([[0, 0, 0], [0, 300, 300], [0, 300, 300]], dtype=float)
        table = smoothed_table(gradients, .25 * counts, counts, axes=())
        np.testing.assert_array_equal(table[:, 0], 0)
        np.testing.assert_array_equal(table[0, :], 0)
        self.assertGreater(table[1, 1], 0)
        self.assertLess(table[1, 2], 0)
        self.assertLessEqual(np.abs(table).max(), .35)

    def test_training_and_scoring_use_the_same_five_exclusion_tables(self):
        fitting = pd.DataFrame({'id': np.arange(60)})
        query = pd.DataFrame({'id': [112, 108, 103]})
        labels = np.tile([0, 1], 30)
        offsets = fitting.id.to_numpy() * .5
        references = list(StratifiedKFold(5, shuffle=True, random_state=SPLIT_SEED).split(labels, labels))

        def fake_encoder(fit, y, margins, scored, pairs):
            np.testing.assert_array_equal(y, labels[fit.id])
            np.testing.assert_array_equal(margins, offsets[fit.id])
            self.assertTrue(set(fit.id).isdisjoint(scored.id))
            return np.column_stack([scored.id, np.repeat(fit.id.sum(), len(scored))]).astype(np.float32)

        with patch('refine_eda.residual_features', side_effect=fake_encoder), \
             patch('refine_eda.smooth_features', side_effect=fake_encoder):
            original, smooth, inference = eda_matrices(fitting, query, labels, offsets, [('left', 'right')])
        self.assertEqual(len(inference), 5)
        np.testing.assert_array_equal(original[0], smooth[0])
        np.testing.assert_array_equal(original[1][:, 0], query.id)
        for (reference, excluded), scored in zip(references, inference, strict=True):
            np.testing.assert_array_equal(scored[:, 0], query.id)
            np.testing.assert_array_equal(scored[:, 1], fitting.id.iloc[reference].sum())
            np.testing.assert_array_equal(original[0][excluded, 0], fitting.id.iloc[excluded])
            np.testing.assert_array_equal(original[0][excluded, 1], scored[0, 1])


if __name__ == '__main__':
    unittest.main()
