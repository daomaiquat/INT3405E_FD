import unittest

import numpy as np

from improvement_features import amount_parent_feature, residual_pair_features


class AmountParentFeatureTests(unittest.TestCase):
    def test_shrinks_rare_and_unseen_amounts_to_their_tenth_dollar_parent(self):
        keys = np.array([12340, 12341, 12342, 12343, 12345, 12345, *([99900] * 4)])
        labels = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])

        # Reference fraud rate = 3/10. The rare child uses its parent's other
        # 2/4 labels. The unseen child uses the parent's full 3/6 labels.
        want = np.log([196 / 159, 21 / 19, 1])
        got = amount_parent_feature(keys, labels, np.array([12345, 12349, 12359]))

        self.assertEqual(got.dtype, np.dtype(np.float32))
        np.testing.assert_allclose(got, want, rtol=1e-6)

    def test_does_not_use_densely_seen_children_or_unsupported_parent_peers(self):
        keys = np.repeat([2000, 2001, 3050, 3051], [4, 4, 3, 3])
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0])

        got = amount_parent_feature(keys, labels, np.array([2000, 3050, 3060]))

        np.testing.assert_array_equal(got, [0.0, 0.0, 0.0])


class ResidualPairFeatureTests(unittest.TestCase):
    def test_applies_the_signed_newton_step_per_column_and_zero_for_unseen_groups(self):
        groups = np.array([[0, 3], [0, 4], [2, 3], [2, 4]])
        labels = np.array([1, 0, 0, 0])
        expected_fraud = np.array([0.2, 0.3, 0.25, 0.5])
        scoring = np.array([[0, 3], [2, 4], [9, 2], [0, 4], [0, 4]])

        got = residual_pair_features(groups, labels, expected_fraud, scoring)
        want = [[0.5 / 5.37, 0.55 / 5.3475], [-0.75 / 5.4375, -0.8 / 5.46],
                [0.0, 0.0], [0.5 / 5.37, -0.8 / 5.46], [0.5 / 5.37, -0.8 / 5.46]]

        self.assertEqual(got.dtype, np.dtype(np.float32))
        np.testing.assert_allclose(got, want, rtol=1e-6)
        single = residual_pair_features(groups, labels, expected_fraud, scoring[:1])
        np.testing.assert_array_equal(single[0], got[0])

    def test_clips_large_corrections_in_both_directions(self):
        groups = np.repeat([[4, 0], [7, 0]], 32, axis=0)
        labels = np.repeat([1, 0], 32)
        expected_fraud = np.repeat([0.01, 0.99], 32)

        got = residual_pair_features(groups, labels, expected_fraud, np.array([[4, 0], [7, 0]]))

        np.testing.assert_allclose(got, [[1.0, 0.0], [-1.0, 0.0]], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
