import unittest

import numpy as np

from occupancy_tree import pair_occupancy


class PairOccupancyTest(unittest.TestCase):
    def test_removes_fitting_row_from_all_four_counts(self):
        left = np.array([10, 10, 10, 11, 20, 21])
        right = np.array([30, 30, 31, 31, 40, 41])
        fitting, held = pair_occupancy(left, right, 10, 10, np.arange(6), (np.array([0]),))

        np.testing.assert_allclose(fitting[0], [np.log(2), np.log(1.2)], rtol=1e-6)
        np.testing.assert_allclose(held[0], [np.log(3), np.log(1.2)], rtol=1e-6)

    def test_unseen_pair_uses_only_fitting_margins(self):
        left = np.array([10, 10, 10, 11, 20, 21, 11, 12, 11])
        right = np.array([30, 30, 31, 31, 40, 41, 30, 30, 30])
        _, held = pair_occupancy(left, right, 10, 10, np.arange(6), (np.arange(6, 9),))

        np.testing.assert_allclose(held, [[0, -np.log(1.5)], [0, 0], [0, -np.log(1.5)]], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
