import unittest

import numpy as np

from run_density_offset import transfer_density_delta


class TransferDensityDeltaTest(unittest.TestCase):
    def test_transfers_only_the_matched_tree_difference(self):
        logits = np.array([-4., 0., 2.])
        control = np.array([0.4, -0.6, 0.4])
        density = np.array([1.4, -1.6, 0.4])

        np.testing.assert_allclose(
            transfer_density_delta(logits, control, density, 0.25), [-3.75, -0.25, 2.],
        )
        np.testing.assert_array_equal(transfer_density_delta(logits, control, control, 1), logits)

    def test_refuses_mismatched_score_rows(self):
        with self.assertRaises(ValueError):
            transfer_density_delta(np.ones(3), np.ones(2), np.ones(3), 1)


if __name__ == "__main__":
    unittest.main()
