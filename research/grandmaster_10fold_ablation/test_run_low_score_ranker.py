import unittest

import numpy as np

from run_low_score_ranker import apply_correction


class ApplyCorrectionTest(unittest.TestCase):
    def test_gate_and_cap_bound_the_change_to_parent_logits(self):
        base = np.array([-5.0, -3.0, 2.0, 1.0])
        gate = np.array([1.0, 0.5, 0.0, 1.0])
        raw = np.array([2.0, -0.6, 3.0, -1.0])

        np.testing.assert_allclose(apply_correction(base, gate, raw), [-4.6, -3.3, 2.0, 0.6])

    def test_rejects_unaligned_rows(self):
        with self.assertRaises(ValueError):
            apply_correction(np.zeros(3), np.ones(2), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
