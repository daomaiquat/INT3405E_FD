import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_hierarchical_glm_architecture import load_fold, save_fold


class FoldCheckpointTest(unittest.TestCase):
    def test_rejects_checkpoint_with_reordered_validation_rows(self):
        rows = np.array([3, 8])
        held = np.array([[0.3, 0.2], [0.8, 0.6]])
        holdout = np.array([[0.1, 0.4]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold.npz"
            save_fold(path, "signature", rows, held, holdout)
            actual_held, actual_holdout = load_fold(path, "signature", rows, 1)
            np.testing.assert_array_equal(actual_held, held)
            np.testing.assert_array_equal(actual_holdout, holdout)
            with self.assertRaises(ValueError):
                load_fold(path, "signature", rows[::-1], 1)

    def test_rejects_incomplete_predictions_in_checkpoint(self):
        rows = np.array([3, 8])
        held = np.array([[0.3, 0.2], [np.nan, 0.6]])
        holdout = np.array([[0.1, 0.4]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold.npz"
            save_fold(path, "signature", rows, held, holdout)
            with self.assertRaises(ValueError):
                load_fold(path, "signature", rows, 1)


if __name__ == "__main__":
    unittest.main()
