from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from export_hierarchical_submission import replace_dense_meta_input, write_submission
from test_run_ablation import MODELS, archived_decomposition


class HierarchicalSubmissionTests(unittest.TestCase):
    def test_dense_replacement_preserves_the_offsets_of_existing_trees(self):
        baseline = dict(zip(MODELS, [np.array([value], dtype=float) for value in (-4, -2, -1, -2, -3, -4, -2)]))
        actual = replace_dense_meta_input(archived_decomposition(), baseline, np.array([-1.0]))
        np.testing.assert_array_equal(actual, [[-4, -1, 2, 1, 0, -4, -2]])

    def test_export_rejects_wrong_id_order_and_writes_kaggle_schema(self):
        with TemporaryDirectory(dir=Path.cwd()) as scratch:
            path = Path(scratch) / "submission.csv"
            ids = np.array([699_960, 699_961])
            with self.assertRaisesRegex(ValueError, "ID"):
                write_submission(path, ids[::-1], np.array([0.25, 0.75]), ids)
            self.assertFalse(path.exists())
            write_submission(path, ids, np.array([0.25, 0.75]), ids)
            self.assertEqual(path.read_text(encoding="utf-8").splitlines(),
                             ["id,is_fraud", "699960,0.25", "699961,0.75"])


if __name__ == "__main__":
    unittest.main()
