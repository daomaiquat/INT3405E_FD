import unittest

import numpy as np
import pandas as pd

from run_density_tree import cross_fit_inputs, paired_auc_delta


class MemorizingEncoder:
    def fit(self, frame, labels):
        self.label_by_key = dict(zip(frame["key"], labels, strict=True))
        return self

    def transform(self, frame):
        return np.array([[self.label_by_key.get(key, -1)] for key in frame["key"]], dtype=np.float32)


class CrossFitTest(unittest.TestCase):
    def test_training_rows_are_encoded_without_their_own_labels(self):
        raw = pd.DataFrame({"key": range(8)})
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        scoring = pd.DataFrame({"key": [4, 3, 9]})

        (fit, query), = cross_fit_inputs(raw, labels, scoring, (MemorizingEncoder,), inner_folds=2, seed=22)

        np.testing.assert_array_equal(fit.ravel(), [-1] * 8)
        np.testing.assert_array_equal(query.ravel(), [0, 1, -1])

    def test_auc_difference_counts_ties_as_half(self):
        labels = [0, 0, 1, 1]
        baseline = [0.2, 0.6, 0.4, 0.6]
        candidate = [0.2, 0.4, 0.5, 0.7]

        result = paired_auc_delta(labels, candidate, baseline)

        self.assertAlmostEqual(result["delta_roc_auc"], 0.375)
        self.assertLess(result["ci95_low"], result["delta_roc_auc"])
        self.assertGreater(result["ci95_high"], result["delta_roc_auc"])


if __name__ == "__main__":
    unittest.main()
