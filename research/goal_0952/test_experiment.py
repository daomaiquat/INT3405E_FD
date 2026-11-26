import unittest

import numpy as np
import pandas as pd

from evaluate_and_export import assess, auc_by_base_score_band, combine
from residual_models import fit_codes, pair_table


class ExperimentTests(unittest.TestCase):
    def test_delong_gets_signed_labels(self):
        labels = np.array([0, 1, 0, 1], dtype=np.uint8)

        def delong(y, treatment, control):
            np.testing.assert_array_equal(y[np.argsort(-y)][:2], [1, 1])
            return {'diff': .25, 'ci_low': .1, 'ci_high': .4}

        measures = assess({'delong_paired_test': delong}, labels, np.array([-2, 2, -1, 1]),
                          np.array([-2, 2, 1, 0]))
        self.assertEqual(measures['delta_auc'], .25)

    def test_blending_probabilities_preserves_probability_weights(self):
        from scipy.special import expit, logit

        result = combine(logit(np.array([.1, .7])), logit(np.array([.5, .3])), .1, 'probability')
        np.testing.assert_allclose(expit(result), [.14, .66])

    def test_unseen_categories_have_a_separate_neutral_bucket(self):
        fit, query, size = fit_codes(pd.Series(['c', 'a', 'a', 'c']), pd.Series(['b', 'a']))
        table = pair_table(fit, fit, size, size, np.array([0, 1, 0, 1]), np.full(4, .5))
        np.testing.assert_array_equal(query, [0, 1])
        self.assertEqual(table[0, 1], 0)

    def test_supported_residual_interaction_retains_direction(self):
        left = np.repeat([0, 0, 1, 1], 100)
        right = np.repeat([0, 1, 0, 1], 100)
        labels = (left == right).astype(np.uint8)
        table = pair_table(left, right, 2, 2, labels, np.full(400, .5))
        self.assertGreater(table[0, 0], 0)
        self.assertLess(table[0, 1], 0)

    def test_score_band_deltas_account_for_the_total_auc_change(self):
        from sklearn.metrics import roc_auc_score

        labels = np.array([0, 1, 0, 0, 1, 1])
        reference = np.array([.03, .04, .12, .15, .15, .22])
        treatment = np.array([.05, .07, .10, .14, .16, .22])
        records = auc_by_base_score_band(labels, reference, treatment)
        self.assertEqual(sum(row['fraud'] for row in records), 3)
        self.assertAlmostEqual(sum(row['contribution_to_auc_delta'] for row in records),
                               roc_auc_score(labels, treatment) - roc_auc_score(labels, reference))


if __name__ == '__main__':
    unittest.main()
