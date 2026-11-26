import unittest

import numpy as np
import pandas as pd

from class_conditional_density import ClassConditionalDensity


def frame(hours, risk=35.0, auth="OTP"):
    return pd.DataFrame({
        "hours_since_last_txn": hours,
        "velocity_score": 14.7,
        "merchant_risk_score": risk,
        "amount_usd": 23.71,
        "auth_method": auth,
    })


def score(model, features):
    return pd.DataFrame(model.transform(features), columns=model.feature_names)


class ClassConditionalDensityTests(unittest.TestCase):
    def test_equal_distributions_give_zero_likelihood_ratio_under_class_imbalance(self):
        negative = frame([2.03, 2.03, 2.04, 3.17] * 409)
        positive = frame([2.03, 2.03, 2.04, 3.17] * 12)
        labels = [0] * len(negative) + [1] * len(positive)
        model = ClassConditionalDensity().fit(pd.concat([negative, positive]), labels)

        query = frame([2.03, 2.04, 2.05, 3.17, 19.11])
        result = score(model, query).filter(like="_llr")
        np.testing.assert_allclose(result, 0, atol=1e-7)

    def test_global_coarse_ratios_track_class_likelihood_with_both_classes(self):
        negative = frame([2.03] * 20 + [3.17] * 80)
        positive = frame([2.03] * 80 + [3.17] * 20)
        model = ClassConditionalDensity().fit(
            pd.concat([negative, positive]), [0] * 100 + [1] * 100
        )

        ratios = score(model, frame([2.03, 3.17]))["hours_since_last_txn_global_coarse_llr"]
        self.assertGreater(ratios.iloc[0], 1.0)
        self.assertLess(ratios.iloc[1], -1.0)

    def test_context_can_reverse_an_exact_elapsed_spike_within_a_coarse_bin(self):
        low_risk = frame([2.03] * 10 + [2.04] * 80 + [2.03] * 80 + [2.04] * 10)
        high_risk = frame([2.03] * 80 + [2.04] * 10 + [2.03] * 10 + [2.04] * 80, risk=65)
        labels = [0] * 90 + [1] * 90
        model = ClassConditionalDensity().fit(pd.concat([low_risk, high_risk]), labels * 2)

        result = score(model, frame([2.03, 2.04, 2.03], risk=[35, 35, 65]))
        np.testing.assert_allclose(result["hours_since_last_txn_global_exact_llr"], 0, atol=1e-7)
        np.testing.assert_allclose(result["hours_since_last_txn_context_coarse_llr"], 0, atol=1e-7)
        for name in ("hours_since_last_txn_context_exact_llr", "hours_since_last_txn_local_spike_llr"):
            self.assertGreater(result.loc[0, name], 0.8)
            self.assertLess(result.loc[1, name], -0.8)
            self.assertLess(result.loc[2, name], -0.8)

    def test_unseen_and_under_supported_contexts_use_global_distributions(self):
        known = frame([2.03] * 20 + [2.04] * 80 + [2.03] * 80 + [2.04] * 20)
        sparse = frame([2.03] * 4, auth="PIN")
        labels = [0] * 100 + [1] * 100 + [0] * 4
        model = ClassConditionalDensity().fit(pd.concat([known, sparse]), labels)

        query = frame([2.03, 2.03, 2.06, 17.02, np.nan],
                      risk=[35, 35, 98, 98, 35],
                      auth=["PIN", "unseen", "OTP", "OTP", "unseen"])
        result = score(model, query)
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        for resolution in ("coarse", "exact"):
            np.testing.assert_array_equal(
                result[f"hours_since_last_txn_context_{resolution}_llr"],
                result[f"hours_since_last_txn_global_{resolution}_llr"],
            )
        np.testing.assert_array_equal(result["hours_since_last_txn_context_exact_weight"], 0)
        self.assertGreater(result.loc[0, "hours_since_last_txn_global_exact_llr"], 0.75)

    def test_transform_uses_only_fitted_labels_and_preserves_row_order(self):
        train = frame([2.03] * 80 + [3.17] * 80)
        labels = np.r_[np.zeros(80), np.ones(80)]
        model = ClassConditionalDensity().fit(train, labels)
        reverse = ClassConditionalDensity().fit(train, 1 - labels)
        query = frame([3.17, 2.03, 9.99]).set_index(pd.Index([19, -5, 2]))

        actual = model.transform(query)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (3, len(model.feature_names)))
        self.assertLessEqual(len(model.feature_names), 25)
        np.testing.assert_array_equal(actual, model.transform(query.assign(id=[7, 8, 9], is_fraud=1)))
        np.testing.assert_array_equal(actual[[2, 0]], model.transform(query.iloc[[2, 0]]))
        np.testing.assert_allclose(
            score(model, query).filter(like="_llr"),
            -score(reverse, query).filter(like="_llr"), atol=1e-7,
        )

    def test_single_class_is_neutral_and_an_empty_transform_keeps_width(self):
        model = ClassConditionalDensity().fit(frame([2.03, 3.17]), [0, 0])

        result = score(model, frame([2.03, 9.99]))
        np.testing.assert_array_equal(result, 0)
        self.assertEqual(model.transform(frame([])).shape, (0, len(model.feature_names)))

    def test_fit_rejects_labels_with_wrong_shape_or_non_binary_values(self):
        train = frame([2.03, 3.17])
        for labels in ([1], [[0], [1]], [0, 2], [0, np.nan]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                ClassConditionalDensity().fit(train, labels)


if __name__ == "__main__":
    unittest.main()
