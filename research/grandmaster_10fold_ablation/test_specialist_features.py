import unittest

import numpy as np
import pandas as pd

from specialist_features import FEATURE_COLUMNS, GROUP_COLUMNS, SpecialistFeatures, hard_group_mask


def raw_rows():
    return pd.DataFrame({
        "velocity_score": [10.0, 25.5, 80.1],
        "merchant_risk_score": [3.0, 16.5, 40.0],
        "hours_since_last_txn": [0.25, 2.0, 11.0],
        "amount_usd": [10.0, 25.0, 160.0],
        "account_balance_usd": [150.0, 625.0, 300.0],
        "txn_count_last_24h": [1, 4, 8],
        "prior_disputes": [0, 0, 2],
        "auth_method": ["otp", "pin", "otp"],
        "channel": ["online", "in-store", "in-store"],
        "merchant_category": ["shopping", "groceries", "groceries"],
        "device_type": ["mobile", "desktop", "mobile"],
    })


class SpecialistFeaturesTests(unittest.TestCase):
    def test_gate_requires_zero_cvv_retries_and_each_flag_false(self):
        frame = pd.DataFrame({name: [False] * 9 for name in GROUP_COLUMNS[:-1]}, dtype="boolean")
        frame["cvv_retry_count"] = [0, 1, 0, 0, 0, 0, 0, 0, 0]
        for row, flag in enumerate(GROUP_COLUMNS[:-1], start=2):
            frame.loc[row, flag] = True
        frame.loc[8, GROUP_COLUMNS[0]] = pd.NA

        np.testing.assert_array_equal(hard_group_mask(frame), [True] + [False] * 8)

    def test_transform_uses_fitted_statistics_with_any_batch_and_labels(self):
        train = raw_rows()
        self.assertEqual(set(train.columns), set(FEATURE_COLUMNS))
        query = train.iloc[[1]].assign(is_fraud=0)
        other = train.iloc[[2]].assign(velocity_score=10_000.0, is_fraud=1)
        encoder = SpecialistFeatures().fit(train.assign(is_fraud=[1, 0, 0]))

        single = encoder.transform(query)
        batch = encoder.transform(pd.concat([other, query.assign(is_fraud=1)]))

        self.assertEqual(single.dtype, np.float32)
        self.assertEqual(single.shape, (1, batch.shape[1]))
        self.assertEqual(single[0, 0], 1.0)
        np.testing.assert_array_equal(single[0], batch[1])
        self.assertTrue(np.isfinite(batch).all())

    def test_unseen_category_contributes_no_one_hot_value(self):
        train = raw_rows()
        encoder = SpecialistFeatures().fit(train)
        known = encoder.transform(train.iloc[[0]])
        unknown = encoder.transform(train.iloc[[0]].assign(auth_method="unseen"))

        diff = (known - unknown).ravel()
        np.testing.assert_array_equal(diff[diff != 0], [1.0])

    def test_transform_handles_constant_training_numeric_columns(self):
        train = raw_rows().iloc[[0]]
        features = SpecialistFeatures().fit(train).transform(train)

        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features.sum(), 5.0)


if __name__ == "__main__":
    unittest.main()
