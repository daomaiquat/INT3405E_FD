import unittest

import numpy as np
import pandas as pd

from density_tree_inputs import DensityTreeInputs


def official_rows():
    return pd.DataFrame({
        "id": [60, 12, 44, 27],
        "amount_usd": [10.25, 10.25, 75.00, 100.50],
        "hours_since_last_txn": [0.25, 3.0, 4.0, 7.0],
        "txn_count_last_24h": [1, 7, 2, 1],
        "distance_from_home_km": [2.5, 4.4, 55.2, 5.1],
        "card_age_months": [48, 48, 61, 18],
        "customer_age": [30, 30, 42, 35],
        "account_balance_usd": [300.0, 600.00, 2000.0, 500.0],
        "cvv_retry_count": [0, 1, 2, 0],
        "velocity_score": [11.9, 23.6, 23.6, 66.5],
        "time_of_day_hour": [0, 3, 10, 20],
        "day_of_week": [0, 3, 5, 6],
        "merchant_risk_score": [27.9, 23.5, 23.5, 44.25],
        "prior_disputes": [0, 1, 0, 2],
        "merchant_category": ["Retail", "Travel", "Travel", "Retail"],
        "card_type": ["Visa", "Visa", "Mastercard", "Mastercard"],
        "auth_method": pd.Categorical(
            ["OTP", "PIN", "OTP", "PIN"], categories=["PIN", "OTP", "Unobserved"],
        ),
        "channel": ["Online", "Online", "POS", "POS"],
        "device_type": ["Phone", "Tablet", "Laptop", "Phone"],
        "is_foreign_transaction": [True, False, False, False],
        "is_new_merchant": [False, True, False, False],
        "used_vpn": [False, True, False, False],
        "ip_country_mismatch": [False, False, True, False],
        "billing_shipping_mismatch": [False, False, True, False],
        "is_ai_generated_scam_attempt": [False, False, False, True],
        "is_fraud": [0, 0, 1, 1],
    })


class DensityTreeInputsTests(unittest.TestCase):
    def setUp(self):
        self.train = official_rows()
        self.labels = np.array([1, 0, 0, 0], dtype=np.int8)

    def test_fit_uses_all_official_fields_and_ignores_id_and_frame_label(self):
        encoder = DensityTreeInputs().fit(self.train, self.labels)
        without_bookkeeping = self.train.drop(columns=["id", "is_fraud"])
        other = DensityTreeInputs().fit(without_bookkeeping, self.labels)

        self.assertEqual(encoder.feature_names[:24], tuple(without_bookkeeping.columns))
        self.assertEqual(encoder.feature_names, other.feature_names)
        np.testing.assert_array_equal(
            encoder.transform(self.train.assign(id=-100, is_fraud=1)),
            other.transform(without_bookkeeping),
        )

    def test_unknown_categories_and_amounts_back_off_to_fold_prior(self):
        encoder = DensityTreeInputs().fit(self.train, self.labels)
        score = pd.concat([self.train.iloc[[0]]] * 4, ignore_index=True)
        score["auth_method"] = pd.Categorical(
            ["OTP", "Unobserved", "New", None], categories=["New", "OTP", "Unobserved"],
        )
        score["amount_usd"] = [10.25, 10.251, 700.0, np.nan]
        result = encoder.transform(score)
        cols = {name: i for i, name in enumerate(encoder.feature_names)}

        np.testing.assert_array_equal(result[:, cols["auth_method"]], [0, -1, -1, -1])
        for name in ("auth_method", "amount_usd"):
            np.testing.assert_array_equal(result[:, cols[f"{name}_exact_count"]], [2, 0, 0, 0])
            np.testing.assert_allclose(result[:, cols[f"{name}_exact_frequency"]], [0.5, 0, 0, 0])
            np.testing.assert_allclose(result[:, cols[f"{name}_exact_target"]], [6 / 22, 0.25, 0.25, 0.25])
        self.assertTrue(np.isfinite(result).all())

    def test_scoring_batches_produce_the_same_finite_float32_rows(self):
        encoder = DensityTreeInputs().fit(self.train, self.labels)
        score = self.train.drop(columns=["id", "is_fraud"]).assign(
            account_balance_usd=[np.inf, 0.0, -1000.25, 300.0],
        )
        result = encoder.transform(score)
        cols = {name: i for i, name in enumerate(encoder.feature_names)}

        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        self.assertEqual(result.shape, (len(score), len(encoder.feature_names)))
        self.assertEqual(len(set(encoder.feature_names)), result.shape[1])
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[1], encoder.transform(score.iloc[1:2])[0])
        np.testing.assert_array_equal(result[:, cols["amount_usd_fractional_units"]], [25, 25, 0, 50])
        self.assertAlmostEqual(result[1, cols["log_amount_to_balance"]], np.log1p(10.25), places=6)


if __name__ == "__main__":
    unittest.main()
