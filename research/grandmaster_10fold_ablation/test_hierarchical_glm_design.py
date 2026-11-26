import unittest

import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix_csr

from hierarchical_glm_design import HierarchicalGLMDesign


def raw_frame(n=36):
    positions = np.arange(n)
    return pd.DataFrame({
        "amount_usd": 20.0 + positions,
        "hours_since_last_txn": 4.5,
        "txn_count_last_24h": 2,
        "distance_from_home_km": 3.0,
        "card_age_months": 10,
        "customer_age": 32,
        "account_balance_usd": 1000.0,
        "cvv_retry_count": 0,
        "velocity_score": positions.astype(float),
        "time_of_day_hour": 16,
        "day_of_week": 1,
        "merchant_risk_score": 50.0 + positions,
        "prior_disputes": 0,
        "merchant_category": np.where(positions % 4 == 0, "travel", "retail"),
        "card_type": "credit",
        "auth_method": np.where(positions % 2 == 0, "OTP", "PIN"),
        "channel": "mobile",
        "device_type": "phone",
        "is_foreign_transaction": False,
        "is_new_merchant": False,
        "used_vpn": False,
        "ip_country_mismatch": False,
        "billing_shipping_mismatch": False,
        "is_ai_generated_scam_attempt": False,
    })


class HierarchicalGLMDesignTests(unittest.TestCase):
    def test_unseen_amounts_keep_only_supported_ancestors(self):
        train = raw_frame(24)
        train["amount_usd"] = [18.00] * 12 + [20.01] * 12
        score = pd.concat([train.iloc[:1]] * 5, ignore_index=True)
        score["amount_usd"] = [20.01, 20.02, 20.11, 21.02, 30.01]
        matrix = HierarchicalGLMDesign().fit(train).transform(score)

        self.assertTrue(isspmatrix_csr(matrix))
        self.assertEqual(matrix.dtype, np.float32)
        scales = []
        for row in range(4):
            difference = matrix[row] - matrix[row + 1]
            self.assertEqual(difference.nnz, 1)
            scales.append(difference.data[0])
        self.assertTrue(all(0 < first < second for first, second in zip(scales, scales[1:])))

    def test_rare_amount_and_repeated_scoring_amount_use_fit_vocabulary(self):
        train = raw_frame(24)
        train["amount_usd"] = [18.00] * 12 + [20.01] + [20.02] * 11
        score = pd.concat([train.iloc[:1]] * 30, ignore_index=True)
        score["amount_usd"] = [20.03] + [20.04] * 28 + [20.05]
        fitted = HierarchicalGLMDesign().fit(train)

        matrix = fitted.transform(score)
        self.assertEqual((matrix[0] - matrix[1]).nnz, 0)
        self.assertEqual((matrix[1] - fitted.transform(score.iloc[1:2])).nnz, 0)

    def test_interactions_append_the_same_main_effect_matrix(self):
        train = raw_frame()
        fitted = HierarchicalGLMDesign().fit(train)
        additive = fitted.transform(train)
        complete = fitted.transform(train, interactions=True)

        self.assertEqual((complete[:, :additive.shape[1]] - additive).nnz, 0)
        extra = complete[:, additive.shape[1]:]
        self.assertGreater(extra.nnz, 0)
        self.assertLessEqual(extra.getnnz(axis=1).max(), 12)

    def test_category_interaction_contrast_has_zero_fit_weighted_mean(self):
        train = raw_frame()
        fitted = HierarchicalGLMDesign().fit(train)
        score = pd.concat([train.iloc[:1]] * 2, ignore_index=True)
        score["amount_usd"] = 45.0
        score["velocity_score"] = 17.5
        score["merchant_risk_score"] = 67.5
        score["merchant_category"] = ["retail", "travel"]
        main_width = fitted.transform(score).shape[1]
        interaction = fitted.transform(score, interactions=True)[:, main_width:]

        self.assertGreater(interaction.nnz, 0)
        np.testing.assert_allclose(
            (3 * interaction[0] + interaction[1]).toarray(), 0, atol=1e-7,
        )

    def test_id_and_label_have_no_effect_on_fit_or_transform(self):
        train = raw_frame()
        extended = train.assign(id="not_a_feature", is_fraud="not_a_feature")
        first = HierarchicalGLMDesign().fit(train).transform(extended, interactions=True)
        second = HierarchicalGLMDesign().fit(extended).transform(train, interactions=True)

        self.assertEqual(first.shape, second.shape)
        self.assertEqual((first - second).nnz, 0)


if __name__ == "__main__":
    unittest.main()
