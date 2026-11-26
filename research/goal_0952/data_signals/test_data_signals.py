import unittest

import numpy as np
import pandas as pd

from overlooked_features import digit_tables, pack_context, peer_features, physical


class DataSignalTests(unittest.TestCase):
    def test_derived_time_wraps_to_the_previous_day(self):
        row = dict(amount_usd=105., account_balance_usd=100., hours_since_last_txn=2.,
                   txn_count_last_24h=0, customer_age=20, card_age_months=35, velocity_score=12,
                   time_of_day_hour=0, day_of_week=0, is_foreign_transaction=False, ip_country_mismatch=True)
        out = physical(pd.DataFrame([row]))
        self.assertAlmostEqual(out['previous_week_phase'][0], 166.5 / 168)
        self.assertEqual(out['short_interval_empty_count'][0], 1.)
        self.assertEqual(out['spend_exceeds_balance'][0], 1.)

    def test_digit_interaction_detects_dependence_beyond_the_marginals(self):
        left = np.repeat([0, 0, 1, 1], 100)
        right = np.repeat([0, 1, 0, 1], 100)
        labels = (left == right).astype(np.uint8)
        response, rarity = digit_tables(left, right, labels, np.zeros(400))
        self.assertGreater(response[0, 0], response[0, 1])
        np.testing.assert_allclose(rarity[:2, :2], 0)

    def test_parent_prefix_allows_coarse_context_backoff(self):
        left, right, size = pack_context(('a', 'b', 'c'), {
            'a': (np.array([1]), np.array([1]), 2), 'b': (np.array([2]), np.array([2]), 3),
            'c': (np.array([3]), np.array([0]), 4),
        })
        self.assertEqual(size, 24)
        np.testing.assert_array_equal(left // 4, right // 4)

    def test_peer_features_for_one_row_ignore_other_scoring_rows(self):
        frame = pd.DataFrame({
            'velocity_score': np.arange(60) * .25, 'txn_count_last_24h': np.arange(60) % 8,
            'hours_since_last_txn': np.arange(60) * .25, 'auth_method': ['OTP', 'PIN'] * 30,
            'channel': ['APP', 'POS'] * 30, 'amount_usd': np.arange(60) + 1.,
            'merchant_category': ['Travel', 'Utilities'] * 30, 'account_balance_usd': np.arange(60) + 100.,
            'distance_from_home_km': np.arange(60) * .5, 'is_foreign_transaction': [True, False] * 30,
            'used_vpn': [False, False, True] * 20, 'merchant_risk_score': np.arange(60) / 2.,
            'billing_shipping_mismatch': [True, False] * 30, 'cvv_retry_count': np.arange(60) % 3,
            'device_type': ['iPhone', 'Android'] * 30, 'prior_disputes': np.arange(60) % 2,
            'card_age_months': np.arange(60) % 31 + 20, 'customer_age': np.arange(60) % 40 + 18,
            'card_type': ['Visa', 'Other'] * 30,
        })
        labels = (np.arange(50) % 9 == 0).astype(np.uint8)
        only = peer_features(frame.iloc[:50], labels, frame.iloc[50:51])
        together = peer_features(frame.iloc[:50], labels, frame.iloc[50:])
        self.assertEqual(only.shape, (1, 28))
        self.assertTrue(np.isfinite(together).all())
        np.testing.assert_array_equal(only[0], together[0])


if __name__ == '__main__':
    unittest.main()
