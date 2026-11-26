import unittest

import numpy as np

from risk_gated_meta import fit_gated_stacker


class RiskGatedMetaTests(unittest.TestCase):
    def test_gating_learns_distinct_rankings_and_uses_the_same_gate_for_new_batches(self):
        anchor = np.repeat(np.linspace(-3.0, 3.0, 256), 2)
        signal = np.tile([-1.0, 1.0], 256)
        features = np.column_stack((anchor, anchor, signal, -signal))
        labels = np.where(anchor < 2.3, signal > 0, signal < 0).astype(float)
        model = fit_gated_stacker(features, labels, 0.01, (0.0, np.zeros(4)))

        check = np.array([[-2, -2, 1, -1], [-2, -2, -1, 1], [5, 5, 1, -1], [5, 5, -1, 1]])
        logits = model.predict_logit(check)
        self.assertGreater(logits[0], logits[1])
        self.assertLess(logits[2], logits[3])
        np.testing.assert_allclose(logits, [model.predict_logit(check[i:i+1])[0] for i in range(4)])
        self.assertTrue((model.low_weights >= 0).all() and (model.high_weights >= 0).all())

    def test_equal_training_quantiles_reduce_to_a_single_finite_linear_fit(self):
        features = np.column_stack((np.zeros((8, 2)), np.tile([-1., 1.], 4)))
        labels = np.tile([0, 1], 4)
        model = fit_gated_stacker(features, labels, 0.01, (0.0, np.zeros(3)))

        np.testing.assert_allclose(model.low_weights, model.high_weights, atol=1e-10)
        logits = model.predict_logit(features[:2])
        self.assertTrue(np.isfinite(logits).all())
        self.assertLess(logits[0], 0)
        self.assertGreater(logits[1], 0)


if __name__ == "__main__":
    unittest.main()
