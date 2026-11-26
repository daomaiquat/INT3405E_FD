import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit, logit

from low_score_rank_objective import LowScoreObjective, low_score_gate


class LowScoreObjectiveTests(unittest.TestCase):
    def test_gate_uses_frozen_logit_with_half_weight_at_three_percent(self):
        center = logit(0.03)
        assert_allclose(
            low_score_gate([center - 0.7, center, center + 0.7]),
            [0.7310585786300049, 0.5, 0.2689414213699951],
        )

    def test_control_matches_binary_cross_entropy_with_unclipped_logits(self):
        base = logit([0.001, 0.03, 0.90])
        labels = np.array([0, 1, 1])
        corrections = np.array([-60.0, 2.5, 1.0])
        gate = low_score_gate(base)
        corrected = base + gate * corrections
        outcome_probability = expit(corrected * (2 * labels - 1))
        objective = LowScoreObjective(labels, base, rank_weight=0.0, seed=22)
        loss, gradient, hessian = objective.loss_gradient_hessian(corrections)

        assert_allclose(loss, -np.log(outcome_probability).mean(), atol=1e-15)
        assert_allclose(gradient, (expit(corrected) - labels) * gate, atol=1e-15)
        assert_allclose(hessian, expit(corrected) * expit(-corrected) * gate**2)
        self.assertLess(gradient[0], 1e-28)

    def test_pairwise_derivatives_match_finite_differences_on_both_endpoints(self):
        base = logit([0.08, 0.016, 0.011, 0.003, 0.04])
        labels = np.array([1, 1, 0, 0, 0])
        corrections = np.array([-0.6, 0.25, 0.42, -0.3, 0.11])
        objective = LowScoreObjective(labels, base, rank_weight=0.01, seed=32)
        _, gradient, hessian = objective.loss_gradient_hessian(corrections)
        lightweight_gradient, lightweight_hessian = objective.gradient_hessian(corrections)
        assert_array_equal(lightweight_gradient, gradient)
        assert_array_equal(lightweight_hessian, hessian)
        step = 1e-5
        for row in range(len(labels)):
            perturbation = np.zeros_like(corrections)
            perturbation[row] = step
            upper_loss, upper_gradient, _ = objective.loss_gradient_hessian(corrections + perturbation)
            lower_loss, lower_gradient, _ = objective.loss_gradient_hessian(corrections - perturbation)
            self.assertAlmostEqual(gradient[row], labels.size * (upper_loss - lower_loss) / (2 * step), places=8)
            self.assertAlmostEqual(hessian[row], (upper_gradient[row] - lower_gradient[row]) / (2 * step), places=8)

    def test_pairwise_loss_weights_low_fraud_three_times_and_includes_high_fraud(self):
        base = logit([0.01, 0.20, 0.03])
        labels = np.array([1, 1, 0])
        gate = low_score_gate(base)
        # Every corrected logit is zero, so each pair contributes log(2).
        corrections = -base / gate
        control = LowScoreObjective(labels, base, rank_weight=0.0, seed=11)
        ranker = LowScoreObjective(labels, base, rank_weight=0.01, seed=11)
        control_loss, control_grad, _ = control.loss_gradient_hessian(corrections)
        rank_loss, rank_grad, _ = ranker.loss_gradient_hessian(corrections)

        assert_allclose(rank_loss - control_loss, 0.01 * np.log(2), rtol=1e-13)
        expected_rank_grad = np.array([-0.01125, -0.00375, 0.015])
        assert_allclose((rank_grad - control_grad) / gate, expected_rank_grad, atol=1e-15)

    def test_near_pairs_are_sampled_with_uniform_coverage_and_importance_correction(self):
        base = np.r_[-5.0, -5.0, np.full(1000, -8.0)]
        labels = np.r_[1, np.zeros(base.size - 1)]
        gate = low_score_gate(base)
        corrections = -base / gate
        control = LowScoreObjective(labels, base, rank_weight=0.0, seed=17)
        ranker = LowScoreObjective(labels, base, rank_weight=0.01, seed=17)
        repeated = LowScoreObjective(labels, base, rank_weight=0.01, seed=17)
        rank_gradient, _ = ranker.gradient_hessian(corrections)
        repeated_gradient, _ = repeated.gradient_hessian(corrections)
        control_gradient, _ = control.gradient_hessian(corrections)
        rank_contribution = (rank_gradient - control_gradient) / gate

        assert_array_equal(rank_gradient, repeated_gradient)
        self.assertGreater(np.count_nonzero(rank_contribution[2:]), 0)
        self.assertLessEqual(np.count_nonzero(rank_contribution[2:]), 32)
        self.assertGreater(rank_contribution[1], 0)
        assert_allclose(rank_contribution[1] / rank_contribution[1:].sum(), 1 / 1001, rtol=0.04)


if __name__ == "__main__":
    unittest.main()
