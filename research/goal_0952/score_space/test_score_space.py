import unittest

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from score_space import MAX_DELTA, ResidualSurface, quadratic, scales, squared_distances, transform
from soft_targets import targets


class ScoreSpaceTests(unittest.TestCase):
    def test_radial_distance_matches_rowwise_euclidean_squared(self):
        a = np.array([[1., 2.], [3., 0.]], dtype=np.float32)
        b = np.array([[1., 2.], [-1., 1.], [2., 1.]], dtype=np.float32)
        np.testing.assert_allclose(squared_distances(a, b), ((a[:, None] - b[None, :])**2).sum(axis=-1))

    def test_quadratic_excludes_anchor_square_and_constant_inputs_are_finite(self):
        values = np.array([[2., 3.], [4., 5.]], dtype=np.float32)
        np.testing.assert_allclose(quadratic(values), .25 * np.array([[6., 9.], [20., 25.]]))
        constant = np.ascontiguousarray(np.ones((20, 7)))
        np.testing.assert_equal(np.isfinite(transform(constant, scales(constant))).all(), True)

    def test_residual_surface_learns_a_disagreement_interaction_on_unseen_rows(self):
        rng = np.random.default_rng(85)
        logits = rng.normal(size=(12000, 7))
        base = .7 * logits[:, 0] - 1.5
        y = rng.random(len(base)) < expit(base + .8 * logits[:, 2] * logits[:, 3])
        with threadpool_limits(limits=2):
            smoother = ResidualSurface('quadratic').fit(logits[:9000], base[:9000], y[:9000])
            update = smoother.predict(logits[9000:], base[9000:])
        self.assertLessEqual(np.abs(update).max(), MAX_DELTA + 1e-8)
        self.assertLess(log_loss(y[9000:], expit(base[9000:] + update)), log_loss(y[9000:], expit(base[9000:])))

    def test_soft_label_uses_local_teacher_but_preserves_label_majority(self):
        logits = logit(np.array([.25, .75, .95, .05]))
        mixed = targets(np.array([0, 1, 0, 1]), logits)
        np.testing.assert_allclose(mixed, [.05, .95, .19, .81], atol=1e-7)
        with self.assertRaises(ValueError):
            targets(np.array([0, 1, 2, 0]), logits)


if __name__ == '__main__':
    unittest.main()
