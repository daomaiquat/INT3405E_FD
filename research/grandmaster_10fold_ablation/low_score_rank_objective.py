"""Pointwise and sampled pairwise losses for gated logit corrections."""

import numpy as np
from scipy.special import expit, logit


_GATE_CENTER = logit(0.03)
_GATE_WIDTH = 0.7
_LOW_FRAUD_LOGIT = logit(0.02)
_NEAR_LOGIT_RADIUS = 1.5
_NEAR_PAIRS = 64
_UNIFORM_PAIRS = 32


def low_score_gate(base_logits):
    """Return the rowwise multiplier for corrections to frozen logits."""
    return expit((_GATE_CENTER - np.asarray(base_logits, dtype=np.float64)) / _GATE_WIDTH)


def _sample_pairs(base_logits, positive_rows, negative_rows, seed):
    negative_rows = negative_rows[np.argsort(base_logits[negative_rows], kind="stable")]
    negative_logits = base_logits[negative_rows]
    positive_logits = base_logits[positive_rows]
    lower = np.searchsorted(negative_logits, positive_logits - _NEAR_LOGIT_RADIUS, side="left")
    upper = np.searchsorted(negative_logits, positive_logits + _NEAR_LOGIT_RADIUS, side="right")
    empty_windows = upper == lower
    lower[empty_windows] = 0
    upper[empty_windows] = negative_rows.size

    random = np.random.default_rng(seed)
    near = random.integers(lower[:, None], upper[:, None], size=(positive_rows.size, _NEAR_PAIRS))
    uniform = random.integers(negative_rows.size, size=(positive_rows.size, _UNIFORM_PAIRS))
    sampled_positions = np.concatenate((near, uniform), axis=1)

    # Balance the proposal against a uniform negative for either draw source.
    in_window = (sampled_positions >= lower[:, None]) & (sampled_positions < upper[:, None])
    near_share = _NEAR_PAIRS / (_NEAR_PAIRS + _UNIFORM_PAIRS)
    relative_proposal = (1.0 - near_share) + in_window * (
        near_share * negative_rows.size / (upper - lower)
    )[:, None]
    fraud_weight = np.where(positive_logits < _LOW_FRAUD_LOGIT, 3.0, 1.0)
    return negative_rows[sampled_positions], fraud_weight[:, None] / relative_proposal


class LowScoreObjective:
    """Fit corrections on frozen logits with fixed pairs for the whole fit."""

    def __init__(self, labels, base_logits, *, rank_weight: float, seed: int):
        self._base = np.array(base_logits, dtype=np.float64)
        labels = np.asarray(labels)
        if self._base.ndim != 1 or not self._base.size or not np.isfinite(self._base).all():
            raise ValueError("Expected a nonempty vector of finite base logits.")
        if labels.shape != self._base.shape or not np.isin(labels, [0, 1]).all():
            raise ValueError("Expected one binary label per base logit.")
        if not np.isfinite(rank_weight) or rank_weight < 0:
            raise ValueError("Expected a finite nonnegative ranking weight.")

        self._signs = np.where(labels == 1, -1.0, 1.0)
        self._gate = low_score_gate(self._base)
        self._gate_squared = self._gate**2
        self._sampled_negatives = None
        if rank_weight:
            self._positive_rows = np.flatnonzero(labels == 1)
            negative_rows = np.flatnonzero(labels == 0)
            if self._positive_rows.size and negative_rows.size:
                self._sampled_negatives, weights = _sample_pairs(
                    self._base, self._positive_rows, negative_rows, seed,
                )
                self._pair_weights = weights * (rank_weight * self._base.size / weights.sum())

    def loss_gradient_hessian(self, raw_corrections):
        """Return mean loss and the gradient and diagonal Hessian of the summed loss."""
        return self._evaluate(raw_corrections, include_loss=True)

    def gradient_hessian(self, raw_corrections):
        """Return the gradient and diagonal Hessian of the summed loss for the booster."""
        _, gradient, hessian = self._evaluate(raw_corrections, include_loss=False)
        return gradient, hessian

    def _evaluate(self, raw_corrections, *, include_loss):
        raw = np.asarray(raw_corrections, dtype=np.float64)
        if raw.shape != self._base.shape or not np.isfinite(raw).all():
            raise ValueError("Expected one finite correction per base logit.")

        logits = self._base + self._gate * raw
        negative_margins = self._signs * logits
        residuals = expit(negative_margins)
        gradient = self._signs * residuals
        hessian = residuals * expit(-negative_margins)
        loss = np.logaddexp(0.0, negative_margins).sum() if include_loss else 0.0

        if self._sampled_negatives is not None:
            negative_pair_margins = logits[self._sampled_negatives] - logits[self._positive_rows, None]
            pair_gradient = self._pair_weights * expit(negative_pair_margins)
            pair_hessian = pair_gradient * expit(-negative_pair_margins)
            negative_rows = self._sampled_negatives.ravel()
            gradient[self._positive_rows] -= pair_gradient.sum(axis=1)
            gradient += np.bincount(negative_rows, weights=pair_gradient.ravel(), minlength=logits.size)
            hessian[self._positive_rows] += pair_hessian.sum(axis=1)
            hessian += np.bincount(negative_rows, weights=pair_hessian.ravel(), minlength=logits.size)
            if include_loss:
                loss += (self._pair_weights * np.logaddexp(0.0, negative_pair_margins)).sum()

        gradient *= self._gate
        hessian *= self._gate_squared
        return float(loss / logits.size), gradient, hessian
