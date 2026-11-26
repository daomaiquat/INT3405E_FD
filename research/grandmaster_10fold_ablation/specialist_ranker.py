"""Learn a signed correction to frozen logits for one transaction cohort."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


_BALANCED_MATCHED = 6
_BALANCED_UNIFORM = 2
_BALANCED_PAIRS = _BALANCED_MATCHED + _BALANCED_UNIFORM


def _nearby_windows(logits, positive_rows, negative_rows):
    sorted_rows = negative_rows[np.argsort(logits[negative_rows], kind="stable")]
    sorted_scores = logits[sorted_rows]
    positive_scores = logits[positive_rows]
    first = np.searchsorted(sorted_scores, positive_scores - 0.25, side="left")
    stop = np.searchsorted(sorted_scores, positive_scores + 0.25, side="right")

    insertion = np.searchsorted(sorted_scores, positive_scores)
    left = np.maximum(insertion - 1, 0)
    right = np.minimum(insertion, len(sorted_rows) - 1)
    nearest = np.where(
        positive_scores - sorted_scores[left] <= sorted_scores[right] - positive_scores,
        left, right,
    )
    empty = first == stop
    first[empty] = nearest[empty]
    stop[empty] = nearest[empty] + 1
    return sorted_rows, first, stop


def _sample_negative_rows(logits, positive_rows, negative_rows, random, pairing):
    if pairing == "uniform":
        return random.choice(negative_rows, size=(len(positive_rows), 12))

    sorted_rows, first, stop = _nearby_windows(logits, positive_rows, negative_rows)
    matched = sorted_rows[random.integers(first[:, None], stop[:, None], size=(len(positive_rows), 8))]
    uniform = random.choice(negative_rows, size=(len(positive_rows), 4))
    return np.column_stack((matched, uniform))


def _balanced_negative_rows(logits, positive_rows, negative_rows, random):
    sorted_rows, first, stop = _nearby_windows(logits, positive_rows, negative_rows)
    first, stop = first[:, None], stop[:, None]
    matched = random.integers(first, stop, size=(len(positive_rows), _BALANCED_MATCHED))
    uniform = random.integers(len(sorted_rows), size=(len(positive_rows), _BALANCED_UNIFORM))
    positions = np.column_stack((matched, uniform))
    in_window = (first <= positions) & (positions < stop)
    target = 1.0 / len(sorted_rows)
    proposal = (
        (_BALANCED_MATCHED / _BALANCED_PAIRS) * in_window / (stop - first)
        + (_BALANCED_UNIFORM / _BALANCED_PAIRS) * target
    )
    return sorted_rows[positions], target / proposal


def _balanced_pair_design(features, inside, labels, logits, random):
    inside_rows = np.flatnonzero(inside)
    inside_positive = np.flatnonzero(inside & (labels == 1))
    inside_negative = np.flatnonzero(inside & (labels == 0))
    outside_positive = np.flatnonzero(~inside & (labels == 1))
    outside_negative = np.flatnonzero(~inside & (labels == 0))
    blocks = (
        (inside_positive, inside_negative),
        (inside_positive, outside_negative),
        (outside_positive, inside_negative),
    )
    active = [(positive, negative) for positive, negative in blocks if positive.size and negative.size]
    if not active:
        raise ValueError("Balanced ranking requires a pair affected by the cohort correction.")

    rows = sum(len(positive) * _BALANCED_PAIRS for positive, _ in active)
    differences = np.empty((rows, features.shape[1]), dtype=np.float64)
    offsets = np.empty(rows, dtype=np.float64)
    mass = np.empty(rows, dtype=np.float64)
    start = 0
    for positive, negative in active:
        sampled, importance = _balanced_negative_rows(logits, positive, negative, random)
        stop = start + sampled.size
        design = differences[start:stop].reshape((*sampled.shape, features.shape[1]))
        design.fill(0)
        if inside[positive[0]]:
            design[:] = features[np.searchsorted(inside_rows, positive), None, :]
        if inside[negative[0]]:
            design -= features[np.searchsorted(inside_rows, sampled)]
        offsets[start:stop] = (logits[positive, None] - logits[sampled]).ravel()
        # Population pair count divided by the number of sampled pairs in
        # this stratum, with the proposal correction applied per pair.
        mass[start:stop] = importance.ravel() * (len(positive) * len(negative) / sampled.size)
        start = stop
    mass /= mass.sum()
    return differences, offsets, mass


def _pair_design(features, inside, inside_rows, positive, normal, labels, logits, random, pairing):
    positive_rows = inside_rows[positive]
    inside_normals = inside_rows[normal]
    outside_normals = np.flatnonzero(~inside & (labels == 0))
    if not inside_normals.size:
        inside_normals = outside_normals
    if not outside_normals.size:
        outside_normals = inside_normals
    if not inside_normals.size:
        raise ValueError("Ranking pairs require at least one normal transaction.")

    paired_rows = np.column_stack((
        _sample_negative_rows(logits, positive_rows, inside_normals, random, pairing),
        _sample_negative_rows(logits, positive_rows, outside_normals, random, pairing),
    ))
    differences = np.empty((*paired_rows.shape, features.shape[1]), dtype=np.float64)
    differences[:] = features[positive, None, :]
    paired_inside = inside[paired_rows]
    # Outside rows retain only their original logit and have zero correction.
    differences[paired_inside] -= features[np.searchsorted(inside_rows, paired_rows[paired_inside])]
    offsets = logits[positive_rows, None] - logits[paired_rows]
    return differences.reshape(-1, features.shape[1]), offsets.ravel()


def fit_specialist(features, inside, labels, base_logits, pairing, seed, *, rank_weight=0.005, l2=0.0005):
    """Fit on aligned rows, with each feature row corresponding to a true inside mask entry."""
    features = np.asarray(features)
    inside = np.asarray(inside)
    labels = np.asarray(labels)
    logits = np.asarray(base_logits, dtype=np.float64)
    if inside.ndim != 1 or inside.dtype.kind != "b" or labels.shape != inside.shape or logits.shape != inside.shape:
        raise ValueError("Expected a boolean cohort mask and one label and baseline logit per row.")
    if features.ndim != 2 or features.shape[1] == 0 or features.shape[0] != inside.sum():
        raise ValueError("Features must include exactly the cohort rows in original order.")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(logits).all() or not np.isfinite(features).all():
        raise ValueError("Expected binary labels and finite baseline logits and features.")
    if pairing not in ("pointwise", "uniform", "matched", "balanced"):
        raise ValueError("Expected pointwise, uniform, matched, or balanced pairing.")
    if not np.isfinite(l2) or l2 < 0 or not np.isfinite(rank_weight) or rank_weight < 0:
        raise ValueError("The L2 penalty and ranking loss weight must be finite and nonnegative.")

    inside_rows = np.flatnonzero(inside)
    inside_labels = labels[inside_rows]
    positive = np.flatnonzero(inside_labels == 1)
    normal = np.flatnonzero(inside_labels == 0)
    if not positive.size:
        raise ValueError("The cohort must contain at least one fraud transaction.")

    pointwise_seed, ranking_seed = np.random.SeedSequence(seed).spawn(2)
    pointwise_random = np.random.default_rng(pointwise_seed)
    sampled_normal = pointwise_random.choice(normal, size=min(normal.size, 32 * positive.size), replace=False)
    sampled = np.concatenate((positive, sampled_normal))
    design = np.asarray(features[sampled], dtype=np.float64, order="C")
    offsets = logits[inside_rows[sampled]]
    signs = 2.0 * inside_labels[sampled] - 1.0
    mass = np.full(len(sampled), 1.0 / len(inside_rows))
    if sampled_normal.size:
        mass[positive.size:] *= normal.size / sampled_normal.size

    pair_features = pair_offsets = pair_mass = None
    if pairing != "pointwise" and rank_weight:
        ranking_random = np.random.default_rng(ranking_seed)
        if pairing == "balanced":
            pair_features, pair_offsets, pair_mass = _balanced_pair_design(
                features, inside, labels, logits, ranking_random,
            )
        else:
            pair_features, pair_offsets = _pair_design(
                features, inside, inside_rows, positive, normal, labels, logits, ranking_random, pairing,
            )

    def loss_and_gradient(weights):
        margins = -signs * (offsets + design @ weights)
        loss = mass @ np.logaddexp(0.0, margins) + 0.5 * l2 * (weights @ weights)
        gradient = design.T @ (-mass * signs * expit(margins)) + l2 * weights
        if pair_features is not None:
            pair_margins = -(pair_offsets + pair_features @ weights)
            pair_losses = np.logaddexp(0.0, pair_margins)
            pair_residuals = expit(pair_margins)
            if pair_mass is None:
                loss += rank_weight * pair_losses.mean()
                gradient -= (rank_weight / len(pair_margins)) * (pair_features.T @ pair_residuals)
            else:
                loss += rank_weight * (pair_mass @ pair_losses)
                gradient -= rank_weight * (pair_features.T @ (pair_mass * pair_residuals))
        return float(loss), gradient

    fitted = minimize(
        loss_and_gradient, np.zeros(features.shape[1]), method="L-BFGS-B", jac=True,
        options={"maxiter": 500, "gtol": 1e-8, "ftol": 1e-12, "maxls": 40},
    )
    if not np.isfinite(fitted.fun) or not np.isfinite(fitted.x).all() or not np.isfinite(fitted.jac).all():
        raise RuntimeError(f"Specialist fit returned a nonfinite solution ({fitted.message}).")
    if not fitted.success and np.max(np.abs(fitted.jac)) > 1e-6:
        raise RuntimeError(f"Specialist fit did not converge ({fitted.message}).")
    return fitted.x
