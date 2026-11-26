"""Instrument the archived fold runner for three isolated ablations."""

from copy import copy
from textwrap import dedent, indent

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from improvement_features import amount_parent_feature, residual_pair_features


def _reference_probabilities(sparse, sparse_inputs, dense, dense_inputs, to_logit):
    sparse_logit = to_logit(sparse.predict_proba(sparse_inputs)[:, 1])
    dense_logit = to_logit(dense.predict_proba(dense_inputs)[:, 1])
    return expit(0.5 * (sparse_logit + dense_logit))


def _residual_lookup(numeric, keys, fit_rows, y_fit, base_probabilities, *destinations):
    risk = numeric["merchant_risk_score"].to_numpy()
    velocity = numeric["velocity_score"].to_numpy()
    auth = keys["cat_auth_method"].to_numpy()
    quantiles = np.linspace(0, 1, 17)[1:-1]
    risk_edges = np.unique(np.quantile(risk[fit_rows], quantiles))
    velocity_edges = np.unique(np.quantile(velocity[fit_rows], quantiles))
    auth_values = np.unique(auth[fit_rows])

    def groups(risk, velocity, auth):
        risk_bin = np.digitize(risk, risk_edges)
        velocity_bin = np.digitize(velocity, velocity_edges)
        auth_index = np.searchsorted(auth_values, auth)
        match = (auth_index < len(auth_values)) & (
            auth_values[np.minimum(auth_index, len(auth_values) - 1)] == auth
        )
        auth_index = np.where(match, auth_index, len(auth_values))
        return np.column_stack((
            velocity_bin * (len(auth_values) + 1) + auth_index,
            risk_bin * (len(velocity_edges) + 1) + velocity_bin,
        ))

    reference = groups(risk[fit_rows], velocity[fit_rows], auth[fit_rows])
    scoring = []
    for num, key, rows in destinations:
        scoring_risk = num["merchant_risk_score"].to_numpy()
        scoring_velocity = num["velocity_score"].to_numpy()
        scoring_auth = key["cat_auth_method"].to_numpy()
        if rows is not None:
            scoring_risk = scoring_risk[rows]
            scoring_velocity = scoring_velocity[rows]
            scoring_auth = scoring_auth[rows]
        scoring.append(groups(scoring_risk, scoring_velocity, scoring_auth))
    encoded = residual_pair_features(
        reference, y_fit, base_probabilities, np.concatenate(scoring, axis=0)
    )
    return np.split(encoded, np.cumsum([len(group) for group in scoring[:-1]], dtype=int))


def _fit_hierarchical_dense(X_tr, X_va, X_te, key_tr, key_te, tr_idx, va_idx, y, seeds, to_logit):
    amount_key = "exact_amount_usd"
    training_keys = key_tr[amount_key].to_numpy()[tr_idx]
    validation_keys = key_tr[amount_key].to_numpy()[va_idx]
    test_keys = key_te[amount_key].to_numpy()
    feature = np.zeros(len(y), dtype=np.float32)
    for seed in seeds:
        split = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fit, held in split.split(training_keys, y):
            feature[held] += amount_parent_feature(
                training_keys[fit], y[fit], training_keys[held]
            ) / len(seeds)

    eval_feature = amount_parent_feature(
        training_keys, y, np.concatenate((validation_keys, test_keys))
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature.reshape(-1, 1))
    model = LogisticRegression(C=0.10, max_iter=600, tol=1e-5, solver="lbfgs")
    model.fit(np.column_stack((X_tr, scaled)), y)
    scaled_eval = scaler.transform(eval_feature.reshape(-1, 1))
    del scaled
    z_va = to_logit(model.predict_proba(
        np.column_stack((X_va, scaled_eval[:len(X_va)]))
    )[:, 1])
    z_te = to_logit(model.predict_proba(
        np.column_stack((X_te, scaled_eval[len(X_va):]))
    )[:, 1])
    return z_va, z_te


def _aligned_dense_logits(encoder, numeric, keys, labels, fit, held, to_logit):
    pack = encoder.fit_transform_fold(
        numeric, keys, labels, fit, held,
        spline_scale=1.5, low_card_scale=1.2, high_card_scale=0.6, compute_te=True,
    )
    # This variant refits the Dense half only. Release the unused OHE matrices.
    del pack["X_sp_tr"], pack["X_sp_va"]
    scaler = StandardScaler()
    dense_train = np.hstack((pack["Z_sp_tr"], scaler.fit_transform(pack["te_tr"])))
    dense_held = np.hstack((pack["Z_sp_va"], scaler.transform(pack["te_va"])))
    del pack
    model = LogisticRegression(C=0.10, max_iter=300, tol=1e-4, solver="lbfgs")
    model.fit(dense_train, labels[fit])
    return to_logit(model.predict_proba(dense_held)[:, 1]).astype(np.float32)


def _replace_exact(source, marker, replacement):
    if source.count(marker) != 1:
        raise ValueError(f"Archived Grandmaster fold source differs at {marker.strip()!r}.")
    return source.replace(marker, replacement, 1)


def _before(source, marker, snippet):
    whitespace = marker[:len(marker) - len(marker.lstrip())]
    return _replace_exact(source, marker, indent(dedent(snippet).strip(), whitespace) + "\n" + marker)


def make_experimental_runner(stage_source: str, namespace: dict):
    """Return the archived branch runner plus one changed output per candidate."""
    # The baseline stays the archived source. Exact anchors make any future
    # archive incompatibility explicit, and both variants share its XGB block.
    xgb_start = "    r_va_xgb, r_te_xgb = np.zeros(len(va_idx)), np.zeros(n_te)\n"
    xgb_end = "    r_va_cat, r_te_cat = np.zeros(len(va_idx)), np.zeros(n_te)\n"
    if stage_source.count(xgb_start) != 1 or stage_source.count(xgb_end) != 1:
        raise ValueError("Cannot find the original offset XGBoost fit block.")
    fit_xgb_block = xgb_start + stage_source.split(xgb_start, 1)[1].split(xgb_end, 1)[0]
    fit_xgb = (
        "def _fit_offset_xgb(X_tr, X_va, X_te, y_outer_tr, off_tr, off_va, off_te,\n"
        "                    seeds, f, xgb_n, xgb_dev, va_idx, n_te):\n"
        + fit_xgb_block + "    return r_va_xgb, r_te_xgb\n"
    )

    source = _replace_exact(
        stage_source,
        "    del pack_out, Xd_tr, Xd_va, Xd_te\n",
        indent(dedent("""
            # Each residual lookup sees labels only in the scoring split's fit set.
            reference_p = _reference_probabilities(
                clf_sp, pack_out["X_sp_tr"], clf_de, Xd_tr, to_logit_10
            )
            residual_va, residual_te = _residual_lookup(
                num_tr, key_tr, tr_idx, y_outer_tr, reference_p,
                (num_tr, key_tr, va_idx), (num_te, key_te, None),
            )
            del pack_out, reference_p
            hierarchy_va, hierarchy_te = _fit_hierarchical_dense(
                Xd_tr, Xd_va, Xd_te, key_tr, key_te, tr_idx, va_idx, y_outer_tr,
                enc_outer_glm.te_seeds, to_logit_10,
            )
            del Xd_tr, Xd_va, Xd_te
        """).strip(), "    ") + "\n",
    )
    source = _before(
        source,
        "    in_skf = StratifiedKFold(n_splits=inner_offset_folds, shuffle=True, random_state=500 + f)\n",
        """
        residual_tr = np.zeros((len(tr_idx), 2), dtype=np.float32)
        aligned_dense = np.zeros(len(tr_idx), dtype=np.float32)
        aligned_encoder = _copy_encoder(enc_inner_glm)
        aligned_encoder.te_seeds = list(enc_outer_glm.te_seeds)
        """,
    )
    source = _replace_exact(
        source,
        "        del in_pk, Xi_tr, Xi_va\n",
        indent(dedent("""
            # These two models were trained wholly inside abs_itr. Their scores
            # can depend on labels in abs_itr but never on labels for abs_iva.
            reference_p = _reference_probabilities(
                in_sp, in_pk["X_sp_tr"], in_de, Xi_tr, to_logit_10
            )
            residual_tr[i_va_rel], = _residual_lookup(
                num_tr, key_tr, abs_itr, y_all[abs_itr], reference_p,
                (num_tr, key_tr, abs_iva),
            )
            del in_pk, Xi_tr, Xi_va, reference_p
            aligned_dense[i_va_rel] = _aligned_dense_logits(
                aligned_encoder, num_tr, key_tr, y_all, abs_itr, abs_iva, to_logit_10
            )
        """).strip(), "        ") + "\n",
    )
    source = _before(source, "    dt2 = time.perf_counter() - t2\n", """
        aligned_off_tr = (0.5 * (sp_tr_nest + aligned_dense)).astype(np.float32)
        aligned_va, aligned_te = _fit_offset_xgb(
            X_tr, X_va, X_te, y_outer_tr, aligned_off_tr, off_va, off_te,
            seeds, f, xgb_n, xgb_dev, va_idx, n_te,
        )
        # All baseline trees are done. Rebind matrices one at a time so the
        # original and augmented inputs are not held in memory together.
        X_tr = np.column_stack((X_tr, residual_tr))
        X_va = np.column_stack((X_va, residual_va))
        X_te = np.column_stack((X_te, residual_te))
        residual_xgb_va, residual_xgb_te = _fit_offset_xgb(
            X_tr, X_va, X_te, y_outer_tr, off_tr, off_va, off_te,
            seeds, f, xgb_n, xgb_dev, va_idx, n_te,
        )
    """)
    return_start = '    return {\n        "va": {\n'
    source = _replace_exact(
        source, return_start,
        '    return {\n'
        '        "variants": {\n'
        '            "residual_te": {"va": residual_xgb_va, "te": residual_xgb_te},\n'
        '            "hierarchical_amount": {"va": hierarchy_va, "te": hierarchy_te},\n'
        '            "aligned_offset": {"va": aligned_va, "te": aligned_te},\n'
        '        },\n'
        '        "va": {\n',
    )
    scope = {
        **namespace,
        "_copy_encoder": copy,
        "_reference_probabilities": _reference_probabilities,
        "_residual_lookup": _residual_lookup,
        "_fit_hierarchical_dense": _fit_hierarchical_dense,
        "_aligned_dense_logits": _aligned_dense_logits,
    }
    exec(compile(fit_xgb + "\n\n" + source, "<Grandmaster ablation folds>", "exec"), scope)
    return scope["run_fold_stage1_stage2"]
