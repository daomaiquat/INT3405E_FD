"""Train the validated Dense GLM change and export a 10-fold Kaggle submission."""

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from branch_runner import _fit_hierarchical_dense
from run_ablation import NOTEBOOK_SHA256, ROOT, file_hash, load_archive


BASELINE = ROOT.parent / "grandmaster_refit" / "full"
OUT_DIR = ROOT / "submission_hierarchical_amount"
DENSE_GLM = "02f_DeGLM"
CACHE_ID_FIELDS = (
    "train_sha256", "test_sha256", "notebook_sha256", "run_mode", "train_rows", "test_rows",
    "folds", "seed", "environment", "interpret_core", "xgb_device", "catboost_task_type",
)


def replace_dense_meta_input(decompose, original_logits, dense_logits):
    matrix, fields = decompose(original_logits)
    # Retained boosted trees were fitted with the original Blend50, so their
    # residuals must be computed before substituting the improved Dense GLM.
    matrix[:, fields.index(DENSE_GLM)] = dense_logits
    return matrix


def write_submission(path: Path, ids, probabilities, template_ids):
    ids = np.asarray(ids)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    template_ids = np.asarray(template_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or not np.array_equal(ids, template_ids):
        raise ValueError("Test IDs must match the submission template in order.")
    if probabilities.shape != ids.shape or not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("Expected one finite probability in [0, 1] per test ID.")
    temporary = path.with_suffix(".tmp.csv")
    pd.DataFrame({"id": ids, "is_fraud": probabilities}).to_csv(temporary, index=False)
    readback = pd.read_csv(temporary)
    if list(readback.columns) != ["id", "is_fraud"]:
        raise ValueError("Unexpected submission CSV columns.")
    np.testing.assert_array_equal(readback["id"], template_ids)
    np.testing.assert_allclose(readback["is_fraud"], probabilities, atol=1e-14, rtol=1e-12)
    temporary.replace(path)


def load_baseline_folds(metadata, folds, model_names, train_count, test_count):
    identity = {field: metadata[field] for field in CACHE_ID_FIELDS}
    signature = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    oof = np.full((train_count, len(model_names)), np.nan)
    test = np.zeros((test_count, len(model_names)))
    cache_hashes = {}
    for fold, fit, held in folds:
        path = BASELINE / "artifacts" / f"fold_{fold + 1:02d}.npz"
        cache_hashes[path.name] = file_hash(path)
        with np.load(path, allow_pickle=False) as saved:
            if saved["signature"].item() != signature or not np.array_equal(saved["fit_rows"], fit) or not np.array_equal(saved["held_rows"], held):
                raise ValueError(f"Cached 10-fold predictions do not match this split: {path}")
            val_predictions, test_predictions = saved["held"], saved["test"]
            if val_predictions.shape != (len(held), len(model_names)) or test_predictions.shape != test.shape:
                raise ValueError(f"Unexpected cached prediction shape: {path}")
            oof[held] = val_predictions
            test += test_predictions / len(folds)
    if not np.isfinite(oof).all() or not np.isfinite(test).all():
        raise ValueError("Cached baseline predictions have missing or nonfinite rows.")
    return dict(zip(model_names, oof.T)), dict(zip(model_names, test.T)), cache_hashes


def main():
    started = perf_counter()
    validation = json.loads((ROOT / "validation" / "metrics.json").read_text(encoding="utf-8"))
    for name, expected in validation["experiment_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"The validated implementation differs at {name}.")
    baseline_metadata = json.loads((BASELINE / "metrics.json").read_text(encoding="utf-8"))
    train_path, test_path = ROOT.parents[1] / 'data' / "train.csv", ROOT.parents[1] / 'data' / "test.csv"
    train_hash, test_hash = file_hash(train_path), file_hash(test_path)
    if (
        train_hash != validation["train_sha256"]
        or train_hash != baseline_metadata["train_sha256"]
        or test_hash != baseline_metadata["test_sha256"]
        or NOTEBOOK_SHA256 != baseline_metadata["notebook_sha256"]
    ):
        raise ValueError("Data or notebook differs from the validated experiment and local baseline.")

    print("Loading the Grandmaster 10-fold configuration", flush=True)
    nb, _ = load_archive(OUT_DIR, smoke=False)
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    template = pd.read_csv(ROOT.parents[1] / 'data' / "sample_submission.csv", usecols=["id"])["id"]
    if (len(train), len(test)) != (699_960, 299_984) or not test["id"].equals(template) or not template.is_unique:
        raise ValueError("Expected the complete official datasets and the official sequence of test IDs.")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Unexpected nulls in the official data.")
    labels = train["is_fraud"].to_numpy(dtype=np.int8)
    test_ids = test["id"].to_numpy()
    cv = nb["StratifiedKFold"](n_splits=nb["N_FOLDS"], shuffle=True, random_state=nb["GLOBAL_SEED"])
    folds = [(i, train_idx, hold_idx) for i, (train_idx, hold_idx) in enumerate(cv.split(labels, labels))]
    base_oof, base_test, cache_hashes = load_baseline_folds(
        baseline_metadata, folds, nb["LOCKED_MODELS"], len(train), len(test)
    )
    print(f"Verified {len(folds)} local 10-fold checkpoints; training the changed Dense GLM only", flush=True)

    identity = {
        "variant": "hierarchical_amount",
        "train_sha256": train_hash,
        "test_sha256": test_hash,
        "notebook_sha256": NOTEBOOK_SHA256,
        "folds": len(folds),
        "cv_seed": nb["GLOBAL_SEED"],
        "train_rows": len(train),
        "test_rows": len(test),
        "local_10fold_cache_sha256": cache_hashes,
        "implementation_sha256": {**validation["experiment_sha256"], "export_hierarchical_submission.py": file_hash(Path(__file__))},
        "environment": nb["ENV_INFO"],
    }
    signature = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    dense_oof = np.full(len(train), np.nan)
    dense_test = np.zeros(len(test))
    num_tr, key_tr = nb["build_digit_trie_views"](train, selective_digits=True)
    num_te, key_te = nb["build_digit_trie_views"](test, selective_digits=True)
    del train, test
    encoder = nb["FastFoldInductiveEncoder"](te_smooths=(10.0, 50.0), te_seeds=(101, 202, 303))

    for fold, fit, held in folds:
        path = nb["ART_DIR"] / f"dense_fold_{fold + 1:02d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if saved["signature"].item() != signature or not np.array_equal(saved["held_rows"], held):
                    raise ValueError(f"Dense checkpoint does not match this run: {path}")
                dz_held, dz_test = saved["held"], saved["test"]
            print(f"Dense GLM fold {fold + 1}/{len(folds)} restored", flush=True)
        else:
            t_fold = perf_counter()
            print(f"Dense GLM fold {fold + 1}/{len(folds)} fitting {len(fit):,} rows", flush=True)
            pack = encoder.fit_transform_fold(
                num_tr, key_tr, labels, fit, held, num_te, key_te,
                spline_scale=1.5, low_card_scale=1.2, high_card_scale=0.6, compute_te=True,
            )
            del pack["X_sp_tr"], pack["X_sp_va"], pack["X_sp_te"]
            scaler = nb["StandardScaler"]()
            xd_tr = np.hstack((pack["Z_sp_tr"], scaler.fit_transform(pack["te_tr"]))).astype(np.float32, copy=False)
            xd_va = np.hstack((pack["Z_sp_va"], scaler.transform(pack["te_va"]))).astype(np.float32, copy=False)
            xd_te = np.hstack((pack["Z_sp_te"], scaler.transform(pack["te_te"]))).astype(np.float32, copy=False)
            del pack
            dz_held, dz_test = _fit_hierarchical_dense(
                xd_tr, xd_va, xd_te, key_tr, key_te, fit, held, labels[fit], encoder.te_seeds, nb["to_logit_10"],
            )
            del xd_tr, xd_va, xd_te
            nb["gc"].collect()
            if not np.isfinite(dz_held).all() or not np.isfinite(dz_test).all():
                raise ValueError(f"Nonfinite Dense prediction at fold {fold + 1}.")
            temporary = path.with_suffix(".tmp.npz")
            np.savez(temporary, signature=signature, held_rows=held, held=dz_held, test=dz_test)
            temporary.replace(path)
            print(f"Dense GLM fold {fold + 1} saved in {perf_counter() - t_fold:.1f}s", flush=True)
        if dz_held.shape != (len(held),) or dz_test.shape != dense_test.shape:
            raise ValueError(f"Dense checkpoint has incorrect shapes: {path}")
        dense_oof[held] = dz_held
        dense_test += dz_test / len(folds)

    if not np.isfinite(dense_oof).all() or not np.isfinite(dense_test).all():
        raise ValueError("Dense GLM predictions are incomplete or nonfinite.")
    del num_tr, key_tr, num_te, key_te
    nb["gc"].collect()

    print("Selecting baseline meta regularization and fitting the improved head", flush=True)
    stack = nb["run_stage4_stacking"](base_oof, base_test, labels, folds, meta_inner_folds=nb["META_INNER_FOLDS"])
    baseline_auc = float(roc_auc_score(labels, stack["final_oof"]))
    np.testing.assert_allclose(baseline_auc, baseline_metadata["baseline_oof"]["oof_auc"], rtol=0, atol=1e-8)
    oof_matrix = replace_dense_meta_input(nb["build_resdecomp"], base_oof, dense_oof)
    test_matrix = replace_dense_meta_input(nb["build_resdecomp"], base_test, dense_test)
    test_logits = np.zeros(len(test_ids), dtype=np.float64)
    y_fit = labels.astype(np.float64)
    for (_, fit, _), l2 in zip(folds, stack["l2s"], strict=True):
        intercept, weights = nb["_fit_nonneg_l2_weights"](oof_matrix[fit], y_fit[fit], l2)
        test_logits += (intercept + test_matrix @ weights) / len(folds)
    scores = nb["from_logit_10"](test_logits)
    if np.array_equal(scores, stack["final_te"]):
        raise ValueError("The improved submission unexpectedly equals the original predictions.")

    result = OUT_DIR / "submission.csv"
    write_submission(result, test_ids, scores, template.to_numpy())
    metadata = {
        **identity,
        "baseline_meta_l2_by_fold": list(map(float, stack["l2s"])),
        "source_holdout_roc_auc": validation["variants"]["hierarchical_amount"]["roc_auc"],
        "submission_sha256": file_hash(result),
        "mean_test_probability": float(scores.mean()),
        "mean_absolute_change_vs_local_10fold": float(np.abs(scores - stack["final_te"]).mean()),
        "test_prediction_rows_verified": len(scores),
        "elapsed_seconds": perf_counter() - started,
    }
    temporary = OUT_DIR / "manifest.tmp.json"
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(OUT_DIR / "manifest.json")
    print(f"Verified submission: {result} with {len(scores):,} ordered test IDs", flush=True)


if __name__ == "__main__":
    main()
