"""Compare the archived Grandmaster architecture with a full-train refit."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT.parent / "Model tham khảo" / "Grandmaster" / "notebook.ipynb"
NOTEBOOK_SHA256 = "5b408b5f21688211083b3212e5c31ee86a060ac106011ae7cf3e06fa02170aad"


def file_hash(path):
    with path.open("rb") as file:
        digest = sha256()
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_notebook(data_dir, out_dir, smoke):
    if file_hash(NOTEBOOK) != NOTEBOOK_SHA256:
        raise ValueError("Grandmaster notebook differs from the archived source.")
    os.environ["DATA_DIR_OVERRIDE"] = str(data_dir)
    os.environ["OUT_DIR_OVERRIDE"] = str(out_dir)
    os.environ["SMOKE_TEST"] = "1" if smoke else "0"
    os.environ["MPLBACKEND"] = "Agg"
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    namespace = {"__name__": "grandmaster_notebook"}
    # Reuse the original configuration, encoders, metrics, and fold runners.
    for index in (1, 5, 7, 9):
        source = "".join(cells[index]["source"])
        exec(compile(source, f"{NOTEBOOK}:cell_{index + 1}", "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run the archive's small configuration.")
    args = parser.parse_args()
    started = perf_counter()
    data_dir = ROOT.parents[1] / "data"
    out_dir = ROOT / ("smoke" if args.smoke else "full")
    print(f"Loading Grandmaster for {out_dir.name} run", flush=True)
    nb = load_notebook(data_dir, out_dir, args.smoke)
    np, pd = nb["np"], nb["pd"]
    train_path, test_path = (data_dir / name for name in ("train.csv", "test.csv"))
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    sample_ids = pd.read_csv(data_dir / "sample_submission.csv", usecols=["id"])["id"]
    if not test["id"].equals(sample_ids) or not test["id"].is_unique:
        raise ValueError("Test IDs are not aligned with the official submission template.")
    if args.smoke:
        sampler = nb["StratifiedShuffleSplit"](
            n_splits=1, train_size=nb["SMOKE_TR_N"], random_state=nb["GLOBAL_SEED"]
        )
        keep, _ = next(sampler.split(train, train["is_fraud"]))
        train = train.iloc[np.sort(keep)].reset_index(drop=True)
        test = test.iloc[:nb["SMOKE_TE_N"]].reset_index(drop=True)

    y = train["is_fraud"].to_numpy(dtype=np.int8)
    test_ids = test["id"].to_numpy()
    if not args.smoke and (len(y), len(test)) != (699_960, 299_984):
        raise ValueError("Expected the complete official train and test datasets.")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Unexpected missing values in the CSVs.")
    print(f"Loaded {len(y):,} training rows, {int(y.sum()):,} positive, {len(test):,} test rows", flush=True)

    identity = {
        "train_sha256": file_hash(train_path),
        "test_sha256": file_hash(test_path),
        "notebook_sha256": NOTEBOOK_SHA256,
        "run_mode": out_dir.name,
        "train_rows": len(y),
        "test_rows": len(test),
        "folds": nb["N_FOLDS"],
        "seed": nb["GLOBAL_SEED"],
        "environment": nb["ENV_INFO"],
        "interpret_core": nb["importlib"].metadata.version("interpret-core"),
        "xgb_device": nb["XGB_DEVICE"],
        "catboost_task_type": nb["CAT_TASK_TYPE"],
    }
    signature = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    models = nb["LOCKED_MODELS"]
    encoder = nb["FastFoldInductiveEncoder"]
    outer_glm = encoder(te_smooths=(10.0, 50.0), te_seeds=(101, 202, 303))
    inner_glm = encoder(te_smooths=(10.0, 50.0), te_seeds=(101,))
    tree = encoder(te_smooths=(20.0,), te_seeds=(101, 202, 303), include_pair_interactions=True)

    print("Building feature views", flush=True)
    features = nb["build_digit_trie_views"]
    num_tr, key_tr = features(train, selective_digits=True)
    num_te, key_te = features(test, selective_digits=True)
    num_cols = [name for name in num_tr if not name.startswith("log1p_")]
    cat_cols = nb["CAT_COLS"]
    ebm_tr = pd.concat([num_tr[num_cols], train[cat_cols].astype(str)], axis=1)
    ebm_te = pd.concat([num_te[num_cols], test[cat_cols].astype(str)], axis=1)
    del train, test

    def score_split(label, fold, fit_rows, held_rows, full_refit=False):
        path = nb["ART_DIR"] / f"{label}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if (
                    saved["signature"].item() != signature
                    or not np.array_equal(saved["fit_rows"], fit_rows)
                    or not np.array_equal(saved["held_rows"], held_rows)
                ):
                    raise ValueError(f"Checkpoint does not match this run: {path}")
                held, tests = saved["held"], saved["test"]
            print(f"{label}: restored checkpoint", flush=True)
        else:
            print(f"{label}: fitting {len(fit_rows):,} labeled rows", flush=True)
            t0 = perf_counter()
            if full_refit:
                # The original prediction helpers require a nonempty holdout.
                # Supply one unlabeled test row outside all training indices.
                num = pd.concat([num_tr, num_te.iloc[:1]], ignore_index=True)
                keys = pd.concat([key_tr, key_te.iloc[:1]], ignore_index=True)
            else:
                num, keys = num_tr, key_tr
            res = nb["run_fold_stage1_stage2"](
                f=fold, tr_idx=fit_rows, va_idx=held_rows, num_tr=num, key_tr=keys,
                y_all=y, num_te=num_te, key_te=key_te, enc_outer_glm=outer_glm,
                enc_inner_glm=inner_glm, enc_tree=tree, inner_offset_folds=nb["INNER_OFFSET_FOLDS"],
                seeds=nb["SEEDS"], lgb_n=nb["LGB_N_EST"], xgb_n=nb["XGB_N_EST"],
                cat_n=nb["CAT_N_EST"], stxgb_n=nb["STXGB_N_EST"],
                xgb_dev=nb["XGB_DEVICE"], cat_task=nb["CAT_TASK_TYPE"],
            )
            del num, keys
            print(f"{label}: GLMs {res['dt1']:.1f}s, trees {res['dt2']:.1f}s, fitting EBM", flush=True)
            ebm_input = pd.concat([ebm_tr, ebm_te.iloc[:1]], ignore_index=True) if full_refit else ebm_tr
            ebm_held, ebm_test, _ = nb["run_fold_ebm"](
                fold, fit_rows, held_rows, ebm_input, y, ebm_te, nb["EBM_MAX_BINS"], nb["EBM_INTERACTIONS"]
            )
            res["va"]["04_EBM"], res["te"]["04_EBM"] = ebm_held, ebm_test
            held = np.column_stack([res["va"][name] for name in models])
            tests = np.column_stack([res["te"][name] for name in models])
            if not np.isfinite(held).all() or not np.isfinite(tests).all():
                raise ValueError(f"Non-finite predictions in {label}.")
            if full_refit:
                np.testing.assert_allclose(held[0], tests[0], rtol=1e-5, atol=1e-5)
            temp = path.with_suffix(".tmp.npz")
            np.savez(temp, signature=signature, fit_rows=fit_rows, held_rows=held_rows, held=held, test=tests)
            temp.replace(path)
            print(f"{label}: checkpoint saved after {perf_counter() - t0:.1f}s", flush=True)
        if held.shape != (len(held_rows), len(models)) or tests.shape != (len(test_ids), len(models)):
            raise ValueError(f"Unexpected checkpoint shape: {path}")
        return held, tests

    all_rows = np.arange(len(y), dtype=np.int64)
    _, full_test = score_split("full_base", 0, all_rows, np.array([len(y)]), full_refit=True)
    print(f"All seven base models refit on {len(y):,} labeled rows", flush=True)
    splitter = nb["StratifiedKFold"](n_splits=nb["N_FOLDS"], shuffle=True, random_state=nb["GLOBAL_SEED"])
    oof = np.full((len(y), len(models)), np.nan)
    fold_test = np.zeros_like(full_test)
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    folds = []
    for fold, (fit_rows, held_rows) in enumerate(splitter.split(all_rows, y)):
        held, tests = score_split(f"fold_{fold + 1:02}", fold, fit_rows, held_rows)
        oof[held_rows] = held
        fold_ids[held_rows] = fold
        fold_test += tests / nb["N_FOLDS"]
        folds.append((fold, fit_rows, held_rows))

    if not np.isfinite(oof).all() or (fold_ids < 0).any():
        raise ValueError("OOF predictions do not cover every training row.")
    print("Fitting and comparing the meta-models", flush=True)
    def as_dict(scores):
        return {name: scores[:, i] for i, name in enumerate(models)}

    oof_dict, fold_dict = as_dict(oof), as_dict(fold_test)
    transform = nb["build_resdecomp"]
    meta_oof, meta_cols = transform(oof_dict)
    meta_fold, _ = transform(fold_dict)
    meta_full, _ = transform(as_dict(full_test))
    stack = nb["run_stage4_stacking"](oof_dict, fold_dict, y, folds, meta_inner_folds=nb["META_INNER_FOLDS"])
    link = nb["from_logit_10"]
    intercept, weights = stack["bs"].mean(), stack["ws"].mean(axis=0)
    baseline = stack["final_te"]
    np.testing.assert_allclose(link(intercept + meta_fold @ weights), baseline, rtol=1e-12, atol=1e-12)
    fixed_stack = link(intercept + meta_full @ weights)

    # Fit the final meta-model on clean OOF input rows, not in-sample base predictions.
    l2_counts = Counter(stack["l2s"])
    l2 = max(l2_counts, key=lambda value: (l2_counts[value], value))
    full_intercept, full_weights = nb["_fit_nonneg_l2_weights"](meta_oof, y.astype(np.float64), l2)
    full_stack = link(full_intercept + meta_full @ full_weights)

    def compare(scores):
        delta = scores - baseline
        k = max(1, int(np.ceil(0.01 * len(scores))))
        risk_rows = np.argpartition(scores, -k)[-k:]
        baseline_rows = np.argpartition(baseline, -k)[-k:]
        return {
            "mean_probability": float(scores.mean()),
            "median_probability": float(np.median(scores)),
            "spearman_vs_fold_ensemble": float(nb["stats"].spearmanr(scores, baseline).statistic),
            "mean_absolute_probability_change": float(np.mean(np.abs(delta))),
            "top_1_percent_overlap_fraction": len(np.intersect1d(risk_rows, baseline_rows)) / k,
        }

    for name, scores in [("submission_full_refit.csv", full_stack), ("submission_refit_fixed_stack.csv", fixed_stack)]:
        if scores.shape != test_ids.shape or not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
            raise ValueError(f"Invalid submission predictions: {name}")
        path = out_dir / name
        temp = path.with_suffix(".tmp.csv")
        pd.DataFrame({"id": test_ids, "is_fraud": scores}).to_csv(temp, index=False)
        check = pd.read_csv(temp)
        np.testing.assert_array_equal(check["id"].to_numpy(), test_ids)
        np.testing.assert_allclose(check["is_fraud"].to_numpy(), scores, rtol=1e-12, atol=1e-14)
        temp.replace(path)

    metrics = {
        **identity,
        "full_base_train_rows": len(all_rows),
        "full_meta_oof_rows": len(meta_oof),
        "baseline_oof": nb["evaluate_oof"](y, stack["final_oof"], fold_ids),
        "baseline_test_mean_probability": float(baseline.mean()),
        "refit_base_only": compare(fixed_stack),
        "refit_base_and_meta": compare(full_stack),
        "full_meta_l2": l2,
        "full_meta_intercept": full_intercept,
        "full_meta_weights": dict(zip(meta_cols, full_weights.tolist())),
        "elapsed_seconds": perf_counter() - started,
    }
    metrics_temp = out_dir / "metrics.tmp.json"
    metrics_temp.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    metrics_temp.replace(out_dir / "metrics.json")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Verified output files in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
