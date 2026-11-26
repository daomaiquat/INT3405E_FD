"""Compare three changes to Grandmaster on one shared stratified holdout."""

import argparse
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT.parent / "Model tham khảo" / "Grandmaster" / "notebook.ipynb"
NOTEBOOK_SHA256 = "5b408b5f21688211083b3212e5c31ee86a060ac106011ae7cf3e06fa02170aad"
HOLDOUT_FRACTION = 0.15
HOLDOUT_SEED = 4242
VARIANTS = ("residual_te", "hierarchical_amount", "aligned_offset")


def file_hash(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def load_archive(out_dir, smoke):
    if file_hash(NOTEBOOK) != NOTEBOOK_SHA256:
        raise ValueError("The reference notebook changed since the experiment was defined.")
    os.environ["SMOKE_TEST"] = "1" if smoke else "0"
    os.environ["DATA_DIR_OVERRIDE"] = str(ROOT.parents[1] / "data")
    os.environ["OUT_DIR_OVERRIDE"] = str(out_dir)
    os.environ["MPLBACKEND"] = "Agg"
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    namespace = {"__name__": "grandmaster_archive"}
    # Load only the configuration and function definitions. The archive's
    # top-level training and export cells are never executed here.
    for index in (1, 5, 7, 9):
        source = "".join(cells[index]["source"])
        exec(compile(source, f"{NOTEBOOK}:cell_{index + 1}", "exec"), namespace)
    return namespace, "".join(cells[9]["source"])


def build_candidate_matrices(decompose, raw_baseline, candidate_logits):
    baseline, columns = decompose(raw_baseline)
    blend50 = 0.5 * (raw_baseline["02e_SpGLM"] + raw_baseline["02f_DeGLM"])
    dense_column = columns.index("02f_DeGLM")
    xgb_column = columns.index("delta_03b_OffXGB")
    candidates = {name: baseline.copy() for name in VARIANTS}
    # The retained trees still used the original Blend50. Recomputing their
    # residuals from the replacement Dense GLM would change four meta inputs.
    candidates["hierarchical_amount"][:, dense_column] = candidate_logits["hierarchical_amount"]
    for name in ("residual_te", "aligned_offset"):
        candidates[name][:, xgb_column] = candidate_logits[name] - blend50
    return candidates


def main():
    from branch_runner import make_experimental_runner
    from evaluate_ablation import evaluate_variants

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use the archive's small training configuration.")
    args = parser.parse_args()
    started = perf_counter()
    out_dir = ROOT / ("smoke" if args.smoke else "validation")
    print(f"Loading Grandmaster for {out_dir.name}", flush=True)
    nb, stage_source = load_archive(out_dir, args.smoke)
    np, pd = nb["np"], nb["pd"]
    train_path = ROOT.parents[1] / 'data' / "train.csv"
    frame = pd.read_csv(train_path)
    if len(frame) != 699_960 or frame.isna().any().any() or not frame["id"].is_unique:
        raise ValueError("Expected the complete official labeled training data with unique IDs.")
    if args.smoke:
        sample = nb["StratifiedShuffleSplit"](n_splits=1, train_size=nb["SMOKE_TR_N"], random_state=nb["GLOBAL_SEED"])
        keep, _ = next(sample.split(frame["id"], frame["is_fraud"]))
        frame = frame.iloc[np.sort(keep)].reset_index(drop=True)
    splitter = nb["StratifiedShuffleSplit"](n_splits=1, test_size=HOLDOUT_FRACTION, random_state=HOLDOUT_SEED)
    develop_rows, holdout_rows = next(splitter.split(frame["id"], frame["is_fraud"]))
    develop = frame.iloc[np.sort(develop_rows)].reset_index(drop=True)
    holdout = frame.iloc[np.sort(holdout_rows)].reset_index(drop=True)
    del frame
    y_dev = develop["is_fraud"].to_numpy(dtype=np.int8)
    y_holdout = holdout.pop("is_fraud").to_numpy(dtype=np.int8)
    holdout_ids = holdout["id"].to_numpy()
    otp_mask = (holdout["auth_method"].eq("OTP") & holdout["velocity_score"].between(22.1, 24.0, inclusive="left")).to_numpy()
    print(f"Development: {len(develop):,}; locked holdout: {len(holdout):,}; stratified seed {HOLDOUT_SEED}", flush=True)

    identity = {
        "mode": out_dir.name,
        "train_sha256": file_hash(train_path),
        "notebook_sha256": NOTEBOOK_SHA256,
        "experiment_sha256": {name: file_hash(ROOT / name) for name in ("run_ablation.py", "branch_runner.py", "improvement_features.py")},
        "develop_rows": len(develop),
        "develop_frauds": int(y_dev.sum()),
        "holdout_rows": len(holdout),
        "holdout_frauds": int(y_holdout.sum()),
        "holdout_ids_sha256": sha256(holdout_ids.tobytes()).hexdigest(),
        "holdout_fraction": HOLDOUT_FRACTION,
        "holdout_seed": HOLDOUT_SEED,
        "cv_folds": nb["N_FOLDS"],
        "cv_seed": nb["GLOBAL_SEED"],
        "packages": nb["ENV_INFO"],
        "interpret_core": nb["importlib"].metadata.version("interpret-core"),
        "xgb_device": nb["XGB_DEVICE"],
        "catboost_task_type": nb["CAT_TASK_TYPE"],
    }
    signature = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    model_names = nb["LOCKED_MODELS"]
    encoder = nb["FastFoldInductiveEncoder"]
    enc_outer = encoder(te_smooths=(10.0, 50.0), te_seeds=(101, 202, 303))
    enc_inner = encoder(te_smooths=(10.0, 50.0), te_seeds=(101,))
    enc_tree = encoder(te_smooths=(20.0,), te_seeds=(101, 202, 303), include_pair_interactions=True)
    run_fold = make_experimental_runner(stage_source, nb)

    feature_views = nb["build_digit_trie_views"]
    num_dev, key_dev = feature_views(develop, selective_digits=True)
    num_hold, key_hold = feature_views(holdout, selective_digits=True)
    keep_num = [name for name in num_dev if not name.startswith("log1p_")]
    cat_cols = nb["CAT_COLS"]
    ebm_dev = pd.concat([num_dev[keep_num], develop[cat_cols].astype(str)], axis=1)
    ebm_hold = pd.concat([num_hold[keep_num], holdout[cat_cols].astype(str)], axis=1)
    del develop, holdout

    cv = nb["StratifiedKFold"](n_splits=nb["N_FOLDS"], shuffle=True, random_state=nb["GLOBAL_SEED"])
    folds = [(f, tr, va) for f, (tr, va) in enumerate(cv.split(np.zeros(len(y_dev)), y_dev))]
    oof = np.full((len(y_dev), len(model_names)), np.nan)
    candidate_oof = np.full((len(y_dev), len(VARIANTS)), np.nan)
    baseline_holdout = np.zeros((len(y_holdout), len(model_names)))
    candidate_holdout = np.zeros((len(y_holdout), len(VARIANTS)))

    for fold, tr_idx, va_idx in folds:
        path = nb["ART_DIR"] / f"fold_{fold + 1:02d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if (saved["signature"].item() != signature or not np.array_equal(saved["va_idx"], va_idx)):
                    raise ValueError(f"Checkpoint does not match current inputs: {path}")
                base_va, base_hold = saved["baseline_va"], saved["baseline_hold"]
                change_va, change_hold = saved["candidate_va"], saved["candidate_hold"]
            print(f"Fold {fold + 1}/{len(folds)} restored", flush=True)
        else:
            t_fold = perf_counter()
            print(f"Fold {fold + 1}/{len(folds)} fitting {len(tr_idx):,} rows", flush=True)
            result = run_fold(
                f=fold, tr_idx=tr_idx, va_idx=va_idx, num_tr=num_dev, key_tr=key_dev, y_all=y_dev,
                num_te=num_hold, key_te=key_hold, enc_outer_glm=enc_outer, enc_inner_glm=enc_inner,
                enc_tree=enc_tree, inner_offset_folds=nb["INNER_OFFSET_FOLDS"], seeds=nb["SEEDS"],
                lgb_n=nb["LGB_N_EST"], xgb_n=nb["XGB_N_EST"], cat_n=nb["CAT_N_EST"],
                stxgb_n=nb["STXGB_N_EST"], xgb_dev=nb["XGB_DEVICE"], cat_task=nb["CAT_TASK_TYPE"],
            )
            print(f"Fold {fold + 1}: GLMs and encoders {result['dt1']:.1f}s, trees {result['dt2']:.1f}s, fitting EBM", flush=True)
            ebm_va, ebm_test, _ = nb["run_fold_ebm"](
                fold, tr_idx, va_idx, ebm_dev, y_dev, ebm_hold, nb["EBM_MAX_BINS"], nb["EBM_INTERACTIONS"]
            )
            result["va"]["04_EBM"], result["te"]["04_EBM"] = ebm_va, ebm_test
            base_va = np.column_stack([result["va"][name] for name in model_names])
            base_hold = np.column_stack([result["te"][name] for name in model_names])
            change_va = np.column_stack([result["variants"][name]["va"] for name in VARIANTS])
            change_hold = np.column_stack([result["variants"][name]["te"] for name in VARIANTS])
            temp = path.with_suffix(".tmp.npz")
            if not all(np.isfinite(x).all() for x in (base_va, base_hold, change_va, change_hold)):
                raise ValueError(f"Non-finite logit at fold {fold + 1}")
            np.savez(temp, signature=signature, va_idx=va_idx, baseline_va=base_va, baseline_hold=base_hold,
                     candidate_va=change_va, candidate_hold=change_hold)
            temp.replace(path)
            print(f"Fold {fold + 1} checkpoint saved after {perf_counter() - t_fold:.1f}s", flush=True)
        oof[va_idx] = base_va
        candidate_oof[va_idx] = change_va
        baseline_holdout += base_hold / len(folds)
        candidate_holdout += change_hold / len(folds)

    if not all(np.isfinite(x).all() for x in (oof, candidate_oof, baseline_holdout, candidate_holdout)):
        raise ValueError("Incomplete OOF coverage or invalid holdout predictions.")
    base_oof_dict = dict(zip(model_names, oof.T))
    base_holdout_dict = dict(zip(model_names, baseline_holdout.T))
    candidate_dev = build_candidate_matrices(nb["build_resdecomp"], base_oof_dict, dict(zip(VARIANTS, candidate_oof.T)))
    candidate_hold = build_candidate_matrices(nb["build_resdecomp"], base_holdout_dict, dict(zip(VARIANTS, candidate_holdout.T)))
    del num_dev, key_dev, num_hold, key_hold, ebm_dev, ebm_hold
    nb["gc"].collect()
    print("Fitting the meta-models on development data and evaluating the shared holdout", flush=True)
    metrics, predictions = evaluate_variants(
        nb, base_oof_dict, base_holdout_dict, candidate_dev, candidate_hold, y_dev, folds, y_holdout, otp_mask
    )
    if not all(p.shape == y_holdout.shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() for p in predictions.values()):
        raise ValueError("Holdout probability validation failed.")
    prediction_path = out_dir / "holdout_predictions.csv"
    prediction_temp = prediction_path.with_suffix(".tmp.csv")
    pd.DataFrame({"id": holdout_ids, "is_fraud": y_holdout, **predictions}).to_csv(prediction_temp, index=False)
    prediction_temp.replace(prediction_path)
    report = {**identity, **metrics, "elapsed_seconds": perf_counter() - started}
    report_path = out_dir / "metrics.json"
    report_temp = report_path.with_suffix(".tmp.json")
    report_temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    report_temp.replace(report_path)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Verified predictions and metrics written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
