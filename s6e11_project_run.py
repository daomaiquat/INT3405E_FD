"""Foundation, CV contract and prediction contract for project S6E11.

Default: data audit + locked folds (plan.md steps 0-2). `--stage contract --smoke`
additionally runs the step 3 infrastructure check and writes only `submission_smoke.csv`.
`--stage baseline` runs selected step-5 baselines; use one flag per model or `--models`.
`--stage complexity` runs step 6 ablations; `--stage encoding` builds step 7 TE artifacts.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _cli_threads() -> int | None:
    argv = sys.argv[1:]
    for index, token in enumerate(argv):
        if token == "--threads" and index + 1 < len(argv):
            return int(argv[index + 1])
        if token.startswith("--threads="):
            return int(token.split("=", 1)[1])
    return None


def _limit_threads(count: int | None) -> None:
    if count is not None:
        if count < 1:
            raise ValueError("--threads must be at least 1")
        for name in THREAD_VARS:
            os.environ[name] = str(count)


_threads = _cli_threads()
_limit_threads(_threads)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.artifacts import (environment, fold_needs_run, locked_requirements,  # noqa: E402
                           read_fold_status, read_json, sha256_file, submission_path,
                           write_fold_status, write_json)
from src.data import audit_data, cast_target, load_data, quality_table, validate_frames  # noqa: E402
from src.eda import run_eda  # noqa: E402
from src.metrics import auc_score, summarize_oof  # noqa: E402
from src.models import BASELINE_IDS  # noqa: E402
from src.predictions import join_oof, join_test, prediction_frame, shuffle_for_check  # noqa: E402
from src.smoke_model import fit_contract_model, load_bundle, predict_bundle, save_bundle  # noqa: E402
from src.training import run_baseline  # noqa: E402
from src.complexity import run_complexity  # noqa: E402
from src.encoding import run_encoding  # noqa: E402
from src.te_models import TE_MODELS, run_te_models  # noqa: E402
from src.validation import (assign_folds, early_stop_assignments,  # noqa: E402
                            partition_for_fold, smoke_sample, validate_partitions)

ROOT = Path(__file__).resolve().parent


def _source_signature():
    return {str(p.relative_to(ROOT)): sha256_file(p)
            for p in [Path(__file__), *sorted((ROOT / "src").glob("*.py"))]}


def _resolve_run_dir(output_parent: Path, resume, config: dict, data_dir: Path,
                      smoke: bool, models: tuple[str, ...] = (), stage=None):

    if not resume:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
        if smoke:
            run_id += "_smoke"
        return output_parent / run_id, run_id, False
    if resume == "latest":
        existing = sorted((p for p in output_parent.iterdir()
                           if p.is_dir() and (p / "run_config.json").is_file()),
                          key=lambda p: p.name)
        if not existing:
            raise ValueError(f"--resume latest: no prior run in {output_parent}")
        run_dir = existing[-1]
    else:
        run_dir = output_parent / resume
    previous = read_json(run_dir / "run_config.json")
    if stage in ("complexity", "encoding", "te_models"):
        if (previous.get("stage") != stage or previous.get("source_signature") != _source_signature()
                or read_json(run_dir / "environment.json")["packages"] != environment()["packages"]):
            step = {"complexity": "Step 6", "encoding": "Step 7", "te_models": "Phase 8"}[stage]
            raise ValueError(f"{step} resume requires the same stage, code and package versions; use a new run")
    if (previous.get("config") != config or previous.get("data_dir") != str(data_dir)
            or previous.get("smoke") != smoke):
        raise ValueError("--resume requires matching config, data_dir and smoke mode")
    if models:
        previous_models = previous.get("baseline_models")
        if previous_models and sorted(previous_models) != sorted(models):
            raise ValueError("--resume requires the same selected baseline models")
    return run_dir, run_dir.name, True


def _run_contract_stage(run_dir: Path, train: pd.DataFrame, test: pd.DataFrame,
                        sample: pd.DataFrame, folds: pd.DataFrame, early: pd.DataFrame,
                        config: dict, n_folds: int, state: dict) -> dict:
    """Step 3 smoke: fit each fold, assemble id-keyed outputs, verify the bundle."""
    target = config["target"]
    experiment_id = "contract_smoke"
    exp_dir = run_dir / experiment_id
    bundle_dir = exp_dir / "model_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    status_path = exp_dir / "fold_status.json"
    metrics_path = exp_dir / "fold_metrics.json"
    status = read_fold_status(status_path)
    persisted_metrics = read_json(metrics_path) if metrics_path.is_file() else {}
    oof_outputs, test_outputs, fold_metrics = {}, {}, {}

    for fold in range(n_folds):
        oof_path = exp_dir / f"oof_fold_{fold}.csv"
        test_path = exp_dir / f"test_fold_{fold}.csv"
        bundle_path = bundle_dir / f"fold_{fold}.joblib"
        required = [oof_path, test_path, bundle_path]
        reusable = (state.get("resumed") and not fold_needs_run(status, fold, required)
                    and "train_auc" in persisted_metrics.get(str(fold), {}))
        if reusable:
            oof_outputs[fold] = pd.read_csv(oof_path)
            test_outputs[fold] = pd.read_csv(test_path)
            fold_metrics[str(fold)] = persisted_metrics[str(fold)]
            logging.info("Reusing completed fold %s", fold)
            continue
        status[str(fold)] = "started"
        write_fold_status(status_path, status)
        try:
            part = partition_for_fold(train, folds, early, config, fold)
            bundle_a = fit_contract_model(train.iloc[part["A"]], config)
            predict_bundle(bundle_a, train.iloc[part["E"]])
            bundle = fit_contract_model(train.iloc[part["T"]], config)
            valid_prediction = predict_bundle(bundle, train.iloc[part["V"]])
            test_prediction = predict_bundle(bundle, test)
            save_bundle(bundle, bundle_path)
            reloaded_prediction = predict_bundle(load_bundle(bundle_path), train.iloc[part["V"]])
            roundtrip = float(np.max(np.abs(valid_prediction - reloaded_prediction)))
            if roundtrip > 1e-12:
                raise ValueError(f"fold {fold}: bundle roundtrip changed predictions")
            determinism = None
            if fold == 0:
                repeated = fit_contract_model(train.iloc[part["A"]], config)
                determinism = float(np.max(np.abs(
                    predict_bundle(repeated, train.iloc[part["A"]])
                    - predict_bundle(bundle_a, train.iloc[part["A"]]))))
            oof_frame = prediction_frame(train["id"].iloc[part["V"]].to_numpy(), valid_prediction, fold)
            test_frame = prediction_frame(test["id"].to_numpy(), test_prediction, fold)
            oof_frame.to_csv(oof_path, index=False)
            test_frame.to_csv(test_path, index=False)
            oof_outputs[fold] = oof_frame
            test_outputs[fold] = test_frame
            fold_metrics[str(fold)] = {
                "sizes": {name: len(values) for name, values in part.items()},
                "train_auc": auc_score(train[target].iloc[part["T"]],
                                       predict_bundle(bundle, train.iloc[part["T"]])),
                "n_trees": bundle["n_trees"],
                "convergence_warnings": bundle["warnings"],
                "bundle_roundtrip_max_abs_diff": roundtrip,
                "same_seed_max_abs_diff": determinism,
                "reused": False,
            }
            status[str(fold)] = "completed"
            write_fold_status(status_path, status)
            logging.info("Fold %s fit; bundle roundtrip diff %.2e", fold, roundtrip)
        except Exception:
            status[str(fold)] = "failed"
            write_fold_status(status_path, status)
            raise

    oof = join_oof(oof_outputs, folds, config)
    test_by_fold = join_test(test_outputs, sample, n_folds)
    shuffled_oof = join_oof({k: shuffle_for_check(v, 100 + k) for k, v in oof_outputs.items()},
                            folds, config)
    shuffled_test = join_test({k: shuffle_for_check(v, 200 + k) for k, v in test_outputs.items()},
                              sample, n_folds)
    pd.testing.assert_frame_equal(oof, shuffled_oof)
    pd.testing.assert_frame_equal(test_by_fold, shuffled_test)

    constant_auc = auc_score(oof[target], np.full(len(oof), 0.5))
    rng = np.random.default_rng(config["model_seed"])
    random_auc = auc_score(oof[target], rng.random(len(oof)))
    if abs(constant_auc - 0.5) > 1e-12:
        raise ValueError("constant prediction did not score AUC 0.50")

    oof.to_csv(exp_dir / "oof.csv", index=False)
    test_by_fold.to_csv(exp_dir / "test_by_fold.csv", index=False)
    metrics = summarize_oof(
        oof,
        train_auc=float(np.mean([m["train_auc"] for m in fold_metrics.values() if "train_auc" in m])),
        n_trees=0,
    )
    write_json(exp_dir / "metrics.json", metrics)
    write_json(exp_dir / "fold_metrics.json", fold_metrics)
    write_json(exp_dir / "contract_checks.json", {
        "oof_rows": len(oof), "test_rows": len(test_by_fold),
        "oof_ids_unique": bool(oof["id"].is_unique),
        "each_train_row_scored_by_own_fold": True,
        "shuffled_input_joins_identical": True,
        "constant_prediction_auc": constant_auc,
        "independent_random_auc": random_auc,
        "bundle_roundtrip_max_abs_diff": max(m.get("bundle_roundtrip_max_abs_diff", 0.0)
                                             for m in fold_metrics.values()),
        "same_seed_max_abs_diff_fold_0": fold_metrics["0"].get("same_seed_max_abs_diff"),
        "fold_status": status,
    })
    submission = submission_path(run_dir, True)
    pd.DataFrame({"id": test_by_fold["id"], target: test_by_fold["prediction"]}).to_csv(
        submission, index=False)
    log_path = run_dir / "experiment_log.csv"
    pd.DataFrame([{
        "experiment_id": experiment_id, "run_id": state["run_id"], "status": "completed",
        "event_time": datetime.now(timezone.utc).isoformat(), "resumed": bool(state.get("resumed")),
        "model": "contract_logreg_smoke", "oof_auc": metrics["oof_auc"],
        "mean_auc": metrics["mean_auc"], "std_auc": metrics["std_auc"],
        "note": "infrastructure check only; not a baseline result",
    }]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    logging.info("Contract OOF AUC %.6f; smoke submission %s", metrics["oof_auc"], submission.name)
    return {"experiment_id": experiment_id, "metrics": metrics, "fold_status": status,
            "submission": submission.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("setup", "audit", "folds", "eda", "contract", "baseline",
                                            "complexity", "encoding", "te_models"),
                        default="setup")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Parent directory for a fresh run")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--threads", type=int, help="Limit BLAS/OMP pools before numeric imports")
    parser.add_argument("--resume", nargs="?", const="latest", metavar="RUN_ID",
                        help="Reuse an existing run with identical config/data instead of a new run")
    parser.add_argument("--models", nargs="+", choices=BASELINE_IDS, metavar="MODEL",
                        help="Select step-5 baselines, e.g. --models B02 B05")
    parser.add_argument("--all-baselines", action="store_true",
                        help="Select every B00-B07 baseline in one run")
    parser.add_argument("--te-models", nargs="+", choices=tuple(TE_MODELS), default=None,
                        help="Phase 8 models; default: T01 GLM, T02 LGBM, T03 XGB")
    parser.add_argument("--encoding-dir", type=Path, help="Existing step-7 run or encoding directory; auto-detected by default")
    parser.add_argument("--check-cache-only", action="store_true", help="Phase 8: validate cache without fitting any models")
    for model_id in BASELINE_IDS:
        parser.add_argument(f"--{model_id.lower()}", action="append_const", const=model_id,
                            dest="model_flags", help=f"Select {model_id} baseline")
    args = parser.parse_args()
    if args.stage != "te_models" and (args.te_models or args.encoding_dir or args.check_cache_only):
        parser.error("TE model/cache flags require --stage te_models")
    te_selected = list(dict.fromkeys(args.te_models or TE_MODELS))
    selected = list(dict.fromkeys((args.models or []) + (args.model_flags or [])))
    if args.all_baselines:
        selected = list(BASELINE_IDS)
    if args.stage == "baseline" and not selected:
        parser.error("--stage baseline needs --models, --all-baselines or a per-model flag")
    if args.stage != "baseline" and selected:
        parser.error("baseline flags require --stage baseline")
    if args.stage == "contract":
        args.smoke = True
    if args.stage == "eda":
        args.smoke = False
    if _threads is not None and args.threads != _threads:
        raise ValueError("--threads value disagrees with the early thread limit")
    config = read_json(args.config)
    data_dir = (args.data_dir or ROOT / config["data_dir"]).resolve()
    output_parent = args.output_dir or (Path("/kaggle/working/s6e11_project")
                                        if Path("/kaggle/working").is_dir() else ROOT / "outputs")
    run_dir, run_id, resumed = _resolve_run_dir(output_parent.resolve(), args.resume,
                                                 config, data_dir, args.smoke, tuple(selected), args.stage)
    if resumed and args.stage == "te_models":
        previous = read_json(run_dir / "run_config.json")
        if previous.get("te_models") != te_selected or previous.get("check_cache_only") != args.check_cache_only:
            raise ValueError("Phase 8 resume requires the same models and cache-check mode")

    run_dir.mkdir(parents=True, exist_ok=resumed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(run_dir / "run.log", mode="a" if resumed else "w",
                                                      encoding="utf-8")])
    state = {"run_id": run_id, "stage": args.stage, "smoke": args.smoke, "status": "started",
             "data_dir": str(data_dir), "config": config, "threads": args.threads,
             "resumed": resumed, "baseline_models": selected if args.stage == "baseline" else []}
    state["source_signature"] = _source_signature()
    if args.stage == "te_models":
        state.update(te_models=te_selected, check_cache_only=args.check_cache_only)
    write_json(run_dir / "run_config.json", state)
    env = environment()
    write_json(run_dir / "environment.json", env)
    snapshot = run_dir / "source_snapshot"
    (snapshot / "src").mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    shutil.copy2(args.config, snapshot / "config.json")
    for source in (ROOT / "src").glob("*.py"):
        shutil.copy2(source, snapshot / "src" / source.name)
    try:
        submission_path(run_dir, args.smoke)
        train, test, sample, manifest, _ = load_data(data_dir, config)
        write_json(run_dir / "data_manifest.json", manifest)
        audit = audit_data(train, test, sample, config, manifest)
        write_json(run_dir / "data_audit.json", audit)
        table = quality_table(audit)
        table.to_csv(run_dir / "data_quality.csv", index=False)
        failed = table.loc[~table["ok"]]
        if len(failed):
            raise ValueError(f"data quality checks failed: {failed['check'].tolist()}")
        logging.info("Verified raw data: %s train, %s test, %s positive rows",
                     len(train), len(test), audit["label"]["positive_rows"])
        train = cast_target(train, config["target"])
        n_folds = config["n_folds"]
        if args.smoke:
            train, test, sample = smoke_sample(train, test, sample, config)
            n_folds = config["smoke"]["n_folds"]
        summary = validate_frames(train, test, sample, config)
        state["active_data"] = summary
        state["seeds"] = {"fold": config["seed"], "model": config["model_seed"],
                          "encoder": config["encoder_seed"], "early_stop": config["early_stop_seed"]}

        folds = assign_folds(train, config["target"], n_folds, config["seed"])
        folds.to_csv(run_dir / "fold_assignments.csv", index=False)
        state["fold_hash"] = sha256_file(run_dir / "fold_assignments.csv")
        state["n_folds"] = n_folds
        early = early_stop_assignments(train, folds, config, n_folds)
        early.to_csv(run_dir / "early_stop_assignments.csv", index=False)
        state["early_stop_hash"] = sha256_file(run_dir / "early_stop_assignments.csv")
        partitions = validate_partitions(train, folds, early, config, n_folds)
        write_json(run_dir / "cv_contract.json", {
            "fold_spec": f"StratifiedKFold(n_splits={n_folds}, shuffle=True, random_state={config['seed']})",
            "seeds": state["seeds"],
            "hashes": {"fold_assignments.csv": state["fold_hash"],
                       "early_stop_assignments.csv": state["early_stop_hash"]},
            "splits": {"T": "outer train = 80% of training rows",
                       "A": "fit part of T = 72% of training rows",
                       "E": "early-stop part of T = 8% of training rows",
                       "V": "outer validation = 20% of training rows"},
            "test_rows_used_for_labeled_fit": 0,
            "partitions": {str(k): v for k, v in partitions.items()},
        })
        if args.stage in ("setup", "folds"):
            fold_summary = folds.groupby("fold")[config["target"]].agg(["size", "sum", "mean"])
            fold_summary.to_csv(run_dir / "fold_summary.csv")
        logging.info("Locked %s folds and A/E/V splits (seeds: %s)", n_folds, state["seeds"])
        if args.stage == "eda":
            state["eda"] = run_eda(run_dir / "eda", train, test, folds, config)
        if args.stage == "contract":
            state["contract"] = _run_contract_stage(
                run_dir, train, test, sample, folds, early, config, n_folds, state)
        if args.stage == "baseline":
            state["baselines"] = [run_baseline(model_id, run_dir, train, test, sample, folds,
                                               early, config, n_folds, state)
                                  for model_id in selected]
        if args.stage == "complexity":
            state["complexity"] = run_complexity(run_dir, train, test, sample, folds,
                                                early, config, n_folds, state)
        if args.stage == "encoding":
            state["encoding"] = run_encoding(run_dir, train, test, sample, folds,
                                             early, config, n_folds, state, manifest)
        if args.stage == "te_models":
            state["phase8"] = run_te_models(ROOT, run_dir, train, test, sample, folds, early,
                                             config, n_folds, state, manifest, args.encoding_dir,
                                             te_selected, args.check_cache_only)
        state["status"] = "completed"
        if args.stage == "baseline":
            state["model_training"] = f"step 5 baselines run: {', '.join(selected)}"
        elif args.stage == "complexity":
            state["model_training"] = "step 6: tree complexity ablations and one-fold learning curve"
        elif args.stage == "encoding":
            state["model_training"] = "step 7: cross-fitted m-estimate target encoding and leakage guards"
        elif args.stage == "te_models":
            state["model_training"] = ("phase 8: cache check only" if args.check_cache_only
                                       else "phase 8: " + ", ".join(te_selected))
        elif args.stage == "contract":
            state["model_training"] = "contract smoke only; baselines are selected with --stage baseline"
        else:
            state["model_training"] = "not started; follow plan.md"
        state["data_quality_rows"] = len(table)
        (run_dir / "requirements.lock.txt").write_text(locked_requirements(env), encoding="utf-8")
        if not args.smoke:
            try:
                (ROOT / "requirements.lock.txt").write_text(locked_requirements(env), encoding="utf-8")
                state["project_lock"] = str(ROOT / "requirements.lock.txt")
            except OSError:
                state["project_lock"] = "skipped: project folder is read-only"
        submission_path(run_dir, args.smoke)
        write_json(run_dir / "run_config.json", state)
        logging.info("Foundation complete: %s", run_dir)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(run_dir / "run_config.json", state)
        logging.exception("Setup failed")
        raise


if __name__ == "__main__":
    main()
