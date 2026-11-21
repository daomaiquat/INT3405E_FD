"""Export selected fold models and replay them against the captured test predictions."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logit

from models import fit_predict, predict_from_state
from run_plan import read_json, sha256, write_json


FORMAT_VERSION = 1
REPLAY_TOLERANCE = 1e-6
CHECKPOINT_TOLERANCE = 5e-6
# The CPU and CUDA XGBoost predictors sum float32 tree contributions in
# different orders. Keep that cross-device allowance separate from GLM/LGB.
XGBOOST_REPLAY_TOLERANCE = 2e-5
OFFSET_CLIP = 1e-7


def _json_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _difference(actual, expected, label, tolerance):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"{label}: missing or non-finite predictions")
    if np.any((actual < 0) | (actual > 1)) or np.any((expected < 0) | (expected > 1)):
        raise ValueError(f"{label}: invalid probabilities")
    difference = float(np.max(np.abs(actual - expected), initial=0.0))
    if difference > tolerance:
        raise ValueError(f"{label}: max absolute difference {difference:.8g} exceeds {tolerance:.8g}")
    return difference


def _margin(probabilities):
    return logit(np.clip(np.asarray(probabilities, dtype=np.float64), OFFSET_CLIP, 1 - OFFSET_CLIP))


def _offset_spec(spec):
    return replace(
        spec,
        family="glm",
        groups=("glm_raw", *spec.groups[1:]),
        boost=False,
        params={"C": 1.0, "solver": "lbfgs", "max_iter": 2000, "tol": 0.0001},
    )


def _model_details(details):
    if "RTX 4060" not in details["device"]:
        raise RuntimeError(f"GPU could not be verified for bundle refit: {details['device']}")
    return {
        "state": details["model_state"],
        "device": details["device"],
        "version": details["version"],
        "settings": details["settings"],
        "fit_seconds": details["fit_seconds"],
        "rounds": details["rounds"],
    }


def _refit(run, spec, reference, matrices, offset_spec, offset_matrices, rounds):
    x_fit, x_test = matrices
    offsets = None
    glm = None
    if offset_spec is not None:
        x_glm_fit, x_glm_test = offset_matrices
        params = run.model_params(offset_spec, offset_matrices, reference)
        params = dict(params, include_model_state=True)
        (p_train, p_test), details = fit_predict(
            "glm", x_glm_fit, run.y[reference], (x_glm_fit, x_glm_test),
            params=params, seed=spec.model_seed,
        )
        glm = _model_details(details)
        offsets = (_margin(p_train), None, (_margin(p_test),))
    params = run.model_params(spec, matrices, reference)
    params = dict(params, include_model_state=True)
    (test,), details = fit_predict(
        spec.family, x_fit, run.y[reference], (x_test,), params=params,
        seed=spec.model_seed, rounds=rounds or None, offsets=offsets,
    )
    model = _model_details(details)
    if model["rounds"] != rounds:
        raise ValueError(f"{spec.name}: checkpoint used {rounds} rounds, refit used {model['rounds']}")
    return test, {"model": model, "glm_offset": glm}


def _replay(spec, states, x_test, x_glm_test):
    query_offsets = None
    if spec.boost:
        (probabilities,) = predict_from_state("glm", states["glm_offset"]["state"], (x_glm_test,))
        query_offsets = (_margin(probabilities),)
    (test,) = predict_from_state(spec.family, states["model"]["state"], (x_test,), query_offsets)
    return test


def _cached_record(path, provenance):
    if not path.is_file():
        return None
    try:
        record = read_json(path)
        if (record["provenance"] == provenance
                and record["state_sha256"] == _json_digest(record["states"])):
            return record
    except (ValueError, KeyError, TypeError):
        pass
    return None


def _selected_bundles(run, specs, output, *, allow_fit):
    output = Path(output).resolve()
    if allow_fit:
        output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": FORMAT_VERSION,
        "source_run": run.manifest,
        "preprocessing": "Rebuild fold features from the locked competition inputs and member specification.",
        "glm_offset_clip": OFFSET_CLIP,
        "members": {},
    }
    test_predictions = {}
    for spec in specs:
        if spec.name in test_predictions:
            raise ValueError(f"Duplicate bundle member {spec.name}")
        if spec.family == "xgb":
            replay_tolerance = checkpoint_tolerance = XGBOOST_REPLAY_TOLERANCE
        else:
            replay_tolerance, checkpoint_tolerance = REPLAY_TOLERANCE, CHECKPOINT_TOLERANCE
        source = run.output / spec.name
        spec_json = json.loads(json.dumps(asdict(spec)))
        if spec_json != read_json(source / "config.json"):
            raise ValueError(f"Bundle specification differs from checkpoint: {spec.name}")
        spec_hash = _json_digest(spec_json)
        _, expected_mean = run.load(spec.name)
        member = output / spec.name
        if allow_fit:
            member.mkdir(exist_ok=True)
        fold_replays = []
        fold_records = []
        for number, part in enumerate(run.partitions, start=1):
            source_fold = source / f"fold_{number}.npz"
            source_details = read_json(source / f"fold_{number}.json")
            rounds = int(source_details["rounds"])
            with np.load(source_fold) as saved:
                if not np.array_equal(saved["ids"], run.ids[part["V"]]):
                    raise ValueError(f"Checkpoint validation IDs differ: {spec.name} fold {number}")
                expected = np.asarray(saved["test"], dtype=np.float64)
            if expected.shape != (len(run.test_indices),):
                raise ValueError(f"Invalid checkpoint test shape: {spec.name} fold {number}")
            provenance = {
                "format_version": FORMAT_VERSION,
                "spec_sha256": spec_hash,
                "source_run": run.manifest,
                "source_fold_sha256": sha256(source_fold),
                "fold": number,
                "rounds": rounds,
                "glm_offset_clip": OFFSET_CLIP if spec.boost else None,
            }
            path = member / f"fold_{number}.json"
            record = _cached_record(path, provenance)
            label = f"{spec.name} fold {number}"
            if record is None and not allow_fit:
                raise ValueError(f"No matching fold model state: {path}")

            reference = part["T"]
            queries = (run.test_indices,)
            matrices = run.matrices(spec, reference, queries)
            offset_spec = _offset_spec(spec) if spec.boost else None
            offset_matrices = run.matrices(offset_spec, reference, queries) if spec.boost else None
            x_glm_test = offset_matrices[1] if offset_matrices is not None else None
            if record is None:
                print(f"{label}: refitting selected bundle model on GPU", flush=True)
                fitted, states = _refit(run, spec, reference, matrices, offset_spec, offset_matrices, rounds)
                replay = _replay(spec, states, matrices[1], x_glm_test)
                replay_difference = _difference(replay, fitted, label + " CPU replay", replay_tolerance)
                checkpoint_difference = _difference(replay, expected, label + " checkpoint", checkpoint_tolerance)
                record = {
                    "provenance": provenance,
                    "states": states,
                    "state_sha256": _json_digest(states),
                    "max_abs_cpu_vs_refit": replay_difference,
                    "max_abs_cpu_vs_checkpoint": checkpoint_difference,
                }
                write_json(path, record)
            else:
                replay = _replay(spec, record["states"], matrices[1], x_glm_test)
                checkpoint_difference = _difference(replay, expected, label + " checkpoint", checkpoint_tolerance)
            fold_replays.append(replay)
            fold_records.append({
                "fold": number,
                "state_file": str(path.relative_to(output).as_posix()),
                "state_file_sha256": sha256(path),
                "rounds": rounds,
                "max_abs_cpu_vs_refit": record["max_abs_cpu_vs_refit"],
                "max_abs_cpu_vs_checkpoint": checkpoint_difference,
            })
            del matrices, offset_matrices, x_glm_test
        mean_replay = np.column_stack(fold_replays).mean(axis=1)
        difference = _difference(mean_replay, expected_mean, spec.name + " fold mean", checkpoint_tolerance)
        test_predictions[spec.name] = mean_replay
        manifest["members"][spec.name] = {
            "spec": spec_json,
            "spec_sha256": spec_hash,
            "folds": fold_records,
            "cpu_replay_tolerance": replay_tolerance,
            "checkpoint_tolerance": checkpoint_tolerance,
            "max_abs_cpu_mean_vs_checkpoint": difference,
        }
    return test_predictions, manifest


def export_members(run, specs, output):
    """Refit missing selected states on GPU, then save the verified fold bundle manifest."""
    predictions, manifest = _selected_bundles(run, specs, output, allow_fit=True)
    write_json(Path(output) / "manifest.json", manifest)
    return predictions, manifest


def verify_members(run, specs, output):
    """Replay saved states on CPU without refitting and verify the published manifest hashes."""
    predictions, manifest = _selected_bundles(run, specs, output, allow_fit=False)
    saved = read_json(Path(output) / "manifest.json")
    if (saved["format_version"] != FORMAT_VERSION or saved["source_run"] != manifest["source_run"]
            or saved["glm_offset_clip"] != OFFSET_CLIP):
        raise ValueError("Bundle manifest does not match the supplied run")
    for name, member in manifest["members"].items():
        previous = saved["members"][name]
        if (member["spec_sha256"] != previous["spec_sha256"]
                or [fold["state_file_sha256"] for fold in member["folds"]]
                != [fold["state_file_sha256"] for fold in previous["folds"]]):
            raise ValueError(f"Stored manifest hashes differ for {name}")
    return predictions, manifest
