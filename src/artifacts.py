"""Small, explicit helpers for reproducible run artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def environment() -> dict:
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "lightgbm", "xgboost", "catboost"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": versions}


def locked_requirements(env: dict) -> str:
    lines = ["# Exact versions recorded from a completed run; do not upgrade between ablation branches."]
    for name, installed in env["packages"].items():
        if installed is not None:
            lines.append(f"{name}=={installed}")
    return "\n".join(lines) + "\n"


def read_fold_status(path: Path) -> dict:
    return read_json(path) if path.is_file() else {}


def write_fold_status(path: Path, status: dict) -> None:
    write_json(path, status)


def fold_needs_run(status: dict, fold: int, required_files: list[Path]) -> bool:
    return (status.get(str(fold)) != "completed"
            or any(not path.is_file() for path in required_files))


OFFICIAL_SUBMISSION = "submission.csv"
SMOKE_SUBMISSION = "submission_smoke.csv"


def submission_path(run_dir: Path, smoke: bool) -> Path:
    if smoke:
        official = run_dir / OFFICIAL_SUBMISSION
        if official.exists():
            raise RuntimeError(f"smoke run must not contain {OFFICIAL_SUBMISSION}: {official}")
        return run_dir / SMOKE_SUBMISSION
    return run_dir / OFFICIAL_SUBMISSION


def save_bundle(bundle: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array")
        joblib.dump(bundle, path)


def load_bundle(path: Path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array")
        return joblib.load(path)
