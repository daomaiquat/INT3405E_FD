"""RTX 4060 training backends for the local credit-card fraud experiments."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from functools import lru_cache
from io import StringIO
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.special import expit


TREE_DEFAULTS = {
    "lgb": {
        "learning_rate": 0.03,
        "num_leaves": 4,
        "min_child_samples": 50,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    },
    "xgb": {
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    },
}


@lru_cache(maxsize=1)
def _cuda_name():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable.")
    name = torch.cuda.get_device_name(0)
    if "NVIDIA" not in name or "RTX 4060" not in name:
        raise RuntimeError(f"CUDA device 0 is {name!r}; RTX 4060 is required.")
    return name


@lru_cache(maxsize=1)
def _opencl_ids():
    from verify_gpu import find_nvidia_opencl

    return find_nvidia_opencl()


@lru_cache(maxsize=1)
def _opencl_cache_directory():
    return TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix=".opencl-cache-")


@contextmanager
def _opencl_cache():
    # Boost.Compute caches compiled Windows kernels in APPDATA. Give each process
    # a writable, isolated cache before the library resolves that path.
    previous = os.environ.get("APPDATA")
    os.environ["APPDATA"] = _opencl_cache_directory().name
    try:
        yield
    finally:
        if previous is None:
            del os.environ["APPDATA"]
        else:
            os.environ["APPDATA"] = previous


def _offset_vector(values, size, role):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid {role} logits: expected {size} finite values.")
    return values


def _tree_offsets(offsets, x_train, early, queries):
    if offsets is None:
        return None, None, (None,) * len(queries)
    train, eval_scores, query_scores = offsets
    if (early is None) != (eval_scores is None) or len(query_scores) != len(queries):
        raise ValueError("Provide offsets for every role present in the tree fit.")
    train = _offset_vector(train, len(x_train), "training")
    if early is not None:
        eval_scores = _offset_vector(eval_scores, len(early[0]), "early-stop")
    query_scores = tuple(
        _offset_vector(scores, len(matrix), f"query {index}")
        for index, (scores, matrix) in enumerate(zip(query_scores, queries))
    )
    return train, eval_scores, query_scores


def _fit_lgb(
    x_train, y_train, queries, early, offsets, settings, cap, patience, seed, include_state
):
    import lightgbm as lgb

    platform_id, device_id = _opencl_ids()
    train_offset, early_offset, query_offsets = offsets
    config = dict(
        settings,
        objective="binary",
        metric="auc",
        device_type="gpu",
        gpu_platform_id=platform_id,
        gpu_device_id=device_id,
        verbosity=1,
        seed=seed,
    )
    if train_offset is not None:
        config.update(boost_from_average=False, sigmoid=1.0)
    # Capture startup evidence and suppress repeated diagnostics after device verification.
    startup = StringIO()
    device_name = None

    def require_gpu(env):
        nonlocal device_name
        if env.iteration != env.begin_iteration:
            return
        match = re.search(
            r"Using GPU Device: ([^\n]*RTX 4060[^\n]*?), Vendor: NVIDIA\b",
            startup.getvalue(),
            re.IGNORECASE,
        )
        if match is None:
            raise RuntimeError("LightGBM GPU startup did not confirm RTX 4060:\n" + startup.getvalue())
        device_name = match.group(1)

    require_gpu.before_iteration = True
    callbacks = [require_gpu]
    train = lgb.Dataset(x_train, label=y_train, init_score=train_offset)
    validation = None
    if early is not None:
        x_early, y_early = early
        validation = lgb.Dataset(x_early, label=y_early, init_score=early_offset, reference=train)
        callbacks.append(lgb.early_stopping(patience, first_metric_only=True, verbose=False))
    start = perf_counter()
    with _opencl_cache(), redirect_stdout(startup):
        model = lgb.train(
            config, train, num_boost_round=cap, valid_sets=[validation] if early is not None else None,
            valid_names=["early"] if early is not None else None, callbacks=callbacks,
        )
    fit_seconds = perf_counter() - start
    if device_name is None:
        raise RuntimeError("LightGBM returned without running the RTX 4060 verification callback.")
    chosen = int(model.best_iteration or model.current_iteration())
    details = {
        "device": f"OpenCL {platform_id}:{device_id}: {device_name}",
        "fit_seconds": fit_seconds,
        "rounds": chosen,
        "early_auc": float(model.best_score["early"]["auc"]) if early is not None else None,
        "settings": config,
        "version": lgb.__version__,
    }
    predictions = []
    for matrix, offset in zip(queries, query_offsets):
        if offset is None:
            scores = model.predict(matrix, num_iteration=chosen, num_threads=config["num_threads"])
        else:
            residual = model.predict(
                matrix, raw_score=True, num_iteration=chosen, num_threads=config["num_threads"]
            )
            scores = expit(offset + residual)
        predictions.append(np.asarray(scores, dtype=np.float64))
    if include_state:
        details["model_state"] = model.model_to_string(num_iteration=chosen)
    return tuple(predictions), details


def _fit_xgb(
    x_train, y_train, queries, early, offsets, settings, cap, patience, seed, include_state
):
    import xgboost as xgb

    gpu_name = _cuda_name()
    if not xgb.build_info().get("USE_CUDA"):
        raise RuntimeError("The installed XGBoost build does not include CUDA support.")
    train_offset, early_offset, query_offsets = offsets
    config = dict(
        settings,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cuda:0",
        seed=seed,
    )

    class RequireCuda(xgb.callback.TrainingCallback):
        def before_training(self, model):
            device = json.loads(model.save_config())["learner"]["generic_param"]["device"]
            if device != "cuda:0":
                raise RuntimeError(f"XGBoost selected {device!r}; cuda:0 is required.")
            return model

    start = perf_counter()
    matrix_options = {"max_bin": config["max_bin"], "nthread": config["nthread"]}
    train = xgb.QuantileDMatrix(x_train, label=y_train, base_margin=train_offset, **matrix_options)
    evaluations = []
    if early is not None:
        x_early, y_early = early
        validation = xgb.QuantileDMatrix(
            x_early, label=y_early, base_margin=early_offset, ref=train, **matrix_options
        )
        evaluations.append((validation, "early"))
    model = xgb.train(
        config, train, num_boost_round=cap, evals=evaluations,
        early_stopping_rounds=patience if early is not None else None,
        verbose_eval=False, callbacks=[RequireCuda()],
    )
    fit_seconds = perf_counter() - start
    actual = json.loads(model.save_config())["learner"]["generic_param"]["device"]
    if actual != "cuda:0":
        raise RuntimeError(f"XGBoost finished on {actual!r}; CUDA was required.")
    chosen = int(model.best_iteration + 1 if early is not None else model.num_boosted_rounds())
    details = {
        "device": f"{actual}: {gpu_name}",
        "fit_seconds": fit_seconds,
        "rounds": chosen,
        "early_auc": float(model.best_score) if early is not None else None,
        "settings": config,
        "version": xgb.__version__,
    }
    predictions = tuple(
        np.asarray(
            model.predict(
                xgb.DMatrix(matrix, base_margin=offset, nthread=config["nthread"]),
                iteration_range=(0, chosen),
            ),
            dtype=np.float64,
        )
        for matrix, offset in zip(queries, query_offsets)
    )
    if include_state:
        details["model_state"] = bytes(model[:chosen].save_raw(raw_format="json")).decode("utf-8")
    return predictions, details


def _standardized_columns(columns, width):
    if columns is None:
        return list(range(width))
    if isinstance(columns, (int, np.integer)):
        if not 0 <= columns <= width:
            raise ValueError("GLM standardize_cols count is out of range.")
        return list(range(int(columns)))
    selected = [int(index) for index in columns]
    if len(set(selected)) != len(selected) or any(not 0 <= index < width for index in selected):
        raise ValueError("GLM standardize_cols contains repeated or out-of-range feature indices.")
    return selected


def _glm_loss(design, labels, weights, l2):
    from torch.nn import functional as F

    return F.binary_cross_entropy_with_logits(design @ weights, labels) + (
        0.5 * l2 * weights[:-1].square().sum()
    )


def _glm_gradient(design, labels, weights, l2):
    margins = design @ weights
    probabilities = margins.sigmoid()
    gradient = design.T @ (probabilities - labels) / len(labels)
    gradient[:-1] += l2 * weights[:-1]
    return gradient, margins, probabilities


def _newton_cholesky(design, labels, weights, l2, max_iter, tol):
    import torch

    penalty = torch.full_like(weights, l2)
    penalty[-1] = 0.0
    for iteration in range(max_iter):
        gradient, margins, probabilities = _glm_gradient(design, labels, weights, l2)
        if float(gradient.abs().max()) <= tol:
            return weights, iteration, "gradient tolerance"
        curvature = (probabilities * (-margins).sigmoid()).sqrt()
        weighted_design = design * curvature[:, None]
        hessian = weighted_design.T @ weighted_design / len(labels)
        del weighted_design
        hessian.diagonal().add_(penalty)
        factor = torch.linalg.cholesky(hessian)
        step = torch.cholesky_solve(gradient[:, None], factor).squeeze(1)
        base_loss = _glm_loss(design, labels, weights, l2)
        expected_drop = gradient @ step
        for reduction in range(25):
            rate = 0.5 ** reduction
            candidate = weights - rate * step
            if _glm_loss(design, labels, candidate, l2) <= base_loss - 1e-4 * rate * expected_drop:
                weights = candidate
                break
        else:
            return weights, iteration, "line search stalled"
    return weights, max_iter, "iteration limit"


def _lbfgs(design, labels, weights, l2, max_iter, tol):
    import torch

    weights.requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [weights], max_iter=max_iter, tolerance_grad=tol, tolerance_change=0.0,
        history_size=25, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = _glm_loss(design, labels, weights, l2)
        loss.backward()
        return loss

    optimizer.step(closure)
    iterations = int(optimizer.state[weights].get("n_iter", 0))
    return weights.detach(), iterations, (
        "iteration limit" if iterations >= max_iter else "LBFGS stopping condition"
    )


def _fit_glm(x_train, y_train, queries, settings, include_state):
    import torch

    gpu_name = _cuda_name()
    config = {
        "C": 1.0, "max_iter": 2000, "tol": 0.0001, "solver": "lbfgs",
        "dtype": "float64", "standardize_cols": None,
    }
    if settings.keys() - config.keys():
        raise ValueError(f"Unknown GLM settings: {sorted(settings.keys() - config.keys())}")
    config.update(settings)
    solvers = {"lbfgs": _lbfgs, "newton-cholesky": _newton_cholesky}
    dtype = {"float64": torch.float64, "float32": torch.float32}.get(config["dtype"])
    solver = solvers.get(config["solver"])
    regularization, tol, max_iter = float(config["C"]), float(config["tol"]), int(config["max_iter"])
    if dtype is None or solver is None or regularization <= 0 or tol <= 0 or max_iter < 1:
        raise ValueError(f"Invalid GLM settings: {config}")
    labels = np.asarray(y_train)
    if labels.shape != (len(x_train),) or not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("GLM training requires both binary classes.")
    x_train = np.asarray(x_train)
    width = x_train.shape[1]
    columns = _standardized_columns(config["standardize_cols"], width)
    config["standardize_cols"] = columns
    device = torch.device("cuda:0")
    torch.cuda.synchronize(device)
    start = perf_counter()
    labels = torch.as_tensor(labels, dtype=dtype, device=device)
    indices = torch.tensor(columns, dtype=torch.long, device=device)

    def make_design(matrix, center=None, scale=None):
        values = torch.as_tensor(np.asarray(matrix), dtype=dtype, device=device)
        design = torch.empty((len(values), width + 1), dtype=dtype, device=device)
        design[:, :-1] = values
        design[:, -1] = 1
        if columns:
            selected = values[:, indices]
            if center is None:
                scale, center = torch.std_mean(selected, dim=0, correction=0)
                scale = torch.where(scale > 0, scale, 1)
            design[:, indices] = (selected - center) / scale
        return design, center, scale

    design, center, scale = make_design(x_train)
    weights = torch.zeros(width + 1, dtype=dtype, device=device)
    weights[-1] = torch.logit(labels.mean())
    # The sklearn objective uses an unpenalized intercept and L2 strength 1/C
    # against summed loss. Dividing both terms by n gives the objective here.
    l2 = 1.0 / (regularization * len(labels))
    weights, iterations, reason = solver(design, labels, weights, l2, max_iter, tol)
    gradient, _, _ = _glm_gradient(design, labels, weights, l2)
    gradient_max = float(gradient.abs().max())
    loss = float(_glm_loss(design, labels, weights, l2))
    if not np.isfinite([gradient_max, loss]).all():
        raise RuntimeError("GLM optimization returned a non-finite loss or gradient.")
    torch.cuda.synchronize(device)
    fit_seconds = perf_counter() - start
    del design, labels

    predictions = []
    with torch.inference_mode():
        for matrix in queries:
            probabilities = np.empty(len(matrix), dtype=np.float64)
            for start in range(0, len(matrix), 131072):
                rows = slice(start, start + 131072)
                design, _, _ = make_design(matrix[rows], center, scale)
                probabilities[rows] = (design @ weights).sigmoid().cpu().numpy()
            predictions.append(probabilities)
    converged = gradient_max <= tol
    details = {
        "device": f"{device}: {gpu_name} (PyTorch CUDA)",
        "fit_seconds": fit_seconds,
        "rounds": 0,
        "iterations": iterations,
        "converged": converged,
        "gradient_inf_norm": gradient_max,
        "objective": loss,
        "stop_reason": "gradient tolerance" if converged else reason,
        "settings": config,
        "version": torch.__version__,
    }
    if include_state:
        details["model_state"] = {
            "coefficients": weights[:-1].cpu().tolist(),
            "intercept": float(weights[-1]),
            "standardize_cols": columns,
            "center": center.cpu().tolist() if center is not None else [],
            "scale": scale.cpu().tolist() if scale is not None else [],
        }
    return tuple(predictions), details


def fit_predict(
    family: str, x_train, y_train, queries: tuple, *, early: tuple | None = None,
    params: dict | None = None, seed: int = 42, rounds: int | None = None, offsets: tuple | None = None
) -> tuple[tuple[np.ndarray, ...], dict]:
    """Fit on supplied rows; optional early data selects tree rounds for a separate refit."""
    family = {"lightgbm": "lgb", "xgboost": "xgb"}.get(family, family)
    settings = dict(params or {})
    include_state = bool(settings.pop("include_model_state", False))
    if family == "glm":
        if early is not None or offsets is not None or rounds not in (None, 0):
            raise ValueError("GLM accepts neither early stopping, tree rounds nor offsets.")
        return _fit_glm(x_train, y_train, queries, settings, include_state)

    backends = {"lgb": _fit_lgb, "xgb": _fit_xgb}
    if family not in backends:
        raise ValueError(f"Unknown family: {family!r}")
    patience = int(settings.pop("patience", settings.pop("early_stopping_rounds", 200)))
    cap = int(settings.pop("max_iterations", 6000))
    cap = int(rounds) if rounds is not None else cap
    if cap < 1 or patience < 1:
        raise ValueError("Tree rounds and patience must be positive.")
    if family == "lgb":
        settings = {"max_bin": 255, "num_threads": 4, **TREE_DEFAULTS[family], **settings}
    else:
        settings = {"max_bin": 256, "nthread": 4, **TREE_DEFAULTS[family], **settings}
    checked_offsets = _tree_offsets(offsets, x_train, early, queries)
    return backends[family](
        x_train, y_train, queries, early, checked_offsets, settings, cap, patience, int(seed),
        include_state,
    )


def predict_from_state(
    family: str, state: dict | str, queries: tuple, query_offsets: tuple | None = None
) -> tuple[np.ndarray, ...]:
    """Replay a saved binary model on CPU using the same query matrices and optional logits."""
    family = {"lightgbm": "lgb", "xgboost": "xgb"}.get(family, family)
    if query_offsets is None:
        offsets = (None,) * len(queries)
    else:
        if family == "glm":
            raise ValueError("GLM replay does not accept offsets.")
        if len(query_offsets) != len(queries):
            raise ValueError("Provide one offset vector for every query matrix.")
        offsets = tuple(
            _offset_vector(values, len(matrix), f"query {index}")
            for index, (values, matrix) in enumerate(zip(query_offsets, queries))
        )

    if family == "glm":
        coefficients = np.asarray(state["coefficients"], dtype=np.float64)
        columns = np.asarray(state["standardize_cols"], dtype=np.intp)
        center = np.asarray(state["center"], dtype=np.float64)
        scale = np.asarray(state["scale"], dtype=np.float64)
        intercept = float(state["intercept"])

        def predict(matrix, _offset):
            design = np.array(matrix, dtype=np.float64, copy=True)
            design[:, columns] = (design[:, columns] - center) / scale
            return expit(design @ coefficients + intercept)

    elif family == "lgb":
        import lightgbm as lgb

        model = lgb.Booster(model_str=state)

        def predict(matrix, offset):
            if offset is None:
                return model.predict(matrix, num_iteration=-1, num_threads=4)
            residual = model.predict(matrix, raw_score=True, num_iteration=-1, num_threads=4)
            return expit(offset + residual)

    elif family == "xgb":
        import xgboost as xgb

        model = xgb.Booster()
        model.load_model(bytearray(state.encode("utf-8")))
        model.set_param({"device": "cpu", "nthread": 4})

        def predict(matrix, offset):
            return model.predict(xgb.DMatrix(matrix, base_margin=offset, nthread=4))

    else:
        raise ValueError(f"Unknown family: {family!r}")

    return tuple(
        np.asarray(predict(matrix, offset), dtype=np.float64) if len(matrix)
        else np.empty(0, dtype=np.float64)
        for matrix, offset in zip(queries, offsets)
    )
