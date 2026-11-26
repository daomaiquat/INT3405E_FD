"""Matched CUDA MLP comparison of BCE, focal, and pairwise ranking losses."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / "plan_runs"
FEATURE_CONFIG = PLAN / "results" / "full" / "XGB-D03" / "config.json"
sys.path.insert(0, str(PLAN))
from run_plan import Experiment, PlanRun, read_json, sha256, write_json, write_npz


VARIANTS = {
    "MLP-BCE": ("bce", None),
    "MLP-FOCAL-G1": ("focal", 1.0),
    "MLP-FOCAL-G2": ("focal", 2.0),
    "MLP-PAIRWISE": ("pairwise", None),
}
EPOCHS = 12
BATCH_SIZE = 4096
WIDTHS = (128, 64)
DROPOUT = 0.10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
NEGATIVES_PER_POSITIVE = 32
FEATURE_GROUPS = ("raw", "te13", "cents", "digits")


def cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch chưa nhận được CUDA.")
    name = torch.cuda.get_device_name(0)
    if "RTX 4060" not in name:
        raise RuntimeError(f"Thí nghiệm yêu cầu RTX 4060, CUDA đang dùng {name!r}.")
    return torch.device("cuda:0"), f"cuda:0: {name} (PyTorch CUDA)"


def network(input_width):
    first, second = WIDTHS
    return nn.Sequential(
        nn.Linear(input_width, first), nn.LayerNorm(first), nn.GELU(), nn.Dropout(DROPOUT),
        nn.Linear(first, second), nn.LayerNorm(second), nn.GELU(), nn.Dropout(DROPOUT),
        nn.Linear(second, 1), nn.Flatten(start_dim=0),
    )


def pointwise_loss(logits, targets, gamma):
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if gamma is not None:
        # With binary labels exp(-BCE) is p_t, and expm1 is accurate for small BCE.
        losses = losses * (-torch.expm1(-losses)).pow(gamma)
    return losses.mean()


def pair_indexes(targets, rng):
    positive = np.flatnonzero(targets == 1)
    negative = np.flatnonzero(targets == 0)
    if not positive.size or not negative.size:
        return None
    sampled = rng.choice(negative, size=(len(positive), NEGATIVES_PER_POSITIVE), replace=True)
    return np.column_stack((positive, sampled))


def pairwise_loss(logits, indexes):
    margins = logits[indexes[:, 1:]] - logits[indexes[:, :1]]
    return F.softplus(margins).mean()


def probability_scores(model, values):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([
            model(batch).double().sigmoid().cpu().numpy()
            for batch in values.split(BATCH_SIZE)
        ])


def config_for(name, source, epochs, source_hash):
    objective, gamma = VARIANTS[name]
    return {
        "name": name,
        "family": "mlp",
        "parent": "" if name == "MLP-BCE" else "MLP-BCE",
        "feature_source": source.name,
        "feature_config_sha256": source_hash,
        "groups": list(source.groups),
        "encoder_seed": source.encoder_seed,
        "smoothing": source.smoothing,
        "model_seed": source.model_seed,
        "scaler": "StandardScaler fitted on outer training matrix",
        "hidden_widths": list(WIDTHS),
        "activation": "GELU",
        "normalization": "LayerNorm",
        "dropout": DROPOUT,
        "optimizer": "AdamW fused",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "sampling": "ordinary random shuffle without replacement in each epoch",
        "class_weights": None,
        "target_prevalence": "unchanged in outer validation and test",
        "objective": objective,
        "focal_gamma": gamma,
        "focal_alpha": None,
        "pair_sampling": ("within-batch negative sampling with replacement"
                          if objective == "pairwise" else None),
        "negatives_per_positive": NEGATIVES_PER_POSITIVE if objective == "pairwise" else None,
        "score_kind": "sigmoid_probability" if objective == "bce" else "uncalibrated_sigmoid_margin",
        "calibration_evaluated": False,
        "log_loss_note": ("recorded by the shared evaluator; use AUC/AP for the loss comparison"),
        "torch_version": str(torch.__version__),
    }


def fit_fold(config, fold, device, train, labels, valid, test, device_name):
    start = perf_counter()
    objective, gamma = config["objective"], config["focal_gamma"]
    seed = config["model_seed"] + fold - 1
    torch.manual_seed(seed)
    model = network(train.shape[1])
    initialization = hashlib.sha256()
    for parameter in model.parameters():
        initialization.update(parameter.detach().numpy().tobytes())
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True,
    )
    row_rng = np.random.default_rng(seed)
    pair_rng = np.random.default_rng(seed + 10_000)
    batch_order = hashlib.sha256()
    y_gpu = torch.as_tensor(labels, dtype=torch.float32, device=device)
    losses = []
    skipped = 0
    n_pairs = 0
    model.train()
    for epoch in range(config["epochs"]):
        cpu_order = row_rng.permutation(len(labels))
        batch_order.update(cpu_order.astype("<i8", copy=False).tobytes())
        order = torch.as_tensor(cpu_order, device=device)
        total_loss = torch.zeros((), device=device)
        contributions = 0
        for offset in range(0, len(labels), BATCH_SIZE):
            rows = order[offset:offset + BATCH_SIZE]
            if objective == "pairwise":
                indexes = pair_indexes(labels[cpu_order[offset:offset + BATCH_SIZE]], pair_rng)
                if indexes is None:
                    skipped += 1
                    continue
                n_pairs += len(indexes) * NEGATIVES_PER_POSITIVE
            optimizer.zero_grad(set_to_none=True)
            logits = model(train[rows])
            loss = (
                pairwise_loss(logits, torch.as_tensor(indexes, device=device))
                if objective == "pairwise" else pointwise_loss(logits, y_gpu[rows], gamma)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
            contributions += 1
        if not contributions:
            raise RuntimeError(f"{config['name']} has no valid batches in epoch {epoch + 1}")
        epoch_loss = float(total_loss.item()) / contributions
        if not np.isfinite(epoch_loss):
            raise ValueError(f"{config['name']} fold {fold} loss is not finite")
        losses.append(epoch_loss)
        print(f"{config['name']} fold {fold} | epoch {epoch + 1}/{config['epochs']}"
              f" | loss {epoch_loss:.6f}", flush=True)
    torch.cuda.synchronize(device)
    fit_seconds = perf_counter() - start
    scores = tuple(probability_scores(model, values) for values in (valid, test))
    details = {
        "fold": fold,
        "device": device_name,
        "fit_seconds": fit_seconds,
        "seconds": perf_counter() - start,
        "epochs": config["epochs"],
        "feature_count": train.shape[1],
        "epoch_loss": losses,
        "seed": seed,
        "initialization_sha256": initialization.hexdigest(),
        "batch_order_sha256": batch_order.hexdigest(),
        "skipped_batches": skipped,
        "pair_count": n_pairs,
    }
    return scores, details


def load_fold(directory, fold, validation_ids, test_ids):
    prediction_path = directory / f"fold_{fold}.npz"
    metric_path = directory / f"fold_{fold}.json"
    if not prediction_path.is_file() or not metric_path.is_file():
        return None
    with np.load(prediction_path, allow_pickle=False) as saved:
        if (not np.array_equal(saved["ids"], validation_ids)
                or not np.array_equal(saved["test_ids"], test_ids)):
            raise ValueError(f"Checkpoint ID mismatch for {directory.name} fold {fold}")
        return (saved["valid"], saved["test"]), read_json(metric_path)


def validate_fold(name, scores, labels, test_ids, details, device_name):
    for values, shape in zip(scores, (labels.shape, test_ids.shape), strict=True):
        if values.shape != shape or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{name} fold prediction is outside the expected size or range")
    auc = float(roc_auc_score(labels, scores[0]))
    if details["device"] != device_name or not np.isclose(details["auc"], auc, rtol=0, atol=1e-12):
        raise ValueError(f"{name} fold checkpoint disagrees with device or validation AUC")


def run_experiments(run, source, names, device, device_name, epochs):
    configs = {name: config_for(name, source, epochs, sha256(FEATURE_CONFIG)) for name in names}
    oof = {name: np.full(len(run.y), np.nan, dtype=np.float64) for name in names}
    test_by_fold = {name: np.empty((len(run.test_ids), len(run.partitions)), dtype=np.float64)
                    for name in names}
    histories = {name: [] for name in names}
    for name, config in configs.items():
        directory = run.output / name
        directory.mkdir(exist_ok=True)
        config_path = directory / "config.json"
        if config_path.is_file() and read_json(config_path) != config:
            raise ValueError(f"Matched experiment config changed for {name}")
        write_json(config_path, config)

    for fold, part in enumerate(run.partitions, start=1):
        reference, validation = part["T"], part["V"]
        labels = run.y[reference]
        matches = set()
        train = valid = test = None
        for name, config in configs.items():
            directory = run.output / name
            previous = load_fold(directory, fold, run.ids[validation], run.test_ids)
            if previous is None:
                if train is None:
                    started = perf_counter()
                    matrices = run.matrices(source, reference, (validation, run.test_indices))
                    scaler = StandardScaler(copy=False).fit(matrices[0])
                    train, valid, test = (
                        torch.as_tensor(scaler.transform(matrix), dtype=torch.float32, device=device)
                        for matrix in matrices
                    )
                    del matrices
                    print(f"Fold {fold} | {train.shape[1]} features | preprocessing "
                          f"{perf_counter() - started:.1f}s | {device_name}", flush=True)
                scores, details = fit_fold(config, fold, device, train, labels, valid, test, device_name)
                details.update(auc=float(roc_auc_score(run.y[validation], scores[0])),
                               train_rows=len(reference), valid_rows=len(validation),
                               train_positives=int(labels.sum()),
                               valid_positives=int(run.y[validation].sum()))
                validate_fold(name, scores, run.y[validation], run.test_ids, details, device_name)
                write_npz(directory / f"fold_{fold}.npz", ids=run.ids[validation], test_ids=run.test_ids,
                          valid=scores[0], test=scores[1])
                write_json(directory / f"fold_{fold}.json", details)
            else:
                scores, details = previous
                validate_fold(name, scores, run.y[validation], run.test_ids, details, device_name)
            matches.add((details["initialization_sha256"], details["batch_order_sha256"]))
            oof[name][validation], test_by_fold[name][:, fold - 1] = scores
            histories[name].append(details)
            print(f"{name} fold {fold}: AUC {details['auc']:.6f} | {details['seconds']:.1f}s"
                  f" | skipped {details['skipped_batches']} | pairs {details['pair_count']:,}",
                  flush=True)
        if len(matches) != 1:
            raise RuntimeError(f"Fold {fold} variants have different initialization or batch order")

    for name, config in configs.items():
        results = run.save_predictions(
            name, oof[name], test_by_fold[name].mean(axis=1), config, histories[name],
            test_by_fold=test_by_fold[name],
        )
        print(f"{name} | AUC {results['oof_auc']:.6f} | AP {results['average_precision']:.6f}"
              f" | {config['score_kind']}", flush=True)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", choices=tuple(VARIANTS), nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    names = tuple(name for name in VARIANTS if not args.only or name in args.only)
    output = args.output or HERE / "results" / ("mlp-smoke" if args.smoke else "mlp")
    source = Experiment(**read_json(FEATURE_CONFIG))
    if source.name != "XGB-D03" or tuple(source.groups) != FEATURE_GROUPS:
        raise ValueError("XGB-D03 saved feature contract differs from this comparison")
    device, device_name = cuda_device()
    torch.set_float32_matmul_precision("high")
    epochs = 2 if args.smoke else EPOCHS
    print(f"{device_name} | PyTorch {torch.__version__} | {epochs} fixed epochs", flush=True)
    run = PlanRun(output, smoke=args.smoke)
    run_experiments(run, source, names, device, device_name, epochs)


if __name__ == "__main__":
    cli()
