"""Compare self-supervised pretraining methods on the locked S6E11 folds."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits


HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / "plan_runs"
RESULTS = HERE / "results"
sys.path.insert(0, str(PLAN))
from run_plan import PlanRun, metric_row, read_json, sha256, write_json, write_npz


@dataclass(frozen=True)
class Variant:
    architecture: str
    objective: str | None
    batch: int
    pretrain_batch: int
    finetune_epochs: int
    pretrain_epochs: int = 3


VARIANTS = {
    "mlp-scratch": Variant("mlp", None, 4096, 4096, 6, 0),
    "mlp-masked": Variant("mlp", "masked", 4096, 4096, 6),
    "mlp-scarf": Variant("mlp", "scarf", 4096, 512, 6),
    "ft-scratch": Variant("ft", None, 2048, 2048, 5, 0),
    "ft-masked": Variant("ft", "masked", 2048, 2048, 5),
    "tabnet-scratch": Variant("tabnet", None, 4096, 4096, 4, 0),
    "tabnet-masked": Variant("tabnet", "masked", 4096, 4096, 4),
}


def model_source(architecture):
    return HERE / ("tabnet_model.py" if architecture == "tabnet" else "neural_models.py")


def new_model(architecture, numeric_count, category_sizes):
    if architecture == "tabnet":
        from tabnet_model import build_tabnet

        return build_tabnet(numeric_count, category_sizes)

    from neural_models import build_model

    return build_model(architecture, numeric_count, category_sizes)


def cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch chưa nhận được GPU CUDA.")
    name = torch.cuda.get_device_name(0)
    if "RTX 4060" not in name:
        raise RuntimeError(f"Yêu cầu NVIDIA RTX 4060, CUDA nhận {name}.")
    torch.set_float32_matmul_precision("high")
    return torch.device("cuda:0"), f"cuda:0: {name}"


class FoldInputs:
    def __init__(self, run, part, device):
        numeric = run.config["numeric_columns"]
        categorical = run.config["categorical_columns"] + run.config["flag_columns"]
        self.numeric_count = len(numeric)
        sizes = []

        train = run.train.iloc[part["T"]]
        valid = run.train.iloc[part["V"]]
        frames = (train, valid, run.test)
        numbers = train[numeric].to_numpy(dtype=np.float64)
        mean = numbers.mean(axis=0)
        scale = numbers.std(axis=0)
        scale[scale == 0] = 1
        values = [(frame[numeric].to_numpy(dtype=np.float64) - mean) / scale for frame in frames]
        values = [array.astype(np.float32) for array in values]

        for column in categorical:
            known = sorted(train[column].dropna().unique().tolist())
            mapping = {value: offset for offset, value in enumerate(known, 1)}
            sizes.append(len(mapping) + 1)
            values = [
                np.column_stack((array, frame[column].map(mapping).fillna(0).to_numpy(dtype=np.float32)))
                for array, frame in zip(values, frames, strict=True)
            ]

        if not all(np.isfinite(array).all() for array in values):
            raise ValueError("Non-finite values after preprocessing")
        self.category_sizes = tuple(sizes)
        self.train, self.valid, self.test = (
            torch.as_tensor(array, device=device) for array in values
        )
        self.labels = torch.as_tensor(run.y[part["T"]], dtype=torch.float32, device=device)


def train_epochs(model, fold_data, *, epochs, batch_size, seed, objective, use_amp):
    torch.manual_seed(seed)
    device = fold_data.train.device
    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4, fused=True)
    torch.cuda.synchronize()
    start = perf_counter()
    model.train()
    history = []
    phase = objective or "bce"
    for epoch in range(epochs):
        order = torch.randperm(len(fold_data.train), generator=generator, device=device)
        losses = []
        for indexes in order.split(batch_size):
            if indexes.numel() < 2:
                # Both contrastive loss and TabNet batch normalization need a pair.
                indexes = torch.cat((indexes, order[:1]))
            batch = fold_data.train[indexes]
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with amp:
                if objective:
                    loss = model.pretrain_loss(batch, objective)
                else:
                    score = model(batch).float()
                    loss = binary_cross_entropy_with_logits(score, fold_data.labels[indexes])
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().float())
        mean_loss = torch.stack(losses).mean().item()
        if not np.isfinite(mean_loss):
            raise ValueError(f"Non-finite {phase} loss at epoch {epoch + 1}")
        history.append(mean_loss)
        print(f"  {phase} epoch {epoch + 1}/{epochs}: loss {mean_loss:.6f}", flush=True)
    return history, perf_counter() - start


def probability(model, values, batch_size):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([
            model(batch).float().sigmoid().cpu().numpy() for batch in values.split(batch_size)
        ])


def configuration(spec, smoke, device_name):
    config = asdict(spec)
    if smoke:
        config["batch"] = min(256, spec.batch)
        config["pretrain_batch"] = min(256, spec.pretrain_batch)
        config["finetune_epochs"] = 1
        config["pretrain_epochs"] = int(spec.objective is not None)
    config.update(
        schema=1,
        seed=42,
        learning_rate=0.001,
        weight_decay=1e-4,
        architecture_source_sha256=sha256(model_source(spec.architecture)),
        device=device_name,
        torch_version=str(torch.__version__),
        amp=spec.architecture != "tabnet" and torch.cuda.is_bf16_supported(),
        smoke=smoke,
    )
    return config


def fit_fold(config, fold, fold_data, directory):
    print(f"{directory.name} fold {fold + 1} | {config['device']}", flush=True)
    seed = config["seed"] + fold
    torch.manual_seed(seed)
    model = new_model(config["architecture"], fold_data.numeric_count, fold_data.category_sizes)
    model.to(fold_data.train.device)
    pretrain_seconds, pretrain_history = 0.0, []
    if config["objective"]:
        pretrain_history, pretrain_seconds = train_epochs(
            model, fold_data, epochs=config["pretrain_epochs"], batch_size=config["pretrain_batch"],
            seed=seed + 5000, objective=config["objective"], use_amp=config["amp"],
        )
        path = directory / f"encoder_{fold + 1}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save({
            "state": {name: weight.detach().cpu() for name, weight in model.state_dict().items()},
            "numeric_count": fold_data.numeric_count,
            "category_sizes": fold_data.category_sizes,
        }, temporary)
        temporary.replace(path)
    finetune_history, finetune_seconds = train_epochs(
        model, fold_data, epochs=config["finetune_epochs"], batch_size=config["batch"],
        seed=seed + 10_000, objective=None, use_amp=config["amp"],
    )
    valid = probability(model, fold_data.valid, config["batch"])
    test = probability(model, fold_data.test, config["batch"])
    return valid, test, {
        "fold": fold + 1,
        "device": config["device"],
        "pretrain_seconds": pretrain_seconds,
        "finetune_seconds": finetune_seconds,
        "pretrain_loss_by_epoch": pretrain_history,
        "finetune_loss_by_epoch": finetune_history,
    }


def run_variants(run, names, device, device_name):
    configs = {}
    for name in names:
        directory = run.output / name
        directory.mkdir(exist_ok=True)
        config = configuration(VARIANTS[name], run.smoke, device_name)
        path = directory / "config.json"
        if path.exists() and read_json(path) != config:
            raise ValueError(f"Configuration mismatch for {directory}. Use a new output directory.")
        if not path.exists():
            write_json(path, config)
        configs[name] = config

    for fold, part in enumerate(run.partitions):
        fold_data = None
        for name, config in configs.items():
            directory = run.output / name
            checkpoint = directory / f"fold_{fold + 1}"
            details, predictions = checkpoint.with_suffix(".json"), checkpoint.with_suffix(".npz")
            encoder = directory / f"encoder_{fold + 1}.pt"
            if details.exists() and predictions.exists() and (not config["objective"] or encoder.exists()):
                print(f"{name} fold {fold + 1}: resume {read_json(details)['auc']:.6f}", flush=True)
                continue
            if fold_data is None:
                fold_data = FoldInputs(run, part, device)
            valid, test, record = fit_fold(config, fold, fold_data, directory)
            record["auc"] = metric_row(run.y[part["V"]], valid)["oof_auc"]
            write_npz(predictions, indexes=part["V"], valid=valid, test=test)
            write_json(details, record)
            print(f"{name} fold {fold + 1}: AUC {record['auc']:.6f}", flush=True)
        del fold_data

    for name in names:
        combine_folds(run, name)


def combine_folds(run, name):
    directory = run.output / name
    oof = np.full(len(run.y), np.nan, dtype=np.float32)
    test_by_fold = []
    records = []
    for fold, part in enumerate(run.partitions, 1):
        with np.load(directory / f"fold_{fold}.npz", allow_pickle=False) as saved:
            indexes, valid, test = (saved[key] for key in ("indexes", "valid", "test"))
            if not np.array_equal(indexes, part["V"]):
                raise ValueError(f"Wrong validation indices for {name} fold {fold}")
            if valid.shape != (len(indexes),) or test.shape != (len(run.test_ids),):
                raise ValueError(f"Wrong prediction shapes for {name} fold {fold}")
            if not (np.isfinite(valid).all() and np.isfinite(test).all()):
                raise ValueError(f"Non-finite predictions for {name} fold {fold}")
            oof[indexes] = valid
            test_by_fold.append(test)
        records.append(read_json(directory / f"fold_{fold}.json"))
    test_by_fold = np.stack(test_by_fold, axis=1)
    test = test_by_fold.mean(axis=1)
    metrics = {
        "variant": name, "status": "completed", **run.manifest,
        **metric_row(run.y, oof), "folds": records,
        "config": read_json(directory / "config.json"),
    }
    write_npz(directory / "predictions.npz", ids=run.ids, test_ids=run.test_ids,
              fold=run.folds, oof=oof, test=test, test_by_fold=test_by_fold)
    write_json(directory / "metrics.json", metrics)
    print(f"{name} complete: OOF AUC {metrics['oof_auc']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--smoke", action="store_true", help="Run reduced data, two folds and one epoch")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    device, device_name = cuda_device()
    output = args.output_dir or RESULTS / ("smoke" if args.smoke else "full")
    run = PlanRun(output, smoke=args.smoke)
    run_variants(run, args.variants, device, device_name)


if __name__ == "__main__":
    main()
