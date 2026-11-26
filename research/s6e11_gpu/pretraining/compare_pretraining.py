"""Check the OOF results and compare representation and prediction blending."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from run_pretraining import (
    PLAN, RESULTS, VARIANTS, FoldInputs, PlanRun, combine_folds, cuda_device,
    metric_row, new_model, read_json, sha256, write_json, write_npz,
)
from models import fit_predict


BASE = PLAN / "results" / "full"
FUSION = "xgb-d03-mlp-masked-embeddings"
CONTROLS = {
    "mlp-masked": "mlp-scratch",
    "mlp-scarf": "mlp-scratch",
    "ft-masked": "ft-scratch",
    "tabnet-masked": "tabnet-scratch",
}
BLEND_WEIGHT = 0.05


def load_scores(run, directory):
    path = directory / "predictions.npz"
    with np.load(path, allow_pickle=False) as payload:
        for key, expected in (("ids", run.ids), ("fold", run.folds), ("test_ids", run.test_ids)):
            if not np.array_equal(payload[key], expected):
                raise ValueError(f"{path} has a different {key} order")
        valid, test = payload["oof"], payload["test"]
    for scores, size in ((valid, len(run.y)), (test, len(run.test_ids))):
        if (scores.shape != (size,) or not np.isfinite(scores).all()
                or np.any(scores < 0) or np.any(scores > 1)):
            raise ValueError(f"Invalid predictions in {path}")
    recorded = read_json(directory / "metrics.json")
    if abs(roc_auc_score(run.y, valid) - recorded["oof_auc"]) > 1e-9:
        raise ValueError(f"Stored AUC does not match predictions: {path}")
    for key in ("fold_hash", "early_stop_hash", "inputs", "smoke"):
        if recorded[key] != run.manifest[key]:
            raise ValueError(f"Stored {key} does not match this run: {path}")
    return valid, test, recorded


def embeddings(model, data):
    model.eval()
    with torch.inference_mode():
        return tuple(
            np.concatenate([model.encode(batch).float().cpu().numpy() for batch in values.split(4096)])
            for values in (data.train, data.valid, data.test)
        )


def run_fusion(run, device):
    directory = run.output / FUSION
    directory.mkdir(exist_ok=True)
    source = run.output / "mlp-masked"
    source_config = read_json(source / "config.json")
    base_config = read_json(BASE / "XGB-D03" / "config.json")
    base_metrics = read_json(BASE / "XGB-D03" / "metrics.json")
    if not run.smoke:
        load_scores(run, BASE / "XGB-D03")
    rounds = [20] * len(run.partitions) if run.smoke else [item["rounds"] for item in base_metrics["folds"]]
    config = {
        "architecture": "xgb", "objective": "binary:logistic", "parent": "XGB-D03",
        "original_groups": base_config["groups"], "embedding_source": "mlp-masked",
        "encoder_before_supervised_finetune": True, "encoder_config_sha256": sha256(source / "config.json"),
        "encoder_states_sha256": [sha256(source / f"encoder_{fold}.pt") for fold in range(1, len(rounds) + 1)],
        "rounds_by_fold": rounds, "base_config": base_config,
    }
    path = directory / "config.json"
    if path.exists() and read_json(path) != config:
        raise ValueError(f"Stored configuration differs: {path}")
    write_json(path, config)
    for fold, (part, trees) in enumerate(zip(run.partitions, rounds, strict=True), 1):
        artifact = directory / f"fold_{fold}"
        if artifact.with_suffix(".json").exists() and artifact.with_suffix(".npz").exists():
            print(f"{FUSION} fold {fold}: resumed", flush=True)
            continue
        start = perf_counter()
        inputs = FoldInputs(run, part, device)
        weights = torch.load(source / f"encoder_{fold}.pt", map_location="cpu", weights_only=True)
        if (weights["numeric_count"], weights["category_sizes"]) != (inputs.numeric_count, inputs.category_sizes):
            raise ValueError(f"Preprocessing differs from saved encoder for fold {fold}")
        model = new_model(source_config["architecture"], inputs.numeric_count, inputs.category_sizes)
        model.load_state_dict(weights["state"])
        model.to(device)
        vectors = embeddings(model, inputs)
        del model, inputs
        matrices = run.builder.make(
            part["T"], (part["V"], run.test_indices), groups=tuple(base_config["groups"]),
            smoothing=base_config["smoothing"], seed=base_config["encoder_seed"], logit_te=False,
        )
        matrices = tuple(np.column_stack((matrix, vector)) for matrix, vector in zip(matrices, vectors, strict=True))
        (valid, test), details = fit_predict(
            "xgb", matrices[0], run.y[part["T"]], matrices[1:], params=base_config["params"],
            seed=base_config["model_seed"], rounds=trees,
        )
        metrics = {
            "fold": fold, "auc": roc_auc_score(run.y[part["V"]], valid), "rounds": trees,
            "device": details["device"], "extra_columns": vectors[0].shape[1],
            "pretrain_seconds": 0.0, "finetune_seconds": perf_counter() - start,
        }
        write_npz(artifact.with_suffix(".npz"), indexes=part["V"], valid=valid, test=test)
        write_json(artifact.with_suffix(".json"), metrics)
        print(f"{FUSION} fold {fold}: AUC {metrics['auc']:.6f} | {details['device']}", flush=True)
    combine_folds(run, FUSION)


def result_row(run, scores, source=None):
    candidate = metric_row(run.y, scores)
    candidate["fold_auc"] = [roc_auc_score(run.y[part["V"]], scores[part["V"]]) for part in run.partitions]
    if source is not None:
        candidate["delta"] = candidate["oof_auc"] - roc_auc_score(run.y, source)
        candidate["fold_delta"] = [
            auc - roc_auc_score(run.y[part["V"]], source[part["V"]])
            for auc, part in zip(candidate["fold_auc"], run.partitions, strict=True)
        ]
    return candidate


def render_report(run):
    scores = {name: load_scores(run, run.output / name) for name in (*VARIANTS, FUSION)}
    final, final_test, final_metrics = load_scores(run, BASE / "final")
    final, final_test = final.astype(np.float64), final_test.astype(np.float64)
    tree, _, tree_metrics = load_scores(run, BASE / "XGB-D03")
    lines = [
        "# S6E11 · Kết quả tiền huấn luyện trên RTX 4060", "",
        f"Đủ {len(run.y):,} dòng train, {len(run.test_ids):,} dòng test và 5 fold đã chốt. "
        "Mỗi encoder tiền huấn luyện trên phần train của fold. Số epoch cố định trước, "
        "phần validation ngoài chỉ dùng để tính điểm. Các đầu phân loại dùng BCE.", "",
        "Mỗi hướng tiền huấn luyện chạy 3 epoch. MLP, FT-Transformer và TabNet tinh chỉnh lần lượt "
        "6, 5 và 4 epoch, bằng ngân sách huấn luyện có nhãn của đối chứng tương ứng. MLP và "
        "FT-Transformer che 7/24 cột. SCARF thay 14/24 cột. TabNet dùng tỷ lệ che 20% và đặt sàn "
        "phương sai 0,1 trong hàm khôi phục để giảm ảnh hưởng của các nhóm hiếm.", "",
        f"Mốc XGB-D03 **{tree_metrics['oof_auc']:.6f}** và FINAL **{final_metrics['oof_auc']:.6f}**. "
        "Δ AUC của mỗi mô hình tiền huấn luyện được tính so với cùng kiến trúc khởi tạo ngẫu nhiên.", "",
        "## Mô hình riêng", "",
        "| Cấu hình | OOF ROC AUC | Δ AUC | Fold tăng | AP | Log-loss | Giây pretrain | Giây phân loại |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    all_metrics = {"references": {"XGB-D03": tree_metrics["oof_auc"], "FINAL": final_metrics["oof_auc"]},
                   "standalone": {}, "prediction_blend": {}}
    for name in VARIANTS:
        valid, _, record = scores[name]
        comparator = CONTROLS.get(name)
        result = result_row(run, valid, scores[comparator][0] if comparator else None)
        result["control"] = comparator
        result["pretrain_seconds"] = sum(fold["pretrain_seconds"] for fold in record["folds"])
        result["finetune_seconds"] = sum(fold["finetune_seconds"] for fold in record["folds"])
        all_metrics["standalone"][name] = result
        improvement = f"{result['delta']:+.6f}" if comparator else "—"
        wins = f"{sum(change > 0 for change in result['fold_delta'])}/5" if comparator else "—"
        lines.append(f"| [{name}](full/{name}/metrics.json) | {result['oof_auc']:.6f} | {improvement} | "
                     f"{wins} | {result['average_precision']:.6f} | {result['log_loss']:.6f} | "
                     f"{result['pretrain_seconds']:.1f} | {result['finetune_seconds']:.1f} |")
    lines += ["", "## AUC theo fold", "", "| Cấu hình | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, row in all_metrics["standalone"].items():
        lines.append(f"| `{name}` | " + " | ".join(f"{value:.6f}" for value in row["fold_auc"]) + " |")
    lines += ["", "## Gộp dự đoán với FINAL", "",
              "Tỷ lệ cố định 95% FINAL và 5% đầu ra mô hình mới. Cùng ID và cùng phép chia fold. "
              "Đối chứng scratch cũng được gộp ở tỷ lệ 95/5 để đối chiếu tiền huấn luyện.", "",
              "| Thành viên mới | OOF ROC AUC | Δ so FINAL | Δ so gộp scratch | Fold tăng so FINAL | AP |",
              "|---|---:|---:|---:|---:|---:|"]
    blend_dir = run.output / "fixed-blends"
    blend_dir.mkdir(exist_ok=True)
    blend_controls = {
        name: roc_auc_score(run.y, (1 - BLEND_WEIGHT) * final + BLEND_WEIGHT * scores[name][0].astype(np.float64))
        for name in sorted(set(CONTROLS.values()))
    }
    for name, control in CONTROLS.items():
        valid, test, _ = scores[name]
        valid = (1 - BLEND_WEIGHT) * final + BLEND_WEIGHT * valid.astype(np.float64)
        test = (1 - BLEND_WEIGHT) * final_test + BLEND_WEIGHT * test.astype(np.float64)
        result = result_row(run, valid, final)
        result["scratch_blend_auc"] = blend_controls[control]
        result["delta_vs_scratch_blend"] = result["oof_auc"] - blend_controls[control]
        all_metrics["prediction_blend"][name] = result
        write_npz(blend_dir / f"final-95-{name}-05.npz", ids=run.ids, test_ids=run.test_ids,
                  fold=run.folds, oof=valid, test=test)
        wins = sum(change > 0 for change in result["fold_delta"])
        lines.append(f"| `{name}` | {result['oof_auc']:.6f} | {result['delta']:+.6f} | "
                     f"{result['delta_vs_scratch_blend']:+.6f} | {wins}/5 | {result['average_precision']:.6f} |")
    fusion = result_row(run, scores[FUSION][0], tree)
    all_metrics["feature_blend"] = fusion
    fusion_wins = sum(change > 0 for change in fusion["fold_delta"])
    lines += ["", "## Thêm embedding vào XGB-D03", "",
              f"Thêm {scores[FUSION][2]['folds'][0]['extra_columns']} cột từ encoder MLP đã đóng băng trước tinh chỉnh. "
              "Các nhóm đầu vào gốc và số vòng boosting từng fold được giữ theo XGB-D03.", "",
              f"[{FUSION}](full/{FUSION}/metrics.json) đạt **{fusion['oof_auc']:.6f}**, "
              f"Δ **{fusion['delta']:+.6f}** so với XGB-D03, tăng ở {fusion_wins}/5 fold. "
              f"So với FINAL, Δ **{fusion['oof_auc'] - final_metrics['oof_auc']:+.6f}**.", "",
              "Các so sánh dùng cùng 5 fold khảo sát. Số giây là tổng thời gian huấn luyện theo fold. "
              "Embedding dùng encoder học từ dữ liệu gốc của fold, trước khi mạng học nhãn gian lận.", "",
              "## Chạy lại", "",
              "Chạy từ thư mục dự án với môi trường Python hỗ trợ CUDA đã dùng cho các lượt GPU. "
              "TabNet dùng `pytorch-tabnet==4.1.0`. Mỗi quá trình tiếp tục từ các fold đã lưu nếu cấu hình khớp.", "",
              "```powershell", "python s6e11_gpu/pretraining/run_pretraining.py",
              "python s6e11_gpu/pretraining/compare_pretraining.py fusion",
              "python s6e11_gpu/pretraining/compare_pretraining.py report", "```", ""]
    summary = run.output.parent / "summary.md"
    temporary = summary.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(summary)
    write_json(summary.with_suffix(".json"), all_metrics)
    print(f"Đã kiểm tra dự đoán OOF và ghi {summary}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fusion", "report"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "report" and args.smoke:
        parser.error("report compares full-data runs")
    run = PlanRun(RESULTS / ("smoke" if args.smoke else "full"), smoke=args.smoke)
    if args.stage == "fusion":
        device, _ = cuda_device()
        run_fusion(run, device)
    else:
        render_report(run)


if __name__ == "__main__":
    main()
