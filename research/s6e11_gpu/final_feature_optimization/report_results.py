"""Verify the XGBoost branches and compare original and modified FINAL predictions."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PLAN = HERE.parent / "plan_runs"
REFERENCE = PLAN / "results" / "full"
RESULTS = HERE / "results"
EXPERIMENTS = RESULTS / "full"
ADDED_COLUMNS = {"PAIR-TE": 8, "RARE-TE": 2, "PAIR-RARE-TE": 10}
MIN_DELTA = 0.000100
MIN_WINS = 4
BASE_COLUMNS = 109

sys.path.insert(0, str(PLAN))
from run_plan import metric_row, read_json, sha256, write_json


def confirm_numbers(recorded, expected, description):
    if not np.allclose(recorded, expected, rtol=0, atol=1e-10):
        raise ValueError(f"Saved {description} differs from the recomputed values")


def evaluate(name, oof, y, folds):
    if oof.shape != y.shape or not np.all(np.isfinite(oof) & (oof >= 0) & (oof <= 1)):
        raise ValueError(f"Invalid OOF scores for {name}")
    return {
        "name": name,
        **metric_row(y, oof),
        "folds": [
            {"fold": index + 1, "auc": float(roc_auc_score(y[folds == index], oof[folds == index]))}
            for index in range(5)
        ],
    }


def load_run(directory, name, ids, y, folds, manifest, *, gpu=False, width=None):
    metrics = read_json(directory / "metrics.json")
    if metrics["name"] != name or metrics["status"] != "completed":
        raise ValueError(f"Unexpected run name or status in {directory}")
    for key in ("smoke", "rows", "test_rows", "fold_hash", "early_stop_hash", "inputs", "models_use_llm"):
        if metrics[key] != manifest[key]:
            raise ValueError(f"{name}: different {key}")

    with np.load(directory / "predictions.npz", allow_pickle=False) as predictions:
        np.testing.assert_array_equal(predictions["ids"], ids)
        np.testing.assert_array_equal(predictions["fold"], folds)
        oof = predictions["oof"]
    checked = evaluate(name, oof, y, folds)
    for key in ("oof_auc", "average_precision", "log_loss"):
        confirm_numbers(metrics[key], checked[key], f"{name} {key}")
    records = metrics["folds"]
    if [entry["fold"] for entry in records] != [1, 2, 3, 4, 5]:
        raise ValueError(f"Unexpected fold metadata for {name}")
    for detail, expected in zip(records, checked["folds"], strict=True):
        confirm_numbers(detail["auc"], expected["auc"], f"{name} fold {detail['fold']} AUC")
        if gpu:
            for stage in (detail, detail["early"]):
                if ("cuda:" not in stage["device"] or "RTX 4060" not in stage["device"]
                        or stage["rounds"] <= 0):
                    raise ValueError(f"Unexpected GPU or number of trees for {name} fold {detail['fold']}")
        if width is not None and (detail["feature_count"], detail["extra_feature_count"]) != (width, width - BASE_COLUMNS):
            raise ValueError(f"Unexpected feature width for {name} fold {detail['fold']}")
    return checked, metrics, oof


def compare(run, baseline):
    per_fold = [new["auc"] - old["auc"] for new, old in zip(run["folds"], baseline["folds"], strict=True)]
    return {
        **run,
        "delta_auc": run["oof_auc"] - baseline["oof_auc"],
        "delta_ap": run["average_precision"] - baseline["average_precision"],
        "fold_deltas": per_fold,
        "fold_wins": sum(change > 0 for change in per_fold),
    }


def score_rows(reference, candidates, *, tree=False):
    width_header = " Số cột |" if tree else ""
    width_alignment = "---:|" if tree else ""
    width = f" {BASE_COLUMNS} |" if tree else ""
    lines = [
        f"| Cấu hình |{width_header} OOF ROC AUC | Δ AUC | AP | Δ AP | Log-loss | Fold tăng |",
        f"|---|{width_alignment}---:|---:|---:|---:|---:|---:|",
        f"| `{reference['name']}` |{width} {reference['oof_auc']:.6f} | — | "
        f"{reference['average_precision']:.6f} | — | {reference['log_loss']:.6f} | — |",
    ]
    for candidate in candidates:
        width = f" {BASE_COLUMNS + candidate['added_columns']} |" if tree else ""
        lines.append(
            f"| `{candidate['name']}` |{width} {candidate['oof_auc']:.6f} | {candidate['delta_auc']:+.6f} | "
            f"{candidate['average_precision']:.6f} | {candidate['delta_ap']:+.6f} | "
            f"{candidate['log_loss']:.6f} | {candidate['fold_wins']}/5 |"
        )
    return lines


def fold_rows(reference, candidates):
    names = " | ".join(f"Δ {candidate['name']}" for candidate in candidates)
    lines = [f"| Fold | AUC FINAL | {names} |", "|---:|---:|" + "---:|" * len(candidates)]
    for index, fold in enumerate(reference["folds"]):
        changes = " | ".join(f"{run['fold_deltas'][index]:+.6f}" for run in candidates)
        lines.append(f"| {fold['fold']} | {fold['auc']:.6f} | {changes} |")
    return lines


def render(original_tree, original_final, replay, trees, swaps, anchored):
    lines = [
        "# Tối ưu đặc trưng cho tổ hợp FINAL", "",
        "Ba nhánh XGBoost chạy riêng trên RTX 4060 với 699.960 dòng train và cùng 5 fold của `XGB-D03`. "
        "Trong mỗi fold, mô hình chọn số cây trên phần giữ lại bên trong train rồi huấn luyện lại trên toàn bộ phần train của fold.", "",
        "| Phiên bản | Cách tạo đặc trưng | Số cột mới |",
        "|---|---|---:|",
        "| `PAIR-TE` | Target Encoding cho 8 tương tác thuộc hai nhóm P1/P2 | 8 |",
        "| `RARE-TE` | Target Encoding hai tầng cho số tiền và số dư tài khoản | 2 |",
        "| `PAIR-RARE-TE` | Gộp cả hai nhóm trên | 10 |", "",
        "Với khóa hiếm, mỗi biến có tối đa 32 khoảng phân vị được tính trên phần đang dùng để huấn luyện. "
        "Tỷ lệ gian lận của khoảng được làm trơn về tỷ lệ chung với hệ số 50. "
        "Tỷ lệ của giá trị chính xác đến cent được làm trơn về tỷ lệ của khoảng với hệ số 20. "
        "Giá trị chưa gặp dùng tỷ lệ của khoảng. Thống kê mã hóa một dòng train được tính từ các dòng ở fold mã hóa khác.", "",
        "## Điểm từng nhánh XGBoost", "",
        *score_rows(original_tree, trees, tree=True),
        "", "## Thay nhánh XGBoost trong FINAL", "",
        "Mỗi biến thể thay `XGB-D03` bằng nhánh XGBoost mới. `GLM-NEWTON`, phép chuẩn hóa và trọng số logistic "
        "ở tầng gộp giữ theo tổ hợp gốc của từng fold. Tầng gộp được tái dựng từ 25 tệp dự đoán ngoài fold bên trong của "
        "`GLM-NEWTON` và 25 tệp của `XGB-D03`, dùng dữ liệu train của từng fold ngoài. "
        f"Khi tái dựng `FINAL`, sai khác xác suất lớn nhất so với bản lưu là **{replay['max_abs_probability_difference']:.2g}** "
        f"và OOF ROC AUC khớp ở **{replay['oof_auc']:.6f}**.", "",
        *score_rows(original_final, swaps),
        "", *fold_rows(original_final, swaps),
        "", "## Trung bình 50/50 với FINAL gốc", "",
        "Lấy trung bình xác suất OOF của `FINAL` gốc và từng biến thể thay nhánh theo tỷ lệ cố định 50/50, "
        "sau khi đối chiếu thứ tự ID và fold. Phép thử này dùng các dự đoán đã tính ở trên.", "",
        *score_rows(original_final, anchored),
        "", *fold_rows(original_final, anchored),
    ]
    tested_blends = [*swaps, *anchored]
    best = max(tested_blends, key=lambda run: run["oof_auc"])
    passing = [run["name"] for run in tested_blends
               if run["delta_auc"] >= MIN_DELTA and run["fold_wins"] >= MIN_WINS]
    lines.extend([
        "", "## Đánh giá", "",
        f"Trong sáu phép thử, `{best['name']}` có AUC cao nhất với **{best['oof_auc']:.6f}**, "
        f"chênh **{best['delta_auc']:+.6f}** so với `FINAL` và tăng ở **{best['fold_wins']}/5 fold**.", "",
        "Mốc sàng lọc là tăng ít nhất **0.000100 AUC** và tăng ở ít nhất **4/5 fold**. "
        "Đây là so sánh trên các fold phát triển của dự án, với các trọng số gộp cố định.", "",
    ])
    if passing:
        names = ", ".join(f"`{name}`" for name in passing)
        lines.append(f"{names} đạt mốc sàng lọc và là ứng viên cho bước xác nhận tiếp theo.")
    else:
        lines.append("Cả sáu phép thử đều chưa qua mốc sàng lọc. Kết quả ủng hộ dùng `FINAL` gốc.")
    lines.extend([
        "",
        "OOF là dự đoán ngoài fold. AP (Average Precision) tổng hợp độ chính xác và độ bao phủ ở các mức xếp hạng. "
        "Log-loss đánh giá chất lượng xác suất dự đoán, với giá trị thấp hơn là tốt hơn.", "",
        "[Số liệu đối chiếu](comparison.json) · [Chi tiết tái dựng FINAL](frozen_reference.json) · "
        "[Sàng lọc đặc trưng từ EDA](../../../eda_review/ket_qua_eda.md) · "
        "[Kế hoạch và mốc sàng lọc](../../../plan.md)", "",
    ])
    return "\n".join(lines)


def main():
    names = [f"{prefix}-{suffix}" for suffix in ADDED_COLUMNS for prefix in ("XGB-D03", "FINAL")]
    run_folders = [REFERENCE / "XGB-D03", REFERENCE / "final", *(EXPERIMENTS / name for name in names)]
    required = [RESULTS / "frozen_reference.json", REFERENCE / "manifest.json"]
    required.extend(folder / filename for folder in run_folders for filename in ("metrics.json", "predictions.npz"))
    missing = [str(path.relative_to(PROJECT)) for path in required if not path.is_file()]
    if missing:
        sys.exit("Kết quả đầy đủ chưa sẵn sàng. Còn thiếu:\n" + "\n".join(missing))

    manifest = read_json(REFERENCE / "manifest.json")
    if manifest["smoke"] or manifest["models_use_llm"]:
        raise ValueError("Expected the approved full-data run without LLMs")
    for filename, digest in manifest["inputs"].items():
        if sha256(PROJECT / filename) != digest:
            raise ValueError(f"Official data file changed: {filename}")
    data = pd.read_csv(PROJECT.parent / 'data' / "train.csv", usecols=["id", "is_fraud"])
    ids, y = data["id"].to_numpy(), data["is_fraud"].to_numpy(dtype=np.int8)
    if len(y) != manifest["rows"] or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Unexpected training data length or labels")
    with np.load(REFERENCE / "XGB-D03" / "predictions.npz", allow_pickle=False) as values:
        folds = values["fold"]
    if folds.shape != y.shape or not np.array_equal(np.unique(folds), np.arange(5)):
        raise ValueError("Expected five official folds in the original run")

    contract = (ids, y, folds, manifest)
    original_tree, _, _ = load_run(REFERENCE / "XGB-D03", "XGB-D03", *contract, gpu=True)
    original_final, _, final_oof = load_run(REFERENCE / "final", "FINAL", *contract)
    replay = read_json(RESULTS / "frozen_reference.json")
    confirm_numbers(replay["oof_auc"], original_final["oof_auc"], "FINAL replay AUC")
    if not 0 <= replay["max_abs_probability_difference"] <= 1e-5:
        raise ValueError("FINAL replay probability difference exceeds tolerance")

    trees, swaps, anchored = [], [], []
    for suffix, extra_width in ADDED_COLUMNS.items():
        tree_name, blend_name = f"XGB-D03-{suffix}", f"FINAL-{suffix}"
        tree, trained, _ = load_run(
            EXPERIMENTS / tree_name, tree_name, *contract, gpu=True, width=BASE_COLUMNS + extra_width,
        )
        blend, evaluated, blend_oof = load_run(EXPERIMENTS / blend_name, blend_name, *contract)
        trees.append({
            **compare(tree, original_tree),
            "added_columns": extra_width,
            "rounds": [record["rounds"] for record in trained["folds"]],
        })
        swap = compare(blend, original_final)
        confirm_numbers(evaluated["delta_auc"], swap["delta_auc"], f"{blend_name} Δ AUC")
        confirm_numbers(evaluated["delta_ap"], swap["delta_ap"], f"{blend_name} Δ AP")
        if evaluated["fold_wins"] != swap["fold_wins"]:
            raise ValueError(f"Unexpected fold win count for {blend_name}")
        swaps.append(swap)
        anchor_name = f"50/50 FINAL + {suffix}"
        anchor = evaluate(anchor_name, 0.5 * final_oof + 0.5 * blend_oof, y, folds)
        anchored.append(compare(anchor, original_final))

    comparison = {"reference_tree": original_tree, "reference_final": original_final,
                  "branches": trees, "swaps": swaps, "anchored": anchored}
    output = RESULTS / "ket_qua_toi_uu_dac_trung_final.md"
    staged = output.with_suffix(".md.tmp")
    staged.write_text(render(original_tree, original_final, replay, trees, swaps, anchored), encoding="utf-8")
    write_json(RESULTS / "comparison.json", comparison)
    staged.replace(output)
    print(output)


if __name__ == "__main__":
    main()
