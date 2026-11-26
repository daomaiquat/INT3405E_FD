"""Verify saved S6E11 feature ablations and report their comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
PLAN_RUNS = PROJECT / "s6e11_gpu" / "plan_runs"
REFERENCE = PLAN_RUNS / "results" / "full"
RESULTS = HERE / "results"
FULL = RESULTS / "full"
BASE = "XGB-D03"
BASE_COLUMNS = 109
N_FOLDS = 5
MIN_DELTA = 0.0001
MIN_WINS = 4
VARIANTS = {
    "XGB-D03-CHANNEL-6H": 20,
    "XGB-D03-VELOCITY-RESIDUAL": 1,
}

sys.path.insert(0, str(PLAN_RUNS))
from run_plan import assign_folds, early_stop_assignments, frame_hash, metric_row, read_json, sha256, write_json


def require_equal(saved, expected, label):
    if not np.array_equal(saved, expected):
        raise ValueError(f"{label} differs from the official reference")


def require_close(saved, expected, label):
    if not np.allclose(saved, expected, rtol=0, atol=1e-10):
        raise ValueError(f"{label} differs from the recomputed value")


def load_contract():
    manifest = read_json(REFERENCE / "manifest.json")
    if manifest["smoke"] or manifest["models_use_llm"]:
        raise ValueError("Expected the original full-data, non-LLM run")
    for filename in ("train.csv", "test.csv", "sample_submission.csv"):
        if sha256(PROJECT / filename) != manifest["inputs"][filename]:
            raise ValueError(f"{filename} no longer matches the original run")
    train = pd.read_csv(PROJECT.parent / 'data' / "train.csv", usecols=["id", "is_fraud"])
    test = pd.read_csv(PROJECT.parent / 'data' / "test.csv", usecols=["id"])
    sample = pd.read_csv(PROJECT.parent / 'data' / "sample_submission.csv", usecols=["id"])
    require_equal(sample["id"].to_numpy(), test["id"].to_numpy(), "Submission IDs")
    require_equal([len(train), len(test)], [manifest["rows"], manifest["test_rows"]], "Data row counts")
    config = read_json(PROJECT.parent / "configs" / "default.json")
    assignment = assign_folds(train, "is_fraud", N_FOLDS, config["seed"])
    early = early_stop_assignments(train, assignment, config, N_FOLDS)
    if (frame_hash(assignment) != manifest["fold_hash"]
            or frame_hash(early) != manifest["early_stop_hash"]):
        raise ValueError("The reconstructed fold tables differ from the original run")
    return manifest, train, test, assignment["fold"].to_numpy()


def valid_scores(scores, shape, label):
    if scores.shape != shape or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"Unexpected probability array for {label}")


def checked_feature_manifest():
    manifest = read_json(FULL / "feature_manifest.json")
    if manifest["baseline"] != BASE or manifest["baseline_columns"] != BASE_COLUMNS:
        raise ValueError("Unexpected baseline in the feature manifest")
    if {name: feature["expected_columns"] for name, feature in manifest["variants"].items()} != VARIANTS:
        raise ValueError("Unexpected variants or feature widths in the feature manifest")
    paths = {
        "baseline_config_sha256": REFERENCE / BASE / "config.json",
        "baseline_predictions_sha256": REFERENCE / BASE / "predictions.npz",
        "runner_sha256": HERE / "run_ablation.py",
    }
    for key, path in paths.items():
        if sha256(path) != manifest[key]:
            raise ValueError(f"{path} differs from the feature manifest")
    for name in ("run_plan.py", "features.py", "models.py"):
        if sha256(PLAN_RUNS / name) != manifest["pipeline_hashes"][name]:
            raise ValueError(f"{name} differs from the feature manifest")
    return manifest


def verify_xgb_fold(folder, number, details, train_ids, valid, oof, test_by_fold, baseline=None):
    path = folder / f"fold_{number}.json"
    if read_json(path) != details:
        raise ValueError(f"Per-fold metadata mismatch in {path}")
    with np.load(folder / f"fold_{number}.npz", allow_pickle=False) as saved:
        require_equal(saved["ids"], train_ids[valid], f"Validation IDs in {path}")
        require_equal(saved["valid"], oof[valid], f"Validation scores in {path}")
        require_equal(saved["test"], test_by_fold[:, number - 1], f"Test scores in {path}")
    require_equal([details["train_rows"], details["valid_rows"]],
                  [len(train_ids) - valid.sum(), valid.sum()], f"Training sizes in {path}")
    if baseline is not None:
        added = VARIANTS[folder.name]
        require_equal([details["base_feature_count"], details["extra_feature_count"], details["feature_count"]],
                      [BASE_COLUMNS, added, BASE_COLUMNS + added], f"Feature widths in {path}")
    for stage, row in (("early", details["early"]), ("refit", details)):
        if not row["device"].lower().startswith("cuda:") or "RTX 4060" not in row["device"]:
            raise ValueError(f"Unexpected GPU for {stage} in {path}")
        if not row["settings"]["device"].startswith("cuda") or row["rounds"] < 1:
            raise ValueError(f"Unexpected GPU settings or tree count in {path}")
        if baseline is not None:
            original = baseline["early"] if stage == "early" else baseline
            if row["settings"] != original["settings"] or row["version"] != original["version"]:
                raise ValueError(f"XGBoost settings differ in {stage} of {path}")
    if details["rounds"] != details["early"]["rounds"]:
        raise ValueError(f"Refit did not use the selected tree count in {path}")


def checked_run(folder, contract, *, xgb, baseline=None):
    manifest, train, test, folds = contract
    metrics = read_json(folder / "metrics.json")
    if metrics["status"] != "completed" or any(metrics[key] != value for key, value in manifest.items()):
        raise ValueError(f"Incomplete run or mismatched manifests in {folder}")
    train_ids, test_ids = train["id"].to_numpy(), test["id"].to_numpy()
    if xgb:
        config = read_json(folder / "config.json")
        if config != metrics["config"] or config["name"] != folder.name:
            raise ValueError(f"Saved specification mismatch in {folder}")
        if baseline is not None:
            original = baseline["config"]
            shared = original.keys() - {"name", "parent"}
            if config["parent"] != BASE or any(config[key] != original[key] for key in shared):
                raise ValueError(f"XGB-D03 baseline configuration changed in {folder}")

    with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
        require_equal(saved["ids"], train_ids, f"Train IDs in {folder}")
        require_equal(saved["test_ids"], test_ids, f"Test IDs in {folder}")
        require_equal(saved["fold"], folds, f"Fold assignments in {folder}")
        oof, predictions = saved["oof"], saved["test"]
        valid_scores(oof, (len(train_ids),), folder.name)
        valid_scores(predictions, (len(test_ids),), folder.name)
        if xgb:
            test_by_fold = saved["test_by_fold"]
            valid_scores(test_by_fold, (len(test_ids), N_FOLDS), folder.name)
            require_close(predictions, test_by_fold.mean(axis=1), f"Test fold mean in {folder}")

    labels = train["is_fraud"].to_numpy(dtype=np.int8)
    scores = metric_row(labels, oof)
    for key, value in scores.items():
        require_close(metrics[key], value, f"{key} in {folder}")
    require_equal([entry["fold"] for entry in metrics["folds"]], list(range(1, N_FOLDS + 1)),
                  f"Fold numbering in {folder}")
    fold_aucs, rounds = [], []
    for index, details in enumerate(metrics["folds"]):
        valid = folds == index
        score = float(roc_auc_score(labels[valid], oof[valid]))
        require_close(details["auc"], score, f"Fold {index + 1} AUC in {folder}")
        fold_aucs.append(score)
        if xgb:
            original_fold = None if baseline is None else baseline["folds"][index]
            verify_xgb_fold(folder, index + 1, details, train_ids, valid, oof, test_by_fold, original_fold)
            rounds.append(details["rounds"])
    return {"name": metrics["name"], **scores, "fold_aucs": fold_aucs, "rounds": rounds,
            "seconds": metrics.get("seconds"), "config": metrics["config"], "folds": metrics["folds"]}


def compare(candidate, reference):
    delta = candidate["oof_auc"] - reference["oof_auc"]
    deltas = [new - original for new, original in zip(candidate["fold_aucs"], reference["fold_aucs"], strict=True)]
    wins = sum(change > 0 for change in deltas)
    return {**candidate, "delta_auc": delta, "delta_ap": candidate["average_precision"] - reference["average_precision"],
            "delta_log_loss": candidate["log_loss"] - reference["log_loss"], "fold_deltas": deltas,
            "fold_wins": wins, "passed_screen": delta >= MIN_DELTA and wins >= MIN_WINS}


def make_report(baseline, final, variants):
    lines = [
        "# S6E11 · Thử thêm hai đặc trưng vào XGB-D03", "",
        "Cả hai cấu hình chạy trên NVIDIA RTX 4060 với 699.960 dòng train, 299.984 dòng test và cùng 5 fold với `XGB-D03`. "
        "Mỗi cấu hình thêm riêng đặc trưng của mình, giữ cấu hình XGBoost gốc và dùng cùng cách chọn số cây. "
        "Ở mỗi fold, mô hình chọn số cây trên phần giữ lại bên trong train, rồi huấn luyện lại trên toàn bộ phần train của fold.", "",
        "- **Kênh giao dịch × khung giờ 6 tiếng.** Ghép kênh giao dịch với một trong bốn khung giờ. Mỗi cặp tạo một cột 0/1 (one-hot).",
        "- **Velocity so với nhịp giao dịch tương tự.** Lấy `velocity_score` trừ trung vị của nhóm có số giao dịch và thời gian kể từ giao dịch trước tương tự. "
        "Các mốc chia số giao dịch là 3, 6, 10. Các mốc chia thời gian là 0,25, 1, 6 và 24 giờ. "
        "Với nhóm dưới 250 dòng, dùng trung vị chung. Mọi trung vị đều tính trên phần dữ liệu dùng để huấn luyện ở bước tương ứng.", "",
        "## Kết quả gộp", "",
        "| Cấu hình | Số cột | OOF ROC AUC | Δ AUC với XGB-D03 | AP | Δ AP | Log-loss | Số fold tăng |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| `XGB-D03` gốc | {BASE_COLUMNS} | {baseline['oof_auc']:.6f} | — | {baseline['average_precision']:.6f} | — | {baseline['log_loss']:.6f} | — |",
    ]
    for result in variants:
        lines.append(f"| `{result['name']}` | {BASE_COLUMNS + VARIANTS[result['name']]} | {result['oof_auc']:.6f} | {result['delta_auc']:+.6f} | "
                     f"{result['average_precision']:.6f} | {result['delta_ap']:+.6f} | {result['log_loss']:.6f} | {result['fold_wins']}/5 |")
    lines.extend([
        "", f"Mô hình tổ hợp `FINAL` hiện có OOF ROC AUC **{final['oof_auc']:.6f}** và AP **{final['average_precision']:.6f}**. "
        "Các Δ trong bảng dùng `XGB-D03` gốc làm đối chứng trực tiếp.", "",
        "OOF là điểm tính từ dự đoán trên phần dữ liệu giữ lại của từng fold. AP (Average Precision) là chỉ số tổng hợp khả năng tìm đúng các ca gian lận ở các ngưỡng xếp hạng. "
        "Log-loss thấp hơn nghĩa là dự đoán xác suất phù hợp với nhãn hơn.", "",
        "## Chênh lệch AUC từng fold và số cây", "",
        "| Fold | AUC gốc | AUC khung giờ | Δ khung giờ | AUC velocity | Δ velocity | Cây gốc | Cây khung giờ | Cây velocity |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    hour, velocity = variants
    for index in range(N_FOLDS):
        lines.append(f"| {index + 1} | {baseline['fold_aucs'][index]:.6f} | "
                     f"{hour['fold_aucs'][index]:.6f} | {hour['fold_deltas'][index]:+.6f} | "
                     f"{velocity['fold_aucs'][index]:.6f} | {velocity['fold_deltas'][index]:+.6f} | "
                     f"{baseline['rounds'][index]} | {hour['rounds'][index]} | {velocity['rounds'][index]} |")
    lines.extend(["", "## Quyết định", "",
                  "Mốc sàng lọc là tăng ít nhất **0.000100 OOF ROC AUC** và cải thiện ở ít nhất **4/5 fold**.", ""])
    passed = [f"`{result['name']}`" for result in variants if result["passed_screen"]]
    if passed:
        lines.append(f"{', '.join(passed)} qua mốc sàng lọc. Các kết quả dùng cùng 5 fold phát triển. "
                     "Ưu tiên kiểm định lại trên phép chia riêng trước khi đưa vào `FINAL`.")
    else:
        lines.append("Cả hai ứng viên đều dưới mốc sàng lọc. Kết quả ủng hộ giữ cấu hình `XGB-D03` hiện tại.")
    lines.extend(["", "Bảng được tính lại từ dự đoán đã lưu và đối chiếu với nhãn chính thức, thứ tự ID, phép chia fold, "
                  "mã băm SHA-256 của ba CSV và nhật ký hai giai đoạn chạy CUDA của từng fold. "
                  "[Số liệu đầy đủ](comparison.json) · [Lần sàng lọc trước khi huấn luyện](../../ket_qua_eda.md)", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-reference", action="store_true", help="Verify the existing reference only and write no report")
    args = parser.parse_args()
    if not args.verify_reference:
        for path in [FULL / "manifest.json", FULL / "feature_manifest.json", *(FULL / name / "metrics.json" for name in VARIANTS)]:
            if not path.is_file():
                parser.exit(2, f"Kết quả huấn luyện đầy đủ chưa có tại {path}\n")

    contract = load_contract()
    baseline = checked_run(REFERENCE / BASE, contract, xgb=True)
    final = checked_run(REFERENCE / "final", contract, xgb=False)
    if args.verify_reference:
        print(f"Verified XGB-D03 {baseline['oof_auc']:.6f} and FINAL {final['oof_auc']:.6f} on all five folds.")
        return

    if read_json(FULL / "manifest.json") != contract[0]:
        raise ValueError("The ablation run's manifest differs from the original full-data manifest")
    feature_manifest = checked_feature_manifest()
    variants = [compare(checked_run(FULL / name, contract, xgb=True, baseline=baseline), baseline) for name in VARIANTS]
    summary = {"manifest": contract[0], "feature_manifest": feature_manifest, "reference": baseline, "final": final,
               "screening": {"min_delta_auc": MIN_DELTA, "min_fold_wins": MIN_WINS}, "variants": variants}
    report = RESULTS / "ket_qua_thu_dac_trung.md"
    staged = report.with_suffix(".md.tmp")
    staged.write_text(make_report(baseline, final, variants), encoding="utf-8")
    write_json(RESULTS / "comparison.json", summary)
    staged.replace(report)
    print(report)


if __name__ == "__main__":
    main()
