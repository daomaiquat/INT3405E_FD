"""Verify completed S6E11 artifacts and render Vietnamese progress notebooks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import nbformat as nb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REPO = PROJECT.parent
RESULTS = HERE / "results" / "full"
NOTEBOOKS = HERE / "notebooks"


@dataclass(frozen=True)
class Stage:
    slug: str
    title: str
    steps: str
    command: str
    pattern: str
    expected: tuple[str, ...]
    method: str
    gap: str

    def matches(self, name):
        return re.fullmatch(self.pattern, name) is not None


STAGES = (
    Stage(
        "01-te", "Mã hóa theo nhãn cho 13 và 24 cột", "7–8", "--stage te",
        r"GPU-(?:B02|S6-L04|S6-X02)|T0[1-4]",
        ("GPU-B02", "GPU-S6-L04", "GPU-S6-X02", "T01", "T02", "T03", "T04"),
        "So sánh GLM, LightGBM và XGBoost thêm TE13 với ba đối chứng GPU chỉ dùng đặc trưng gốc. "
        "TE ước lượng tỷ lệ gian lận có làm trơn bằng 20 mẫu. Năm fold nội bộ tách nhãn của từng "
        "dòng huấn luyện khỏi cách mã hóa dòng đó. GLM dùng logit của TE, cây dùng tỷ lệ. "
        "T04 mở rộng mô hình TE13 dẫn đầu sang 24 cột.",
        "T04 chọn họ mô hình theo chính điểm khảo sát này. Kết quả cần được đọc theo từng đối chứng.",
    ),
    Stage(
        "02-digit", "Thử nghiệm khóa chữ số", "9", "--stage features --compare-xgb-digits", r"(?:XGB-)?D0[1-3]",
        ("D01", "D02", "D03", "XGB-D01", "XGB-D02", "XGB-D03"),
        "D01 thêm 6 giá trị phần trăm của đơn vị, chẳng hạn 37 trong 1371.37. "
        "D02 mã hóa theo nhãn 36 khóa chữ số từ sáu cột số. D03 kết hợp hai nhóm. "
        "Mỗi lượt giữ cùng mô hình TE làm đối chứng. LightGBM dùng T04 và XGBoost dùng T03.",
        "Trình chạy đo D03 cùng D01 và D02 để so sánh trên cùng fold. Kết quả chữ số gắn với "
        "cấu trúc của bộ dữ liệu tổng hợp.",
    ),
    Stage(
        "03-pairs-count", "Cặp tương tác, tần suất và biến dẫn xuất", "10", "--stage features",
        r"P1(?:-P2(?:-P3)?)?|COUNT|DERIVED", ("P1", "P1-P2", "P1-P2-P3", "COUNT", "DERIVED"),
        "P1, P2 và P3 lần lượt mở rộng GLM với 4, 8 và 13 khóa tương tác. Ngưỡng chia phân vị "
        "được học trong mỗi tập tham chiếu. COUNT thêm log(1+tần suất) từ dữ liệu huấn luyện. "
        "DERIVED đo riêng bình phương số cờ và tỷ lệ số tiền trên số dư.",
        "Chênh lệch của P2 và P3 đo cả nhóm khóa vừa thêm. Cần phép bỏ từng khóa nếu muốn quy kết "
        "lợi ích cho một tương tác riêng.",
    ),
    Stage(
        "04-tuning", "Điều chỉnh siêu tham số có giới hạn", "11", "--stage advanced --compare-xgb-digits --tune-family xgb --seed-anchor XGB-D03",
        r"T01-M50|GLM-C.+|GLM-NEWTON|TUNE-.+",
        ("T01-M50", "GLM-C0.1", "GLM-C0.3", "GLM-NEWTON", "TUNE-LAMBDA3", "TUNE-ETA005", "TUNE-NOSUB"),
        "T01-M50 đổi hệ số làm trơn từ 20 lên 50. GLM thử C và bộ giải Newton riêng. "
        "Cây thử điều chuẩn, tốc độ học và lấy mẫu. File metric lưu log-loss, số vòng cùng "
        "trạng thái hội tụ khi trình chạy thu được các trường đó.",
        "Các cấu hình được so trên bộ fold khảo sát. Bảng chỉ mô tả các cấu hình đã chạy.",
    ),
    Stage(
        "05-glm-boost", "GLM làm nền cho cây", "12", "--stage advanced --compare-xgb-digits --tune-family xgb --seed-anchor XGB-D03", r"GLM-BOOST-.+",
        ("GLM-BOOST-LGB", "GLM-BOOST-XGB"),
        "GLM cung cấp logit ban đầu. Cây học phần hiệu chỉnh trên logit rồi chuyển tổng "
        "về xác suất. Quy trình chọn số cây giữ riêng tập dừng sớm, sau đó huấn luyện lại "
        "trên toàn bộ phần huấn luyện của fold.",
        "Các tệp dự đoán đủ để tính lại AUC. Khả năng nạp lại trọn bộ GLM, tiền xử lý và cây "
        "cần kiểm tra bằng model bundle riêng.",
    ),
    Stage(
        "06-seed", "Trung bình dự đoán qua ba seed", "13", "--stage advanced --compare-xgb-digits --tune-family xgb --seed-anchor XGB-D03",
        r".+-S(?:71|93)|SEED3-(?:MEAN|RANK)", ("SEED3-MEAN", "SEED3-RANK"),
        "Giữ nguyên outer fold và chạy lại XGB-D03 với một seed mô hình khác và một seed bộ mã hóa khác. "
        "So trung bình xác suất với trung bình thứ hạng của ba lượt.",
        "S71 đổi seed của cây, S93 đổi seed của bộ mã hóa. MLP là nhánh tùy chọn "
        "và không thuộc danh sách thí nghiệm của trình chạy này.",
    ),
    Stage(
        "07-ensemble-final", "Ensemble, xác nhận và tệp nộp", "14–19", "--stage all --compare-xgb-digits --tune-family xgb --seed-anchor XGB-D03 --stack-lgb D03 --stack-xgb XGB-D03",
        r"(?:FINAL|(?:ALL3|TREES2)-(?:MEAN|RANK)|.+-R125)", (),
        "Phân biệt meta CV dùng dự đoán OOF đã lưu với xác nhận lồng (nested CV), nơi mỗi "
        "phần giữ lại được dự đoán bằng pipeline học hoàn toàn trong phần huấn luyện tương ứng. "
        "Hai họ cây dùng D03 tương ứng để so cách gộp. Khi bỏ một thành viên vẫn đạt ngưỡng "
        "sàng lọc và giảm dưới 0.0001 AUC, xác nhận lồng ưu tiên tổ hợp gọn hơn. "
        "Các hiện vật của chặng cuối ghi nguồn dự đoán, cách refit và tệp nộp nếu có.",
        "Điểm khảo sát và điểm xác nhận dùng hai quy trình khác nhau. Tệp nộp được kiểm tra "
        "cục bộ và chỉ có điểm Kaggle khi có kết quả nộp thực tế.",
    ),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, title, tolerance=1e-12):
    require(np.isclose(float(actual), expected, rtol=0, atol=tolerance),
            f"Sai khác {title}: {actual} / {expected}")


def load_basis(manifest):
    official = read_json(REPO / "data" / "manifest.json")
    config = read_json(REPO / "configs" / "default.json")
    controls = read_json(REPO / "configs" / "phase8_controls.json")["T01"]
    require(manifest["smoke"] is False and manifest["models_use_llm"] is False,
            "Thư mục không phải lượt chạy đầy đủ")
    require(manifest["fold_hash"] == controls["fold_hash"]
            and manifest["early_stop_hash"] == controls["early_stop_hash"],
            "Mã kiểm tra fold không khớp cấu hình đối chứng")
    for name, info in official["input_files"].items():
        require(digest(PROJECT.parent / "data" / name) == manifest["inputs"][name] == info["sha256"],
                f"Mã SHA-256 sai cho {name}")
    y_name = config["target"]
    train = pd.read_csv(PROJECT.parent / 'data' / "train.csv", usecols=["id", y_name])
    test_ids = pd.read_csv(PROJECT.parent / 'data' / "test.csv", usecols=["id"])["id"].to_numpy()
    ids = train["id"].to_numpy()
    y = train[y_name].to_numpy(dtype=np.int8)
    require(len(ids) == manifest["rows"] == official["train_rows"]
            and len(test_ids) == manifest["test_rows"] == official["test_rows"],
            "Số dòng không khớp manifest")
    folds = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(config["n_folds"], shuffle=True, random_state=config["seed"])
    for fold, (_, valid) in enumerate(splitter.split(ids, y)):
        folds[valid] = fold
    fold_csv = pd.DataFrame({"id": ids, y_name: y, "fold": folds}).to_csv(index=False, lineterminator="\n")
    require(hashlib.sha256(fold_csv.encode()).hexdigest() == manifest["fold_hash"],
            "Phép chia từ dữ liệu gốc khác manifest")
    return ids, y, test_ids, folds


def verify_run(directory, manifest, basis):
    name = "FINAL" if directory.name.casefold() == "final" else directory.name
    metrics = read_json(directory / "metrics.json")
    require(metrics["name"] == name and metrics["status"] == "completed", "Lượt chưa hoàn tất")
    require(all(metrics[key] == value for key, value in manifest.items()), "Manifest của lượt không khớp")
    config_path = directory / "config.json"
    if "family" in metrics["config"] or config_path.is_file():
        require(read_json(config_path) == metrics["config"], "Cấu hình đã lưu không khớp")
    ids, y, test_ids, fold = basis
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        for field, expected in (("ids", ids), ("test_ids", test_ids), ("fold", fold)):
            require(np.array_equal(saved[field], expected), f"Sai {field} trong dự đoán")
        oof, test = saved["oof"], saved["test"]
    for values, shape in ((oof, y.shape), (test, test_ids.shape)):
        require(values.shape == shape and np.isfinite(values).all()
                and ((values >= 0) & (values <= 1)).all(),
                "Kích thước hoặc miền giá trị dự đoán sai")
    for key, metric in (("oof_auc", roc_auc_score), ("average_precision", average_precision_score),
                        ("log_loss", log_loss)):
        close(metrics[key], metric(y, oof), key)
    folds = metrics["folds"]
    require([item["fold"] for item in folds] == list(range(1, int(fold.max()) + 2)), "Thiếu fold")
    for item in folds:
        mask = fold == item["fold"] - 1
        close(item["auc"], roc_auc_score(y[mask], oof[mask]), f"AUC fold {item['fold']}")
        if "valid_rows" in item:
            require(item["valid_rows"] == mask.sum(), "Sai số dòng trong fold")
        if "family" in metrics["config"]:
            require("RTX 4060" in item.get("device", ""), "Thiếu xác nhận RTX 4060")
            if item.get("early"):
                require("RTX 4060" in item["early"].get("device", ""), "Fold dừng sớm thiếu xác nhận RTX 4060")
    return metrics


def collect(results=RESULTS):
    results = Path(results)
    errors = []
    csv_rows = {}
    csv_path = results / "comparison.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            csv_rows = {row["name"]: row for row in csv.DictReader(stream)}
    pending = {path.name: len(list(path.glob("fold_*.json")))
               for path in results.iterdir()
               if path.is_dir() and (path / "config.json").is_file() and not (path / "metrics.json").is_file()} \
        if results.is_dir() else {}
    paths = sorted(path for path in results.glob("*/metrics.json") if path.parent.name != "nested")
    if not paths:
        return {}, pending, errors
    try:
        manifest = read_json(results / "manifest.json")
        basis = load_basis(manifest)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return {}, pending, [f"Chưa thể xác minh dữ liệu và fold · {exc}"]
    verified = {}
    for path in paths:
        name = "FINAL" if path.parent.name.casefold() == "final" else path.parent.name
        try:
            metrics = verify_run(path.parent, manifest, basis)
            csv_row = csv_rows.get(name)
            if csv_row:
                for key in ("oof_auc", "average_precision", "log_loss"):
                    close(csv_row[key], metrics[key], f"CSV {key}", 5.1e-11)
                for fold in metrics["folds"]:
                    close(csv_row[f"fold_{fold['fold']}"], fold["auc"], "CSV AUC theo fold", 5.1e-11)
            verified[name] = metrics
        except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
            errors.append(f"`{name}` · {exc}")
    if missing := sorted(verified.keys() - csv_rows.keys()):
        errors.append("Bảng CSV chưa có " + ", ".join(f"`{name}`" for name in missing)
                      + ". Các dòng này đã được đối chiếu trực tiếp từ dự đoán và JSON.")
    for item in verified.values():
        parent = verified.get(item["config"].get("parent"))
        if parent and "delta_oof_auc" in item:
            try:
                close(item["delta_oof_auc"], item["oof_auc"] - parent["oof_auc"], "delta AUC")
                require(len(item["delta_by_fold"]) == len(item["folds"]), "Thiếu delta fold")
                for recorded, row, peer in zip(item["delta_by_fold"], item["folds"], parent["folds"], strict=True):
                    close(recorded, row["auc"] - peer["auc"], "delta AUC theo fold")
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"`{item['name']}` · {exc}")
    return verified, pending, errors


def stage_rows(stage, records):
    return sorted((item for name, item in records.items() if stage.matches(name)),
                  key=lambda item: (-item["oof_auc"], item["name"]))


def number(value, *, signed=False, digits=6):
    return "—" if value is None else f"{value:{'+' if signed else ''}.{digits}f}"


def table(rows, *, folds=False):
    if not rows:
        return "Chưa có kết quả đầy đủ đã xác minh cho nhóm này."
    headers = ["Mã", "Đối chứng", "ROC AUC OOF", "Δ AUC", "AP", "Log-loss", "Giây"]
    if folds:
        headers.extend(f"Fold {i}" for i in range(1, 6))
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join([":---", ":---", *[ "---:" for _ in headers[2:]]]) + " |"]
    for row in rows:
        parent = row["config"].get("parent")
        values = [
            f"`{row['name']}`", f"`{parent}`" if parent else "—", number(row["oof_auc"]),
            number(row.get("delta_oof_auc"), signed=True), number(row["average_precision"]),
            number(row["log_loss"]), number(row["seconds"], digits=1) if row["seconds"] else "—",
        ]
        if folds:
            by_fold = {item["fold"]: item["auc"] for item in row["folds"]}
            values.extend(number(by_fold.get(i)) for i in range(1, 6))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def stage_status(stage, rows, pending):
    actual = {row["name"] for row in rows}
    waiting = [name for name in stage.expected if name not in actual]
    running = [(name, count) for name, count in pending.items() if stage.matches(name)]
    lines = []
    if running:
        details = ", ".join(f"`{name}` ({count}/5 tệp metric)" for name, count in running)
        lines.append(f"Đã có checkpoint theo fold cho {details}.")
    if waiting:
        lines.append("Chưa có hiện vật đầy đủ đã xác minh cho "
                     + ", ".join(f"`{name}`" for name in waiting) + ".")
    return "\n\n".join(lines)


def nested_summary(results):
    directory = results / "nested"
    path = directory / "metrics.json"
    if not path.is_file():
        completed = sum(re.fullmatch(r"outer_\d+\.npz", item.name) is not None
                        for item in directory.glob("outer_*.npz"))
        return f"Xác nhận lồng đã lưu {completed}/5 dự đoán outer fold." if completed else ""
    try:
        manifest = read_json(results / "manifest.json")
        ids, y, _, folds = load_basis(manifest)
        metrics = read_json(path)
        require(metrics["smoke"] is False and metrics["fold_hash"] == manifest["fold_hash"]
                and metrics["outer_folds"] == metrics["inner_folds"] == 5,
                "Sai cấu hình xác nhận lồng")
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            require(np.array_equal(saved["ids"], ids) and np.array_equal(saved["fold"], folds),
                    "Sai ID hoặc fold")
            scores = saved["oof"]
        require(scores.shape == y.shape and np.isfinite(scores).all()
                and ((scores >= 0) & (scores <= 1)).all(), "Dự đoán outer không hợp lệ")
        close(metrics["oof_auc"], roc_auc_score(y, scores), "AUC xác nhận lồng")
        close(metrics["average_precision"], average_precision_score(y, scores), "AP xác nhận lồng")
        close(metrics["log_loss"], log_loss(y, scores), "log-loss xác nhận lồng")
        for fold, recorded in enumerate(metrics["fold_auc"]):
            keep = folds == fold
            close(recorded, roc_auc_score(y[keep], scores[keep]), f"AUC outer fold {fold + 1}")
        require(len(metrics["fold_auc"]) == 5, "Thiếu AUC theo outer fold")
        return (f"Đã xác minh ROC AUC {metrics['oof_auc']:.6f} qua xác nhận lồng trên 5 outer fold. "
                + "AUC theo fold lần lượt là " + ", ".join(number(score) for score in metrics["fold_auc"])
                + ". Các họ mô hình và cấu hình tầng hai được chọn ở giai đoạn khảo sát.")
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        return f"Chưa xác minh được kết quả lồng · {exc}"


def final_summary(results, *, markdown_links=True):
    sections = []
    development_path = results / "ensemble" / "ensemble_summary.json"
    if development_path.is_file():
        try:
            development = read_json(development_path)
            rows = sorted(development["metrics"], key=lambda row: -row["auc"])
            lines = ["| Khảo sát ensemble | Phương pháp | ROC AUC | Δ so mô hình đơn |",
                     "|:---|:---|---:|---:|"]
            lines.extend(
                f"| `{row['candidate']}` | `{row['method']}` | {number(row['auc'])} "
                f"| {number(row.get('delta_auc_vs_best_base'), signed=True)} |"
                for row in rows[:8]
            )
            label = ("[Bảng khảo sát đầy đủ](ensemble/ensemble_metrics.csv)."
                     if markdown_links else "Xem `ensemble/ensemble_metrics.csv`.")
            sections.append("Các điểm dưới đây do meta CV trên OOF ghi lại. Đã hiển thị 8 dòng "
                            "cao nhất khi có đủ dữ liệu. " + label + "\n\n" + "\n".join(lines))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            sections.append(f"Chưa đọc được hồ sơ khảo sát ensemble · {exc}")
    if message := nested_summary(results):
        sections.append(message)
    final = results / "final"
    path = final / "selection.json"
    if not path.is_file():
        sections.append("Chưa có `final/selection.json`. Chưa xác nhận quyết định và tệp nộp cuối.")
        return "\n\n".join(sections)
    try:
        selection = read_json(path)
        require(selection["smoke"] is False, "Đây là quyết định của lượt chạy thử")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        sections.append(f"Đã thấy `final/selection.json` nhưng chưa đọc được · {exc}")
        return "\n\n".join(sections)
    label = ("[Hồ sơ chọn phương án cuối](final/selection.json)."
             if markdown_links else "Xem `final/selection.json`.")
    sections.append(f"Chọn `{selection.get('selected', '—')}` với ROC AUC OOF "
                    f"{number(selection.get('oof_auc'))}. Mô hình đơn đối chứng là "
                    f"`{selection.get('best_single', '—')}`. Loại điểm "
                    f"`{selection.get('score_kind', '—')}`. " + label)
    if selection.get("evidence"):
        lines = ["| Ứng viên cuối | Δ ROC AUC | Fold tăng | Khoảng bootstrap 95% | Qua sàng lọc |",
                 "|:---|---:|---:|:---|:---|"]
        for name, row in sorted(selection["evidence"].items(), key=lambda item: -item[1]["delta"]):
            ci = row["bootstrap"]
            low, high = number(ci["ci_low"], signed=True), number(ci["ci_high"], signed=True)
            lines.append(f"| `{name}` | {number(row['delta'], signed=True)} "
                         f"| {sum(delta > 0 for delta in row['delta_by_fold'])} "
                         f"| [{low}, {high}] | {'Có' if row['passes_screen'] else 'Không'} |")
        sections.append("Bootstrap ghép cặp theo lớp dùng dự đoán cố định. Khoảng dưới đây "
                        "đo biến thiên lấy mẫu của bộ dự đoán đã lưu.\n\n" + "\n".join(lines))
    path = final / "submission.csv"
    if path.is_file():
        try:
            scored = pd.read_csv(path, float_precision="round_trip")
            sample = pd.read_csv(PROJECT.parent / 'data' / "sample_submission.csv", usecols=["id"])
            require(list(scored.columns) == ["id", "is_fraud"]
                    and scored["id"].equals(sample["id"])
                    and scored["id"].is_unique and scored["is_fraud"].between(0, 1).all(),
                    "cột, thứ tự ID hoặc điểm dự đoán sai")
            with np.load(results / "FINAL" / "predictions.npz", allow_pickle=False) as saved:
                require(np.array_equal(scored["id"].to_numpy(), saved["test_ids"])
                        and np.array_equal(scored["is_fraud"].to_numpy(), saved["test"]),
                        "điểm trong CSV khác dự đoán lượt FINAL")
            sha256 = digest(path)
            require(sha256 == selection["submission"]["sha256"], "SHA-256 khác hồ sơ cuối")
            label = f"[`{path.name}`](final/{path.name})" if markdown_links else f"`{path.name}`"
            sections.append(f"{label} qua kiểm tra cột, {len(scored):,} ID, điểm dự đoán và "
                            f"SHA-256 `{sha256}`.")
        except (ValueError, OSError, KeyError) as exc:
            sections.append(f"`{path.name}` chưa qua kiểm tra · {exc}")
    else:
        sections.append("Chưa có tệp nộp CSV để kiểm tra.")
    return "\n\n".join(sections)


def read_stage(stage_slug, results=RESULTS):
    stage = next(item for item in STAGES if item.slug == stage_slug)
    records, pending, errors = collect(results)
    rows = stage_rows(stage, records)
    return rows, stage_status(stage, rows, pending), errors


def render_markdown(results, records, pending, errors):
    completed = len(records)
    best = max(records.values(), key=lambda item: item["oof_auc"]) if records else None
    present = f"Đã xác minh {completed} lượt chạy đầy đủ."
    if best:
        present += f" ROC AUC OOF cao nhất trong các lượt đã xác minh là {best['oof_auc']:.6f} (`{best['name']}`)."
    body = [
        "# S6E11 · Kết quả và tiến độ thí nghiệm GPU",
        present,
        "ROC AUC đo khả năng xếp hạng giao dịch gian lận. OOF là dự đoán cho một giao dịch "
        "từ mô hình huấn luyện trên các phần còn lại. AP tóm tắt đường precision–recall. "
        "Δ AUC so với đối chứng được ghi cho từng lượt. Cột giây lấy tổng thời gian "
        "ghi theo fold, có thể gồm tiền xử lý và suy luận.",
        "Bảng được tính lại từ các tệp `predictions.npz` và đối chiếu với `metrics.json`, "
        "`config.json` khi có, cùng `comparison.csv`. Kết quả smoke và kiểm tra mô phỏng "
        "được dùng để kiểm tra mã và nằm ngoài bảng điểm đầy đủ.",
        "Phạm vi đối chiếu là Bước 7–19, dòng 298–598. [Kế hoạch dự án](../../../../INT3405E_FD/plan.md#L298)",
        "## Bảng kết quả",
        table(sorted(records.values(), key=lambda row: -row["oof_auc"]), folds=True),
        "## Tiến độ theo hướng",
    ]
    for stage in STAGES:
        rows = stage_rows(stage, records)
        status = stage_status(stage, rows, pending)
        body.append(f"### {stage.title}")
        body.append(f"Bước {stage.steps} · {len(rows)} lượt xác minh. "
                    f"[Notebook](../../notebooks/{stage.slug}.ipynb)")
        if status:
            body.append(status)
        body.append(stage.gap)
    body.extend(["## Hiện vật cuối", final_summary(results)])
    if errors:
        body.extend(["## Điểm cần kiểm tra", "\n".join(f"- {error}" for error in errors)])
    return "\n\n".join(body) + "\n"


def render_notebook(stage, results, records, pending, errors):
    rows = stage_rows(stage, records)
    command = f"& '..\\.build\\venv\\Scripts\\python.exe' -B -X utf8 run_plan.py {stage.command}"
    reproduction = (
        "Chạy từ thư mục `plan_runs` với môi trường Python đã dùng cho RTX 4060. "
        "Trình chạy tiếp tục từ checkpoint có cấu hình khớp. Cell dưới đọc kết quả theo mặc định.\n\n"
        f"```powershell\n{command}\n```"
    )
    if stage.slug == "01-te":
        reproduction += (
            "\n\nDùng `--stage te --only-te T01`, `T02` hoặc `T03` để chạy một cặp "
            "thí nghiệm và đối chứng. T04 cần điểm của cả T01–T03 để chọn họ mô hình."
        )
    lookup = dedent(f"""
        import os
        import subprocess
        import sys
        from pathlib import Path

        here = Path.cwd().resolve()
        roots = [Path(os.environ["S6E11_PLAN_RUNS"])] if "S6E11_PLAN_RUNS" in os.environ else []
        for base in (here, *here.parents):
            roots.extend((base, base / "plan_runs", base / "s6e11_gpu" / "plan_runs",
                          base / "Predicting Credit Card Fraud" / "s6e11_gpu" / "plan_runs"))
        plan_dir = next((path for path in roots if (path / "run_plan.py").is_file()), None)
        if plan_dir is None:
            raise FileNotFoundError("Mở notebook trong dự án hoặc đặt S6E11_PLAN_RUNS.")

        CHAY_LAI = False
        if CHAY_LAI:
            subprocess.run([sys.executable, "-B", "-X", "utf8", str(plan_dir / "run_plan.py"),
                            *{stage.command.split()!r}],
                           cwd=plan_dir, check=True)
    """).strip()
    snapshot = table(rows, folds=True)
    if status := stage_status(stage, rows, pending):
        snapshot += "\n\n" + status
    if stage.slug == "07-ensemble-final":
        snapshot += "\n\n" + final_summary(results, markdown_links=False)
    if errors:
        snapshot += "\n\n" + f"Có {len(errors)} lưu ý khi xác minh. Xem cell bên dưới hoặc báo cáo tiến độ."
    readback = dedent(f"""
        import numpy as np
        import pandas as pd
        from IPython.display import Markdown, display

        sys.path.insert(0, str(plan_dir))
        from report_results import final_summary, read_stage

        data_dir = plan_dir / "results" / "full"
        rows, pending, notes = read_stage({stage.slug!r}, data_dir)
        if rows:
            points = [{{"mã": row["name"], "đối chứng": row["config"].get("parent"),
                       "ROC AUC OOF": row["oof_auc"], "AP": row["average_precision"],
                       "delta AUC": row.get("delta_oof_auc"), "log-loss": row["log_loss"],
                       **{{f"fold {{item['fold']}}": item["auc"] for item in row["folds"]}}}}
                      for row in rows]
            display(pd.DataFrame(points))
        if pending:
            display(Markdown(pending))
        if notes:
            display(Markdown("\\n".join(f"- {{note}}" for note in notes)))
        if {stage.slug == '07-ensemble-final'!r}:
            display(Markdown(final_summary(data_dir, markdown_links=False)))
    """).strip()
    details = dedent("""
        if rows:
            record = rows[0]
            directory = data_dir / record["name"]
            display(record["config"])
            display(pd.DataFrame([
                {key: item.get(key) for key in
                 ("fold", "auc", "device", "seconds", "iterations", "converged", "rounds")}
                for item in record["folds"]
            ]))
            with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
                display({name: saved[name].shape for name in ("ids", "fold", "oof", "test_ids", "test")})
    """).strip()
    cells = [
        nb.v4.new_markdown_cell(f"# {stage.title}\n\nBước {stage.steps} của kế hoạch S6E11.\n\n{stage.method}"),
        nb.v4.new_markdown_cell(f"## Chạy lại\n\n{reproduction}"),
        nb.v4.new_code_cell(lookup),
        nb.v4.new_markdown_cell("## Kết quả tại thời điểm tạo notebook\n\n" + snapshot),
        nb.v4.new_markdown_cell("## Đọc dữ liệu kết quả hiện tại\n\n"
                                "Cell này xác minh những lượt hoàn tất từ dự đoán OOF và tệp metric."),
        nb.v4.new_code_cell(readback),
        nb.v4.new_markdown_cell("## Xem cấu hình và log theo fold\n\n"
                                "Hiển thị lượt có AUC cao nhất trong nhóm hiện tại."),
        nb.v4.new_code_cell(details),
        nb.v4.new_markdown_cell("## Phạm vi diễn giải\n\n" + stage.gap),
    ]
    notebook = nb.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (S6E11 GPU)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    for index, cell in enumerate(notebook.cells, start=1):
        cell.id = f"{stage.slug}-{index:02d}"
        if cell.cell_type == "code":
            compile(cell.source, f"<{stage.slug}>", "exec")
    nb.validate(notebook)
    return nb.writes(notebook)


def atomic_text(path, content):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--notebooks", type=Path, default=NOTEBOOKS)
    args = parser.parse_args()
    records, pending, errors = collect(args.results)
    args.results.mkdir(parents=True, exist_ok=True)
    args.notebooks.mkdir(parents=True, exist_ok=True)
    atomic_text(args.results / "comparison.md", render_markdown(args.results, records, pending, errors))
    for stage in STAGES:
        rendered = render_notebook(stage, args.results, records, pending, errors)
        atomic_text(args.notebooks / f"{stage.slug}.ipynb", rendered)
    print(f"{len(records)} lượt đã xác minh, {len(pending)} lượt có checkpoint, "
          f"{len(errors)} lưu ý, {len(STAGES)} notebook.")


if __name__ == "__main__":
    main()
