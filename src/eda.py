"""Step 4 EDA: hypothesis-generating tables with discovery/confirmation separation.

No model is tuned here. Every fraud-rate table carries row and fraud counts, and
specific tails/pairs are chosen on the discovery half before being checked on the
disjoint confirmation half.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .artifacts import write_json

DIGIT_COLUMNS = ("amount_usd", "hours_since_last_txn", "distance_from_home_km",
                 "account_balance_usd", "velocity_score", "merchant_risk_score")
KEY_TYPES = ("cents", "int", "int10", "tenths", "lead1", "lead2")
PAIRS = (
    ("P1_merchant_risk", (("merchant_category", "raw"), ("merchant_risk_score", "decile"))),
    ("P1_foreign_ip", (("is_foreign_transaction", "raw"), ("ip_country_mismatch", "raw"))),
    ("P1_category_amount", (("merchant_category", "raw"), ("amount_usd", "decile"))),
    ("P1_count_velocity", (("txn_count_last_24h", "raw"), ("velocity_score", "decile"))),
    ("P2_category_age", (("merchant_category", "raw"), ("customer_age", "raw"))),
    ("P2_amount_risk", (("amount_usd", "decile"), ("merchant_risk_score", "decile"))),
    ("P2_hours_velocity", (("hours_since_last_txn", "decile"), ("velocity_score", "decile"))),
    ("P2_age_risk", (("customer_age", "raw"), ("merchant_risk_score", "decile"))),
    ("P3_new_dow", (("is_new_merchant", "raw"), ("day_of_week", "raw"))),
    ("P3_scam_velocity", (("is_ai_generated_scam_attempt", "raw"), ("velocity_score", "decile"))),
    ("P3_age_card_age", (("customer_age", "raw"), ("card_age_months", "raw"))),
    ("P3_hour_dow", (("time_of_day_hour", "raw"), ("day_of_week", "raw"))),
    ("P3_count_age", (("txn_count_last_24h", "raw"), ("customer_age", "raw"))),
)


def _rate_table(labels, target, column: str, overall: float, min_rows: int = 1) -> pd.DataFrame:
    names = pd.Series(np.asarray(labels)).astype(str).reset_index(drop=True)
    values = pd.Series(np.asarray(target)).reset_index(drop=True)
    frame = pd.DataFrame({"group": names, "y": values.to_numpy()})
    grouped = frame.groupby("group", sort=False).agg(n_rows=("y", "size"), n_fraud=("y", "sum"))
    grouped = grouped.loc[grouped["n_rows"] >= min_rows].copy()
    grouped["fraud_rate"] = grouped["n_fraud"] / grouped["n_rows"]
    grouped["lift"] = grouped["fraud_rate"] / overall
    result = grouped.reset_index()
    result.insert(0, "column", column)
    return result[["column", "group", "n_rows", "n_fraud", "fraud_rate", "lift"]]


def _stability(discovery: pd.DataFrame, confirmation: pd.DataFrame,
               overall: float, min_rows: int) -> dict:
    merged = discovery.merge(confirmation[["group", "n_rows", "n_fraud", "fraud_rate"]],
                             on="group", suffixes=("_disc", "_conf"))
    merged = merged.loc[(merged["n_rows_disc"] >= min_rows)
                        & (merged["n_rows_conf"] >= min_rows)]
    correlation = float(merged["fraud_rate_disc"].corr(merged["fraud_rate_conf"])) if len(merged) >= 3 else 0.0
    top = discovery.nlargest(min(5, len(discovery)), "fraud_rate")
    confirmation_map = confirmation.set_index("group")["fraud_rate"]
    hits = int(sum(bool(confirmation_map.get(group, -1.0) > overall) for group in top["group"]))
    return {
        "groups_discovery": int(len(discovery)), "groups_confirmation": int(len(confirmation)),
        "groups_in_both": int(len(merged)),
        "rate_correlation": correlation if np.isfinite(correlation) else 0.0,
        "top5_confirmed_above_base": hits, "top5_count": int(len(top)),
        "max_lift_discovery": float(discovery["lift"].max()) if len(discovery) else 0.0,
        "max_lift_confirmation": float(confirmation["lift"].max()) if len(confirmation) else 0.0,
    }


def imbalance_summary(train: pd.DataFrame, config: dict) -> dict:
    labels = train[config["target"]].to_numpy()
    positives = int(labels.sum())
    return {"rows": int(len(labels)), "positive_rows": positives,
            "negative_rows": int(len(labels)) - positives,
            "fraud_rate": float(labels.mean()),
            "negative_to_positive_ratio": float((len(labels) - positives) / positives)}


def univariate_tables(train: pd.DataFrame, config: dict, overall: float) -> dict:
    target = config["target"]
    labels = train[target].to_numpy()
    numeric_summary, numeric_tables = [], {}
    for column in config["numeric_columns"]:
        values = train[column].to_numpy(dtype=np.float64)
        numeric_summary.append({"column": column, "univariate_auc": roc_auc_score(labels, values),
                                "distinct": int(train[column].nunique()),
                                "min": float(values.min()), "max": float(values.max())})
        if train[column].nunique() <= 25:
            groups = train[column].astype(str)
        else:
            groups = pd.qcut(train[column], 10, duplicates="drop").astype(str)
        numeric_tables[column] = _rate_table(groups, labels, column, overall)
    categorical_tables = {c: _rate_table(train[c], labels, c, overall)
                          for c in config["categorical_columns"]}
    flag_tables = {c: _rate_table(train[c].astype(int), labels, c, overall)
                   for c in config["flag_columns"]}
    n_flags = train[config["flag_columns"]].sum(axis=1).astype(int)
    flags_table = _rate_table(n_flags, labels, "n_flags", overall)
    no_flag = train.loc[n_flags == 0]
    flag_effects = []
    for column, table in flag_tables.items():
        rates = table.set_index("group")["fraud_rate"]
        flag_effects.append({"column": column, "rate_when_true": float(rates.get("1", np.nan)),
                             "rate_when_false": float(rates.get("0", np.nan)),
                             "lift_when_true": float(table.set_index("group")["lift"].get("1", np.nan))})
    return {
        "numeric_summary": pd.DataFrame(numeric_summary).sort_values("univariate_auc", ascending=False),
        "numeric_tables": numeric_tables,
        "categorical_tables": categorical_tables,
        "flag_tables": flag_tables,
        "flags_table": flags_table,
        "no_flag": {"rows": int(len(no_flag)), "fraud_rows": int(no_flag[target].sum()),
                    "fraud_rate": float(no_flag[target].mean()) if len(no_flag) else 0.0,
                    "share_of_all_fraud": float(no_flag[target].sum() / int(labels.sum()))},
        "flag_effects": pd.DataFrame(flag_effects),
    }


def structure_tables(train: pd.DataFrame, test: pd.DataFrame, folds: pd.DataFrame,
                     config: dict) -> pd.DataFrame:
    target = config["target"]
    valid_ids = folds.loc[folds["fold"] == 0, "id"]
    valid = train.loc[train["id"].isin(valid_ids)]
    rows = []
    for column in config["numeric_columns"] + config["categorical_columns"]:
        train_values = set(train[column])
        rows.append({
            "column": column,
            "kind": "numeric" if column in config["numeric_columns"] else "categorical",
            "train_distinct": int(train[column].nunique()),
            "test_distinct": int(test[column].nunique()),
            "top_value": str(train[column].value_counts().index[0]),
            "top_share": float(train[column].value_counts(normalize=True).iloc[0]),
            "unseen_rows_test": int((~test[column].isin(train_values)).sum()),
            "unseen_rows_valid_fold0": int((~valid[column].isin(train_values)).sum()),
        })
    return pd.DataFrame(rows)


def split_halves(train: pd.DataFrame, config: dict):
    settings = config["eda"]
    first, second = train_test_split(
        np.arange(len(train)), test_size=settings["discovery_fraction"],
        stratify=train[config["target"]], random_state=settings["seed"])
    return train.iloc[np.sort(first)].reset_index(drop=True), train.iloc[np.sort(second)].reset_index(drop=True)


def _digit_labels(values: np.ndarray, key_type: str) -> pd.Series:
    units = np.rint(values * 100).astype(np.int64)
    if key_type == "cents":
        return pd.Series(units % 100).astype(str)
    integer_part = units // 100
    if key_type == "int":
        return pd.Series(integer_part).astype(str)
    if key_type == "int10":
        return pd.Series(units // 1000).astype(str)
    if key_type == "tenths":
        return pd.Series(units // 10).astype(str)
    text = pd.Series(integer_part).astype(str)
    length = text.str.len().astype(str)
    if key_type == "lead1":
        return length + "|" + text.str[0]
    return length + "|" + text.str[:2]


def digit_tables(discovery: pd.DataFrame, confirmation: pd.DataFrame, config: dict,
                 overall: float) -> dict:
    min_rows = config["eda"]["min_digit_group"]
    target = config["target"]
    discovery_rows, confirmation_rows, stability_rows = [], [], []
    for column in DIGIT_COLUMNS:
        for key_type in KEY_TYPES:
            name = f"{column}__{key_type}"
            disc_labels = _digit_labels(discovery[column].to_numpy(dtype=np.float64), key_type)
            conf_labels = _digit_labels(confirmation[column].to_numpy(dtype=np.float64), key_type)
            disc = _rate_table(disc_labels, discovery[target], name, overall, min_rows)
            conf = _rate_table(conf_labels, confirmation[target], name, overall, min_rows)
            discovery_rows.append(disc)
            confirmation_rows.append(conf)
            stability_rows.append({"column": column, "key_type": key_type,
                                   **_stability(disc, conf, overall, min_rows)})
    return {"discovery": pd.concat(discovery_rows, ignore_index=True),
            "confirmation": pd.concat(confirmation_rows, ignore_index=True),
            "stability": pd.DataFrame(stability_rows)}


def _decile_edges(values: np.ndarray, quantiles: int = 10) -> np.ndarray:
    return np.unique(np.quantile(values, np.linspace(0, 1, quantiles + 1)))


def _decile_labels(values: np.ndarray, edges: np.ndarray) -> pd.Series:
    return pd.Series(np.searchsorted(edges[1:-1], values, side="right")).astype(str)


def _pair_labels(frame: pd.DataFrame, specs, edges: dict) -> pd.Series:
    pieces = []
    for column, mode in specs:
        if mode == "decile":
            pieces.append(_decile_labels(frame[column].to_numpy(dtype=np.float64), edges[column]))
        else:
            pieces.append(frame[column].astype(str).reset_index(drop=True))
    result = pieces[0].astype(str).reset_index(drop=True)
    for piece in pieces[1:]:
        result = result + "|" + piece.astype(str).reset_index(drop=True)
    return result


def pair_tables(discovery: pd.DataFrame, confirmation: pd.DataFrame, config: dict,
                overall: float) -> dict:
    min_rows = config["eda"]["min_pair_group"]
    target = config["target"]
    needed = {column for _, specs in PAIRS for column, mode in specs if mode == "decile"}
    edges = {column: _decile_edges(discovery[column].to_numpy(dtype=np.float64)) for column in needed}
    discovery_rows, confirmation_rows, stability_rows = [], [], []
    for name, specs in PAIRS:
        disc = _rate_table(_pair_labels(discovery, specs, edges), discovery[target], name,
                           overall, min_rows)
        conf = _rate_table(_pair_labels(confirmation, specs, edges), confirmation[target], name,
                           overall, min_rows)
        discovery_rows.append(disc)
        confirmation_rows.append(conf)
        stability_rows.append({"pair": name, **_stability(disc, conf, overall, min_rows)})
    return {"discovery": pd.concat(discovery_rows, ignore_index=True),
            "confirmation": pd.concat(confirmation_rows, ignore_index=True),
            "stability": pd.DataFrame(stability_rows)}


def train_test_shift(train: pd.DataFrame, test: pd.DataFrame, config: dict) -> pd.DataFrame:
    settings = config["eda"]
    rng = np.random.default_rng(settings["seed"])
    rows = []
    for column in config["numeric_columns"]:
        source = train[column].to_numpy(dtype=np.float64)
        target = test[column].to_numpy(dtype=np.float64)
        spread = np.sqrt((source.var() + target.var()) / 2)
        size = settings["ks_rows_per_side"]
        sample_source = source[rng.choice(len(source), min(size, len(source)), replace=False)]
        sample_target = target[rng.choice(len(target), min(size, len(target)), replace=False)]
        rows.append({"column": column, "kind": "numeric",
                     "train_mean": float(source.mean()), "test_mean": float(target.mean()),
                     "smd": float((target.mean() - source.mean()) / spread) if spread else 0.0,
                     "ks_statistic": float(ks_2samp(sample_source, sample_target).statistic)})
    for column in config["categorical_columns"]:
        source = train[column].value_counts(normalize=True)
        target = test[column].value_counts(normalize=True)
        categories = source.index.union(target.index)
        distance = 0.5 * float((source.reindex(categories, fill_value=0)
                                - target.reindex(categories, fill_value=0)).abs().sum())
        rows.append({"column": column, "kind": "categorical", "total_variation": distance})
    return pd.DataFrame(rows)


def adversarial_validation(train: pd.DataFrame, test: pd.DataFrame, config: dict) -> dict:
    settings = config["eda"]
    size = settings["adversarial_rows_per_side"]
    source = train.sample(n=min(size, len(train)), random_state=settings["seed"])
    target = test.sample(n=min(size, len(test)), random_state=settings["seed"])
    features = pd.concat([source.drop(columns=["id", config["target"]]), target.drop(columns=["id"])],
                         ignore_index=True)
    labels = np.r_[np.zeros(len(source), dtype=np.int8), np.ones(len(target), dtype=np.int8)]
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), config["numeric_columns"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), config["categorical_columns"]),
        ("flags", "passthrough", config["flag_columns"]),
    ])
    model = make_pipeline(preprocessor, LogisticRegression(
        C=1.0, max_iter=500, random_state=config["model_seed"]))
    splitter = StratifiedKFold(3, shuffle=True, random_state=config["model_seed"])
    fold_auc = {}
    for fold, (fit_index, valid_index) in enumerate(splitter.split(features, labels)):
        model.fit(features.iloc[fit_index], labels[fit_index])
        prediction = model.predict_proba(features.iloc[valid_index])[:, 1]
        fold_auc[str(fold)] = float(roc_auc_score(labels[valid_index], prediction))
    return {"rows_train_sample": len(source), "rows_test_sample": len(target),
            "fold_auc": fold_auc, "mean_auc": float(np.mean(list(fold_auc.values()))),
             "interpretation": "AUC gần 0.5 không chứng minh train/test cùng phân phối."}



def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def build_findings(imbalance: dict, univariate: dict, structure: pd.DataFrame,
                   digits: dict, pairs: dict, shift: pd.DataFrame, adversarial: dict,
                   config: dict) -> list[dict]:
    overall = imbalance["fraud_rate"]
    numeric_summary = univariate["numeric_summary"]
    strongest = numeric_summary.iloc[0]
    flags = univariate["flags_table"].set_index("group")
    no_flag = univariate["no_flag"]
    cvv = univariate["numeric_tables"]["cvv_retry_count"].set_index("group")
    auth = univariate["categorical_tables"]["auth_method"].sort_values("fraud_rate")
    low_auth, high_auth = auth.iloc[0], auth.iloc[-1]
    def rate_of(table: pd.DataFrame, key: str) -> float:
        return float(table.loc[key, "fraud_rate"]) if key in table.index else 0.0
    cvv_rates = " / ".join(_pct(rate_of(cvv, str(value))) for value in range(3))
    cvv_third_rows = int(cvv.loc["3", "n_rows"]) if "3" in cvv.index else 0
    stable_digits = digits["stability"].loc[digits["stability"]["groups_in_both"] >= 10]
    digit_best = stable_digits.sort_values(["rate_correlation", "top5_confirmed_above_base"],
                                           ascending=False).iloc[0]
    stable_pairs = pairs["stability"].loc[pairs["stability"]["groups_in_both"] >= 20]
    pair_best = stable_pairs.sort_values(["rate_correlation", "top5_confirmed_above_base"],
                                         ascending=False).iloc[0]
    top_shift = shift.loc[shift["kind"] == "numeric"].sort_values("ks_statistic", ascending=False).iloc[0]
    unverified = "observed_in_eda_unverified_by_ablation"

    return [
         {"id": "F1", "finding": f"Train có {imbalance['positive_rows']:,} dòng dương trên "
                                 f"{imbalance['rows']:,} dòng ({_pct(overall)}).",
          "hypothesis": "Giữ ROC AUC làm metric chính và giữ StratifiedKFold; không dùng accuracy "
                        "và không mặc định class weight/SMOTE.",

         "priority": "high", "status": "confirmed_from_data", "tested_in": "steps 2, 5"},
         {"id": "F2", "finding": f"Numeric đơn biến mạnh nhất là {strongest['column']} "
                                 f"(AUC {strongest['univariate_auc']:.3f}); "
                                 f"số giá trị riêng từ 2 đến {int(structure['train_distinct'].max()):,}.",
          "hypothesis": "Thử target encoding 13 cột số với smoothing m=20 và cây nhỏ; so sánh "
                        "với raw/log để đo riêng giá trị TE.",

         "priority": "high", "status": unverified, "tested_in": "steps 7-8"},
         {"id": "F3", "finding": f"No flag: {no_flag['rows']:,} dòng, {no_flag['fraud_rows']:,} fraud "
                                 f"({_pct(no_flag['fraud_rate'])}), chiếm "
                                 f"{_pct(no_flag['share_of_all_fraud'])} tổng fraud. Tỷ lệ fraud "
                                 f"theo n_flags: 0 → {_pct(flags.loc['0', 'fraud_rate'])}, "
                                 f"1 → {_pct(flags.loc['1', 'fraud_rate'])}, "
                                 f"2 → {_pct(flags.loc['2', 'fraud_rate'])}.",
          "hypothesis": "Giữ 6 cờ raw; thử nghiệm n_flags và one-hot/nhóm phi tuyến thành nhóm riêng "
                        "để phân tích lỗi no-flag.",

         "priority": "high", "status": unverified, "tested_in": "steps 5, 10, 17"},
         {"id": "F4", "finding": f"cvv_retry_count phi tuyến: fraud rate tại 0/1/2 là {cvv_rates}; "
                                 f"giá trị 3 có {cvv_third_rows} dòng.",
          "hypothesis": "Không ép tuyến tính: giữ dạng one-hot/spline hoặc TE có smoothing; "
                        "đối chiếu số liệu guide bằng bảng số của chính project.",

         "priority": "high", "status": unverified, "tested_in": "steps 5, 7-8"},
         {"id": "F5", "finding": f"auth_method phân hóa mạnh: {low_auth['group']} "
                                 f"{_pct(low_auth['fraud_rate'])} so với {high_auth['group']} "
                                 f"{_pct(high_auth['fraud_rate'])}.",
          "hypothesis": "TE cho các danh mục mạnh, one-hot cho danh mục yếu; quyết định bằng ablation "
                        "chứ không đánh giá bằng phán đoán đơn biến.",

         "priority": "medium", "status": unverified, "tested_in": "steps 7-8"},
         {"id": "F6", "finding": f"Chia discovery/confirmation có {_pct(overall)} fraud ở cả hai "
                                 f"nửa. Key ổn định nhất: {digit_best['column']}__{digit_best['key_type']} "
                                 f"corr={digit_best['rate_correlation']:.2f}, "
                                 f"top5 xác nhận {int(digit_best['top5_confirmed_above_base'])}/"
                                 f"{int(digit_best['top5_count'])}.",
          "hypothesis": "Đưa cents numeric (D01) và 36 digit TE (D02) vào ablation riêng; đây là "
                        "cấu trúc dữ liệu tổng hợp, không mặc định có ý nghĩa nghiệp vụ.",

         "priority": "medium", "status": unverified, "tested_in": "steps 9"},
         {"id": "F7", "finding": f"Pair ổn định nhất: {pair_best['pair']} corr="
                                 f"{pair_best['rate_correlation']:.2f}, top5 xác nhận "
                                 f"{int(pair_best['top5_confirmed_above_base'])}/"
                                 f"{int(pair_best['top5_count'])}; các group đủ n hiện có "
                                 f"{int(pair_best['groups_in_both'])}.",
          "hypothesis": "Thử pair ổn định này trước khi mở rộng nhóm khác; chỉ kết luận tương tác "
                        "khi ablation cải thiện AUC.",

         "priority": "medium", "status": unverified, "tested_in": "step 10"},
         {"id": "F8", "finding": f"Lệch train/test lớn nhất theo KS là {top_shift['column']} "
                                 f"(KS={top_shift['ks_statistic']:.4f}); adversarial AUC="
                                 f"{adversarial['mean_auc']:.4f}.",
          "hypothesis": "Giữ random CV làm chính và ghi limit; không khẳng định cùng phân phối chỉ "
                        "vì AUC gần 0.5 hoặc p-value lớn.",
          "priority": "medium", "status": unverified,

         "tested_in": "steps 2, 15"},
    ]


def _md_table(frame: pd.DataFrame, max_rows: int = 8) -> str:
    view = frame.head(max_rows)
    columns = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in view.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_report(output_dir: Path, imbalance: dict, univariate: dict, structure: pd.DataFrame,
                  digits: dict, pairs: dict, shift: pd.DataFrame, adversarial: dict,
                  findings: list[dict]) -> None:
    lines = [
        "# EDA dẫn tới giả thuyết (plan.md bước 4)",
        "",
        f"Dữ liệu: {imbalance['rows']:,} dòng train, {imbalance['positive_rows']:,} fraud "
        f"({_pct(imbalance['fraud_rate'])}). Mọi bảng tỷ lệ đều có số dòng và số fraud.",
        "",
        "## Phát hiện và giả thuyết",
        "",
        "| ID | Phát hiện | Giả thuyết | Ưu tiên | Trạng thái | Kiểm tra ở |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in findings:
        lines.append(f"| {item['id']} | {item['finding']} | {item['hypothesis']} | "
                     f"{item['priority']} | {item['status']} | {item['tested_in']} |")
    lines += ["", "## Numeric đơn biến", "", _md_table(univariate["numeric_summary"], 13), "",
              "## Cờ và n_flags", "", _md_table(univariate["flags_table"], 8), "",
              "## Không có flag nào", "",
              f"{univariate['no_flag']['rows']:,} dòng; {univariate['no_flag']['fraud_rows']:,} fraud; "
              f"tỷ lệ {_pct(univariate['no_flag']['fraud_rate'])}; "
              f"chiếm {_pct(univariate['no_flag']['share_of_all_fraud'])} tổng fraud.", "",
              "## Cấu trúc và giá trị chưa gặp", "", _md_table(structure, 20), "",
              "## Độ ổn định của digit key (discovery → confirmation)", "",
              _md_table(digits["stability"].sort_values("rate_correlation", ascending=False), 12), "",
              "## Độ ổn định của pair key", "",
              _md_table(pairs["stability"].sort_values("rate_correlation", ascending=False), 13), "",
              "## Train/test shift", "", _md_table(shift.sort_values("ks_statistic", ascending=False), 20), "",
              f"Adversarial AUC 3 fold: {adversarial['fold_auc']}, mean "
              f"{adversarial['mean_auc']:.4f}. Chỉ là bằng chứng bổ sung.", "",
              "## Lưu ý", "",
              "- Discovery/confirmation là hai nửa phân tầng không giao nhau; không chọn đuôi "
              "thập phân hay pair hot trên toàn train rồi gọi là đã xác nhận.",
              "- Phát hiện là quan sát EDA; trừ F1 (fact của dữ liệu) các giả thuyết đều chưa "
              "được kiểm chứng bằng ablation/ensemble.",
              "- AUC gần 0.5 của adversarial không chứng minh train/test cùng phân phối.",
              "- Số liệu của guide (no-flag, CVV, risk score) được tính lại từ CSV chính thức ở bảng trên.",
              ""]
    (output_dir / "eda_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_eda(output_dir: Path, train: pd.DataFrame, test: pd.DataFrame, folds: pd.DataFrame,
            config: dict) -> dict:
    """Write all step 4 tables and return a compact JSON-safe summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = float(train[config["target"]].mean())
    imbalance = imbalance_summary(train, config)
    univariate = univariate_tables(train, config, overall)
    structure = structure_tables(train, test, folds, config)
    discovery, confirmation = split_halves(train, config)
    digits = digit_tables(discovery, confirmation, config, overall)
    pairs = pair_tables(discovery, confirmation, config, overall)
    shift = train_test_shift(train, test, config)
    adversarial = adversarial_validation(train, test, config)
    findings = build_findings(imbalance, univariate, structure, digits, pairs, shift,
                              adversarial, config)

    write_json(output_dir / "imbalance.json", imbalance)
    univariate["numeric_summary"].to_csv(output_dir / "univariate_numeric_summary.csv", index=False)
    for column, table in univariate["numeric_tables"].items():
        table.to_csv(output_dir / f"univariate_numeric_{column}.csv", index=False)
    for column, table in univariate["categorical_tables"].items():
        table.to_csv(output_dir / f"univariate_categorical_{column}.csv", index=False)
    for column, table in univariate["flag_tables"].items():
        table.to_csv(output_dir / f"univariate_flag_{column}.csv", index=False)
    univariate["flags_table"].to_csv(output_dir / "flags_n_flags.csv", index=False)
    univariate["flag_effects"].to_csv(output_dir / "flag_effects.csv", index=False)
    structure.to_csv(output_dir / "cardinality_unseen.csv", index=False)
    digits["discovery"].to_csv(output_dir / "digit_discovery.csv", index=False)
    digits["confirmation"].to_csv(output_dir / "digit_confirmation.csv", index=False)
    digits["stability"].to_csv(output_dir / "digit_stability.csv", index=False)
    pairs["discovery"].to_csv(output_dir / "pair_discovery.csv", index=False)
    pairs["confirmation"].to_csv(output_dir / "pair_confirmation.csv", index=False)
    pairs["stability"].to_csv(output_dir / "pair_stability.csv", index=False)
    shift.to_csv(output_dir / "train_test_shift.csv", index=False)
    write_json(output_dir / "adversarial.json", adversarial)
    write_json(output_dir / "findings.json", findings)
    _write_report(output_dir, imbalance, univariate, structure, digits, pairs, shift,
                  adversarial, findings)
    return {"imbalance": imbalance, "findings": len(findings),
            "discovery_rows": int(len(discovery)), "confirmation_rows": int(len(confirmation)),
            "adversarial_mean_auc": adversarial["mean_auc"],
            "report": str(output_dir / "eda_report.md")}
