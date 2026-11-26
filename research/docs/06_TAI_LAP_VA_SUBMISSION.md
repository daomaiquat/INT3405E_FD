# Tái lập thí nghiệm và kiểm tra submission

[Về README](README.md) · [Bảng điểm và danh mục CSV lịch sử](05_BANG_DIEM_VA_ARTEFACT.md)

## Điểm vào và nguồn đầu vào

Chạy các lệnh từ thư mục gốc `INT3405E_FD/`. Mã nền nằm ở `src/` và `s6e11_gpu/`, còn các runner nghiên cứu nằm trong `research/`. Lấy ba CSV chính thức từ Kaggle và đặt vào `data/` để chạy huấn luyện hoặc đối chiếu với dữ liệu cuộc thi. Git lưu manifest SHA-256. Mã giao dịch `id` dùng để ghép hàng, kiểm tra fold và viết CSV. [Hướng dẫn dữ liệu](../../data/README.md) · [Manifest](../../data/manifest.json) · [Hàm nạp dữ liệu và checkpoint](../goal_0952/grandmaster_inputs.py) · [Quy tắc dự án](../README.md#quy-tắc-dự-án)

| Tệp | Dòng dữ liệu | Cột | Vai trò |
|---|---:|---|---|
| `data/train.csv` | 699.960 | `id`, 24 đặc trưng, `is_fraud` | Huấn luyện, tạo fold, holdout và audit có nhãn. |
| `data/test.csv` | 299.984 | `id`, 24 đặc trưng | Suy luận. Thống kê không nhãn dùng để kiểm phân bố. |
| `data/sample_submission.csv` | 299.984 | `id,is_fraud` | Xác định tên cột và thứ tự ID của mọi submission. |

Môi trường thử nghiệm dùng Python 3.12, NumPy, SciPy, pandas, scikit-learn, XGBoost, LightGBM và CatBoost. Các lượt cây chạy trên RTX 4060 theo runner, còn GLM, EBM và LightGBM tùy cấu hình có thể chạy CPU. Khi tái lập thí nghiệm, xem phiên bản thư viện và thiết bị trong manifest của lượt chạy. [Môi trường Grandmaster](../grandmaster_refit/full/metrics.json) · [Khóa phiên bản repo](../../requirements.lock.txt)

## Xác minh hai CSV phát hành

Từ gốc repo, chạy lệnh đầu với Python 3.11 trở lên để kiểm tra hai CSV trong bản clone bằng thư viện chuẩn. Chạy lệnh thứ hai sau khi đặt CSV test chính thức vào `data/` để kiểm thêm SHA-256, thứ tự ID và tập giao dịch được chọn từ 404 mức tiền. [Mã kiểm tra](../verify_release.py)

```powershell
python research/verify_release.py
python research/verify_release.py --test data/test.csv
```

Phép kiểm xác nhận SHA-256 của hai CSV, 299.984 ID cùng thứ tự và xác suất hợp lệ. Nó đối chiếu 4.448 dòng tăng đúng theo hệ số log-odds và xác nhận các dòng còn lại giữ nguyên chuỗi xác suất. Manifest lưu riêng hệ số chấm holdout và hệ số xuất test. [Manifest bản cuối](../goal_0952/deep_eda_measurement/amount_hole_submission.json)

| Thành phần | SHA-256 |
|---|---|
| `train.csv` | `87eaa10b8d17a7e4f657b8da1c66242431338bdd8b6ad45861521ac0c0e9fbd5` |
| `test.csv` | `10f516959020357b565abaaa4fd25baae97fc2f536d0805695bb7cc15c1660f0` |
| `sample_submission.csv` | `13fc6ed8b73de5329884ec490f2a0b750cf29a7cda772b78cfa0cf486f6f35fa` |
| Grandmaster EDA gốc | `9aa21cdd324a0913c8f029dc4ebe1efcd3f51a8323c77ce93f13f283eba6f3d0` |
| Grandmaster EDA + số tiền thưa | `7068b053cb3e39a07b544c95d9f66b98958e0722d8669a82f465284062af1287` |

[Manifest đầu vào](../../data/manifest.json) · [Manifest Grandmaster EDA](../goal_0952/evaluation.json) · [Manifest bản cuối](../goal_0952/deep_eda_measurement/amount_hole_submission.json)

### Mã tối giản để tái tạo xác suất bản cuối

Chạy từ gốc repo với Python đã cài pandas, NumPy và SciPy. Ví dụ dùng bản điểm Grandmaster EDA đã lưu, `data/test.csv`, `data/sample_submission.csv` và hệ số trong manifest để tính lại xác suất của bản cuối. Lệnh `verify` của runner xuất còn đối chiếu SHA-256 của cả ba CSV đầu vào và của các tệp dự đoán. [Runner đầy đủ](../goal_0952/deep_eda_measurement/export_amount_hole.py)

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

data = Path("data")
goal = Path("research") / "goal_0952"
lane = goal / "deep_eda_measurement"
manifest = json.loads((lane / "amount_hole_submission.json").read_text(encoding="utf-8"))
test = pd.read_csv(data / "test.csv", usecols=["id", "amount_usd"])
sample = pd.read_csv(data / "sample_submission.csv")
assert test["id"].equals(sample["id"])

with np.load(goal / "selected_predictions.npz", allow_pickle=False) as saved:
    np.testing.assert_array_equal(saved["test_ids"], test["id"])
    probability = saved["test"].copy()

scaled_amount = test["amount_usd"].to_numpy() * 100
amount_ticks = np.rint(scaled_amount).astype(np.int64)
on_grid = np.abs(scaled_amount - amount_ticks) <= 1e-6
affected = on_grid & np.isin(amount_ticks, manifest["frozen_amount_ticks"])
assert int(affected.sum()) == manifest["updated_test_rows"] == 4_448
probability[affected] = expit(logit(probability[affected]) + manifest["full_refit_logit_shift"])

submission = pd.read_csv(lane / manifest["file"])
assert submission["id"].equals(sample["id"])
np.testing.assert_allclose(submission["is_fraud"], probability, atol=1e-14, rtol=0)
print(f"Verified {len(submission):,} probabilities")
```

## Các runner theo phase

| Phase | Chạy hoặc đọc từ | Mã và kết quả |
|---|---|---|
| Audit, fold, EDA và baseline | Từ gốc repo, runner chọn `--stage audit`, `folds`, `eda`, `contract`, `baseline`, `complexity`, `encoding` hoặc `te_models`. | [Runner](../../s6e11_project_run.py) · [Cách chạy](../../README.md) · [Tiến độ](../../reports/steps_progress.md) |
| GPU 5 fold và FINAL | `run_plan.py` có `--stage te/features/advanced/all`, `--smoke`, `--data-dir` và `--output`. Quy trình xác nhận FINAL được mô tả riêng. | [Runner](../../s6e11_gpu/plan_runs/run_plan.py) · [README GPU](../../s6e11_gpu/plan_runs/README.md) · [Bảng điểm](../../s6e11_gpu/plan_runs/published/full/comparison.csv) |
| Lệch nhãn, tiền huấn luyện, ablation | XGBoost và MLP cho lệch nhãn · masked reconstruction, SCARF và nối embedding · so nhóm TE cặp và mức hiếm. | [XGBoost](../s6e11_gpu/imbalance/run_xgb.py) · [MLP](../s6e11_gpu/imbalance/run_mlp.py) · [Tiền huấn luyện](../s6e11_gpu/pretraining/run_pretraining.py) · [Ablation](../s6e11_gpu/final_feature_optimization/run_optimization.py) |
| Nhiều seed và refit | Các runner dưới `s6e11_gpu` lưu dự đoán từng thành viên, mã seed encoder/mô hình và số vòng. | [Seed XGBoost](../s6e11_gpu/seed3_improvement/run_seed_bagging.py) · [Refit mô hình đơn](../s6e11_gpu/single_full_refit/run_full_refit.py) · [Refit blend](../s6e11_gpu/blend_full_refit/run_blend_refits.py) · [Seed LightGBM](../s6e11_gpu/lightgbm_seed5/run_lgbm_seed5.py) |
| Grandmaster | Tái lập toàn bộ bằng `run_refit.py`. Phép ablation tách holdout bằng `run_ablation.py`. Mỗi runner có `--smoke` cho cấu hình nhỏ. | [Refit](../grandmaster_refit/run_refit.py) · [Ablation](../grandmaster_10fold_ablation/run_ablation.py) · [Mã xuất phân cấp](../grandmaster_10fold_ablation/export_hierarchical_submission.py) |
| Bảy hướng mới 5 fold | Các runner bên cạnh báo cáo cho spline, occupancy, copula, lân cận, giờ giao dịch và blend. Kiểm tra nhiều seed cho blend cố định. | [Báo cáo và lệnh](../novel_model_experiments/BAO_CAO.md) · [Mã blend](../novel_model_experiments/fixed_blend.py) · [Mã kiểm tra](../novel_model_experiments/verify_results.py) |
| Grandmaster EDA 85/10/5 | Tạo EDA, các nhánh `development/full`, rồi đánh giá và xuất. `evaluation.json` xác nhận công thức được chọn. | [EDA](../goal_0952/eda_from_scratch.py) · [XGB](../goal_0952/residual_models.py) · [LightGBM](../goal_0952/lightgbm_branch.py) · [Spline](../goal_0952/spline_branch.py) · [Xuất](../goal_0952/evaluate_and_export.py) |
| Mật độ, đặc trưng đồng cảnh, kiểm tra độ ổn định | Các thư mục `followup`, `robust_validation`, `data_signals` và `score_space` trong `goal_0952`. | [Followup](../goal_0952/followup/BAO_CAO_FOLLOWUP.md) · [Độ ổn định](../goal_0952/robust_validation/BAO_CAO_DO_ON_DINH.md) · [Đồng cảnh](../goal_0952/data_signals/BAO_CAO_DAC_DIEM.md) · [Không gian điểm](../goal_0952/score_space/BAO_CAO_HUONG_KHAC.md) |
| Refinement, rủi ro và ba hướng song song | Giữ Grandmaster EDA làm mốc. Các phase `development/reverse/export` đã được lưu và xác minh riêng. | [Refinement](../goal_0952/grandmaster_eda_refinement/refine_eda.py) · [Theo rủi ro và cặp bổ sung](../goal_0952/grandmaster_eda_risk/risk_eda.py) · [Ba hướng](../goal_0952/BAO_CAO_THU_SONG_SONG.md) |
| EDA sâu và bản cuối | Sàng lọc nhóm giá trị, audit hành vi, kiểm tương tác ba biến, phân rã lỗi xếp hạng và xuất một hiệu chỉnh. | [Báo cáo](../goal_0952/BAO_CAO_EDA_SAU.md) · [Sàng lọc](../goal_0952/deep_eda_measurement/measurement_eda.py) · [Bản cuối](../goal_0952/deep_eda_measurement/export_amount_hole.py) |

### Ví dụ kiểm tra repo và chạy GPU

Chạy từ thư mục `INT3405E_FD/`. Chỉ định môi trường Python đã cài thư viện theo repo khi chạy thật.

```powershell
python .\s6e11_project_run.py --stage audit
python .\s6e11_project_run.py --stage folds
python .\s6e11_project_run.py --stage contract --threads 4
python .\s6e11_gpu\plan_runs\run_plan.py --stage te --only-te T01 T02 T03 T04 --smoke
```

Phép smoke dùng tập con và số fold giảm. Kết quả toàn bộ dẫn trong chương baseline và bảng chỉ mục.

### Ví dụ tái lập các báo cáo EDA mới

Chạy từ gốc repo sau khi cài môi trường nghiên cứu và tạo các checkpoint (kết quả trung gian) của Grandmaster. `measurement_eda.py` dựng lại EDA số. Lệnh `export` ghi lại CSV ứng viên và manifest vào thư mục nghiên cứu.

```powershell
python research/goal_0952/deep_eda_measurement/measurement_eda.py
python research/goal_0952/deep_eda_behavior/audit_behavior.py
python research/goal_0952/deep_eda_triples/triple_eda.py
python research/goal_0952/deep_eda_triples/plot_triples.py
python research/goal_0952/deep_eda_triples/rank_errors.py
python research/goal_0952/deep_eda_triples/audit_amount_holes.py
python research/goal_0952/deep_eda_measurement/export_amount_hole.py export
python research/goal_0952/deep_eda_measurement/export_amount_hole.py verify
```

Để chạy lại EDA sâu, cần các checkpoint theo fold ở `research/grandmaster_10fold_ablation/validation/artifacts/`, `research/grandmaster_refit/full/artifacts/` và `research/grandmaster_10fold_ablation/submission_hierarchical_amount/artifacts/`. Đây là đầu ra cục bộ cần tái tạo qua refit, ablation và xuất phân cấp. Nhánh Grandmaster EDA cũng đọc dự đoán trung gian của các mô hình thành phần. Bản phát hành lưu `research/goal_0952/selected_predictions.npz` với `holdout_ids`, `holdout`, `test_ids` và `test`. Khi tái lập trên máy khác, đối chiếu SHA-256 đầu vào, phiên bản thư viện và sai số log-odds của nhánh GPU. [Chuỗi huấn luyện](../goal_0952/BAO_CAO_0952.md) · [Mã khôi phục đối chứng](../goal_0952/followup/run_followup.py)

## Chọn đúng tệp Kaggle

| Tệp | Vai trò |
|---|---|
| [Grandmaster EDA gốc](../goal_0952/submission_grandmaster_eda.csv) | Đối chứng đúng của nhánh 85/10/5. |
| [Grandmaster EDA + số tiền thưa](../goal_0952/deep_eda_measurement/submission_grandmaster_eda_amount_hole.csv) | Ứng viên mới của nhánh người dùng chọn. |
| Expanded half-density (`goal_0952/followup/submission_context.csv`) | Ứng viên mật độ ổn định hơn các biến thể trực tiếp ở một số phép thử. |
| Bậc hai theo bất đồng (`goal_0952/score_space/submission_quadratic.csv`) | AUC cao nhất trên holdout chung trong sổ đo · phép đảo vai trò cho Δ khác dấu. |
| Grandmaster phân cấp (`grandmaster_10fold_ablation/submission_hierarchical_amount/submission.csv`) | Test CSV của biến thể trước Goal 0.952. |
| Dự đoán holdout 104.994 dòng (`grandmaster_10fold_ablation/validation/holdout_predictions.csv`) | Dùng AUC, AP, log loss nội bộ với nhãn train và ID holdout. |

Submission hợp lệ có cột `id,is_fraud`, 299.984 ID duy nhất theo đúng thứ tự mẫu, giá trị hữu hạn trong [0,1] và SHA-256 khớp manifest. Hai CSV có liên kết trong bảng là tệp phát hành. Các đường dẫn còn lại ghi nhận vị trí của kết quả lịch sử trong lần chạy tương ứng. [Danh mục](05_BANG_DIEM_VA_ARTEFACT.md)

## Phạm vi nguồn

Đã đọc mục cấu trúc và cách chạy trong README repo, các khai báo CLI của runner GPU, Grandmaster, EDA và xuất ứng viên, các mục định danh dòng, công thức và kết quả trong các manifest được dẫn. Các lệnh trên trích từ runner và báo cáo có sẵn. Việc tổng hợp tài liệu xác minh đường dẫn, số liệu, cấu trúc Markdown và CSV ứng viên từ các tệp lưu cục bộ.
