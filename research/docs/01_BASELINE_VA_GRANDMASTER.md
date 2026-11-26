# Từ baseline đến Grandmaster

Vòng xây lại pipeline S6E11 đưa ROC AUC OOF từ **0,946965** của XGBoost B06 đến **0,950345** của tổ hợp GPU `FINAL` qua xác nhận lồng 5×5. Kho tham chiếu có một dòng phát triển riêng, với `Grandmaster` đạt **0,951267** trên CV 10 fold đã khóa và **0,95154** Public LB theo bản lưu nộp Kaggle. Khi lấy Grandmaster làm đối chứng và thử tiếp trên tập giữ lại 15%, `hierarchical_amount` chỉ tăng **0,00000227 AUC**. Mỗi điểm dưới đây đi kèm cách chia dữ liệu và nguồn tương ứng.

Các pipeline thi đấu huấn luyện từ CSV chính thức bằng mô hình học máy trên dữ liệu bảng. Quy tắc dự án giới hạn dữ liệu và đầu ra mô hình ở nguồn này, với toàn bộ fit và dự đoán chạy độc lập với LLM. [Quy tắc dự án](../README.md#quy-tắc-dự-án)

## 1. Các đối chứng và quy trình trước khi thêm đặc trưng

Kế hoạch ngày 17/11/2026 ghi bốn đối chứng từ pipeline cũ. Trên phép đo OOF gộp, XGBoost Optuna đạt 0,946986008, LightGBM Optuna 0,946968350, trung bình hạng ba cây 0,947291807 và logistic stack năm mô hình 0,947251050. XGBoost Optuna có mean AUC 5 fold 0,947023036, khác điểm gộp do cách tổng hợp. Các số này là mốc lịch sử do kế hoạch ghi lại. Git của `INT3405E_FD` xác nhận các mốc xuất bản `ba8a24d` (pipeline), `5e3026b` (kết quả GPU), `9e28db9` (tái lập GPU) và `a1ce856` (đóng gói mã mô hình). [Kế hoạch, mục 3](../../plan.md#3-hiện-trạng-thứ-gì-được-kế-thừa) · [README dự án](../../README.md)

Trước khi chọn mô hình, runner kiểm tra 699.960 dòng train với 11.689 ca gian lận, 299.984 dòng test, schema, ID và checksum. Thiết kế chính dùng 5 fold phân tầng seed 42, seed mô hình 42, seed dừng sớm 7 và seed encoder 2026. Mỗi lượt chia thành tập huấn luyện ngoài `T` và tập đánh giá ngoài `V`. Với boosting, runner tách 10% `T` làm `E` để chọn số vòng khi học trên `A`, sau đó huấn luyện lại với số vòng đã chọn trên toàn `T` rồi chấm `V`. ID dùng để ghép dự đoán. Smoke của runner chính lấy 20.000 dòng train, 1.000 dòng test và 2 fold để kiểm tra luồng dữ liệu, lưu bundle và tiếp tục lượt chạy dở. Nhóm `kaggle_practice` tạo dữ liệu giả lập về lỗi sản xuất, với biến như dây chuyền, nhà cung cấp và nhiệt độ, để kiểm tra notebook. Điểm test của bài thực hành dùng phép chia 14.000/3.000/3.000 dòng với 4,37% dương ở test. [Cấu hình](../../configs/default.json) · [Phân vai fold](../../src/validation.py) · [Bộ sinh dữ liệu, ô 3](../kaggle_practice/logistic-te.ipynb) · [Chỉ số thực hành GLM](../kaggle_practice/results/logistic-te-metrics.json)

### Thang B00–B07

Run Kaggle `20261117T184301Z_75acd6` chạy đủ dữ liệu. Bảng lấy số làm tròn từ phép kiểm tính lại trên dự đoán đã lưu. Baseline theo prior huấn luyện có thể cho điểm hơi khác nhau giữa các fold, nên OOF gộp của B00 là 0,499965 thay vì đúng 0,5. B02 giữ raw numeric, sáu cờ, `log1p` của số tiền và số dư, cùng one-hot cho năm cột phân loại. Mỗi fold học bộ chuẩn hóa và từ điển phân loại của nó. B03/B04 chỉ dùng 13 cột số và sáu cờ. B05/B06 dùng số, cờ và one-hot. B07 đưa cột phân loại vào CatBoost theo dạng chuỗi. [Registry mô hình](../../src/models.py) · [Mã đặc trưng](../../src/features.py) · [Biên bản tính lại](../../reports/review_steps_0_5_checks.json)

| Mã | Mô hình và cấu hình | OOF ROC AUC | Log-loss |
|---|---|---:|---:|
| B00 | Điểm hằng bằng prior, tỷ lệ gian lận trong phần train | 0,499965 | 0,084900 |
| B01 | Điểm ngẫu nhiên độc lập | 0,497404 | 1,000134 |
| B02 | Logistic L2, `C=1`, L-BFGS | 0,941879 | 0,047045 |
| B03 | Cây quyết định `depth=4`, tối thiểu 50 mẫu/lá | 0,798327 | 0,065511 |
| B04 | Random Forest 300 cây, `max_features=sqrt`, tối thiểu 5 mẫu/lá | 0,917245 | 0,053891 |
| B05 | LightGBM 15 lá, `min_child_samples=50` | 0,946349 | 0,045780 |
| **B06** | **XGBoost `hist`, độ sâu 3, `min_child_weight=5`** | **0,946965** | **0,045490** |
| B07 | CatBoost độ sâu 4, `l2_leaf_reg=3`, Bernoulli subsample 0,8 | 0,945808 | 0,045804 |

Ba mô hình boosting đặt learning rate 0,03, trần 6.000 vòng, patience 200. Biên bản ghi thiếu source snapshot của Step 5 và 7 lỗi `PermissionError` trong 35 lần chạy test kiểm tra lại. Các số OOF/log-loss của cả tám cấu hình được tính lại trong cùng biên bản. [Cấu hình baseline](../../configs/default.json) · [Biên bản tính lại](../../reports/review_steps_0_5_checks.json)

### Cây nông và lượng dữ liệu

Step 6 `20261118T091556Z_367a2c` giữ bộ đặc trưng raw/flags/one-hot và kiểm tra độ phức tạp. LightGBM 4 lá tăng **0,001015**, thắng 5/5 fold và giảm chênh AUC train–OOF từ 0,01317 xuống 0,00545 so với 15 lá. XGBoost độ sâu 2 tăng **0,000352**, thắng 4/5 fold và giảm chênh từ 0,00780 xuống 0,00453 so với độ sâu 3. Cây nông cần nhiều vòng hơn, với trung vị 2.734 vòng cho LightGBM 4 lá và 3.137 vòng cho XGBoost độ sâu 2. Ở learning curve, khi dùng 10%, 25%, 50%, 100% của `T` trong một fold, LightGBM 4 lá lần lượt đạt 0,940543, 0,942791, 0,944187 và 0,945818 AUC trên cùng `V`. Đây là bằng chứng dùng thêm dữ liệu có ích trong fold vừa đo. [Nhật ký Step 6](../../reports/steps_progress.md#kết-quả-bước-6) · [Kiểm tra kết quả](../../reports/step6_review_checks.json) · [Runner độ phức tạp](../../src/complexity.py)

| LightGBM 4 lá | 8 lá | 15 lá | 31 lá | XGBoost depth 2 | depth 3 |
|---:|---:|---:|---:|---:|---:|
| **0,947365** | 0,946754 | 0,946350 | 0,946221 | **0,947317** | 0,946965 |

Step 7 `20261118T134537Z_0364fc` xây target encoding (TE), tức dùng tỷ lệ gian lận đã làm trơn của các giá trị cùng khóa. 13 cột số dùng khóa `round(100*x)`, `m=20`, 5 inner fold với seed 2026. Trong `T` và `A`, mỗi dòng nhận TE từ các inner fold còn lại. `V`/test dùng thống kê trên `T`, còn `E` dùng `A`. Hoán vị nhãn rồi dựng lại đạt 0,500843 AUC. Phép minh họa rò rỉ cho 0,875117 khi làm đúng và 0,913150 khi để lộ nhãn, với `valid_registry=false`. Runner chính có cache và smoke `T01–T03` đã kiểm tra đồng nhất từ raw đến bundle. Số liệu huấn luyện đủ dữ liệu `T01–T04` dưới đây đến từ runner GPU riêng. [Nhật ký Step 7–8](../../reports/steps_progress.md#bước-7--target-encoding-full-data-local) · [Mã TE](../../src/target_encoding.py) · [Hồ sơ smoke T01–T03](../../reports/phase8_smoke_checks.json)

Tái lập từ `INT3405E_FD` với CSV gốc trong `data/` hoặc dùng `--data-dir`. `--stage contract` tự bật smoke. `--stage te_models` đọc cache Step 7 tương thích với dữ liệu, fold và tham số. [Runner](../../s6e11_project_run.py) · [Hướng dẫn](../../README.md)

```powershell
python .\s6e11_project_run.py --stage contract --threads 4
python .\s6e11_project_run.py --stage baseline --all-baselines --threads 4
python .\s6e11_project_run.py --stage complexity --threads 4
python .\s6e11_project_run.py --stage encoding --threads 4
python .\s6e11_project_run.py --stage te_models --te-models T01 T02 T03 --threads 4
```

## 2. Hai vòng notebook GPU và pipeline GPU theo kế hoạch

Hai vòng notebook RTX 4060 dùng 699.960 dòng và 5 fold phân tầng seed 42. Vòng 1 so các mô hình riêng trên cùng fold, vòng 2 thử thêm họ mô hình và gộp hạng. LightGBM dùng OpenCL. XGBoost, CatBoost và mô hình PyTorch trong các vòng này chạy GPU. AP (Average Precision) tóm tắt precision–recall cho lớp gian lận. [Vòng 1 và mã notebook](../s6e11_gpu/results/comparison.md) · [Vòng 2 và mã notebook](../s6e11_gpu/round2/results/comparison.md)

| Vòng | Mô hình | OOF AUC | AP |
|---|---|---:|---:|
| 1 | LightGBM 4 lá + TE | **0,948329** | 0,516402 |
| 1 | XGBoost depth 2 + TE | 0,948311 | 0,518785 |
| 1 | Logistic + TE | 0,947300 | 0,513789 |
| 1 | MLP với embedding phân loại | 0,941456 | 0,486874 |
| 1 | FT-Transformer | 0,938332 | 0,474786 |
| 2 | Gộp hạng LightGBM/XGBoost 50/50 | **0,948474** | 0,518606 |
| 2 | XGBoost + TE theo hai cặp | 0,948193 | 0,518245 |
| 2 | CatBoost GPU + TE số | 0,947156 | 0,517428 |
| 2 | TabM-mini | 0,942350 | 0,489214 |

Đối chứng chạy lại vòng 2 ghi LightGBM 0,948323 và XGBoost 0,948311. Gộp hạng tăng ở cả 5 fold so với từng thành phần. TE theo cặp làm giảm 0,000118 AUC khi so trực tiếp với XGBoost chạy lại. [Kết quả đầy đủ](../s6e11_gpu/round2/results/comparison.md)

### Từ T01 đến XGB-D03

Pipeline theo kế hoạch dùng cùng 5 fold seed 42. Mỗi chênh lệch dưới đây dùng đối chứng của chính runner GPU, có thể khác nhẹ với Step 5–6 hoặc các notebook GPU độc lập. GLM dùng log-odds của TE, cây dùng tỷ lệ được mã hóa. [Kết quả GPU chi tiết](../s6e11_gpu/plan_runs/results/full/comparison.md) · [Định nghĩa thí nghiệm](../../s6e11_gpu/plan_runs/run_plan.py) · [Mã đặc trưng](../../s6e11_gpu/plan_runs/features.py)

| Phép thử | Kết quả OOF AUC | Kết luận trong thử nghiệm |
|---|---|---|
| `T01`, `T02`, `T03` | 0,947261, 0,948527, 0,948495 | TE 13 cột lần lượt tăng 0,005371 cho GLM, 0,001295 cho LightGBM, 0,001356 cho XGBoost |
| `T04`, `T01-M50` | 0,948574 và 0,947208 | LightGBM thêm TE của đủ 24 cột chỉ tăng 0,000047. Đổi `m=20` thành `m=50` ở GLM giảm 0,000053 |
| `D01` / `D02` / `D03` trên LightGBM | 0,948784 / 0,950124 / 0,950131 | Cent riêng, nhóm khóa chữ số, rồi dùng cả hai trên nền `T04` |
| `XGB-D01` / `XGB-D02` / `XGB-D03` | 0,948623 / 0,950074 / **0,950182** | Cùng ba phép thử trên nền `T03`. `XGB-D03` là cây đơn tốt nhất trong bảng chính |
| `P1` / `P1-P2` / `P1-P2-P3` | 0,947484 / 0,947581 / 0,947385 | Bốn cặp đầu và bốn cặp thứ hai có ích cho GLM. Năm cặp thêm ở P3 làm giảm AUC |
| `COUNT` / `DERIVED` | 0,946825 / 0,948542 | Count và biến dẫn xuất giảm lần lượt 0,001749 và 0,000032 so với `T04` |
| `GLM-C0.1` / `GLM-C0.3` / `GLM-NEWTON` | 0,947558 / 0,947605 / **0,947656** | `GLM-NEWTON` giữ `C=1` của `P1-P2`, dùng solver Newton-Cholesky và đạt điểm GLM cao nhất trong luồng |
| `TUNE-ETA005` / `TUNE-LAMBDA3` / `TUNE-NOSUB` / `XGB-D03-R125` | 0,950060 / 0,950045 / 0,950104 / 0,950146 | Learning rate, tăng phạt, bỏ lấy mẫu và tăng vòng 25% đều giảm AUC so với `XGB-D03` |
| `GLM-BOOST-LGB` / `GLM-BOOST-XGB` | 0,948623 / 0,948544 | Thêm logit GLM làm điểm khởi đầu cho cây, tăng nhẹ 0,000097/0,000050 trên hai đối chứng TE |

Khóa chữ số dùng sáu cột `amount_usd`, `hours_since_last_txn`, `distance_from_home_km`, `account_balance_usd`, `velocity_score`, `merchant_risk_score`. Nhóm có cent, phần nguyên, phần nguyên bỏ hàng đơn vị, giá trị cắt đến một số lẻ và độ dài kèm 1 hoặc 2 chữ số đầu. Mã hóa cặp `P1/P2` kết hợp loại merchant, cờ giao dịch, tuổi, tần suất và nhóm phân vị của rủi ro, số tiền, thời gian hoặc velocity. Thống kê phụ thuộc dữ liệu được học trong phần train tương ứng. Cách thử GLM-boost ở vòng GPU dùng margin train từ GLM đã khớp trên phần train tương ứng. Grandmaster về sau dựng margin train qua inner OOF để đo cách làm chặt hơn. [Bộ dựng đặc trưng](../s6e11_gpu/plan_runs/features.py) · [Luồng fit GPU](../s6e11_gpu/plan_runs/run_plan.py) · [Grandmaster, mục 6](<../Model tham khảo/Grandmaster/FINAL_RESEARCH_REPORT.md#6-key-methodological-findings--stacking-ablations-exp08>)

### Chọn `FINAL`

`FINAL` kết hợp logit chuẩn hóa từ `GLM-NEWTON` và `XGB-D03` bằng Logistic Regression. Trong mỗi fold ngoài, tầng logistic học từ OOF của mô hình gốc tạo qua 5 fold con. OOF AUC **0,950345**, AP **0,529376**, log-loss **0,044347**. So với `XGB-D03`, mức tăng **0,000163**, tăng 5/5 fold, bootstrap ghép cặp 95% **[+0,000025, +0,000291]** trên dự đoán đã chốt. Các fold lần lượt đạt 0,949878, 0,951163, 0,955174, 0,951022 và 0,944769. Lượt khảo sát trên ma trận OOF chung cho 0,950339 với hai mô hình và 0,950394 với ba mô hình. Điểm ba mô hình là kết quả khảo sát. Các phương án trung bình `ALL3-MEAN` và `TREES2-MEAN` lần lượt đạt 0,950278 và 0,950276. [Cấu hình được chọn](../../s6e11_gpu/plan_runs/published/full/final/selection.json) · [Chỉ số lồng nhau](../../s6e11_gpu/plan_runs/published/full/nested/metrics.json) · [Mã hoàn thiện](../../s6e11_gpu/plan_runs/finalize.py)

Tái lập từ thư mục gốc `INT3405E_FD` sau khi đặt CSV chính thức vào `data/`. Kết quả và hướng dẫn notebook theo từng nhóm thử có trong README GPU. [Hướng dẫn chạy](../../s6e11_gpu/plan_runs/README.md) · [Notebook tổ hợp](../../s6e11_gpu/plan_runs/notebooks/07-ensemble-final.ipynb)

```powershell
python s6e11_gpu/plan_runs/run_plan.py --stage te --only-te T01 T02 T03 --smoke --data-dir data
python s6e11_gpu/plan_runs/run_plan.py --stage all --compare-xgb-digits --tune-family xgb --seed-anchor XGB-D03 --stack-lgb D03 --stack-xgb XGB-D03 --data-dir data
```

### Các lượt sau `FINAL`

| Hướng | Kết quả và cách đọc |
|---|---|
| Gộp XGBoost nhiều seed | `SEED3-MEAN` 0,950269278. Thêm cặp seed cây/TE `71/2028` hoặc `103/2032` cho 0,950303108 và 0,950309071. Thêm cả hai thành `SEED5-MEAN` đạt 0,950331140, tăng 0,000061862 ở 5/5 fold. Khoảng bootstrap cho chênh lệch là [+0,000015747, +0,000108718], trong khi ngưỡng chọn đã đặt là +0,000100. [Điểm từng tổ hợp](../s6e11_gpu/seed3_improvement/ket_qua_seed3_mean.md) · [Runner](../s6e11_gpu/seed3_improvement/run_seed_bagging.py) |
| TE theo cặp và khóa hiếm | Trên `XGB-D03`, thêm 8 TE cặp đạt 0,949618, thêm 2 TE phân cấp cho số tiền/số dư đạt 0,950063, gộp cả hai đạt 0,949582. Tầng khóa hiếm gom tối đa 32 phân vị, dùng `m=50` về prior chung rồi `m=20` cho khóa cent về tỷ lệ của phân vị. Khi thay thành phần trong `FINAL`, cao nhất là `FINAL-RARE-TE` 0,950260. Gộp 50/50 biến thể đó với `FINAL` đạt 0,950311. [Báo cáo](../s6e11_gpu/final_feature_optimization/results/ket_qua_toi_uu_dac_trung_final.md) · [Runner](../s6e11_gpu/final_feature_optimization/run_optimization.py) |
| Lệch nhãn | `XGB-W2`, `XGB-W4`, `XGB-RANDOM` và `XGB-HARD` lần lượt đạt 0,950104 / 0,950079 / 0,950177 / 0,950141, dưới `XGB-D03` 0,950182. `XGB-HARD` đặt hệ số 3 cho 10% mẫu âm rủi ro cao nhất theo OOF nội bộ và có AP cao hơn đối chứng chọn ngẫu nhiên 0,000607. MLP BCE đạt 0,944899, focal `γ=1/2` đạt 0,944267/0,944007 và pairwise đạt 0,941213. [Bảng và cấu hình](../../s6e11_gpu/imbalance/RESULTS.md) · [Runner](../../s6e11_gpu/imbalance/README.md) |
| Tiền huấn luyện tabular | Che cột rồi học khôi phục cải thiện MLP từ 0,941240 lên 0,941555, FT-Transformer từ 0,941599 lên 0,942241 và TabNet từ 0,733782 lên 0,753023. MLP-SCARF thay một phần cột bằng giá trị từ dòng khác đạt 0,941207. Gộp 5% xác suất mô hình tiền huấn luyện vào `FINAL` giảm AUC trong cả bốn phép thử. Thêm 64 embedding MLP vào cây cho 0,949660. [Kết quả và cách chạy](../s6e11_gpu/pretraining/results/summary.md) · [Runner](../s6e11_gpu/pretraining/run_pretraining.py) |
| Chia lại CV seed 1337 và 2718 | Runner huấn luyện `FINAL` mới theo 5 outer × 3 inner fold mỗi seed. Hỗn hợp 80% `FINAL` + 20% `D03` LightGBM đạt 0,950338888 (+0,000068764, 4/5 fold) và 0,950530313 (+0,000103781, 5/5 fold), trung bình **0,950434601** trên hai cách chia. Thêm 10% `MLP-BCE` cho +0,000109419 và +0,000038581. Hiệu chỉnh riêng nhóm không bật cờ giảm khoảng 0,000012 ở cả hai seed. [Seed 1337](../s6e11_gpu/followup/results/full/seed_1337/metrics.json) · [Seed 2718](../s6e11_gpu/followup/results/full/seed_2718/metrics.json) · [Runner](../s6e11_gpu/followup/run_followup.py) |
| Hỗn hợp 70/20/10 và thử nghiệm bổ sung | 70% `FINAL` + 20% `D03` + 10% `MLP-BCE` đạt 0,95045126 ở seed 42 (+0,00010605) và tăng 5/5 fold trên từng seed 42, 1337, 2718. 90% `FINAL` + 10% spline đạt 0,95045057 ở seed 42, trọng số 10% được chọn sau khi xem fold đầu. Hiệu chỉnh thời gian đạt 0,95035247. Cây lân cận đa độ rộng 0,95029222, tương tác theo lớp 0,95022205, mật độ cặp 0,94986082 và spline tương tác riêng 0,94576566. Đây là lượt nối tiếp dòng GPU. [Báo cáo, code và CSV](../novel_model_experiments/BAO_CAO.md) |

Seed của cây/encoder thay nguồn ngẫu nhiên trong quá trình fit trên cùng fold. Seed 1337/2718 đổi cách chia outer fold của cùng tập giao dịch. **Refit 100%** là một lần học trên toàn bộ 699.960 dòng để dự đoán test. `XGB-D03` refit lấy trung vị 2.131 vòng từ CV, `D03` LightGBM dùng 1.563 vòng. Các runner refit còn xuất bản đơn `XGB-D03-S71-E2028`, `XGB-D03-R125`, `XGB-RANDOM`, `XGB-HARD` và bản gộp `FINAL`, `FINAL3`, `SEED3-MEAN`, `SEED5-MEAN`. `LGBM-SEED5-MEAN` dùng 5 thành viên LightGBM 4 lá với các cặp seed `42/2026`, `71/2026`, `42/2028`, `71/2028`, `103/2032`, mỗi thành viên đủ 1.563 vòng và trọng số 0,2. Trên test, Spearman của `LGBM-SEED5-MEAN` so với `D03` full là 0,999120. Các cột `reference_oof_auc` của bản refit dẫn lại kết quả CV đã có. Tệp test chưa gắn nhãn nên bảng xuất refit chỉ cho biết thống kê dự đoán, sự trùng ID và độ tương quan. [Refit mô hình đơn](../s6e11_gpu/single_full_refit/run_full_refit.py) · [Refit tổ hợp](../s6e11_gpu/blend_full_refit/results/summary.json) · [Hồ sơ LightGBM 5 seed](../s6e11_gpu/lightgbm_seed5/results/LGBM-SEED5-MEAN/metrics.json)

## 3. Kho tham chiếu `V1`, `V2` và `Grandmaster`

Kho `Model tham khảo` ghi ba lần nộp và ba phiên bản kiến trúc. `V1` đóng băng trước lúc lập kế hoạch xây lại pipeline ngày 17/11, nên dòng thời gian theo ngày chồng lên dòng GPU ở mục 2. Các private score để trống trong hồ sơ. Có ghi nhận của người dùng trong chat “Tóm tắt model tham khảo” rằng lần tải lại hiển thị 0,95154 cho các tệp đã thử. Ghi nhận đó chưa được đối chiếu với metadata Kaggle của các lượt tải lại, nên bảng giữ điểm theo ref và ngày của kho lưu gốc. Mục tiêu **0,953** từng được nêu trong trao đổi là mốc nghiên cứu phía trước. Nhánh `goal_0952` về sau đặt mốc **0,952** theo giao thức riêng, ở tài liệu giai đoạn kế tiếp. [Bảng đối chiếu bản lưu](<../Model tham khảo/VERSION_COMPARISON.md>) · [Mục tiêu `goal_0952`](../goal_0952/BAO_CAO_0952.md)

| Bản | Nguồn gốc lưu trong kho | Giao thức local và OOF AUC | Public LB được ghi lại |
|---|---|---|---|
| `V1` | `versions/v1_frozen/`, `ref=58183027`, 10/11/2026 | 5 fold seed 42, **0,9508989** | **0,95069** |
| `V2` | Bản tên gốc `V3`, `research_v3/submission_v3.csv`, `ref=58486891`, 18/11/2026 | 5 fold, tầng gộp lồng nhau seed 42, **0,9510771** | **0,95092** |
| `Grandmaster` | `independent_kaggle_research/`, `ref=58693191`, 22/11/2026 | CV xác nhận 10 fold seed 2026, **0,951267252810** | **0,95154** |

- **V1.** Ba nhánh là Logistic L2 trên spline/TE (0,9504425), XGBoost độ sâu 2 trung bình seed `(7, 42, 2026)` (0,9504396) và LightGBM độ sâu 2 với 4 lá (0,9504212). Dùng 36 từ điển khóa đa độ phân giải, TE làm trơn ngoài fold, one-hot, đếm tần suất và B-spline bậc 3 với 7 nút trên `merchant_risk_score`. Gộp phân vị hạng theo 45% GLM, 35% XGB và 20% LGB. CSV có 299.984 dòng, hạng trung bình 0,500002. Thang hạng phục vụ xếp thứ tự giao dịch. [README và nguồn gốc](<../Model tham khảo/V1/README.md>) · [Notebook V1](<../Model tham khảo/V1/notebook.ipynb>)
- **V2/V3.** Hai GLM `A3`/`A4` kết hợp spline 10 nút đều trên `merchant_risk_score`/`velocity_score` với spline 7 nút theo phân vị. Mã nguồn làm tròn giá trị số bằng `np.round` trên `float64` trước khi tạo khóa. TE dùng hoán vị inner độc lập theo khóa và gộp 5 seed. Các nhánh LightGBM `B2/B3/B5/B6`, XGBoost `C0/C2` và CatBoost `D1/D2` bổ sung cây siêu nông độc lập hoặc học phần dư trên GLM margin nested (`eta_off=1,0`). Gộp 10 mô hình bằng `simplex_stdlogit` (logit chuẩn hóa theo fold, trọng số không âm tổng bằng 1), tiếp theo là Platt scaling một chiều để hiệu chỉnh xác suất. Kết quả tăng 0,0001782 so với V1 và thắng 5/5 fold theo hồ sơ. Log-loss 0,0440757, PR-AUC 0,5337296, Brier 0,010693, xác suất test trung bình 0,016599. `C2_XGB_GLMBoost_5Seed` là thành phần mạnh nhất, OOF 0,9508547. [README và nguồn gốc](<../Model tham khảo/V2/README.md>) · [Notebook V2 gốc V3](<../Model tham khảo/V2/notebook.ipynb>)
- **Grandmaster.** Thiết kế độc lập từ ba CSV chính thức và chốt cấu hình trên Dev5 seed 42 trước khi xác nhận 10 fold seed 2026. Đặc trưng phân cấp chữ số thích hợp với tín hiệu rời rạc của `customer_age`, `merchant_risk_score`, `velocity_score`, `hours_since_last_txn` và cent của `amount_usd`. Đối chứng exact trên `time_of_day_hour`, `distance_from_home_km` và `card_age_months` làm giảm AUC so với spline trơn trong phép thử báo cáo. Pipeline khóa 7 mô hình thuộc 4 nhóm và dùng `ResDecomp_NonNeg`. [README Grandmaster](<../Model tham khảo/Grandmaster/README.md>) · [Báo cáo, mục 4–8](<../Model tham khảo/Grandmaster/FINAL_RESEARCH_REPORT.md>) · [Notebook Grandmaster](<../Model tham khảo/Grandmaster/notebook.ipynb>)

### Thiết kế và ablation của Grandmaster gốc

`02e_SpGLM` dựng ma trận thưa 73.277 chiều trong thí nghiệm phát triển, với indicator đa cấp theo tiền tố/hậu tố/cent, one-hot và B-spline 10 nút. Spline là các hàm cơ sở cho phép đường rủi ro uốn theo dữ liệu. `02f_DeGLM` dùng các khóa đó cùng log-odds TE thực nghiệm Bayes `m=[10,50]` qua 3 seed. Lấy trung bình hai logit làm `Blend50`. Trong mỗi fold ngoài, margin của các dòng train cho nhóm cây học phần dư đến từ inner 3-fold OOF. Ba cây phần dư, một cây độc lập và EBM bổ sung tín hiệu cho hai GLM. EBM (Explainable Boosting Machine) là mô hình cộng các hiệu ứng theo từng biến và một số cặp tương tác. [Định nghĩa và kết quả thành phần](<../Model tham khảo/Grandmaster/FINAL_RESEARCH_REPORT.md#7-final-standalone-notebook-results--learned-ensemble-s6e11_independent_grandmasteripynb>) · [Notebook, ô 6, 8 và 10](<../Model tham khảo/Grandmaster/notebook.ipynb>)

| Thành phần | Kiến trúc đã khóa | OOF AUC 10 fold |
|---|---|---:|
| `02e_SpGLM` | Logistic thưa L2 `C=0,03`, digit trie và spline | 0,949502 |
| `02f_DeGLM` | Logistic dày L2 `C=0,10`, spline và TE `m=10/50` | 0,949881 |
| `03a_OffLGB` | `Blend50` offset, LightGBM 4 lá `extra_trees`, 297 vòng, 3 seed | 0,950802 |
| `03b_OffXGB` | `Blend50` offset, XGBoost `lossguide`, 4 lá, `max_bin=1024`, 174 vòng, 3 seed | **0,950816** |
| `03c_OffCAT` | `Blend50` offset, CatBoost `depth=4`, `SymmetricTree`, 429 vòng, 3 seed | 0,950742 |
| `03d_StXGB` | XGBoost depth 2, `max_bin=1024`, cặp phân vị, 2.346 vòng, 3 seed | 0,950247 |
| `04_EBM` | EBM GA2M, `max_bins=1024`, 25 tương tác, 4 outer bags, đầu vào không TE | 0,948936 |

Ở tầng gộp, mỗi cây offset đóng góp `delta = logit_cây − logit_Blend50`. Logistic có phạt L2 và hệ số không âm học trên các dòng ngoài fold đang chấm, với mức phạt được chọn trong meta inner CV. Dùng `eps=1e-10` khi đổi sang logit để giữ thứ tự của nhiều xác suất rất nhỏ. Trên Dev5, shallow meta-tree đạt 0,949761, probit rank L2 0,950502, logit L2 0,950565, simplex không âm 0,950702, và phân rã phần dư với 14 mô hình 0,950764. Lược xuống 7 mô hình đạt **0,950793**. Bỏ các nhóm offset, EBM, GLM hoặc cây độc lập lần lượt giảm 0,000213, 0,000153, 0,000094 và 0,000063. Trong sổ thử, exact one-hot toàn bộ, pair one-hot thô, Chow–Liu và các mạng phần dư cũng được thử. Mạng PLE-ResNet, TabM và DCNv2 cho tín hiệu tương quan cao với `Blend50` trong phép sàng lọc, nên bị lược khi chốt. [Sổ thí nghiệm và ablation, mục 5–6](<../Model tham khảo/Grandmaster/FINAL_RESEARCH_REPORT.md>)

`EXP09` xác nhận thiết kế khóa với OOF **0,951265**. Notebook gốc độc lập đạt **0,951267252810**, DeLong 95% [0,949405, 0,953129], log-loss 0,043960, PR-AUC 0,535517 và Brier 0,010668. Mức tăng trên dự đoán ghép cặp so với cây đơn `03b_OffXGB` là +0,000451 (`p=1,21×10⁻⁸`). Thiết kế Dev5 huấn luyện trên 80% dữ liệu mỗi fold và Confirm10 dùng 90%, nên chênh OOF giữa hai lần xác nhận đi kèm thay đổi quy mô train. Trong notebook gốc, tầng gộp loại các dòng fold đang chấm khi fit meta trên ma trận OOF của mô hình gốc. `FINAL` GPU ở trên chạy lại hai mô hình cơ sở trong từng vòng ngoài để tạo đầu vào meta. [Báo cáo xác nhận, mục 7–8](<../Model tham khảo/Grandmaster/FINAL_RESEARCH_REPORT.md>) · [Notebook, ô 8 và 16](<../Model tham khảo/Grandmaster/notebook.ipynb>)

Để chạy lại notebook lưu trữ, sao chép notebook sang thư mục chạy, dùng CSV chính thức và thiết lập đường dẫn theo README tương ứng. V1 dùng `CONFIG["DATA_DIR"]`, V2 tự dò `data/` hoặc đường dẫn notebook theo `resolve_project_paths()`, Grandmaster dùng `DATA_DIR_OVERRIDE` nếu CSV ở thư mục riêng. V1/V2 bật `CONFIG["QUICK_RUN"]` để thử luồng 2 fold. Với Grandmaster trên PowerShell, một phiên thử luồng riêng dùng `$env:SMOKE_TEST='1'` rồi chạy lệnh dưới đây, và đưa biến về `'0'` để chạy đủ 10 fold. [V1](<../Model tham khảo/V1/README.md>) · [V2](<../Model tham khảo/V2/README.md>) · [Grandmaster](<../Model tham khảo/Grandmaster/README.md>)

```powershell
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 notebook.ipynb
```

## 4. Grandmaster tái huấn luyện tại máy local

`run_refit.py` đối chiếu SHA-256 notebook gốc và chỉ nạp bốn ô chứa cấu hình và hàm. Runner đọc ba CSV chính thức rồi tự huấn luyện bảy mô hình. Lượt full dựng lại OOF 10 fold seed 2026 trên 699.960 dòng, cho **0,9512804488775675 AUC**, PR-AUC **0,535612**, log-loss **0,043956**, Brier **0,010668** và DeLong CI 95% **[0,949419, 0,953142]**. Metadata ghi Python 3.12, XGBoost CUDA, CatBoost GPU và `interpret-core` 0.7.8. Smoke dùng 25.000 train, 5.000 test, 2 fold và ghi 0,931190 AUC của đối chứng. [Runner tái huấn luyện](../grandmaster_refit/run_refit.py) · [Full](../grandmaster_refit/full/metrics.json) · [Smoke](../grandmaster_refit/smoke/metrics.json)

| Dự đoán test trên 299.984 dòng | Trung bình | Spearman so với gộp fold | Phần top 1% trùng |
|---|---:|---:|---:|
| Gộp theo 10 fold của đối chứng local | 0,016603640 | — | — |
| Bảy mô hình gốc refit 100%, giữ tầng gộp | 0,016704831 | 0,999528214 | 98,03% |
| Bảy mô hình gốc refit 100%, fit lại meta trên toàn bộ OOF | 0,016704342 | 0,999530917 | 98,03% |

Lượt fit lại meta dùng 699.960 hàng OOF từ mô hình gốc, `L2=1e-5`, rồi áp hệ số đó vào dự đoán test do bảy mô hình refit tạo ra. Hai CSV là `submission_refit_fixed_stack.csv` (`grandmaster_refit/full/submission_refit_fixed_stack.csv`) và `submission_full_refit.csv` (`grandmaster_refit/full/submission_full_refit.csv`). OOF 0,951280 thuộc đối chứng gộp fold của chính lần chạy. Đây là số đo local riêng với 0,951267 của notebook và Public LB 0,95154 của bản tham chiếu. [Cách tính và xuất tệp](../grandmaster_refit/run_refit.py) · [Thông tin refit](../grandmaster_refit/full/metrics.json)

## 5. Ablation Grandmaster và nhánh `hierarchical_amount`

Tách `train.csv` phân tầng theo seed **4242** trước khi huấn luyện. Tập phát triển có **594.966 dòng, 9.936 fraud**. Tập giữ lại có **104.994 dòng, 1.753 fraud**. Mỗi ứng viên tạo OOF trên 10 fold seed 2026 của tập phát triển, fit meta bằng đầu vào OOF rồi dự đoán tập giữ lại chung. Smoke của riêng runner này dùng 21.250 phát triển và 3.750 giữ lại với 2 fold. Các thử nghiệm tiếp theo dùng lại phép chia 15% nhiều lần, nên các mức tăng nhỏ trên tập giữ lại cần được đọc trong phạm vi vòng nghiên cứu này. [Luồng ablation](../grandmaster_10fold_ablation/run_ablation.py) · [Hàm đánh giá](../grandmaster_10fold_ablation/evaluate_ablation.py) · [Full](../grandmaster_10fold_ablation/validation/metrics.json) · [Smoke](../grandmaster_10fold_ablation/smoke/metrics.json)

| Ứng viên | Thay đổi độc lập | AUC giữ lại | Δ so đối chứng |
|---|---|---:|---:|
| Grandmaster local | 7 đầu vào meta, cùng dữ liệu phát triển | 0,950946659 | — |
| `residual_te` | Mã hóa phần sai lệch dự đoán của hai cặp `velocity × auth` và `merchant risk × velocity` theo tối đa 16 khoảng phân vị. Dùng tổng `y−p` chia cho tổng `p(1−p)+5`, chặn giá trị về [−1,1], rồi thử trong cây XGB offset | 0,950944531 | −0,000002127 |
| `hierarchical_amount` | Bổ sung đúng một TE phân cấp vào Dense GLM. Khi khóa cent có dưới 4 mẫu và các khóa khác cùng nhóm 0,1 USD có ít nhất 4 mẫu, làm trơn tỷ lệ nhóm về prior (`m=50`), rồi cent về nhóm (`m=10`). Train dùng TE cross-fit. Khi thay logit Dense GLM ở meta, giữ nguyên định nghĩa `Blend50` của 3 cây cũ | **0,950948929** | **+0,000002271** |
| `aligned_offset` | Thay nhánh XGB offset. Trong mỗi inner fold, fit Dense GLM với cả ba seed TE của encoder ngoài thay cho một seed. Giữ margin dùng để dự đoán validation/test | 0,950945647 | −0,000001011 |

`hierarchical_amount` có DeLong CI 95% **[−0,000002713, +0,000007255]**, `p=0,3718`. Đây là mức tăng không rõ trên holdout. Runner xuất biến thể này từ 10 fold trên toàn bộ 699.960 dòng, kiểm tra lại cache của bảy mô hình, đưa Dense GLM mới vào ma trận meta và xác nhận đủ **299.984** dự đoán test. Mean test 0,016602174, thay đổi xác suất tuyệt đối trung bình 0,000010992 so với bản gộp 10 fold local. Trường `source_holdout_roc_auc` của manifest dẫn về phép thử 15%. [Công thức đặc trưng](../grandmaster_10fold_ablation/improvement_features.py) · [Mã fit nhánh](../grandmaster_10fold_ablation/branch_runner.py) · [Mã xuất CSV](../grandmaster_10fold_ablation/export_hierarchical_submission.py) · [Manifest](../grandmaster_10fold_ablation/submission_hierarchical_amount/manifest.json) · CSV (`grandmaster_10fold_ablation/submission_hierarchical_amount/submission.csv`)

### Kiểm tra tiếp trên và quanh đối chứng phân cấp

| Thử nghiệm | Đầu vào, mô hình và bằng chứng |
|---|---|
| Logistic phân cấp | Thiết kế thưa thêm hiệu ứng đồng thời ở mốc 10 USD, 1 USD, 0,1 USD và cent với phạt L2 mạnh dần ở khóa chi tiết. So bản cộng tính với bản thêm tích spline theo merchant risk/velocity/amount và nhóm ngữ cảnh. AUC riêng giữ lại 0,946503905 / 0,946599689. Meta chọn trọng số bổ sung bằng 0 cho cả hai, nên stack giữ 0,950948929. [Thiết kế](../grandmaster_10fold_ablation/hierarchical_glm_design.py) · [Runner](../grandmaster_10fold_ablation/run_hierarchical_glm_architecture.py) · [Chỉ số](../grandmaster_10fold_ablation/hierarchical_glm_architecture/metrics.json) |
| Meta học xếp hạng, chia theo rủi ro và cách gộp fold | Pairwise loss, risk-gated, gộp cặp logit và trung bình xác suất lần lượt cho 0,950923402 / 0,950944128 / 0,950948051 / 0,950945277 trên cùng holdout, dưới `hierarchical_amount`. Pairwise loss giảm 0,000025527, `p=0,0447` trong phép so được lưu. [Runner](../grandmaster_10fold_ablation/run_meta_improvements.py) · [Chỉ số](../grandmaster_10fold_ablation/meta_improvements/metrics.json) |
| Hiệu chỉnh phần dư sau stack | Hiệu chỉnh logit trên dự đoán OOF, chọn L2=100 từ tập phát triển. Holdout AUC 0,950877331 (−0,000071598), log-loss tăng từ 0,043873761 lên 0,043895047. Báo cáo phụ có nhóm OTP và velocity `[22,1; 24,0)`. [Runner](../grandmaster_10fold_ablation/run_post_stack_residual.py) · [Chỉ số](../grandmaster_10fold_ablation/post_stack_residual/metrics.json) |
| Ranker nhóm 6 cờ tắt và 0 lần thử lại CVV | Thử pointwise và các cách lấy cặp uniform, matched, balanced trên nhóm rủi ro khó. Cả bốn chọn biên độ 0,25, đạt lần lượt 0,950939558 / 0,950936470 / 0,950938442 / 0,950938796, còn đối chứng 0,950948929. Matched tăng chất lượng xếp fraud của nhóm so với mẫu âm ngoài nhóm, đồng thời làm giảm chất lượng xếp fraud ngoài nhóm so với mẫu âm trong nhóm. [Tạo nhóm](../grandmaster_10fold_ablation/specialist_features.py) · [Runner](../grandmaster_10fold_ablation/run_hard_group_ranker.py) · [Chỉ số](../grandmaster_10fold_ablation/hard_group_ranker/metrics.json) · [Balanced](../grandmaster_10fold_ablation/hard_group_ranker_balanced/metrics.json) |
| Mật độ theo lớp trên cây riêng | Lấy tỉ số khả năng theo lớp ở khóa rộng, khóa chính xác và theo ngữ cảnh cho `hours_since_last_txn`, `velocity_score`, `merchant_risk_score`, `amount_usd` và tín hiệu đột biến cục bộ theo giờ. XGB có mật độ đạt 0,948840183, hơn cây cùng cấu hình 0,000445791 (CI [−0,000084760, +0,000976343]), dưới Grandmaster 0,002108746. [Mã mật độ](../grandmaster_10fold_ablation/class_conditional_density.py) · [Runner](../grandmaster_10fold_ablation/run_density_tree.py) · [Chỉ số](../grandmaster_10fold_ablation/density_tree_validation/metrics.json) |
| Mật độ theo lớp trên cây offset | Cây học phần dư với mật độ đạt 0,950590528, đối chứng 0,950473163. Thay hẳn thành phần cây trong stack cho 0,950915672. Chỉ thêm chênh giữa hai cây qua hệ số đóng băng của `03b_OffXGB` đạt 0,950974833 (+0,000025903, CI [−0,000009348, +0,000061154]). [Runner](../grandmaster_10fold_ablation/run_density_offset.py) · [Chỉ số](../grandmaster_10fold_ablation/density_offset_validation/metrics.json) |
| Ranker vùng điểm thấp và phép chia đảo vai trò | Học trong giao thức đảo vai trò từ 104.994 dòng lịch sử, chấm trong 10 fold của 594.966 dòng còn lại. AUC trọng số theo số cặp dương–âm mỗi fold là 0,950540733 cho đối chứng, 0,950125147 cho pointwise và 0,950033907 cho pairwise. Cả hai nhánh giảm ở 10/10 fold so với đối chứng. [Runner](../grandmaster_10fold_ablation/run_low_score_ranker.py) · [Chỉ số](../grandmaster_10fold_ablation/low_score_rank_validation/metrics.json) |

Chạy các lệnh sau từ gốc repo với môi trường dùng cho Grandmaster. Bản xuất phân cấp cần kết quả full của cả refit và ablation. Các script kiểm tra dấu vết dữ liệu và thứ tự ID trước khi dùng cache. Các lệnh xuất tệp cục bộ. [Runner refit](../grandmaster_refit/run_refit.py) · [Runner ablation](../grandmaster_10fold_ablation/run_ablation.py) · [Runner xuất](../grandmaster_10fold_ablation/export_hierarchical_submission.py)

```powershell
python research/grandmaster_refit/run_refit.py --smoke
python research/grandmaster_refit/run_refit.py
python research/grandmaster_10fold_ablation/run_ablation.py --smoke
python research/grandmaster_10fold_ablation/run_ablation.py
python research/grandmaster_10fold_ablation/export_hierarchical_submission.py
```

`hierarchical_amount` trở thành một đối chứng để theo dõi mức tăng nhỏ từ trường số tiền. Các nhánh sau tiếp tục với EDA làm lại từ dữ liệu gốc, cây mật độ, spline phần dư, gộp mô hình và các tập xác nhận bổ sung. [EDA](02_EDA.md) · [Thử nghiệm sau FINAL](04_MO_HINH_VA_THU_NGHIEM.md) · [Bảng điểm và tệp chạy](05_BANG_DIEM_VA_ARTEFACT.md)

## Phạm vi nguồn đã đọc

- Đọc `plan.md` đoạn mở đầu và mục 1–3, các tiêu đề trong phần còn lại, toàn bộ `steps_progress.md`, `INT3405E_FD/README.md`, `configs/default.json`, `reports/review_steps_0_5_checks.json`, và các đoạn cấu hình hoặc tiền xử lý tương ứng trong `src/models.py` và `src/features.py`. Kiểm tra lịch sử Git hiện có trong `INT3405E_FD`. [Kế hoạch](../../plan.md) · [Nhật ký](../../reports/steps_progress.md) · [Biên bản baseline](../../reports/review_steps_0_5_checks.json)
- Đọc toàn bộ bảng đối chiếu GPU vòng 1/vòng 2, hai bản `plan_runs/.../comparison.md`, README runner GPU, báo cáo mất cân bằng, tiền huấn luyện, thêm seed, TE cặp/hiếm và báo cáo nhánh mô hình bổ sung. Đọc phần cấu hình/phân nhóm/fit trong `run_plan.py`, `features.py`, phần fit và đánh giá trong `followup/run_followup.py`, các khóa `oof_auc`, `delta_auc`, `fold_wins` của hai `metrics.json` follow-up, và hồ sơ refit tổ hợp, LightGBM 5 seed. [Báo cáo GPU](../s6e11_gpu/plan_runs/results/full/comparison.md) · [Báo cáo mô hình bổ sung](../novel_model_experiments/BAO_CAO.md)
- Đọc toàn bộ README của kho, V1, V2, Grandmaster, `VERSION_COMPARISON.md` và các mục 1–9 của `FINAL_RESEARCH_REPORT.md`. Trong các notebook lưu, đối chiếu dòng cấu hình, làm tròn, spline, TE, mô hình và tầng gộp trong các ô code V1 3/7/9/13/15/17/21/23, V2 3/7/8/10/12/14/16/18/22, Grandmaster 2/6/8/10/14/16/20. Các điểm Public LB trong bảng lấy từ metadata ghi trong README của kho. [Đối chiếu kho](<../Model tham khảo/VERSION_COMPARISON.md>) · [Báo cáo Grandmaster](<../Model tham khảo/Grandmaster/FINAL_RESEARCH_REPORT.md>)
- Đọc `grandmaster_refit/run_refit.py` phần nạp ô notebook, chia fold, refit, meta và xuất CSV, cùng toàn bộ `full/metrics.json` và `smoke/metrics.json`. Trong `grandmaster_10fold_ablation` đọc runner chính, định nghĩa và điều kiện của ba nhánh, các trường đánh giá tổng và so sánh trong 10 tệp `metrics.json` ở thư mục con, manifest xuất `hierarchical_amount` và các đoạn tái lập hiển thị trong mã runner. Bảng dùng số đo đã lưu, thể hiện đến 6 hoặc 9 chữ số tùy độ lớn chênh lệch. [Refit](../grandmaster_refit/full/metrics.json) · [Ablation](../grandmaster_10fold_ablation/validation/metrics.json) · [Manifest phân cấp](../grandmaster_10fold_ablation/submission_hierarchical_amount/manifest.json)
