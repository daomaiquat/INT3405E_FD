# Predicting Credit Card Fraud · Từ baseline đến Grandmaster EDA

**Ứng viên cuối của nhánh Grandmaster EDA là bản hiệu chỉnh 404 mức tiền thưa cục bộ.** ROC AUC trên tập giữ lại tăng từ **0.95122445** lên **0.95131585**, Average Precision từ **0.53823528** lên **0.53852282**, log loss giảm từ **0.04383599** xuống **0.04380602**. Khoảng tin cậy 95% cho Δ AUC là **[−0.00002229, +0.00020508]**. CSV có đủ 299.984 dòng, hiệu chỉnh đúng 4.448 dòng test. Grandmaster EDA gốc là đối chứng. Mục tiêu local 0.952 còn cách 0.00068415 theo ứng viên này. [Submission ứng viên](../goal_0952/deep_eda_measurement/submission_grandmaster_eda_amount_hole.csv) · [Đối chứng](../goal_0952/submission_grandmaster_eda.csv) · [Kết quả và khoảng tin cậy](../goal_0952/deep_eda_measurement/measurement_metrics.json)

**Mốc Kaggle ghi trong kho là 0.95154 Public LB của Grandmaster tham khảo.** Bản hiệu chỉnh bậc hai theo bất đồng có AUC **0.95134001**, cao nhất trong các ứng viên trên holdout chung được lập chỉ mục. Mức tăng trên đối chứng mật độ là +0.00002777, còn Δ AUC có trọng số ở phép 10 fold đảo vai trò là −0.00002963. Ứng viên số tiền được đề xuất theo hướng Grandmaster EDA mà chủ nhân đã chọn, với một hệ số nhỏ và bằng chứng lặp lại về log loss. Điểm Kaggle của các ứng viên gần đây chưa có bản ghi xác minh trong phạm vi tổng hợp. [Kho Public LB](<../Model tham khảo/VERSION_COMPARISON.md>) · [Holdout bậc hai](../goal_0952/score_space/holdout_metrics.json) · [10 fold đảo vai trò](../goal_0952/score_space/reversed_metrics.json)

## Mục lục

| Tệp | Nội dung |
|---|---|
| [01 · Baseline và Grandmaster](01_BASELINE_VA_GRANDMASTER.md) | Thiết lập repo, baseline cây và tuyến tính, notebook GPU, D03, xử lý lệch nhãn, tiền huấn luyện, ensemble, seed/refit, so sánh V1/V2/Grandmaster. |
| [02 · EDA](02_EDA.md) | Từ điển dữ liệu, chất lượng, lệch nhãn, phân phối train–test, tín hiệu đơn biến, phần sai lệch sau mô hình và vùng lỗi xếp hạng. |
| [03 · Feature engineering](03_FEATURE_ENGINEERING.md) | Cột đầu vào, công thức, mã hóa ngoài fold, các nhóm đặc trưng theo mô hình và kết quả ablation. |
| [04 · Mô hình và thử nghiệm](04_MO_HINH_VA_THU_NGHIEM.md) | Các nhánh từ bảy thử nghiệm OOF đến mục tiêu 0.952, cải tiến qua nhiều lượt, kết quả âm và điều kiện lựa chọn. |
| [05 · Bảng điểm và tệp](05_BANG_DIEM_VA_ARTEFACT.md) | 394 dòng chỉ số từ 91 tệp kết quả, cùng danh mục 71 CSV lịch sử và hai CSV phát hành. |
| [06 · Tái lập và submission](06_TAI_LAP_VA_SUBMISSION.md) | Cách kiểm tra đầu vào, chạy hoặc xác minh từng phase, phân biệt CSV holdout với CSV Kaggle và định danh bản cuối. |
| [07 · Lịch sử chat và phase](07_LICH_SU_CHAT_VA_PHASE.md) | Các quyết định theo thời gian, phạm vi lịch sử đã đọc, từ mục tiêu ban đầu đến EDA sâu. |

Các tệp dự đoán có liên kết mở trực tiếp trong repo. Đường dẫn dạng mã của các CSV lịch sử ghi vị trí đầu ra của từng lượt chạy. Đặt dữ liệu Kaggle chính thức trong `data/` để tái lập thí nghiệm. [Hướng dẫn tái lập](06_TAI_LAP_VA_SUBMISSION.md)

## Lộ trình từ baseline tới bản cuối

Các dòng dùng chế độ đánh giá khác nhau. Đọc cột phép đo trước khi so hai AUC. Dấu — biểu thị chỉ số chưa được ghi trong nguồn gắn với dòng đó.

| Phase và mô hình | Phép đo | ROC AUC | AP | Log loss | Đặc trưng hoặc thay đổi chính |
|---|---|---:|---:|---:|---|
| B06 đối chứng nền | OOF 5 fold theo nhật ký | 0.946965 | — | — | Cấu hình nền đã chốt sau bước audit và đối chứng. [Nhật ký](../../reports/steps_progress.md) |
| S6-L04 | OOF theo nhật ký bước 6 | 0.947365 | — | — | LightGBM giới hạn 4 lá. [Nhật ký](../../reports/steps_progress.md) |
| LightGBM + TE vòng GPU 1 | OOF 5 fold | 0.94832930 | 0.51640167 | — | Thêm target encoding, tức ước lượng tỷ lệ gian lận theo nhóm trên các fold khác. [Số liệu](../s6e11_gpu/results/lightgbm-te.json) |
| Rank blend vòng GPU 2 | OOF 5 fold | 0.94847399 | 0.51860590 | — | Gộp thứ hạng các nhánh LightGBM và XGBoost. [Số liệu](../s6e11_gpu/round2/results/rank-blend.json) |
| XGB-D03 | OOF 5 fold | 0.95018230 | 0.52876265 | 0.04442177 | 109 cột từ 24 trường gốc, TE 13 trường số, phần thập phân và khóa chữ số. [Bảng điểm](../../s6e11_gpu/plan_runs/published/full/comparison.csv) |
| FINAL | OOF lồng 5×5 | 0.95034521 | 0.52937620 | 0.04434665 | GLM-NEWTON + XGB-D03 với tầng gộp học ngoài fold. [Hồ sơ FINAL](../../s6e11_gpu/plan_runs/published/full/final/metrics.json) |
| 80% FINAL + 20% D03 | Trung bình AUC OOF của hai seed | 0.95043460 | — | — | Cố định tỷ trọng LightGBM D03, 9/10 fold tăng. [Seed 1337](../s6e11_gpu/followup/results/full/seed_1337/metrics.json) · [Seed 2718](../s6e11_gpu/followup/results/full/seed_2718/metrics.json) |
| 70% FINAL + 20% D03 + 10% MLP-BCE | OOF 5 fold seed 42 | 0.95045126 | 0.53027823 | 0.04430444 | Thêm đa dạng từ mạng nơ-ron huấn luyện bằng binary cross-entropy. [Số liệu](../novel_model_experiments/results/fixed_blend_metrics.json) |
| Grandmaster tham khảo | OOF bảo vệ 10 fold / Public LB | 0.951267 / 0.95154 | — | 0.043960 OOF | 2 GLM, 3 cây học sai lệch, XGBoost riêng, EBM, tầng gộp không âm. [Kho tham khảo](<../Model tham khảo/VERSION_COMPARISON.md>) |
| Grandmaster tái lập local | OOF đối chứng 10 fold | 0.95128045 | — | 0.04395569 | Tái lập bảy nhánh. Các tệp refit toàn bộ train có thống kê test riêng. [Tái lập](../grandmaster_refit/full/metrics.json) |
| Grandmaster nguyên bản / phân cấp | Cùng holdout 104.994 dòng | 0.95094666 / 0.95094893 | 0.53814886 / 0.53818509 | 0.04387461 / 0.04387376 | Thay mã hóa Dense GLM của số tiền bằng điều chuẩn phân cấp. [Ablation](../grandmaster_10fold_ablation/validation/metrics.json) |
| Grandmaster EDA | Cùng holdout | 0.95122445 | 0.53823528 | 0.04383599 | Nhánh XGBoost offset có thêm tương tác EDA, gộp 85/10/5 với LightGBM và spline. [27 công thức](../goal_0952/evaluation.json) |
| Expanded half-density | Cùng holdout | 0.95131224 | 0.53828373 | 0.04382592 | Thử nhánh mật độ theo lớp và giao dịch lân cận, gộp nửa nhánh mở rộng. [Số liệu](../goal_0952/followup/context_metrics.json) |
| Hiệu chỉnh bậc hai theo bất đồng | Cùng holdout | **0.95134001** | 0.53864400 | 0.04381021 | Dùng tích bậc hai giữa độ lệch điểm của bảy đầu ra Grandmaster. [Số liệu](../goal_0952/score_space/holdout_metrics.json) |
| **Grandmaster EDA + số tiền thưa** | Cùng holdout | **0.95131585** | **0.53852282** | **0.04380602** | Một cờ cho 404 mức cent, một hệ số hiệu chỉnh log-odds. [Số liệu](../goal_0952/deep_eda_measurement/measurement_metrics.json) |

## Cấu hình bản cuối

- **Nền Grandmaster phân cấp.** Hồi quy logistic thưa với spline và khóa chữ số nhiều mức, hồi quy logistic dày có TE, ba bộ tăng cường LightGBM/XGBoost/CatBoost học phần sai lệch, XGBoost riêng và EBM. Tầng gộp không âm có các đầu vào sai lệch so với GLM nền. Dense GLM dùng điều chuẩn phân cấp cho số tiền. [Kiến trúc](<../Model tham khảo/Grandmaster/FINAL_RESEARCH_REPORT.md>) · [Mã phân cấp](../grandmaster_10fold_ablation/improvement_features.py)
- **XGBoost học phần sai lệch.** Điểm nền được đưa vào dưới dạng `base_margin`. Dùng **122 cột** gồm 109 cột D03, 12 cặp tương tác EDA đã trừ hiệu ứng từng biến và 1 cột chữ số cuối của khoảng cách. Huấn luyện 224 vòng, độ sâu tối đa 3, tốc độ 0,025 và L2 = 40. Mã hóa có nhãn dùng 5 fold con. [Mã D03](../../s6e11_gpu/plan_runs/features.py) · [Nhánh XGBoost](../goal_0952/residual_models.py) · [12 cặp đã chọn](../goal_0952/residual_development_metrics.json)
- **Hai nhánh bổ sung.** LightGBM D03 tối đa 4 lá, 1.600 vòng và spline logistic có các tương tác được sàng lọc trên phần dư. Tỷ trọng lần lượt 0,10 và 0,05 trên log-odds. [LightGBM](../goal_0952/lightgbm_branch.py) · [Spline](../goal_0952/spline_branch.py) · [Bộ trộn](../goal_0952/evaluate_and_export.py)
- **Hiệu chỉnh EDA số tiền.** Tập (H) gồm 404 mức cent xác định từ tần suất không dùng nhãn ở cửa sổ ±5 và ±25 cent trên tập phát triển. Học một độ dịch log-odds dùng log loss, Ridge 25 và giới hạn ±0,30. Độ dịch khi chấm holdout là +0,27198694. Độ dịch dùng xuất test sau khi ước lượng trên toàn bộ train là +0,28729082. [Quy tắc](../goal_0952/deep_eda_measurement/METHOD.md) · [Manifest bản cuối](../goal_0952/deep_eda_measurement/amount_hole_submission.json)

```text
z_EDA = 0.85 × z_OffsetXGB + 0.10 × z_LightGBM_D03 + 0.05 × z_Spline
z_final_test = z_EDA + 0.287290817291 × 1[số tiền thuộc H]
is_fraud = sigmoid(z_final_test)
```

`z` là log-odds, logarit của tỷ lệ giữa xác suất gian lận và xác suất giao dịch hợp lệ. SHA-256 của CSV ứng viên là `7068b053cb3e39a07b544c95d9f66b98958e0722d8669a82f465284062af1287`. [Mã xuất và kiểm tra CSV](../goal_0952/deep_eda_measurement/export_amount_hole.py)

## Dữ liệu và cách đọc điểm

- **Dữ liệu.** Train có 699.960 dòng, 11.689 ca gian lận, 24 đặc trưng. Test có 299.984 dòng. `id` dùng để căn dữ liệu và xuất tệp. Pipeline học từ CSV của cuộc thi và dự đoán tự tạo, chạy bằng ML thông thường theo quy tắc dự án. [Manifest đầu vào](../../data/manifest.json) · [Quy tắc](../README.md#quy-tắc-dự-án)
- **OOF.** Out-of-fold ghép dự đoán trên các fold giữ lại rồi tính AUC một lần. Trung bình AUC từng fold và trung bình qua hai seed là phép thống kê khác. Đánh giá lồng dùng vòng trong cho mã hóa, dừng sớm hoặc tầng gộp. [Quy trình gốc](../../s6e11_gpu/plan_runs/README.md)
- **Holdout và phép đảo.** Holdout Grandmaster cố định seed 4242, 104.994 dòng, 1.753 ca gian lận. Phần phát triển có 594.966 dòng. Phép 10 fold đảo vai trò trong các lượt sau thường chỉ xét nhánh thay đổi, học trên 104.994 dòng và đánh giá theo từng fold của 594.966 dòng còn lại. Vì holdout từng tham gia lựa chọn qua nhiều vòng, độ tin cậy của mức tăng nhỏ cần được xem cùng kiểm tra ngoài fold và log loss. [Hồ sơ holdout](../grandmaster_10fold_ablation/validation/metrics.json) · [Ví dụ đảo vai trò](../goal_0952/parallel_tree_calibration/reverse_metrics.json)
- **Chỉ số.** ROC AUC đo xác suất một ca gian lận được xếp trên một ca hợp lệ, hòa điểm tính nửa. Average Precision (AP) nhạy với việc đưa gian lận lên đầu danh sách. Log loss và Brier đo chất lượng xác suất và càng thấp càng tốt. Các file nộp test có chỉ số kiểm tra tệp, còn điểm theo nhãn phụ thuộc phép chấm đã ghi trong bảng. [Bảng điểm đầy đủ](05_BANG_DIEM_VA_ARTEFACT.md)

## Phạm vi nguồn

Tổng hợp các lượt có kết quả và quyết định trong năm cuộc chat về dự án từ 19 đến 26/11/2026. Đối chiếu phần tóm tắt mô hình và đặc trưng với các hàm, cấu hình và trường số liệu được dẫn. Nhật ký ghi riêng lời người dùng báo về Kaggle, kho tham khảo và kết quả được tính cục bộ. [Phạm vi từng chat](07_LICH_SU_CHAT_VA_PHASE.md) · [Nguồn và phép đo từng dòng](05_BANG_DIEM_VA_ARTEFACT.md)
