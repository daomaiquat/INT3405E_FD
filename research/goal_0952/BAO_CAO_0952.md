# Predicting Credit Card Fraud · EDA và thử nghiệm hướng tới ROC AUC 0,952

Mục tiêu dùng ROC AUC, tức khả năng xếp một giao dịch gian lận cao hơn một giao dịch hợp lệ. Tôi lấy tập giữ lại seed 4242 của Grandmaster làm phép so sánh chính. Tập này có 104.994 giao dịch và 1.753 ca gian lận. Các mô hình dùng để đánh giá trên tập này chỉ học từ 594.966 giao dịch phát triển. Dự án đã dùng tập giữ lại này trong các lượt trước. Nó được dùng để so sánh và chọn cấu hình trong lượt này. [Cách chấm Kaggle](https://www.kaggle.com/competitions/playground-series-s6e11/overview/abstract) · [Hồ sơ chia tập gốc](../grandmaster_10fold_ablation/validation/metrics.json)

## Kết quả so với mục tiêu

**Cấu hình tốt nhất đạt ROC AUC 0,95122445.** Khoảng cách tới 0,952 là **0,00077555**. Tôi chọn 85% Grandmaster đã bổ sung XGBoost 224 vòng, 10% LightGBM bốn lá và 5% spline tương tác. Tỷ trọng áp dụng trên log-odds, tức logarit của tỷ lệ cược, rồi chuyển về xác suất. [Tệp submission](submission_grandmaster_eda.csv) · [Toàn bộ 27 cấu hình](evaluation.json)

| Phương án | ROC AUC giữ lại | Δ so với Grandmaster phân cấp |
|---|---:|---:|
| Grandmaster gốc | 0,95094666 | −0,00000227 |
| Grandmaster phân cấp số tiền | 0,95094893 | — |
| Grandmaster + XGBoost, 224 vòng với đặc trưng cũ | 0,95104590 | +0,00009697 |
| Grandmaster + XGBoost, 224 vòng với đặc trưng cũ và EDA | 0,95112021 | +0,00017128 |
| Grandmaster + XGBoost, 224 vòng lấy trung bình 5 seed | 0,95106722 | +0,00011829 |
| XGBoost 224 vòng + tổ hợp 70/20/10 có MLP | 0,95118135 | +0,00023242 |
| **XGBoost 224 vòng + LightGBM + spline** | **0,95122445** | **+0,00027552** |

- So với cây cùng 224 vòng, bổ sung đặc trưng EDA tăng 0,00007431. Cộng trực tiếp hiệu chỉnh EDA vào điểm lại giảm 0,00002369 so với Grandmaster. Kết quả ủng hộ để cây học cách dùng các tín hiệu này. Cây xếp hạng vùng điểm thấp, 448–672 vòng và trung bình năm seed đều thấp hơn công thức được chọn. [Các nhánh phần sai lệch](residual_development_metrics.json) · [Đối chứng và đường học dài hơn](residual_extended_development_metrics.json) · [Kiểm tra xếp hạng](residual_ranked_development_metrics.json) · [Lấy trung bình năm seed](residual_bagged_development_metrics.json)
- Tổ hợp thắng tăng 0,00027552 so với Grandmaster phân cấp. Khoảng DeLong ghép cặp 95% là [+0,00002221, +0,00052883], tính có điều kiện trên dự đoán cố định và chưa hiệu chỉnh việc chọn phương án theo chính tập này. Average Precision đạt 0,53823528 và log-loss 0,04383599. Các số đo là kiểm chứng local trên tập giữ lại.
- Các ca gian lận nằm trong 90% giao dịch có điểm Grandmaster thấp nhất đóng góp 0,00026051 AUC, khoảng 95% mức tăng ròng. Phần đóng góp là thay đổi tỷ lệ xếp đúng giữa từng ca gian lận trong nhóm đó và toàn bộ giao dịch hợp lệ giữ lại.
- Đối chứng Grandmaster phân cấp đạt 0,95128092 trên đánh giá khảo sát 10 fold toàn bộ train với tầng gộp dùng ma trận OOF chung. Đánh giá này khác quy trình giữ lại 15% của bảng trên. Khi xuất submission, các nhánh được tái lập hoặc huấn luyện với toàn bộ 699.960 giao dịch hợp lệ theo thiết kế từng mô hình.

## EDA làm lại từ CSV gốc

| Kiểm tra | Kết quả |
|---|---|
| Quy mô và lệch nhãn | 699.960 dòng train, 299.984 dòng test, 24 đặc trưng. Có 11.689 ca gian lận, chiếm 1,670%. |
| Chất lượng và ID | Không có ô thiếu hay vô hạn. ID duy nhất. Không có bản ghi trùng trên toàn bộ 24 đặc trưng trong train, trong test hoặc giữa hai tập. ID chỉ dùng để kiểm tra, đối chiếu và xuất tệp. |
| Dịch chuyển phân bố | KS lớn nhất trên cột số 0,003825 ở `day_of_week`. Khoảng cách tổng biến thiên lớn nhất trên cột phân loại và cờ 0,002310 ở `auth_method`. |
| Tín hiệu đơn biến ngoài fold | `merchant_risk_score` 0,741200, `velocity_score` 0,706536, `cvv_retry_count` 0,705663, `auth_method` 0,684998. CVV có Average Precision cao nhất trong 24 biến, đạt 0,085704. |
| Khóa số thưa | 6.668 dòng test có số dư chính xác chưa thấy trong train. Với số tiền là 1.617 dòng. Dữ liệu có nhiều mức làm tròn, nên khóa hiếm cần điều chuẩn hoặc quay về nhóm rộng hơn. |

[Kiểm tra toàn bộ dữ liệu](data_audit.json) · [Điểm đủ 24 biến](univariate_signal.json) · [Biểu đồ EDA](eda_overview.png)

Tôi dùng dự đoán ngoài fold của Grandmaster trên tập phát triển để tìm tín hiệu còn sót. Với 276 cặp biến, tôi bỏ phần giải thích bởi từng biến riêng, rồi ước lượng hướng hiệu chỉnh từ một nửa dữ liệu và kiểm trên nửa còn lại, theo cả hai chiều. Mỗi ô phải đủ số dòng và độ hỗ trợ theo xác suất dự kiến. 32 cặp cho mức giảm log-loss xấp xỉ dương ở bước sàng lọc. Kết quả này dùng để chọn giả thuyết huấn luyện.

| Cặp | Ô đủ hỗ trợ | Spearman hai nửa | Mức giảm log-loss xấp xỉ × 1 triệu |
|---|---:|---:|---:|
| Số giao dịch 24 giờ × loại thẻ | 37 | 0,513 | 11,42 |
| Giờ từ giao dịch trước × lệch quốc gia IP | 24 | 0,454 | 6,62 |
| Số lần thử CVV × loại thẻ | 14 | 0,868 | 5,72 |
| Thiết bị × merchant mới | 16 | 0,347 | 4,60 |
| Điểm rủi ro merchant × lệch địa chỉ thanh toán/giao hàng | 20 | 0,338 | 4,19 |

- Sau khi lấy Grandmaster làm mốc, các khóa chính xác của thời gian, velocity và điểm rủi ro merchant chưa cho mức giảm log-loss lặp lại qua hai nửa. Chữ số cuối của khoảng cách cho kết quả dương trong bước sàng lọc. Tôi dùng tín hiệu này cùng các cặp đã chọn để thử huấn luyện.
- 90% giao dịch có điểm dự đoán thấp nhất chứa 1.594 ca, chiếm 16% số ca gian lận. Chúng đóng góp khoảng 76% số cặp gian lận–hợp lệ mà Grandmaster xếp sai trên tập phát triển. Vì vậy tôi thử riêng một hàm mất mát xếp hạng theo cặp, ưu tiên vùng điểm thấp, và so với cây học phần sai lệch thông thường.

[Kết quả sàng lọc, độ hỗ trợ và phân tích lỗi xếp hạng](residual_eda.json)

## Cách kết hợp các thành phần mạnh nhất

- **Nền Grandmaster.** Dùng hai GLM với spline và mã hóa chữ số, ba cây nông học phần sai lệch của GLM, XGBoost cây nông có độ phân giải cao, và EBM. Tầng gộp có trọng số không âm. Bản `hierarchical_amount` thêm điều chuẩn theo cấp số tiền cho GLM. Tái dựng xác suất đối chứng từ các checkpoint local trên đúng ID. [Grandmaster gốc](<../Model tham khảo/Grandmaster/FINAL_RESEARCH_REPORT.md>) · [Manifest phân cấp số tiền](../grandmaster_10fold_ablation/submission_hierarchical_amount/manifest.json)
- **Nhánh cây mới.** Học phần sai lệch trên log-odds của toàn bộ tổ hợp Grandmaster. Dùng đặc trưng gốc, mã hóa chữ số và target encoding của `XGB-D03`. Bổ sung 12 cặp từ EDA và chữ số cuối của khoảng cách. Các thống kê đặc trưng có nhãn cho một dòng train được học trên các fold con khác. So với XGBoost cùng cấu hình chỉ dùng các đặc trưng đã có và so 96, 224, 448, 672 vòng.
- **Đa dạng mô hình.** Huấn luyện lại MLP-BCE trên các fold Grandmaster, spline tương tác và LightGBM bốn lá theo thiết kế D03. Bản LightGBM dùng CPU với 1.600 vòng và tốc độ học 0,05. Thử lại công thức trộn xác suất 70/20/10 từ lượt trước và các tỷ lệ nhỏ của spline. Nhánh hỗn hợp phân bố rời rạc theo lớp kiểm tra thêm mô hình tạo sinh cho các kiểu giao dịch tiềm ẩn. [Các thí nghiệm 5 fold trước](../novel_model_experiments/BAO_CAO.md)

## Tái lập

Chạy từ `goal_0952`. Dùng các môi trường Python local của dự án đã có sẵn. Nhánh Grandmaster dùng các checkpoint local tạo từ CSV chính thức, và toàn bộ pipeline chạy ML thông thường.

```powershell
$cpu = '..\grandmaster_refit\.refit-env\Scripts\python.exe'
$gpu = '..\s6e11_gpu\.build\venv\Scripts\python.exe'
& $cpu -B eda_from_scratch.py
& $gpu -B mlp_branch.py development
& $gpu -B mlp_branch.py full
& $cpu -B spline_branch.py
& $cpu -B lightgbm_branch.py
& $gpu -B latent_profiles.py development
& $gpu -B latent_profiles.py full
foreach ($mode in @('development', 'full')) {
    & $cpu -B residual_models.py $mode
    & $cpu -B residual_models.py $mode --extended
    & $cpu -B residual_models.py $mode --ranked
    & $cpu -B residual_models.py $mode --bagged
}
& $cpu -B evaluate_and_export.py --extended --bagged
& $cpu -B -m unittest discover -v
```

## Phạm vi nguồn đã đọc

- `train.csv`, `test.csv`, `sample_submission.csv` đã được đọc toàn bộ để audit, tính phân phối, xác minh lưới số, huấn luyện và kiểm tra danh tính dòng. Sàng lọc phần sai lệch có nhãn dùng 594.966 dòng phát triển. EDA tổng quan dùng mọi dòng và trường dữ liệu chính thức.
- Đã đọc toàn bộ `eda_review/ket_qua_eda.md`, `grandmaster_refit/run_refit.py`, `grandmaster_10fold_ablation/run_ablation.py`, và `grandmaster_10fold_ablation/evaluate_ablation.py`. Đã đối chiếu toàn bộ manifest của Grandmaster `hierarchical_amount`, metadata tập giữ lại và các mảng điểm từng fold dùng cho lượt này.
- Đã đọc mục 2–3 trong `Model tham khảo/Grandmaster/README.md`, mục 4–7 trong `FINAL_RESEARCH_REPORT.md` và các đoạn tạo đặc trưng, huấn luyện tầng gộp, tách phần sai lệch trong notebook. Điểm Grandmaster toàn bộ 10 fold dùng tầng gộp trên ma trận dự đoán OOF chung nên được ghi riêng với kết quả tập giữ lại.
