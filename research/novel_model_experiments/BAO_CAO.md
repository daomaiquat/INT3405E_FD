# Predicting Credit Card Fraud · Thử nghiệm bổ sung

## Kết quả

**Ưu tiên `submission_fixed_blend.csv`.** Tổ hợp xác suất 70/20/10 đạt ROC AUC 0,95045126 trên seed 42, tăng 0,00010605 so với `FINAL`. Tổ hợp tăng ở cả 5 fold trên từng cách chia trong ba cách chia đã kiểm tra. [Kết quả đầy đủ](results/fixed_blend_metrics.json)

| Hướng | ROC AUC OOF | Δ so với `FINAL` | Tệp nộp |
|---|---:|---:|---|
| Tổ hợp cố định 70/20/10 | 0,95045126 | +0,00010605 | CSV (`novel_model_experiments/results/submission_fixed_blend.csv`) |
| Tổ hợp 90% `FINAL` + 10% spline | 0,95045057 | +0,00010536 | CSV (`novel_model_experiments/results/submission_spline_blend.csv`) |
| Hiệu chỉnh theo mốc thời gian chính xác | 0,95035247 | +0,00000726 | CSV (`novel_model_experiments/results/submission_hours_residual.csv`) |
| XGBoost với lân cận đa độ rộng | 0,95029222 | −0,00005299 | CSV (`novel_model_experiments/results/submission_neighborhood.csv`) |
| XGBoost với tương tác theo lớp | 0,95022205 | −0,00012316 | CSV (`novel_model_experiments/results/submission_copula.csv`) |
| LightGBM với mật độ cặp giá trị | 0,94986082 | −0,00048439 | CSV (`novel_model_experiments/submission_occupancy.csv`) |
| Hồi quy spline tương tác | 0,94576566 | −0,00457955 | CSV (`novel_model_experiments/submission_spline_interactions.csv`) |

| Seed của tổ hợp cố định | Δ so với `FINAL` | Fold cải thiện | Khoảng bootstrap 95% ghép cặp |
|---:|---:|---:|---:|
| 42 | +0,00010605 | 5/5 | [+0,00004118, +0,00017687] |
| 1337 | +0,00017334 | 5/5 | [+0,00010193, +0,00025526] |
| 2718 | +0,00014130 | 5/5 | [+0,00006071, +0,00022081] |

- Tổ hợp spline tăng AUC ở 5/5 fold. Mô hình spline tương tác riêng đạt 0,94576566, thấp hơn đối chứng cộng tính 0,94591771 trên cả năm fold. Đối chứng cộng tính chạm giới hạn 220 vòng tối ưu. Trọng số 10% của tổ hợp spline được chọn sau khi xem fold đầu. [Mô hình spline](spline_interactions_metrics.json) · [Tổ hợp spline](results/spline_blend_metrics.json)
- Lân cận đa độ rộng tăng 0,00011926 AUC ở 4/5 fold so với XGBoost đối chứng giữ đúng cấu hình. Tương tác theo lớp tăng 0,00004908 ở 3/5 fold. Cả hai khoảng DeLong 95% của mức tăng đều chứa 0. Mật độ cặp giá trị giảm 0,00009990 trên fold duy nhất có đối chứng LightGBM cùng cấu hình. [So sánh ghép cặp](results/verification.json) · [Đối chứng mật độ](occupancy_metrics.json)
- Hiệu chỉnh thời gian tăng AUC 0,00000726 và giảm AP 0,00003087 so với `FINAL`. Khi đo OOF, xác suất kỳ vọng lấy từ các mô hình cơ sở dự đoán ngoài fold con và tầng gộp đã khớp trên phần train ngoài. Khi xuất test, hiệu chỉnh học trên dự đoán OOF hoàn chỉnh của `FINAL`. [Chi tiết hiệu chỉnh](results/hours_residual_metrics.json)

Cả bảy CSV có đúng 299.984 hàng dữ liệu, đúng cột `id,is_fraud` và đúng thứ tự ID mẫu. Các mức điểm dùng kết quả kiểm chứng cục bộ. [Biên bản kiểm tra CSV](results/verification.json)

## Dự án đã làm được gì

- Quy trình chính đã kiểm tra dữ liệu, dựng 5 fold phân tầng và huấn luyện mô hình trên 699.960 giao dịch, trong đó 11.689 giao dịch gian lận. Cây nông giúp giảm khớp quá mức. Target encoding (mã hóa giá trị bằng tỷ lệ gian lận đo trên dữ liệu huấn luyện) và nhóm đặc trưng chữ số tạo bước tăng rõ hơn điều chỉnh trọng số lớp. `XGB-D03` đạt 0,950182 AUC. Lượt đổi seed cây và encoder `XGB-D03-S71-E2028` đạt 0,950201. `FINAL` đạt 0,950345 khi huấn luyện lại hai tầng trong từng fold ngoài. [Tiến độ bước 0–8](../../reports/steps_progress.md) · [37 lượt GPU](../s6e11_gpu/plan_runs/results/full/comparison.md) · [68 cấu hình đơn](../s6e11_gpu/xep_hang_mo_hinh_don.md) · [Thử thêm seed](../s6e11_gpu/seed3_improvement/ket_qua_seed3_mean.md)
- Thêm target encoding theo cặp vào `XGB-D03` giảm AUC 0,000564. Mã hóa khóa hiếm giảm 0,000120. Thử khung giờ theo kênh, điểm velocity tương đối, focal loss, lấy mẫu khó và tiền huấn luyện đều đã có đối chứng. Nhánh MLP có AUC riêng thấp hơn cây nhưng khi thêm một tỷ lệ nhỏ vào `FINAL` đã cho dấu hiệu bổ trợ ở hai seed khác. [Ablation đặc trưng](../s6e11_gpu/final_feature_optimization/results/ket_qua_toi_uu_dac_trung_final.md) · [EDA và đối chứng](../eda_review/feature_ablation/results/ket_qua_thu_dac_trung.md) · [Lệch nhãn](../s6e11_gpu/imbalance/ket_qua_thu_nghiem_lech_nhan.md) · [Tiền huấn luyện](../s6e11_gpu/pretraining/results/summary.md) · [Seed 1337](../s6e11_gpu/followup/results/full/seed_1337/metrics.json) · [Seed 2718](../s6e11_gpu/followup/results/full/seed_2718/metrics.json)
- Nhánh Grandmaster đã thử GLM với spline và phân cấp chữ số, cây học phần sai lệch của GLM, EBM, mạng nơ-ron và gộp mô hình. Bản refit lưu AUC 0,951280 trên 10 fold với tầng gộp dùng ma trận OOF chung. Bản `hierarchical_amount` có AUC 0,950949 trên cùng tập giữ lại 15% đã dùng ở nhiều thí nghiệm. Hai con số mô tả quy trình đo riêng. Lượt này dùng `FINAL` làm mốc chung để so sánh ghép cặp trên seed 42. [Sổ thí nghiệm Grandmaster](<../Model tham khảo/Grandmaster/FINAL_RESEARCH_REPORT.md>) · [Refit 10 fold](../grandmaster_refit/full/metrics.json) · [Tập giữ lại 15%](../grandmaster_10fold_ablation/validation/metrics.json)

## Giả thuyết và cách triển khai

| Hướng | Giả thuyết | Kiểm tra trực tiếp |
|---|---|---|
| Lân cận đa độ rộng | Trong dữ liệu làm tròn, một giá trị chính xác có thể có rủi ro khác các giá trị liền kề. Giá trị hiếm cần dựa vào lân cận rộng hơn. | Hai cửa sổ quanh mỗi giá trị của sáu biến số. Đặc trưng ghi tỷ lệ lân cận có làm trơn, mức lệch ở giá trị chính xác và số quan sát. Thêm vào cây nông với đối chứng cùng cấu hình. |
| Tương tác theo lớp | Một cặp biến có thể đồng xuất hiện khác nhau giữa giao dịch gian lận và hợp lệ, sau khi đã tính đến ảnh hưởng của từng biến. | Tạo 25 tỉ số hợp lý theo lớp (class-conditional likelihood ratio) sau khi trừ phần giải thích bởi phân bố biên. Làm trơn ô thưa về giả định hai biến độc lập. |
| Mật độ cặp giá trị | Trình sinh dữ liệu có thể lặp lại một số cặp chính xác nhiều hơn dự kiến trong cùng vùng giá trị. | Đếm bảy cặp giá trị số, thêm độ lệch so với tần suất kỳ vọng trong ô thô. Thống kê học không cần nhãn và loại dòng đang mã hóa khỏi số đếm của chính nó. |
| Spline tương tác | Có thể mô tả tương tác trơn và giảm nhiễu so với chia ô cứng. | Hồi quy logistic ridge (hồi quy có phạt hệ số lớn) trên spline, chữ số và biến phân loại. Chọn tương tác bằng phần sai lệch còn lại trên train, rồi đo với mô hình spline cộng tính. |
| Tổ hợp trọng số cố định | LightGBM và MLP mắc các lỗi xếp hạng khác nhau nên có thể bổ trợ cho `FINAL`. | Gộp xác suất 70% `FINAL`, 20% `D03`, 10% `MLP-BCE`. So với từng nhánh bổ sung riêng và `FINAL` trên seed 42, 1337 và 2718. |
| Hiệu chỉnh theo thời gian chính xác | Một phần tín hiệu tại các mốc 0,01 giờ có thể còn sót sau khi dùng `FINAL`. | So số ca gian lận thực tế với kỳ vọng từ dự đoán lồng nhau trên phần train của mỗi fold. Học một hiệu chỉnh log-odds có điều chuẩn, lấy lân cận ±0,20 giờ làm mốc. |
| Tổ hợp spline | Spline có thể xếp hạng đúng một số cặp giao dịch mà cây xếp sai dù AUC riêng thấp hơn. | Gộp theo log-odds với trọng số 90% `FINAL` và 10% spline tương tác. Khảo sát này được chọn sau khi quan sát fold đầu. |

Chẩn đoán thăm dò theo hai nửa ID chẵn/lẻ cho thấy mức lệch rủi ro quanh các mốc `hours_since_last_txn` lặp lại rõ hơn số tiền hoặc khoảng cách. Với các mốc có ít nhất 40 giao dịch trong mỗi nửa, tương quan Spearman của phần sai lệch chuẩn hóa quanh xu hướng làm trơn Gaussian là 0,272 cho thời gian (băng thông 0,20 giờ), 0,044 cho số tiền và 0,057 cho khoảng cách. Các phép kiểm tra này dùng toàn bộ dòng của CSV chính thức và chỉ các cột cần thiết để tính kết quả. ID dùng để chia kiểm tra, đối chiếu hàng và tạo tệp nộp.

## Cách đo và tái lập

- ROC AUC là điểm Kaggle dùng để đánh giá khả năng xếp giao dịch gian lận cao hơn giao dịch hợp lệ. OOF (out of fold) là dự đoán cho dòng do mô hình huấn luyện ở các fold còn lại tạo ra. Các phép đo chính dùng cùng `StratifiedKFold` 5 fold, seed 42. Thống kê có nhãn của nhánh cây mới được tính trong 5 fold con của phần train, seed 2026. [Thể lệ chấm điểm Kaggle](https://www.kaggle.com/competitions/playground-series-s6e11/overview/abstract)
- XGBoost dùng 1.650 cây cố định trên CUDA với cùng tham số cho hai nhánh và đối chứng. Phép so sánh ghép cặp dùng cùng hàng và fold. Khoảng 95% DeLong trong `verification.json` mô tả bất định có điều kiện trên các dự đoán cố định. Ba seed của tổ hợp cố định chia lại cùng tập giao dịch. Các seed bổ sung hỗ trợ kiểm tra độ ổn định theo cách chia dữ liệu.
- Các pipeline mới dùng dữ liệu CSV chính thức. Tổ hợp và hiệu chỉnh dùng lại dự đoán local của các lượt đã xác minh trên chính dữ liệu đó. Kiểm tra `verification.json` đối chiếu SHA-256 dữ liệu, toàn bộ fold, thứ tự ID, khoảng xác suất và sự trùng khớp giữa giá trị CSV và mảng dự đoán.

Chạy từ thư mục `novel_model_experiments` với Python trong `grandmaster_refit/.refit-env`. Các tệp `.py` chứa toàn bộ tham số thực nghiệm.

```powershell
$python = '..\grandmaster_refit\.refit-env\Scripts\python.exe'
& $python -B multiscale_tree.py
& $python -B occupancy_tree.py --control
& $python -B alternative_model.py
& $python -B fixed_blend.py --output results/rerun_fixed_blend
& $python -B hours_residual.py
& $python -B blend_spline.py
& $python -B -m unittest discover -v
& $python -B verify_results.py
```

## Phạm vi nguồn đã đọc

- Đọc toàn bộ `steps_progress.md`, `s6e11_gpu/xep_hang_mo_hinh_don.md`, `s6e11_gpu/plan_runs/results/full/comparison.md`, `s6e11_gpu/seed3_improvement/ket_qua_seed3_mean.md`, `s6e11_gpu/final_feature_optimization/results/ket_qua_toi_uu_dac_trung_final.md`, `s6e11_gpu/pretraining/results/summary.md`, `s6e11_gpu/imbalance/ket_qua_thu_nghiem_lech_nhan.md`, `eda_review/ket_qua_eda.md` và `eda_review/feature_ablation/results/ket_qua_thu_dac_trung.md`.
- Đọc mục 4–7 của `Model tham khảo/Grandmaster/FINAL_RESEARCH_REPORT.md`, mục 2 trong ba `README.md` của V1, V2 và Grandmaster, mục 1–2 của `Model tham khảo/VERSION_COMPARISON.md`, cùng các đoạn chia fold và huấn luyện tầng gộp trong runner và notebook liên quan. Đối chiếu toàn bộ `grandmaster_refit/full/metrics.json`, `grandmaster_10fold_ablation/validation/metrics.json` và manifest của `hierarchical_amount`.
- Đọc tiêu đề và phép chia các bước trong `plan.md` để định vị tài liệu. Đối chiếu ID, nhãn, fold và xác suất đã lưu của sáu lượt chính. Kiểm tra Kaggle Overview ở các mục Evaluation và Submission File.
