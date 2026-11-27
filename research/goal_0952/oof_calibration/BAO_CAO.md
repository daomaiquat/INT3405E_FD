# Hiệu chỉnh Grandmaster EDA bằng điểm ngoài fold

**Ứng viên AUC tốt nhất lượt này là `smooth_amount_distance`**, đạt **0,95138773** và log loss **0,04380525** trên holdout chung. AUC tăng **0,00016328** so với Grandmaster EDA gốc. Khoảng tin cậy DeLong ghép cặp 95% cho mức tăng là **[−0,00001844, +0,00034500]**. Average Precision giảm nhẹ, từ 0,53823528 xuống 0,53821569. Kết quả đến từ dữ liệu train chính thức, với holdout đã được dùng ở các lượt trước. [Kết quả](development_metrics.json)

## Các giả thuyết

Giữ đúng điểm Grandmaster EDA đã chốt, gồm **85% XGBoost, 10% LightGBM và 5% spline** trên thang logit. Logit là điểm mà hàm sigmoid đổi thành xác suất. Hiệu chỉnh cộng một lượng nhỏ vào logit này. Mỗi phép đo có thể đối chiếu trực tiếp với bản gốc. [Báo cáo mô hình gốc](../BAO_CAO_GRANDMASTER_EDA_GOC.md) · [Mã hiệu chỉnh](calibrate.py)

1. **Học lại hệ số cho đúng 404 mức tiền cũ.** Dùng danh sách cent đã chốt và học hệ số mới từ phần dư của tổ hợp ngoài fold. Lượt trước học hệ số từ Grandmaster phân cấp. Điều chuẩn L2 = 25 và giới hạn ±0,30 logit.
2. **Hiệu chỉnh mềm theo hõm mật độ.** Đếm tần suất ở từng mức tiền và khoảng cách, rồi so với các điểm lân cận trên lưới 0,01. Dùng cả cửa sổ ±5 và ±25 nấc, bỏ điểm trung tâm. Mỗi cột có một biến tuyến tính theo độ thưa đã chuẩn hóa và một biến chỉ tác động thêm ở mức thưa mạnh. Các vùng ít dữ liệu, ngoài lưới và gần biên được giảm hoặc bỏ hiệu chỉnh. So riêng số tiền với số tiền cộng khoảng cách. L2 theo cột là 40 cho số tiền, 60 cho khoảng cách. Biên điều chỉnh tối đa là ±0,30 logit.
3. **Điều chỉnh theo độ lệch giữa các nhánh.** Học từ ba chênh lệch so với nhánh XGBoost. Đối chứng tuyến tính có ba hệ số. Bản phụ thuộc rủi ro nhân thêm mỗi độ lệch với một hàm sigmoid của điểm Grandmaster, tạo sáu hệ số. Dùng L2 = 120, áp dụng nửa bước và giới hạn ±0,15 logit. [Nguồn giả thuyết về mật độ](../deep_eda_measurement/BAO_CAO.md) · [Danh sách 404 giá trị](../deep_eda_measurement/amount_hole_submission.json)

Hồi quy logistic lấy điểm gốc làm offset, tức điểm bắt đầu cố định khi khớp phần sai lệch. Hàm mất mát là log loss với điều chuẩn L2. Mô hình học thêm hệ số chặn và hệ số theo mức rủi ro gốc để tránh gán sai lệch chung cho nhóm, rồi chỉ chuyển các hệ số theo đặc trưng sang điểm dự đoán. Cấu hình các biến thể được cố định trước khi chấm holdout.

## Cách kiểm chứng

- **Phát triển.** Dùng 594.966 dòng, chia ba fold phân tầng bằng seed 6271. Trong từng fold, huấn luyện lại theo cấu hình gốc của ba nhánh trên 396.644 dòng để lấy điểm cho 198.322 dòng còn lại. Tính tần suất không nhãn trên cùng tập tham chiếu của từng fold. Học hệ số hiệu chỉnh từ hai fold và đo log loss trên fold còn lại. Tiêu chí sàng lọc là giảm log loss trung bình và giảm ở ít nhất hai trong ba fold.
- **Holdout.** Học hệ số một lần trên toàn bộ 594.966 dòng có điểm ngoài fold, áp dụng vào đúng điểm Grandmaster EDA gốc trên 104.994 giao dịch, gồm 1.753 giao dịch gian lận. Đại lượng ROC AUC đo khả năng xếp hạng, Average Precision tổng hợp precision–recall, và log loss đo chất lượng xác suất.
- **Phạm vi của điểm ngoài fold.** Ba nhánh cuối được huấn luyện lại riêng trong từng fold. Biên Grandmaster tầng dưới lấy từ cache ngoài fold huấn luyện ở các lượt trước và danh sách 12 cặp EDA giữ cố định. Do đó phép sàng lọc ba fold là kiểm tra chuyển giao của tầng cuối trên nền có sẵn. Mô hình spline ở một số lần fit chạm trần 220 vòng tối ưu của cấu hình gốc. [Mã dựng điểm ngoài fold](build_oof.py) · [Lịch sử các fold](development_oof_manifest.json) · [Sàng lọc theo log loss](development_screen.json)

## Kết quả

| Biến thể | ROC AUC holdout | Δ AUC so với gốc | Average Precision | Log loss | Fold phát triển giảm log loss |
|---|---:|---:|---:|---:|---:|
| Grandmaster EDA gốc | 0,95122445 | — | 0,53823528 | 0,04383599 | — |
| Hiệu chỉnh 404 giá trị từ lượt trước | 0,95131585 | +0,00009140 | 0,53852282 | 0,04380602 | — |
| Học lại hệ số cho 404 giá trị | 0,95131806 | +0,00009361 | 0,53853305 | 0,04380524 | 2/3 |
| Độ thưa số tiền | 0,95133868 | +0,00011423 | 0,53841693 | 0,04380594 | 2/3 |
| **Độ thưa số tiền và khoảng cách** | **0,95138773** | **+0,00016328** | 0,53821569 | 0,04380525 | 2/3 |
| Điều chỉnh tuyến tính giữa các nhánh | 0,95121566 | −0,00000879 | 0,53838688 | 0,04383390 | 3/3 |
| Điều chỉnh giữa các nhánh theo mức rủi ro | 0,95122263 | −0,00000182 | 0,53839200 | 0,04383222 | 3/3 |

- Thêm khoảng cách vào số tiền cho mức tăng AUC **0,00004905** trên holdout, với khoảng tin cậy 95% **[−0,00007963, +0,00017772]**. Biến thể ghép tác động lên 17.669 dòng holdout. Hai trong ba fold phát triển tăng AUC, với các mức lần lượt **+0,00013819**, **+0,00002241** và **−0,00004449**. [Số liệu từng fold, khoảng tin cậy và vùng điểm](development_metrics.json) · [So sánh ghép cặp](release_verification.json)
- Bản trộn tập con kèm 404 mức tiền ở lượt trước đạt **0,95133949** trên cùng holdout. Bản độ thưa số tiền và khoảng cách cao hơn **0,00004824** về AUC. Mốc 0,952 cách **0,00061227**. Điểm Kaggle cho bản mới chưa được đo. [Kết quả lượt trước](../honest_leaves/amount_hole_subforest_metrics.json)
- Bản học lại đúng **404 mức tiền** có Average Precision tốt nhất lượt này. Nó chỉ cộng biên cho 1.644 dòng holdout và 4.448 dòng test. Trong ba fold phát triển, nhóm này có lần lượt 67, 68 và 56 ca gian lận, so với 50,17, 48,99 và 54,93 ca dự kiến. Ở fold thứ ba, dự báo gốc đã sát và hiệu chỉnh làm log loss tăng. Hai bản điều chỉnh giữa các nhánh giảm log loss trong phát triển và trên holdout, còn AUC holdout giảm nhẹ. [Kiểm tra nhóm 404 mức tiền](release_verification.json)

## Submission và tái lập

Hệ số xuất test được học từ điểm ngoài fold trên **699.960 dòng train**. Bản độ thưa số tiền và khoảng cách đổi điểm của 50.280 giao dịch test, tương đương **16,76%**. Mỗi CSV có **299.984 dòng**, cột `id,is_fraud`, đúng thứ tự ID và xác suất khớp điểm đã lưu trong sai số tuyệt đối `1e-14`. [Manifest điểm ngoài fold trên toàn train](full_oof_manifest.json) · [Hệ số và SHA-256](export-full_metrics.json) · [Mã kiểm tra độc lập](verify.py)

| Ứng viên | CSV |
|---|---|
| Độ thưa số tiền và khoảng cách | `submission_grandmaster_eda_smooth_amount_distance.csv` |
| Độ thưa số tiền | `submission_grandmaster_eda_smooth_amount.csv` |
| Học lại 404 mức tiền | `submission_grandmaster_eda_refit_amount_hole.csv` |
| Điều chỉnh tuyến tính giữa các nhánh | `submission_grandmaster_eda_linear_branch_adjustment.csv` |
| Điều chỉnh giữa các nhánh theo rủi ro | `submission_grandmaster_eda_risk_branch_adjustment.csv` |

Các lệnh chạy từ thư mục `oof_calibration`. Mã đọc CSV chính thức từ `data/` và mặc định tìm cache trong `research/`. Ví dụ đặt `GRANDMASTER_CACHE_PROJECT` tới thư mục cha của repo, nơi chứa cache ở workspace đã chạy. Môi trường đã chạy dùng Python 3.12 và XGBoost CUDA 3.4.1.

```powershell
$env:GRANDMASTER_CACHE_PROJECT = (Resolve-Path '..\..\..\..').Path
$python = 'python'
& $python -B build_oof.py development --device cuda
& $python -B calibrate.py development
& $python -B build_oof.py full --device cuda
& $python -B calibrate.py export-full
& $python -B verify.py export-full
```

Phạm vi nguồn đã dùng gồm mục 3–4 trong `BAO_CAO_GRANDMASTER_EDA_GOC.md`, mục **Tín hiệu có thể sử dụng** và **Tệp dự đoán ứng viên** của báo cáo đo lưới số, trường `holdout_variants.amount_hole_subforest` của lượt trộn tập con, và mã huấn luyện từng nhánh trong `lightgbm_branch.py`, `spline_branch.py`, `residual_models.py` cùng các hàm thiết kế `SplineDesign`, `fit_logistic`. Đọc `train.csv`, `test.csv` và `sample_submission.csv` để xác minh SHA-256 và ID. Chỉ dùng dữ liệu thi chính thức trong pipeline.
