# S6E11 · Thử cải thiện trọng số và mật độ theo ngữ cảnh

**ROC AUC cao nhất là 0,95131224**, tăng 0,00008779 so với bản 0,95122445. Khoảng cách tới 0,952 là 0,00068776. Phương án này có mức tăng dương trên cả hai nửa của tập giữ lại (+0,00003123 và +0,00014186). Khoảng tin cậy DeLong ghép cặp 95% là [−0,00003230, +0,00020788]. Tôi xếp nó là ứng viên thử nghiệm vì khoảng này chứa 0 và tập giữ lại đã được dùng để chọn mô hình. Submission ứng viên (`goal_0952/followup/submission_context.csv`) · [Kết quả mật độ mở rộng](context_metrics.json)

## Các phép thử trên cùng tập giữ lại

22 biến thể mới được so với cùng bản đối chứng trên 104.994 giao dịch, trong đó 1.753 ca gian lận. Mô hình và các bộ mã hóa học tham số từ 594.966 dòng phát triển. Hai nửa được chia phân tầng cùng seed 6401 để kiểm tra chiều thay đổi ngay trong tập giữ lại. [Vòng cây và trọng số](holdout_metrics.json) · [Hiệu chỉnh tương tác chung](structured_metrics.json) · [Vòng mở rộng ngữ cảnh](context_metrics.json)

| Cấu hình | ROC AUC | Δ với đối chứng |
|---|---:|---:|
| Đối chứng Grandmaster + XGBoost + LightGBM + spline | 0,95122445 | — |
| XGBoost với mật độ theo ngữ cảnh cũ | 0,95129466 | +0,00007021 |
| XGBoost với mật độ cũ và lân cận đa độ rộng | 0,95126146 | +0,00003701 |
| XGBoost với mật độ mở rộng theo EDA | 0,95131181 | +0,00008736 |
| **Trộn 50/50 hai nhánh mật độ** | **0,95131224** | **+0,00008779** |
| XGBoost ưu tiên vùng điểm thấp bằng trọng số | 0,95115070 | −0,00007375 |
| Hiệu chỉnh logistic có điều chuẩn, hệ số 0,5 | 0,95119591 | −0,00002854 |
| Hiệu chỉnh tương tác riêng cho vùng điểm thấp, hệ số 0,5 | 0,95101509 | −0,00020936 |

Mật độ theo ngữ cảnh đo một mức giá trị xuất hiện khác nhau thế nào giữa lớp gian lận và hợp lệ, sau khi biết ngữ cảnh giao dịch. Nhóm mở rộng có 10 đặc trưng, gồm số lần thử CVV theo loại thẻ và hình thức xác thực, thời gian kể từ giao dịch trước theo IP và kênh, velocity theo số giao dịch cùng hình thức xác thực. Nhóm hiếm quay về phân bố chung. Mã hóa cho dòng huấn luyện dùng nhãn từ các fold con khác.

Tổ hợp mới giữ nguyên 10% LightGBM và 5% spline. Phần 85% XGBoost chia đều cho hai cây học sai lệch từ Grandmaster có các nhóm đặc trưng mật độ khác nhau. Gộp trên log-odds rồi đưa về xác suất. Average Precision tăng từ 0,53823528 lên 0,53828373. Log-loss giảm từ 0,04383599 xuống 0,04382592. [Chi tiết phép gộp](context_expansion.py)

## Tái lập và nguồn dữ liệu

Chạy từ thư mục `goal_0952/followup`.

```powershell
$python = '..\..\grandmaster_refit\.refit-env\Scripts\python.exe'
& $python -B run_followup.py development
& $python -B structured_offset.py
& $python -B context_expansion.py
& $python -B run_followup.py export
& $python -B context_expansion.py --export expanded --mix-old-density
& $python -B -m unittest discover -v
```

Đã đọc toàn bộ báo cáo `goal_0952/BAO_CAO_0952.md`, toàn bộ dòng của `train.csv`, `test.csv` và `sample_submission.csv`, cùng hồ sơ kết quả và các mảng dự đoán trên đúng ID của các nhánh tham gia. Mô hình chỉ dùng dữ liệu CSV của cuộc thi và các dự đoán đã được huấn luyện local trên chính dữ liệu đó. [Phép đo gốc](../BAO_CAO_0952.md) · [Manifest của CSV mới](context_submission.json)
