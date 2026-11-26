# Kiểm tra các biến thể của Grandmaster EDA

**Grandmaster EDA gốc vẫn là lựa chọn chính.** Trên tập giữ lại 104.994 giao dịch, mô hình đạt ROC AUC **0,95122445**, cách mục tiêu 0,952 khoảng 0,00077555. Cách lấy trung bình điểm từ năm bảng EDA cải thiện trong phép kiểm tra đảo vai trò, nhưng mức tăng chưa lặp lại trên tập giữ lại gốc. Điểm trong báo cáo đến từ nhãn train chính thức. Ba tệp mới đã được xác minh ở máy local. [Submission gốc](../submission_grandmaster_eda.csv)

## Tín hiệu đã kiểm tra

Giữ công thức 85% Grandmaster với XGBoost học phần sai lệch, 10% LightGBM D03 và 5% spline tương tác. Các tỷ trọng áp dụng trên log-odds, tức logarit của tỷ lệ giữa xác suất gian lận và xác suất hợp lệ. Giữ nguyên 109 cột D03, 12 cặp EDA, chữ số cuối của khoảng cách và XGBoost 224 cây sâu tối đa 3 tầng. Cây tiếp tục dùng hàm mất mát phân loại logistic với cấu hình gốc.

1. **Đồng nhất kích thước bảng EDA lúc học và lúc chấm.** Khi học cây, mỗi dòng nhận đặc trưng từ bảng ước lượng trên 80% dòng tham chiếu để loại trừ nhãn của chính nó. Bản gốc dùng bảng 100% lúc chấm. Bản mới cho chính cây gốc chấm riêng qua năm bảng 80%, sau đó lấy trung bình log-odds. Cách này giữ ý nghĩa các cặp đã phát hiện và kiểm tra mức nhạy của cây với thay đổi biên phân vị và độ hỗ trợ.
2. **Làm mượt các ô tương tác EDA.** Với trục số chia theo phân vị, mỗi ô chia 10% thống kê cho từng ô liền kề. Trục phân loại giữ các nhóm riêng. Thay ngưỡng hỗ trợ cứng bằng hệ số giảm dần cho ô ít thông tin. Vẫn loại ảnh hưởng của từng biến riêng, giữ điều chuẩn gốc và giữ đặc trưng chữ số cuối của khoảng cách.
3. **Trộn nửa nhánh làm mượt.** Lấy trung bình log-odds của hai nhánh XGBoost gốc và làm mượt bên trong phần 85%. Tỷ lệ 50/50 được đặt trước khi chấm.

## Kết quả kiểm chứng

| Bản | ROC AUC holdout gốc | Δ AUC holdout | Δ AUC đảo vai trò, có trọng số theo số cặp | Fold tăng AUC | Fold giảm log loss |
|---|---:|---:|---:|---:|---:|
| Grandmaster EDA gốc | **0,95122445** | — | — | — | — |
| Trung bình năm bảng EDA | 0,95121087 | −0,00001358 | **+0,00005404** | **7/10** | **8/10** |
| Làm mượt cặp EDA | 0,95121100 | −0,00001345 | −0,00006778 | 1/10 | 2/10 |
| Trộn 50% nhánh làm mượt | 0,95121931 | −0,00000514 | −0,00002392 | 1/10 | 2/10 |

- **Holdout gốc** dùng 594.966 dòng để học và 104.994 dòng để chấm. Grandmaster EDA gốc cũng dẫn đầu về Average Precision, log loss và Brier. Khoảng DeLong 95% của chênh lệch AUC ở bản trung bình năm bảng là [−0,00004510, +0,00001794], tính trên dự đoán cố định. Tập này đã được dự án dùng để chọn mô hình ở các lượt trước. [Điểm holdout](holdout_metrics.json)
- **Đảo vai trò** dùng 104.994 dòng tham chiếu để học nhánh XGBoost và chấm lần lượt trên 10 fold của 594.966 dòng còn lại. Mỗi fold dùng tầng nền Grandmaster đã học ngoài fold đó và tầng gộp riêng. Phép thử kiểm tra riêng nhánh XGBoost trong điều kiện ít dữ liệu tham chiếu hơn. Danh sách 12 cặp kế thừa đã được chọn trong các lượt trước từ cùng bộ nhãn. [Điểm từng fold](reverse_metrics.json)
- **Vùng điểm ảnh hưởng nhiều nhất.** Với bản trung bình năm bảng, 183 ca gian lận trong khoảng phân vị điểm gốc 75–90 đóng góp −0,00001270 vào mức thay đổi AUC holdout. Nhóm 50–75 đóng góp +0,00000685. Trên test, trung vị thay đổi xác suất tuyệt đối là 0,00000779 và phân vị 99 là 0,00178. Các dải điểm chỉ dùng để mô tả kết quả sau khi chấm. [Chi tiết theo phân vị](score_band_diagnostics.json)

## Submission và cách tái lập

Tất cả các biến thể dùng đủ 699.960 dòng train khi huấn luyện để xuất test. Mỗi CSV có 299.984 dòng, đúng cột `id,is_fraud`, đúng thứ tự ID và xác suất hữu hạn trong [0, 1]. SHA-256 của bản gốc được đối chiếu với manifest. Sai số log-odds lớn nhất khi chạy lại XGBoost gốc là 0,00000191. Công thức chấm dùng điểm gốc lưu sẵn và cộng chênh lệch giữa các nhánh chạy lại cùng lượt.

- Trung bình năm bảng EDA (`goal_0952/grandmaster_eda_refinement/submission_grandmaster_eda_matched_encoding_size.csv`)
- Làm mượt cặp EDA (`goal_0952/grandmaster_eda_refinement/submission_grandmaster_eda_smoothed_pairs.csv`)
- Trộn 50% nhánh làm mượt (`goal_0952/grandmaster_eda_refinement/submission_grandmaster_eda_smoothed_pairs_half.csv`)

Chạy từ thư mục `grandmaster_eda_refinement` với môi trường đã có của dự án.

```powershell
$python = '..\..\grandmaster_refit\.refit-env\Scripts\python.exe'
& $python -B refine_eda.py development
& $python -B refine_eda.py reverse
& $python -B refine_eda.py export
& $python -B verify_refinement.py
& $python -B -m unittest discover -v
```

## Phạm vi nguồn đã đọc

- [BAO_CAO_0952.md](../BAO_CAO_0952.md), toàn bộ báo cáo, gồm EDA gốc, công thức mô hình và kết quả giữ lại.
- [evaluation.json](../evaluation.json), các trường cấu hình được chọn, AUC giữ lại và SHA-256. [residual_development_metrics.json](../residual_development_metrics.json), dấu vân tay dữ liệu và danh sách 12 cặp đã chọn.
- [residual_models.py](../residual_models.py), các hàm mã hóa bảng, tách fold con và huấn luyện XGBoost. [eda_from_scratch.py](../eda_from_scratch.py), các hàm thống kê và tách hiệu ứng từng biến trong bảng. [run_followup.py](../followup/run_followup.py), các đoạn tái lập điểm gốc, cấu hình XGBoost và tạo ma trận đầu vào.
- `train.csv`, `test.csv` và `sample_submission.csv` chính thức, toàn bộ dòng và cột khi kiểm tra định danh, huấn luyện và xuất dự đoán. Dự đoán lưu theo fold được đối chiếu với cách chia tập gốc.
