# Tối ưu Grandmaster EDA bằng tín hiệu tương tác còn sót

**Bản tuyến tính gộp 10 cặp EDA bổ sung đạt ROC AUC 0,95123214 trên holdout, tăng 0,00000769 so với Grandmaster EDA gốc.** Trên 10 fold đảo vai trò, chênh lệch có trọng số theo số cặp giao dịch là +0,00000014. Grandmaster EDA gốc vẫn là lựa chọn chính. Bản tuyến tính là ứng viên phù hợp nhất để đối chiếu vì thay đổi nhỏ và cho mức chênh cùng chiều trong hai phép thử. Mục tiêu 0,952 vẫn còn cách 0,00076786 theo AUC tốt nhất của lượt này. [Đối chứng](../submission_grandmaster_eda.csv) · CSV tuyến tính (`goal_0952/grandmaster_eda_risk/submission_grandmaster_eda_extra_pairs_linear.csv`)

## Hai giả thuyết

1. **Hiệu ứng cặp biến thay đổi theo rủi ro nền.** Giữ nguyên 12 cặp đã dùng. Trong mỗi ô, ước lượng độ dốc của phần sai lệch giữa nhãn và xác suất Grandmaster theo điểm rủi ro nền. Hàm tanh quanh mức xác suất 3% làm giới hạn ảnh hưởng ở hai đầu. Phép tính tách độ dốc khỏi sai lệch trung bình trong ô và hiệu ứng đơn biến, rồi co các ô ít thông tin. Cross-fitting dùng năm fold để đặc trưng của mỗi dòng train được học từ nhãn ngoài fold.
2. **Các cặp ngoài danh sách 12 vẫn có thể bổ sung tín hiệu.** Từ kết quả sàng lọc cũ, lấy tất cả cặp chưa dùng có mức giảm log loss chéo xấp xỉ ít nhất 0,000001. Quy tắc này chọn 10 cặp, gồm số lần tranh chấp × lệch quốc gia IP, kênh × VPN và số lần thử CVV × cờ lừa đảo AI. Dùng bảng sai lệch kiểu gốc để tính từng cặp và cộng thành một cột. Danh sách đầy đủ và độ lớn tín hiệu nằm trong hồ sơ thí nghiệm. [Sàng lọc gốc](../residual_eda.json) · [10 cặp đã dùng](extra_development_metrics.json)

Mỗi giả thuyết được thử theo ba cách. Thêm tín hiệu vào XGBoost 224 cây sâu tối đa 3 tầng. Lấy trung bình hai đầu ra XGBoost cũ và mới bên trong tỷ trọng 85% của nhánh cây. Hoặc giữ cây gốc và thêm một hiệu chỉnh tuyến tính Ridge học theo log loss trên các đặc trưng ngoài fold. Hệ số của tổng 10 cặp được giới hạn [0, 1] để tự về 0 khi tập học thiếu bằng chứng cùng chiều. Hiệu chỉnh tuyến tính chỉ dùng nửa biên ước lượng. Tỷ trọng LightGBM 10% và spline 5% được bảo toàn trên thang log-odds.

## Kết quả local

| Phương án | ROC AUC holdout | Δ AUC holdout | Δ AUC đảo vai trò | Fold tăng AUC | Δ log loss holdout |
|---|---:|---:|---:|---:|---:|
| Grandmaster EDA gốc | 0,95122445 | — | — | — | — |
| Rủi ro × EDA qua XGBoost | 0,95112217 | −0,00010228 | −0,00004679 | 3/10 | +0,00001781 |
| Rủi ro × EDA, trộn nửa nhánh | 0,95117764 | −0,00004681 | −0,00000589 | 4/10 | +0,00000705 |
| Rủi ro × EDA tuyến tính | 0,95122158 | −0,00000287 | −0,00000083 | 4/10 | −0,00000050 |
| 10 cặp bổ sung qua XGBoost | 0,95123132 | +0,00000687 | −0,00000988 | 4/10 | **−0,00000693** |
| 10 cặp bổ sung, trộn nửa nhánh | 0,95122902 | +0,00000458 | −0,00000194 | 4/10 | −0,00000385 |
| **10 cặp bổ sung tuyến tính** | **0,95123214** | **+0,00000769** | **+0,00000014** | 3/10 | −0,00000014 |

- Tập giữ lại (holdout) có 104.994 giao dịch, mô hình học từ 594.966 dòng. Khoảng DeLong 95% ghép cặp của Δ AUC tuyến tính là [−0,00000233, +0,00001770]. Phép đo dùng dự đoán cố định và holdout đã được dùng ở các vòng trước. [Holdout nhánh rủi ro](development_metrics.json) · [Holdout nhánh bổ sung](extra_development_metrics.json)
- Phép đảo vai trò dùng 104.994 dòng để học nhánh XGBoost và 594.966 dòng còn lại chia thành 10 fold để chấm. Tầng nền từng fold được lấy từ mô hình đã chừa fold cần chấm. Chênh lệch AUC là trung bình có trọng số theo số cặp gian lận–hợp lệ trong từng fold. Bản tuyến tính bổ sung tăng rất nhẹ ở ba fold và bằng đối chứng ở bảy fold. Việc chọn danh sách cặp kế thừa vẫn dựa vào cùng bộ train lịch sử. [10 fold nhánh rủi ro](reverse_metrics.json) · [10 fold nhánh bổ sung](extra_reverse_metrics.json)
- Khi dùng tín hiệu theo rủi ro trong cây, phần lớn mức mất AUC đến từ các ca gian lận trong dải phân vị điểm gốc 50–90. Với bản tuyến tính bổ sung, bốn dải đến phân vị 95 đóng góp dương, hai dải cao hơn đóng góp âm nhỏ. Các dải là chẩn đoán sau đánh giá. Hệ số Ridge của tín hiệu bổ sung là 0,1808 khi huấn luyện holdout và 0,2035 khi huấn luyện để chấm test. Khi xuất test, trung vị thay đổi xác suất tuyệt đối là 0,00000183. [Phân rã theo dải điểm](extra_development_metrics.json) · [Hệ số refit](extra_full_metrics.json)

### Đối chứng xáo trộn cột bổ sung

Xáo cột tổng 10 cặp trong từng nhóm điểm nền, giữ 20 nhóm và một seed cố định. Phép thử giữ số lượng cột và phân phối của cột mới theo nhóm rủi ro, đồng thời phá liên hệ giữa giao dịch và giá trị gốc. XGBoost đạt 0,95120693 với cột xáo và 0,95123132 với cột thật. Chênh lệch là +0,00002439, với khoảng DeLong 95% [−0,00001776, +0,00006654]. Đây là một phép xáo đơn trên holdout. [Kết quả đối chứng xáo trộn](extra_placebo_metrics.json)

## CSV đã xuất

Mỗi tệp dùng mô hình huấn luyện lại với 699.960 dòng train và có 299.984 dòng test, hai cột `id,is_fraud`, đúng thứ tự ID và xác suất hợp lệ. SHA-256 của bản gốc và từng CSV được đối chiếu với manifest. Mức lệch log-odds lớn nhất khi chạy lại nhánh cây gốc là 0,00000191. Công thức dùng dự đoán gốc đã lưu và cộng chênh lệch đo trong cùng lượt chạy.

| Dùng tín hiệu trong mô hình | Rủi ro × EDA | 10 cặp bổ sung |
|---|---|---|
| XGBoost | CSV (`goal_0952/grandmaster_eda_risk/submission_grandmaster_eda_risk_conditioned_tree.csv`) | CSV (`goal_0952/grandmaster_eda_risk/submission_grandmaster_eda_extra_pairs_tree.csv`) |
| Trộn nửa nhánh XGBoost | CSV (`goal_0952/grandmaster_eda_risk/submission_grandmaster_eda_risk_conditioned_half.csv`) | CSV (`goal_0952/grandmaster_eda_risk/submission_grandmaster_eda_extra_pairs_half.csv`) |
| Hiệu chỉnh tuyến tính | CSV (`goal_0952/grandmaster_eda_risk/submission_grandmaster_eda_risk_conditioned_linear.csv`) | CSV (`goal_0952/grandmaster_eda_risk/submission_grandmaster_eda_extra_pairs_linear.csv`) |

## Tái lập và phạm vi nguồn

Chạy từ thư mục `grandmaster_eda_risk`.

```powershell
$python = '..\..\grandmaster_refit\.refit-env\Scripts\python.exe'
foreach ($family in @('risk', 'extra')) {
    & $python -B risk_eda.py development --family $family
    & $python -B risk_eda.py reverse --family $family
    & $python -B risk_eda.py export --family $family
    & $python -B verify_risk_eda.py --family $family
}
& $python -B audit_extra.py
& $python -B -m unittest discover -v
```

- Đã đọc toàn bộ báo cáo Grandmaster EDA của lượt trước, báo cáo các đặc điểm dễ bỏ sót và báo cáo học từ dự đoán có sẵn. [Grandmaster EDA](../grandmaster_eda_refinement/BAO_CAO_GRANDMASTER_EDA.md) · [Đặc điểm dữ liệu](../data_signals/BAO_CAO_DAC_DIEM.md) · [Không gian điểm](../score_space/BAO_CAO_HUONG_KHAC.md)
- Đã đọc mục kết quả xếp hạng các cặp dương và cấu trúc của tệp EDA gốc. Runner đọc danh sách cặp đầy đủ khi áp dụng ngưỡng. Đã đọc đoạn mã hóa, đối chiếu nhánh gốc, tạo ma trận và đảo vai trò trong các runner trước. [Cặp EDA](../residual_eda.json) · [Mã hóa gốc](../residual_models.py) · [Ma trận gốc](../followup/run_followup.py) · [Runner lượt trước](../grandmaster_eda_refinement/refine_eda.py)
- Ba CSV chính thức được đọc toàn bộ dòng và cột trong quy trình kiểm tra định danh, huấn luyện và xuất dự đoán. Các chỉ số được tính lại từ dự đoán lưu cùng ID chính thức và cách chia tập gốc.
