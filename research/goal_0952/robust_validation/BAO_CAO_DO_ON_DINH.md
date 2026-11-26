# S6E11 · Kiểm tra cải tiến có điều chuẩn

## Thiết kế

Tôi thử một cấu hình mới cố định. Mỗi khoảng số liên tục chia 20% số đếm cho từng khoảng liền kề. Các ngữ cảnh hiếm được kéo về phân bố chung mạnh hơn. Cây XGBoost giảm từ độ sâu 3 xuống 2, tăng L2 từ 40 lên 60 và giữ 224 vòng. Bản phối hợp lấy trung bình log-odds của nhánh XGBoost hiện tại và nhánh điều chuẩn, cùng LightGBM và spline như trước. XGBoost vẫn học bằng mất mát xác suất. Một phép đối chứng giữ độ sâu và hệ số điều chuẩn cũ để đo riêng tác động của việc làm trơn mật độ.

- **Tập giữ lại 104.994 dòng, seed 4242.** So sánh toàn bộ tổ hợp trên đúng các dòng và nhãn đã dùng ở lượt trước. Tập này đã tham gia chọn các cấu hình trong lịch sử dự án. [Kết quả điều chuẩn](forward_metrics.json) · [Đối chứng chỉ làm trơn](smoothed_forward_metrics.json)
- **10 fold đảo vai trò, 594.966 dòng đánh giá.** Trong mỗi fold, bảy mô hình nền Grandmaster học trên chín fold còn lại. Tầng gộp, bộ mã hóa và cây học sai lệch học trên 104.994 dòng tham chiếu tách riêng. Tầng gộp tạo dự đoán ngoài fold cho phần tham chiếu bằng ba fold con. Mỗi fold đánh giá nằm ngoài tất cả các bước khớp tham số. Phép kiểm cô lập nhánh mật độ, với ít dữ liệu cho cây học sai lệch hơn và không có hai nhánh LightGBM và spline. Các đặc trưng đã có lịch sử khảo sát trên bộ train. Điểm tuyệt đối của hai quy trình cần đọc theo đúng phép chia và kiến trúc. [Kết quả 10 fold](reversed_metrics.json) · [Đối chứng chỉ làm trơn](smoothed_reversed_metrics.json)
- **Hiệu chỉnh xác suất.** Thử thêm một lớp logistic đơn điệu có hai hệ số, phạt độ lệch khỏi công thức gốc. Đánh giá qua 5 fold trên dự đoán lưu của tập giữ lại. AUC trong từng fold giữ nguyên. [Kết quả hiệu chỉnh](calibration_metrics.json)

## Kết quả bản điều chuẩn

| Phép đo | Đối chứng | Trộn 50% nhánh điều chuẩn | Δ |
|---|---:|---:|---:|
| AUC của toàn bộ tổ hợp trên tập giữ lại | 0,95131224 | 0,95126523 | −0,00004701 |
| AP của toàn bộ tổ hợp trên tập giữ lại | 0,53828373 | 0,53828048 | −0,00000325 |
| Log-loss của toàn bộ tổ hợp trên tập giữ lại | 0,04382592 | 0,04383366 | +0,00000774 |
| AUC 10 fold, trọng số theo số cặp dương–âm | 0,95018999 | 0,95025811 | +0,00006812 |
| AP gộp 10 fold | 0,53066892 | 0,53088297 | +0,00021405 |
| Log-loss gộp 10 fold | 0,04433998 | 0,04431729 | −0,00002269 |
| Brier gộp 10 fold | 0,01073419 | 0,01073196 | −0,00000222 |

Trên phép kiểm đảo vai trò, AUC tăng ở 8/10 fold. Xu hướng của AUC và log-loss khác nhau giữa hai phép đo. Tôi giữ bản 0,95131224 làm lựa chọn chính. Bản cây nông có giá trị đối chiếu vì nhánh này cải thiện đồng thời AUC, AP và log-loss trên phép kiểm 10 fold. Bản hiện tại (`goal_0952/followup/submission_context.csv`) · CSV điều chuẩn để đối chiếu (`goal_0952/robust_validation/submission_regularized.csv`)

Khi chỉ làm trơn và giữ cấu hình cây cũ, bản trộn đạt 0,95129606 trên tập giữ lại. Trên 10 fold đảo vai trò, AUC có trọng số chỉ tăng 0,00000232, AP giảm 0,00010759 và log-loss tăng 0,00000123. Kết quả ủng hộ tác động của cấu hình cây điều chuẩn trong phép kiểm 10 fold.

Hiệu chỉnh logistic giảm log-loss ở 3/5 fold. Log-loss gộp tăng từ 0,04382592 lên 0,04383683. Brier giảm nhẹ từ 0,01062983 xuống 0,01062718. Các fold hiệu chỉnh dùng hệ số khác nhau, nên AUC gộp của chúng có thể đổi dù thứ tự điểm trong từng fold được giữ nguyên.

## Tái lập và phạm vi đọc

Chạy từ `goal_0952/robust_validation`.

```powershell
$python = '..\..\grandmaster_refit\.refit-env\Scripts\python.exe'
& $python -B robust_density.py forward
& $python -B robust_density.py reverse
& $python -B robust_density.py forward --smooth-only
& $python -B robust_density.py reverse --smooth-only
& $python -B calibrate.py
& $python -B robust_density.py export
& $python -B -m unittest discover -v
```

Đã đọc toàn bộ `goal_0952/followup/BAO_CAO_FOLLOWUP.md`, toàn bộ ba CSV chính thức và các mảng dự đoán của mô hình so sánh. Phần tái kiểm tra chia tập đọc các mục xác minh nguồn và tạo điểm theo fold trong `grandmaster_inputs.py`, cùng đoạn `anchor_logits` và vòng lặp đánh giá trong `run_low_score_ranker.py`. [Hồ sơ lần trước](../followup/BAO_CAO_FOLLOWUP.md) · [Manifest tệp mới](submission_manifest.json)
