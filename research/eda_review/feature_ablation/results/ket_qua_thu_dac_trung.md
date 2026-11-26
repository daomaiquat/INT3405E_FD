# S6E11 · Thử thêm hai đặc trưng vào XGB-D03

Cả hai cấu hình chạy trên NVIDIA RTX 4060 với 699.960 dòng train, 299.984 dòng test và cùng 5 fold với `XGB-D03`. Mỗi cấu hình thêm riêng đặc trưng của mình, giữ cấu hình XGBoost gốc và dùng cùng cách chọn số cây. Ở mỗi fold, mô hình chọn số cây trên phần giữ lại bên trong train, rồi huấn luyện lại trên toàn bộ phần train của fold.

- **Kênh giao dịch × khung giờ 6 tiếng.** Ghép kênh giao dịch với một trong bốn khung giờ. Mỗi cặp tạo một cột 0/1 (one-hot).
- **Velocity so với nhịp giao dịch tương tự.** Lấy `velocity_score` trừ trung vị của nhóm có số giao dịch và thời gian kể từ giao dịch trước tương tự. Các mốc chia số giao dịch là 3, 6, 10. Các mốc chia thời gian là 0,25, 1, 6 và 24 giờ. Với nhóm dưới 250 dòng, dùng trung vị chung. Mọi trung vị đều tính trên phần dữ liệu dùng để huấn luyện ở bước tương ứng.

## Kết quả gộp

| Cấu hình | Số cột | OOF ROC AUC | Δ AUC với XGB-D03 | AP | Δ AP | Log-loss | Số fold tăng |
|---|---:|---:|---:|---:|---:|---:|---:|
| `XGB-D03` gốc | 109 | 0.950182 | — | 0.528763 | — | 0.044422 | — |
| `XGB-D03-CHANNEL-6H` | 129 | 0.950108 | -0.000074 | 0.528639 | -0.000124 | 0.044431 | 1/5 |
| `XGB-D03-VELOCITY-RESIDUAL` | 110 | 0.950083 | -0.000099 | 0.528574 | -0.000189 | 0.044439 | 3/5 |

Mô hình tổ hợp `FINAL` hiện có OOF ROC AUC **0.950345** và AP **0.529376**. Các Δ trong bảng dùng `XGB-D03` gốc làm đối chứng trực tiếp.

OOF là điểm tính từ dự đoán trên phần dữ liệu giữ lại của từng fold. AP (Average Precision) là chỉ số tổng hợp khả năng tìm đúng các ca gian lận ở các ngưỡng xếp hạng. Log-loss thấp hơn nghĩa là dự đoán xác suất phù hợp với nhãn hơn.

## Chênh lệch AUC từng fold và số cây

| Fold | AUC gốc | AUC khung giờ | Δ khung giờ | AUC velocity | Δ velocity | Cây gốc | Cây khung giờ | Cây velocity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.949852 | 0.949775 | -0.000076 | 0.949432 | -0.000420 | 2194 | 2047 | 2150 |
| 2 | 0.950884 | 0.950873 | -0.000011 | 0.950898 | +0.000014 | 1536 | 1511 | 1648 |
| 3 | 0.954957 | 0.955007 | +0.000050 | 0.955084 | +0.000127 | 1771 | 1751 | 1684 |
| 4 | 0.950990 | 0.950798 | -0.000193 | 0.950994 | +0.000004 | 2306 | 3001 | 2257 |
| 5 | 0.944603 | 0.944510 | -0.000093 | 0.944446 | -0.000157 | 2131 | 2170 | 2132 |

## Quyết định

Mốc sàng lọc là tăng ít nhất **0.000100 OOF ROC AUC** và cải thiện ở ít nhất **4/5 fold**.

Cả hai ứng viên đều dưới mốc sàng lọc. Kết quả ủng hộ giữ cấu hình `XGB-D03` hiện tại.

Bảng được tính lại từ dự đoán đã lưu và đối chiếu với nhãn chính thức, thứ tự ID, phép chia fold, mã băm SHA-256 của ba CSV và nhật ký hai giai đoạn chạy CUDA của từng fold. [Số liệu đầy đủ](comparison.json) · [Lần sàng lọc trước khi huấn luyện](../../ket_qua_eda.md)
