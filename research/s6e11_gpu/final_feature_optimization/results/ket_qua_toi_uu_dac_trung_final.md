# Tối ưu đặc trưng cho tổ hợp FINAL

Ba nhánh XGBoost chạy riêng trên RTX 4060 với 699.960 dòng train và cùng 5 fold của `XGB-D03`. Trong mỗi fold, mô hình chọn số cây trên phần giữ lại bên trong train rồi huấn luyện lại trên toàn bộ phần train của fold.

| Phiên bản | Cách tạo đặc trưng | Số cột mới |
|---|---|---:|
| `PAIR-TE` | Target Encoding cho 8 tương tác thuộc hai nhóm P1/P2 | 8 |
| `RARE-TE` | Target Encoding hai tầng cho số tiền và số dư tài khoản | 2 |
| `PAIR-RARE-TE` | Gộp cả hai nhóm trên | 10 |

Với khóa hiếm, mỗi biến có tối đa 32 khoảng phân vị được tính trên phần đang dùng để huấn luyện. Tỷ lệ gian lận của khoảng được làm trơn về tỷ lệ chung với hệ số 50. Tỷ lệ của giá trị chính xác đến cent được làm trơn về tỷ lệ của khoảng với hệ số 20. Giá trị chưa gặp dùng tỷ lệ của khoảng. Thống kê mã hóa một dòng train được tính từ các dòng ở fold mã hóa khác.

## Điểm từng nhánh XGBoost

| Cấu hình | Số cột | OOF ROC AUC | Δ AUC | AP | Δ AP | Log-loss | Fold tăng |
|---|---:|---:|---:|---:|---:|---:|---:|
| `XGB-D03` | 109 | 0.950182 | — | 0.528763 | — | 0.044422 | — |
| `XGB-D03-PAIR-TE` | 117 | 0.949618 | -0.000564 | 0.526957 | -0.001805 | 0.044557 | 0/5 |
| `XGB-D03-RARE-TE` | 111 | 0.950063 | -0.000120 | 0.528477 | -0.000285 | 0.044447 | 1/5 |
| `XGB-D03-PAIR-RARE-TE` | 119 | 0.949582 | -0.000601 | 0.526451 | -0.002312 | 0.044593 | 0/5 |

## Thay nhánh XGBoost trong FINAL

Mỗi biến thể thay `XGB-D03` bằng nhánh XGBoost mới. `GLM-NEWTON`, phép chuẩn hóa và trọng số logistic ở tầng gộp giữ theo tổ hợp gốc của từng fold. Tầng gộp được tái dựng từ 25 tệp dự đoán ngoài fold bên trong của `GLM-NEWTON` và 25 tệp của `XGB-D03`, dùng dữ liệu train của từng fold ngoài. Khi tái dựng `FINAL`, sai khác xác suất lớn nhất so với bản lưu là **0** và OOF ROC AUC khớp ở **0.950345**.

| Cấu hình | OOF ROC AUC | Δ AUC | AP | Δ AP | Log-loss | Fold tăng |
|---|---:|---:|---:|---:|---:|---:|
| `FINAL` | 0.950345 | — | 0.529376 | — | 0.044347 | — |
| `FINAL-PAIR-TE` | 0.949952 | -0.000393 | 0.528129 | -0.001247 | 0.044460 | 0/5 |
| `FINAL-RARE-TE` | 0.950260 | -0.000086 | 0.529168 | -0.000208 | 0.044364 | 1/5 |
| `FINAL-PAIR-RARE-TE` | 0.949882 | -0.000463 | 0.527546 | -0.001830 | 0.044490 | 0/5 |

| Fold | AUC FINAL | Δ FINAL-PAIR-TE | Δ FINAL-RARE-TE | Δ FINAL-PAIR-RARE-TE |
|---:|---:|---:|---:|---:|
| 1 | 0.949878 | -0.000387 | -0.000189 | -0.000476 |
| 2 | 0.951163 | -0.000601 | +0.000009 | -0.000871 |
| 3 | 0.955174 | -0.000178 | -0.000036 | -0.000186 |
| 4 | 0.951022 | -0.000410 | -0.000018 | -0.000472 |
| 5 | 0.944769 | -0.000289 | -0.000108 | -0.000317 |

## Trung bình 50/50 với FINAL gốc

Lấy trung bình xác suất OOF của `FINAL` gốc và từng biến thể thay nhánh theo tỷ lệ cố định 50/50, sau khi đối chiếu thứ tự ID và fold. Phép thử này dùng các dự đoán đã tính ở trên.

| Cấu hình | OOF ROC AUC | Δ AUC | AP | Δ AP | Log-loss | Fold tăng |
|---|---:|---:|---:|---:|---:|---:|
| `FINAL` | 0.950345 | — | 0.529376 | — | 0.044347 | — |
| `50/50 FINAL + PAIR-TE` | 0.950281 | -0.000064 | 0.529523 | +0.000147 | 0.044352 | 1/5 |
| `50/50 FINAL + RARE-TE` | 0.950311 | -0.000034 | 0.529320 | -0.000056 | 0.044352 | 1/5 |
| `50/50 FINAL + PAIR-RARE-TE` | 0.950249 | -0.000096 | 0.529260 | -0.000116 | 0.044367 | 1/5 |

| Fold | AUC FINAL | Δ 50/50 FINAL + PAIR-TE | Δ 50/50 FINAL + RARE-TE | Δ 50/50 FINAL + PAIR-RARE-TE |
|---:|---:|---:|---:|---:|
| 1 | 0.949878 | -0.000048 | -0.000085 | -0.000097 |
| 2 | 0.951163 | -0.000125 | +0.000018 | -0.000244 |
| 3 | 0.955174 | +0.000027 | -0.000013 | +0.000024 |
| 4 | 0.951022 | -0.000096 | -0.000000 | -0.000127 |
| 5 | 0.944769 | -0.000020 | -0.000043 | -0.000037 |

## Đánh giá

Trong sáu phép thử, `50/50 FINAL + RARE-TE` có AUC cao nhất với **0.950311**, chênh **-0.000034** so với `FINAL` và tăng ở **1/5 fold**.

Mốc sàng lọc là tăng ít nhất **0.000100 AUC** và tăng ở ít nhất **4/5 fold**. Đây là so sánh trên các fold phát triển của dự án, với các trọng số gộp cố định.

Cả sáu phép thử đều chưa qua mốc sàng lọc. Kết quả ủng hộ dùng `FINAL` gốc.

OOF là dự đoán ngoài fold. AP (Average Precision) tổng hợp độ chính xác và độ bao phủ ở các mức xếp hạng. Log-loss đánh giá chất lượng xác suất dự đoán, với giá trị thấp hơn là tốt hơn.

[Số liệu đối chiếu](comparison.json) · [Chi tiết tái dựng FINAL](frozen_reference.json) · [Sàng lọc đặc trưng từ EDA](../../../eda_review/ket_qua_eda.md) · [Kế hoạch và mốc sàng lọc](../../../../plan.md)
