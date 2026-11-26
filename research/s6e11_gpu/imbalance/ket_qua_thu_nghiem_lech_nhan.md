# S6E11 · Thử nghiệm xử lý lệch nhãn

Đã chạy **8 cấu hình mới trên NVIDIA RTX 4060** và đối chiếu với `XGB-D03` đã lưu. Cấu hình cây có AUC cao nhất trong nhóm là **`XGB-D03` (0.950182)**. Cấu hình MLP có AUC cao nhất trong nhóm là **`MLP-BCE` (0.944899)**.

Tất cả kết quả dùng cùng **5 fold** trên **699.960** giao dịch, gồm **11.689** giao dịch gian lận. Validation giữ nguyên tỷ lệ gian lận gần **1,67%**. Các chỉ số được tính lại từ dự đoán OOF và khớp với file kết quả từng lượt. Điểm mô hình cuối hiện có là **0.950345**, thu được từ lần xác nhận lồng nhau của tổ hợp. [Kết quả gốc](../plan_runs/results/full/comparison.md)

Precision@1% là tỷ lệ gian lận trong 1% giao dịch được xếp cao nhất. Recall@1% là tỷ lệ giao dịch gian lận được tìm thấy trong nhóm đó. Giây là tổng thời gian ghi theo fold. Thời gian `XGB-D03` gồm cả lượt chọn số vòng, còn các biến thể XGBoost mới dùng số vòng đã lưu, nên cột này không đo cùng một phạm vi công việc.

## XGBoost · Trọng số lớp và mẫu hợp lệ dễ bị báo nhầm

`XGB-W2` và `XGB-W4` lần lượt đặt trọng số lớp gian lận là 2 và 4. `XGB-HARD` tăng trọng số lên 3 cho 10% giao dịch hợp lệ có điểm rủi ro cao nhất từ dự đoán ngoài fold trong phần huấn luyện. `XGB-RANDOM` tăng đúng mức đó cho cùng số giao dịch hợp lệ chọn ngẫu nhiên. Các lượt giữ số vòng `[2194, 1536, 1771, 2306, 2131]` theo từng fold của `XGB-D03`, đã chọn trên tập dừng sớm trong phần huấn luyện. Kết quả đánh giá tác động của trọng số ở ngân sách số vòng cố định.

| Cấu hình | OOF ROC AUC | Δ AUC | Fold tăng | AP | Precision@1% | Recall@1% | Giây |
|---|---:|---:|---:|---:|---:|---:|---:|
| `XGB-D03` | 0.950182 | — | — | 0.528763 | 66.40% | 39.76% | 231.4 |
| `XGB-W2` | 0.950104 | -0.000079 | 1/5 | 0.528413 | 66.40% | 39.76% | 55.4 |
| `XGB-W4` | 0.950079 | -0.000103 | 2/5 | 0.527904 | 66.14% | 39.61% | 51.8 |
| `XGB-RANDOM` | 0.950177 | -0.000005 | 4/5 | 0.528397 | 66.33% | 39.72% | 47.2 |
| `XGB-HARD` | 0.950141 | -0.000041 | 2/5 | 0.529004 | 66.46% | 39.80% | 47.6 |

`XGB-RANDOM` đạt **0.950177**, Δ AUC **-0.000005** với **4/5** fold tăng. Δ Average Precision là **-0.000366**.

Nhóm này chưa có cấu hình đồng thời tăng ít nhất 0.000100 AUC và cải thiện ít nhất 4/5 fold so với đối chứng.

**So sánh mẫu khó với chọn ngẫu nhiên.** Lấy `XGB-RANDOM` làm đối chứng trực tiếp. `XGB-HARD` đạt **0.950141**, Δ AUC **-0.000036** với **2/5** fold tăng. Δ Average Precision là **+0.000607**.

## MLP · Focal Loss và học xếp hạng theo cặp

Bốn cấu hình dùng cùng 109 đặc trưng, hai tầng ẩn 128 và 64 nút, dropout 0,10, AdamW, batch 4.096 và 12 epoch. Trọng số khởi tạo và thứ tự batch được đối chiếu trong từng fold. `MLP-BCE` dùng binary cross-entropy. `MLP-FOCAL-G1` và `MLP-FOCAL-G2` giảm ảnh hưởng của mẫu dễ với `γ=1` và `γ=2`. `MLP-PAIRWISE` học để giao dịch gian lận được điểm cao hơn giao dịch hợp lệ được lấy cặp trong batch.

| Cấu hình | OOF ROC AUC | Δ AUC | Fold tăng | AP | Precision@1% | Recall@1% | Giây |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MLP-BCE` | 0.944899 | — | — | 0.503905 | 64.56% | 38.66% | 13.8 |
| `MLP-FOCAL-G1` | 0.944267 | -0.000632 | 0/5 | 0.502758 | 64.27% | 38.49% | 14.3 |
| `MLP-FOCAL-G2` | 0.944007 | -0.000891 | 0/5 | 0.501783 | 64.44% | 38.59% | 14.4 |
| `MLP-PAIRWISE` | 0.941213 | -0.003686 | 0/5 | 0.487224 | 63.20% | 37.85% | 18.9 |

`MLP-FOCAL-G1` đạt **0.944267**, Δ AUC **-0.000632** với **0/5** fold tăng. Δ Average Precision là **-0.001147**.

Nhóm này chưa có cấu hình đồng thời tăng ít nhất 0.000100 AUC và cải thiện ít nhất 4/5 fold so với đối chứng.

Đầu ra focal và pairwise được dùng như điểm xếp hạng chưa hiệu chỉnh. Việc biến đổi qua sigmoid đưa điểm về khoảng 0–1 để lưu, với mục đích đo AUC và AP. Số batch chỉ có một lớp được bỏ qua khi tính mất mát theo cặp là **0** trên toàn bộ 5 fold.

## AUC theo từng fold

| Cấu hình | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| `XGB-D03` | 0.949852 | 0.950884 | 0.954957 | 0.950990 | 0.944603 |
| `XGB-W2` | 0.949736 | 0.950966 | 0.954909 | 0.950983 | 0.944332 |
| `XGB-W4` | 0.949920 | 0.950891 | 0.954939 | 0.950910 | 0.944163 |
| `XGB-RANDOM` | 0.949647 | 0.950899 | 0.954973 | 0.951069 | 0.944667 |
| `XGB-HARD` | 0.949803 | 0.950938 | 0.954894 | 0.951058 | 0.944515 |
| `MLP-BCE` | 0.944761 | 0.945673 | 0.949624 | 0.946083 | 0.938606 |
| `MLP-FOCAL-G1` | 0.943693 | 0.945257 | 0.949119 | 0.945367 | 0.938308 |
| `MLP-FOCAL-G2` | 0.943153 | 0.945193 | 0.949131 | 0.945040 | 0.938066 |
| `MLP-PAIRWISE` | 0.941411 | 0.941908 | 0.945067 | 0.942883 | 0.935313 |

## Kết luận

Giữ `XGB-D03` làm mô hình cây trong tổ hợp hiện tại. Các thay đổi trọng số và cách chọn mẫu khó trong lượt thử này chưa đạt tiêu chí thay thế.

## Hồ sơ kết quả

- [`XGB-W2`](results/xgb/XGB-W2/metrics.json)
- [`XGB-W4`](results/xgb/XGB-W4/metrics.json)
- [`XGB-RANDOM`](results/xgb/XGB-RANDOM/metrics.json)
- [`XGB-HARD`](results/xgb/XGB-HARD/metrics.json)
- [`MLP-BCE`](results/mlp/MLP-BCE/metrics.json)
- [`MLP-FOCAL-G1`](results/mlp/MLP-FOCAL-G1/metrics.json)
- [`MLP-FOCAL-G2`](results/mlp/MLP-FOCAL-G2/metrics.json)
- [`MLP-PAIRWISE`](results/mlp/MLP-PAIRWISE/metrics.json)
