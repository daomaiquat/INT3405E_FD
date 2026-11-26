# Cải tiến SEED3-MEAN

Giữ `SEED3-MEAN` theo tiêu chí đã đặt. Ứng viên có OOF AUC cao nhất là `SEED5-MEAN` với **0.950331140**, chênh **+0.000061862**.

Đã huấn luyện hai thành viên XGBoost mới trên **NVIDIA RTX 4060**, mỗi thành viên chạy đủ 5 fold. Tổng thời gian theo fold là **305.6 giây**. So sánh dùng cùng 699.960 dòng train, 299.984 dòng test và cách chia fold của đối chứng. Pipeline dùng dữ liệu cuộc thi và các dự đoán tự huấn luyện trong dự án.

## Các tổ hợp

Đối chứng lấy trung bình xác suất từ ba nhánh có cặp seed cây và seed mã hóa theo nhãn (target encoding) là `42/2026`, `71/2026` và `42/2028`. Hai nhánh mới giữ cấu hình cây, đặc trưng và quy trình dừng sớm của `XGB-D03`.

| Nhánh mới | Seed cây | Seed mã hóa | OOF ROC AUC riêng | Giây theo fold |
|---|---:|---:|---:|---:|
| `XGB-D03-S71-E2028` | 71 | 2028 | 0.950200547 | 170.5 |
| `XGB-D03-S103-E2032` | 103 | 2032 | 0.950121476 | 135.0 |

`SEED4-CROSS-MEAN` thêm cặp `71/2028`. `SEED4-NEW-MEAN` thêm cặp `103/2032`. `SEED5-MEAN` thêm cả hai. Thành viên trong mỗi tổ hợp có trọng số bằng nhau.

## Kết quả trên năm fold

| Tổ hợp | OOF ROC AUC | Δ AUC | Fold tăng | Bootstrap 95% cho Δ | Average precision | Log-loss |
|---|---:|---:|---:|---:|---:|---:|
| `SEED3-MEAN` | 0.950269278 | — | — | — | 0.528975397 | 0.044409259 |
| `SEED4-CROSS-MEAN` | 0.950303108 | +0.000033831 | 5/5 | [-0.000001014, +0.000068749] | 0.529076182 | 0.044399258 |
| `SEED4-NEW-MEAN` | 0.950309071 | +0.000039793 | 4/5 | [-0.000003160, +0.000082377] | 0.529264892 | 0.044391143 |
| `SEED5-MEAN` | 0.950331140 | +0.000061862 | 5/5 | [+0.000015747, +0.000108718] | 0.529295983 | 0.044385929 |

Δ AUC theo từng fold so với đối chứng.

| Tổ hợp | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| `SEED4-CROSS-MEAN` | +0.000003095 | +0.000018739 | +0.000051537 | +0.000019386 | +0.000082764 |
| `SEED4-NEW-MEAN` | +0.000024022 | +0.000061831 | -0.000017206 | +0.000107105 | +0.000055819 |
| `SEED5-MEAN` | +0.000027237 | +0.000063478 | +0.000031297 | +0.000104802 | +0.000113762 |

Tôi dùng mức cải thiện tối thiểu **0,000100 OOF AUC** và **4/5 fold tăng** từ kế hoạch dự án. Tôi thêm điều kiện cận dưới bootstrap 95% lớn hơn 0 để quyết định thay đối chứng. Khoảng bootstrap được tính trên 500 mẫu phân tầng ghép cặp từ các dự đoán đã lưu và chỉ phản ánh dao động do lấy mẫu các dòng.

[Notebook đã chạy](SEED3-MEAN_improvement.ipynb) · [Số liệu chi tiết](results/evaluation.json)
