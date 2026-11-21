# S6E11 · Thử nghiệm xử lý lệch nhãn

[Kết quả tổng hợp](RESULTS.md) so sánh trọng số lớp và mẫu âm khó (hard negatives) cho XGBoost, cùng ba hàm mất mát cho MLP. Cả hai nhóm dùng cùng phép chia 5 fold và chỉ dùng dữ liệu chính thức của cuộc thi. Nhãn của fold xác thực chỉ dùng để tính chỉ số.

| Mã nguồn | Chức năng |
|---|---|
| [xgboost_class_weights_and_hard_negatives.py](xgboost_class_weights_and_hard_negatives.py) | Chạy bốn biến thể XGBoost với số vòng cố định theo XGB-D03 |
| [mlp_bce_focal_and_pairwise.py](mlp_bce_focal_and_pairwise.py) | So sánh BCE, hai biến thể focal và mất mát xếp hạng theo cặp |
| [report_imbalance_experiments.py](report_imbalance_experiments.py) | Kiểm tra dự đoán theo ID rồi tính lại bảng kết quả |

## Chạy thử

Chuẩn bị ba tệp CSV chính thức và môi trường CUDA theo [hướng dẫn GPU](../plan_runs/README.md). Chạy từ thư mục gốc của repo. Dùng `--data-dir` để chỉ định thư mục khác nếu cần. Hai lệnh chạy thử dùng một tập con của cùng dữ liệu chính thức và tự tạo mọi đầu vào cần cho lượt thử.

```powershell
python s6e11_gpu/imbalance/xgboost_class_weights_and_hard_negatives.py --smoke --data-dir data
python s6e11_gpu/imbalance/mlp_bce_focal_and_pairwise.py --smoke --data-dir data
```

## Chạy đầy đủ

Chạy quy trình [đối chứng GPU](../plan_runs/README.md#chạy-lại) với `--stage all` trước khi chạy các biến thể cây. Lượt đối chứng tạo các dự đoán XGB-D03 và 25 tệp dự đoán chéo bên trong phần huấn luyện. Trình chạy biến thể cây kiểm tra dữ liệu, cấu hình và ID của các tệp đó trước khi chọn mẫu âm khó. Nếu đối chứng nằm ngoài thư mục mặc định, truyền `--plan-results <thư mục full>` cho trình chạy XGBoost và trình tạo báo cáo.

```powershell
python s6e11_gpu/imbalance/xgboost_class_weights_and_hard_negatives.py --validate-only --data-dir data
python s6e11_gpu/imbalance/xgboost_class_weights_and_hard_negatives.py --data-dir data
python s6e11_gpu/imbalance/mlp_bce_focal_and_pairwise.py --data-dir data
python s6e11_gpu/imbalance/report_imbalance_experiments.py --data-dir data
```

`--validate-only` kiểm tra đối chứng trên CPU. Các lượt huấn luyện lưu đầu ra cục bộ trong `results/xgb/` và `results/mlp/`. Trình tạo báo cáo ghi `RESULTS.local.md` để có thể đối chiếu với `RESULTS.md` đi kèm mã nguồn.
