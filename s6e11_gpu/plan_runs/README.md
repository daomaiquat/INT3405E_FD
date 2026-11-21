# S6E11 · Huấn luyện trên RTX 4060

Bộ notebook trình bày các lượt T01–T04, D01–D03, cặp tương tác và tần suất, tinh chỉnh tham số, GLM-boost, thử seed và xác nhận tổ hợp. Mỗi lượt đủ dữ liệu dùng **699.960 giao dịch, 5 fold phân tầng và seed 42**. Hai họ cây chọn số vòng trên tập dừng sớm trong phần huấn luyện rồi huấn luyện lại trên toàn phần đó. [Báo cáo 37 kết quả đã xác minh](published/full/comparison.md)

Mô hình cuối gồm GLM-NEWTON và XGBoost D03. Đánh giá lồng nhau đạt **OOF ROC AUC 0.950345**. Kết quả lưu trong `published/full` là chỉ số tổng hợp và siêu dữ liệu của lượt chạy gốc. Các notebook hiển thị được kết quả đã lưu ngay khi clone repo. [Notebook xác nhận tổ hợp](notebooks/07-ensemble-final.ipynb)

## Chuẩn bị dữ liệu và môi trường

Tải `train.csv`, `test.csv` và `sample_submission.csv` từ [trang Data của S6E11](https://www.kaggle.com/competitions/playground-series-s6e11/data). Đặt ba tệp cùng nhau trong `data/` của repo, hoặc chỉ định thư mục bằng `--data-dir`. Trình chạy kiểm tra SHA-256 trước lượt huấn luyện đầy đủ.

Lượt gốc dùng Python 3.12 trên Windows, NVIDIA RTX 4060, PyTorch CUDA 12.8, XGBoost CUDA và LightGBM OpenCL. Dùng môi trường Python có driver NVIDIA và các bản thư viện hỗ trợ GPU.

```powershell
python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r s6e11_gpu/plan_runs/requirements.txt
python s6e11_gpu/plan_runs/verify_gpu.py torch
python s6e11_gpu/plan_runs/verify_gpu.py boost
```

## Chạy lại

Chạy các lệnh sau từ thư mục gốc của repo. Bỏ `--data-dir data` nếu đang dùng đường dẫn mặc định. Thay `data` bằng thư mục riêng nếu lưu dữ liệu ở chỗ khác.

```powershell
python s6e11_gpu/plan_runs/run_plan.py --stage te --only-te T01 T02 T03 --smoke --data-dir data
python s6e11_gpu/plan_runs/run_plan.py --stage all --compare-xgb-digits --tune-family xgb --seed-anchor XGB-D03 --stack-lgb D03 --stack-xgb XGB-D03 --data-dir data
python s6e11_gpu/plan_runs/report_results.py --data-dir data
```

Lượt đầy đủ lưu cấu hình, chỉ số, dự đoán từng fold và `final/submission.csv` trong `s6e11_gpu/plan_runs/results/full/`. Trình chạy tiếp tục từ các fold đã hoàn thành khi cấu hình và dữ liệu khớp. Dữ liệu cuộc thi, dự đoán theo giao dịch, tệp nộp, trạng thái mô hình và thư mục chạy được lưu cục bộ.

## Notebook

1. [Mã hóa theo nhãn](notebooks/01-te.ipynb)
2. [Khóa chữ số](notebooks/02-digit.ipynb)
3. [Cặp tương tác và tần suất](notebooks/03-pairs-count.ipynb)
4. [Tinh chỉnh tham số](notebooks/04-tuning.ipynb)
5. [GLM-boost](notebooks/05-glm-boost.ipynb)
6. [Gộp qua ba seed](notebooks/06-seed.ipynb)
7. [Tổ hợp và xác nhận lồng nhau](notebooks/07-ensemble-final.ipynb)
