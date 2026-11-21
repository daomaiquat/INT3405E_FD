# S6E11 · Danh mục mã mô hình

Tên notebook ghi rõ vòng thử, họ mô hình và kỹ thuật. Các lượt huấn luyện chính dùng dữ liệu chính thức của S6E11, 5 fold phân tầng và GPU RTX 4060. Có **37 kết quả** trong lượt đánh giá chính và **8 cấu hình** trong lượt thử xử lý lệch nhãn. [Bảng kết quả chính](plan_runs/published/full/comparison.md)

## Mô hình đối chứng B00–B07

| Mã | Cấu hình |
|---|---|
| `B00` | Điểm hằng bằng tỷ lệ gian lận trong phần train |
| `B01` | Điểm ngẫu nhiên |
| `B02` | Hồi quy logistic L2 |
| `B03` | Cây quyết định sâu 4 |
| `B04` | Random Forest với 300 cây |
| `B05` | LightGBM |
| `B06` | XGBoost |
| `B07` | CatBoost |

Mã cấu hình ở [src/models.py](../src/models.py). Chạy một hoặc nhiều mô hình bằng `python s6e11_project_run.py --stage baseline --models B02 B05 B06` từ thư mục gốc của repo. [Hướng dẫn runner chính](../README.md)

## Notebook các vòng GPU ban đầu

| Vòng | Mô hình hoặc phép gộp | Notebook |
|---|---|---|
| 1 | LightGBM + Target Encoding | [round1-lightgbm-target-encoding](model_notebooks/round1-lightgbm-target-encoding.ipynb) |
| 1 | XGBoost + Target Encoding | [round1-xgboost-target-encoding](model_notebooks/round1-xgboost-target-encoding.ipynb) |
| 1 | Logistic Regression + Target Encoding | [round1-logistic-target-encoding](model_notebooks/round1-logistic-target-encoding.ipynb) |
| 1 | MLP với embedding cho cột phân loại | [round1-mlp-categorical-embeddings](model_notebooks/round1-mlp-categorical-embeddings.ipynb) |
| 1 | FT-Transformer | [round1-ft-transformer](model_notebooks/round1-ft-transformer.ipynb) |
| 2 | XGBoost + Target Encoding theo cặp | [round2-xgboost-pair-target-encoding](model_notebooks/round2-xgboost-pair-target-encoding.ipynb) |
| 2 | LightGBM và XGBoost gộp hạng | [round2-lightgbm-xgboost-rank-blend](model_notebooks/round2-lightgbm-xgboost-rank-blend.ipynb) |
| 2 | CatBoost + Target Encoding | [round2-catboost-target-encoding](model_notebooks/round2-catboost-target-encoding.ipynb) |
| 2 | TabM-mini | [round2-tabm-mini](model_notebooks/round2-tabm-mini.ipynb) |

## Thí nghiệm theo kế hoạch và mô hình `FINAL`

| Nhóm | Mã cấu hình tiêu biểu | Notebook |
|---|---|---|
| Target Encoding | `T01`–`T04` | [01-te](plan_runs/notebooks/01-te.ipynb) |
| Khóa chữ số | `D01`–`D03`, `XGB-D01`–`XGB-D03` | [02-digit](plan_runs/notebooks/02-digit.ipynb) |
| Cặp đặc trưng và tần suất | `P1`, `P1-P2`, `P1-P2-P3`, `COUNT`, `DERIVED` | [03-pairs-count](plan_runs/notebooks/03-pairs-count.ipynb) |
| Tinh chỉnh tham số | `T01-M50`, `GLM-NEWTON`, `GLM-C0.1`, `GLM-C0.3`, `TUNE-*` | [04-tuning](plan_runs/notebooks/04-tuning.ipynb) |
| GLM làm điểm khởi đầu cho cây | `GLM-BOOST-LGB`, `GLM-BOOST-XGB` | [05-glm-boost](plan_runs/notebooks/05-glm-boost.ipynb) |
| Đổi seed và gộp dự đoán | `XGB-D03-S71`, `XGB-D03-S93`, `SEED3-*` | [06-seed](plan_runs/notebooks/06-seed.ipynb) |
| Tổ hợp và xác nhận | `ALL3-*`, `TREES2-*`, `XGB-D03-R125`, `FINAL` | [07-ensemble-final](plan_runs/notebooks/07-ensemble-final.ipynb) |

`FINAL` là tổ hợp Logistic Regression của `GLM-NEWTON` và `XGB-D03`. OOF ROC AUC qua xác nhận lồng nhau 5×5 là **0.950345**. [Mã huấn luyện và hướng dẫn chạy lại](plan_runs/README.md)

## Thử nghiệm xử lý lệch nhãn

Nhánh XGBoost gồm `XGB-W2`, `XGB-W4`, `XGB-HARD` và `XGB-RANDOM`. Nhánh MLP gồm `MLP-BCE`, `MLP-FOCAL-G1`, `MLP-FOCAL-G2` và `MLP-PAIRWISE`. [Mã và kết quả xử lý lệch nhãn](imbalance/README.md)

Dữ liệu cuộc thi, dự đoán theo giao dịch, tệp nộp và trọng số mô hình lưu cục bộ. Đường dẫn dữ liệu và các yêu cầu thư viện nằm trong hướng dẫn từng nhóm.
