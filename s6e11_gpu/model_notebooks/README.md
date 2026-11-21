# Notebook GPU S6E11 · Vòng 1 và vòng 2

Các notebook chạy đánh giá trên tập huấn luyện chính thức. Chế độ đầy đủ dùng 5 fold phân tầng với seed 42. Môi trường đã dùng là Windows, Python 3.12 và NVIDIA GeForce RTX 4060 Laptop GPU. Các kiểm tra thiết bị trong notebook giữ yêu cầu RTX 4060.

| Vòng | Notebook | Thư viện huấn luyện |
|---|---|---|
| 1 | [LightGBM với mã hóa theo nhãn](round1-lightgbm-target-encoding.ipynb) | LightGBM 4.6.0 có hỗ trợ OpenCL |
| 1 | [XGBoost với mã hóa theo nhãn](round1-xgboost-target-encoding.ipynb) | XGBoost 3.4.1 có hỗ trợ CUDA |
| 1 | [Logistic Regression với mã hóa theo nhãn](round1-logistic-target-encoding.ipynb) | PyTorch CUDA |
| 1 | [MLP với véc-tơ biểu diễn biến phân loại](round1-mlp-categorical-embeddings.ipynb) | PyTorch CUDA |
| 1 | [FT-Transformer](round1-ft-transformer.ipynb) | PyTorch CUDA |
| 2 | [XGBoost với mã hóa theo nhãn cho hai cặp biến](round2-xgboost-pair-target-encoding.ipynb) | XGBoost 3.4.1 có hỗ trợ CUDA |
| 2 | [Gộp thứ hạng LightGBM và XGBoost](round2-lightgbm-xgboost-rank-blend.ipynb) | PyTorch CUDA và hai lượt cây vòng 1 |
| 2 | [CatBoost với mã hóa theo nhãn](round2-catboost-target-encoding.ipynb) | CatBoost 1.2.10 có đúng một GPU CUDA hiển thị |
| 2 | [TabM-mini](round2-tabm-mini.ipynb) | PyTorch CUDA và TabM 0.0.3 |

## Chuẩn bị

1. Tạo môi trường Python 3.12. Cài driver NVIDIA phù hợp và kiểm tra lệnh `nvidia-smi` chạy được. LightGBM sử dụng OpenCL của NVIDIA và kiểm tra nền tảng OpenCL ở vị trí 1 hoặc 0. XGBoost cần bản đầy đủ có CUDA.
2. Cài các thư viện cho kernel Python. Đây là các phiên bản thư viện chính ghi trong lượt chạy gốc.

   ```sh
   python -m pip install jupyterlab ipykernel "numpy==2.5.5" "pandas==3.1.1" "scikit-learn==1.9.1"
   python -m pip install "lightgbm==4.6.0" "xgboost==3.4.1" "catboost==1.2.10"
   python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
   python -m pip install "tabm==0.0.3"
   ```

3. Đặt `train.csv` hoặc ZIP chính thức `playground-series-s6e11.zip` trong `data/` của repo, hoặc đặt biến môi trường `S6E11_DATA_DIR` trỏ tới thư mục chứa tệp. Notebook cũng tìm trong thư mục cha của repo nếu biến môi trường chưa đặt. Dữ liệu phải khớp SHA-256 trong [manifest](../../data/manifest.json).
4. Khởi động `python -m jupyterlab` từ repo. Mở notebook, chọn kernel của môi trường trên rồi chạy lần lượt các ô. Nếu kernel bắt đầu bên ngoài repo, đặt `S6E11_REPO` trỏ tới thư mục gốc.

## Chạy thử và lưu kết quả

- Đặt `S6E11_SMOKE=1` để kiểm tra nhanh trên 20.000 dòng phân tầng và huấn luyện một fold. Notebook gộp thứ hạng yêu cầu hai lượt cây đầy đủ.
- Để chạy gộp thứ hạng, chạy LightGBM và XGBoost vòng 1 đủ 5 fold trước. Mỗi notebook tạo tệp dự đoán ngoài tập huấn luyện (OOF) theo ID. Notebook gộp đối chiếu ID, fold và mã SHA-256 của dữ liệu trước khi tính hạng.
- Kết quả tạo khi chạy nằm trong `local_results/`, tách theo vòng và chế độ `full` hoặc `smoke`. Thư mục này được Git bỏ qua. Có thể đổi thư mục kết quả bằng `S6E11_NOTEBOOK_OUTPUT`; dùng cùng giá trị cho hai notebook đối chứng và notebook gộp.
