# S6E11 · Kết quả huấn luyện GPU

Dữ liệu huấn luyện có 699.960 giao dịch và 11.689 nhãn gian lận. Các mô hình dùng cùng 5 phần chia phân tầng với seed 42.

| Notebook | ROC AUC OOF | Average Precision | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fit (giây) | Toàn lượt (giây) | Thiết bị |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [LightGBM 4 lá + TE](../lightgbm-te.ipynb) | 0.948329 | 0.516402 | 0.946953 | 0.949707 | 0.953471 | 0.948874 | 0.942825 | 23.73 | 33.52 | OpenCL 1:0: NVIDIA GeForce RTX 4060 Laptop GPU |
| [XGBoost độ sâu 2 + TE](../xgboost-te.ipynb) | 0.948311 | 0.518785 | 0.947114 | 0.949628 | 0.953324 | 0.948757 | 0.942915 | 15.84 | 23.36 | cuda:0: NVIDIA GeForce RTX 4060 Laptop GPU |
| [Hồi quy logistic + TE](../logistic-te.ipynb) | 0.947300 | 0.513789 | 0.946438 | 0.948720 | 0.952651 | 0.947457 | 0.941383 | 1.41 | 9.49 | NVIDIA GeForce RTX 4060 Laptop GPU (PyTorch CUDA) |
| [MLP](../mlp.ipynb) | 0.941456 | 0.486874 | 0.941203 | 0.943095 | 0.946657 | 0.942088 | 0.937009 | 17.00 | 20.63 | cuda:0: NVIDIA GeForce RTX 4060 Laptop GPU (PyTorch CUDA) |
| [FT-Transformer](../ft-transformer.ipynb) | 0.938332 | 0.474786 | 0.939143 | 0.943263 | 0.947186 | 0.941992 | 0.937370 | 31.77 | 35.71 | cuda:0: NVIDIA GeForce RTX 4060 Laptop GPU (PyTorch CUDA) |

- Điểm OOF được tính trên các phần giữ lại của `train.csv`.
- LightGBM dùng OpenCL để tính histogram trên GPU. Các mô hình còn lại dùng CUDA để huấn luyện.
- Số giây fit là tổng cả 5 fold. Thời gian toàn lượt còn gồm tiền xử lý, suy luận và tính điểm.
