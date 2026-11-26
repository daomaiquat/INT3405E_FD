# S6E11 · Đối chiếu vòng 2 trên RTX 4060

Có 699.960 giao dịch và 11.689 nhãn gian lận. Tất cả các hướng dùng 5 fold phân tầng, seed 42.

| Notebook | ROC AUC gộp | Δ so LightGBM vòng 1 | Average Precision | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fit hoặc gộp (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [Gộp hạng LightGBM và XGBoost](../rank-blend.ipynb) | 0.948474 | +0.000145 | 0.518606 | 0.947197 | 0.949820 | 0.953542 | 0.948956 | 0.943025 | 0.029 |
| [XGBoost + TE theo cặp](../xgboost-pair-te.ipynb) | 0.948193 | -0.000136 | 0.518245 | 0.946807 | 0.949378 | 0.953218 | 0.948636 | 0.943073 | 26.164 |
| [CatBoost GPU + TE số](../catboost-te.ipynb) | 0.947156 | -0.001173 | 0.517428 | 0.945438 | 0.948447 | 0.952827 | 0.947685 | 0.941646 | 138.541 |
| [TabM-mini](../tabm-mini.ipynb) | 0.942350 | -0.005980 | 0.489214 | 0.941594 | 0.943664 | 0.947724 | 0.942498 | 0.937576 | 67.276 |

- LightGBM vòng 1 đạt 0.948329. Khi tạo lại OOF, LightGBM đạt 0.948323 và XGBoost đạt 0.948311.
- XGBoost thêm hai cặp có chênh lệch -0.000118 AUC so với XGBoost đối chứng chạy lại.
- Tổ hợp 50/50 mất 0.029 giây để gộp trên CUDA. Hai lượt tạo OOF mất 142.58 giây tính trong vòng CV, gồm tiền xử lý và đánh giá.
- Tổ hợp có ROC AUC cao hơn cả hai mô hình thành phần ở 5/5 fold.
- Ba mô hình mới xác nhận RTX 4060 ở cả 5 fold. Mỗi tệp OOF lưu ID, fold và dự đoán. Điểm trong bảng được tính lại từ các tệp này.
