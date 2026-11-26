# Xếp hạng mô hình đơn S6E11

Đã đối chiếu **68 cấu hình** có kết quả đủ 5 fold trên 699.960 giao dịch. **XGB-D03** đứng đầu với OOF ROC AUC **0.950182**. Thứ hạng dùng AUC gộp chưa làm tròn. AUC cao hơn nghĩa là mô hình xếp giao dịch gian lận lên trước tốt hơn.

## Mô hình tốt nhất theo họ

| Họ mô hình | Cấu hình tốt nhất | OOF ROC AUC | Hạng chung |
|---|---|---:|---:|
| XGBoost | `XGB-D03` | 0.950182 | 1 |
| LightGBM | `D03` | 0.950131 | 5 |
| Hồi quy logistic | `GLM-NEWTON` | 0.947656 | 30 |
| CatBoost | `CatBoost + TE · vòng 2` | 0.947156 | 42 |
| MLP | `MLP-BCE` | 0.944899 | 51 |
| TabM | `TabM-mini · vòng 2` | 0.942350 | 54 |
| FT-Transformer | `ft-masked` | 0.942241 | 55 |
| Random Forest | `B04` | 0.917245 | 65 |
| Decision Tree | `B03` | 0.798327 | 66 |
| TabNet | `tabnet-masked` | 0.753023 | 67 |

## Toàn bộ cấu hình đã đối chiếu

AP (Average Precision) tóm tắt precision–recall. Dấu — cho biết nguồn lưu điểm cũ thiếu AP.

| Hạng | Cấu hình | Họ mô hình | OOF ROC AUC | AP | Đợt thử |
|---:|---|---|---:|---:|---|
| 1 | [`XGB-D03`](plan_runs/results/full/XGB-D03/metrics.json) | XGBoost | 0.95018230 | 0.528763 | Kế hoạch chính |
| 2 | [`XGB-RANDOM`](imbalance/results/xgb/XGB-RANDOM/metrics.json) | XGBoost | 0.95017683 | 0.528397 | Lệch nhãn |
| 3 | [`XGB-D03-R125`](plan_runs/results/full/XGB-D03-R125/metrics.json) | XGBoost | 0.95014625 | 0.529329 | Kế hoạch chính |
| 4 | [`XGB-HARD`](imbalance/results/xgb/XGB-HARD/metrics.json) | XGBoost | 0.95014087 | 0.529004 | Lệch nhãn |
| 5 | [`D03`](plan_runs/results/full/D03/metrics.json) | LightGBM | 0.95013086 | 0.527604 | Kế hoạch chính |
| 6 | [`D02`](plan_runs/results/full/D02/metrics.json) | LightGBM | 0.95012422 | 0.528206 | Kế hoạch chính |
| 7 | [`XGB-D03-CHANNEL-6H`](../eda_review/feature_ablation/results/full/XGB-D03-CHANNEL-6H/metrics.json) | XGBoost | 0.95010814 | 0.528639 | EDA và thêm đặc trưng |
| 8 | [`TUNE-NOSUB`](plan_runs/results/full/TUNE-NOSUB/metrics.json) | XGBoost | 0.95010368 | 0.528598 | Kế hoạch chính |
| 9 | [`XGB-W2`](imbalance/results/xgb/XGB-W2/metrics.json) | XGBoost | 0.95010351 | 0.528413 | Lệch nhãn |
| 10 | [`XGB-D03-S93`](plan_runs/results/full/XGB-D03-S93/metrics.json) | XGBoost | 0.95009142 | 0.527930 | Kế hoạch chính |
| 11 | [`XGB-D03-VELOCITY-RESIDUAL`](../eda_review/feature_ablation/results/full/XGB-D03-VELOCITY-RESIDUAL/metrics.json) | XGBoost | 0.95008310 | 0.528574 | EDA và thêm đặc trưng |
| 12 | [`XGB-W4`](imbalance/results/xgb/XGB-W4/metrics.json) | XGBoost | 0.95007915 | 0.527904 | Lệch nhãn |
| 13 | [`XGB-D02`](plan_runs/results/full/XGB-D02/metrics.json) | XGBoost | 0.95007439 | 0.528382 | Kế hoạch chính |
| 14 | [`XGB-D03-S71`](plan_runs/results/full/XGB-D03-S71/metrics.json) | XGBoost | 0.95007415 | 0.527848 | Kế hoạch chính |
| 15 | [`XGB-D03-RARE-TE`](final_feature_optimization/results/full/XGB-D03-RARE-TE/metrics.json) | XGBoost | 0.95006260 | 0.528477 | Tối ưu đặc trưng |
| 16 | [`TUNE-ETA005`](plan_runs/results/full/TUNE-ETA005/metrics.json) | XGBoost | 0.95005989 | 0.528456 | Kế hoạch chính |
| 17 | [`TUNE-LAMBDA3`](plan_runs/results/full/TUNE-LAMBDA3/metrics.json) | XGBoost | 0.95004546 | 0.528513 | Kế hoạch chính |
| 18 | [`xgb-d03-mlp-masked-embeddings`](pretraining/results/full/xgb-d03-mlp-masked-embeddings/metrics.json) | XGBoost | 0.94966046 | 0.525347 | Tiền huấn luyện |
| 19 | [`XGB-D03-PAIR-TE`](final_feature_optimization/results/full/XGB-D03-PAIR-TE/metrics.json) | XGBoost | 0.94961810 | 0.526957 | Tối ưu đặc trưng |
| 20 | [`XGB-D03-PAIR-RARE-TE`](final_feature_optimization/results/full/XGB-D03-PAIR-RARE-TE/metrics.json) | XGBoost | 0.94958154 | 0.526451 | Tối ưu đặc trưng |
| 21 | [`D01`](plan_runs/results/full/D01/metrics.json) | LightGBM | 0.94878365 | 0.521507 | Kế hoạch chính |
| 22 | [`XGB-D01`](plan_runs/results/full/XGB-D01/metrics.json) | XGBoost | 0.94862328 | 0.521631 | Kế hoạch chính |
| 23 | [`T04`](plan_runs/results/full/T04/metrics.json) | LightGBM | 0.94857413 | 0.520055 | Kế hoạch chính |
| 24 | [`DERIVED`](plan_runs/results/full/DERIVED/metrics.json) | LightGBM | 0.94854209 | 0.520315 | Kế hoạch chính |
| 25 | [`T02`](plan_runs/results/full/T02/metrics.json) | LightGBM | 0.94852669 | 0.518542 | Kế hoạch chính |
| 26 | [`T03`](plan_runs/results/full/T03/metrics.json) | XGBoost | 0.94849461 | 0.521020 | Kế hoạch chính |
| 27 | [`LightGBM + TE · vòng 1`](results/lightgbm-te.json) | LightGBM | 0.94832930 | 0.516402 | GPU vòng 1 |
| 28 | [`XGBoost + TE · vòng 1`](results/xgboost-te.json) | XGBoost | 0.94831147 | 0.518785 | GPU vòng 1 |
| 29 | [`XGBoost + TE theo cặp · vòng 2`](round2/results/xgboost-pair-te.json) | XGBoost | 0.94819325 | 0.518245 | GPU vòng 2 |
| 30 | [`GLM-NEWTON`](plan_runs/results/full/GLM-NEWTON/metrics.json) | Hồi quy logistic | 0.94765641 | 0.515512 | Kế hoạch chính |
| 31 | [`GLM-C0.3`](plan_runs/results/full/GLM-C0.3/metrics.json) | Hồi quy logistic | 0.94760502 | 0.515563 | Kế hoạch chính |
| 32 | [`P1-P2`](plan_runs/results/full/P1-P2/metrics.json) | Hồi quy logistic | 0.94758091 | 0.515491 | Kế hoạch chính |
| 33 | [`GLM-C0.1`](plan_runs/results/full/GLM-C0.1/metrics.json) | Hồi quy logistic | 0.94755777 | 0.515294 | Kế hoạch chính |
| 34 | [`P1`](plan_runs/results/full/P1/metrics.json) | Hồi quy logistic | 0.94748391 | 0.515070 | Kế hoạch chính |
| 35 | [`P1-P2-P3`](plan_runs/results/full/P1-P2-P3/metrics.json) | Hồi quy logistic | 0.94738532 | 0.514208 | Kế hoạch chính |
| 36 | [`S6-L04`](../../reports/step6_review_checks.json) | LightGBM | 0.94736526 | — | Độ phức tạp cây |
| 37 | [`S6-X02`](../../reports/step6_review_checks.json) | XGBoost | 0.94731671 | — | Độ phức tạp cây |
| 38 | [`Logistic + TE · vòng 1`](results/logistic-te.json) | Hồi quy logistic | 0.94729970 | 0.513789 | GPU vòng 1 |
| 39 | [`T01`](plan_runs/results/full/T01/metrics.json) | Hồi quy logistic | 0.94726101 | 0.513583 | Kế hoạch chính |
| 40 | [`GPU-S6-L04`](plan_runs/results/full/GPU-S6-L04/metrics.json) | LightGBM | 0.94723215 | 0.513262 | Kế hoạch chính |
| 41 | [`T01-M50`](plan_runs/results/full/T01-M50/metrics.json) | Hồi quy logistic | 0.94720801 | 0.513007 | Kế hoạch chính |
| 42 | [`CatBoost + TE · vòng 2`](round2/results/catboost-te.json) | CatBoost | 0.94715633 | 0.517428 | GPU vòng 2 |
| 43 | [`GPU-S6-X02`](plan_runs/results/full/GPU-S6-X02/metrics.json) | XGBoost | 0.94713909 | 0.513930 | Kế hoạch chính |
| 44 | [`B06`](../../reports/review_steps_0_5_checks.json) | XGBoost | 0.94696513 | — | Baseline Kaggle |
| 45 | [`COUNT`](plan_runs/results/full/COUNT/metrics.json) | LightGBM | 0.94682509 | 0.505820 | Kế hoạch chính |
| 46 | [`S6-L08`](../../reports/step6_review_checks.json) | LightGBM | 0.94675421 | — | Độ phức tạp cây |
| 47 | [`S6-L15`](../../reports/step6_review_checks.json) | LightGBM | 0.94634983 | — | Độ phức tạp cây |
| 48 | [`B05`](../../reports/review_steps_0_5_checks.json) | LightGBM | 0.94634907 | — | Baseline Kaggle |
| 49 | [`S6-L31`](../../reports/step6_review_checks.json) | LightGBM | 0.94622065 | — | Độ phức tạp cây |
| 50 | [`B07`](../../reports/review_steps_0_5_checks.json) | CatBoost | 0.94580786 | — | Baseline Kaggle |
| 51 | [`MLP-BCE`](imbalance/results/mlp/MLP-BCE/metrics.json) | MLP | 0.94489872 | 0.503905 | Lệch nhãn |
| 52 | [`MLP-FOCAL-G1`](imbalance/results/mlp/MLP-FOCAL-G1/metrics.json) | MLP | 0.94426659 | 0.502758 | Lệch nhãn |
| 53 | [`MLP-FOCAL-G2`](imbalance/results/mlp/MLP-FOCAL-G2/metrics.json) | MLP | 0.94400727 | 0.501783 | Lệch nhãn |
| 54 | [`TabM-mini · vòng 2`](round2/results/tabm-mini.json) | TabM | 0.94234968 | 0.489214 | GPU vòng 2 |
| 55 | [`ft-masked`](pretraining/results/full/ft-masked/metrics.json) | FT-Transformer | 0.94224129 | 0.485509 | Tiền huấn luyện |
| 56 | [`GPU-B02`](plan_runs/results/full/GPU-B02/metrics.json) | Hồi quy logistic | 0.94189036 | 0.489712 | Kế hoạch chính |
| 57 | [`B02`](../../reports/review_steps_0_5_checks.json) | Hồi quy logistic | 0.94187887 | — | Baseline Kaggle |
| 58 | [`ft-scratch`](pretraining/results/full/ft-scratch/metrics.json) | FT-Transformer | 0.94159885 | 0.483873 | Tiền huấn luyện |
| 59 | [`mlp-masked`](pretraining/results/full/mlp-masked/metrics.json) | MLP | 0.94155513 | 0.489459 | Tiền huấn luyện |
| 60 | [`MLP · vòng 1`](results/mlp.json) | MLP | 0.94145617 | 0.486874 | GPU vòng 1 |
| 61 | [`mlp-scratch`](pretraining/results/full/mlp-scratch/metrics.json) | MLP | 0.94124013 | 0.488468 | Tiền huấn luyện |
| 62 | [`MLP-PAIRWISE`](imbalance/results/mlp/MLP-PAIRWISE/metrics.json) | MLP | 0.94121284 | 0.487224 | Lệch nhãn |
| 63 | [`mlp-scarf`](pretraining/results/full/mlp-scarf/metrics.json) | MLP | 0.94120676 | 0.487462 | Tiền huấn luyện |
| 64 | [`FT-Transformer · vòng 1`](results/ft-transformer.json) | FT-Transformer | 0.93833215 | 0.474786 | GPU vòng 1 |
| 65 | [`B04`](../../reports/review_steps_0_5_checks.json) | Random Forest | 0.91724524 | — | Baseline Kaggle |
| 66 | [`B03`](../../reports/review_steps_0_5_checks.json) | Decision Tree | 0.79832716 | — | Baseline Kaggle |
| 67 | [`tabnet-masked`](pretraining/results/full/tabnet-masked/metrics.json) | TabNet | 0.75302268 | 0.094014 | Tiền huấn luyện |
| 68 | [`tabnet-scratch`](pretraining/results/full/tabnet-scratch/metrics.json) | TabNet | 0.73378241 | 0.085424 | Tiền huấn luyện |

## Phạm vi đối chiếu

- Đã kiểm SHA-256 của ba CSV chính thức và đối chiếu thứ tự ID cùng phép chia fold. **52 cấu hình** được tính lại trực tiếp từ mảng dự đoán OOF. **16 cấu hình** lấy điểm từ báo cáo đã lưu, gồm baseline Kaggle, độ phức tạp cây và GPU vòng 1.
- Mỗi dòng là một cấu hình tạo một điểm dự đoán. Ví dụ, XGBoost nhận embedding đã đóng băng vẫn có một mô hình chấm điểm cuối. Các bản gộp dự đoán, mô hình thử nhanh, điểm hằng/ngẫu nhiên và bản chạy lại đối chứng trùng được loại khỏi bảng.
- Hai cấu hình GLM-boost dùng GLM và cây trong cùng pipeline tạo điểm. OOF AUC của `GLM-BOOST-LGB` là 0.948623 và `GLM-BOOST-XGB` là 0.948544. [Bảng kết quả kế hoạch chính](plan_runs/results/full/comparison.md)
- Hai kết quả Optuna cũ còn xuất hiện trong bảng lịch sử. Chúng nằm ngoài thứ hạng vì chưa có dữ liệu để đối chiếu ID và fold. [Bảng lịch sử](../../plan.md#L63)
- Kết quả mới và cũ dùng cùng phép chia 5 fold. Thứ hạng mô tả điểm đo được của từng cấu hình đã chạy, với cách tạo đặc trưng và quy trình huấn luyện tương ứng. Chênh lệch nhỏ cần được đọc cùng đánh giá theo fold.
