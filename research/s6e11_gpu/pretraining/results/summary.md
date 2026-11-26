# S6E11 · Kết quả tiền huấn luyện trên RTX 4060

Đủ 699,960 dòng train, 299,984 dòng test và 5 fold đã chốt. Mỗi encoder tiền huấn luyện trên phần train của fold. Số epoch cố định trước, phần validation ngoài chỉ dùng để tính điểm. Các đầu phân loại dùng BCE.

Mỗi hướng tiền huấn luyện chạy 3 epoch. MLP, FT-Transformer và TabNet tinh chỉnh lần lượt 6, 5 và 4 epoch, bằng ngân sách huấn luyện có nhãn của đối chứng tương ứng. MLP và FT-Transformer che 7/24 cột. SCARF thay 14/24 cột. TabNet dùng tỷ lệ che 20% và đặt sàn phương sai 0,1 trong hàm khôi phục để giảm ảnh hưởng của các nhóm hiếm.

Mốc XGB-D03 **0.950182** và FINAL **0.950345**. Δ AUC của mỗi mô hình tiền huấn luyện được tính so với cùng kiến trúc khởi tạo ngẫu nhiên.

## Mô hình riêng

| Cấu hình | OOF ROC AUC | Δ AUC | Fold tăng | AP | Log-loss | Giây pretrain | Giây phân loại |
|---|---:|---:|---:|---:|---:|---:|---:|
| [mlp-scratch](full/mlp-scratch/metrics.json) | 0.941240 | — | — | 0.488468 | 0.047232 | 0.0 | 14.4 |
| [mlp-masked](full/mlp-masked/metrics.json) | 0.941555 | +0.000315 | 4/5 | 0.489459 | 0.047121 | 16.0 | 13.6 |
| [mlp-scarf](full/mlp-scarf/metrics.json) | 0.941207 | -0.000033 | 1/5 | 0.487462 | 0.047322 | 77.6 | 14.1 |
| [ft-scratch](full/ft-scratch/metrics.json) | 0.941599 | — | — | 0.483873 | 0.048165 | 0.0 | 40.1 |
| [ft-masked](full/ft-masked/metrics.json) | 0.942241 | +0.000642 | 3/5 | 0.485509 | 0.047531 | 42.7 | 45.6 |
| [tabnet-scratch](full/tabnet-scratch/metrics.json) | 0.733782 | — | — | 0.085424 | 0.077921 | 0.0 | 67.6 |
| [tabnet-masked](full/tabnet-masked/metrics.json) | 0.753023 | +0.019240 | 4/5 | 0.094014 | 0.077087 | 38.7 | 64.9 |

## AUC theo fold

| Cấu hình | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---:|---:|---:|---:|---:|
| `mlp-scratch` | 0.940926 | 0.942851 | 0.946039 | 0.941767 | 0.936464 |
| `mlp-masked` | 0.940917 | 0.943175 | 0.946652 | 0.942496 | 0.936866 |
| `mlp-scarf` | 0.941376 | 0.942641 | 0.945916 | 0.941419 | 0.935458 |
| `ft-scratch` | 0.939259 | 0.944218 | 0.947855 | 0.942410 | 0.937599 |
| `ft-masked` | 0.940441 | 0.944398 | 0.947598 | 0.942548 | 0.937014 |
| `tabnet-scratch` | 0.713220 | 0.702353 | 0.763325 | 0.753229 | 0.744434 |
| `tabnet-masked` | 0.720845 | 0.757241 | 0.783168 | 0.749279 | 0.759029 |

## Gộp dự đoán với FINAL

Tỷ lệ cố định 95% FINAL và 5% đầu ra mô hình mới. Cùng ID và cùng phép chia fold. Đối chứng scratch cũng được gộp ở tỷ lệ 95/5 để đối chiếu tiền huấn luyện.

| Thành viên mới | OOF ROC AUC | Δ so FINAL | Δ so gộp scratch | Fold tăng so FINAL | AP |
|---|---:|---:|---:|---:|---:|
| `mlp-masked` | 0.950292 | -0.000054 | +0.000004 | 1/5 | 0.529180 |
| `mlp-scarf` | 0.950289 | -0.000056 | +0.000002 | 1/5 | 0.529155 |
| `ft-masked` | 0.950291 | -0.000054 | -0.000004 | 0/5 | 0.529019 |
| `tabnet-masked` | 0.949165 | -0.001180 | -0.000173 | 0/5 | 0.529118 |

## Thêm embedding vào XGB-D03

Thêm 64 cột từ encoder MLP đã đóng băng trước tinh chỉnh. Các nhóm đầu vào gốc và số vòng boosting từng fold được giữ theo XGB-D03.

[xgb-d03-mlp-masked-embeddings](full/xgb-d03-mlp-masked-embeddings/metrics.json) đạt **0.949660**, Δ **-0.000522** so với XGB-D03, tăng ở 0/5 fold. So với FINAL, Δ **-0.000685**.

Các so sánh dùng cùng 5 fold khảo sát. Số giây là tổng thời gian huấn luyện theo fold. Embedding dùng encoder học từ dữ liệu gốc của fold, trước khi mạng học nhãn gian lận.

## Chạy lại

Chạy từ thư mục dự án với môi trường Python hỗ trợ CUDA đã dùng cho các lượt GPU. TabNet dùng `pytorch-tabnet==4.1.0`. Mỗi quá trình tiếp tục từ các fold đã lưu nếu cấu hình khớp.

```powershell
python s6e11_gpu/pretraining/run_pretraining.py
python s6e11_gpu/pretraining/compare_pretraining.py fusion
python s6e11_gpu/pretraining/compare_pretraining.py report
```
