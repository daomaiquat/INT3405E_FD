# Nguồn dữ liệu duy nhất

Ba file gốc đặt tại đây:

| File | Số dòng |
|---|---:|
| `train.csv` | 699.960 |
| `test.csv` | 299.984 |
| `sample_submission.csv` | 299.984 |

`manifest.json` lưu SHA256 và kích thước, đối chiếu từ run Optuna đầy đủ ban đầu. Runner kiểm tra checksum trước khi nạp dữ liệu, không sửa CSV gốc. Train có 11.689 nhãn gian lận.

Không chép train/test vào folder run hoặc legacy. Khi chạy Kaggle, gắn chính folder dữ liệu này cùng code. Các CSV bị loại khỏi Git bằng `.gitignore`; metadata và checksum vẫn được quản lý cùng code.
