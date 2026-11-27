# Hồ sơ nghiên cứu S6E11 từ baseline đến Grandmaster EDA

[README và cấu hình bản cuối](docs/README.md) · [Lịch sử từng phase](docs/07_LICH_SU_CHAT_VA_PHASE.md) · [Bảng điểm và danh mục CSV](docs/05_BANG_DIEM_VA_ARTEFACT.md) · [Tái lập](docs/06_TAI_LAP_VA_SUBMISSION.md)

Gói nghiên cứu lưu tám chương Markdown, mã các thí nghiệm, các báo cáo và số liệu nguồn. Bản phát hành gồm **Grandmaster EDA gốc** và **Grandmaster EDA hiệu chỉnh mức tiền thưa**. Trên tập giữ lại 104.994 dòng, AUC tăng từ 0.95122445 lên 0.95131585, log loss giảm từ 0.04383599 xuống 0.04380602. Holdout đã được sử dụng trong các lượt trước và khoảng tin cậy mức tăng AUC chứa 0. [Báo cáo EDA sâu](goal_0952/BAO_CAO_EDA_SAU.md) · [Đối chứng](goal_0952/submission_grandmaster_eda.csv) · [CSV ứng viên](goal_0952/deep_eda_measurement/submission_grandmaster_eda_amount_hole.csv)

## Mã các phiên bản gần đây

| Phiên bản | Mã và kết quả |
|---|---|
| Baseline, đặc trưng và các thử nghiệm đã lưu | [Bảng điểm theo phase](docs/05_BANG_DIEM_VA_ARTEFACT.md) · [Lịch sử thử nghiệm](docs/07_LICH_SU_CHAT_VA_PHASE.md) |
| FINAL trộn LightGBM 20%, trộn MLP 10% và hiệu chỉnh nhóm không bật cờ | [Huấn luyện và đánh giá theo seed](s6e11_gpu/followup/run_followup.py) · [Xuất ba submission](s6e11_gpu/followup/export_submissions.py) |
| Grandmaster EDA gốc 85/10/5 | [Báo cáo kỹ thuật](goal_0952/BAO_CAO_GRANDMASTER_EDA_GOC.md) · [Notebook](goal_0952/grandmaster_eda.ipynb) |
| Trộn cây, hiệu chỉnh lá và ghép số tiền | [Các phiên bản, mã chạy và kết quả](goal_0952/honest_leaves/BAO_CAO.md) |
| Năm bản hiệu chỉnh học ngoài fold | [Các phiên bản, mã chạy và kết quả](goal_0952/oof_calibration/BAO_CAO.md) |
| EDA về độ thưa và độ lặp lại | [Báo cáo và mã kiểm tra](goal_0952/oof_calibration/detailed_eda/BAO_CAO_EDA.md) |

Mã các thí nghiệm Grandmaster gần đây nạp dữ liệu chính thức từ `data/` và mặc định tìm cache trong `research/`. Nếu đã có cache ở nơi khác, đặt `GRANDMASTER_CACHE_PROJECT` tới thư mục chứa `goal_0952/`, `grandmaster_10fold_ablation/` và `grandmaster_refit/`. Mã Python luôn nạp từ repo. Các lệnh trong báo cáo tạo CSV thí nghiệm, điểm dự đoán trung gian và checkpoint ở máy chạy.

Để tái tạo ba CSV của lượt FINAL từ điểm seed 42 đã lưu, chạy lệnh sau từ gốc repo. Tham số `--gpu-results` trỏ tới thư mục chứa `plan_runs/results/full/` và `imbalance/results/mlp/`. [Mã xuất CSV](s6e11_gpu/followup/export_submissions.py)

```powershell
python -B research/s6e11_gpu/followup/export_submissions.py --gpu-results ../s6e11_gpu
```

## Kiểm tra bản phát hành

Chạy từ thư mục gốc của repo với Python 3.11 trở lên.

```powershell
python -B research/verify_release.py
python -B research/verify_release.py --test data/test.csv
```

Lệnh đầu kiểm checksum hai CSV, thứ tự và tính duy nhất của 299.984 ID, giá trị xác suất và đúng 4.448 dòng thay đổi theo công thức đã chốt. Khi có `data/test.csv`, lệnh thứ hai còn kiểm SHA-256 của dữ liệu chính thức và dùng 404 mức số tiền để đối chiếu các ID được hiệu chỉnh. [Manifest mô hình](goal_0952/deep_eda_measurement/amount_hole_submission.json) · [Manifest xác minh](goal_0952/deep_eda_measurement/release_verification.json)

Tái tạo CSV chính xác đến từng byte bằng NumPy và SciPy. Đoạn sau lưu tệp mới trong thư mục kết quả local.

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
python -B research/replay_release.py --output outputs/amount_hole_replayed.csv
```

Script dùng 4.448 log-odds gốc từ archive và giữ chuỗi xác suất gốc trên các dòng còn lại. SHA-256 của kết quả phải khớp `7068b053cb3e39a07b544c95d9f66b98958e0722d8669a82f465284062af1287`. Dự đoán và thước đo holdout vẫn tự kiểm tra được bằng dữ liệu train chính thức cùng `selected_predictions.npz`. [Mã tái tạo](replay_release.py) · [Hướng dẫn đầy đủ](docs/06_TAI_LAP_VA_SUBMISSION.md)

## Cấu trúc

| Thư mục | Vai trò |
|---|---|
| [docs](docs/README.md) | Tám chương từ baseline, EDA và đặc trưng đến thử nghiệm, bảng điểm và bàn giao. |
| [goal_0952](goal_0952/BAO_CAO_0952.md) | Runner, score và các phase cải tiến Grandmaster EDA. |
| [grandmaster_refit](grandmaster_refit/run_refit.py) · [grandmaster_10fold_ablation](grandmaster_10fold_ablation/run_ablation.py) | Mã tái lập Grandmaster và các phép kiểm 10 fold, cùng hồ sơ điểm. |
| [novel_model_experiments](novel_model_experiments/BAO_CAO.md) · [eda_review](eda_review/ket_qua_eda.md) | Các hướng 5 fold và kiểm tra đặc trưng giai đoạn đầu. |
| [s6e11_gpu](s6e11_gpu/xep_hang_mo_hinh_don.md) | Các phép thử bổ sung, báo cáo lịch sử và cấu hình đã đo. Pipeline GPU chính của repo vẫn ở [s6e11_gpu/plan_runs](../s6e11_gpu/plan_runs/README.md). |
| [Model tham khảo](<Model tham khảo/VERSION_COMPARISON.md>) | Thiết kế và các điểm đã ghi của V1, V2 và Grandmaster. |

Đặt ba CSV chính thức ở [data](../data/README.md). Git bỏ qua chúng, môi trường Python và các checkpoint lớn sinh từ huấn luyện. Các đường dẫn CSV lịch sử trong bảng điểm chỉ tên hiện vật ở workspace cũ. Repo mang hai CSV cần cho bản phát hành và các ảnh chụp số liệu dùng đối chiếu. Các runner đã chuyển sang đường dẫn `data/` trong repo. Khi huấn luyện lại theo phase, tạo checkpoint và manifest cùng lượt theo hướng dẫn của runner. [Phạm vi bản phát hành](docs/06_TAI_LAP_VA_SUBMISSION.md)

## Quy tắc dự án

- Pipeline dùng dữ liệu được cung cấp cho cuộc thi và các dự đoán do mô hình trong dự án tự huấn luyện.
- Huấn luyện và suy luận chạy bằng ML thông thường. Trợ lý có thể hỗ trợ mã và phân tích. Dữ liệu, đặc trưng, nhãn, embedding và dự đoán trong pipeline được tạo từ dữ liệu cuộc thi và mã đã mô tả.
- Điểm OOF, holdout, đảo vai trò và Public Leaderboard được ghi kèm nguồn và giao thức. CSV bản phát hành là dự đoán test chưa có điểm Kaggle trong hồ sơ này.
