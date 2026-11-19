# Kết quả tổ chức lại workspace — 17/11/2026

Workspace được tổ chức lại ngay tại `D:\2026\INT2403E`, theo yêu cầu gom dữ liệu, giữ phần lịch sử hữu ích và bỏ các bản thừa.

## Đã giữ và chuyển

- Ba CSV gốc → `data/`; checksum khớp dữ liệu của run Optuna đầy đủ ban đầu.
- Hai tài liệu tham khảo chính → `docs/references/`.
- Code, findings/kế hoạch cũ, notebook EDA và hình ảnh → `legacy/`.
- Baseline OOF/test → `legacy/runs/baselines/`.
- Optuna full run → `legacy/runs/optuna/20261113T183409930114Z/`.
- Linear stack, model đã học và submission → `legacy/runs/blend/`.
- Hai file cũ khác nội dung trong RAR → `legacy/package_snapshots/rar_snapshot/`.

## Đã xóa

Tổng **3.045 file**, khoảng **594,12 MB** (566,6 MiB):

- `artifacts(2)`: cả 31 file trùng SHA256 với baseline còn giữ.
- Các bản sao code, dữ liệu, dự đoán trong `s6e11_kaggle_ready`; mọi file bị bỏ đã đối chiếu SHA256 với nguồn được giữ.
- RAR đóng gói cũ: đã đọc và xác minh đủ 21 file bên trong; 19 file trùng, hai file khác đã được lưu riêng.
- `mini_input`, `mini_runs`, `optuna_test_runs`, `*_smoke`: dữ liệu thử pipeline và kết quả cỡ nhỏ. Cấu hình các run còn trong `legacy/experiment_summaries/smoke_runs.json`.
- `optuna_deps`: thư viện cài kèm có thể tạo lại từ môi trường Python.
- Các ảnh EDA và submission XGBoost trùng nội dung, bytecode cũ và thư mục lồng đã rỗng.

Các path xóa/chuyển đều được kiểm tra là đường dẫn tuyệt đối nằm trong workspace; vùng `.git` và thư mục cấu hình được bảo vệ. Không sửa nội dung ba CSV gốc hoặc manifest/log của các run lịch sử.

## Nền tảng mới đã kiểm tra

- Sáu kiểm tra tự động về schema, ID, thứ tự submission và fold đã qua.
- Chạy runner trên dữ liệu đầy đủ từ thư mục làm việc khác đã thành công.
- Fold mới khớp hoàn toàn 699.960 dòng của run Optuna gốc.
- Chạy smoke trên 20.000 train/1.000 test, 2 fold đã thành công.
- Chỉ còn một `train.csv`, một `test.csv`, một `sample_submission.csv` trong project.

`s6e11_project_run.py` hiện thực hiện kiểm tra dữ liệu và tạo fold. Các giai đoạn huấn luyện mới sẽ được làm tiếp theo `plan.md`.

Chi tiết từng thao tác: [changes.json](changes.json). Kết quả kiểm tra nội dung RAR: [rar_contents.json](rar_contents.json). Số liệu tổng hợp: [summary.json](summary.json).

`reorganize.ps1` là bản lưu để kiểm toán thao tác đã thực hiện; không chạy lại script migration này. File `summary.json` đánh dấu migration đã hoàn thành.
