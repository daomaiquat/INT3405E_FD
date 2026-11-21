# S6E11 — Credit Card Fraud

Dự án phát hiện gian lận thẻ tín dụng trên Kaggle Playground Series S6E11. Đặt dữ liệu gốc trong `data/`. Kế hoạch đầy đủ ở [plan.md](plan.md).

Kết quả GPU local mới nhất đạt **OOF ROC AUC 0.950345** với GLM-NEWTON và XGBoost D03 qua xác nhận lồng nhau 5×5. [Báo cáo kết quả GPU](s6e11_gpu/plan_runs/published/full/comparison.md)

## Cấu trúc hiện tại

```text
INT2403E/
  plan.md                     # kế hoạch project từ đầu đến cuối
  s6e11_project_run.py         # runner: audit, fold, EDA và contract smoke
  configs/default.json        # cấu hình dùng chung
  data/                       # train/test/sample + checksum manifest
  src/                        # nạp dữ liệu, audit, EDA, features, models, training, artifacts
  tests/                      # kiểm tra schema, ID, fold, EDA, baseline và chất lượng dữ liệu
  docs/references/            # guide và bài học tham khảo
  legacy/                     # code, notebook và kết quả lịch sử hữu ích
  reports/reorganization/     # nhật ký chuyển/xóa file
  outputs/                    # các run mới, không đưa vào Git
```

## Chạy tại máy local

Môi trường Python hiện tại đã có các thư viện cần thiết. Nếu tạo môi trường mới:

```powershell
python -m pip install -r requirements.txt
```

Chạy kiểm tra dữ liệu đầy đủ và tạo cùng 5 fold, seed 42:

```powershell
python .\s6e11_project_run.py
```

Chạy kiểm tra nhanh trên mẫu phân tầng, 2 fold:

```powershell
python .\s6e11_project_run.py --smoke
```

Các stage hỗ trợ: `setup`, `audit`, `folds`, `eda`, `contract`, `baseline`, `complexity`, `encoding`, `te_models`. Cờ dùng chung: `--smoke`, `--resume [RUN_ID|latest]`, `--threads N`, `--data-dir PATH`, `--output-dir PATH`, `--config PATH`. Với `--stage baseline`, chọn mô hình bằng `--b00` … `--b07`, `--models B02 B05` hoặc `--all-baselines`; phase 8 chọn bằng `--te-models T01 T02 T03`. `--threads` giới hạn BLAS/OMP. Smoke không tạo `submission.csv` chính thức.

Chạy kiểm tra hạ tầng OOF/test/bundle (bước 3, tự bật smoke):

```powershell
python .\s6e11_project_run.py --stage contract --threads 4
```

Chạy EDA toàn dữ liệu và ghi các bảng giả thuyết (bước 4):

```powershell
python .\s6e11_project_run.py --stage eda --threads 4
```

Chạy từng baseline bước 5 (code đã sẵn; chạy khi bạn muốn có số liệu):

```powershell
python .\s6e11_project_run.py --stage baseline --b02 --threads 4
python .\s6e11_project_run.py --stage baseline --models B02 B05 B06 --threads 4
python .\s6e11_project_run.py --stage baseline --all-baselines --threads 4
```

Bước 6 — ablation cây nhỏ và learning curve (không chạy lại B00–B07):

```powershell
python .\s6e11_project_run.py --stage complexity --threads 4
```

Trên Kaggle:

```python
!python "$(find /kaggle/input -name s6e11_project_run.py -print -quit)" --stage complexity --threads 4
```

Thử pipeline nhanh bằng cách thêm `--smoke`. Stage này chạy 6 cấu hình: LightGBM
4/8/15/31 lá, XGBoost depth 2/3. LightGBM dùng AUC-only early stopping; giữ nguyên
feature set raw/flags/one-hot. Learning curve dùng LightGBM 4 lá trên fold 0,
các tỷ lệ outer train 10/25/50/100%; đây là chẩn đoán một fold.

Đầu ra trong `<run_id>/complexity/`: `comparison.csv` (delta so control, số cây,
train gap và thời gian), `complexity_learning_curve.png`, `learning_curve/`,
`summary.json` và các thư mục candidate chứa OOF/test/bundle. Các estimator vẫn dùng
1 thread như baseline để giữ cấu hình fit; `--threads` giới hạn BLAS/OMP. Full run
mới là số liệu để kết luận; smoke không đủ ngân sách số cây để xếp hạng mô hình.
Resume dùng `--stage complexity --resume RUN_ID` với code/config/môi trường không đổi.

## Bước 7 — Target encoding cross-fit

Trên Kaggle (upload code mới như phần dưới, giữ nguyên dataset cũ):

```python
!python "$(find /kaggle/input -name s6e11_project_run.py -print -quit)" --stage encoding --threads 4
```

Stage này mã hoá 13 cột số với m-estimate `m=20`, khoá nguyên theo `round(100*x)`, 5 inner
fold phân tầng cùng `encoder_seed=2026`. Và test dùng bảng fit trên toàn T; E dùng bảng của
A, không dùng nhãn E. Có thể thêm `--smoke` để kiểm tra đường chạy.

Đầu ra trong `<run_id>/encoding/`: `config.json`, `fold_<k>/` với `T_crossfit.csv`,
`V_from_T.csv`, `test_from_T.csv`, `A_crossfit.csv`, `E_from_A.csv`, `inner_folds_*.csv`,
bảng TE và `cache_index.json`; `checks_summary.json`; `summary.json`. Gate gồm đổi nhãn V,
kiểm tra inner fold tự loại trừ, shuffle nhãn và ví dụ rò rỉ có chủ đích. `leak_demo/` luôn
`valid_registry=false`, không được dùng làm đặc trưng hay model hợp lệ. Không tạo submission.

## Phase 8 — Mô hình với TE đã có

Code tự tìm run Step7 hợp lệ trong `Step7/`, `outputs/` hoặc các input Kaggle,
đối chiếu dữ liệu/fold/tham số và đọc lại TE theo ID. Không cần chạy lại bước 7.

Chạy đầy đủ T01–T03 tại local:

```powershell
python .\s6e11_project_run.py --stage te_models --threads 4
```

Nếu muốn chạy GLM trước, thêm `--te-models T01`. Hai cây chạy riêng bằng
`--te-models T02 T03`. T01 là GLM + logit TE, T02 là LGBM 4 lá + TE, T03 là XGB
depth 2 + TE; các mô hình giữ raw/log/flags/one-hot như đối chứng tương ứng.
Giai đoạn này chỉ dùng TE 13 cột số. Nhánh T04/24 cột thực hiện sau khi có kết quả.

Chỉ định cache nếu không muốn tự tìm:

```powershell
python .\s6e11_project_run.py --stage te_models --encoding-dir .\Step7\20261118T134537Z_0364fc --threads 4
```

Trên Kaggle, gắn code mới (`src/`, `configs/`, bao gồm `phase8_controls.json`), dữ liệu
raw như cũ và run Step7. Tối thiểu run Step7 cần `encoding/`, `data_manifest.json`,
`fold_assignments.csv`, `early_stop_assignments.csv`; có thể upload nguyên folder
`Step7/` đã có, không cần nhân bản dữ liệu gốc.

```python
!python "$(find /kaggle/input -name s6e11_project_run.py -print -quit)" --stage te_models --threads 4
```

Thêm `--check-cache-only` để chỉ kiểm tra cache, hoặc `--smoke` để chạy mẫu với cache
smoke tương ứng. Smoke không lấy một phần cache full vì các inner fold sẽ khác nhau.
Cache khác môi trường vẫn dùng được nếu dữ liệu, vai trò fold và tham số encode khớp.

Đầu ra `<run_id>/te_models/`:

- `comparison.csv`: AUC/log-loss, delta gộp và từng fold so B02, S6-L04, S6-X02.
- `T01/`, `T02/`, `T03/`: OOF, test theo fold, metric, config, checkpoint và bundle.
- Bundle chứa bảng TE của T, preprocessing và model để dự đoán từ raw trên dữ liệu
  mới. Khi chấm train, phải dùng TE cross-fit như runner; không dùng full-T TE làm OOF.
- GLM có hệ số theo fold; `controls_used.json` ghi đối chứng từ output thật của bước
  5/6, không cần mang lại toàn bộ bundle mô hình cũ.

Phase 8 không tự tạo submission/stack. Đã kiểm tra cache full 5 fold, smoke T01–T03
và raw-inference roundtrip; chưa có kết quả huấn luyện full-data phase 8.

Kiểm tra tự động:

```powershell
python -m unittest discover -s tests -v
```

Runner tìm dữ liệu theo vị trí file code nên có thể gọi từ thư mục làm việc khác. Mỗi lần chạy tạo một `outputs/<run_id>/` mới, gồm `data_manifest.json`, `data_audit.json`, `data_quality.csv`, `environment.json`, `requirements.lock.txt`, source snapshot và `fold_assignments.csv`. Stage `eda` thêm `<run_id>/eda/` với `eda_report.md`, `findings.json` và các bảng CSV; stage `contract` thêm `<run_id>/contract_smoke/`; stage `baseline` thêm `<run_id>/baselines/<model>/` cộng `baseline_comparison.csv` và `baseline_oof_spearman.csv`; stage `complexity` thêm `<run_id>/complexity/`; stage `encoding` thêm `<run_id>/encoding/`. Baseline, complexity và encoding không tạo submission. Môi trường được khóa lần đầu sau run đầy đủ thành công tại `requirements.lock.txt`; không nâng thư viện giữa các nhánh ablation. Run lỗi có trạng thái `failed` trong `run_config.json`.

## Chạy trên Kaggle

Upload cùng một folder chứa `s6e11_project_run.py`, `src/`, `configs/` và `data/` làm Dataset rồi Add Input vào Notebook. Có thể chọn các thư mục này ngay từ project; không cần tạo thêm bản sao dữ liệu hoặc gói ZIP.

```python
!python "$(find /kaggle/input -name s6e11_project_run.py -print -quit)"
```

Đầu ra: `/kaggle/working/s6e11_project/<run_id>/`.

## Trạng thái thực hiện

- Đã làm (bước 0–4): dọn folder, hợp nhất dữ liệu và xác minh checksum, lưu kết quả cũ, khóa môi trường, runner có `--resume`/`--threads`, kiểm tra schema/ID, audit chất lượng dữ liệu, chốt fold/A/E/V và ba seed, metric thống nhất, ghép OOF/test theo ID, smoke fit + bundle/resume, EDA toàn dữ liệu với 8 giả thuyết có số mẫu. Báo cáo chi tiết từng bước: [reports/steps_progress.md](reports/steps_progress.md).
- Bước 5: B00–B07 đã chạy full-data trên Kaggle, `run_id=20261117T184301Z_75acd6`; artifacts copy tại `Step5/`. Best baseline là B06 XGBoost với OOF AUC 0,946965.
- Bước 6: ablation cây đã chạy full, `run_id=20261118T091556Z_367a2c`; artifacts copy tại `Step6/`. Winner: S6-L04 (LGBM 4 lá, OOF 0,947365) và S6-X02 (XGB depth 2, OOF 0,947317). Chi tiết: [reports/steps_progress.md](reports/steps_progress.md).
- Bước 7: encoder cross-fit đã chạy full local, `run_id=20261118T134537Z_0364fc`; artifacts tại `Step7/`. Gate rò rỉ đạt, shuffle-label AUC 0,500843; chưa tạo submission hay OOF mô hình T01–T04.
- Phase 8: code T01–T03 đã sẵn, dùng lại Step7; kiểm tra cache full và smoke đã qua. Chờ chạy full-model để đánh giá lợi ích TE.
- Chưa triển khai trong runner mới: feature engineering nâng cao, tuning và stacking. Thực hiện tiếp theo `plan.md`; baseline/complexity/encoding chỉ là development CV và không tạo submission mới.
- Linear stack cũ và submission cũ được giữ tại [legacy/runs/blend](legacy/runs/blend). Đây là kết quả lịch sử, không phải kết quả huấn luyện của runner mới.

Chi tiết dữ liệu: [data/README.md](data/README.md). Lịch sử còn giữ: [legacy/README.md](legacy/README.md). Nhật ký dọn dẹp: [reports/reorganization/README.md](reports/reorganization/README.md).
