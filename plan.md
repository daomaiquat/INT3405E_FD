# Kế hoạch xây dựng lại dự án S6E11 từ đầu

**Bài toán:** phát hiện gian lận thẻ tín dụng trên Kaggle Playground Series S6E11.  
**Ngày lập:** 17/11/2026.  
**Trạng thái:** đã tổ chức lại workspace, tạo nền tảng kiểm tra dữ liệu/fold, chạy EDA đầy đủ (bước 0–4), chạy xong baseline B00–B07 (bước 5) và ablation độ phức tạp cây + learning curve một fold (bước 6) trên Kaggle. Target encoding cross-fit bước 7 đã chạy full-data trên local và qua các gate rò rỉ; artifacts tại `Step7/`. Báo cáo từng bước ở [reports/steps_progress.md](reports/steps_progress.md).

## 1. Mục tiêu và phạm vi dự án

Xây một quy trình có thể chạy lại từ dữ liệu gốc đến submission, chứng minh được mỗi thay đổi giúp hay không giúp, và có đủ bằng chứng để viết báo cáo môn học.

### 1.1. Phạm vi dữ liệu

- Chỉ dùng `train.csv`, `test.csv`, `sample_submission.csv` của cuộc thi.
- Mọi mô hình đều tự huấn luyện; không dùng trọng số huấn luyện sẵn hoặc dự đoán của người khác.
- Tài liệu bên ngoài dùng để đặt giả thuyết, không dùng điểm của tác giả làm kết quả dự án.
- `id` dùng ghép dữ liệu, không đưa vào mô hình ở quy trình chính.
- Chỉ nhãn trong train được dùng để học. Không có nhãn test để tự tính AUC test.

### 1.2. Tiêu chí thành công

1. Dữ liệu, fold, cấu hình và phiên bản thư viện được ghi lại.
2. Có baseline, thí nghiệm đặc trưng, mô hình cải tiến và ensemble được đánh giá cùng quy trình.
3. Target encoding, early stopping, preprocessing và stacking có kiểm tra rò rỉ.
4. Báo cáo có giả thuyết, đối chứng, kết quả từng fold, ablation, phân tích lỗi và giới hạn.
5. Upload một folder lên Kaggle, chạy một lệnh để thực thi pipeline đã chọn; lưu log và kết quả vào một thư mục run mới.

**Thước đo chính được chốt trước thí nghiệm:** ROC AUC trên OOF gộp. Luôn kèm mean/std AUC 5 fold và delta từng fold. Không đổi sang phép đo khác chỉ vì một ứng viên trông tốt hơn ở phép đo đó. Average Precision (AP, kèm PR curve), log-loss và các chỉ số theo ngưỡng phục vụ phân tích bổ sung.

Các mốc khoảng `0.949–0.951` hoặc cao hơn trong tài liệu chỉ là tham khảo. Mục tiêu của dự án là cải thiện được kiểm chứng, không phải đạt một AUC được cam kết trước.

## 2. Cách kết hợp tài liệu tham khảo

Tài liệu chính:

- [Guide kỹ thuật](docs/references/credit-card-fraud-s6e11-guide.md): TE theo từng giá trị, khóa chữ số, cặp tương tác, GLM + cây, refit và hướng mạng nơ-ron.
- [Bài học ML thuần](docs/references/s6e11_pure_ml_lessons.md): thứ tự ưu tiên, ablation, mức tăng do nhiễu và giới hạn của bằng chứng.
- [Kế hoạch môn học ban đầu](legacy/docs/s6e11_ml_course_plan.md): yêu cầu EDA, baseline, bias–variance, learning curve và báo cáo.
- [Kế hoạch cải thiện từ trạng thái cũ](legacy/docs/s6e11_next_steps_plan.md): các khoảng trống cụ thể trong code hiện có.

### 2.1. Những điểm phải hiểu lại trước khi thực hiện

| Nội dung trong các tài liệu | Quyết định cho dự án này |
|---|---|
| Có nhiều thang điểm và mức tăng khác nhau | Ghi là kết quả tham khảo, không cộng các mức tăng từ các pipeline khác nhau |
| Một guide xem chữ số là nguồn tăng điểm lớn; tài liệu kia ưu tiên tính tổng quát | Thử TE giá trị thô trước; khóa chữ số là nhóm ablation riêng, ghi rõ đặc thù dữ liệu tổng hợp |
| Solver `newton-cholesky` và `lbfgs` được báo là thắng theo hai hướng trái nhau | Kiểm tra hội tụ trên cùng thiết kế đầu vào; không chọn solver vì tên hoặc kết quả ở pipeline khác |
| Count được báo vừa có ích vừa không có ích | Thử riêng trên mô hình TE đã chốt, không thêm sẵn vào mọi pipeline |
| Có công thức nhân số cây với 1.1 hoặc 1.25 khi refit | Đây là giả thuyết cần kiểm tra, không phải quy tắc toán học bắt buộc |
| “5-fold trên OOF” được gọi là nested stacking | Phân biệt CV tầng meta với nested CV của toàn hệ thống; bước xác nhận cuối phải xét cả mô hình gốc |
| AUC shuffle gần 0.50 được xem là chứng minh sạch hoàn toàn | Dùng như một phép kiểm tra cùng kiểm tra nguồn dữ liệu, fold và preprocessing |
| Tỷ lệ gian lận cùng tăng ở hai biến được gọi là tương tác | Có thể do cộng tính trên log-odds; chỉ kết luận cần tương tác khi ablation cải thiện |
| Cột yếu được yêu cầu luôn giữ, hoặc cây được nói là không thể học từng giá trị | Giữ làm đối chứng, rồi quyết định bằng ablation; cây có thể học nhưng có thể cần nhiều nhánh |

Hai lưu ý học thuật:

- Logit cộng tính tương ứng với **odds** kết hợp theo dạng nhân, không có nghĩa xác suất rủi ro nhân trực tiếp với nhau.
- `n_flags` là tổng của sáu cờ. Với Logistic Regression đã có đủ sáu cờ tuyến tính, thêm đúng tổng này không tự tạo tương tác mới. Nếu muốn kiểm tra phi tuyến theo số cờ, thử one-hot hoặc spline của `n_flags` thành một nhóm riêng.

## 3. Hiện trạng: thứ gì được kế thừa

Đã có dữ liệu đầy đủ: **699.960 dòng train, 299.984 dòng test, 11.689 nhãn dương**, khoảng **1,67%**. Schema gồm `id`, 24 đặc trưng và nhãn `is_fraud` trong train.

| Kết quả cũ | Mean AUC 5 fold | OOF AUC gộp | Cách dùng |
|---|---:|---:|---|
| XGBoost Optuna | 0.947023036 | 0.946986008 | Mốc mô hình đơn |
| LightGBM Optuna | 0.947000935 | 0.946968350 | Mốc mô hình đơn |
| Trung bình hạng ba cây | 0.947326781 | 0.947291807 | Đối chứng ensemble đơn giản |
| Logistic stack năm mô hình | 0.947323591 | 0.947251050 | Đối chứng stack đã học |

Nguồn: `legacy/runs/blend/comparison.csv`. Run Optuna đầy đủ: `legacy/runs/optuna/20261113T183409930114Z/`.

Những file cũ được giữ làm nguồn đối chiếu:

- `legacy/docs/s6e11_eda_findings.md`, `legacy/docs/s6e11_baselines_findings.md`: ý tưởng và bảng lịch sử, cần kiểm tra lại các kết luận chưa có số liệu thực.
- `legacy/runs/baselines/`: baseline OOF/test; các mảng `.npy` không có `id` nên cần xác minh thứ tự trước khi nhập vào hệ thống mới.
- `legacy/scripts/`: code EDA, baseline, Optuna, tune cây và linear stack cũ.
- `legacy/runs/blend/`: dự đoán và linear meta model đã fit, dùng làm đối chứng lịch sử.
- Các gói trùng, thư viện vendored và dữ liệu/run mini đã được dọn; cấu hình mini còn ở `legacy/experiment_summaries/`. Nhật ký chi tiết ở `reports/reorganization/`.

“Làm lại từ đầu” nghĩa là xây lại quy trình có cấu trúc và tái lập các mốc quan trọng. Giữ code, mô hình và kết quả cũ hữu ích trong `legacy/`; dữ liệu gốc chỉ có một nguồn tại `data/`. Các kết quả mới có `run_id` riêng và không ghi đè lịch sử.

## 4. Cấu trúc dự án và hợp đồng chạy trên Kaggle

### Bước 0 — Thiết lập folder dự án

**Mục đích:** có một nguồn code và một bộ dữ liệu đầu vào rõ ràng.

```text
INT2403E/                   # chính workspace hiện tại, không lồng thêm project
  s6e11_project_run.py       # điểm chạy duy nhất
  README.md                 # cách chạy và trạng thái kết quả
  requirements.txt          # môi trường đã kiểm tra
  configs/
    default.json            # cấu hình các giai đoạn chính
  data/
    train.csv
    test.csv
    sample_submission.csv
  src/
    data.py                 # schema, ID, checksum
    validation.py           # fold, early-stop split, metric
    features.py             # raw, log, flags, digits, pair keys
    encoders.py             # cross-fitted TE và count
    models.py               # GLM, LGBM, XGB, CatBoost, RF
    training.py             # một quy trình fit và refit chung
    ensemble.py             # averaging, stack, nested confirmation
    diagnostics.py          # ablation, bootstrap, lỗi theo nhóm
    artifacts.py            # log, bảng, dự đoán và model bundle
  tests/
    test_data_contract.py
    test_encoding_boundaries.py
    test_submission.py
  reports/
    experiment_log.csv
    findings.md
```

Đây là cấu trúc đích. Runner, `data/`, `configs/`, các module data/eda/validation/artifacts/metrics/predictions/smoke_model/features/models/training/complexity/target_encoding/encoding và kiểm tra nền tảng đã có; các module ensemble/diagnostics sẽ được thêm ở các bước tiếp theo. Nhật ký triển khai từng bước ở [reports/steps_progress.md](reports/steps_progress.md).
 `docs/` và `legacy/` nằm ngay trong workspace. Khi chạy Kaggle, upload code/config/data cùng một folder, không đóng ZIP hoặc tạo thêm bản sao dữ liệu theo mặc định.

Đầu ra Kaggle dự kiến:

```text
/kaggle/working/s6e11_project/<run_id>/
  run_config.json
  environment.json
  data_manifest.json
  source_snapshot/
  fold_assignments.csv
  feature_groups.json
  experiment_log.csv
  run.log
  events.jsonl
  <experiment_id>/
    metrics.json
    fold_metrics.csv
    oof.csv
    test_by_fold.csv
    model_bundle/
  final/
    submission.csv
    selection.json
```

**Đầu việc:**

- [x] Đưa ba CSV gốc vào `data/`, kiểm tra checksum so với nguồn hiện tại.
- [x] Tạo các module nền tảng dùng chung: data, validation và artifacts.
- [x] Ghi phiên bản Python, NumPy, pandas, SciPy, scikit-learn, LightGBM, XGBoost, CatBoost và joblib trong mỗi run.
- [x] Khóa môi trường sau lần chạy thành công; không nâng thư viện giữa hai nhánh ablation.
- [x] Thiết kế `--stage`, `--smoke`, `--resume`, `--threads`, `--data-dir`, `--output-dir`.
- [x] Quy định smoke không tạo file mang tên submission chính thức.

Lệnh đã có cho phần kiểm tra dữ liệu và tạo fold; sau này cùng entry point này sẽ được mở rộng để huấn luyện:

```python
!python "$(find /kaggle/input -name s6e11_project_run.py -print -quit)"
```

Hiện runner hỗ trợ `--stage setup|audit|folds|eda|contract|baseline|complexity|encoding`, `--smoke`, `--resume`, `--threads`, `--data-dir` và `--output-dir`; mặc định kiểm tra dữ liệu đầy đủ, ghi bảng chất lượng và tạo fold. `--stage eda` chạy toàn bộ dữ liệu và ghi bảng giả thuyết; `--stage contract` tự bật smoke chỉ để kiểm tra hạ tầng OOF/test/bundle; `--stage baseline` huấn luyện B00–B07; `--stage complexity` chạy ablation cây bước 6; `--stage encoding` tạo artifact TE cross-fit và chạy các gate rò rỉ bước 7. Baseline chạy tại `run_id=20261117T184301Z_75acd6`, complexity tại `run_id=20261118T091556Z_367a2c`; bước 7 chạy full local tại `run_id=20261118T134537Z_0364fc`. MLP/LM vẫn cần bật rõ trong config. Script đọc dữ liệu theo vị trí file, không phụ thuộc tên tài khoản Kaggle; trên Kaggle, kết quả ghi vào `/kaggle/working`.

**Điều kiện xong:** chạy smoke từ một thư mục làm việc khác vẫn tìm đúng dữ liệu, tạo run riêng và ghi đủ cấu hình.

## 5. Nền tảng dữ liệu và đánh giá

### Bước 1 — Kiểm tra dữ liệu gốc

**Đầu việc:**

- [x] Kiểm tra số dòng, tên cột, kiểu dữ liệu, missing và giá trị vô hạn.
- [x] Cast nhãn sang 0/1, xác nhận có đủ hai lớp.
- [x] Kiểm tra `id` duy nhất, train/test không trùng ID, sample có đúng ID và thứ tự test.
- [x] Tách 13 cột số, 5 cột phân loại, 6 cờ; loại `id` và nhãn khỏi feature list.
- [x] Kiểm tra số âm, phạm vi hợp lý và danh mục chưa gặp trong test.
- [x] Kiểm tra giao dịch trùng toàn bộ đặc trưng và nhóm trùng có nhãn mâu thuẫn.
- [x] Kiểm tra độ chính xác thập phân trước khi tạo khóa từ số thực.
- [x] Ghi checksum, kích thước file và số dòng vào `data_manifest.json`.

Không tự xóa dòng trùng hoặc outlier. Nếu có dấu hiệu các dòng cùng một thực thể, kiểm tra xem random split có gây phụ thuộc giữa train/validation hay không; nếu cần GroupKFold, tạo nhánh đánh giá riêng và giải thích vì sao không so thẳng AUC với random CV.

**Đầu ra:** bảng chất lượng dữ liệu và manifest.  
**Điều kiện xong:** mọi sai khác so với schema dự kiến đều được giải thích hoặc xử lý có ghi log.

### Bước 2 — Chốt quy trình CV trước khi thêm mô hình mạnh

**Đầu việc:**

- [x] Tạo hoặc tái sử dụng đúng fold theo `id`: `StratifiedKFold(5, shuffle=True, random_state=42)`.
- [x] Lưu `id,is_fraud,fold` và checksum của phép chia.
- [x] Các run cùng một nhóm ablation dùng chung fold và early-stop split.
- [x] Chốt seed mô hình, seed encoder và seed early stop thành ba trường riêng.
- [x] Với cây: trong 80% outer train T, tách 90% A để fit và 10% E để chọn số cây. E chiếm 8% toàn train, A chiếm 72%.
- [x] Outer validation V, chiếm 20%, chỉ dùng chấm sau khi fit xong.
- [x] Sau khi chọn số cây trên E, fit lại pipeline trên toàn T rồi chấm V.
- [x] Fit scaler, one-hot, binning và mọi thống kê được học trong đúng phần train tương ứng.

Không gọi một tập hold-out mới cắt từ dữ liệu đã dùng phát triển trước đây là “chưa từng đụng tới”. Điểm lịch sử đã được xem trên toàn bộ dữ liệu. Kế hoạch dùng CV phát triển và bước xác nhận toàn pipeline ở Bước 15; giới hạn do chọn mô hình nhiều lần vẫn cần ghi trong báo cáo.

**Đầu ra:** fold cố định, early-stop split, hàm tính metric thống nhất.  
**Điều kiện xong:** các nhóm không trùng dòng; validation có đủ hai lớp; test không tham gia fit các bước có nhãn.

### Bước 3 — Xây runner và các phép kiểm tra cần thiết

Mọi mô hình phải trả cùng định dạng:

- OOF: `id,is_fraud,fold,prediction`.
- Test: `id,fold_0,...,fold_4,prediction`; prediction là trung bình các fold ở chế độ mặc định.
- Metric: OOF AUC gộp, AUC từng fold, mean/std, train AUC, log-loss, thời gian, số cây và cảnh báo hội tụ.

**Đầu việc:**

- [x] Ghép dự đoán theo `id`, kiểm tra one-to-one, không ghép mù theo vị trí.
- [x] Mỗi dòng train nhận đúng một dự đoán từ mô hình không học dòng đó trong outer fit.
- [x] Kiểm tra hằng số cho AUC 0.50; dự đoán ngẫu nhiên độc lập quanh 0.50, không ép phải đúng 0.5000.
- [x] Kiểm tra chạy lại cùng seed cho sai khác dự đoán trong dung sai đã ghi.
- [x] Kiểm tra validation/test bị đảo thứ tự vẫn được ghép đúng theo ID.
- [x] Chạy smoke trên mẫu có phân tầng để xác nhận fit, ghi file và đọc lại model bundle.
- [x] Ghi trạng thái `started/completed/failed` để resume không bỏ qua fold bị lỗi.

**Điều kiện xong:** các kiểm tra này qua trước khi tin kết quả TE hoặc stack. Không cần viết test mô phỏng mọi chi tiết thư viện; tập trung ranh giới dữ liệu và output.

### Bước 4 — Làm EDA dẫn tới giả thuyết

| Câu hỏi | Cần tính/vẽ | Quyết định dự kiến |
|---|---|---|
| Mất cân bằng đến đâu? | Số nhãn dương, fraud rate | Chọn metric và phân tầng |
| Cột nào có tín hiệu đơn biến? | Fraud rate theo giá trị/phân vị, số mẫu mỗi nhóm | Raw, log, spline hay TE |
| Cờ có cộng dồn? | Fraud rate theo `n_flags`, kèm số mẫu | Ablation số cờ, không mặc định có lợi |
| Bao nhiêu fraud không có cờ? | Tỷ lệ fraud và các mẫu no-flag | Trọng tâm phân tích lỗi |
| Các giá trị số lặp lại thế nào? | Cardinality, tần suất, tỷ lệ chưa gặp trong validation/test | Ưu tiên TE và smoothing |
| Chữ số có tín hiệu ổn định? | Fraud rate theo cents/digit trên hai phần phân tầng | Có đưa nhánh digit vào ablation |
| Hai biến có cần tương tác riêng? | Bảng chéo có đủ số mẫu | Pair key cho GLM |
| Train/test có khác biệt rõ? | Phân phối, danh mục, adversarial validation nếu cần | Đánh giá mức phù hợp của random CV |

**Đầu việc:**

- [x] Mỗi bảng fraud rate kèm số dòng và số fraud để tránh kết luận từ nhóm quá hiếm.
- [x] Kiểm tra lại các con số no-flag, CVV, risk score trong guide trước khi trích vào báo cáo như số của dự án.
- [x] Tách phần discovery/confirmation khi chọn các đuôi thập phân hoặc cặp rất cụ thể; không lựa chọn một đuôi “hot” trên toàn train rồi coi như đã xác nhận.
- [x] Không dùng p-value hoặc adversarial AUC gần 0.50 để kết luận chắc chắn train/test cùng phân phối.
- [x] Ghi 5–8 phát hiện, mỗi phát hiện có giả thuyết mô hình tương ứng và trạng thái đã/chưa kiểm chứng.

**Đầu ra:** báo cáo EDA, bảng cardinality/unseen và danh sách giả thuyết có thứ tự ưu tiên.  
**Điều kiện xong:** có đủ lý do để thử cây nhỏ, TE, digit và pair; không cần vẽ mọi biểu đồ có thể có.

## 6. Baseline và độ phức tạp mô hình

### Bước 5 — Tái lập thang baseline

> Cập nhật 18/11/2026: registry B00–B07, preprocessing, early-stop/refit và cờ CLI đã có
> (`--stage baseline`, `--b00` … `--b07`, `--models`, `--all-baselines`). Đã chạy full-data
> trên Kaggle, `run_id=20261117T184301Z_75acd6`; số liệu và kiểm tra artifact ở
> [reports/steps_progress.md](reports/steps_progress.md).

| ID | Mô hình | Đầu vào và mục đích |
|---|---|---|
| B00 | Hằng số theo prior train | Mốc AUC 0.50 |
| B01 | Điểm ngẫu nhiên | Kiểm tra đánh giá |
| B02 | Logistic Regression L2 | Raw numeric, log cho amount/balance, cờ, one-hot; scaler trong fold |
| B03 | Decision Tree depth 4 | Quy tắc đơn giản và ví dụ bias cao |
| B04 | Random Forest | Bagging; một cấu hình cố định, không quét rộng |
| B05 | LightGBM | Cấu hình đối chứng đã chốt, ghi rõ số lá |
| B06 | XGBoost | Cấu hình đối chứng đã chốt, ghi rõ depth |
| B07 | CatBoost | Đặc trưng phân loại theo cách của CatBoost |

**Đầu việc:**

- [x] Chạy cùng outer fold; chỉ khác preprocessing cần thiết cho từng họ mô hình.
- [x] Giữ raw numeric và các cờ cho cây; không chuẩn hóa cây chỉ vì GLM cần scaler.
- [x] One-hot học từ train, xử lý danh mục chưa gặp; không target encode ở giai đoạn này.
- [x] Với GLM, ghi solver, C, tolerance, số vòng và cảnh báo hội tụ.
- [x] Với cây, dùng cùng quy trình chọn số cây/refit đã định; mô hình số cây cố định phải ghi rõ chế độ.
- [x] So với số cũ để tìm lỗi lớn, không yêu cầu khớp tuyệt đối khi môi trường hoặc quy trình đã đổi.
- [x] Chỉ làm một ablation class weighting nếu cần bằng chứng cho báo cáo; không mặc định SMOTE hoặc trọng số lớp tốt cho AUC.

**Đầu ra:** bảng baseline, train–CV gap và Spearman OOF giữa mô hình.  
**Điều kiện xong:** hiểu được vì sao GLM/cây nông/RF/GBDT khác nhau. Các mốc cũ khoảng 0.942 cho Logistic và 0.944–0.947 cho GBDT dùng để kiểm tra sơ bộ, không là điều kiện điểm bắt buộc.

### Bước 6 — Kiểm tra cây nhỏ và learning curve

**Đầu việc:**

- [x] LightGBM: thử `num_leaves={4,8,15,31}` trên cùng feature set.
- [x] XGBoost: thử `max_depth={2,3}`; thêm depth 4 nếu cần hoàn thiện đường cong.
- [x] Giữ learning rate, regularization, subsampling và cách chọn số cây thống nhất trong mỗi đường cong.
- [x] Khởi đầu learning rate 0.03, trần 6.000 cây, patience 200; chỉ tăng trần nếu log cho thấy mô hình còn cải thiện khi chạm trần.
- [x] Nếu LightGBM dùng `subsample<1`, đặt `subsample_freq>0` để lấy mẫu dòng thực sự hoạt động.
- [x] Vẽ AUC, train–CV gap và thời gian theo độ phức tạp.
- [x] Learning curve trên một mô hình đại diện: lấy mẫu phân tầng phần train của fold, giữ cùng validation; ghi rõ nếu chỉ đo trên một fold.

**Lưu ý:** LightGBM 4 lá không đảm bảo cây cân bằng depth 2; `num_leaves` và `max_depth` là hai ràng buộc khác nhau. [Tham số LightGBM](https://lightgbm.readthedocs.io/en/stable/Parameters.html)

**Đầu ra:** hai cấu hình cây đối chứng cho các bước TE.  
**Điều kiện xong:** chọn cấu hình theo AUC và chi phí, không mặc định cây nhỏ chắc chắn thắng như tài liệu.

## 7. Phát triển đặc trưng theo từng nhóm

### Bước 7 — Xây target encoder có cross-fitting

Với khóa giá trị v, trên phần dữ liệu tham chiếu R được phép:

```text
prior_R = mean(y_R)
TE_R(v) = (sum(y_R tại v) + m * prior_R) / (count_R(v) + m)
```

**Quy trình cho mỗi outer fold:**

1. Giữ riêng outer train T và validation V.
2. Chia T thành 5 inner fold. Mỗi phần nhận TE tính từ 4 phần còn lại.
3. Fit bảng TE trên toàn T để transform V và test; giá trị mới về prior của T.
4. Với early stopping, tách A/E trước rồi lặp lại cross-fitting chỉ trong A. E dùng bảng của A, tuyệt đối không lấy nhãn E vào đặc trưng A.
5. Sau chọn số cây, tạo lại cross-fitted TE trong toàn T để fit model cuối của fold.
6. Lưu cấu hình và cache theo checksum dữ liệu, ID của phần tham chiếu, nhóm cột, m, inner split, seed và phiên bản code.

**Đầu việc:**

- [x] Khởi đầu `m=20`, một seed encoder, 5 inner fold có phân tầng.
- [x] Encode 13 cột số trước; giữ raw numeric, cờ và one-hot làm phần đối chứng.
- [x] Dùng giá trị nguyên hóa theo độ chính xác gốc làm khóa ổn định; không làm tròn thô ngoài thiết kế thí nghiệm.
- [x] Trường hợp nhóm hiếm, khóa chưa gặp, bool và missing có quy tắc rõ ràng.
- [x] Dùng tự cài đặt theo công thức m-estimate: `fit_transform` tương đương cross-fit trên train, `transform` dùng bảng của phần train cho V/test.
- [x] Đối chiếu công thức trên ví dụ nhỏ tính tay và kiểm tra ranh giới fold bằng unit test và artifact check.

`fit(X,y).transform(X)` trên chính train khác với `fit_transform` có cross-fitting; không thay thế lẫn nhau. [Tài liệu TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)

**Kiểm tra bắt buộc:**

- [x] Thay nhãn của outer V không làm đổi encoder hoặc mô hình học từ T.
- [x] Dòng được giữ lại trong inner fold không tham gia tổng nhãn/prior dùng encode chính nó.
- [x] Xáo trộn nhãn và chạy lại encoder + model cho AUC quanh 0.50 trong biến thiên lấy mẫu; không ép khoảng 0.48–0.50 vì điểm có thể nằm trên 0.50 do ngẫu nhiên.
- [x] Ví dụ TE rò rỉ có chủ đích, nếu dùng để giảng giải, được đánh dấu riêng và bị loại khỏi registry mô hình hợp lệ.

**Điều kiện xong:** các phép kiểm tra qua trước khi chạy đủ cây TE. Một điểm CV cao bất thường là lý do kiểm tra lại, không phải tự động xác nhận tiến bộ.

### Bước 8 — Đo lợi ích per-value TE trên ba họ mô hình

| ID | Mô hình mới | So với |
|---|---|---|
| T01 | GLM + logit TE 13 cột số | B02 |
| T02 | LightGBM nhỏ + TE 13 cột số | Cây raw đã chọn ở Bước 6 |
| T03 | XGBoost nhỏ + TE 13 cột số | Cây raw đã chọn ở Bước 6 |
| T04 | Mô hình TE tốt nhất, mở rộng TE sang cả 24 cột | Chính mô hình đó với TE 13 cột |

**Đầu việc:**

- [ ] GLM lấy `logit(clip(TE,1e-6,1-1e-6))`, chuẩn hóa trong fold rồi học Logistic Regression.
- [ ] Giữ phần raw/log/one-hot giống đối chứng, để hiệu quả TE đo được riêng.
- [ ] Cây nhận TE dưới dạng số; không cần logit chỉ vì GLM dùng logit.
- [ ] Ghi delta theo cột/nhóm khi cần biết TE của các cột yếu có thực sự giúp.
- [ ] Chỉ sau kết quả đầu mới so `m=20` với `m=50` trên GLM tốt nhất; không đổi m đồng thời đổi solver và feature set.
- [ ] Đánh giá mô hình đơn và khả năng bổ sung cho ensemble; không giữ một mô hình chỉ vì nó khác tên.

**Thứ tự chạy:** GLM TE trước để kiểm tra nhanh, rồi LightGBM TE, XGBoost TE, sau đó mới mở rộng all-column TE.

**Đầu ra:** OOF/test của các mô hình TE hợp lệ và bảng delta.  
**Điều kiện xong:** chọn được nhánh TE có lợi hoặc giải thích được vì sao không tái hiện lợi ích trong tài liệu.

### Bước 9 — Ablation khóa chữ số

Chỉ bắt đầu sau khi TE giá trị thô đã được kiểm tra. Sáu cột đề xuất:

`amount_usd`, `hours_since_last_txn`, `distance_from_home_km`, `account_balance_usd`, `velocity_score`, `merchant_risk_score`.

Sau khi xác nhận dữ liệu có độ chính xác phù hợp, đặt `units=round(100*x)` dưới dạng số nguyên. Với ví dụ `x=1371.37`:

| Khóa | Công thức/ý nghĩa | Ví dụ |
|---|---|---:|
| `cents` | `units % 100` | 37 |
| `int` | `units // 100` | 1371 |
| `int10` | `units // 1000` | 137 |
| `tenths` | `units // 10` | 13713 |
| `lead1` | Độ dài phần nguyên + chữ số đầu | `4\|1` |
| `lead2` | Độ dài phần nguyên + hai chữ số đầu | `4\|13` |

**Đầu việc:**

- [ ] Tạo 36 khóa chỉ từ X; kiểm tra số 0, số dưới 1, giá trị tròn, chữ số 0 ở phần cents và sai số float.
- [ ] Tạo tên cột theo nguồn để không trùng giữa các biến.
- [ ] Chạy D01: mô hình TE tốt nhất + 6 cột cents dạng numeric.
- [ ] Chạy D02: cùng mô hình + TE của 36 digit keys.
- [ ] Chạy D03: cùng mô hình + TE digit keys và cents numeric, chỉ khi D01/D02 cho cơ sở để kết hợp.
- [ ] Nếu có lợi, bỏ từng nhóm khóa để xác định nhóm đóng góp; không suy ra 36 khóa đều cần thiết.

**Đầu ra:** bảng ablation raw TE → cents → digit TE → kết hợp.  
**Điều kiện xong:** lợi ích qua nhiều fold và qua kiểm tra ổn định. Báo cáo đây là đặc trưng theo cấu trúc dữ liệu tổng hợp, cần kiểm chứng lại nếu chuyển sang dữ liệu thật.

### Bước 10 — Cặp tương tác và count

Ưu tiên pair keys cho GLM vì mô hình này chưa tự biểu diễn mọi tương tác. Không sinh mọi cặp của tất cả cột.

Danh mục 13 cặp từ guide, dùng như danh sách ứng viên:

| Nhóm | Khóa |
|---|---|
| P1, thử trước | `merchant_category × merchant_risk_score`; `is_foreign_transaction × ip_country_mismatch`; `merchant_category × decile(amount_usd)`; `txn_count_last_24h × decile(velocity_score)` |
| P2 | `merchant_category × customer_age`; `decile(amount_usd) × decile(merchant_risk_score)`; `decile(hours_since_last_txn) × decile(velocity_score)`; `customer_age × decile(merchant_risk_score)` |
| P3 | `is_new_merchant × day_of_week`; `is_ai_generated_scam_attempt × decile(velocity_score)`; `customer_age × card_age_months`; `time_of_day_hour × day_of_week`; `txn_count_last_24h × customer_age` |

**Đầu việc:**

- [ ] Với decile, fit biên phân vị chỉ trên phần train phù hợp; xử lý biên trùng và giá trị ngoài miền.
- [ ] Ghép khóa bằng tuple hoặc cách biểu diễn không gây va chạm.
- [ ] Target encode các khóa bằng quy trình Bước 7, khởi đầu cùng m với GLM đối chứng.
- [ ] Thêm P1 rồi P2/P3 theo nhóm, giữ raw features; không ép phải dùng đủ 13 cặp.
- [ ] Count là thí nghiệm riêng: thêm `log1p(count)` theo khóa vào mô hình TE tốt nhất.
- [ ] Mặc định count chỉ fit từ train tương ứng. Count train+test không nhãn là một nhánh dùng phân phối test, cần ghi rõ và đánh giá riêng.
- [ ] Thử `n_flags` phi tuyến hoặc tỷ lệ amount/balance nếu có giả thuyết từ EDA; không thêm gộp vào nhánh pair để quy kết điểm tăng sai.

**Đầu ra:** bảng đóng góp pair/count và danh sách feature groups được giữ.  
**Điều kiện xong:** mỗi nhóm được giữ có lợi ích riêng hoặc có bằng chứng bổ sung trong stack.

## 8. Mô hình nâng cao và ngân sách thử nghiệm

### Bước 11 — Hoàn thiện GLM và tuning có giới hạn

**Đầu việc:**

- [ ] Kiểm tra `n_iter_`, cảnh báo hội tụ, log-loss và AUC trước khi đổi solver.
- [ ] Khởi đầu `lbfgs` với đủ số vòng và tolerance hợp lý; thử `newton-cholesky` trên cùng X, C và tiêu chí hội tụ nếu còn nghi vấn.
- [ ] Theo dõi RAM khi dùng `newton-cholesky`; Hessian có chi phí bộ nhớ bậc hai theo số đặc trưng × số lớp.
- [ ] Chỉ quét `C` trong một tập nhỏ như `{0.1,0.3,1}` trên GLM tốt nhất.
- [ ] Chỉ tune các cây đã có feature set tốt, giới hạn trước số trial, ví dụ tối đa 12 trial cho một họ được chọn.
- [ ] Tách tìm cấu hình khỏi bước xác nhận cuối; kết quả chọn trên development CV được ghi là development score.
- [ ] Tất cả trial giữ OOF/test và log cấu hình; không chỉ lưu trial thắng.

Logistic Regression tối ưu loss phân loại có điều chuẩn, không trực tiếp tối ưu ROC AUC. Cấu hình cuối được chọn theo quy trình đánh giá đã chốt. [Tài liệu LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

**Điều kiện xong:** chọn được ít cấu hình có thể giải thích và tái lập; dừng khi ngân sách hết hoặc các thay đổi chỉ dao động trong nhiễu.

### Bước 12 — GLM làm nền cho cây nhỏ

**Giả thuyết:** GLM học phần cộng tính; cây sửa phần còn sai.

**Đầu việc:**

- [ ] Trong phần train A, fit encoder, scaler và GLM chỉ bằng A; lấy `z_A=decision_function(A)`.
- [ ] Tính `z_E` cho tập early stop bằng đúng pipeline học từ A.
- [ ] LightGBM nhận `init_score=z_A`, eval nhận `z_E`; thử 4 lá trước.
- [ ] Sau chọn số cây, fit lại GLM và pipeline trên toàn T rồi fit lại cây với số cây cố định.
- [ ] Dự đoán V/test bằng `sigmoid(z_GLM + tree_raw_margin)`; không cộng hai xác suất.
- [ ] Kiểm tra trường hợp cây đóng góp bằng 0 phải trả đúng xác suất GLM.
- [ ] Lưu cả GLM, preprocessing và cây trong cùng model bundle; load lại phải tái tạo dự đoán.
- [ ] XGBoost với `base_margin` là biến thể sau, nếu LightGBM có cơ sở tốt; phải cấp margin nhất quán khi dự đoán.

LightGBM cần cộng initial score vào dự đoán raw rồi mới chuyển sang xác suất. [Giải thích từ nhóm LightGBM](https://github.com/lightgbm-org/LightGBM/issues/5219)

**Đầu ra:** mô hình GLM-boost và đối chứng GLM/cây đứng riêng.  
**Điều kiện xong:** cải thiện có thể xác nhận, hoặc có lỗi khác giúp stack. Không mặc định nó là mô hình đơn mạnh nhất trước khi chạy.

### Bước 13 — Seed bagging và mô hình khác họ

**Nhánh chính:**

- [ ] Chỉ bag 3 seed cho một hoặc hai cấu hình tốt nhất, giữ outer fold cố định.
- [ ] Thay seed encoder và seed cây theo các trường config riêng để biết đang đo nguồn biến thiên nào.
- [ ] So trung bình xác suất và trung bình hạng theo lựa chọn đã giới hạn trước.
- [ ] Với TE có cùng T và m, bảng transform V/test từ toàn T không tự thay đổi chỉ vì đổi inner seed; khác biệt đến từ feature train và mô hình fit lại.

**Nhánh GPU tùy chọn:**

- [ ] Sau khi GLM/cây/stack cơ bản ổn định mới thử một MLP tự huấn luyện từ đầu, có raw numeric, categorical/embeddings và/hoặc TE.
- [ ] Chọn epoch bằng tập nội bộ hoặc cố định trước; không chọn epoch dựa trên outer V.
- [ ] Đo thời gian một fold và đóng góp vào stack trước khi mở thêm kiến trúc.
- [ ] Các thông số RealMLP trong guide là cấu hình tham khảo, phải đối chiếu API và môi trường lúc triển khai.

**Nhánh nghiên cứu, không bắt buộc để hoàn thành project:** Target-First LM tự huấn luyện từ đầu. Chỉ thực hiện nếu còn ngân sách sau phần báo cáo; tokenizer/weights không lấy từ mô hình huấn luyện sẵn nếu giữ đúng phạm vi. Khi suy luận phải đánh giá hai giả thuyết nhãn, không được đưa nhãn thật của V/test vào input. Cách tính likelihood và prior phải được kiểm tra riêng trước khi tin AUC.

**Điều kiện xong:** mô hình mới phải giúp kết quả hoặc giúp kiểm chứng giả thuyết; không cần 8–15 thành viên mới coi là project hoàn chỉnh.

## 9. Ensemble, xác nhận và quyết định cuối

### Bước 14 — So sánh các cách kết hợp rồi học stack

**Đối chứng bắt buộc:** best single, trung bình xác suất, trung bình hạng, Logistic stack.

**Đầu việc:**

- [ ] Registry chỉ nhận OOF/test đủ kích thước, đúng ID/nhãn/fold; chặn smoke, file trùng và TE rò rỉ.
- [ ] Thử stack từ các mô hình gốc; không đưa output của một stack làm input cho stack khác.
- [ ] Giữ năm họ mô hình hiện có trong bảng đối chứng; thêm lần lượt GLM TE, cây TE/digit và GLM-boost.
- [ ] Với stack mặc định, clip xác suất, lấy logit và StandardScaler trong từng meta train fold.
- [ ] Fit Logistic Regression với L2; không đặt trước phần trăm cho từng mô hình.
- [ ] Chấm meta CV bằng hai phép chia seed 7 và 8 để kiểm tra độ ổn định của tầng hai.
- [ ] Thử logit-rank như một ablation riêng sau stack logit. Với chế độ đánh giá có thể áp dụng cho dữ liệu mới, fit empirical CDF của từng cột dự đoán trên meta train, áp dụng cho meta validation/test rồi clip và logit; không fit scaler/CDF bằng nhãn validation.
- [ ] Chỉ quét `C={0.01,0.1,1}` sau khi tập thành viên tương đối ổn định.
- [ ] Leave-one-out từng thành viên; một hệ số âm nhỏ không đủ để kết luận mô hình vô ích hay hữu ích.

Hệ số Logistic trên feature đã logit/chuẩn hóa không phải trọng số phần trăm. Mô hình này học theo log-loss, còn việc chọn ensemble dùng ROC AUC.

**Đầu ra:** bảng ensemble, hệ số, leave-one-out và hai bộ meta CV.  
**Điều kiện xong:** ensemble có lợi so với đối chứng hoặc chọn mô hình đơn giản hơn nếu chênh lệch không rõ.

### Bước 15 — Xác nhận toàn pipeline và đo bất định

CV của meta model trên OOF đã lưu **không phải** nested CV đầy đủ. Một số mô hình đã tạo feature OOF cho meta train có thể từng học nhãn thuộc meta validation. Đổi seed meta hoặc dùng chung base fold không tự giải quyết hết phụ thuộc này.

**Quy trình xác nhận ensemble đã chọn:**

1. Khóa phạm vi cấu hình và thành viên trước khi xem kết quả xác nhận.
2. Trong mỗi outer T, tạo base OOF nội bộ chỉ từ T để fit meta.
3. Mỗi base inner fit tiếp tục làm TE cross-fit và early stopping đúng ranh giới.
4. Nếu còn chọn C hoặc cấu hình, việc chọn đó diễn ra trong T, không dùng outer V.
5. Fit lại base model trên T, tạo feature cho V rồi dự đoán bằng meta học trong T.
6. Gộp dự đoán outer V để báo điểm xác nhận của cả hệ thống.

Với 3 thành viên, 5 outer và 5 inner cho OOF tầng meta, riêng phần base cần khoảng `5 × (5+1) × 3 = 90` lượt fit, chưa tính chọn số cây và chọn cấu hình. Vì vậy chỉ xác nhận sâu tập thành viên đã chốt. Việc đã chọn họ mô hình qua nhiều thí nghiệm trên cùng dataset vẫn phải được công bố; bước này không làm dữ liệu cũ thành một test set độc lập chưa từng xem.

Lựa chọn tham số và chấm trên cùng CV có thể tạo điểm lạc quan. [Ví dụ nested/non-nested CV của scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

**Quy tắc sàng lọc và kiểm tra:**

- [ ] Đối chiếu delta AUC gộp và cả 5 delta theo fold trên cùng tập dòng.
- [ ] Ngưỡng sàng lọc ban đầu: delta OOF AUC ít nhất 0.0001 và có lợi ở ít nhất 4/5 fold; đây là bộ lọc thực dụng, không phải kiểm định thống kê.
- [ ] Ứng viên tăng nhỏ hoặc thêm độ phức tạp lớn cần thêm cách chia xác nhận toàn pipeline hoặc bootstrap ghép cặp.
- [ ] Bootstrap khởi đầu 500 lần, lấy cùng chỉ số cho hai bộ dự đoán, phân tầng hai lớp; báo phân phối delta và khoảng percentile 95%.
- [ ] Ghi rõ bootstrap trên prediction cố định không bao gồm toàn bộ bất định do huấn luyện lại hay chọn mô hình nhiều lần.
- [ ] Nếu đang chọn các thành viên tăng rất nhỏ, thử null test: mô hình cũ cộng nhiễu, đi qua cùng bước lựa chọn; xem quy trình có giữ thành viên không thêm thông tin hay không.
- [ ] Không dùng độ lệch chuẩn AUC giữa fold như sai số chuẩn của delta giữa hai mô hình.

**Điều kiện xong:** có một lựa chọn cuối và lý do chọn được hỗ trợ bằng dữ liệu; không cần mọi thí nghiệm đều tăng điểm.

### Bước 16 — Refit cho test và chọn chế độ suy luận

**Phương án mặc định:** trung bình dự đoán từ 5 outer model đã refit trên toàn T; meta cuối học trên toàn bộ base OOF. Cách này giữ cấu trúc gần pipeline đã đánh giá.

**Phương án full-train refit, thử riêng nếu cần:**

- [ ] Chọn số cây bằng quy trình nội bộ; dùng thống kê best iteration như median hoặc mean đã chốt trước.
- [ ] So hệ số số cây 1.0 và 1.25 bằng thí nghiệm mô phỏng refit trên phần train của validation, trước khi quyết định cho test.
- [ ] Tạo TE cross-fit cho các dòng train ngay cả khi fit trên 100% train; không encode train bằng nhãn của chính từng dòng.
- [ ] Transform test bằng bảng TE fit từ toàn train.
- [ ] Fit lại toàn bộ preprocessing, GLM/base model và lưu bundle đầy đủ.
- [ ] Đo khác biệt phân phối và tương quan giữa test prediction refit và fold average; không chọn cách refit chỉ vì public score cao hơn sau nhiều lần thử.
- [ ] Kiểm tra tầng meta còn phù hợp khi base test chuyển từ fold average sang một full-train model; không đổi ngầm nguồn feature test sau khi đã chấm stack.

**Đầu ra:** `final/selection.json` ghi rõ nguồn OOF, cách refit, model IDs, transform và cách tạo test prediction.  
**Điều kiện xong:** load lại bundle và dự đoán tái tạo được file nộp trong dung sai đã ghi.

## 10. Giải thích kết quả và hoàn thành project

### Bước 17 — Phân tích lỗi và ý nghĩa thực tế

**Đầu việc:**

- [ ] Chỉ dùng dự đoán OOF/outer xác nhận để phân tích lỗi; không trình bày lỗi trên train như lỗi tổng quát hóa.
- [ ] Xem fraud bị xếp thấp và giao dịch hợp lệ bị xếp cao theo nhóm no-flag, auth, CVV, merchant category, tuổi và giá trị hiếm/unseen.
- [ ] So best single và stack trên cùng nhóm để biết ensemble sửa lỗi nào.
- [ ] Báo recall/precision tại top 0.5%, 1%, 2% điểm cao nhất như mô phỏng ngân sách kiểm tra; ghi rõ đây là phân tích offline.
- [ ] Vẽ PR curve; nếu bàn về xác suất, thêm calibration curve, Brier score hoặc log-loss.
- [ ] Không chọn ngưỡng để tối đa AUC: ROC AUC không dùng một ngưỡng quyết định cố định. Nếu cần ngưỡng, chọn theo mục tiêu riêng trên train/inner validation.
- [ ] Permutation importance theo nhóm raw/TE/digit/pair; dùng SHAP trên mẫu nếu thật sự cần giải thích cây.
- [ ] Nêu rõ feature importance không chứng minh quan hệ nhân quả; chữ số synthetic không mặc định có ý nghĩa nghiệp vụ.

**Đầu ra:** 3–5 nhóm lỗi có số liệu và đề xuất cụ thể, thay vì chỉ danh sách feature importance.

### Bước 18 — Kiểm tra submission và lưu mốc Kaggle

**Đầu việc:**

- [ ] File đúng hai cột `id,is_fraud` và đủ 299.984 dòng cho phiên bản dữ liệu hiện tại.
- [ ] ID duy nhất, cùng tập và thứ tự với sample; không NaN/inf; điểm trong miền phù hợp.
- [ ] Ghi rõ output là xác suất Logistic hay điểm hạng; điểm hạng 0–1 chưa phải xác suất được hiệu chỉnh.
- [ ] Gắn SHA256 và run ID với file submission; không nhầm `submission.csv` từ run cũ ở thư mục gốc.
- [ ] Chỉ nộp các mốc có lý do: baseline mạnh, TE/digit tốt nhất, ensemble cuối.
- [ ] Ghi public score sau khi nộp như một quan sát bổ sung; không báo private score khi chưa có.
- [ ] Không tự động gửi submission; script tạo file, log và đường dẫn đầu ra để người làm dự án nộp.

**Điều kiện xong:** người khác có thể truy ngược file nộp về dữ liệu, code, cấu hình và phép chia fold.

### Bước 19 — Báo cáo môn học và bàn giao

Đề cương báo cáo:

1. Bài toán, dữ liệu và phạm vi tự huấn luyện.
2. Metric, mất cân bằng lớp và thiết kế validation.
3. EDA với các giả thuyết dẫn tới đặc trưng.
4. Baseline và bias–variance/learning curve.
5. TE và ví dụ rò rỉ, kèm cách kiểm tra.
6. Ablation raw TE, digit, pair, count và độ phức tạp cây.
7. GLM/cây/GLM-boost; các nhánh không hiệu quả.
8. Ensemble học trọng số, đóng góp từng thành viên và xác nhận.
9. Phân tích lỗi, giới hạn dữ liệu synthetic và sai số đánh giá.
10. Kết luận, cách chạy lại và hướng tiếp theo.

**Bảng/hình tối thiểu:**

- [ ] Bảng dữ liệu và tỷ lệ lớp.
- [ ] 5–8 phát hiện EDA có số mẫu.
- [ ] Bảng baseline với metric và thời gian.
- [ ] Đường cong độ phức tạp và learning curve.
- [ ] Sơ đồ ranh giới outer/inner/early stop/TE.
- [ ] Bảng ablation theo nhóm đặc trưng.
- [ ] Bảng ensemble, hệ số và leave-one-out.
- [ ] Phân phối bootstrap delta và phân tích lỗi.
- [ ] Nhật ký thí nghiệm thành công lẫn thất bại.

**Bàn giao:** README có một lệnh Kaggle; môi trường đã kiểm tra; run cuối cùng; submission; model bundles; hướng dẫn đọc metric. Chạy lại một lượt smoke và kiểm tra lại manifest/bundle trước khi kết thúc.

## 11. Thứ tự thực hiện và ngân sách

| Chặng | Bước | Ưu tiên | Kết quả phải có trước khi tiếp tục |
|---|---|---|---|
| A | 0–4 | Bắt buộc | Folder, dữ liệu, CV, runner và EDA rõ ràng |
| B | 5–6 | Bắt buộc | Baseline và cấu hình cây đối chứng |
| C | 7–8 | Ưu tiên cải thiện cao nhất | TE đã kiểm tra và GLM/cây TE |
| D | 9–10 | Thí nghiệm có kiểm soát | Đóng góp digit/pair/count được đo riêng |
| E | 11–12 | Sau khi đặc trưng ổn | Tuning giới hạn và GLM-boost |
| F | 13–15 | Chỉ mở rộng ứng viên tốt | Bagging, stack và xác nhận cuối |
| G | 16–19 | Bắt buộc để bàn giao | Refit, lỗi, submission và báo cáo |

### 11.1. Vòng chạy đầu tiên

1. Kiểm tra runner bằng smoke.
2. Tái lập baseline cần thiết để có đối chứng đúng pipeline mới.
3. Chốt cây raw nhỏ và chạy `T01 → T02 → T03`.
4. Chỉ sau đó mới chạy T04, digit hoặc pair.
5. Lấy thời gian một fold thực tế để ước lượng toàn run; không ấn định giờ CPU/GPU từ số của người khác.

Một thí nghiệm 5 fold có 5 lần fit cuối. Nếu mỗi fold cây cần chọn số cây rồi refit, thường có 10 lần fit cây. TE inner fold còn có chi phí tính thống kê, nhưng không mặc định thêm 5 lần fit cây.

### 11.2. Cách kiểm soát tài nguyên

- Không nhân đồng thời mọi tổ hợp model × feature group × m × C × seed.
- Giai đoạn đầu một seed; chỉ bag cấu hình tốt nhất.
- CPU cho GLM/GBDT; GPU chỉ bật khi mô hình cụ thể cần và có benchmark.
- Cache phải có khóa dữ liệu/phép chia đầy đủ; resume chỉ dùng lại artifact đã hoàn thành và đúng cấu hình.
- Lưu model bundle của ứng viên quan trọng; các trial khác tối thiểu có OOF/test/cấu hình/log.
- Nếu một nhánh không cải thiện sau ngân sách đã đặt, ghi kết luận và chuyển sang báo cáo; không cần làm LM để đạt tiêu chuẩn project.

## 12. Nhật ký và định nghĩa hoàn thành

Mỗi thí nghiệm ghi tối thiểu:

```text
experiment_id, run_id, hypothesis, parent_experiment,
data_hash, code_version, fold_hash, feature_groups,
model, params, encoder_seed, model_seed, early_stop_seed,
oof_auc, fold_0_auc, ..., fold_4_auc, mean_auc, std_auc,
delta_vs_parent, train_auc, logloss, fit_seconds,
status, keep_or_drop, reason
```

Không điền kết quả tham khảo vào các cột metric của thí nghiệm chưa chạy. Mỗi đoạn kết luận trong báo cáo cần trỏ về bảng/run cụ thể.

Project hoàn thành khi:

- [ ] Có một folder với dữ liệu và code chạy được trên Kaggle bằng một lệnh.
- [ ] Có quy trình CV và kiểm tra rò rỉ đã thực hiện, không chỉ mô tả.
- [ ] Có baseline và bằng chứng về ảnh hưởng của ít nhất các nhóm raw/TE/digit hoặc pair.
- [ ] Có ít nhất một mô hình tuyến tính và một mô hình cây được huấn luyện, so sánh và giải thích.
- [ ] Có ensemble học từ OOF, đối chứng đơn giản và ghi rõ giới hạn của đánh giá.
- [ ] Có cấu hình cuối, bundle suy luận, submission truy xuất được nguồn gốc.
- [ ] Có phân tích lỗi, nhật ký các thử nghiệm không giúp và báo cáo tái lập được.

**Việc bắt đầu khi chuyển từ kế hoạch sang triển khai:** thực hiện Bước 0–4 để tạo nền tảng sạch và EDA dẫn tới giả thuyết, sau đó tái lập baseline và ưu tiên Logistic Regression với target encoding. Các mô hình phức tạp hơn chỉ được xây trên nền đánh giá đã qua kiểm tra.
