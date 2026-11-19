# Tiến độ dự án S6E11

*Cập nhật: 18/11/2026 — đã xong bước 0–7; encoder bước 7 đã chạy full-data trên local, tiếp theo là mô hình TE bước 8.*

## Nguyên tắc

Ưu tiên **thực nghiệm chạy ổn, kết quả đúng, không leakage và nâng cấp có lý do**.
Giữ đối chứng/config chính nhất quán. Không quá khắt khe về môi trường, hash/newline
hoặc chi tiết phụ khi chúng không ảnh hưởng kết quả; không chạy lại phần đã hợp lệ.

## Trạng thái

| Phần | Trạng thái | Kết luận giữ lại |
|---|---|---|
| Bước 0–4: dữ liệu, CV, runner, EDA | Xong | Dữ liệu tập trung, 5 fold cố định, có giả thuyết EDA |
| Bước 5: B00–B07 | Xong | Best baseline: B06 XGBoost, OOF AUC **0.946965** |
| Bước 6: cây nhỏ và learning curve | **Xong, đã kiểm tra output** | Chọn **LightGBM 4 lá + XGBoost depth 2** làm đối chứng cho TE |
| Bước 7: encoder | **Xong, chạy full local** | Gate rò rỉ đạt; artifact tại `Step7/20261118T134537Z_0364fc/` |
| Bước 8: mô hình TE | **Code và smoke sẵn** | Dùng cache Step7; chờ full-model T01–T03 |

## Kết quả bước 6

Run đầy đủ: [20261118T091556Z_367a2c](../Step6/20261118T091556Z_367a2c).
Metric chính là **ROC AUC trên OOF gộp**.

| Cấu hình | OOF AUC | Delta so control cùng họ | Fold tăng | Train–OOF gap | Tổng thời gian các fold |
|---|---:|---:|---:|---:|---:|
| LightGBM 15 lá — control | 0.946350 | — | — | 0.01317 | 20,9 phút |
| **LightGBM 4 lá** | **0.947365** | **+0.001015** | **5/5** | **0.00545** | 39,7 phút |
| LightGBM 8 lá | 0.946754 | +0.000404 | 4/5 | 0.00893 | 26,0 phút |
| LightGBM 31 lá | 0.946221 | −0.000129 | 2/5 | 0.01749 | 15,9 phút |
| XGBoost depth 3 — control | 0.946965 | — | — | 0.00780 | 27,9 phút |
| **XGBoost depth 2** | **0.947317** | **+0.000352** | **4/5** | **0.00453** | 38,6 phút |

Thời gian gồm chọn cây, refit và các thao tác dự đoán/kiểm tra trong fold.
Chi tiết: [comparison.csv](../Step6/20261118T091556Z_367a2c/complexity/comparison.csv).

### Kết luận và config được chọn

- **Cây nhỏ tốt hơn trong thí nghiệm này:** AUC tăng và train–OOF gap giảm, phù hợp
  với giả thuyết giảm overfit. LightGBM 4 lá tăng ở cả 5 fold, XGBoost depth 2 ở 4/5.
- Hai cấu hình thắng chỉ cách nhau khoảng **0.000049 AUC**: giữ cả hai cho TE;
  chưa kết luận LightGBM chắc chắn tốt hơn trên test.
- Cây nhỏ cần nhiều vòng boosting hơn nên chậm hơn. Không cấu hình nào chạm trần
  6.000 cây; chưa cần tăng trần hoặc chạy lại bước 6.

| Tham số giữ cho bước tiếp theo | LightGBM | XGBoost |
|---|---|---|
| Độ phức tạp | `num_leaves=4` | `max_depth=2` |
| Learning rate | 0.03 | 0.03 |
| Regularization | `min_child_samples=50, reg_lambda=1` | `min_child_weight=5, reg_lambda=1` |
| Lấy mẫu | `subsample=0.9, subsample_freq=1, colsample_bytree=0.9` | `subsample=0.9, colsample_bytree=0.9` |
| Chọn số cây | AUC trên E, patience 200, tối đa 6.000 | AUC trên E, patience 200, tối đa 6.000 |
| Cây thực tế trong 5 fold | 2.177–3.407; median 2.734 | 2.044–4.129; median 3.137 |

Số cây thực tế là kết quả early stopping, không phải số cố định cho mọi fold.
Đầu vào vẫn là raw numeric + 6 cờ + one-hot categorical, **chưa có TE**.

## Learning curve: dữ liệu còn có ích

LightGBM 4 lá, **chỉ fold 0**, validation cố định 139.992 dòng:

| Phần outer train sử dụng | Số dòng train | AUC validation |
|---|---:|---:|
| 10% | 55.996 | 0.940543 |
| 25% | 139.992 | 0.942791 |
| 50% | 279.984 | 0.944187 |
| 100% | 559.968 | 0.945818 |

AUC tăng theo lượng dữ liệu, chưa thấy bão hòa rõ trên fold này. Giữ refit trên toàn
bộ phần train của mỗi fold. Đây là chẩn đoán một fold, không phải AUC 5-fold.

[Bảng learning curve](../Step6/20261118T091556Z_367a2c/complexity/learning_curve/learning_curve.csv)
· [Hình kết quả](../Step6/20261118T091556Z_367a2c/complexity/complexity_learning_curve.png).

## Kiểm tra output: đạt, có thể đi tiếp

- Run completed, full-data, không phải smoke; fold và early-stop split giống bước 5.
- Trong cùng họ, các config chỉ khác số lá/depth; đối chứng phù hợp.
- Cả 6 cấu hình có OOF đủ 699.960 dòng, test đủ 299.984 dòng, đúng ID/nhãn/fold;
  dự đoán hữu hạn trong `[0,1]`; trung bình test theo 5 fold đúng.
- AUC từng fold, AUC gộp, log-loss và các delta tính lại khớp artifact; đủ 5 fold completed.
- Learning curve giữ V cố định, A/E phân hoạch phần train được lấy mẫu;
  điểm 100% khớp AUC fold 0 của S6-L04.
- Luồng fit trong source snapshot chọn cây trên A/E, refit T rồi chấm V;
  không thấy dùng nhãn outer V để chọn số cây trong luồng này.

Bằng chứng chi tiết: [step6_review_checks.json](step6_review_checks.json).
**Không có vấn đề cần chạy lại bước 0–6.** Không mở lại các việc phụ đã qua để chặn tiến độ.

## Bước 7 — Target encoding full-data local

Code: `src/target_encoding.py`, `src/encoding.py`.
Run full: [20261118T134537Z_0364fc](../Step7/20261118T134537Z_0364fc), local data,
`smoke=false`, `status=completed`. Smoke trước đó: `outputs/20261118T133334Z_aebe53_smoke/`.

Cấu hình chốt: 13 cột số, `m=20`, `key_scale=100`, 5 outer fold, mỗi T/A có 5 inner fold
phân tầng cùng `encoder_seed=2026`. Mỗi fold tạo 559.968 dòng `T_crossfit`, 139.992
`V_from_T`, 299.984 `test_from_T`, 503.971 `A_crossfit`, 55.997 `E_from_A`.

| Kiểm tra | Kết quả |
|---|---|
| Công thức m-estimate, unseen về prior | Đúng; `m=20`, 13 cột số, seed 2026 |
| Cross-fit trong T và A | 5 fold × 2 scope × 13 cột, inner holdout không vào prior/tổng nhãn của chính nó |
| V/test từ T, E từ A | Đúng; V/test lấy bảng toàn T, E lấy bảng toàn A |
| Tính lại độc lập từ CSV gốc | 806.000 giá trị TE ở smoke khớp, sai số lớn nhất 0 |
| Đổi nhãn outer V | Feature và dự đoán kiểm tra không đổi, max diff = 0 |
| Shuffle nhãn, dựng lại encoder + model | AUC **0.500843** full (smoke 0.517073), nằm trong sanity gate `[0.35,0.65]` |
| Leak demo | Clean 0.875117 / leaky 0.913150, `valid_registry=false` |
| Unit test | **42/42 qua** |

Không phát hiện leakage trong phạm vi đã kiểm tra. Leak demo chỉ để giải thích, bị loại
khỏi registry; **không phải kết quả GLM + TE bước 8**. Full run không tạo submission.

Bằng chứng: [checks_summary.json](../Step7/20261118T134537Z_0364fc/encoding/checks_summary.json),
[summary.json](../Step7/20261118T134537Z_0364fc/encoding/summary.json),
[step7_review_checks.json](step7_review_checks.json). Môi trường local khác lock Kaggle
(NumPy 2.5.4/pandas 3.1.1/sklearn 1.7.2 so với 2.1.3/2.3.3/1.6.1); ghi nhận khi
đối chiếu kết quả. Không cần chạy lại encoding chỉ vì môi trường khác: cache có thể
dùng lại khi dữ liệu, fold và tham số encode khớp.

Hai lưu ý khi mở rộng, không chặn tiến độ:

- Guard runtime inner fold chủ yếu kiểm tra assignment/prior; đối chiếu giá trị TE với
  công thức mới chỉ chạy độc lập trên smoke. Khoảng shuffle `[0.35,0.65]` là sanity gate
  rộng, không thay thế phân tích leakage đầy đủ của từng mô hình bước 8.
- Encoder hiện dành cho 13 cột số. Chưa dùng thẳng cho cột categorical dạng chuỗi:
  `integer_keys` đang coerce chuỗi không phải số thành missing. Chỉ bổ sung xử lý
  khóa categorical khi thực hiện nhánh TE 24 cột.

## Việc tiếp theo: bước 8

**Code đã chuẩn bị:** `--stage te_models`, mặc định T01/T02/T03. Đã đọc lại cache
full Step7 đủ 5 fold và chạy smoke cả ba mô hình. Bốn test mới qua; OOF/test smoke
đúng kích thước, metric khớp và bundle dự đoán từ raw khớp đường dùng TE cache
(max diff = 0). Chưa huấn luyện full-data phase 8 và chưa kết luận TE tăng điểm.
Bằng chứng: [phase8_smoke_checks.json](phase8_smoke_checks.json).

```powershell
python .\s6e11_project_run.py --stage te_models --threads 4
```

Tự tìm cache `Step7/20261118T134537Z_0364fc/encoding/`; muốn GLM trước thì thêm
`--te-models T01`. Đầu ra `outputs/<run_id>/te_models/comparison.csv` sẽ so trực tiếp
với B02 / S6-L04 / S6-X02, không chạy lại các đối chứng này.

1. **GLM + TE trước:** giữ raw/log/one-hot và config như B02, chỉ thêm logit TE
   13 cột số để đo riêng đóng góp encoding; scaler học trong train tương ứng.
2. Giữ encoder đã kiểm tra: `m=20`, inner 5 fold, seed 2026; V/test không tham gia
   học nhãn. Khi chọn cây, tách A/E trước rồi mới fit encoder trong A.
3. **Hai cây + TE sau:** LightGBM 4 lá + TE so với S6-L04; XGBoost depth 2 + TE
   so với S6-X02. Giữ các config còn lại, fold và cách early stopping.
4. Chạy đủ 5 outer fold rồi ghi OOF AUC, delta từng fold và kết luận giữ/bỏ.
   Chưa có cơ sở nói TE tăng điểm trước khi có thí nghiệm này.
5. Digit/pair/GLM-boost để các bước sau; chưa thêm cùng lúc vào nhánh TE đầu tiên.

Nhật ký/review dài của giai đoạn trước đã chuyển sang
[legacy/docs](../legacy/docs/steps_progress_before_step6_review.md).
