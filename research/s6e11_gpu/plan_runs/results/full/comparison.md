# S6E11 · Kết quả và tiến độ thí nghiệm GPU

Đã xác minh 37 lượt chạy đầy đủ. ROC AUC OOF cao nhất trong các lượt đã xác minh là 0.950345 (`FINAL`).

ROC AUC đo khả năng xếp hạng giao dịch gian lận. OOF là dự đoán cho một giao dịch từ mô hình huấn luyện trên các phần còn lại. AP tóm tắt đường precision–recall. Δ AUC so với đối chứng được ghi cho từng lượt. Cột giây lấy tổng thời gian ghi theo fold, có thể gồm tiền xử lý và suy luận.

Bảng được tính lại từ các tệp `predictions.npz` và đối chiếu với `metrics.json`, `config.json` khi có, cùng `comparison.csv`. Kết quả smoke và kiểm tra mô phỏng được dùng để kiểm tra mã và nằm ngoài bảng điểm đầy đủ.

Phạm vi đối chiếu là Bước 7–19, dòng 298–598. [Kế hoạch dự án](../../../../../plan.md#L298)

## Bảng kết quả

| Mã | Đối chứng | ROC AUC OOF | Δ AUC | AP | Log-loss | Giây | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FINAL` | `XGB-D03` | 0.950345 | +0.000163 | 0.529376 | 0.044347 | — | 0.949878 | 0.951163 | 0.955174 | 0.951022 | 0.944769 |
| `ALL3-MEAN` | `XGB-D03` | 0.950278 | +0.000095 | 0.529063 | 0.044400 | — | 0.949636 | 0.951153 | 0.955191 | 0.951014 | 0.944623 |
| `TREES2-MEAN` | `XGB-D03` | 0.950276 | +0.000094 | 0.528939 | 0.044406 | — | 0.949860 | 0.951070 | 0.955019 | 0.951198 | 0.944514 |
| `ALL3-RANK` | `XGB-D03` | 0.950276 | +0.000093 | 0.528497 | 0.926446 | — | 0.949657 | 0.951243 | 0.955181 | 0.951008 | 0.944529 |
| `SEED3-MEAN` | `XGB-D03` | 0.950269 | +0.000087 | 0.528975 | 0.044409 | — | 0.949866 | 0.950974 | 0.955095 | 0.951069 | 0.944623 |
| `TREES2-RANK` | `XGB-D03` | 0.950267 | +0.000085 | 0.528893 | 0.931237 | — | 0.949845 | 0.951051 | 0.955004 | 0.951204 | 0.944512 |
| `SEED3-RANK` | `XGB-D03` | 0.950263 | +0.000081 | 0.528939 | 0.931015 | — | 0.949846 | 0.950968 | 0.955090 | 0.951069 | 0.944623 |
| `XGB-D03` | `T03` | 0.950182 | +0.001688 | 0.528763 | 0.044422 | 231.4 | 0.949852 | 0.950884 | 0.954957 | 0.950990 | 0.944603 |
| `XGB-D03-R125` | `XGB-D03` | 0.950146 | -0.000036 | 0.529329 | 0.044384 | 346.8 | 0.949779 | 0.950968 | 0.954931 | 0.950869 | 0.944525 |
| `D03` | `T04` | 0.950131 | +0.001557 | 0.527604 | 0.044475 | 871.3 | 0.949598 | 0.951024 | 0.954852 | 0.951197 | 0.944163 |
| `D02` | `T04` | 0.950124 | +0.001550 | 0.528206 | 0.044435 | 1115.7 | 0.949649 | 0.951012 | 0.954810 | 0.951146 | 0.944175 |
| `TUNE-NOSUB` | `XGB-D03` | 0.950104 | -0.000079 | 0.528598 | 0.044444 | 401.8 | 0.949684 | 0.950965 | 0.954947 | 0.950777 | 0.944443 |
| `XGB-D03-S93` | `XGB-D03` | 0.950091 | -0.000091 | 0.527930 | 0.044470 | 338.1 | 0.949597 | 0.950775 | 0.955045 | 0.950776 | 0.944449 |
| `XGB-D02` | `T03` | 0.950074 | +0.001580 | 0.528382 | 0.044444 | 217.4 | 0.949728 | 0.950997 | 0.954809 | 0.950887 | 0.944387 |
| `XGB-D03-S71` | `XGB-D03` | 0.950074 | -0.000108 | 0.527848 | 0.044479 | 385.2 | 0.949702 | 0.950915 | 0.954834 | 0.950920 | 0.944399 |
| `TUNE-ETA005` | `XGB-D03` | 0.950060 | -0.000122 | 0.528456 | 0.044458 | 279.8 | 0.949692 | 0.950717 | 0.955062 | 0.950767 | 0.944679 |
| `TUNE-LAMBDA3` | `XGB-D03` | 0.950045 | -0.000137 | 0.528513 | 0.044450 | 360.8 | 0.949733 | 0.950859 | 0.954945 | 0.950742 | 0.944433 |
| `D01` | `T04` | 0.948784 | +0.000210 | 0.521507 | 0.044890 | 898.4 | 0.947576 | 0.949905 | 0.954315 | 0.949367 | 0.943274 |
| `XGB-D01` | `T03` | 0.948623 | +0.000129 | 0.521631 | 0.044920 | 157.2 | 0.947424 | 0.949972 | 0.953856 | 0.949065 | 0.943195 |
| `GLM-BOOST-LGB` | `T02` | 0.948623 | +0.000097 | 0.522113 | 0.044872 | 396.8 | 0.947683 | 0.950164 | 0.953800 | 0.948633 | 0.943055 |
| `T04` | `T02` | 0.948574 | +0.000047 | 0.520055 | 0.044973 | 487.1 | 0.947379 | 0.950095 | 0.953644 | 0.949273 | 0.942742 |
| `GLM-BOOST-XGB` | `T03` | 0.948544 | +0.000050 | 0.521212 | 0.044924 | 153.6 | 0.947633 | 0.950265 | 0.953487 | 0.948543 | 0.942984 |
| `DERIVED` | `T04` | 0.948542 | -0.000032 | 0.520315 | 0.044972 | 809.5 | 0.947170 | 0.950143 | 0.953734 | 0.949127 | 0.942875 |
| `T02` | `GPU-S6-L04` | 0.948527 | +0.001295 | 0.518542 | 0.045057 | 406.6 | 0.947342 | 0.949910 | 0.953597 | 0.949303 | 0.942775 |
| `T03` | `GPU-S6-X02` | 0.948495 | +0.001356 | 0.521020 | 0.044963 | 138.3 | 0.947440 | 0.949946 | 0.953561 | 0.948919 | 0.942902 |
| `GLM-NEWTON` | `P1-P2` | 0.947656 | +0.000075 | 0.515512 | 0.045275 | 44.2 | 0.946464 | 0.949029 | 0.953155 | 0.947935 | 0.941843 |
| `GLM-C0.3` | `P1-P2` | 0.947605 | +0.000024 | 0.515563 | 0.045281 | 42.5 | 0.946500 | 0.948953 | 0.953050 | 0.947871 | 0.941740 |
| `P1-P2` | `P1` | 0.947581 | +0.000097 | 0.515491 | 0.045286 | 40.6 | 0.946502 | 0.948846 | 0.953047 | 0.947864 | 0.941744 |
| `GLM-C0.1` | `P1-P2` | 0.947558 | -0.000023 | 0.515294 | 0.045300 | 50.9 | 0.946486 | 0.948825 | 0.953019 | 0.947905 | 0.941690 |
| `P1` | `T01` | 0.947484 | +0.000223 | 0.515070 | 0.045314 | 26.3 | 0.946547 | 0.949016 | 0.952914 | 0.947636 | 0.941467 |
| `P1-P2-P3` | `P1-P2` | 0.947385 | -0.000196 | 0.514208 | 0.045363 | 42.6 | 0.946229 | 0.948721 | 0.952837 | 0.947988 | 0.941226 |
| `T01` | `GPU-B02` | 0.947261 | +0.005371 | 0.513583 | 0.045406 | 19.4 | 0.946496 | 0.948613 | 0.952620 | 0.947446 | 0.941290 |
| `GPU-S6-L04` | — | 0.947232 | — | 0.513262 | 0.045409 | 591.6 | 0.945810 | 0.948485 | 0.951833 | 0.947570 | 0.942810 |
| `T01-M50` | `T01` | 0.947208 | -0.000053 | 0.513007 | 0.045436 | 20.9 | 0.946282 | 0.948593 | 0.952604 | 0.947335 | 0.941375 |
| `GPU-S6-X02` | — | 0.947139 | — | 0.513930 | 0.045408 | 148.9 | 0.945917 | 0.948347 | 0.951603 | 0.947423 | 0.942902 |
| `COUNT` | `T04` | 0.946825 | -0.001749 | 0.505820 | 0.046290 | 702.9 | 0.945270 | 0.949068 | 0.951247 | 0.946996 | 0.941942 |
| `GPU-B02` | — | 0.941890 | — | 0.489712 | 0.047038 | 20.6 | 0.941376 | 0.943491 | 0.946932 | 0.941524 | 0.936274 |

## Tiến độ theo hướng

### Mã hóa theo nhãn cho 13 và 24 cột

Bước 7–8 · 7 lượt xác minh. [Notebook](../../notebooks/01-te.ipynb)

T04 chọn họ mô hình theo chính điểm khảo sát này. Kết quả cần được đọc theo từng đối chứng.

### Thử nghiệm khóa chữ số

Bước 9 · 6 lượt xác minh. [Notebook](../../notebooks/02-digit.ipynb)

Trình chạy đo D03 cùng D01 và D02 để so sánh trên cùng fold. Kết quả chữ số gắn với cấu trúc của bộ dữ liệu tổng hợp.

### Cặp tương tác, tần suất và biến dẫn xuất

Bước 10 · 5 lượt xác minh. [Notebook](../../notebooks/03-pairs-count.ipynb)

Chênh lệch của P2 và P3 đo cả nhóm khóa vừa thêm. Cần phép bỏ từng khóa nếu muốn quy kết lợi ích cho một tương tác riêng.

### Điều chỉnh siêu tham số có giới hạn

Bước 11 · 7 lượt xác minh. [Notebook](../../notebooks/04-tuning.ipynb)

Các cấu hình được so trên bộ fold khảo sát. Bảng chỉ mô tả các cấu hình đã chạy.

### GLM làm nền cho cây

Bước 12 · 2 lượt xác minh. [Notebook](../../notebooks/05-glm-boost.ipynb)

Các tệp dự đoán đủ để tính lại AUC. Khả năng nạp lại trọn bộ GLM, tiền xử lý và cây cần kiểm tra bằng model bundle riêng.

### Trung bình dự đoán qua ba seed

Bước 13 · 4 lượt xác minh. [Notebook](../../notebooks/06-seed.ipynb)

S71 đổi seed của cây, S93 đổi seed của bộ mã hóa. MLP là nhánh tùy chọn và không thuộc danh sách thí nghiệm của trình chạy này.

### Ensemble, xác nhận và tệp nộp

Bước 14–19 · 6 lượt xác minh. [Notebook](../../notebooks/07-ensemble-final.ipynb)

Điểm khảo sát và điểm xác nhận dùng hai quy trình khác nhau. Tệp nộp được kiểm tra cục bộ và chỉ có điểm Kaggle khi có kết quả nộp thực tế.

## Hiện vật cuối

Các điểm dưới đây do meta CV trên OOF ghi lại. Đã hiển thị 8 dòng cao nhất khi có đủ dữ liệu. [Bảng khảo sát đầy đủ](ensemble/ensemble_metrics.csv).

| Khảo sát ensemble | Phương pháp | ROC AUC | Δ so mô hình đơn |
|:---|:---|---:|---:|
| `logistic_stack_seed8_c0.1` | `logistic_stack` | 0.950394 | +0.000212 |
| `logistic_stack_seed8_c0.01` | `logistic_stack` | 0.950393 | +0.000211 |
| `rank_logit_stack_seed8_c0.1` | `rank_logit_stack` | 0.950392 | +0.000209 |
| `logistic_stack_seed8_c1` | `logistic_stack` | 0.950390 | +0.000208 |
| `rank_logit_stack_seed8_c0.01` | `rank_logit_stack` | 0.950388 | +0.000206 |
| `rank_logit_stack_seed8_c1` | `rank_logit_stack` | 0.950387 | +0.000204 |
| `logistic_stack_seed7_c0.1` | `logistic_stack` | 0.950386 | +0.000203 |
| `logistic_stack_seed7_c1` | `logistic_stack` | 0.950384 | +0.000202 |

Đã xác minh ROC AUC 0.950345 qua xác nhận lồng trên 5 outer fold. AUC theo fold lần lượt là 0.949878, 0.951163, 0.955174, 0.951022, 0.944769. Các họ mô hình và cấu hình tầng hai được chọn ở giai đoạn khảo sát.

Chọn `NESTED-STACK` với ROC AUC OOF 0.950345. Mô hình đơn đối chứng là `XGB-D03`. Loại điểm `model_probability`. [Hồ sơ chọn phương án cuối](final/selection.json).

Bootstrap ghép cặp theo lớp dùng dự đoán cố định. Khoảng dưới đây đo biến thiên lấy mẫu của bộ dự đoán đã lưu.

| Ứng viên cuối | Δ ROC AUC | Fold tăng | Khoảng bootstrap 95% | Qua sàng lọc |
|:---|---:|---:|:---|:---|
| `NESTED-STACK` | +0.000163 | 5 | [+0.000025, +0.000291] | Có |
| `ALL3-MEAN` | +0.000095 | 4 | [-0.000104, +0.000300] | Không |
| `TREES2-MEAN` | +0.000094 | 4 | [+0.000011, +0.000186] | Không |
| `ALL3-RANK` | +0.000093 | 3 | [-0.000105, +0.000298] | Không |
| `SEED3-MEAN` | +0.000087 | 5 | [+0.000008, +0.000168] | Không |
| `TREES2-RANK` | +0.000085 | 3 | [+0.000002, +0.000179] | Không |
| `SEED3-RANK` | +0.000081 | 4 | [+0.000005, +0.000161] | Không |

`submission.csv` (`s6e11_gpu/plan_runs/results/full/final/submission.csv`) qua kiểm tra cột, 299,984 ID, điểm dự đoán và SHA-256 `4ddddd228ce11ace6e0b5e5468632c2f8df92700b4f26310ce4461fbad53788f`.
