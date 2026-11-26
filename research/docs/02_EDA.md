# Khám phá dữ liệu giao dịch

Tín hiệu nổi bật của dự án là **rủi ro thay đổi ngay giữa các mức số liền kề**, bên cạnh xu hướng theo điểm rủi ro merchant, velocity và số lần nhập lại CVV. Mã hóa chữ số tăng OOF ROC AUC của `XGB-D02` thêm **0,001580** so với `T03`. Về cuối dự án, hiệu chỉnh cho 404 mức tiền thưa cục bộ đưa Grandmaster EDA từ **0,95122445 lên 0,95131585** trên cùng tập giữ lại. Mức tăng này có khoảng tin cậy chứa 0. **0,95131585 là kết quả trên 104.994 dòng train giữ lại.** [Ablation GPU](../../s6e11_gpu/plan_runs/published/full/comparison.md) · [Kết quả amount hole](../goal_0952/deep_eda_measurement/measurement_metrics.json)

## Dữ liệu, nhãn và 24 trường gốc

| Tệp | Số dòng | Số cột và vai trò |
|---|---:|---|
| `train.csv` | 699.960 | 26 cột gồm 24 đặc trưng, `id` và nhãn nhị phân `is_fraud`. Có 11.689 ca dương và 688.271 ca âm, tỷ lệ gian lận khoảng 1,670%. |
| `test.csv` | 299.984 | 25 cột gồm 24 đặc trưng và `id`. |
| `sample_submission.csv` | 299.984 | Hai cột `id,is_fraud` · thứ tự ID khớp test. |

Kiểm tra trên toàn bộ ba CSV ghi nhận 0 ô thiếu, 0 giá trị vô hạn, ID duy nhất và hai tập không giao nhau theo ID. Không có dòng trùng khi so toàn bộ 24 cột trong từng tập hoặc giữa hai tập. `id` là mã dòng phục vụ đối chiếu và xuất tệp. Bộ dữ liệu có thông tin tổng hợp về giao dịch trước, số giao dịch trong 24 giờ và tuổi thẻ. Khóa khách hàng/thẻ và chuỗi giao dịch gốc nằm ngoài phần dữ liệu cung cấp. [Kết quả kiểm cấu trúc và SHA-256](../eda_review/data_audit.json) · [Mã kiểm](../eda_review/data_audit.py) · [Từ điển đặc trưng](../eda_review/tu_dien_24_dac_trung.md)

### 13 trường số

Cột cuối là miền giá trị quan sát trong train. Ba cột `cvv_retry_count`, `txn_count_last_24h` và `prior_disputes` là số đếm rời rạc dù được lưu như trường số. [Miền giá trị và độ chính xác](../eda_review/data_audit.json)

| Trường | Ý nghĩa | Min–max train |
|---|---|---:|
| `amount_usd` | Giá trị giao dịch theo USD | 1–6.872,69 |
| `account_balance_usd` | Số dư tài khoản theo USD | 58,25–99.701,39 |
| `hours_since_last_txn` | Số giờ kể từ giao dịch trước | 0,01–51,98 |
| `txn_count_last_24h` | Số giao dịch trong 24 giờ gần nhất | 0–11 |
| `distance_from_home_km` | Khoảng cách từ địa điểm giao dịch đến nhà khách hàng theo km | 0–57,93 |
| `card_age_months` | Tuổi thẻ theo tháng | 25–73 |
| `customer_age` | Tuổi khách hàng theo năm | 18–81 |
| `cvv_retry_count` | Số lần nhập lại mã bảo mật CVV | 0–3 |
| `velocity_score` | Điểm mức độ dồn dập của hoạt động giao dịch | 0–72,9 |
| `time_of_day_hour` | Giờ thực hiện giao dịch | 0–23 |
| `day_of_week` | Mã ngày trong tuần | 0–6 |
| `merchant_risk_score` | Điểm rủi ro của đơn vị bán hàng | 0–100 |
| `prior_disputes` | Số lần tranh chấp giao dịch trước đây | 0–4 |

### 5 trường phân loại và 6 cờ

| Nhóm | Trường | Ý nghĩa |
|---|---|---|
| Phân loại | `merchant_category` | Ngành hàng, ví dụ `Groceries`, `Fuel`, `Online Retail`. |
| Phân loại | `card_type` | Mạng lưới/thương hiệu thẻ, gồm Visa, Mastercard, Amex, RuPay và Discover. |
| Phân loại | `auth_method` | Phương thức xác thực, gồm 3D Secure, OTP, PIN, sinh trắc học và không xác thực. |
| Phân loại | `channel` | Kênh giao dịch, gồm Online, POS, Contactless, In-App và ATM. |
| Phân loại | `device_type` | Loại thiết bị, ví dụ iPhone, Android Phone, Windows PC và POS Terminal. |
| Cờ | `is_foreign_transaction` | Bật khi giao dịch được đánh dấu là giao dịch nước ngoài. |
| Cờ | `is_new_merchant` | Bật khi đơn vị bán hàng được đánh dấu là mới với khách hàng. |
| Cờ | `used_vpn` | Bật khi kết nối dùng VPN. |
| Cờ | `ip_country_mismatch` | Bật khi quốc gia suy ra từ IP lệch quốc gia đối chiếu. |
| Cờ | `billing_shipping_mismatch` | Bật khi địa chỉ thanh toán và giao hàng khác nhau. |
| Cờ | `is_ai_generated_scam_attempt` | Bật khi giao dịch có cờ dấu hiệu lừa đảo do AI tạo. Đây là trường có sẵn trong CSV. |

Tài liệu cục bộ chưa xác nhận công thức của hai điểm `velocity_score` và `merchant_risk_score`, ngày tương ứng với mã 0, hoặc quốc gia tham chiếu của cờ lệch IP. [Từ điển và phạm vi đối chiếu](../eda_review/tu_dien_24_dac_trung.md) · [Quy tắc nguồn dữ liệu](../README.md#quy-tắc-dự-án)

## Phân phối, độ chính xác và độ hỗ trợ

- **Train và test gần nhau theo các phép đo đã dùng.** KS (Kolmogorov–Smirnov, chênh lệch lớn nhất giữa hai phân phối tích lũy) cao nhất là **0,003825** ở `day_of_week`. TV (total variation, một nửa tổng chênh lệch tỷ trọng tuyệt đối) cao nhất là **0,002310** ở `auth_method`. Mọi mức của cột phân loại trong test đều có ở train. Phép huấn luyện LightGBM phân biệt train/test của Grandmaster đạt AUC **0,500245** qua 5 fold. Các phép đo này giúp đánh giá cách chia phân tầng. Từng nhóm tương tác vẫn có thể khác về tỷ trọng hoặc độ hỗ trợ. [Kiểm phân phối](../eda_review/data_audit.json) · [Báo cáo Grandmaster, mục 4](<../Model tham khảo/Grandmaster/FINAL_RESEARCH_REPORT.md>)
- **Khóa số chính xác có đuôi thưa.** `account_balance_usd` có 43.317 mức train. 6.668 dòng test có mức chưa gặp và 22.722 dòng có mức xuất hiện dưới 5 lần trong train, tính cả mức chưa gặp. Với `amount_usd` có 13.387 mức train, hai số này là 1.617 và 5.743 dòng. Một khóa chi tiết vì vậy cần điều chuẩn hoặc quay về nhóm rộng hơn khi học từ nhãn. [Số dòng theo độ hỗ trợ](../eda_review/data_audit.json)
- **Lưới lưu số có cấu trúc.** Bảy trường số nguyên. Bốn trường tiền, số dư, khoảng cách và khoảng giờ thường theo bước 0,01. Hai điểm velocity và rủi ro merchant chủ yếu theo bước 0,1. Báo cáo ban đầu ghi `merchant_risk_score` đến hai chữ số vì có một ngoại lệ train 19,12. Hai ngoại lệ test theo lưới danh định là số tiền 9,982 và velocity 29,11. Các mô hình có thể giữ giá trị gốc để học xu hướng và chỉ lượng tử hóa khi tạo khóa riêng. [Kiểm độ chính xác ban đầu](../eda_review/data_audit.json) · [Quét chuỗi CSV theo lưới danh định](../goal_0952/deep_eda_measurement/BAO_CAO.md)

## Tín hiệu đơn biến và phân tích lỗi của `FINAL`

Phép đo đơn biến học tỷ lệ gian lận co về tỷ lệ chung với `m=20` trên train của từng fold. Biến số nhiều mức được chia theo nhóm phân vị của phần train, biến ít mức giữ khóa gốc. OOF (out-of-fold, dự đoán ngoài fold) ghép điểm của mỗi dòng từ mô hình đã chừa dòng đó. AP (Average Precision, tổng hợp precision theo recall) giúp đọc hiệu quả trong bài toán có khoảng 1,67% mẫu dương. [Phương pháp và đủ 24 kết quả](../eda_review/univariate_signal.json) · [Mã tái lập](../eda_review/univariate_signal.py)

| Cột | OOF ROC AUC | AP |
|---|---:|---:|
| `merchant_risk_score` | 0,741200 | 0,041200 |
| `velocity_score` | 0,706536 | 0,036882 |
| `cvv_retry_count` | 0,705663 | **0,085704** |
| `auth_method` | 0,684998 | 0,040293 |
| `is_new_merchant` | 0,640089 | 0,032061 |

CVV có AP cao nhất trong 24 cột. Tỷ lệ gian lận tăng từ **0,93%** ở CVV bằng 0 lên **8,75%** ở 1 và **41,66%** ở ít nhất 2. Với 7.000 giao dịch điểm cao nhất, `FINAL` tìm 4.641/11.689 ca, tương ứng precision 66,30% và recall 39,70%. Trong 3.870 ca gian lận không bật cờ nào, 3.262 nằm ngoài top 1% dù tổng kỳ vọng của `FINAL` cho nhóm này là 3.884,8 ca. Tổng xác suất khá sát nhãn thật trong khi nhiều ca vẫn bị xếp thấp trong nội bộ nhóm. [Sàng lọc và phân tích lỗi `FINAL`](../eda_review/ket_qua_eda.md)

| Nhóm có số ca vượt kỳ vọng ở cả fold 4 và 5 | Dòng train | Gian lận thực tế | `FINAL` dự kiến | Dòng test |
|---|---:|---:|---:|---:|
| POS từ 00:00 đến 05:59 | 48.227 | 856 | 788,5 | 20.809 |
| 10% velocity cao nhất so với nhịp tương ứng | 69.750 | 2.911 | 2.765,8 | 29.913 |
| Merchant mới thuộc `Utilities` | 6.256 | 268 | 243,0 | 2.690 |
| Giao dịch nội địa, lệch IP và merchant mới | 1.426 | 402 | 389,6 | 627 |

Các nhóm được khoanh từ fold 1–3 rồi đối chiếu riêng fold 4 và 5. Bảng là số gộp toàn train. Trong 14 phép hiệu chỉnh điểm nhóm, kênh × khung 6 giờ tăng **0,000021 AUC ở 4/5 fold** và velocity lệch trung vị tăng **0,000016 ở 3/5 fold**, cả hai dưới mốc +0,000100. Sàng lọc chữ số riêng kiểm 91 mẫu và chuyển 15 sang fold 4–5. Mẫu dưới 25 USD có đuôi `.50` hoặc `.99` ghi 21 ca so với XGB dự kiến 34,4 trên 2.300 dòng, bằng chứng mô tả cho một nhóm hẹp. Khi **huấn luyện lại** cùng XGB-D03, thêm riêng hai ứng viên lần lượt làm AUC giảm **0,000074** và **0,000099**. [Số liệu 14 nhóm](../eda_review/feature_screen.json) · [Kiểm chữ số](../eda_review/decimal_probe.json) · [Huấn luyện ablation](../eda_review/feature_ablation/results/ket_qua_thu_dac_trung.md)

## Từ chữ số đến sai lệch tương tác Grandmaster

- **Mức chính xác bổ sung cho đường trơn.** Trên nền Spline-GLM Dev5, thêm hiệu ứng mức chính xác tăng AUC rõ ở tuổi khách hàng (+0,001294), rủi ro merchant (+0,000554), velocity (+0,000529) và giờ từ giao dịch trước (+0,000482). One-hot mức chính xác ở giờ trong ngày, khoảng cách và tuổi thẻ đều giảm AUC so với giữ spline. Cent của `amount_usd` có χ² = 370,68, `p = 4,99 × 10⁻³³` trong phép kiểm mô tả của Grandmaster. Ví dụ đuôi `.49` ghi 2,49% gian lận trên 7.274 dòng. [Phát hiện và đối chứng âm, mục 4](<../Model tham khảo/Grandmaster/FINAL_RESEARCH_REPORT.md>)
- **Sau khi có Grandmaster, tín hiệu còn sót dịch sang một số cặp.** Lượt `goal_0952` xét 276 cặp trên 594.966 dòng phát triển. Mỗi ô tính tổng quan sát trừ dự kiến và loại hiệu ứng của từng trục. Phép sàng lọc đòi mỗi nửa có ít nhất 40 dòng và tổng thông tin logistic ít nhất 4 mỗi ô. Có 32 cặp cho giảm log loss xấp xỉ dương khi học và chấm chéo hai chiều. Dẫn đầu là số giao dịch 24 giờ × loại thẻ (37 ô, Spearman 0,513, giảm 11,42 phần triệu), giờ từ giao dịch trước × lệch IP (24 ô, 0,454, 6,62 phần triệu) và CVV × loại thẻ (14 ô, 0,868, 5,72 phần triệu). Các khóa mức chính xác của thời gian, velocity và merchant risk thiếu cải thiện lặp lại trên nền đã mã hóa chữ số. Chữ số cuối của khoảng cách còn tín hiệu dương. [Báo cáo EDA 0,952](../goal_0952/BAO_CAO_0952.md) · [276 cặp và 18 cách biểu diễn số](../goal_0952/residual_eda.json) · [Mã sàng lọc](../goal_0952/eda_from_scratch.py)
- **Nhiều lỗi xếp hạng nằm ngoài nhóm điểm cao nhất.** So 180.981.473 cặp gian lận–hợp lệ trên tập giữ lại, Grandmaster EDA xếp sai tương đương 8.827.471 cặp. 282 ca dương dưới phân vị điểm 90 chỉ chiếm 16,1% ca dương nhưng tham gia 76,8% số cặp sai. Ở phía giao dịch hợp lệ, nhóm từ phân vị 75 trở lên tham gia 81,2% số cặp sai. Mức AUC 0,952 tương ứng sửa ròng khoảng 140.360 cặp trên tập này. [Ma trận và kích thước từng dải](../goal_0952/deep_eda_triples/rank_errors.json) · [Biểu đồ lỗi](../goal_0952/deep_eda_triples/rank_errors.png)

## EDA sâu theo lưới số, hành vi và tương tác ba biến

### 404 mức tiền thưa cục bộ

Đếm số giao dịch `n(c)` trên lưới cent của tập phát triển. Với từng cent `c`, tính số đếm trung bình `q₅(c)` và `q₂₅(c)` của mọi cent trong cửa sổ ±5 và ±25, loại cent đang xét và tính cả mức có số đếm 0. Chọn mức có `n(c) ≥ 10`, cả hai trung bình `≥ 40`, hai tỷ số `(n(c)+5)/(q(c)+5) ≤ 0,4` và cách cả hai biên miền quan sát hơn 25 cent. Danh sách chốt bằng tần suất, gồm các ví dụ 9,84, 10,06, 22,15 và 31,32 USD. [Phương pháp chốt trước](../goal_0952/deep_eda_measurement/METHOD.md) · [Báo cáo lưới số](../goal_0952/deep_eda_measurement/BAO_CAO.md)

| Tập | Dòng trong 404 mức | Gian lận quan sát | Gian lận dự kiến |
|---|---:|---:|---:|
| Phát triển, fold 1–5 | 4.313 | 85 | 74,20 |
| Phát triển, fold 6–10 | 4.368 | 106 | 81,34 |
| Giữ lại | 1.644 | 46 | 31,86 |
| Test | 4.448 | — | — |

- **Bằng chứng trong hai nửa.** 82 nhóm gồm lưới thô, điểm dồn/thưa, biên, đuôi và ngoại lệ. Trong 45 nhóm đủ hỗ trợ, 18 giảm log loss qua cả hai chiều, và nhóm tiền là nhóm duy nhất qua ngưỡng p Bonferroni `0,05/45 = 0,001111`. p hai phía của nhóm là **0,000316**. Nhóm chiếm 1,475% train và 1,483% test. Tỷ lệ số đếm so với lân cận ±5 cent lần lượt là 16,75% trên phát triển và 17,01% trên test. Test xác nhận cấu trúc tần suất, còn tỷ lệ gian lận được chấm bằng nhãn train. [Số liệu kiểm định](../goal_0952/deep_eda_measurement/measurement_metrics.json) · [Độ nhạy chuyển tập](../goal_0952/deep_eda_triples/amount_hole_sensitivity.json)
- **Tác động trên holdout.** Cộng **+0,27198694 logit** học từ tập phát triển vào điểm 1.644 dòng thuộc nhóm. AUC đạt **0,9513158454622589** so với **0,9512244493667037**. AP tăng từ 0,53823528 lên 0,53852282 và log loss giảm từ 0,04383599 xuống 0,04380602. Khoảng DeLong ghép cặp 95% của Δ AUC là **[−0,00002229, +0,00020508]**. Vùng điểm 75–90 đóng góp +0,00006346 trong tổng mức tăng AUC +0,00009140. [Kết quả giữ lại](../goal_0952/deep_eda_measurement/measurement_metrics.json) · [Đóng góp theo dải điểm](../goal_0952/deep_eda_triples/amount_hole_sensitivity.json)
- **Kiểm tra độ nhạy và xuất tệp.** Bỏ lần lượt từng mức giữ mức giảm log loss từ 0,00002318 đến 0,00003267. Bootstrap theo cụm mức tiền cho khoảng mức giảm `[+0,00000161, +0,00006008]`. Hoán vị theo dải tiền 10 USD, dải điểm nền và chữ số cent cuối có `p = 0,002` trên cùng holdout. Khi xuất CSV, dự án giữ danh sách 404 mức, fit lại hệ số bằng 699.960 dòng train và lấy **+0,28729082 logit**, tương đương nhân odds **1,33281**, trên 4.448 dòng test. [Kiểm độ nhạy](../goal_0952/deep_eda_triples/amount_hole_sensitivity.json) · [Manifest và SHA-256](../goal_0952/deep_eda_measurement/amount_hole_submission.json) · [CSV ứng viên](../goal_0952/deep_eda_measurement/submission_grandmaster_eda_amount_hole.csv)

### Các giả thuyết còn lại

| Giả thuyết | Kết quả và giới hạn |
|---|---|
| Khoảng cách có tần suất thấp hơn lân cận | 295 ca trên 15.110 dòng phát triển, so với 260,39 kỳ vọng, hai nửa cùng hướng. Phủ 7.706 test. `p = 0,00690` cao hơn ngưỡng 0,001111. |
| Lưới làm tròn, biên và ngoại lệ | Velocity bằng 0 chiếm khoảng 7% mỗi tập. Trên phát triển, 16 ca so với 17,41 kỳ vọng. Số tiền bội số 0,25 USD ngoài số nguyên ở kênh Online có 83 ca so với 100,77 kỳ vọng và `p = 0,02444`. Ba dòng ngoài lưới danh định quá ít để ước lượng. |
| Chuyển ngày hoặc tuần × kênh | Ứng viên qua hai nửa phát triển. Áp lên Grandmaster EDA giữ lại làm AUC giảm **0,00013785** và log loss tăng. Thời điểm trước được suy ra từ giờ nguyên và khoảng cách thời gian làm tròn, có trạng thái chưa xác định khi khoảng bất định cắt ranh giới. |
| `account_balance_usd − amount_usd` × loại thẻ | Ứng viên qua hai nửa phát triển. AUC giữ lại giảm **0,00009981**. Riêng Visa với hiệu số dư `[0,100)` có 6 ca giữ lại so với 7,17 dự kiến, đảo hướng quan sát ở hai nửa phát triển. |
| Tương tác thuần ba biến | Xét 16 tổ hợp, loại phần giải thích của cả ba cặp. A/B/C có 237.986 / 178.490 / 178.490 dòng. Mỗi ô cần ít nhất 100 dòng và tổng thông tin logistic ≥6. Ba tổ hợp qua sàng lọc hai chiều đều tăng log loss ở C. Ví dụ số giao dịch × khoảng giờ × kênh giảm 2,63 và 2,31 phần triệu ở hai chiều, rồi tăng 7,96 phần triệu tại C. Chín tổ hợp không còn bậc tự do trên các ô đủ hỗ trợ chung để tách hiệu ứng ba biến. |

Nguồn cho các phép thử trên gồm [Báo cáo lưới số](../goal_0952/deep_eda_measurement/BAO_CAO.md), [Kiểm thời gian và số dư](../goal_0952/deep_eda_behavior/BAO_CAO_THOI_GIAN_SO_DU.md), [16 thử nghiệm ba biến](../goal_0952/deep_eda_triples/triple_eda_results.json) và [Mã chiếu bỏ các hiệu ứng cặp](../goal_0952/deep_eda_triples/triple_eda.py).

## Cách đọc các phép đo

| Phép đo | Phạm vi | Cách đọc |
|---|---|---|
| GPU 5 fold, seed 42 | OOF trên 699.960 dòng | So hai đặc trưng trên cùng fold và cùng `XGB-D03`. `FINAL` còn có xác nhận lồng nhau 5×5. |
| Grandmaster Confirm10 | OOF 10 fold trên 699.960 dòng | Bảy mô hình và tầng gộp train local. OOF Grandmaster là 0,951267252810. README Grandmaster lưu public leaderboard **0,95154** của lượt nộp Grandmaster gốc. |
| `goal_0952` phát triển → giữ lại | 594.966 dòng phát triển, 9.936 ca. Giữ lại 104.994 dòng, 1.753 ca. | Dùng cùng tập giữ lại khi so các biến thể toàn tổ hợp. Tập này đã được dùng qua các vòng chọn mô hình, nên dùng Δ, khoảng tin cậy và đối chứng cùng giao thức để đọc kết quả. |
| Kiểm đảo vai trò | Nhánh mới học trên 104.994 dòng tham chiếu. Chấm 10 fold của 594.966 dòng còn lại. | Đánh giá nhánh riêng với tập học nhỏ hơn. Δ AUC trong mỗi fold thường gộp theo số cặp dương–âm, khác AUC gộp của toàn tổ hợp. |
| Test Kaggle | 299.984 dòng có ID, không có nhãn | Theo dõi độ phủ, phân phối và tính hợp lệ của CSV. Nhãn test chỉ được Kaggle dùng để chấm khi nộp. |

Định nghĩa và số liệu đến từ [Báo cáo GPU](../../s6e11_gpu/plan_runs/published/full/comparison.md), [README Grandmaster, mục 1–2](<../Model tham khảo/Grandmaster/README.md>), [Phép chia 0,952](../goal_0952/BAO_CAO_0952.md) và [Thiết kế kiểm đảo vai trò](../goal_0952/BAO_CAO_THU_SONG_SONG.md).

## Phạm vi đối chiếu và tái lập

- Đọc toàn bộ `eda_review/ket_qua_eda.md`, `tu_dien_24_dac_trung.md` và báo cáo ablation hai đặc trưng. Đối chiếu các trường cấu trúc, ID, nhãn, độ hỗ trợ và phân phối của `data_audit.json`, toàn bộ danh sách 24 trường của `univariate_signal.json`, cùng mục sàng lọc và nguồn của từng báo cáo. Các số liệu toàn CSV ở tài liệu này lấy từ kết quả audit đã lưu. [Báo cáo EDA gốc](../eda_review/ket_qua_eda.md) · [Ablation hai đặc trưng](../eda_review/feature_ablation/results/ket_qua_thu_dac_trung.md)
- Đọc mục EDA và mô hình được chọn trong `goal_0952/BAO_CAO_0952.md`, mục 4 của `FINAL_RESEARCH_REPORT.md`, toàn bộ `BAO_CAO_EDA_SAU.md`, `deep_eda_measurement/METHOD.md` và `deep_eda_behavior/BAO_CAO_THOI_GIAN_SO_DU.md`. Đối chiếu các trường `holdout`, `groups`, phần phân rã AUC và ngưỡng hỗ trợ trong JSON của các lượt tương ứng. [EDA sâu tổng hợp](../goal_0952/BAO_CAO_EDA_SAU.md) · [Mã truy nguồn và khớp ID](../goal_0952/grandmaster_inputs.py)
- Điểm vào tái lập theo lịch sử gồm [kiểm cấu trúc](../eda_review/data_audit.py), [tín hiệu đơn biến](../eda_review/univariate_signal.py), [276 cặp trên Grandmaster](../goal_0952/eda_from_scratch.py), [lưới số amount hole](../goal_0952/deep_eda_measurement/measurement_eda.py), [độ nhạy theo mức tiền](../goal_0952/deep_eda_triples/audit_amount_holes.py) và [xuất CSV](../goal_0952/deep_eda_measurement/export_amount_hole.py). Các báo cáo nguồn ghi thư mục chạy, môi trường và kho dự đoán cần có cho từng lượt.
