# S6E11 · Rà soát đặc trưng từ dữ liệu gốc

**Giữ bộ đặc trưng hiện tại.** Trong 14 nhóm được sàng lọc, tốt nhất là **kênh thanh toán × khung giờ**, với Δ OOF ROC AUC **+0.000021** và tăng ở **4/5 fold**. Mức tăng thấp hơn mốc **+0.000100** trong kế hoạch, đồng thời Average Precision (AP) giảm **0.000035**. Nhóm **velocity cao bất thường so với nhịp giao dịch** có sai số lặp lại trên nhiều giao dịch và là ứng viên thứ hai nếu chạy thử nghiệm riêng. [Biểu đồ sàng lọc](eda_feature_screen.png) · [Số liệu 14 nhóm](feature_screen.json)

## Dữ liệu và phạm vi kiểm tra

Đã đọc toàn bộ dòng và cột trong `train.csv` và `test.csv`, kiểm cấu trúc và thứ tự ID trong `sample_submission.csv`, và đối chiếu SHA-256 của cả ba tệp với bản dùng huấn luyện. Phần chấm điểm đọc toàn bộ dự đoán ngoài fold (out-of-fold, OOF) của `XGB-D03` và `FINAL`. Phần sàng lọc đối chiếu thêm 25 tệp dự đoán của `XGB-D03` từ vòng chia lồng nhau, gồm 5 fold trong cho mỗi fold ngoài. [Kiểm tra dữ liệu](data_audit.json) · [Mã sàng lọc](run_feature_screen.py)

| Kiểm tra | Kết quả |
|---|---|
| Kích thước | 699.960 dòng train và 299.984 dòng test. 24 đặc trưng gồm 13 cột số, 5 cột phân loại, 6 cờ. |
| Nhãn | 11.689 giao dịch gian lận trong train, chiếm **1,67%**. Test có 24 đặc trưng và `id`. |
| Chất lượng | 0 ô thiếu hay vô hạn. ID duy nhất, train và test không giao ID. 0 dòng trùng toàn bộ 24 đặc trưng trong hoặc giữa hai tập. |
| Dịch chuyển train–test | Khoảng cách KS lớn nhất ở cột số là `day_of_week`, **0,003825**. Khoảng cách tổng biến thiên lớn nhất của cột phân loại và cờ là `auth_method`, **0,002310**. Tất cả mức phân loại trong test đều xuất hiện ở train. |
| Độ hỗ trợ của khóa số | `account_balance_usd` có **6.668 dòng test** với khóa chưa gặp (**2,223%**) và **22.722 dòng** có khóa xuất hiện dưới 5 lần trong train (**7,574%**, gồm cả khóa chưa gặp). Với `amount_usd` hai số tương ứng là **1.617** và **5.743**. |
| Độ chính xác số | Train có 7 cột số nguyên, `velocity_score` một chữ số thập phân và 5 cột giữ đến hai chữ số thập phân. Trong test, `amount_usd` và `velocity_score` mỗi cột có một giá trị lệch khỏi lưới tương ứng. |

So sánh phân phối ở trên đo từng cột. Các tương tác mới vẫn cần so độ hỗ trợ qua từng fold và trên test. Với khóa số thưa, nếu thử mã hóa thêm thì nên so một phương án quay về nhóm rộng hơn khi khóa hiếm. Giữ giá trị số gốc ở đầu vào và chỉ làm tròn khi tạo khóa riêng. Dữ liệu hiện có các số đếm và thời gian kể từ giao dịch trước ở từng dòng. Tạo lịch sử theo người dùng hoặc thẻ sẽ cần mã định danh và chuỗi giao dịch tương ứng.

## Tín hiệu đã có trong 24 cột gốc

Với từng cột, tôi chia cột số theo phân vị của phần train từng fold, giữ khóa gốc cho cột rời rạc ít mức, rồi dùng tỷ lệ gian lận có làm trơn để dự đoán phần giữ lại. Bảng dưới là 5 cột đứng đầu theo OOF ROC AUC. [Kết quả đủ 24 cột](univariate_signal.json)

| Cột | OOF ROC AUC | AP |
|---|---:|---:|
| `merchant_risk_score` | 0.741200 | 0.041200 |
| `velocity_score` | 0.706536 | 0.036882 |
| `cvv_retry_count` | 0.705663 | **0.085704** |
| `auth_method` | 0.684998 | 0.040293 |
| `is_new_merchant` | 0.640089 | 0.032061 |

`cvv_retry_count` có AP cao nhất trong 24 cột. Tỷ lệ gian lận tăng từ **0,93%** khi CVV bằng 0 lên **8,75%** khi CVV bằng 1 và **41,66%** khi CVV từ 2 trở lên. Với ngưỡng xếp hạng top 1%, `FINAL` tìm được **693/804** ca ở nhóm CVV từ 2 trở lên. Tín hiệu thô mạnh này đã phản ánh khá rõ trong dự đoán.

`XGB-D03` hiện dùng **109** cột gồm 54 cột gốc sau khi mã hóa phân loại, 13 cột mã hóa theo nhãn, 6 cột phần thập phân và 36 cột mã hóa khóa chữ số. Trước đây nhóm khóa chữ số đem lại **+0.001580** cho `XGB-D02` so với `T03`. Các thử nghiệm tần suất `COUNT` và biến dẫn xuất `DERIVED` giảm lần lượt **0.001749** và **0.000032** trên đối chứng LightGBM `T04`. `DERIVED` thử chung bình phương số cờ và tỷ lệ số tiền trên số dư. Nối 64 chiều biểu diễn MLP vào `XGB-D03` cũng giảm **0.000522**. [Kết quả ablation đã chạy](../s6e11_gpu/plan_runs/results/full/comparison.md) · [Kết quả thêm embedding](../s6e11_gpu/pretraining/results/summary.md)

## Sai số còn lại của hai mô hình

Tổng xác suất OOF trong một nhóm cho biết số ca mô hình dự kiến ở nhóm đó. Tôi đối chiếu số ca thực tế với số dự kiến, lấy fold 1–3 để tìm nhóm đáng chú ý rồi xem riêng fold 4 và 5. Bảng ghi tổng toàn bộ train để dễ đọc. Cả bốn nhóm dưới đều có số ca thực tế vượt số `FINAL` dự kiến ở từng fold 4 và 5. [Bảng nhóm và số liệu từng fold](feature_screen.json)

| Nhóm | Dòng train | Ca thực tế | XGB dự kiến | FINAL dự kiến | Dòng test |
|---|---:|---:|---:|---:|---:|
| POS, giờ 00:00–05:59 | 48.227 | 856 | 786,6 | 788,5 | 20.809 |
| 10% dòng có velocity cao nhất so với nhịp giao dịch tương ứng | 69.750 | 2.911 | 2.743,0 | 2.765,8 | 29.913 |
| Merchant mới thuộc `Utilities` | 6.256 | 268 | 237,2 | 243,0 | 2.690 |
| Giao dịch nội địa, IP lệch quốc gia, merchant mới | 1.426 | 402 | 374,7 | 389,6 | 627 |

Khi chỉ rà soát **7.000 giao dịch rủi ro cao nhất**, `FINAL` tìm **4.641/11.689** ca, tương ứng recall **39,70%** và precision **66,30%**. Trong **3.870** ca gian lận không bật cờ nào, **3.262** nằm ngoài top 1%. Riêng nhóm không cờ, `FINAL` dự kiến tổng cộng **3.884,8** ca, gần số thực tế **3.870**. Vì vậy vấn đề đáng chú ý là xếp hạng bên trong nhóm đông này. Việc tăng điểm đồng loạt cho cả nhóm chưa có cơ sở.

## Thử hiệu chỉnh điểm với 14 nhóm

Tôi giữ nguyên dự đoán `XGB-D03` trên các fold ngoài. Với mỗi ứng viên, tôi ước lượng mức chỉnh cho từng nhóm bằng chênh lệch nhãn thật và dự đoán OOF **bên trong phần train** của fold tương ứng. Tôi áp mức chỉnh vào log-odds, tức logarit tỷ lệ cược, trên fold giữ lại. Nhóm cần ít nhất 500 dòng huấn luyện và 5 ca dự kiến mới được chỉnh. Mức chỉnh có co về 0 và bị giới hạn để giảm tác động của nhóm ít hỗ trợ. Mốc sàng lọc trong kế hoạch là tăng ít nhất **0.000100 AUC** và tăng ở **4/5 fold**. [Bước 15 của kế hoạch](../../plan.md#L511) · [Mã và tham số thử nghiệm](run_feature_screen.py)

| Ứng viên | Δ OOF ROC AUC | Fold tăng |
|---|---:|---:|
| Kênh thanh toán × khung giờ 6 tiếng | **+0.000021** | **4/5** |
| Velocity lệch so với nhịp giao dịch | +0.000016 | 3/5 |
| Nước ngoài × lệch IP × merchant mới | +0.000009 | 2/5 |
| Velocity × số cờ mạng | +0.000002 | 2/5 |
| Khoảng tiền × chữ số cuối | +0.000002 | 2/5 |
| Số tiền so với mức điển hình của merchant và kênh | +0.000001 | 3/5 |
| Kênh × thiết bị | −0.000003 | 3/5 |
| Tỷ lệ số tiền trên số dư | −0.000015 | 3/5 |
| Số lần thử CVV × số cờ | −0.000027 | 1/5 |
| Số lần thử CVV × khoảng tiền | −0.000033 | 0/5 |
| Số lần thử CVV × kênh | −0.000033 | 2/5 |
| Khoảng cách so với mức điển hình của kênh và vùng | −0.000035 | 1/5 |
| Loại merchant × merchant mới | −0.000048 | 1/5 |
| Số lần thử CVV × xác thực | −0.000065 | 0/5 |

Các số trên đo tác động của **hiệu chỉnh điểm theo nhóm** trên dự đoán đã có. Thêm cột rồi huấn luyện lại cây là một thí nghiệm riêng. Cùng dữ liệu này đã được dùng để phát triển các mô hình, nên đây vẫn là đánh giá khảo sát. Chưa có nhóm nào qua cả hai tiêu chí sàng lọc.

## Nhóm nên ưu tiên nếu chạy lượt tiếp theo

1. **`channel × hour_block_6h`**. Tạo biến tương tác giữa kênh thanh toán và một trong bốn khung giờ. Thử riêng với `XGB-D03`, giữ fold và cấu hình đối chứng. Đây là ứng viên có mức tăng AUC cao nhất và tăng ở nhiều fold nhất trong phép sàng lọc.
2. **`velocity_score` trừ trung vị trong nhóm nhịp giao dịch tương tự**. Chia nhóm theo `txn_count_last_24h` và `hours_since_last_txn`, tính trung vị chỉ trên phần train mỗi fold rồi lấy hiệu. Thử giữ hiệu dạng liên tục để cây tự chọn ngưỡng. Phép sàng lọc dùng nhóm phân vị của hiệu này, tăng AP **0.000115**, với bằng chứng sai số nhất quán ở nhóm 10% cao nhất.

**Ưu tiên thấp cho việc mở rộng chữ số.** Phép kiểm riêng xét **91 mẫu**, đưa **15** vào kiểm tra ở fold 4–5. Một vài nhóm hẹp đáng ghi nhận. Chẳng hạn số tiền dưới 25 USD có đuôi `.50` hoặc `.99` ghi **21** ca thật so với **34,4** ca XGB dự kiến trên 2.300 dòng của hai fold này. Các mẫu chữ số còn lại thiếu độ nhất quán rộng để ưu tiên thêm một nhóm khóa lớn. [Phép kiểm chữ số](decimal_probe.json)

Giữ hai ứng viên ưu tiên thành **hai phép thử riêng**. Dùng `XGB-D03` **0.950182** làm đối chứng trực tiếp, ghi Δ theo fold cùng AP, rồi mới cân nhắc đưa ứng viên qua ngưỡng vào tổ hợp `FINAL` **0.950345**.
