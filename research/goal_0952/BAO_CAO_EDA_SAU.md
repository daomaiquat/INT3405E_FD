# EDA sâu trên Predicting Credit Card Fraud

**Tín hiệu rõ nhất là các mức số tiền xuất hiện ít bất thường so với những mức liền kề.** Hiệu chỉnh riêng nhóm này đưa ROC AUC trên tập giữ lại từ **0,95122445 lên 0,95131585** và giảm log loss từ **0,04383599 xuống 0,04380602**. Đây là ứng viên tôi ưu tiên thử trên Grandmaster EDA. Mức AUC còn cách 0,952 khoảng **0,00068415**, và khoảng tin cậy cho mức tăng vẫn chứa 0. [Submission mới](deep_eda_measurement/submission_grandmaster_eda_amount_hole.csv) · [Bản gốc](submission_grandmaster_eda.csv)

## 404 mức tiền thưa bất thường

Đếm mỗi mức tiền trên lưới cent của 594.966 dòng phát triển, trước khi nhìn nhãn. Lấy số đếm trung bình của các mức kề trong hai cửa sổ ±0,05 và ±0,25 USD, bỏ mức đang xét và tính cả mức có số đếm bằng 0. Mỗi mức cần ít nhất 10 dòng, mỗi bình quân lân cận cần ít nhất 40 dòng và tỷ số `(số đếm + 5) / (bình quân lân cận + 5)` phải ≤ 0,4 ở cả hai cửa sổ. Loại 25 cent sát hai biên của miền quan sát. Ví dụ 9,84, 10,06, 22,15 và 31,32 USD. Đặc trưng này so tần suất cục bộ và bổ sung cho mã hóa chữ số của Grandmaster. [Quy tắc đã chốt](deep_eda_measurement/METHOD.md)

| Tập | Dòng thuộc 404 mức | Gian lận quan sát | Gian lận dự kiến |
|---|---:|---:|---:|
| Nửa phát triển 1 | 4.313 | 85 | 74,20 |
| Nửa phát triển 2 | 4.368 | 106 | 81,34 |
| Tập giữ lại | 1.644 | **46** | **31,86** |
| Test | 4.448 | — | — |

- **Tín hiệu lặp lại qua hai nửa phát triển.** Trong 82 nhóm đã xét, 45 nhóm đủ mẫu. Nhóm số tiền là nhóm duy nhất vượt ngưỡng Bonferroni đã chốt và giảm log loss qua cả hai chiều học–chấm. Mức p hai phía là 0,000316, với ngưỡng 0,001111. Cột kỳ vọng dùng điểm Grandmaster phân cấp ngoài fold ở phần phát triển và Grandmaster EDA đầy đủ ở tập giữ lại. [Số liệu sàng lọc](deep_eda_measurement/measurement_metrics.json)
- **Cấu trúc tần suất có mặt trên test.** Cả 404 mức đều dưới tần suất bình quân của các mức kề, ở cả hai thang. Tổng số quan sát ở cửa sổ ±0,05 USD bằng 16,75% số đếm kỳ vọng từ mức kề trên phát triển và 17,01% trên test. Nhóm chiếm 1,475% train và 1,483% test. Test chỉ cung cấp kiểm chứng về phân bố của đặc trưng. [Kiểm tra tần suất chuyển tập](deep_eda_triples/amount_hole_sensitivity.json)
- **Kết quả giữ lại ổn định khi bỏ từng mức tiền.** Mức giảm log loss còn 0,00002318–0,00003267 khi lần lượt bỏ từng mức đã quan sát. Khoảng bootstrap theo mức tiền cho mức giảm log loss là [+0,00000161, +0,00006008]. Phép hoán vị trong cùng dải tiền 10 USD, cùng dải điểm Grandmaster và cùng chữ số cent cuối cho p điều kiện 0,002. Đây là các kiểm tra độ nhạy mô tả trên cùng tập giữ lại. [Kiểm tra độ nhạy](deep_eda_triples/amount_hole_sensitivity.json)
- **Tăng AUC tập trung ở vùng điểm trung bình.** Các ca gian lận có điểm Grandmaster ở phân vị 75–90 đóng góp +0,00006346 vào tổng mức tăng +0,00009140. Average Precision tăng từ 0,53823528 lên 0,53852282. Khoảng DeLong ghép cặp 95% cho Δ AUC là [−0,00002229, +0,00020508]. Tập giữ lại 104.994 dòng đã được dự án dùng ở các vòng trước. [Kết quả giữ lại và phân rã AUC](deep_eda_triples/amount_hole_sensitivity.json)

Để xuất submission, tôi giữ danh sách 404 mức và ước lượng lại một hệ số trên 699.960 dòng train bằng điểm Grandmaster phân cấp ngoài fold. Hệ số trên tập phát triển là +0,27199 log-odds và hệ số xuất test là +0,28729. Hệ số test nhân tỷ lệ cược lên khoảng 1,333 cho đúng 4.448 dòng. CSV có 299.984 ID đúng thứ tự, xác suất hợp lệ và dự đoán của các dòng ngoài nhóm khớp bản gốc. [Báo cáo nhóm số](deep_eda_measurement/BAO_CAO.md) · [Manifest và SHA-256](deep_eda_measurement/amount_hole_submission.json)

## Những phát hiện khác

| Giả thuyết | Bằng chứng kiểm tra |
|---|---|
| **Mức khoảng cách thưa so với các mức kề** | 295 ca trên 15.110 dòng phát triển so với 260,39 dự kiến, hai nửa cùng hướng và phủ 7.706 test. p = 0,00690 cao hơn ngưỡng 0,001111 đã đặt. Đây là ứng viên thứ hai để kiểm chứng trên một phép chia mới. |
| Chuyển ngày hoặc tuần × kênh | Tăng nhẹ ở cả hai nửa phát triển, sau đó làm AUC Grandmaster EDA giảm 0,00013785 và log loss tăng trên tập giữ lại. |
| Số dư trừ số tiền × loại thẻ | Cải thiện nhẹ ở cả hai nửa phát triển. Khi chấm Grandmaster EDA trên tập giữ lại, AUC giảm 0,00009981. Ví dụ nhóm Visa có số dư chênh [0, 100) USD có 6 ca, so với 7,17 dự kiến. |
| Tương tác thật sự của ba biến | Sàng lọc 16 tổ hợp sau khi loại cả ba hiệu ứng cặp, co theo độ hỗ trợ. Có 3 tổ hợp lặp lại ở A→B và B→A; cả 3 làm log loss xấu đi trên phần C tách riêng. Trong 9 tổ hợp, các ô đủ hỗ trợ ở cả hai nửa không còn bậc tự do để tách phần tương tác ba biến. |
| Làm tròn thô, biên và ngoại lệ | Có ba điểm ngoài lưới danh định trên gần một triệu dòng. Điểm velocity bằng 0 có khoảng 7% dữ liệu, với 16 ca so với 17,41 dự kiến trong phần phát triển. Các nhóm kiểm tra này chưa vượt tiêu chí đã chốt. |

Phép kiểm ba biến chia 594.966 dòng phát triển thành A = 237.986, B = 178.490 và C = 178.490. Mỗi ô cần ít nhất 100 dòng và tổng thông tin logistic ít nhất 6. Ví dụ số giao dịch 24 giờ × giờ từ giao dịch trước × kênh giảm log loss 2,63 và 2,31 phần triệu trong hai chiều sàng lọc, rồi tăng 7,96 phần triệu ở C. Các điểm nền ngoài fold vẫn có thể chia sẻ dữ liệu huấn luyện. [Chi tiết hành vi](deep_eda_behavior/BAO_CAO_THOI_GIAN_SO_DU.md) · [Biểu đồ 16 tương tác](deep_eda_triples/triple_eda.png) · [Bảng kết quả](deep_eda_triples/triple_eda_results.json)

## Vùng lỗi còn nhiều dư địa

Trong 180.981.473 cặp gian lận–hợp lệ trên holdout, Grandmaster EDA xếp sai tương đương 8.827.471 cặp. Điều chỉnh khoảng **140.360 cặp ròng** trên chính tập này mới đưa AUC lên 0,952. Có **282 ca gian lận** trong 90% điểm thấp nhất, chiếm **16,1%** ca gian lận và tạo **76,8%** số cặp xếp sai. Phía giao dịch hợp lệ, các trường hợp từ phân vị 75 trở lên tham gia **81,2%** số cặp xếp sai. Tín hiệu số tiền góp phần sửa thứ hạng quanh vùng 75–90 là lý do nó đáng ưu tiên hơn việc chỉ hiệu chỉnh xác suất ở nhóm điểm cao nhất. [Ma trận lỗi xếp cặp](deep_eda_triples/rank_errors.png) · [Số cặp và số ca theo nhóm](deep_eda_triples/rank_errors.json)

## Tái lập và phạm vi nguồn

- Đọc toàn bộ báo cáo hành vi và phần sàng lọc, kiểm chứng, phạm vi nguồn của báo cáo giá trị số. Báo cáo vòng trước được đọc từ đầu đến mục Submission để đối chiếu các thử nghiệm đã làm. [Hành vi](deep_eda_behavior/BAO_CAO_THOI_GIAN_SO_DU.md) · [Giá trị số](deep_eda_measurement/BAO_CAO.md) · [Lượt song song](BAO_CAO_THU_SONG_SONG.md)
- Đọc hàm dựng điểm đối chứng trong `run_followup.py`, phần quy tắc điểm thưa và phép xác nhận holdout trong `measurement_eda.py`, phương pháp sàng lọc đã chốt và các trường kết quả được nêu ở trên. [Dựng đối chứng](followup/run_followup.py) · [Mã kiểm tra nhóm số](deep_eda_measurement/measurement_eda.py) · [Phương pháp](deep_eda_measurement/METHOD.md)
- Toàn bộ dòng và cột của ba CSV chính thức được đọc trong quy trình kiểm tra nguồn và khớp ID. Phân tích nhãn dùng tập phát triển và holdout. ID chỉ phục vụ đối chiếu và xuất dự đoán.

Chạy từ thư mục tương ứng với môi trường Python có sẵn tại `..\..\grandmaster_refit\.refit-env\Scripts\python.exe`. Các tệp dưới đây tái lập phần phân tích mới. [Ba biến](deep_eda_triples/triple_eda.py) · [Ma trận lỗi](deep_eda_triples/rank_errors.py) · [Độ nhạy mức tiền](deep_eda_triples/audit_amount_holes.py) · [Xuất CSV](deep_eda_measurement/export_amount_hole.py)
