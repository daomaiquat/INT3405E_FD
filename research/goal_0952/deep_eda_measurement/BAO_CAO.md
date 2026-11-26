# Phân tích các điểm bất thường trong dữ liệu số

**Nhóm mức tiền xuất hiện thưa hơn các mức ngay cạnh là ứng viên có tín hiệu rõ nhất.** Hiệu chỉnh cục bộ từ tập phát triển giảm log-loss trên 104.994 dòng giữ lại từ **0,04383599 xuống 0,04380602**. ROC AUC lên **0,95131585**, tăng **0,00009140** từ Grandmaster EDA. Khoảng tin cậy DeLong ghép cặp 95% cho mức tăng AUC là **[−0,00002229, +0,00020508]**. Đây là ứng viên để kiểm chứng thêm theo ROC AUC. [Kết quả máy đọc được](measurement_metrics.json)

## Tín hiệu có thể sử dụng

Từ 594.966 dòng phát triển, xác định **404 mức tiền** bằng tần số không nhãn trên lưới cent. Với mỗi mức tiền, tính số giao dịch bình quân trên các mức tiền cách nó tối đa 5 cent và tối đa 25 cent. Mỗi bình quân bỏ mức đang xét và tính cả các mức kề có số đếm bằng 0. Chọn mức tiền có ít nhất 10 dòng, hai bình quân đều ít nhất 40 và tỷ số `(số dòng + 5) / (bình quân lân cận + 5)` không quá 0,4 ở cả hai thang. Loại các điểm trong 25 cent kể từ hai đầu miền quan sát. Ví dụ 9,84, 10,06, 31,32 và 22,15 USD. [Phương pháp chốt trước](METHOD.md)

| Tập | Dòng trong nhóm | Gian lận quan sát | Gian lận dự kiến từ đối chứng |
|---|---:|---:|---:|
| Phát triển, fold OOF 1–5 | 4.313 | 85 | 74,20 |
| Phát triển, fold OOF 6–10 | 4.368 | 106 | 81,34 |
| Phát triển gộp | 8.681 | 191 | 155,54 |
| Giữ lại, đối chứng Grandmaster EDA | 1.644 | 46 | 31,86 |
| Test Kaggle | 4.448 | — | — |

- Trong phát triển, tỷ lệ gian lận trong nhóm là **2,20%** so với **1,79%** theo xác suất của Grandmaster phân cấp ngoài fold. Chênh lệch **+0,409 điểm phần trăm** có khoảng tin cậy theo hàng 95% **[+0,155, +0,662]**. Học hệ số trên mỗi nửa rồi áp dụng sang nửa còn lại giảm log-loss của nhóm **0,000237** và **0,000701**. Đây là nhóm duy nhất vượt ngưỡng hiệu chỉnh Bonferroni trong 45 nhóm đủ mẫu, với p hai phía **0,000316** và ngưỡng **0,001111**.
- Trên tập giữ lại, chênh lệch tỷ lệ của nhóm là **+0,860 điểm phần trăm**, khoảng tin cậy 95% **[+0,221, +1,500]**. Cố định hiệu chỉnh **+0,27198694 logit** từ tập phát triển rồi chấm đúng dự đoán Grandmaster EDA đã lưu. Logit là logarit của tỷ lệ cược. Log-loss toàn tập giảm **0,00002997** với khoảng tin cậy ghép cặp 95% **[−0,00005708, −0,00000285]** cho mức thay đổi.
- Nhóm chiếm **1,475%** của 699.960 dòng train và **1,483%** của 299.984 dòng test. Độ phủ tương đương giữa hai tập. Tập giữ lại từng được dùng trong các vòng phát triển trước, nên kết quả vẫn cần được kiểm tra trên nhãn mới để ước lượng ổn định mức tăng AUC. [Kết quả theo nhóm và tập](measurement_metrics.json)

## Các kiểm tra còn lại

| Kiểm tra | Điều quan sát được |
|---|---|
| Khoảng cách có tần số thấp so với các điểm kề | 15.110 dòng phát triển, 295 ca quan sát và 260,39 dự kiến. Hai nửa cùng hướng, nhưng p = 0,00690 còn cao so với ngưỡng 0,001111 của 45 phép thử. Nhóm này phủ 7.706 test. |
| Bội số và cách làm tròn có điều kiện | Nhóm số tiền là bội số 0,25 USD ngoài số nguyên, riêng kênh Online, có 83 ca so với 100,77 dự kiến trên 5.904 dòng phát triển. Nhóm qua hai chiều log-loss, với p = 0,02444. Tổng hợp nhiều trường cùng nằm trên lưới thô cho hai nửa trái chiều và log-loss tăng theo cả hai chiều. |
| Điểm vận tốc bằng 0 | Chiếm 48.499 dòng train và 20.927 test, tương đương khoảng 7%. Trong 41.413 dòng phát triển đúng biên này, có 16 ca so với 17,41 dự kiến. |
| Bước nhảy ở đuôi số dư | Khoảng 30.449,08–34.197,07 USD là bước nhảy tương đối lớn nhất giữa hai giá trị liền kề trong vùng phân vị 99%–99,9% ở tập phát triển. Phía trên bước nhảy có 953 dòng phát triển, 23 ca so với 17,25 dự kiến. Cỡ mẫu và p = 0,08924 cho thấy bằng chứng còn yếu. |
| Độ chính xác | Ba ngoại lệ ngoài lưới danh định. Train có điểm merchant 19,12. Test có số tiền 9,982 và điểm vận tốc 29,11. Quy mô quá nhỏ để ước lượng tác động. |

Đã xét 82 nhóm, gồm các mức lưới, nhóm có điều kiện, điểm dồn và hõm tần suất, biên, đuôi và độ chính xác. Trong 45 nhóm đủ mẫu, 18 nhóm cho Δ log-loss âm qua cả hai chiều. Một nhóm qua ngưỡng ý nghĩa đã chốt và được xác nhận trên tập giữ lại. [Chi tiết các nhóm](measurement_metrics.json)

## Tệp dự đoán ứng viên

Giữ nguyên tập 404 mức tiền rồi ước lượng lại hệ số trên toàn bộ 699.960 dòng train. Grandmaster phân cấp ngoài fold dự kiến 192,93 ca trong 10.325 dòng thuộc nhóm, với 237 ca quan sát. Dùng cùng mức điều chuẩn L2 là 25 và giới hạn hiệu chỉnh ±0,30, hệ số khi xuất test là **+0,28729082 logit**, tương ứng hệ số nhân odds **1,33281**. Odds là tỷ số giữa xác suất xảy ra và xác suất không xảy ra. [Danh sách 404 mức cent và hệ số](amount_hole_submission.json)

Đã đối chiếu 299.984 dòng với mẫu chính thức, thứ tự ID, SHA-256 và xác suất Grandmaster EDA đã lưu. Ứng viên tăng xác suất cho 4.448 dòng theo quy tắc trên. Các số đo giữ lại trong báo cáo dùng hệ số học trên tập phát triển trước khi refit. [CSV ứng viên](submission_grandmaster_eda_amount_hole.csv)

## Phạm vi nguồn

- Đọc toàn bộ dòng và cột trong `train.csv`, `test.csv` và `sample_submission.csv` để xác minh nguồn theo SHA-256 và thứ tự. Quét nguyên văn mọi dòng của sáu trường số thập phân trong `train.csv` và `test.csv`. Phép kiểm nhãn dùng 594.966 dòng phát triển và ứng viên qua sàng lọc được chấm một lần trên 104.994 dòng giữ lại.
- Đọc toàn bộ `grandmaster_inputs.py`, `data_signals/BAO_CAO_DAC_DIEM.md`, cùng các mục cặp chữ số cuối trong `data_signals/overlooked_features.py`, `exact_value_results` trong `residual_eda.json`, phần phương pháp và các trường đầu trong `univariate_signal.json`, hàm `build_inputs` trong `followup/run_followup.py`, và mục `grandmaster_eda` trong `parallel_eda_resolution/development_metrics.json`.
- Dùng mười tệp fold trong `grandmaster_10fold_ablation/validation/artifacts`, dự đoán `grandmaster_hierarchical` trong `residual_development.npz` để kiểm tra tái dựng và dự đoán `holdout` trong `selected_predictions.npz` để chấm xác nhận. Mã giao dịch chỉ phục vụ kiểm tra thứ tự dự đoán.
- Tệp ứng viên dùng dự đoán ngoài fold từ mười tệp dưới `grandmaster_refit/full/artifacts` và các tệp nhánh phân cấp tương ứng, rồi kiểm tra dự đoán `test` trong `selected_predictions.npz` và toàn bộ dòng của `submission_grandmaster_eda.csv`.

## Tái chạy

Chạy từ thư mục chứa báo cáo.

```powershell
$python = '..\..\grandmaster_refit\.refit-env\Scripts\python.exe'
& $python -B measurement_eda.py
& $python -B export_amount_hole.py export
& $python -B export_amount_hole.py verify
```
