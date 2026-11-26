# S6E11 · Kiểm tra các đặc điểm dữ liệu có thể bị bỏ sót

**Hồ sơ đồng cảnh cải thiện chất lượng xác suất trên cả hai phép đo.** AUC của toàn tổ hợp là 0,95129283, thấp hơn 0,00001941 so với bản 0,95131224. Tôi ưu tiên bản cũ cho mục tiêu AUC Kaggle và xuất bản hồ sơ đồng cảnh để đối chiếu. CSV hồ sơ đồng cảnh (`goal_0952/data_signals/submission_peer_profiles.csv`) · Bản đang dẫn theo AUC (`goal_0952/followup/submission_context.csv`)

## Kết quả

| Nhánh được bổ sung | ROC AUC giữ lại | Average Precision | Log-loss |
|---|---:|---:|---:|
| Đối chứng | 0,95131224 | 0,53828373 | 0,04382592 |
| Quan hệ số và thời gian | 0,95129338 | 0,53831928 | 0,04382754 |
| Hồ sơ đồng cảnh | 0,95129283 | **0,53861055** | **0,04381643** |
| Cặp chữ số cuối | 0,95129325 | 0,53806553 | 0,04383836 |
| Gộp cả ba | 0,95125725 | 0,53824256 | 0,04383429 |

- Hồ sơ đồng cảnh tăng AP **0,00032682**, giảm log-loss **0,00000949** và giảm Brier **0,00000439** trên tập giữ lại. Khoảng DeLong ghép cặp 95% của Δ AUC là [−0,00009551, +0,00005670]. Tập giữ lại đã được dùng để chọn mô hình trong dự án.
- Trên phép kiểm đảo vai trò, hồ sơ đồng cảnh giảm log-loss ở **7/10 fold** và tăng AUC ở **5/10 fold**. Log-loss gộp giảm từ 0,04439411 xuống 0,04439171. AP tăng từ 0,52995291 lên 0,52998627. Mức thay đổi AUC có trọng số theo số cặp dương–âm khoảng −0,000007. [Chấm 10 fold](reversed_metrics.json)
- Khi xếp 1.050 giao dịch đầu để rà soát trên tập giữ lại, hai bản đều tìm được **706 ca**. Với 5.250 và 10.500 giao dịch, bản hồ sơ đồng cảnh tìm được 1.280 và 1.470 ca, so với 1.281 và 1.470 của bản cũ. [Ngân sách rà soát](peer_diagnostics.json)

### Nhóm mà log-loss cải thiện rõ hơn

Trong nhóm 11.059 giao dịch có số lần thử CVV cao khi so trong cùng hình thức xác thực, thiết bị và kênh, bản hồ sơ đồng cảnh giảm log-loss 0,00008126. Nhóm này chiếm 10,53% tập giữ lại và 10,51% test. Với nhóm 10.702 giao dịch có điểm merchant cao khi so trong cùng loại merchant, hình thức xác thực và cờ lệch địa chỉ, log-loss giảm 0,00006570. Các nhóm này có thể giao nhau. Các số đo mô tả tác động của toàn bộ nhánh hồ sơ đồng cảnh trên những dòng đó. [Chi tiết đuôi phân phối](peer_diagnostics.json)

## Xây dựng giả thuyết từ dữ liệu

Tôi xét 31 mẫu trên 594.966 dòng phát triển. Với hai nửa phân tầng, mỗi lần ước lượng hiệu chỉnh trên một nửa rồi chấm log-loss trên nửa còn lại. Đối chứng là điểm Grandmaster ngoài fold của tập phát triển. Không có mẫu nào làm giảm log-loss theo cả hai chiều sàng lọc. Đây là phép thử hiệu chỉnh theo nhóm. Thử nghiệm mô hình sau đó cho phép cây học cách kết hợp các tín hiệu với những đặc trưng đã có. [Kết quả sàng lọc](data_audit.json)

| Tình huống | Dòng phát triển | Ca thực tế | Grandmaster dự kiến |
|---|---:|---:|---:|
| Khoảng cách giao dịch dưới 24 giờ và số đếm 24 giờ bằng 0 | 15.169 | 35 | 36,80 |
| Khoảng cách giao dịch ít nhất 24 giờ và số đếm vẫn dương | 13.574 | 199 | 205,88 |
| Số tiền lớn hơn số dư | 6.349 | 159 | 166,40 |
| Tuổi suy ra ở lúc mở thẻ dưới 18 | 35.910 | 146 | 138,46 |
| Cờ giao dịch nước ngoài khác cờ lệch quốc gia IP | 13.881 | 1.300 | 1.289,51 |

Hai cột thời gian và số đếm có thể dùng mốc khác nhau. Các giá trị thời điểm giao dịch trước và tuổi lúc mở thẻ là phép suy ra từ các trường được cung cấp. Tôi dùng chúng làm đặc trưng thống kê.

| Nhóm thử | Nội dung |
|---|---|
| Quan hệ số và thời gian | 16 biến gồm chênh lệch số tiền và số dư, tần suất theo khoảng thời gian, khoảng cách giữa giao dịch so với số đếm, tuổi suy ra khi mở thẻ và thời điểm giao dịch trước theo chu kỳ ngày và tuần. |
| Hồ sơ đồng cảnh | So tốc độ, thời gian, số tiền, khoảng cách, điểm merchant, số lần thử CVV và tranh chấp với giao dịch trong cùng ngữ cảnh từ 2–3 trường. Dùng phân vị, độ hiếm và chênh lệch mật độ theo lớp. Nhóm thưa quay về ngữ cảnh rộng hơn. |
| Cặp chữ số cuối | Tạo 15 cặp từ 6 trường số. Với mỗi cặp, tính độ hiếm so với giả định độc lập và phần sai lệch rủi ro sau khi đã trừ ảnh hưởng của từng chữ số. |
| Gộp cả ba | Bổ sung đồng thời ba nhóm để kiểm tra tác động cộng dồn. |

## Phép kiểm mô hình

Đối chứng là bản **0,95131224**. Tôi dựng lại nhánh XGBoost dùng mật độ mở rộng với 224 vòng và cùng cấu hình, rồi kiểm tra dự đoán bằng tệp đã lưu. Mỗi ứng viên thay đúng nhánh đó, chiếm **42,5% log-odds toàn tổ hợp**. Cây học bằng mất mát xác suất. Các bộ mã hóa có nhãn trong phần train được tính qua 5 fold con. [Cách tạo đặc trưng](overlooked_features.py) · [Kết quả tập giữ lại](holdout_metrics.json)

Kiểm tra đảo vai trò dùng 10 fold của 594.966 dòng phát triển. Mỗi fold lấy dự đoán nền Grandmaster từ mô hình đã chừa fold đó. Tầng gộp và cây học sai lệch dùng 104.994 dòng tham chiếu tách riêng. Phép kiểm cô lập nhánh XGBoost, với ít dữ liệu huấn luyện cho cây hơn toàn bộ tổ hợp. [Kết quả 10 fold](reversed_metrics.json)

Chạy từ `goal_0952/data_signals`.

```powershell
$python = '..\..\grandmaster_refit\.refit-env\Scripts\python.exe'
& $python -B run_signals.py development
& $python -B run_signals.py reverse --candidate peer_profiles
& $python -B run_signals.py export --candidate peer_profiles
& $python -B peer_diagnostics.py
& $python -B -m unittest discover -v
```

## Phạm vi nguồn đã đọc

Đã đọc toàn bộ `BAO_CAO_DO_ON_DINH.md` của lượt trước, toàn bộ dòng và cột của ba CSV chính thức, các đoạn mã tạo fold và dựng lại dự đoán Grandmaster, cùng cấu hình nhánh mật độ trước đó. Phần thống kê và tìm tín hiệu mới dùng 594.966 dòng phát triển. Số liệu giữ lại và 10 fold dùng đúng các ID liệt kê trong tệp dự đoán của từng lượt. [Báo cáo trước](../robust_validation/BAO_CAO_DO_ON_DINH.md)
