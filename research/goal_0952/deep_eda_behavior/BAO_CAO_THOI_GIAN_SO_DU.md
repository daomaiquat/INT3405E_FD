# EDA sâu về thời gian, số tiền và số dư

**Cả hai ứng viên vượt vòng phát triển đều làm giảm AUC và tăng log-loss trên tập giữ lại.** Hai ứng viên là chuyển ngày hoặc vòng tuần kết hợp kênh giao dịch, và hiệu số `account_balance_usd − amount_usd` kết hợp loại thẻ. Bảy giả thuyết còn lại đều làm tăng log-loss ở cả hai nửa phát triển. [Kết quả chi tiết](behavior_audit.json)

## Sàng lọc trên tập phát triển

Tôi dùng điểm Grandmaster phân cấp ngoài fold (out-of-fold, mỗi dòng được chấm bởi mô hình đã chừa fold của dòng đó). Tập phát triển gồm 594.966 dòng với 9.936 ca gian lận. Tôi chia các fold lẻ và chẵn thành hai nửa, mỗi nửa 297.483 dòng và 4.968 ca. Mỗi chiều học một hiệu chỉnh theo nhóm trên nửa này và đánh giá trên nửa kia. Đối chứng chỉ hiệu chỉnh theo sáu dải xác suất cố định. Các hiệu chỉnh theo nhóm và theo nhóm × dải xác suất được co về 0 khi dữ liệu thưa. AUC đo độ đúng khi xếp hạng giao dịch. Log-loss đánh giá chất lượng xác suất, với chênh lệch dương trong bảng nghĩa là ứng viên giảm lỗi.

| Giả thuyết | Δ AUC gộp | Log-loss giảm × 10⁻⁶, nửa lẻ / chẵn | TV phát triển–test |
|---|---:|---:|---:|
| Cặp khung 4 giờ hiện tại và giao dịch trước | −0,00012311 | −55,43 / −64,37 | 0,333% |
| Chuyển ngày/vòng tuần × kênh | **+0,00003343** | **+1,69 / +0,35** | 0,300% |
| Giao dịch đêm qua nửa đêm × số giao dịch 24 giờ | −0,00003654 | −11,25 / −10,53 | 0,142% |
| Khoảng cách thời gian × số tiền × merchant mới | −0,00005888 | −41,94 / −42,88 | 0,343% |
| Số tiền × số giao dịch 24 giờ / số dư | −0,00011094 | −48,99 / −50,01 | 0,088% |
| Số dư trừ số tiền × loại thẻ | +0,00000270 | **+0,79 / +2,37** | 0,258% |
| Tỷ lệ số tiền/số dư × xác thực × lệch địa chỉ | −0,00012767 | −42,09 / −40,32 | 0,478% |
| Khoảng cách thời gian × khoảng cách từ nhà × kênh | −0,00010211 | −37,12 / −44,30 | 0,510% |
| Tỷ lệ số tiền/số dư × tuổi thẻ | −0,00004419 | −37,63 / −36,56 | 0,263% |

TV (total variation) đo chênh lệch tỷ trọng giữa hai phân phối nhóm. Cả 299.984 dòng test đều có nhóm xuất hiện ít nhất một lần trong phần phát triển, theo từng giả thuyết. Với hai ứng viên, số dòng test rơi vào nhóm có dưới 150 dòng tham chiếu lần lượt là 0 và 108. Mức dịch chuyển tỷ trọng nhỏ, còn các quan hệ với nhãn gian lận chỉ đo được trên train.

## Kiểm tra ứng viên trên tập giữ lại

Điều kiện chọn đặt trước yêu cầu hơn 99,5% dòng test có nhóm được hỗ trợ, TV dưới 2%, và một trong hai tiêu chí hiệu quả. Tiêu chí AUC yêu cầu mức tăng gộp ít nhất 0,00001, AUC tăng trên cả hai nửa và log-loss gộp cải thiện hoặc bằng đối chứng. Tiêu chí log-loss yêu cầu cải thiện trên cả hai nửa và mức giảm gộp ít nhất 0,000001. AUC ở hai nửa tăng lần lượt 0,00006503 và 0,00000672 với chuyển ngày/vòng tuần, và 0,00000575 và 0,00000995 với hiệu số dư. Hai hiệu chỉnh được khóa bằng dữ liệu phát triển rồi mới mang sang 104.994 dòng giữ lại với 1.753 ca gian lận. Tập giữ lại từng được dùng để đánh giá và chọn các mô hình trước đó.

| Đối chứng trên tập giữ lại | Δ AUC chuyển ngày/vòng tuần | Log-loss giảm × 10⁻⁶ | Δ AUC hiệu số dư | Log-loss giảm × 10⁻⁶ |
|---|---:|---:|---:|---:|
| Grandmaster phân cấp đã hiệu chỉnh theo dải | −0,00013841 | −29,76 | −0,00009572 | −22,42 |
| Grandmaster phân cấp thô AUC 0,95094893 | −0,00012921 | −29,89 | −0,00009475 | −21,95 |
| Grandmaster EDA AUC 0,95122445 | −0,00013785 | −33,59 | −0,00009981 | −22,11 |
| Tổ hợp mật độ AUC 0,95131224 | −0,00013765 | −33,24 | −0,00009960 | −22,45 |

Tôi cộng cùng một hiệu chỉnh vào log-odds trên từng dòng của cả bốn đối chứng. Điểm Grandmaster EDA lấy từ trường `holdout` của `selected_predictions.npz` và được kiểm tra thứ tự bằng `holdout_ids`. Với Grandmaster EDA, khoảng tin cậy DeLong ghép cặp 95% cho Δ AUC lần lượt là [−0,00031811, +0,00004241] và [−0,00020525, +0,00000563]. Cả hai chứa 0. Điểm ước lượng đều giảm và thiếu bằng chứng để ưu tiên thử huấn luyện lại hai đặc trưng này. [Cấu hình, AP và khoảng tin cậy](behavior_audit.json)

### Hai nhóm có sai lệch mô tả

- Với thẻ Visa và `account_balance_usd − amount_usd` trong [0, 100) USD, hai nửa phát triển có **26/672** và **21/666** ca gian lận, so với **17,82** và **16,90** ca theo đối chứng. Nhóm chiếm 0,225% ở cả phát triển và test. Số ca trên tập giữ lại là **6/241**, thấp hơn **7,17** ca dự kiến từ Grandmaster EDA.
- Với kênh Contactless và giao dịch trước nằm phía bên kia ranh giới mã tuần, số ca ở hai nửa là **17/1.675** và **15/1.601**, so với **20,79** và **19,07** ca theo đối chứng. Tỷ trọng là 0,551% phát triển và 0,516% test. Số ca trên tập giữ lại là **7/612**, so với **7,79** ca dự kiến từ Grandmaster EDA.

Các nhóm trên được chọn sau khi xem sai lệch của nhiều ô trên tập phát triển. Số ca và kỳ vọng giúp khoanh vùng khi phân tích lỗi.

## Giả định và phạm vi nguồn đã đọc

- Giờ hiện tại nguyên đại diện khoảng `[h,h+1)`. Thời gian kể từ lần trước được xét với độ bất định ±0,005 giờ do làm tròn. Khi khoảng suy ra cắt ranh giới, biến giữ trạng thái “chưa xác định”. Mã ngày 0–6 được xem là một vòng tuần tự để suy ra ranh giới mã 0. Hiệu `số dư − số tiền` và tích số tiền với số đếm 24 giờ là phép biến đổi thống kê từ các cột hiện có.
- Đã đọc toàn bộ dòng và cột của `train.csv`, `test.csv` và `sample_submission.csv`. Dùng `id` để xác minh thứ tự và khớp kho điểm. Đặc trưng chỉ lấy từ các cột giao dịch. Đã đối chiếu SHA-256 của cả ba tệp với manifest.
- Đã đọc toàn bộ `goal_0952/data_signals/BAO_CAO_DAC_DIEM.md`, `goal_0952/followup/BAO_CAO_FOLLOWUP.md` và `goal_0952/BAO_CAO_THU_SONG_SONG.md`. Đã đọc phần đặc trưng vật lý và hồ sơ đồng cảnh trong `overlooked_features.py`, danh sách cặp trong `parallel_peer_context/joint_features.py`, phần dựng điểm trong `grandmaster_inputs.py` và `run_followup.py`, cùng hàm đọc điểm và DeLong trong `evaluate_and_export.py`.
- Nhãn dùng để chấm hai nửa thuộc các tập dòng rời nhau. Các mô hình nền trong kho điểm ngoài fold chia sẻ một phần dữ liệu huấn luyện. Đối chứng hiệu chỉnh dải tự nó làm AUC phát triển giảm 0,00004444 và log-loss tăng 0,00000731 so với điểm Grandmaster thô. Vì vậy tôi kiểm tra các hiệu chỉnh bị khóa trên nhiều điểm nền của cùng tập giữ lại.

Chạy lại từ `goal_0952/deep_eda_behavior`.

```powershell
& '..\..\grandmaster_refit\.refit-env\Scripts\python.exe' -B audit_behavior.py
```
