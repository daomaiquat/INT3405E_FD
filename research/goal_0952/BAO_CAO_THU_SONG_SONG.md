# Ba hướng thử song song trên Grandmaster EDA

**Grandmaster EDA gốc vẫn là lựa chọn chính cho ROC AUC.** LightGBM cây một nút tách là ứng viên thử nghiệm cân bằng nhất của lượt này. AUC holdout đạt **0,95122558**, cao hơn bản gốc **0,00000113**. Phép đảo vai trò cho mức tăng AUC có trọng số **0,00005627** và tăng ở 10/10 fold. Đối chứng giảm biên đơn giản cũng thắng 10/10 fold trong phép này. Mức tăng trên holdout còn nhỏ và khoảng tin cậy bao gồm 0. Kết quả tốt nhất lượt này vẫn cách mục tiêu 0,952 khoảng 0,00077442. [Bản gốc](submission_grandmaster_eda.csv) · Bản LightGBM một nút tách (`goal_0952/parallel_tree_calibration/submission_grandmaster_eda_additive_stumps.csv`)

## Ba giả thuyết

1. **Chi tiết bên trong từng ô phân vị EDA.** Chia đôi các khoảng phân vị cũ cho sáu cặp gồm một biến số liên tục và một biến nhóm. So phần sai lệch dự báo giữa hai nửa sau khi trừ phần chung của ô gốc và biến động chung qua các nhóm. Trọng số co dựa vào độ hỗ trợ của cả hai nửa. So ba cách sử dụng, gồm thêm tổng sáu hiệu chỉnh vào XGBoost, trộn nửa nhánh, và thêm hiệu chỉnh Ridge tuyến tính. [Cấu hình và hệ số](parallel_eda_resolution/development_metrics.json)
2. **Bất thường của hai biến xét chung trong ngữ cảnh.** Chuẩn hóa thứ hạng của điểm velocity × giờ từ giao dịch trước và số tiền × điểm rủi ro merchant. Ước lượng tương quan trên giao dịch hợp lệ theo ngữ cảnh và co nhóm thưa về nhóm cha. Ba đại lượng cho mỗi cặp mô tả độ hiếm chung, sai lệch có hướng và tích điểm thứ hạng. Huấn luyện XGBoost bổ sung sáu cột, sau đó thử thay toàn bộ hoặc một nửa nhánh cây. [Cấu hình các cặp và phép giữ lại](parallel_peer_context/holdout_metrics.json)
3. **Nhánh cây cộng đơn biến.** Dùng LightGBM hai lá, 600 vòng, và giữ 122 cột đầu vào cùng điểm nền. Đặt trước tỷ lệ thay 15% nhánh XGBoost, tương ứng 12,75% log-odds toàn tổ hợp ở holdout và test. LightGBM tối đa năm lá là đối chiếu cấu trúc ở cùng tỷ lệ. [Cấu hình cây và trọng số](parallel_tree_calibration/holdout_metrics.json)

Các thống kê có nhãn dùng fold con để mỗi dòng train nhận đặc trưng tính từ các dòng ngoài fold. Bản trộn ba hướng lấy trung bình ba chênh lệch log-odds so với đối chứng. Thành phần được đặt trước gồm hiệu chỉnh phân vị tuyến tính, thay nửa nhánh ngữ cảnh và cây LightGBM một nút tách. Phép trộn kiểm cùng ID, điểm gốc và chỉ số fold. [Mã trộn](parallel_eda_resolution/combine_paths.py)

## Kết quả kiểm chứng local

| Phương án | ROC AUC holdout | Average Precision | Log loss | Δ AUC đảo vai trò | Fold tăng AUC |
|---|---:|---:|---:|---:|---:|
| Grandmaster EDA gốc | 0,95122445 | 0,53823528 | 0,04383599 | — | — |
| Chi tiết phân vị qua XGBoost | 0,95120508 | 0,53822095 | 0,04383828 | −0,00001371 | 4/10 |
| Chi tiết phân vị, trộn nửa nhánh | 0,95121552 | 0,53826113 | 0,04383702 | −0,00000320 | 5/10 |
| Chi tiết phân vị tuyến tính | 0,95122403 | 0,53822355 | 0,04383642 | −0,00000016 | 3/10 |
| Ngữ cảnh xét chung hai biến | 0,95113668 | **0,53908200** | **0,04380857** | +0,00000914 | 4/10 |
| Ngữ cảnh, trộn nửa nhánh | 0,95118936 | 0,53870944 | 0,04381704 | +0,00000743 | 6/10 |
| **LightGBM một nút tách** | **0,95122558** | 0,53823066 | 0,04383578 | +0,00005627 | 10/10 |
| LightGBM tối đa năm lá | 0,95122496 | 0,53821075 | 0,04383563 | +0,00004212 | 10/10 |
| Giảm 15% phần hiệu chỉnh XGBoost | 0,95121511 | 0,53831307 | 0,04383394 | **+0,00013460** | 10/10 |
| Trộn đều ba hướng | 0,95121425 | 0,53838517 | 0,04382859 | +0,00002271 | 9/10 |

- **Điểm holdout** dùng 104.994 dòng và 1.753 ca gian lận. Toàn bộ tổ hợp được huấn luyện bằng 594.966 dòng còn lại. Khoảng DeLong ghép cặp 95% cho mức tăng của cây một nút là [−0,00001580, +0,00001806]. Dự án đã sử dụng holdout và danh sách cặp EDA trên cùng bộ dữ liệu ở các lượt trước. [Điểm holdout của cây](parallel_tree_calibration/holdout_metrics.json)
- **Đảo vai trò** chấm 594.966 dòng qua 10 fold. Nhánh đang thử học trên 104.994 dòng tham chiếu tách riêng và dựa trên nền Grandmaster đã chừa fold đích. Δ AUC là trung bình có trọng số theo số cặp gian lận–hợp lệ của từng fold. Phép này đo nhánh cây với tập tham chiếu nhỏ hơn và không đo lại toàn bộ tổ hợp. [Phân vị](parallel_eda_resolution/reverse_metrics.json) · [Ngữ cảnh](parallel_peer_context/reverse_metrics.json) · [LightGBM](parallel_tree_calibration/reverse_metrics.json)
- **Đối chứng giảm biên** giữ 85% mức hiệu chỉnh từ cây XGBoost về nền. Trong phép đảo vai trò, cách này thắng LightGBM một nút ở cả 10 fold. Trên holdout, cách giảm biên làm AUC giảm 0,00000934 so với bản gốc, còn cây một nút tăng 0,00000113. Kết quả cho thấy cần xem xét quy mô tham chiếu khi diễn giải mức tăng 10 fold. [Đối chứng giảm biên](parallel_eda_resolution/stump_shrink_audit.json)
- **Ngữ cảnh cải thiện xác suất và thứ tự đầu danh sách.** Bản dùng toàn bộ nhánh mới tăng AP 0,00084672 và giảm log loss 0,00002742. Phần mất AUC tập trung ở nhóm ca gian lận dưới phân vị điểm gốc 90. Khi rà soát 1.050 dòng đầu, bản nửa nhánh tìm 706 ca, so với 704 ở bản gốc. [Đóng góp theo dải điểm](parallel_peer_context/holdout_metrics.json)
- **Bản trộn đều** giảm log loss trên holdout và ở 10/10 fold đảo vai trò. AUC holdout giảm 0,00001019. [Kết quả bản trộn](parallel_eda_resolution/parallel_three_way_metrics.json)

## Submission

Mỗi tệp có 299.984 dòng, đúng thứ tự ID chính thức, hai cột `id,is_fraud` và xác suất trong [0, 1]. Dự đoán được kiểm tra lại từ các tệp lưu cùng SHA-256.

| Phương án | CSV |
|---|---|
| Chi tiết phân vị | XGBoost (`goal_0952/parallel_eda_resolution/submission_grandmaster_eda_resolution_tree.csv`) · Trộn nửa (`goal_0952/parallel_eda_resolution/submission_grandmaster_eda_resolution_half.csv`) · Tuyến tính (`goal_0952/parallel_eda_resolution/submission_grandmaster_eda_resolution_linear.csv`) |
| Ngữ cảnh | XGBoost (`goal_0952/parallel_peer_context/submission_grandmaster_eda_joint_context.csv`) · Trộn nửa (`goal_0952/parallel_peer_context/submission_grandmaster_eda_joint_context_half.csv`) |
| Cây LightGBM | Một nút tách (`goal_0952/parallel_tree_calibration/submission_grandmaster_eda_additive_stumps.csv`) · Tối đa năm lá (`goal_0952/parallel_tree_calibration/submission_grandmaster_eda_five_leaf_trees.csv`) |
| Đối chứng giảm biên | Giảm 15% hiệu chỉnh XGBoost (`goal_0952/parallel_eda_resolution/submission_grandmaster_eda_shrink_toward_grandmaster.csv`) |
| Trộn đều | Ba hướng cố định (`goal_0952/parallel_eda_resolution/submission_grandmaster_eda_parallel_three_way.csv`) |

## Tái lập và phạm vi đọc

Chạy trong thư mục tương ứng của từng nhánh. Dùng Python có sẵn tại `..\..\grandmaster_refit\.refit-env\Scripts\python.exe`. Các lệnh chính của mỗi runner là `development`, `reverse` và `export`. Dùng `verify` ở runner phân vị và cây. Sau khi ba runner có đủ dự đoán, chạy `combine_paths.py` rồi `combine_paths.py --verify-only` trong thư mục phân vị. Chạy `audit_stump_shrink.py` và `audit_stump_shrink.py --export` để tái lập đối chứng giảm biên. [Runner phân vị](parallel_eda_resolution/resolution_eda.py) · [Runner ngữ cảnh](parallel_peer_context/run_joint_context.py) · [Runner cây](parallel_tree_calibration/compare_offset_trees.py) · [Đối chứng giảm biên](parallel_eda_resolution/audit_stump_shrink.py)

- Đã đọc toàn bộ báo cáo vòng trước, gồm giả thuyết, kiểm chứng và danh sách tệp, cùng các mục mã hóa cặp, chia fold con và huấn luyện XGBoost trong mã Grandmaster EDA. [Báo cáo vòng trước](grandmaster_eda_risk/BAO_CAO_TOI_UU_EDA.md) · [Mã hóa gốc](residual_models.py)
- Đã đọc cấu hình cặp và phần chuẩn hóa có điều kiện trong nhánh ngữ cảnh, công thức thay nhánh và cấu hình LightGBM, các phần tái dựng và kiểm đối chứng của runner mới. [Đặc trưng ngữ cảnh](parallel_peer_context/joint_features.py) · [Cấu hình cây](parallel_tree_calibration/compare_offset_trees.py)
- Toàn bộ dòng và cột trong ba CSV chính thức được đọc khi huấn luyện và xác minh dữ liệu. Các chỉ số được đối chiếu trên toàn bộ ID trong từng tập kiểm, và mỗi submission được xác minh trên mọi dòng test.
