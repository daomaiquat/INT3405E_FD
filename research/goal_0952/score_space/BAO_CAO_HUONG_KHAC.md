# S6E11 · Học từ dự đoán có sẵn

## Hướng thử

Lượt này giữ bộ đặc trưng giao dịch đã có. Thử nghiệm thứ nhất học trên bảy đầu ra Grandmaster, gồm hai GLM, ba cây học sai lệch, XGBoost và EBM. Tôi tìm phần sai lệch có thể dự báo từ chênh lệch giữa các mô hình, sau khi loại phần giải thích tuyến tính trên điểm cũ. So sánh hai cách biểu diễn, gồm tích bậc hai giữa các độ lệch và mô hình trơn dựa trên khoảng cách đến 48 cụm điểm. Cả hai dùng hàm mất mát xác suất và co hệ số. Khi áp lên tổ hợp hiện tại, mức thay đổi log-odds được giới hạn khoảng 0,09.

Trên 594.966 dòng phát triển, nhóm từ phân vị 80 đến 95 về mức bất đồng chứa 2.808 ca gian lận, so với 2.745,86 ca Grandmaster dự kiến. Nửa có mức bất đồng thấp nhất ghi 3.480 ca, dưới 3.553,99 ca dự kiến. Đây là quan sát mô tả trên tập phát triển. [Thống kê theo mức bất đồng](holdout_metrics.json)

Thử nghiệm thứ hai dùng **80% nhãn thật và 20% xác suất Grandmaster ngoài fold** làm mục tiêu xác suất cho cây XGBoost. Xác suất dùng làm nhãn mềm lấy từ mô hình được huấn luyện local bằng dữ liệu cuộc thi. Cây giữ nguyên đặc trưng, 224 vòng và bộ tham số. Nhãn mềm được so với một đối chứng chỉ giảm biên log-odds của cây về gần nền Grandmaster theo cùng tỷ lệ.

## Kết quả thử nghiệm theo bất đồng

| Phương án | ROC AUC giữ lại | Average Precision | Log-loss |
|---|---:|---:|---:|
| Tổ hợp hiện tại | 0,95131224 | 0,53828373 | 0,04382592 |
| Hiệu chỉnh bậc hai | **0,95134001** | **0,53864400** | **0,04381021** |
| Hiệu chỉnh theo cụm trên điểm trung bình | 0,95130161 | 0,53844498 | 0,04382359 |
| Hiệu chỉnh theo cụm từng fold rồi lấy trung bình | 0,95130037 | 0,53839768 | 0,04382380 |

Bản bậc hai tăng AUC **0,00002777** trên 104.994 dòng giữ lại. Khoảng tin cậy ghép cặp 95% là [−0,00004221, +0,00009775]. Tập giữ lại từng được dùng trong quá trình chọn mô hình của dự án. [Chi tiết đối chiếu](holdout_metrics.json)

Trong phép kiểm 10 fold đảo vai trò, nhánh bậc hai tăng AUC ở 4/10 fold và giảm log-loss ở 6/10 fold. AUC có trọng số theo số cặp giảm khoảng **0,00002963**. Log-loss gộp giảm từ 0,04433998 xuống 0,04433802. Tầng hiệu chỉnh học trên 104.994 dòng tham chiếu tách riêng. Phép kiểm này cô lập phần XGBoost và Grandmaster, với cấu hình tổ hợp khác phép đo toàn pipeline. [Kiểm tra theo fold](reversed_metrics.json)

CSV hiệu chỉnh theo bất đồng (`goal_0952/score_space/submission_quadratic.csv`) · [Manifest và SHA-256](submission_manifest.json)

## Kết quả thử nghiệm nhãn mềm

| Phương án | AUC toàn tổ hợp giữ lại | Log-loss giữ lại | Δ AUC có trọng số, 10 fold đảo vai trò |
|---|---:|---:|---:|
| Đối chứng | 0,95131224 | 0,04382592 | — |
| Giảm biên cây về nền Grandmaster | 0,95129698 | 0,04382538 | +0,00011889 |
| 80% nhãn thật + 20% xác suất ngoài fold | 0,95129512 | 0,04382499 | +0,00012738 |

Nhãn mềm tăng AUC và giảm log-loss ở **10/10 fold** khi nhánh XGBoost học trên 104.994 dòng tham chiếu. Đối chứng chỉ giảm biên hậu huấn luyện cũng tăng cả hai chỉ số ở 10/10 fold và đạt khoảng 93% mức tăng AUC có trọng số của nhãn mềm. Khi nhánh XGBoost học trên 594.966 dòng và được đặt trong toàn bộ tổ hợp, AUC giảm 0,00001712. Kết quả ủng hộ điều chuẩn mạnh hơn trong phép kiểm dùng tập tham chiếu nhỏ. [Kết quả giữ lại](soft_holdout_metrics.json) · [Kết quả 10 fold](soft_reversed_metrics.json) · CSV nhãn mềm (`goal_0952/score_space/submission_soft_targets.csv`)

**Lựa chọn chính cho Kaggle là bản 0,95131224.** Bản bậc hai là ứng viên thử nghiệm với AUC và log-loss tốt hơn trên tập giữ lại, cùng mức giảm log-loss nhỏ trong phép kiểm 10 fold. Mức tăng AUC chưa lặp lại qua hai phép đo. Bản lựa chọn chính (`goal_0952/followup/submission_context.csv`)

## Tái lập và phạm vi đọc

Chạy từ `goal_0952/score_space`.

```powershell
$python = '..\..\grandmaster_refit\.refit-env\Scripts\python.exe'
& $python -B score_space.py development
& $python -B score_space.py reverse
& $python -B score_space.py export --method quadratic
& $python -B soft_targets.py development
& $python -B soft_targets.py reverse
& $python -B soft_targets.py export
& $python -B -m unittest discover -v
```

Đã đọc toàn bộ ba CSV chính thức, toàn bộ `risk_gated_meta.py`, các mục trong `meta_improvements/metrics.json` về tầng gộp và những đoạn tương ứng trong runner Grandmaster và phép kiểm đảo vai trò. Số liệu lượt này dùng toàn bộ ID và xác suất trong các tệp dự đoán của hai phép đo. [Thử tầng gộp trước đây](../../grandmaster_10fold_ablation/meta_improvements/metrics.json)
