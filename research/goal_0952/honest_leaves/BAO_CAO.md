# Grandmaster EDA với ba mô hình XGBoost học trên các tập con

**Ứng viên đề xuất cho lượt này là `amount_hole_subforest`.** Bản này đạt ROC AUC **0,95133949**, Average Precision **0,53868408** và log loss **0,04379804** trên holdout 104.994 giao dịch. So với bản chỉ hiệu chỉnh số tiền, AUC tăng **0,00002365** với khoảng tin cậy DeLong ghép cặp 95% từ **−0,00001987 đến +0,00006716**. Holdout đã được dùng trong các lượt nghiên cứu trước. Mốc **0,952** vẫn chưa đạt và điểm Kaggle của lượt này chưa được đo. [Số liệu và tên tệp submission](amount_hole_subforest_metrics.json)

## Thiết kế thí nghiệm

Grandmaster EDA gốc trộn biên dự đoán từ ba nhánh với tỷ lệ **85% XGBoost, 10% LightGBM và 5% spline**. Biên dự đoán (logit) là điểm trước khi đổi sang xác suất bằng sigmoid. Thí nghiệm dùng lại đúng bộ điểm Grandmaster, bộ đặc trưng D03 và 13 đặc trưng EDA của nhánh XGBoost. D03 gồm 109 cột thuộc nhóm dữ liệu gốc, target encoding, phần lẻ cent và chữ số. Phần EDA thêm 12 cặp có mã hóa phần dư và một đặc trưng chữ số của khoảng cách. Mã hóa phần dư tóm tắt phần rủi ro mà điểm Grandmaster dự kiến chưa giải thích. [Báo cáo Grandmaster EDA gốc](../BAO_CAO_GRANDMASTER_EDA_GOC.md) · [Mã đặc trưng D03](../../../s6e11_gpu/plan_runs/features.py) · [Mã thí nghiệm](run_honest_leaves.py)

- **Trộn tập con.** Chia tập fit thành ba phần phân tầng theo nhãn với seed 18017. Huấn luyện ba mô hình, mỗi mô hình dùng hai phần để học cấu trúc và điểm lá của 224 cây XGBoost sâu tối đa 3. Mỗi mô hình xây lại đặc trưng Mã hóa phần dư bằng cross-fitting, tức là dùng các phần khác để tạo đặc trưng có nhãn cho mỗi dòng training. Lấy trung bình biên của ba mô hình rồi thay 25% nhánh XGBoost gốc, tương ứng **21,25%** tổng trọng số. Công thức là `Grandmaster EDA + 0,2125 × (XGBoost tập con − XGBoost gốc)`.
- **Ước lượng lại điểm lá.** Sau khi học cây, dùng phần thứ ba để tính phần dư `G = sum(y − p)` và độ cong `H = sum(p × (1 − p))` theo lá. Trừ phần dư chung, dùng L2 = 80 và giới hạn hiệu chỉnh mỗi lá trong ±0,15 logit. Chỉ dùng lá có ít nhất 80 giao dịch và `H ≥ 8`. Lấy trung bình hiệu chỉnh của 224 lá mà giao dịch đi qua, nhân 0,5. Đây là một bước hiệu chỉnh duy nhất ở tầng XGBoost. Các biên Grandmaster tầng dưới vẫn lấy từ kho ngoài fold đã lưu.
- **Ghép với số tiền.** Cộng lại đúng mức điều chỉnh đã có cho **404** mức tiền xuất hiện thưa hơn hai mức liền kề. Dùng hệ số **+0,27198694 logit** đã học trên tập phát triển để đo trên holdout và **+0,28729082 logit** từ toàn bộ train để xuất test. Hệ số được áp dụng cho **1.644** dòng holdout và **4.448** dòng test. [Công thức và số liệu lượt EDA trước](../deep_eda_measurement/BAO_CAO.md) · [Danh sách mức tiền và hệ số](../deep_eda_measurement/amount_hole_submission.json) · [Mã ghép hai ứng viên](combine_amount_hole.py)

## Kết quả holdout

Holdout phân tầng với seed 4242 gồm 104.994 giao dịch, trong đó 1.753 là gian lận. Tập phát triển có 594.966 dòng. ROC AUC đo khả năng xếp hạng gian lận, Average Precision tóm tắt đường precision–recall. Log loss và Brier score đo chất lượng xác suất với giá trị thấp hơn là tốt hơn. [Điểm và khoảng tin cậy](development_metrics.json) · [Đối chứng giảm biên](matched_shrink_audit.json) · [Kết quả ghép](amount_hole_subforest_metrics.json)

| Biến thể | ROC AUC | Δ AUC so với gốc | Average Precision | Log loss | Brier |
|---|---:|---:|---:|---:|---:|
| Grandmaster EDA gốc | 0,95122445 | — | 0,53823528 | 0,04383599 | 0,01063173 |
| Giảm 25% phần chênh của nhánh XGBoost về Grandmaster phân cấp | 0,95120613 | −0,00001832 | 0,53834013 | 0,04383335 | 0,01062927 |
| Trộn ba mô hình tập con | 0,95124678 | +0,00002233 | 0,53839526 | 0,04382808 | 0,01062966 |
| Trộn ba mô hình tập con có hiệu chỉnh lá | 0,95124700 | +0,00002255 | 0,53839858 | 0,04382819 | 0,01062970 |
| Chỉ cộng phần hiệu chỉnh lá vào bản gốc | 0,95122367 | −0,00000078 | 0,53823421 | 0,04383658 | 0,01063192 |
| Chỉ hiệu chỉnh số tiền | 0,95131585 | +0,00009140 | 0,53852282 | 0,04380602 | 0,01062591 |
| **Hiệu chỉnh số tiền + trộn ba mô hình tập con** | **0,95133949** | **+0,00011504** | **0,53868408** | **0,04379804** | **0,01062383** |

Khi giữ nguyên ba cấu trúc cây, hiệu chỉnh lá tăng AUC **0,00000022**, khoảng tin cậy từ **−0,00000109 đến +0,00000153**, và log loss tăng **0,00000012**. Phần cải thiện quan sát được chủ yếu đi cùng phép trộn cây. Theo dải điểm gốc của giao dịch gian lận, trộn cây đóng góp dương ở nhóm dưới phân vị 75. Nó kéo giảm AUC khoảng **0,00001269** ở nhóm phân vị 75–90, nơi điều chỉnh số tiền từng đóng góp dương khoảng **0,00006346**. Điều này dẫn đến phép thử ghép hệ số cố định. [Phân tích theo dải điểm](matched_shrink_audit.json) · [Báo cáo hiệu chỉnh số tiền](../deep_eda_measurement/BAO_CAO.md)

Trong lịch sử của cùng holdout, bản hiệu chỉnh bậc hai theo bất đồng đạt AUC **0,95134001** và log loss **0,04381021**. Bản ghép lượt này có điểm AUC thấp hơn khoảng **0,00000051** và log loss thấp hơn khoảng **0,00001217**. [Bảng điểm lượt trước](../score_space/holdout_metrics.json)

## Kiểm tra đảo vai trò trên 10 fold

Ở phép đo này, 104.994 dòng tham chiếu được dùng để học tầng cuối. Mỗi fold trong 594.966 dòng phát triển được chấm riêng bằng nhánh XGBoost có biên Grandmaster. Weighted Δ AUC lấy trọng số theo số cặp gian lận–hợp lệ của từng fold. Thiết kế này kiểm tra độ nhạy khi tập fit của tầng cuối nhỏ hơn. Bản ghép hiệu chỉnh số tiền mới chỉ được đánh giá trên holdout chung. [Điểm từng fold](reverse_metrics.json) · [Đối chứng giảm biên](matched_shrink_audit.json)

| Biến thể | Weighted Δ AUC | Fold tăng AUC | Fold giảm log loss |
|---|---:|---:|---:|
| Giảm 25% phần chênh của nhánh XGBoost về Grandmaster phân cấp | +0,00020699 | 10/10 | 10/10 |
| Trộn ba mô hình tập con | +0,00014657 | 9/10 | 10/10 |
| Trộn ba mô hình tập con có hiệu chỉnh lá | +0,00014726 | 9/10 | 10/10 |
| Chỉ cộng phần hiệu chỉnh lá vào bản gốc | +0,00000141 | 5/10 | 5/10 |

Khi tầng cuối học từ ít dữ liệu hơn, giảm biên về Grandmaster phân cấp cho kết quả tốt nhất. Trên holdout với tập phát triển lớn hơn, giảm biên làm AUC giảm **0,00001832**. Cả hai phép đo cùng cho thấy phần đóng góp riêng của hiệu chỉnh lá quá nhỏ để ưu tiên. Khi muốn thử riêng tác động của việc trộn cây, dùng `subforest_blend`. Điểm số của bản ghép cần xác nhận trên dữ liệu độc lập.

## Bốn tệp submission

Đã huấn luyện lại tầng XGBoost trên toàn bộ 699.960 dòng thông qua ba tập con. Mỗi CSV có **299.984** dòng, đúng thứ tự ID và cột `id,is_fraud` theo mẫu Kaggle. Xác suất sau khi đọc lại khớp với điểm đã lưu trong sai số tuyệt đối `1e-14`. Điểm và SHA-256 của bản Grandmaster gốc khớp với kho. [Kiểm toán](audit_honest_leaves.py) · [Thông tin full refit](full_metrics.json)

| CSV | Chi tiết |
|---|---|
| `submission_grandmaster_eda_amount_hole_subforest.csv` | [Số liệu và SHA-256](amount_hole_subforest_metrics.json) |
| `submission_grandmaster_eda_subforest_blend.csv` | [Manifest](subforest_blend_manifest.json) |
| `submission_grandmaster_eda_honest_leaf_blend.csv` | [Manifest](honest_leaf_blend_manifest.json) |
| `submission_grandmaster_eda_honest_leaf_overlay.csv` | [Manifest](honest_leaf_overlay_manifest.json) |

## Tái lập

Chạy trong thư mục `honest_leaves`. Môi trường tại workspace dùng Python 3.12 và XGBoost 3.4.1 với CUDA. Mã dùng dữ liệu chính thức từ `data/` và mặc định đọc cache từ `research/`. Ở workspace đã chạy thí nghiệm, cache nằm ở thư mục cha của `INT3405E_FD`, nên đặt biến môi trường như ví dụ dưới đây. Đổi đường dẫn nếu cache nằm ở nơi khác. [Ca kiểm tra hàm hiệu chỉnh và trọng số](test_honest_leaves.py)

```powershell
$env:GRANDMASTER_CACHE_PROJECT = (Resolve-Path '..\..\..\..').Path
$python = 'python'
& $python -B -m unittest discover -v
& $python -B run_honest_leaves.py development --device cuda
& $python -B run_honest_leaves.py reverse --device cuda
& $python -B run_honest_leaves.py export --device cuda
& $python -B audit_honest_leaves.py
& $python -B combine_amount_hole.py
```

Nguồn đối chiếu gồm nhóm đặc trưng trong `s6e11_gpu/plan_runs/features.py`, hàm `nested_residual_train` trong cache `residual_models.py`, hàm `build_inputs` trong cache `followup/run_followup.py`, phần công thức và kết quả holdout trong báo cáo hiệu chỉnh số tiền, cùng mục `variants.quadratic` của `score_space/holdout_metrics.json`. Các số đo mới lấy từ mục `variants`, `honest_vs_same_structure`, `folds` nếu có và các mục đối chứng của những tệp JSON liên kết ở trên.
