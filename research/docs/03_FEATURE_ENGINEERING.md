# Thiết kế đặc trưng theo từng giai đoạn

**Target encoding và mã hóa chữ số tạo mức tăng lớn nhất trong các lượt ablation GPU.** Grandmaster tiếp tục khai thác chúng bằng spline và mã hóa nhiều cấp, rồi dùng mô hình cây để học phần sai lệch còn lại. Grandmaster EDA thêm **12 cặp tương tác và chữ số cuối của khoảng cách** vào nhánh XGBoost. Lượt EDA sâu dùng **một cờ từ 404 mức tiền thưa cục bộ** để hiệu chỉnh điểm, đưa ROC AUC giữ lại lên **0,95131585**. [Bảng so sánh GPU](../../s6e11_gpu/plan_runs/published/full/comparison.md) · [Grandmaster](<../Model tham khảo/Grandmaster/notebook.ipynb>) · [Báo cáo 0,952](../goal_0952/BAO_CAO_0952.md) · [EDA sâu](../goal_0952/BAO_CAO_EDA_SAU.md)

## Baseline và ranh giới huấn luyện

Baseline giữ 13 cột số, 6 cờ và 5 cột phân loại. Hồi quy logistic dùng one-hot, tức một cột 0/1 cho mỗi mức phân loại, và thêm `log1p(max(x,0))` cho `amount_usd`, `account_balance_usd`. Cây dùng dữ liệu số thô và biểu diễn phân loại theo cấu hình, gồm one-hot hoặc phân loại gốc của CatBoost. Bộ chuẩn hóa và bảng mã one-hot của baseline được fit trên phần dữ liệu huấn luyện tương ứng. `id` dùng để đối chiếu dòng. [Mã ma trận baseline](../../src/features.py) · [Bộ cấu hình](../../configs/default.json)

| Vai trò | Phạm vi học đặc trưng |
|---|---|
| `T` là train của fold ngoài | Với đặc trưng dùng nhãn, chia 5 fold con phân tầng. Mỗi dòng nhận giá trị từ bảng ước lượng trên các fold con còn lại. Đây là cross-fit, tức mã hóa chéo theo phần giữ lại. |
| `V` và test | Tra bảng đã học trên toàn bộ `T`. Nhãn của `V` dùng để chấm. `test` cung cấp các giá trị cần biến đổi và ID đầu ra. |
| `A` và `E` bên trong `T` | Khi chọn số cây, học và cross-fit đặc trưng trong `A`, sau đó áp bảng `A` lên `E` để dừng sớm. Sau khi chọn số cây, dựng lại đặc trưng trong `T` và fit lại trước khi chấm `V`. |

Mã GPU `FeatureBuilder` có thể cấp cùng mã định danh nội bộ cho một khóa xuất hiện ở train và test. Những mã này phục vụ tra cứu. Từ vựng one-hot, ngưỡng phân vị, số đếm và thống kê có nhãn được xác định theo phần tham chiếu. Bước 7 dùng outer 5, inner 5, `m=20`, `key_scale=100`, seed encoder 2026. Nhật ký ghi đánh giá xáo nhãn cả encoder và mô hình có AUC 0,500843. Thử minh họa rò rỉ cho AUC 0,913150 khi dùng nhãn sai vai trò, và đánh dấu `valid_registry=false`. [Bước 7 trong nhật ký](../../reports/steps_progress.md) · [Mã target encoding](../../src/target_encoding.py) · [Bộ dựng đặc trưng GPU](../../s6e11_gpu/plan_runs/features.py) · [Vòng chọn số cây và refit](../../s6e11_gpu/plan_runs/run_plan.py)

## Target encoding và các nhóm đặc trưng GPU

Gọi `n_g` là số dòng có khóa `g`, `s_g` là số ca gian lận trong nhóm, và `π` là tỷ lệ dương trong tập tham chiếu. Target encoding (mã hóa theo nhãn) dùng **`p_g = (s_g + mπ)/(n_g+m)`**. Giá trị chưa thấy quay về `π` khi `m>0`. Cây nhận `p_g`. GLM nhận `logit(p_g) = log(p_g/(1−p_g))` sau khi chặn xác suất vào `[10⁻⁶,1−10⁻⁶]`. Khóa số trong lượt GPU là `u = round(100x)` với `np.rint`. Cờ dùng khóa 0/1 và trường phân loại dùng giá trị nhóm. [Luồng TE baseline](../../src/features.py) · [Khóa và công thức tra bảng GPU](../../s6e11_gpu/plan_runs/features.py)

| Lượt | Đầu vào và điểm đo cùng đối chứng |
|---|---|
| `T01`, `T02`, `T03` | Thêm TE cho 13 trường số. GLM `T01` đạt 0,947261, +0,005371 so với `GPU-B02`. LightGBM `T02` đạt 0,948527, +0,001295 so với `GPU-S6-L04`. XGBoost `T03` đạt 0,948495, +0,001356 so với `GPU-S6-X02`. |
| `T04`, `T01-M50` | `T04` mở rộng TE từ 13 trường số thành đủ 24 trường trên LightGBM, đạt 0,948574, +0,000047 so với `T02`. `T01-M50` đổi độ co từ 20 lên 50 cho GLM, đạt 0,947208, Δ −0,000053. |
| `D01`, `XGB-D01` | Thêm 6 cột phần dư `u % 100` trực tiếp. LightGBM `D01` đạt 0,948784, +0,000210 so với `T04`. XGBoost `XGB-D01` đạt 0,948623, +0,000129 so với `T03`. |
| `D02`, `XGB-D02` | Thêm 36 cột TE chữ số, 6 khóa mỗi trường. LightGBM đạt 0,950124, +0,001550 so với `T04`. XGBoost đạt 0,950074, +0,001580 so với `T03`. |
| `D03`, `XGB-D03` | Thêm cả 6 phần dư và 36 cột TE chữ số. `D03` LightGBM dùng `te24`, đạt 0,950131, +0,001557. `XGB-D03` dùng `te13`, đạt 0,950182, +0,001688. |
| `COUNT` | `log1p(số dòng tham chiếu có khóa)` từ 24 cột, hỗ trợ cả khóa cặp khi được bật. Trên đối chứng `T04`, AUC 0,946825, Δ −0,001749. |
| `DERIVED` | Thêm `(tổng 6 cờ)²` và `amount_usd/max(account_balance_usd,1)`. Trên `T04`, AUC 0,948542, Δ −0,000032. |

Các điểm trên là OOF ROC AUC trên cùng quy trình GPU 5 fold. `XGB-D03` có **109 cột** gồm 54 cột gốc sau one-hot, 13 TE khóa số, 6 cột phần dư và 36 TE khóa chữ số. `D03` là mã của LightGBM trong bộ chạy, còn `XGB-D03` xác định đúng nhánh XGBoost. [Đối chứng GPU từng cấu hình](../../s6e11_gpu/plan_runs/published/full/comparison.md) · [Báo cáo cấu trúc 109 cột](../eda_review/ket_qua_eda.md) · [Notebook TE](../../s6e11_gpu/plan_runs/notebooks/01-te.ipynb) · [Notebook chữ số](../../s6e11_gpu/plan_runs/notebooks/02-digit.ipynb)

Sáu trường của nhóm chữ số là `amount_usd`, `hours_since_last_txn`, `distance_from_home_km`, `account_balance_usd`, `velocity_score`, `merchant_risk_score`. Từ `u = round(100x)`, `cents` lấy `u mod 100`, `int` lấy `u // 100`, `int10` lấy `u // 1000`, `tenths` lấy `u // 10`, tức khóa toàn phần đến một chữ số thập phân. `lead1`, `lead2` lấy 1 hoặc 2 chữ số đầu của trị tuyệt đối phần nguyên, ghép số chữ số vào khóa để tránh gộp các bậc lớn nhỏ. Các trường điểm vẫn giữ nguyên giá trị thô khi tính đầu vào raw. [Nơi định nghĩa khóa chữ số](../../s6e11_gpu/plan_runs/features.py)

### Mã hóa tương tác khóa cặp

Mỗi ô kết hợp hai khóa rồi dùng công thức TE. Các trục `Q(·)` được chia theo mười nhóm phân vị, lấy ngưỡng từ tập tham chiếu của encoder. Nhánh `P1`, `P1-P2`, `P1-P2-P3` lần lượt thêm các nhóm sau vào `T01`. [Danh sách cặp trong mã](../../s6e11_gpu/plan_runs/features.py) · [Notebook cặp, count và derived](../../s6e11_gpu/plan_runs/notebooks/03-pairs-count.ipynb)

| Nhóm | Cặp được mã hóa | Δ AUC với đối chứng |
|---|---|---:|
| `p1` | Ngành hàng × điểm merchant · nước ngoài × lệch IP · ngành hàng × `Q(amount)` · số giao dịch 24 giờ × `Q(velocity)` | +0,000223 với `T01` |
| `p2` | Ngành hàng × tuổi khách · `Q(amount)` × `Q(merchant risk)` · `Q(hours)` × `Q(velocity)` · tuổi khách × `Q(merchant risk)` | +0,000097 với `P1` |
| `p3` | Merchant mới × ngày · cờ lừa đảo AI × `Q(velocity)` · tuổi khách × tuổi thẻ · giờ × ngày · số giao dịch 24 giờ × tuổi khách | −0,000196 với `P1-P2` |

Nhánh GLM đi tiếp từ `P1-P2` và thử Newton–Cholesky, đạt 0,947656 với `GLM-NEWTON`. Nhánh XGBoost đi tiếp với `XGB-D03`. `FINAL` ghép hai nhánh này qua tầng logistic, OOF 5×5 **0,950345**. [Bảng so sánh từng bước](../../s6e11_gpu/plan_runs/published/full/comparison.md) · [Cấu hình cuối](../../s6e11_gpu/plan_runs/published/full/final/selection.json)

## Các thử nghiệm tín hiệu bổ sung trên bộ 5 fold

Các notebook GPU đời đầu thử `Logistic + TE`, LightGBM bốn lá và XGBoost độ sâu 2. Lượt 2 thêm hai khóa cặp vào XGBoost, đạt 0,948193, thấp hơn 0,000118 so với XGBoost đối chứng chạy lại trên cùng split. Kết quả này đo cấu hình của notebook vòng 2. [Kết quả vòng 1](../s6e11_gpu/results/comparison.md) · [Kết quả vòng 2](../s6e11_gpu/round2/results/comparison.md) · [Notebook của hai cặp](../s6e11_gpu/round2/xgboost-pair-te.ipynb)

| Hướng | Đặc trưng hoặc cách ghép | Kết quả cùng giao thức nguồn |
|---|---|---|
| EDA `channel × hour_block_6h` | `floor(hour/6)` ghép kênh, tạo 20 cột one-hot | XGB-D03 129 cột, AUC 0,950108, Δ −0,000074. |
| EDA `velocity_residual` | `velocity − trung vị nhóm`, nhóm bằng số giao dịch 24h cắt tại 3, 6, 10 và thời gian cắt tại 0,25, 1, 6, 24 giờ. Nhóm dưới 250 mẫu dùng trung vị chung của phần fit. | XGB-D03 110 cột, AUC 0,950083, Δ −0,000099. |
| MLP tiền huấn luyện | Nối biểu diễn 64 chiều từ mô hình học khôi phục trường bị che (masked pretraining) vào XGB-D03 | Δ −0,000522. |
| Lân cận đa độ rộng | Hai cửa sổ quanh mức giá trị của sáu trường số, lấy tỷ lệ đã làm trơn, phần lệch ở đúng mức và hỗ trợ | AUC 0,95029222, dưới `FINAL` 0,00005299. |
| Tương tác theo lớp | 25 tỷ số hợp lý theo lớp sau khi trừ ảnh hưởng phân bố biên, ô thưa co về giả định độc lập | AUC 0,95022205, dưới `FINAL` 0,00012316. |
| Mật độ cặp giá trị | Bảy cặp số chính xác, đếm lệch so với kỳ vọng của ô thô và trừ chính dòng đang mã hóa | AUC 0,94986082, dưới `FINAL` 0,00048439. |
| Spline có tương tác | Logistic Ridge trên spline, chữ số, phân loại và các tương tác chọn từ sai lệch của train | Điểm riêng 0,94576566. Trộn 10% log-odds với `FINAL` cho 0,95045057 trong lượt khảo sát. |
| Hiệu chỉnh thời gian chính xác | Hiệu chỉnh logit theo nhóm 0,01 giờ, so đối chứng OOF lồng nhau và lân cận ±0,20 giờ | 0,95035247, +0,00000726 so với `FINAL`. |

Nguồn chạy gồm [ablation kênh và velocity](../eda_review/feature_ablation/run_ablation.py), [bảng kiểm kết quả ablation](../eda_review/feature_ablation/results/ket_qua_thu_dac_trung.md), [tổng tiền huấn luyện](../s6e11_gpu/pretraining/results/summary.md), [runner lân cận và mật độ theo lớp](../novel_model_experiments/multiscale_tree.py), [runner mật độ cặp](../novel_model_experiments/occupancy_tree.py), [runner spline](../novel_model_experiments/alternative_model.py), [runner giờ chính xác](../novel_model_experiments/hours_residual.py) và [báo cáo thử nghiệm bổ sung](../novel_model_experiments/BAO_CAO.md).

## Đặc trưng Grandmaster và offset

Hai bản lưu trong `Model tham khảo` cho thấy hướng phát triển song song của khối spline và TE. V1 dùng TE làm trơn trên 36 bộ khóa đa độ phân giải, one-hot, tần suất giá trị và spline tự nhiên 7 nút cho điểm merchant. V2 thêm hai bộ spline gồm 10 nút đều trên merchant risk và velocity, cùng spline 7 nút theo phân vị, làm tròn ở `float64` và dùng các hoán vị chia fold con độc lập cho TE. [V1, phần kiến trúc](<../Model tham khảo/V1/README.md>) · [V2, phần kiến trúc](<../Model tham khảo/V2/README.md>)

| Khối | Cách dựng từ dữ liệu giao dịch |
|---|---|
| Dạng số | Giữ số thô, cờ và tổng cờ. Bổ sung log cho các giá trị không âm, cặp sin/cos với chu kỳ giờ 24 và ngày 7. Ví dụ tỷ lệ số tiền `log1p(amount/(max(balance,0)+1))`, tiền trên số giao dịch `log1p(amount/(max(txn24,0)+1))`, nhịp `log1p(velocity/(max(hours,0)+0,1))` và `merchant_risk×velocity/100`. |
| Spline | Cubic B-spline, tức ghép các hàm đa thức bậc 3 để mô tả quan hệ trơn, dùng 10 nút. Học bộ chuẩn hóa và spline trong fold. Các trục được chọn gồm log tiền, số dư, khoảng giờ, khoảng cách, hai điểm rủi ro, tuổi khách/thẻ, giờ và bốn biến tỷ lệ/tích ở trên. |
| Digit trie và one-hot thưa | Từ `round(100x)` cho tiền, số dư, giờ, khoảng cách, lấy mức chính xác, phần nguyên, phần mười, bậc lớn, tiền tố và hậu tố theo trường. Với velocity và rủi ro merchant dùng `round(10x)`. Chọn mức chính xác ở 8 trường được EDA hỗ trợ gồm tuổi khách, rủi ro merchant, velocity, khoảng giờ, số dư, tiền, số giao dịch 24 giờ và CVV. Các khóa ít hơn 4 dòng chung một mã hiếm trong từ vựng học theo outer train. |
| GLM dày và TE nhiều mức co | Dùng khối spline/số và TE dạng **`logit(p_g)−logit(π)`** với `m=10,50`. Ma trận train lấy trung bình TE từ 5 fold con dưới ba seed 101/202/303. Validation/test dùng bảng của outer train. GLM thưa dùng one-hot khối hiếm và spline có hệ số scale khác nhau để điều chỉnh độ phạt L2. |
| Cây và cặp phân vị | Dùng đặc trưng số, mã phân loại, TE với `m=20` và tần suất của nhóm khóa đã chọn. Ngưỡng điểm merchant/velocity chia theo tối đa 16 khoảng và tuổi theo 10 · ghép biến phân vị với cờ, số lần thử, số giao dịch, tuổi hoặc xác thực. EBM (`Explainable Boosting Machine`) dùng đầu vào không có TE và học hiệu ứng cộng tính cùng tương tác cặp. |

Nguồn trực tiếp là các mục 2–5 của [notebook Grandmaster](<../Model tham khảo/Grandmaster/notebook.ipynb>) và mục 4–7 của [báo cáo nghiên cứu Grandmaster](<../Model tham khảo/Grandmaster/FINAL_RESEARCH_REPORT.md>).

Gọi `z_base` là log-odds nền, `f(x)` là biên do cây học. **Offset (OFF, điểm log-odds đặt trước)** đưa dự đoán về **`p(x) = sigmoid(z_base(x)+f(x))`**, với `sigmoid(z)=1/(1+exp(−z))`. XGBoost nhận `base_margin`, LightGBM dùng `init_score` và cộng lại logit nền khi suy luận, CatBoost dùng `baseline` trong `Pool`. Trong lượt GPU `GLM-BOOST-*`, offset train là điểm của GLM đã fit trên phần tham chiếu. Trong Grandmaster offset train được tạo từ **3 fold con của hai GLM** và nền `z_Blend50=(z_sparse+z_dense)/2`. Đây là khác biệt quan trọng khi so hai lượt. `ResDecomp_NonNeg` dùng các cột phần sai lệch **`Δ_m=z_m−z_Blend50`** cho các cây offset, cùng các logit mô hình nền để fit tầng gộp hệ số không âm. [Offset GPU](../../s6e11_gpu/plan_runs/models.py) · [Vai trò offset GPU](../../s6e11_gpu/plan_runs/run_plan.py) · [Offset và tầng gộp Grandmaster, mục 3–6](<../Model tham khảo/Grandmaster/notebook.ipynb>)

### Phân cấp tiền và đặc trưng mật độ

- **`hierarchical_amount`.** Với mức tiền chính xác `c` có dưới 4 dòng, lấy nhóm cha `c//10` (0,10 USD), trừ chính mức con rồi yêu cầu cha có ít nhất 4 dòng. `p_cha=(s_cha+50π)/(n_cha+50)`, `p_con=(s_con+10p_cha)/(n_con+10)`. Đưa `logit(p_con)−logit(π)` vào GLM dày. Cột bằng 0 ngoài miền hỗ trợ. Nhánh GLM mới thay logit GLM dày trong tầng gộp khi xuất bản phân cấp. [Mã phân cấp](../grandmaster_10fold_ablation/improvement_features.py) · [Điểm ghép GLM](../grandmaster_10fold_ablation/branch_runner.py) · [Manifest bản phân cấp](../grandmaster_10fold_ablation/submission_hierarchical_amount/manifest.json)
- **Mật độ theo lớp (class-conditional density).** Lấy hai phân bố của giá trị `v` trong cùng lớp `y=1` và `y=0`, rồi dùng **`LLR(v)=log(P(v|y=1)/P(v|y=0))`**. Các phiên bản theo ngữ cảnh chia chính xác/thô cho khoảng giờ, velocity, điểm merchant và số tiền. Ví dụ khoảng giờ được đổi thành bước 0,01 và ô thô 0,25 giờ. Với độ hỗ trợ `s` là số mẫu của lớp ít hơn trong tập cha, trộn phân bố con với cha bằng `w=s/(s+λ)` trên ngưỡng tối thiểu, `w=0` dưới ngưỡng. LLR ở mức chính xác là tổng LLR ô thô và LLR vị trí trong ô. Có thêm tương phản giữa đúng mốc giờ và lân cận ±2 bước, cùng các trọng số `w` như đặc trưng hỗ trợ. [22 cột và hệ số co](../grandmaster_10fold_ablation/class_conditional_density.py) · [Runner đối chứng dùng offset](../grandmaster_10fold_ablation/run_density_offset.py)

## Grandmaster EDA và các biến thể từ phần sai lệch

Với nhóm `g`, xác suất nền `p_i` và nhãn `y_i`, đặt **`G_g=Σ(y_i−p_i)`**, **`H_g=Σp_i(1−p_i)`**, **`N_g=số dòng`**. `H_g` là thông tin logistic, giúp đo nhóm có đủ dữ liệu để ước lượng thay đổi logit. Sàng lọc cặp loại ảnh hưởng từng trục bằng hồi quy có trọng số `H`. Khi tạo đặc trưng, lấy **`clip(G_centered/(H+25),−0,35,+0,35)`** cho mỗi ô nếu `H ≥ 8` và `N ≥ 80`, còn lại bằng 0. Nhánh chữ số cuối của khoảng cách dùng khóa `round(100×distance) mod 10` và trừ xu hướng chung trước khi co. Mọi cột có nhãn được cross-fit 5 fold cho phần train của cây. [Mã thống kê, hỗ trợ và co hệ số](../goal_0952/residual_models.py) · [Nguồn thứ hạng cặp](../goal_0952/residual_eda.json)

Nhánh XGBoost 224 vòng có **122 cột**, gồm 109 cột `XGB-D03`, 12 cặp EDA và 1 cột chữ số khoảng cách. Cây học logistic trên offset Grandmaster phân cấp. Giữ nguyên 224 vòng, từ bộ cũ sang thêm EDA nâng AUC giữ lại từ **0,95104590** lên **0,95112021**. Tổ hợp chốt theo `BAO_CAO_0952.md` dùng **85% nhánh XGBoost có offset + 10% LightGBM D03 + 5% spline** trên log-odds, đạt **0,95122445**. [Kết quả theo số vòng và cấu hình](../goal_0952/BAO_CAO_0952.md) · [Mã gộp và xuất](../goal_0952/evaluate_and_export.py)

Runner lấy tối đa 12 cặp đầu theo mức giảm log loss xấp xỉ, với ngưỡng lớn hơn 2 phần triệu. Các cặp được lưu theo thứ tự trong trường `selected_residual_pairs`. [Quy tắc chọn](../goal_0952/residual_models.py) · [Danh sách cấu hình phát triển](../goal_0952/residual_development_metrics.json)

| Cột thứ nhất | Cột thứ hai |
|---|---|
| `txn_count_last_24h` | `card_type` |
| `hours_since_last_txn` | `ip_country_mismatch` |
| `cvv_retry_count` | `card_type` |
| `device_type` | `is_new_merchant` |
| `merchant_risk_score` | `billing_shipping_mismatch` |
| `velocity_score` | `billing_shipping_mismatch` |
| `customer_age` | `billing_shipping_mismatch` |
| `channel` | `billing_shipping_mismatch` |
| `account_balance_usd` | `billing_shipping_mismatch` |
| `device_type` | `used_vpn` |
| `prior_disputes` | `is_new_merchant` |
| `account_balance_usd` | `is_foreign_transaction` |

| Lượt | Thay đổi đặc trưng và độ hỗ trợ | Kết quả trên holdout của lượt |
|---|---|---|
| Mật độ mở rộng | Thêm 10 tương phản theo lớp cho các ngữ cảnh như CVV × loại thẻ × xác thực, khoảng giờ × lệch IP × kênh, velocity × số giao dịch × xác thực. Bảng cục bộ so với nền toàn lớp, dùng hệ số làm trơn 35 cho lớp dương và 100 cho lớp âm. Mỗi ngữ cảnh cần ít nhất 8 dương và 120 âm, rồi giới hạn kết quả ±2,5. | Trộn nửa hai nhánh mật độ tăng AUC từ 0,95122445 lên 0,95131224. [Mã](../goal_0952/followup/context_expansion.py) · [Kết quả](../goal_0952/followup/BAO_CAO_FOLLOWUP.md) |
| Hồ sơ đồng cảnh và quan hệ số | Phân vị, độ hiếm và tương phản mật độ theo lớp trong ngữ cảnh 2–3 trường · nhóm thưa dùng ngữ cảnh cha. Đối chiếu thêm 16 biến thời gian/số dư, 15 cặp chữ số cuối và bản gộp cả ba. | Hồ sơ đồng cảnh AUC 0,95129283 so với 0,95131224 · AP tăng 0,00032682 và log loss giảm 0,00000949. Các nhóm thử riêng đều giảm AUC trên holdout. [Mã](../goal_0952/data_signals/overlooked_features.py) · [Kết quả](../goal_0952/data_signals/BAO_CAO_DAC_DIEM.md) |
| Đồng nhất bảng 80% và làm mượt | Khi suy luận, trung bình 5 cách chấm trên bảng EDA học từ 80% tham chiếu. Biến thể khác chuyển 10% thống kê cho từng ô phân vị bên cạnh và giảm dần trọng số của nhóm ít hỗ trợ. | AUC 0,95121087 và 0,95121100 so với 0,95122445. Bản 5 bảng có +0,00005404 khi đảo vai trò. [Mã](../goal_0952/grandmaster_eda_refinement/refine_eda.py) · [Kết quả](../goal_0952/grandmaster_eda_refinement/BAO_CAO_GRANDMASTER_EDA.md) |
| EDA theo rủi ro và 10 cặp thêm | Độ dốc tương tác phụ thuộc `tanh((z−logit(0,03))/1,5)`. Nhánh cặp mới cộng 10 hiệu chỉnh theo ô thành một cột, chọn trong các cặp chưa dùng với mức giảm log loss chéo xấp xỉ ít nhất một phần triệu. | Bản rủi ro qua XGBoost 0,95112217. Bản 10 cặp tuyến tính 0,95123214, +0,00000769, đảo vai trò gần 0. [Mã](../goal_0952/grandmaster_eda_risk/risk_eda.py) · [Kết quả](../goal_0952/grandmaster_eda_risk/BAO_CAO_TOI_UU_EDA.md) |
| Chi tiết trong ô phân vị | Chia đôi các ô phân vị gốc của sáu cặp số × nhóm. Loại mức chung của cặp con và mức lệch chung của trục phân loại. Hệ số co nhân `H_eff/(H_eff+8)` và `min(N_thấp,N_cao)/(min(N_thấp,N_cao)+80)`. | Đưa vào cây đạt 0,95120508 · hiệu chỉnh tuyến tính 0,95122403. [Mã](../goal_0952/parallel_eda_resolution/resolution_eda.py) · [Kết quả](../goal_0952/BAO_CAO_THU_SONG_SONG.md) |
| Ngữ cảnh hai biến | Chuẩn hóa hạng của velocity × khoảng giờ và số tiền × điểm merchant theo ngữ cảnh. Ước lượng tương quan từ ca hợp lệ và co nhóm thưa về cha. Ba cột/cặp ghi độ hiếm chung, sai lệch có hướng và tích điểm hạng. | AUC 0,95113668 so với 0,95122445. AP 0,53908200 và log loss 0,04380857 đều cải thiện. [Mã](../goal_0952/parallel_peer_context/joint_features.py) · [Kết quả](../goal_0952/BAO_CAO_THU_SONG_SONG.md) |
| Điều chuẩn mật độ | Chuyển 20% số đếm sang từng ô liên kề, tăng co ngữ cảnh và giảm chiều sâu XGBoost từ 3 xuống 2. | Tổ hợp 0,95126523 so với 0,95131224. Đảo vai trò tăng AUC có trọng số +0,00006812. [Mã](../goal_0952/robust_validation/robust_density.py) · [Kết quả](../goal_0952/robust_validation/BAO_CAO_DO_ON_DINH.md) |
| Đặc trưng từ điểm mô hình | Dùng độ lệch logit giữa bảy mô hình Grandmaster, bỏ phần tuyến tính của điểm nền, rồi thử tích bậc hai hoặc độ gần 48 cụm đầu ra. Một nhánh riêng dùng `0,8y + 0,2p_OOF` làm nhãn mềm. | Bậc hai đạt 0,95134001 trên holdout nhưng đảo vai trò giảm khoảng 0,00002963 AUC có trọng số. Nhãn mềm đạt 0,95129512 trên holdout, tăng +0,00012738 khi đảo vai trò. [Mã bậc hai/cụm](../goal_0952/score_space/score_space.py) · [Nhãn mềm](../goal_0952/score_space/soft_targets.py) · [Kết quả](../goal_0952/score_space/BAO_CAO_HUONG_KHAC.md) |

Các mức AUC của hai đối chứng `0,95122445` và `0,95131224` đến từ hai công thức gộp khác nhau, lần lượt là Grandmaster EDA gốc và bản gộp hai nhánh mật độ. Kiểm đảo vai trò chấm nhánh riêng với ít dữ liệu tham chiếu hơn. [Đối chứng của nhánh mật độ](../goal_0952/followup/BAO_CAO_FOLLOWUP.md) · [Giao thức kiểm đảo vai trò](../goal_0952/BAO_CAO_THU_SONG_SONG.md)

## Cờ amount hole và các thử nghiệm EDA sâu

**Amount hole** là cờ từ tần suất mức tiền. Chốt 404 mức trên tập phát triển, dùng lưới cent, hai cửa sổ ±5 và ±25, ngưỡng số đếm tối thiểu 10, trung bình lân cận tối thiểu 40 và tỷ lệ `(n+5)/(q+5) ≤ 0,4` ở cả hai cửa sổ. Hàm xuất cộng **`β·I(cent thuộc danh sách)`** vào log-odds Grandmaster EDA. Học `β` bằng logistic với L2 25, giới hạn ±0,30. Hai hệ số là **0,27198694** khi chấm holdout và **0,28729082** khi fit lại toàn train để xuất test. Đây là hiệu chỉnh một cờ hậu huấn luyện trên điểm của tổ hợp, áp cho 4.448 dòng test. [Điều kiện chọn cờ](../goal_0952/deep_eda_measurement/METHOD.md) · [Hiện thực kiểm tra](../goal_0952/deep_eda_measurement/measurement_eda.py) · [Runner xuất](../goal_0952/deep_eda_measurement/export_amount_hole.py) · [Manifest và 404 khóa](../goal_0952/deep_eda_measurement/amount_hole_submission.json)

Các cách suy chuyển ngày hoặc tuần từ giờ và thời gian kể từ giao dịch trước, chênh lệch số dư theo loại thẻ và chiếu bỏ ba hiệu ứng cặp khỏi mỗi bảng ba biến đều được khảo sát. Hai đặc trưng hành vi qua sàng lọc phát triển làm giảm AUC khi áp lên holdout. Ba tổ hợp ba biến đã qua sàng lọc đều làm log loss xấu hơn trên phần C tách riêng. [Tổng hợp EDA và số liệu](02_EDA.md) · [Mã thử đặc trưng hành vi](../goal_0952/deep_eda_behavior/audit_behavior.py) · [Mã thử tương tác ba biến](../goal_0952/deep_eda_triples/triple_eda.py)

## Phạm vi đối chiếu và điểm vào chạy lại

- Đọc toàn bộ `INT3405E_FD/src/features.py` và `INT3405E_FD/s6e11_gpu/plan_runs/features.py`, các đoạn chọn số cây, refit, offset, `run_te()`, `run_features()` và `run_xgb_digit_comparison()` trong `run_plan.py`. Đối chiếu bảng kết quả GPU và mục bước 7 trong `steps_progress.md`. [Hướng dẫn chạy GPU và môi trường](../../s6e11_gpu/plan_runs/README.md)
- Đọc phần khai báo trường và hàm dựng khóa, từ vựng, spline, TE, offset và phân rã của các mục 2–5 notebook Grandmaster. Đọc mục 4–7 của báo cáo nghiên cứu để đối chiếu thiết kế và chỉ số. Với phân cấp và mật độ, xem `amount_parent_feature`, `ClassConditionalDensity` và đoạn fit GLM dày trong `branch_runner.py`. [Notebook chạy độc lập](<../Model tham khảo/Grandmaster/notebook.ipynb>) · [README Grandmaster](<../Model tham khảo/Grandmaster/README.md>)
- Với các vòng 0,952, đọc phần kết quả và mô tả đặc trưng trong báo cáo từng nhánh được dẫn ở bảng. Đối chiếu hàm tính hỗ trợ trong `residual_models.py`, `context_expansion.py`, `resolution_eda.py` và `joint_features.py`, cùng phương pháp, kết quả holdout và manifest của amount hole. Các bảng tóm tắt chỉ số đã lưu. [Hướng dẫn tái lập Grandmaster EDA](../goal_0952/BAO_CAO_0952.md) · [Các điểm vào EDA sâu](../goal_0952/BAO_CAO_EDA_SAU.md)
