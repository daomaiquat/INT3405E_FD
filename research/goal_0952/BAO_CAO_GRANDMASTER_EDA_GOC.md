# Grandmaster EDA gốc · Báo cáo kỹ thuật

**Grandmaster EDA gốc đạt ROC AUC 0,95122445 trên 104.994 giao dịch giữ lại.** Cấu hình `offset224_lgb10_spline05` tăng **0,00027552** so với Grandmaster phân cấp số tiền và còn cách mốc local 0,952 là **0,00077555**. Mô hình gộp 85% điểm từ Grandmaster có XGBoost học phần sai lệch, 10% LightGBM bốn lá và 5% hồi quy logistic với spline tương tác. Tỷ trọng áp dụng trên log-odds, tức logarit của tỷ lệ cược gian lận. [Notebook Grandmaster EDA](grandmaster_eda.ipynb) · [CSV gốc](submission_grandmaster_eda.csv) · [Hồ sơ đánh giá 27 cấu hình](evaluation.json)

## 1. Dữ liệu và thiết kế đánh giá

`train.csv` có 699.960 giao dịch, gồm 11.689 ca gian lận và 688.271 giao dịch hợp lệ. Tỷ lệ gian lận là **1,670%**. `test.csv` có 299.984 giao dịch cần dự đoán. Kiểm tra dữ liệu ghi nhận không có ô thiếu hay vô hạn và không có dòng trùng trên toàn bộ 24 đặc trưng, kể cả giữa train và test. `id` dùng để đối chiếu dòng, xác minh thứ tự và xuất CSV. `is_fraud` là nhãn nhị phân. Pipeline học từ dữ liệu chính thức và các điểm dự đoán tạo lại trong dự án, bằng các mô hình học máy thông thường. [Kiểm tra dữ liệu](data_audit.json) · [Manifest dữ liệu](../../data/manifest.json)

| Nhóm | Số cột | Tên trường |
|---|---:|---|
| Dạng số | 13 | `amount_usd`, `hours_since_last_txn`, `txn_count_last_24h`, `distance_from_home_km`, `card_age_months`, `customer_age`, `account_balance_usd`, `cvv_retry_count`, `velocity_score`, `time_of_day_hour`, `day_of_week`, `merchant_risk_score`, `prior_disputes` |
| Phân loại | 5 | `merchant_category`, `card_type`, `auth_method`, `channel`, `device_type` |
| Cờ nhị phân | 6 | `is_foreign_transaction`, `is_new_merchant`, `used_vpn`, `ip_country_mismatch`, `billing_shipping_mismatch`, `is_ai_generated_scam_attempt` |

Audit theo từng cột ghi nhận chênh lệch train–test lớn nhất ở `day_of_week` theo thống kê KS (0,003825) và `auth_method` theo khoảng cách tổng biến thiên (0,002310). Khi thử từng biến riêng trên 5 fold, `merchant_risk_score` có AUC 0,741200, `velocity_score` 0,706536, `cvv_retry_count` 0,705663 và `auth_method` 0,684998. `cvv_retry_count` đạt Average Precision cao nhất trong 24 biến, ở mức 0,085704. Khóa giá trị chính xác có thể rất thưa, với 1.617 dòng test mang số tiền và 6.668 dòng mang số dư chưa từng gặp trong train. Các phép đo chênh lệch từng cột chỉ mô tả phân phối biên. [Cấu hình 24 trường](../../configs/default.json) · [Kiểm tra theo cột](data_audit.json) · [Kết quả đơn biến](univariate_signal.json) · [Biểu đồ EDA](eda_overview.png)

| Vai trò | Dòng | Ca gian lận | Cách dùng |
|---|---:|---:|---|
| Tập phát triển | 594.966 | 9.936 | Huấn luyện và tìm tín hiệu EDA có nhãn |
| Tập giữ lại (holdout) | 104.994 | 1.753 | So sánh và chọn cấu hình bằng ROC AUC |
| Train đầy đủ | 699.960 | 11.689 | Huấn luyện lại hoặc tái dựng từng nhánh ở chế độ `full` |
| Test chính thức | 299.984 | — | Tính xác suất và xuất CSV theo thứ tự ID chính thức |

Tập giữ lại lấy 15% train, chia phân tầng theo nhãn (stratified split) với seed **4242**. Trong tập phát triển, Grandmaster dùng 10 fold phân tầng, seed **2026**, để tạo dự đoán ngoài fold (out-of-fold, OOF). Mỗi dòng nhận điểm từ mô hình cơ sở đã bỏ dòng đó ra khi huấn luyện. Các điểm nhánh cho holdout hoặc test được lấy trung bình qua 10 fold trước khi gộp ở tầng meta. Target encoding của D03 dùng 5 fold con seed **2026** và 13 đặc trưng EDA dùng 5 fold con riêng seed **62026**. Mỗi bảng có nhãn dùng các dòng huấn luyện của fold tương ứng và chỉ tra bảng cho dòng cần chấm. Khi chấm holdout, nhãn của holdout chỉ đi vào phép đo và chọn cấu hình. [Hồ sơ chia tập](../grandmaster_10fold_ablation/validation/metrics.json) · [Tái dựng OOF](grandmaster_inputs.py) · [Mã đặc trưng D03](../../s6e11_gpu/plan_runs/features.py) · [Mã đặc trưng EDA](residual_models.py)

## 2. Nền Grandmaster phân cấp số tiền

Nền Grandmaster phân cấp kết hợp bảy nhánh. GLM (generalized linear model) ở đây là hồi quy logistic với các biến đổi spline và chữ số. Spline là hàm ghép từng đoạn để học quan hệ phi tuyến. EBM (Explainable Boosting Machine) là mô hình cộng tính có thể học tương tác được chọn. [Kiến trúc Grandmaster tham khảo](<../Model tham khảo/Grandmaster/README.md>) · [Notebook lưu trữ](<../Model tham khảo/Grandmaster/notebook.ipynb>)

| Nhánh | Vai trò trong mô hình nền |
|---|---|
| `02e_SpGLM` | GLM thưa với spline và chỉ báo theo tiền tố, hậu tố, phần thập phân của các biến số |
| `02f_DeGLM` | GLM với spline và target encoding được điều chuẩn, bổ sung một biến phân cấp theo số tiền |
| `03a_OffLGB` | LightGBM bốn lá học sai lệch từ điểm trung bình của hai GLM gốc |
| `03b_OffXGB` | XGBoost nông, `max_bin=1024`, học sai lệch từ cùng điểm trung bình |
| `03c_OffCAT` | CatBoost cây đối xứng độ sâu 4 học sai lệch từ cùng điểm trung bình |
| `03d_StXGB` | XGBoost độc lập, cây độ sâu 2, `max_bin=1024`, có tương tác theo phân vị |
| `04_EBM` | EBM cộng tính, có tối đa 25 tương tác, không dùng target encoding |

Biến phân cấp dùng khóa số tiền theo cent, tìm các mức tiền khác trong cùng khoảng một phần mười USD. Gọi $n_c,s_c$ là số giao dịch và số ca gian lận ở mức tiền đang xét, $n_p,s_p$ là các số tương ứng của những mức tiền lân cận trong cùng khoảng, và $\pi$ là tỷ lệ gian lận của tập tham chiếu. Với $n_c<4$ và $n_p\ge4$, xác suất nhóm rộng là $q_p=(s_p+50\pi)/(n_p+50)$, sau đó $q_c=(s_c+10q_p)/(n_c+10)$. Đặc trưng là $\operatorname{logit}(q_c)-\operatorname{logit}(\pi)$. Trường hợp còn lại nhận 0. Biến này được thêm vào riêng nhánh GLM đặc `02f_DeGLM`. [Mã đặc trưng phân cấp](../grandmaster_10fold_ablation/improvement_features.py) · [Mã huấn luyện nhánh](../grandmaster_10fold_ablation/branch_runner.py)

Tầng gộp (stacking) dùng hồi quy logistic L2, với hệ số không âm và hệ số chặn tự do. Với ba cây có điểm khởi tạo, cột đi vào tầng gộp là $z_{\mathrm{tree}}-z_{\mathrm{Blend50}}$. $z_{\mathrm{Blend50}}=(z_{\mathrm{SpGLM}}+z_{\mathrm{DeGLM,gốc}})/2$ giữ đúng điểm khởi tạo từng dùng để huấn luyện các cây. Sau khi dựng bảy cột, mã mới thay riêng cột `02f_DeGLM` bằng điểm phân cấp. Tầng gộp học trên ma trận OOF và xuất điểm nền $z_{\mathrm{GM}}$, lấy trung bình logit của 10 mô hình meta khi chấm dữ liệu ngoài tập phát triển. [Hàm dựng và gộp điểm](grandmaster_inputs.py) · [Ablation phân cấp](../grandmaster_10fold_ablation/run_ablation.py) · [Manifest Grandmaster phân cấp](../grandmaster_10fold_ablation/submission_hierarchical_amount/manifest.json)

## 3. EDA trên phần sai lệch còn lại

EDA (exploratory data analysis) là phân tích khám phá dữ liệu. Ở đây, phép sàng lọc dựa trên $p_i=\sigma(z_{\mathrm{GM},i})$ OOF của 594.966 dòng phát triển. Các biến số có hơn 25 giá trị được chia nhóm bằng các ngưỡng phân vị 2%, 5%, 10%, 20%, 35%, 50%, 65%, 80%, 90%, 95% và 98%, gộp ngưỡng trùng. Các biến rời rạc giữ từng mức. Tập phát triển được chia đôi phân tầng bằng seed **731**. Mã quét $\binom{24}{2}=276$ cặp. Trong mỗi nửa, một ô nhóm có $n$ dòng, phần sai lệch $G=\sum_i(y_i-p_i)$ và độ cong logistic $H=\sum_i p_i(1-p_i)$, tức lượng thông tin mô hình dùng khi ước lượng thay đổi nhỏ trên logit. [Mã sàng lọc](eda_from_scratch.py) · [Kết quả 276 cặp](residual_eda.json)

Mã ước lượng hiệu ứng hàng và cột $a,b$ bằng 12 lượt cập nhật bình phương tối thiểu có trọng số, thêm 2 vào mẫu số mỗi lượt. Phần tương tác còn lại là $\widetilde G_{jk}=G_{jk}-H_{jk}(a_j+b_k)$. Hai nửa chỉ so trên các ô có $n\ge40$ và $H\ge4$ ở cả hai bên, với ít nhất 5 ô hợp lệ. Hiệu chỉnh thử $\widetilde G/(H+25)$ từ nửa này được kiểm trên nửa kia theo xấp xỉ bậc hai của log loss, rồi đổi vai trò hai nửa. Chỉ số trong bảng là mức giảm log loss xấp xỉ trên mỗi dòng, nhân một triệu. Spearman đo mức tương đồng thứ hạng của tín hiệu chuẩn hóa $\widetilde G/\sqrt H$ giữa hai nửa. Có 32 cặp đạt mức giảm dương và 12 cặp vượt 2,0 được đưa vào XGBoost. [Kết quả sàng lọc](residual_eda.json) · [Danh sách đặc trưng dùng khi huấn luyện](residual_development_metrics.json)

| Cặp đặc trưng | Ô đủ hỗ trợ | Spearman | Mức giảm log loss × 10⁶ |
|---|---:|---:|---:|
| `txn_count_last_24h` × `card_type` | 37 | 0,513 | 11,421 |
| `hours_since_last_txn` × `ip_country_mismatch` | 24 | 0,454 | 6,623 |
| `cvv_retry_count` × `card_type` | 14 | 0,868 | 5,716 |
| `device_type` × `is_new_merchant` | 16 | 0,347 | 4,603 |
| `merchant_risk_score` × `billing_shipping_mismatch` | 20 | 0,338 | 4,194 |
| `velocity_score` × `billing_shipping_mismatch` | 20 | 0,444 | 3,198 |
| `customer_age` × `billing_shipping_mismatch` | 21 | 0,284 | 3,124 |
| `channel` × `billing_shipping_mismatch` | 9 | 0,983 | 2,674 |
| `account_balance_usd` × `billing_shipping_mismatch` | 22 | 0,536 | 2,443 |
| `device_type` × `used_vpn` | 16 | 0,171 | 2,363 |
| `prior_disputes` × `is_new_merchant` | 6 | 0,943 | 2,208 |
| `account_balance_usd` × `is_foreign_transaction` | 23 | 0,491 | 2,109 |

**Đặc trưng thứ 13** dùng chữ số hàng phần trăm của `distance_from_home_km`, tính bằng `round(100 × distance_from_home_km) % 10`. Mười nhóm có đủ hỗ trợ khi sàng lọc, Spearman **0,685** và mức giảm log loss xấp xỉ **9,965 × 10⁻⁶**. Khi huấn luyện, với từng nhóm chữ số $d$, tử số được trừ xu hướng chung theo $G_d-H_d\sum_d G_d/(\sum_d H_d+25)$. [Kết quả theo độ chính xác đo](residual_eda.json) · [Cách tạo cột chữ số](residual_models.py)

Khi tạo đặc trưng cho XGBoost, mỗi bảng được khớp lại trong fold tham chiếu. Với cặp biến, đầu ra là $\operatorname{clip}(\widetilde G/(H+25),-0{,}35,0{,}35)$. Cột chữ số dùng cùng mẫu số và khoảng chặn, với tử số riêng ở trên. Bảng huấn luyện yêu cầu **$n\ge80$** và **$H\ge8$**. Nhóm dưới ngưỡng hoặc chưa gặp nhận 0. Hằng số 25 kéo các ước lượng ít bằng chứng về 0 (shrinkage). [Cách mã hóa và chia fold](residual_models.py)

## 4. Từ 109 đặc trưng D03 đến mô hình gộp

### XGBoost học phần sai lệch · 85%

Nhánh `offset_xgb_224` nhận **122 cột**. Các cột D03 gồm 54 đầu vào gốc, 6 chữ số phần trăm giữ dưới dạng số và 49 cột target encoding (TE), tức thay khóa bằng tỷ lệ gian lận đã làm mượt từ các dòng tham chiếu. Mỗi khóa có tỷ lệ $\operatorname{TE}(k)=(s_k+20\pi)/(n_k+20)$, với $s_k$ là số ca gian lận, $n_k$ là số lần gặp và $\pi$ là tỷ lệ chung của fold huấn luyện. Khóa chưa gặp nhận $\pi$. [Cấu hình D03](../../configs/default.json) · [Cách sinh cột](../../s6e11_gpu/plan_runs/features.py) · [Đầu vào XGBoost](residual_models.py)

| Khối | Cột | Nội dung |
|---|---:|---|
| `raw` | 54 | 13 biến số + 6 cờ + 35 cột one-hot, tức chỉ báo 0/1 cho từng mức trong 5 biến phân loại |
| `te13` | 13 | TE theo giá trị chính xác của 13 biến số |
| `cents` | 6 | Hai chữ số thập phân của `amount_usd`, `hours_since_last_txn`, `distance_from_home_km`, `account_balance_usd`, `velocity_score`, `merchant_risk_score` |
| `digits` | 36 | Sáu TE cho mỗi biến trong `cents`, dùng khóa `cents`, `int`, `int10`, `tenths`, `lead1`, `lead2` |
| **D03** | **109** | Tổng bốn khối trên |
| **EDA** | **13** | 12 tương tác và 1 hiệu ứng chữ số khoảng cách |
| **Tổng đầu vào XGBoost** | **122** | 109 + 13 |

D03 đổi biến số sang đơn vị nguyên bằng cách làm tròn `100 × giá trị`. `int` giữ phần nguyên và `int10` gom theo chục đơn vị, còn `tenths` giữ giá trị ở độ chính xác một chữ số thập phân. `lead1` và `lead2` dùng một hoặc hai chữ số đầu của phần nguyên, kèm độ dài phần nguyên. [Mã khóa chữ số](../../s6e11_gpu/plan_runs/features.py)

XGBoost nhận $z_{\mathrm{GM}}$ làm điểm khởi tạo (`base_margin`) cho hàm mất mát logistic nhị phân. Cây học số hạng $f_{224}(x)$ cộng vào điểm đó. $z_{224}(x)=z_{\mathrm{GM}}(x)+f_{224}(x)$. Cấu hình dùng `hist`, `max_depth=3`, `max_bin=512`, `min_child_weight=35`, `eta=0.025`, L2 `40`, L1 `1.5`, lấy mẫu dòng `0.90`, lấy mẫu cột `0.85` và seed **8337**. Điểm `output_margin=True` đã bao gồm $z_{\mathrm{GM}}$. [Cấu hình và vòng huấn luyện](residual_models.py) · [Metrics nhánh trên train đầy đủ](residual_full_metrics.json)

### LightGBM bốn lá · 10%

Nhánh này học trực tiếp nhãn gian lận bằng `raw + te24 + cents + digits`. `te24` dùng cả 24 trường, nên toàn bộ ma trận có **120 cột** gồm 54 + 24 + 6 + 36. Cấu hình CPU chạy **1.600 vòng** với `num_leaves=4`, `learning_rate=0.05`, `min_data_in_leaf=50`, `max_bin=255`, L2 `1.0`, lấy mẫu dòng và cột đều `0.90`, seed **42**. Xác suất đầu ra được chặn vào $[10^{-12},1-10^{-12}]$ trước khi đổi sang $z_{\mathrm{LGB}}$. [Mã LightGBM](lightgbm_branch.py) · [Cấu hình đã chạy](lightgbm_metrics.json)

### Hồi quy logistic với spline tương tác · 5%

Spline dùng các đường cong theo từng biến số, chỉ báo phân loại, nhóm chữ số và số cờ bật. Sau bước khớp mô hình cộng tính, phép sàng lọc tương tác dùng phần sai lệch của riêng mô hình spline trên tập huấn luyện của nhánh. Mã chọn tối đa **14 cặp**, thêm tích các hàm cơ sở với hệ số khối **0,36**, rồi khớp hồi quy logistic L2 với `C=0.70`, tối đa 220 vòng tối ưu. Cặp `is_foreign_transaction` × `ip_country_mismatch` đứng đầu cả ở chế độ `development` lẫn `full`. Số vòng hội tụ ghi nhận lần lượt là 155 và 167. [Mã nhánh spline](spline_branch.py) · [Thiết kế và điều chuẩn](../novel_model_experiments/alternative_model.py) · [Danh sách tương tác mỗi lần huấn luyện](spline_metrics.json)

### Công thức xuất xác suất

$$
z(x)=0{,}85\,z_{224}(x)+0{,}10\,z_{\mathrm{LGB}}(x)+0{,}05\,z_{\mathrm{spline}}(x),
\qquad
\widehat P(y=1\mid x)=\sigma(z(x))=\frac{1}{1+\exp(-z(x))}.
$$

Mã tính trung bình có trọng số của ba logit, rồi áp dụng sigmoid một lần để đưa điểm về xác suất. Đây là công thức của `offset224_lgb10_spline05`. [Định nghĩa ứng viên và bước xuất tệp](evaluate_and_export.py)

## 5. So sánh đóng góp và độ chắc chắn

ROC AUC đo khả năng xếp giao dịch gian lận cao hơn giao dịch hợp lệ. Average Precision (AP) tóm tắt quan hệ giữa độ chính xác của cảnh báo và tỷ lệ phát hiện gian lận khi quét các ngưỡng. Log loss đo chất lượng xác suất và phạt mạnh dự đoán tự tin sai. Tiêu chí chọn cấu hình ở đây là ROC AUC. [Điểm của tất cả ứng viên](evaluation.json)

| Cấu hình | ROC AUC | Δ AUC so với nền phân cấp | Average Precision | Log loss |
|---|---:|---:|---:|---:|
| Grandmaster gốc | 0,95094666 | −0,00000227 | 0,53814886 | 0,04387461 |
| Grandmaster phân cấp | 0,95094893 | 0,00000000 | 0,53818509 | 0,04387376 |
| Cộng trực tiếp điểm EDA vào nền | 0,95092524 | −0,00002369 | 0,53768621 | 0,04389379 |
| XGBoost D03, 224 vòng | 0,95104590 | +0,00009697 | 0,53773762 | 0,04387143 |
| XGBoost D03 + EDA, 96 vòng | 0,95103456 | +0,00008563 | 0,53822247 | 0,04385955 |
| XGBoost D03 + EDA, 224 vòng | 0,95112021 | +0,00017128 | 0,53803676 | 0,04385635 |
| XGBoost D03 + EDA, 448 vòng | 0,95110415 | +0,00015522 | 0,53754005 | 0,04387368 |
| XGBoost D03 + EDA, 672 vòng | 0,95111153 | +0,00016260 | 0,53696057 | 0,04388682 |
| XGBoost 224 vòng, trung bình 5 seed | 0,95106722 | +0,00011829 | 0,53803845 | 0,04386547 |
| XGBoost 224 vòng + 5% spline trên logit | 0,95115951 | +0,00021058 | 0,53805571 | 0,04384911 |
| XGBoost 224 vòng + 10% LightGBM trên xác suất | 0,95117600 | +0,00022707 | 0,53819290 | 0,04384434 |
| **Grandmaster EDA gốc 85/10/5 trên logit** | **0,95122445** | **+0,00027552** | **0,53823528** | **0,04383599** |

- Ở cùng 224 vòng, thêm cả 13 cột đem lại **+0,00007431 AUC** so với XGBoost D03. Phép so này đo đóng góp chung của 12 cặp và chữ số, do các cột được thêm cùng lượt. Cộng trực tiếp tổng EDA đã chặn vào điểm nền làm giảm AUC, trong khi đưa chúng vào cây cho mức tăng quan sát được. [Đối chứng D03](residual_extended_development_metrics.json) · [Thử cộng trực tiếp và huấn luyện cây](residual_development_metrics.json)
- LightGBM riêng đạt AUC **0,95049680**, spline riêng **0,94693008**. Tổ hợp 85/10/5 thêm **0,00010424 AUC** trên nhánh XGBoost 224 vòng. Kết quả của tổ hợp phản ánh tác động chung của hai nhánh bổ sung và tỷ trọng. Trong 27 ứng viên, AP cao nhất thuộc `grandmaster_lgb20_mlp10` (**0,53899987**) và log loss thấp nhất thuộc `offset224_lgb20_mlp10` (**0,04383246**). [Kết quả theo nhánh và công thức gộp](evaluation.json)
- Các ca gian lận nằm trong 90% giao dịch có điểm nền thấp nhất đóng góp **0,00026052 AUC**, tương đương **94,6%** mức tăng ròng. Chỉ số này tổng hợp thay đổi vị trí của những ca đó so với toàn bộ giao dịch hợp lệ trên holdout. [Đóng góp theo dải điểm nền](evaluation.json)
- Khoảng tin cậy DeLong ghép cặp 95% cho Δ AUC của cấu hình thắng so với nền là **[+0,00002221, +0,00052883]**. Phép tính dùng hai dãy dự đoán trên cùng các giao dịch giữ lại, với cấu hình được xem là cố định. Holdout từng dùng ở các lượt trước và lại dùng để chọn phương án cao nhất trong 27 cấu hình, nên khoảng này chưa tính độ bất định do chọn mô hình và có thể lạc quan. Tầng meta dùng chung ma trận điểm cơ sở OOF giữa các fold, nên điểm OOF meta vẫn có thể phụ thuộc gián tiếp vào nhãn của fold được chấm qua quá trình huấn luyện cơ sở. AUC OOF **0,95128092** của nền phân cấp trên train đầy đủ vì vậy mang tính khảo sát. [Thiết kế và kết quả đánh giá](evaluation.json) · [Cách dựng ma trận OOF](grandmaster_inputs.py) · [Điểm nền OOF](residual_full_metrics.json)
- Điểm **0,95122445** đến từ nhãn train chính thức trong phép giữ lại local. Trong hồ sơ đã đọc, điểm Kaggle gắn được với nguồn nộp là **0,95154** của Grandmaster tham khảo. Hồ sơ chưa xác nhận điểm Public Leaderboard của đúng CSV `submission_grandmaster_eda.csv` theo SHA-256 ở mục 6. [Chỉ mục thực nghiệm của dự án](../docs/README.md) · [Nguồn điểm của Grandmaster tham khảo](<../Model tham khảo/Grandmaster/README.md>)

## 6. Notebook, mã nguồn và bản nộp

| Tệp | Nội dung |
|---|---|
| [grandmaster_eda.ipynb](grandmaster_eda.ipynb) | Notebook của cấu hình 85/10/5, trình bày EDA, các nhánh và kiểm tra kết quả đã lưu |
| [eda_from_scratch.py](eda_from_scratch.py) | Audit, sàng lọc tương tác, lưới chữ số và phân tích theo dải điểm |
| [grandmaster_inputs.py](grandmaster_inputs.py) | Xác minh dữ liệu và checkpoint, dựng lại các phân hoạch và điểm Grandmaster |
| [residual_models.py](residual_models.py) | Tạo 13 cột EDA theo fold con và huấn luyện XGBoost với điểm nền |
| [lightgbm_branch.py](lightgbm_branch.py) · [spline_branch.py](spline_branch.py) | Huấn luyện hai nhánh bổ sung ở `development` và `full` |
| [evaluate_and_export.py](evaluate_and_export.py) | Ghép điểm theo đúng ID, so ứng viên theo ROC AUC và xuất dự đoán đầy đủ |
| [selected_predictions.npz](selected_predictions.npz) | ID và xác suất đã lưu của cấu hình thắng cho holdout và test |

**Tệp nộp** [submission_grandmaster_eda.csv](submission_grandmaster_eda.csv) có **299.984 dòng dữ liệu**, với header `id,is_fraud`. `is_fraud` là xác suất. Giá trị nhỏ nhất khoảng `2.780717226040827e-08`, lớn nhất `0.9995567805256553`, trung bình `0.016486750199620426`. Bước xuất đọc lại CSV để xác minh cột, thứ tự ID và sai số lưu xác suất. SHA-256 của tệp gốc là `9aa21cdd324a0913c8f029dc4ebe1efcd3f51a8323c77ce93f13f283eba6f3d0`. [Kết quả xác minh của bước xuất](evaluation.json)

Notebook đã đối chiếu công thức trên các tệp điểm của cả ba nhánh ở hai chế độ `development` và `full`. Repo Git lưu `selected_predictions.npz` chứa điểm đã gộp. Các tệp `.npz` của từng nhánh có thể đọc từ thư mục `goal_0952` bên cạnh repo trên máy này hoặc từ đường dẫn đặt trong `GRANDMASTER_COMPONENTS`. Khi chỉ có repo, notebook dùng điểm đã gộp để kiểm tra đầu ra. Tạo lại điểm của ba nhánh từ đầu cần các checkpoint của 10 fold Grandmaster. Bước huấn luyện lại trong notebook tắt mặc định và chưa được chạy trong lượt đối chiếu này. [Notebook](grandmaster_eda.ipynb) · [Cách nạp checkpoint](grandmaster_inputs.py) · [Cách nạp điểm thành phần](evaluate_and_export.py)

## 7. Phạm vi nguồn đã đối chiếu

- `data_audit.json` và `data/manifest.json` dùng cho phần phạm vi, schema, số dòng, nhãn, ID, trùng lặp, hỗ trợ theo khóa và kiểm tra phân phối. `univariate_signal.json` dùng cho phương pháp 5 fold và các biến đứng đầu. Báo cáo dựa vào hồ sơ audit đã lưu cho thống kê nguồn. [Audit](data_audit.json) · [Manifest](../../data/manifest.json) · [Kết quả đơn biến](univariate_signal.json)
- `residual_eda.json` dùng các mục `discovery_*`, `risk_blocks`, danh sách `pair_results` và `exact_value_results`. `residual_development_metrics.json`, `residual_extended_development_metrics.json`, `residual_full_metrics.json`, `lightgbm_metrics.json` và `spline_metrics.json` dùng cho danh sách đặc trưng, chế độ chạy và cấu hình. `evaluation.json` được đối chiếu đủ 27 ứng viên, điểm theo nhánh, khoảng tin cậy, dải điểm và định danh đầu ra. [Sàng lọc EDA](residual_eda.json) · [Điểm và định danh đầu ra](evaluation.json)
- `grandmaster_inputs.py`, `eda_from_scratch.py`, `residual_models.py`, `s6e11_gpu/plan_runs/features.py`, `lightgbm_branch.py`, `spline_branch.py` và `evaluate_and_export.py` dùng cho các hàm chia tập, tạo đặc trưng, khớp mô hình, gộp điểm và xuất CSV. `alternative_model.py` dùng phần hằng số, `SplineDesign`, `pair_score` và `fit_logistic`. [Bộ đặc trưng](../../s6e11_gpu/plan_runs/features.py) · [Thiết kế spline](../novel_model_experiments/alternative_model.py)
- Nền Grandmaster được đối chiếu từ `validation/metrics.json`, manifest `submission_hierarchical_amount`, `grandmaster_refit/full/metrics.json`, các đoạn thay cột Dense trong `run_ablation.py`, `_fit_hierarchical_dense` trong `branch_runner.py`, `amount_parent_feature` trong `improvement_features.py`, và các đoạn `LOCKED_MODELS`, `build_resdecomp`, `_fit_nonneg_l2_weights` của notebook lưu trữ. Trạng thái Public Leaderboard tham chiếu mục 1–2 của README Grandmaster và phần mở đầu của `research/docs/README.md`. [Hồ sơ giữ lại](../grandmaster_10fold_ablation/validation/metrics.json) · [Manifest nền phân cấp](../grandmaster_10fold_ablation/submission_hierarchical_amount/manifest.json) · [Hồ sơ tái lập đủ 10 fold](../grandmaster_refit/full/metrics.json) · [README Grandmaster](<../Model tham khảo/Grandmaster/README.md>)
