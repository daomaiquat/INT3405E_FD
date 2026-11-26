# 04. Mô hình và thử nghiệm sau FINAL

Grandmaster EDA đạt ROC AUC holdout **0,9512244494** với tổ hợp log-odds 85% Grandmaster có cây XGBoost hiệu chỉnh, 10% LightGBM và 5% spline. Hiệu chỉnh riêng các mức số tiền thưa nâng bản Grandmaster EDA lên **0,9513158455** và tạo CSV mới. Mức AUC holdout lớn nhất trong các lượt ghi dưới đây là **0,9513400082**, từ hiệu chỉnh bậc hai trên một đối chứng khác, bản mật độ theo ngữ cảnh **0,9513122374**. Các số liệu đều đến từ kiểm chứng local. [Kết quả gốc](../goal_0952/evaluation.json) · [Kết quả bậc hai](../goal_0952/score_space/holdout_metrics.json) · [Kết quả số tiền](../goal_0952/deep_eda_measurement/measurement_metrics.json)

## 1. Cách đọc điểm giữa các lượt

| Phép đo | Dòng dùng để học và chấm | Cách diễn giải |
|---|---|---|
| 5 fold ngoài (OOF) của FINAL | 699.960 giao dịch, 11.689 ca gian lận. Mỗi dòng được chấm bằng dự đoán của mô hình đã chừa fold chứa dòng đó. | So ghép cặp cùng ID, nhãn và phép chia seed 42. Các seed 1337 và 2718 chia lại cùng bộ dữ liệu để kiểm tra mức nhạy theo cách chia. |
| Holdout Grandmaster seed 4242 | 594.966 dòng phát triển, 9.936 ca gian lận. Holdout 104.994 dòng, 1.753 ca. Trong phần phát triển, Grandmaster chia 10 fold phân tầng seed 2026 để lấy điểm nền. | Cùng tập giữ lại đã được dùng nhiều lần trong lịch sử chọn mô hình. Khoảng DeLong ghép cặp có điều kiện trên bộ dự đoán đã cố định. Việc thử nhiều phương án trên cùng holdout vẫn có thể làm chênh lệch quan sát lạc quan. |
| 10 fold đảo vai trò | Chấm lần lượt 10 fold của 594.966 dòng phát triển. Nhánh đang thử học trên 104.994 dòng tham chiếu tách riêng. Nền Grandmaster chừa fold đích. | Chênh lệch AUC thường là trung bình theo số cặp gian lận–hợp lệ của từng fold. Nhiều lượt chỉ so nhánh nền và hiệu chỉnh, bỏ LightGBM và spline. Kết quả phản ánh tác động ở quy mô tham chiếu nhỏ hơn. |
| Refit để xuất test hoặc khảo sát OOF toàn bộ | 699.960 dòng train và 299.984 dòng test chính thức. Điểm khảo sát Grandmaster phân cấp 0,9512809152 lấy từ 10 fold với ma trận OOF chung cho tầng gộp. | Đọc điểm khảo sát theo đúng thiết kế tầng gộp. CSV xuất test có thể xác minh ID và xác suất. Điểm trên nhãn test chưa được xác lập bởi các tệp local này. |

AUC (ROC AUC) đo khả năng xếp ca gian lận cao hơn giao dịch hợp lệ. AP (Average Precision) nhạy với độ chính xác khi rà soát từ điểm cao xuống. Log-loss phạt xác suất tự tin sai, càng thấp càng tốt. Log-odds là logarit của tỷ số \(p/(1-p)\). Trộn trên thang log-odds nghĩa là lấy tổng có trọng số rồi áp sigmoid để trả về xác suất. Các bảng dùng dấu phẩy thập phân. [Tạo và kiểm tra cách chia](../goal_0952/grandmaster_inputs.py) · [Metadata holdout](../grandmaster_10fold_ablation/validation/metrics.json) · [Kiểm chứng 5 fold](../novel_model_experiments/results/verification.json) · [Thiết kế đảo vai trò](../goal_0952/robust_validation/BAO_CAO_DO_ON_DINH.md)

## 2. Bảy thử nghiệm 5 fold sau FINAL

| Hướng | Giả thuyết và cách học | AUC | AP | Log-loss |
|---|---|---:|---:|---:|
| FINAL, đối chứng | Tái sử dụng dự đoán lồng nhau của tổ hợp hiện tại, giữ phép chia seed 42. | 0,95034521 | 0,52937620 | 0,04434665 |
| Trộn xác suất cố định | Mô hình khác họ có thể sửa các lỗi xếp hạng khác nhau. Lấy 70% FINAL, 20% LightGBM D03 và 10% MLP-BCE. | **0,95045126** | **0,53027823** | **0,04430444** |
| Trộn spline | Hồi quy trơn có thể sửa một số cặp xếp sai. Lấy 90% FINAL và 10% spline tương tác trên log-odds. Chọn tỷ lệ sau khi đã nhìn fold 1. | 0,95045057 | 0,53009202 | 0,04430608 |
| Hiệu chỉnh thời gian chính xác | Mốc giờ làm tròn đến 0,01 có thể còn tín hiệu. Dùng phần dư giữa ca thực tế và dự kiến từ điểm FINAL lồng nhau, so với lân cận ±0,20 giờ. Ridge 12, lấy nửa bước ước lượng và chặn bước thô ±0,35. | 0,95035247 | 0,52934534 | 0,04434682 |
| Lân cận đa độ rộng | Các mức chính xác có tần suất khác vùng lân cận. Với sáu biến số, dùng hai cửa sổ, tỷ lệ có làm trơn, sai lệch ở mức chính xác và số quan sát. | 0,95029222 | 0,52909739 | 0,04440046 |
| Tương tác theo lớp (copula) | Hai biến có thể đồng xuất hiện khác nhau theo lớp. Dùng 25 tỉ số hợp lý sau khi trừ phân bố biên và kéo các ô thưa về giả định độc lập. | 0,95022205 | 0,52865376 | 0,04442023 |
| Mật độ cặp chính xác (occupancy) | Một cặp có thể lặp bất thường trong cùng vùng thô. Đếm bảy cặp số và độ lệch so với tần suất kỳ vọng, bỏ chính dòng được mã hóa, rồi đưa vào LightGBM bốn lá. | 0,94986082 | 0,52528231 | 0,04457110 |
| Spline tương tác riêng | Logistic Ridge trên spline, chữ số và biến phân loại. Chọn tương tác tensor bằng phần dư ở tập học, so với spline cộng tính. | 0,94576566 | 0,50953742 | 0,04590274 |

- **Kiểm chứng tổ hợp cố định.** Δ AUC so với FINAL là +0,00010605, +0,00017334 và +0,00014130 tương ứng seed 42, 1337, 2718. Cả ba seed đều tăng ở 5/5 fold. Khoảng bootstrap ghép cặp 95% lần lượt [+0,00004118. +0,00017687], [+0,00010193. +0,00025526], [+0,00006071. +0,00022081]. Trên seed 42, riêng LightGBM 20% đạt 0,95040009, riêng MLP 10% đạt 0,95039776. [Trọng số và mã gộp](../novel_model_experiments/fixed_blend.py) · [Kết quả từng seed](../novel_model_experiments/results/fixed_blend_metrics.json) · CSV 70/20/10 (`novel_model_experiments/results/submission_fixed_blend.csv`)
- **So với cùng họ mô hình.** XGBoost CUDA đối chứng 1.650 cây đạt AUC 0,95017296, AP 0,52922561 và log-loss 0,04439629. Đặc trưng lân cận tăng AUC 0,00011926 ở 4/5 fold và copula tăng 0,00004908 ở 3/5 fold so với cây đó. Cả hai khoảng DeLong chứa 0. Đặc trưng có nhãn dùng 5 fold con seed 2026. Occupancy giảm 0,00009990 ở fold duy nhất có LightGBM đối chứng cùng cấu hình. Spline cộng tính đạt 0,94591771, thắng spline tương tác ở cả năm fold nhưng chạm giới hạn 220 vòng tối ưu. [Cây lân cận và copula](../novel_model_experiments/multiscale_tree.py) · [Occupancy](../novel_model_experiments/occupancy_tree.py) · [Spline và tương tác](../novel_model_experiments/alternative_model.py) · [Đối chứng và CSV](../novel_model_experiments/results/verification.json)
- **Diễn giải tín hiệu thời gian và spline.** Hiệu chỉnh giờ chỉ tăng AUC +0,00000726, giảm AP 0,00003087. Tín hiệu thời gian chính xác ở bước chẩn đoán theo nửa ID có tương quan phần dư Spearman 0,272, cao hơn số tiền 0,044 và khoảng cách 0,057 ở cách kiểm đó. Gộp spline tăng ở 5/5 fold dù spline tương tác riêng yếu hơn spline cộng tính. [Mã giờ](../novel_model_experiments/hours_residual.py) · [Kết quả giờ](../novel_model_experiments/results/hours_residual_metrics.json) · [Mã gộp spline](../novel_model_experiments/blend_spline.py) · [Kết quả gộp spline](../novel_model_experiments/results/spline_blend_metrics.json) · [Báo cáo và cả bảy CSV](../novel_model_experiments/BAO_CAO.md)

## 3. goal_0952 gốc và công thức 85/10/5

Bảy nhánh tái dựng từ checkpoint local là hai GLM dùng spline và phân cấp chữ số, ba cây nông hiệu chỉnh phần dư của GLM, một XGBoost nông có độ phân giải histogram cao và một EBM (Explainable Boosting Machine, mô hình cộng tính học bằng boosting). Tầng gộp ước lượng trọng số không âm có phạt L2 từ đầu ra ngoài fold. Nền *hierarchical_amount* dùng GLM có điều chuẩn theo các cấp của trường số tiền. [Thiết kế Grandmaster, mục 4–7](<../Model tham khảo/Grandmaster/FINAL_RESEARCH_REPORT.md>) · [Nền và cách gộp](../goal_0952/grandmaster_inputs.py) · [Manifest Grandmaster phân cấp](../grandmaster_10fold_ablation/submission_hierarchical_amount/manifest.json)

**Giả thuyết.** Phần dư của tổ hợp Grandmaster có thể còn tương tác giữa các trường. Sàng lọc 276 cặp trên 594.966 dòng phát triển bằng cách trừ ảnh hưởng từng biến và kiểm hai chiều học một nửa, chấm nửa kia. Có 32 cặp giảm log-loss xấp xỉ, chọn 12 cho cây và thêm chữ số cuối của khoảng cách. Các tín hiệu nổi bật gồm số giao dịch 24 giờ × loại thẻ, giờ từ lần trước × lệch quốc gia IP, CVV × loại thẻ, thiết bị × merchant mới. Các khóa chính xác giờ, velocity và điểm merchant chưa lặp lại theo phép sàng lọc. Vùng 90% điểm thấp chứa 1.594 ca gian lận, khoảng 16% số ca trên tập phát triển, tạo khoảng 76% số cặp xếp sai. Đó là căn cứ cho phép thử xếp hạng ưu tiên vùng thấp. [Sàng lọc và dải điểm](../goal_0952/residual_eda.json) · [Mã EDA](../goal_0952/eda_from_scratch.py)

**Cách học.** Offset là log-odds Grandmaster phân cấp dùng làm điểm xuất phát. XGBoost tối ưu mất mát logistic quanh điểm đó qua tham số base margin khi học và chấm. Ma trận có 109 cột D03, 12 cột cặp phần dư và một cột chữ số khoảng cách. Đặc trưng có nhãn cho dòng học tính từ fold con khác, điểm dùng để tính phần dư đến từ dự đoán ngoài fold. Cây chính dùng 224 vòng, sâu 3, eta 0,025, L2 40 và histogram 512 bin. Nhánh độc lập LightGBM D03 có bốn lá, học 1.600 vòng CPU với tốc độ 0,05. MLP-BCE chuẩn hóa các nhóm raw, target encoding, cents và digits, học mười fold và trung bình log-odds khi chấm. Spline tương tác dùng hồi quy logistic Ridge và chọn cặp trên phần học. Mô hình tạo sinh kiểm tra 12 hồ sơ giao dịch tiềm ẩn theo mỗi lớp, 26 vòng cập nhật và lấy mẫu tối đa 160.000 dòng lớp âm. [Cây offset](../goal_0952/residual_models.py) · [LightGBM](../goal_0952/lightgbm_branch.py) · [MLP](../goal_0952/mlp_branch.py) · [Spline](../goal_0952/spline_branch.py) · [Hồ sơ tiềm ẩn](../goal_0952/latent_profiles.py)

| Biến thể holdout | AUC | AP | Log-loss |
|---|---:|---:|---:|
| Grandmaster gốc | 0,95094666 | 0,53814886 | 0,04387461 |
| Grandmaster phân cấp số tiền | 0,95094893 | 0,53818509 | 0,04387376 |
| Cộng trực tiếp hiệu chỉnh EDA | 0,95092524 | 0,53768621 | 0,04389379 |
| XGBoost 224 vòng với đặc trưng cũ | 0,95104590 | 0,53773762 | 0,04387143 |
| XGBoost 96 vòng với EDA | 0,95103456 | 0,53822247 | 0,04385955 |
| XGBoost 224 vòng với EDA | 0,95112021 | 0,53803676 | 0,04385635 |
| XGBoost 448 vòng với EDA | 0,95110415 | 0,53754005 | 0,04387368 |
| XGBoost 672 vòng với EDA | 0,95111153 | 0,53696057 | 0,04388682 |
| XGBoost xếp hạng điểm thấp, 224 vòng | 0,95071530 | 0,53809183 | 0,04390306 |
| XGBoost 224 vòng với EDA, trung bình năm seed | 0,95106722 | 0,53803845 | 0,04386547 |
| MLP-BCE riêng | 0,94633580 | 0,52107990 | 0,04536046 |
| Spline tương tác riêng | 0,94693008 | 0,51673536 | 0,04534873 |
| Hồ sơ tiềm ẩn riêng | 0,93860801 | 0,43991406 | 0,04929144 |
| LightGBM D03 riêng | 0,95049680 | 0,53084652 | 0,04423601 |
| XGBoost 224 vòng + 3% hồ sơ tiềm ẩn, log-odds | 0,95114385 | 0,53799776 | 0,04385689 |
| XGBoost 224 vòng + LightGBM + MLP, xác suất 70/20/10 | 0,95118135 | 0,53866075 | 0,04383246 |
| **XGBoost 224 vòng + LightGBM + spline, log-odds 85/10/5** | **0,95122445** | 0,53823528 | 0,04383599 |

Thêm EDA vào cùng cây 224 vòng tăng AUC 0,00007431. Tổ hợp thắng trong 27 cấu hình, tăng 0,00027552 so với nền phân cấp. Khoảng DeLong 95% là [+0,00002221. +0,00052883] trước hiệu chỉnh do chọn cấu hình trên holdout. Các ca trong 90% điểm thấp đóng góp +0,00026051, gần 95% mức tăng ròng. Tổ hợp MLP 70/20/10 có AP cao hơn và log-loss thấp hơn trong cùng bảng. Mục tiêu đã đặt là AUC nên tệp gốc xuất theo 85/10/5. [Mã gộp và xuất tệp](../goal_0952/evaluate_and_export.py) · [Đủ 27 cấu hình](../goal_0952/evaluation.json) · [CSV Grandmaster EDA gốc](../goal_0952/submission_grandmaster_eda.csv)

## 4. Mật độ theo ngữ cảnh, cây thay thế và hiệu chỉnh chung

Lượt followup so 22 biến thể mới và giữ 10% LightGBM D03 cùng 5% spline. Mỗi thay đổi chỉ động đến nhánh 85%, theo công thức log-odds mới bằng điểm cũ cộng \(0,85\) nhân tỷ lệ thay và chênh lệch giữa cây mới với cây cũ. Mật độ theo lớp hỏi liệu một giá trị xuất hiện đặc biệt nhiều ở ca gian lận trong một ngữ cảnh giao dịch. Nhánh mở rộng dùng 10 ngữ cảnh, gồm CVV theo loại thẻ và xác thực, giờ kể từ lần trước theo IP và kênh, velocity theo số giao dịch và xác thực, số tiền theo ngành merchant và kênh. Mỗi lớp có phân bố chung làm mốc cho ngữ cảnh thưa. Các cột có nhãn dùng 5 fold con. Phép thử riêng thay cây bằng LightGBM hoặc CatBoost, tăng trọng số mất mát logistic ở vùng điểm thấp, hoặc dùng logistic Ridge trên tối đa 24 bảng ô để hiệu chỉnh chung và thêm thành phần riêng ưu tiên vùng dưới mốc 2,3% rủi ro. [Run và luật trộn](../goal_0952/followup/run_followup.py) · [Mật độ mở rộng](../goal_0952/followup/context_expansion.py) · [Hiệu chỉnh có cấu trúc](../goal_0952/followup/structured_offset.py)

| Biến thể holdout | AUC | AP | Log-loss |
|---|---:|---:|---:|
| XGBoost mật độ cũ | 0,95129466 | 0,53812586 | 0,04382983 |
| XGBoost lân cận đa độ rộng | 0,95118606 | 0,53807844 | 0,04385283 |
| XGBoost mật độ cũ và lân cận | 0,95126146 | 0,53816648 | 0,04384377 |
| XGBoost ưu tiên điểm thấp bằng trọng số | 0,95115070 | 0,53798830 | 0,04385528 |
| Hiệu chỉnh bằng LightGBM | 0,95120510 | 0,53828728 | 0,04384132 |
| Hiệu chỉnh bằng CatBoost | 0,95112573 | 0,53826918 | 0,04385863 |
| Gộp ba họ cây | 0,95124289 | 0,53838588 | 0,04383315 |
| Logistic Ridge dùng chung, nửa biên | 0,95119591 | 0,53765572 | 0,04386336 |
| Logistic Ridge ưu tiên điểm thấp, nửa biên | 0,95101509 | 0,53755899 | 0,04388940 |
| Logistic Ridge ưu tiên điểm thấp, toàn biên | 0,95061792 | 0,53602438 | 0,04400071 |
| Nhánh ngữ cảnh nhắm hẹp | 0,95121922 | 0,53828731 | 0,04383991 |
| Nhánh mật độ mở rộng theo EDA | 0,95131181 | 0,53834381 | 0,04382947 |
| **Trộn đều hai cây mật độ trong phần 85%** | **0,95131224** | 0,53828373 | **0,04382592** |

Bản gộp hai mật độ tăng AUC +0,00008779, dương ở cả hai nửa holdout chia phân tầng seed 6401 (+0,00003123 và +0,00014186). Khoảng DeLong 95% [−0,00003230. +0,00020788] chứa 0. Hồ sơ vòng cây đặt ngưỡng tăng +0,00010 và dương trên hai nửa để đổi đối chứng. Tệp ngữ cảnh được xuất làm ứng viên. [Kết quả cây và tỷ trọng](../goal_0952/followup/holdout_metrics.json) · [Kết quả hiệu chỉnh chung](../goal_0952/followup/structured_metrics.json) · [Kết quả mở rộng](../goal_0952/followup/context_metrics.json) · [Manifest ngữ cảnh](../goal_0952/followup/context_submission.json) · CSV ngữ cảnh (`goal_0952/followup/submission_context.csv`)

## 5. Ổn định theo cỡ mẫu và các dấu hiệu bị bỏ sót

**Điều chuẩn mật độ và cây.** Mỗi khoảng chia 20% số đếm cho từng ô bên cạnh, kéo ngữ cảnh hiếm mạnh hơn về phân bố chung, giảm độ sâu XGBoost 3 xuống 2, tăng L2 40 lên 60, đổi max_bin 512 xuống 256 và min_child_weight 35 lên 45, giữ 224 vòng. Trộn đều log-odds cây cũ và cây mới bên trong nhánh đang thử. Đối chứng chỉ làm trơn giữ nguyên cấu hình cây. Biến thể chính đạt holdout **0,95126523 / 0,53828048 / 0,04383366**, dưới mốc ngữ cảnh về cả ba chỉ số. Trong đảo vai trò, AUC trọng số tăng từ 0,95018999 lên 0,95025811, tăng ở 8/10 fold. AP gộp tăng từ 0,53066892 lên 0,53088297 và log-loss từ 0,04433998 xuống 0,04431729. Chỉ làm trơn đạt AUC 0,95129606 ở holdout và Δ AUC đảo vai trò +0,00000232. Lớp hiệu chỉnh xác suất logistic đơn điệu hai hệ số, đánh giá chéo 5 fold trên holdout, giảm Brier nhẹ nhưng làm log-loss gộp tăng lên 0,04383683. AUC từng fold giữ nguyên vì phép biến đổi đơn điệu. [Mã điều chuẩn](../goal_0952/robust_validation/robust_density.py) · [Holdout](../goal_0952/robust_validation/forward_metrics.json) · [10 fold](../goal_0952/robust_validation/reversed_metrics.json) · [Mã hiệu chỉnh](../goal_0952/robust_validation/calibrate.py) · [Kết quả hiệu chỉnh](../goal_0952/robust_validation/calibration_metrics.json) · CSV điều chuẩn (`goal_0952/robust_validation/submission_regularized.csv`)

**Đặc trưng bị bỏ sót.** Sàng 31 quy tắc trên hai nửa của 594.966 dòng, như số tiền vượt số dư, sự lệch giữa giờ và bộ đếm 24h, tuổi suy ra ở lúc mở thẻ và cờ nước ngoài khác quốc gia IP. Không quy tắc nào giảm log-loss cả hai chiều khi hiệu chỉnh riêng theo nhóm. Các runner sau đó thử để cây học từ 16 quan hệ số và thời gian, hồ sơ đồng cảnh (phân vị, độ hiếm và mật độ theo lớp với nhóm cha cho nhóm thưa), và 15 cặp chữ số cuối (độ hiếm dưới giả định độc lập và phần dư sau khi trừ hiệu ứng từng chữ số). Mỗi ứng viên dùng 224 vòng, thay đúng cây mở rộng chiếm 42,5% log-odds trong bản ngữ cảnh, với nhãn mã hóa từ 5 fold con. [Kiểm tra dữ liệu](../goal_0952/data_signals/data_audit.json) · [Mã đặc trưng](../goal_0952/data_signals/overlooked_features.py) · [Mã chạy](../goal_0952/data_signals/run_signals.py)

| Thay cây mở rộng | AUC | AP | Log-loss |
|---|---:|---:|---:|
| Đối chứng ngữ cảnh | 0,95131224 | 0,53828373 | 0,04382592 |
| Quan hệ số và thời gian | 0,95129338 | 0,53831928 | 0,04382754 |
| Hồ sơ đồng cảnh | 0,95129283 | **0,53861055** | **0,04381643** |
| Cặp chữ số cuối | 0,95129325 | 0,53806553 | 0,04383836 |
| Cả ba nhóm | 0,95125725 | 0,53824256 | 0,04383429 |

Hồ sơ đồng cảnh giảm log-loss 7/10 fold đảo vai trò và tăng AUC 5/10 fold, với Δ AUC trọng số khoảng −0,000007. Hai bản giữ được 706 ca trong 1.050 giao dịch đầu. Nhóm CVV tương đối cao trong cùng xác thực, thiết bị và kênh có 11.059 dòng holdout và giảm log-loss 0,00008126. Dấu hiệu phù hợp với cải thiện xác suất ở một số nhóm, trong khi AUC cả tập vẫn thấp hơn đối chứng. [Holdout](../goal_0952/data_signals/holdout_metrics.json) · [Đảo vai trò](../goal_0952/data_signals/reversed_metrics.json) · [Phân tích nhóm](../goal_0952/data_signals/peer_diagnostics.json) · CSV hồ sơ đồng cảnh (`goal_0952/data_signals/submission_peer_profiles.csv`)

## 6. Học trong không gian điểm và dùng nhãn mềm

**Bất đồng giữa bảy mô hình.** Trong tập phát triển, nhóm từ phân vị 80–95 về mức bất đồng có 2.808 ca so với 2.745,86 dự kiến. Phép thử lấy các chênh lệch giữa bảy đầu ra và nền Grandmaster, loại phần tuyến tính đã có, rồi làm hồi quy phần dư có điều chuẩn trên tích bậc hai hoặc độ gần 48 cụm điểm. Khi cộng vào bản ngữ cảnh, lấy nửa mức hiệu chỉnh và giới hạn thay đổi log-odds ở khoảng 0,09. Hai bản cụm lấy điểm đầu vào bình quân hoặc lấy bình quân hiệu chỉnh từng fold. [Mã mô hình không gian điểm](../goal_0952/score_space/score_space.py)

**Nhãn mềm.** Dùng 80% nhãn nhị phân thực và 20% xác suất ngoài fold của Grandmaster làm mục tiêu cho XGBoost 224 vòng, giữ nguyên đặc trưng và cấu hình. Mỗi phương án thay đúng cây mật độ mở rộng, chiếm 42,5% log-odds của cả tổ hợp. Đối chứng sau huấn luyện co log-odds cây về nền Grandmaster 20%. Cả hai tăng AUC và giảm log-loss 10/10 fold khi nhánh học trên 104.994 dòng tham chiếu. Δ AUC trọng số là +0,00012738 cho nhãn mềm và +0,00011889 cho co biên, khoảng 93% mức tăng của nhãn mềm. Trên holdout của toàn tổ hợp học từ 594.966 dòng, cả hai giảm AUC. [Mã nhãn mềm](../goal_0952/score_space/soft_targets.py) · [Kết quả nhãn mềm 10 fold](../goal_0952/score_space/soft_reversed_metrics.json)

| Biến thể holdout | AUC | AP | Log-loss |
|---|---:|---:|---:|
| Đối chứng ngữ cảnh | 0,95131224 | 0,53828373 | 0,04382592 |
| **Hiệu chỉnh bậc hai** | **0,95134001** | **0,53864400** | **0,04381021** |
| Cụm từ điểm trung bình | 0,95130161 | 0,53844498 | 0,04382359 |
| Cụm từng fold rồi lấy trung bình | 0,95130037 | 0,53839768 | 0,04382380 |
| Co biên cây 20% | 0,95129698 | 0,53834061 | 0,04382538 |
| Nhãn mềm Grandmaster | 0,95129512 | 0,53838016 | 0,04382499 |

Bậc hai tăng AUC holdout +0,00002777, với DeLong 95% [−0,00004221. +0,00009775]. Kiểm đảo vai trò tăng AUC 4/10 fold, giảm log-loss 6/10 fold và giảm AUC trọng số 0,00002963. Báo cáo lượt này giữ ngữ cảnh làm lựa chọn chính trong họ mô hình ngữ cảnh. [Holdout bất đồng](../goal_0952/score_space/holdout_metrics.json) · [Đảo vai trò](../goal_0952/score_space/reversed_metrics.json) · [Holdout nhãn mềm](../goal_0952/score_space/soft_holdout_metrics.json) · CSV bậc hai (`goal_0952/score_space/submission_quadratic.csv`) · CSV nhãn mềm (`goal_0952/score_space/submission_soft_targets.csv`)

## 7. Quay lại Grandmaster EDA gốc, cải tiến bảng mã hóa

**Giả thuyết.** Cây gốc khi học thấy bảng cặp ước lượng từ 80% tập tham chiếu, trong khi khi chấm dùng bảng 100%. Có thể đổi biên phân vị và độ hỗ trợ theo kích thước bảng. Biến thể *matched encoding size* chấm chính cây gốc bằng năm bảng 80% rồi lấy trung bình log-odds. Biến thể làm mượt phân 10% thống kê của mỗi ô số cho từng ô liền kề, giữ ô phân loại riêng và dùng co mềm theo độ hỗ trợ. Bản phối 50/50 lấy trung bình cây gốc và cây đã làm mượt trong trọng số 85%. Giữ 109 cột D03, 12 cặp, chữ số khoảng cách và cấu hình XGBoost 224 vòng sâu 3. [Ba biến thể và mã](../goal_0952/grandmaster_eda_refinement/refine_eda.py)

| Biến thể holdout | AUC | AP | Log-loss | Δ AUC 10 fold đảo vai trò |
|---|---:|---:|---:|---:|
| Grandmaster EDA gốc | **0,95122445** | **0,53823528** | **0,04383599** | — |
| Trung bình năm bảng 80% | 0,95121087 | 0,53815394 | 0,04383945 | +0,00005404 |
| Làm mượt bảng cặp | 0,95121100 | 0,53810391 | 0,04384148 | −0,00006778 |
| Trộn nửa cây làm mượt | 0,95121931 | 0,53815166 | 0,04383819 | −0,00002392 |

Trung bình năm bảng tăng AUC 7/10 và giảm log-loss 8/10 fold đảo vai trò. Trên holdout, khoảng DeLong 95% của Δ AUC là [−0,00004510. +0,00001794]. Nhóm ca gian lận ở phân vị rủi ro 75–90 đóng góp −0,00001270 AUC. Hai bản mượt chỉ tăng AUC ở 1/10 fold đảo vai trò. [Holdout](../goal_0952/grandmaster_eda_refinement/holdout_metrics.json) · [10 fold](../goal_0952/grandmaster_eda_refinement/reverse_metrics.json) · [Dải điểm](../goal_0952/grandmaster_eda_refinement/score_band_diagnostics.json) · [CSV và cách tái lập](../goal_0952/grandmaster_eda_refinement/BAO_CAO_GRANDMASTER_EDA.md)

## 8. Tương tác theo rủi ro và mười cặp EDA bổ sung

**Thay đổi theo rủi ro.** Với 12 cặp cũ, ước lượng phần độ dốc của sai lệch theo rủi ro nền bằng hàm tanh quanh xác suất 3%. Tách độ dốc khỏi hiệu ứng trong ô và khỏi tác động đơn biến. Cross-fitting 5 fold giúp mỗi dòng học nhận thống kê có nhãn từ các fold khác. Nhánh phụ chọn 10 cặp chưa dùng có mức giảm log-loss xấp xỉ từ 0,000001 trong sàng lọc cũ. Ví dụ số lần tranh chấp × lệch quốc gia IP, kênh × VPN, CVV × cờ lừa đảo AI. Tổng phần dư của 10 bảng trở thành một cột. Hai nhóm được thử bằng cách thêm vào XGBoost, phối nửa nhánh trong trọng số 85% hoặc giữ cây cũ và cộng nửa biên Ridge tuyến tính. Hệ số của tổng 10 cặp giới hạn trong [0, 1]. [Mã huấn luyện](../goal_0952/grandmaster_eda_risk/risk_eda.py) · [Holdout rủi ro](../goal_0952/grandmaster_eda_risk/development_metrics.json) · [Holdout 10 cặp](../goal_0952/grandmaster_eda_risk/extra_development_metrics.json)

| Biến thể holdout | AUC | AP | Log-loss | Δ AUC 10 fold đảo vai trò |
|---|---:|---:|---:|---:|
| Rủi ro × EDA, cây | 0,95112217 | 0,53804466 | 0,04385380 | −0,00004679 |
| Rủi ro × EDA, nửa cây | 0,95117764 | 0,53815946 | 0,04384304 | −0,00000589 |
| Rủi ro × EDA, tuyến tính | 0,95122158 | 0,53823341 | 0,04383549 | −0,00000083 |
| 10 cặp bổ sung, cây | 0,95123132 | **0,53847140** | **0,04382906** | −0,00000988 |
| 10 cặp bổ sung, nửa cây | 0,95122902 | 0,53836860 | 0,04383213 | −0,00000194 |
| 10 cặp bổ sung, tuyến tính | **0,95123214** | 0,53822991 | 0,04383585 | +0,00000014 |

Ridge tuyến tính cho 10 cặp có Δ AUC holdout +0,00000769, khoảng DeLong 95% [−0,00000233. +0,00001770]. Ở 10 fold đảo vai trò, bản đó tăng nhẹ ba fold, bảy fold bằng đối chứng. Hệ số Ridge là 0,1808 khi học trên tập phát triển và 0,2035 khi refit để xuất. Đối chứng xáo một lần trong 20 nhóm rủi ro giữ phân phối cột, phá liên hệ với giao dịch. AUC của cây có cột xáo là 0,95120693 và cột thật là 0,95123132. Khoảng tin cậy của chênh lệch chứa 0. [10 fold rủi ro](../goal_0952/grandmaster_eda_risk/reverse_metrics.json) · [10 fold cặp bổ sung](../goal_0952/grandmaster_eda_risk/extra_reverse_metrics.json) · [Mã xáo](../goal_0952/grandmaster_eda_risk/audit_extra.py) · [Kết quả xáo](../goal_0952/grandmaster_eda_risk/extra_placebo_metrics.json) · [CSV tuyến tính và năm biến thể khác](../goal_0952/grandmaster_eda_risk/BAO_CAO_TOI_UU_EDA.md)

## 9. Ba hướng thử song song trên Grandmaster EDA gốc

1. **Độ phân giải trong ô EDA.** Chia đôi ô phân vị của sáu cặp số × nhóm, bỏ phần dư chung của ô và biến động chung của các nhóm. Co theo độ hỗ trợ của hai nửa. Thử đặc trưng tổng sáu độ lệch cho cây, phối nửa nhánh và Ridge tuyến tính. [Mã](../goal_0952/parallel_eda_resolution/resolution_eda.py) · [Holdout](../goal_0952/parallel_eda_resolution/development_metrics.json) · [10 fold](../goal_0952/parallel_eda_resolution/reverse_metrics.json)
2. **Hồ sơ ngữ cảnh hai biến.** Tạo sáu cột từ hai cặp *velocity × thời gian từ giao dịch trước* và *số tiền × điểm rủi ro merchant*. Chuẩn hóa theo thứ hạng, ước lượng tương quan trên giao dịch hợp lệ và co nhóm thưa về nhóm cha. Ba cột mỗi cặp đo độ hiếm chung, sai lệch có hướng và tích điểm thứ hạng. Thử thay toàn bộ hoặc một nửa nhánh cây. [Mã đặc trưng](../goal_0952/parallel_peer_context/joint_features.py) · [Mã chạy](../goal_0952/parallel_peer_context/run_joint_context.py) · [Holdout](../goal_0952/parallel_peer_context/holdout_metrics.json) · [10 fold](../goal_0952/parallel_peer_context/reverse_metrics.json)
3. **Cây cộng đơn biến.** Giữ 122 cột và điểm nền, thử LightGBM hai lá 600 vòng rồi thay 15% nhánh XGBoost, tương đương 12,75% log-odds của cả tổ hợp. Cây một nút tách (decision stump, mỗi cây dùng một biến) được so với LightGBM tối đa năm lá và co phần hiệu chỉnh XGBoost về điểm nền 15%. [Mã cây](../goal_0952/parallel_tree_calibration/compare_offset_trees.py) · [Holdout](../goal_0952/parallel_tree_calibration/holdout_metrics.json) · [10 fold](../goal_0952/parallel_tree_calibration/reverse_metrics.json) · [Mã đối chứng](../goal_0952/parallel_eda_resolution/audit_stump_shrink.py)

| Biến thể holdout | AUC | AP | Log-loss | Δ AUC 10 fold đảo vai trò | Fold tăng AUC |
|---|---:|---:|---:|---:|---:|
| Chi tiết phân vị, cây | 0,95120508 | 0,53822095 | 0,04383828 | −0,00001371 | 4/10 |
| Chi tiết phân vị, nửa cây | 0,95121552 | 0,53826113 | 0,04383702 | −0,00000320 | 5/10 |
| Chi tiết phân vị, tuyến tính | 0,95122403 | 0,53822355 | 0,04383642 | −0,00000016 | 3/10 |
| Ngữ cảnh hai biến, cây | 0,95113668 | **0,53908200** | **0,04380857** | +0,00000914 | 4/10 |
| Ngữ cảnh hai biến, nửa cây | 0,95118936 | 0,53870944 | 0,04381704 | +0,00000743 | 6/10 |
| **LightGBM một nút tách** | **0,95122558** | 0,53823066 | 0,04383578 | +0,00005627 | 10/10 |
| LightGBM tối đa năm lá | 0,95122496 | 0,53821075 | 0,04383563 | +0,00004212 | 10/10 |
| Co 15% phần hiệu chỉnh XGBoost | 0,95121511 | 0,53831307 | 0,04383394 | **+0,00013460** | 10/10 |
| Trộn đều ba chênh lệch | 0,95121425 | 0,53838517 | 0,04382859 | +0,00002271 | 9/10 |

LightGBM một nút tách tăng AUC holdout +0,00000113, khoảng DeLong 95% [−0,00001580. +0,00001806]. Co biên thắng nó ở cả 10 fold đảo vai trò nhưng giảm AUC holdout −0,00000934 so với Grandmaster EDA gốc. Nhánh ngữ cảnh đầy đủ tăng AP 0,00084672 và giảm log-loss 0,00002742 trên holdout. Bản nửa nhánh tìm 706 ca trong 1.050 giao dịch đầu, so với 704 của bản gốc. Phép phối ba hướng lấy trung bình chênh lệch log-odds từ phân vị tuyến tính, nửa nhánh ngữ cảnh và LightGBM một nút tách. Log-loss giảm trong cả 10 fold đảo vai trò, còn AUC holdout thấp hơn gốc. [Đối chứng co biên](../goal_0952/parallel_eda_resolution/stump_shrink_audit.json) · [Mã trộn](../goal_0952/parallel_eda_resolution/combine_paths.py) · [Kết quả trộn](../goal_0952/parallel_eda_resolution/parallel_three_way_metrics.json) · [Danh sách chín CSV](../goal_0952/BAO_CAO_THU_SONG_SONG.md)

## 10. EDA sâu và bản xuất hiệu chỉnh mức tiền thưa

**Xác định nhóm trước khi xem nhãn.** Trên 594.966 dòng phát triển, đếm tần suất theo cent. Với một mức tiền \(t\), lấy trung bình số đếm của các mức lân cận trong hai cửa sổ ±0,05 và ±0,25 USD, bỏ \(t\) và tính cả mức có số đếm 0. Giữ mức có ít nhất 10 dòng, bình quân lân cận ít nhất 40 và cả hai tỷ số \((n_t+5)/(\bar n+5)\) không quá 0,4. Loại vùng 25 cent sát hai biên của miền quan sát. Quy tắc giữ 404 mức, ví dụ 9,84 và 10,06 USD. [Quy tắc và tiêu chí](../goal_0952/deep_eda_measurement/METHOD.md) · [Mã tạo và chấm nhóm](../goal_0952/deep_eda_measurement/measurement_eda.py)

**Tập học và phép đo.** Hai nửa phát triển gồm các fold 1–5 và 6–10 trên nền Grandmaster phân cấp ngoài fold. Nhóm số tiền có 85 ca trên 4.313 dòng (74,20 dự kiến) và 106 ca trên 4.368 dòng (81,34 dự kiến). Sàng 82 nhóm, 45 đủ mẫu. Nhóm 404 mức là nhóm duy nhất giảm log-loss cả hai chiều và vượt ngưỡng hiệu chỉnh đa kiểm định Bonferroni (p = 0,000316 với ngưỡng 0,001111). Dùng hiệu chỉnh logistic Ridge 25, giới hạn bước ±0,30. Hệ số học từ phát triển +0,27198694 được cộng vào log-odds của **Grandmaster EDA gốc** trên đúng 1.644 dòng holdout thuộc nhóm. Quan sát 46 ca so với 31,86 ca mà đối chứng dự kiến. [Số liệu sàng lọc và holdout](../goal_0952/deep_eda_measurement/measurement_metrics.json)

| Tổ hợp holdout | ROC AUC | Average Precision | Log-loss |
|---|---:|---:|---:|
| Grandmaster EDA gốc | 0,9512244494 | 0,5382352787 | 0,0438359879 |
| Grandmaster EDA + mức tiền thưa | **0,9513158455** | **0,5385228224** | **0,0438060217** |

Chênh lệch AUC +0,00009140, khoảng DeLong 95% [−0,00002229. +0,00020508]. Các ca ở phân vị điểm nền 75–90 đóng góp +0,00006346. Bỏ lần lượt từng mức cho mức giảm log-loss từ 0,00002318 đến 0,00003267. Bootstrap theo mức tiền cho khoảng [+0,00000161. +0,00006008] của mức giảm log-loss. Hoán vị trong cùng dải 10 USD, dải điểm nền và chữ số cent cuối cho p điều kiện 0,002. Những kiểm tra này mô tả độ nhạy trên holdout đã được dùng lại. Cả 404 mức tiếp tục thưa theo hai cửa sổ trên test, nơi dữ liệu chỉ cho phép đo tần suất. [Độ nhạy](../goal_0952/deep_eda_triples/amount_hole_sensitivity.json) · [Mã xác minh](../goal_0952/deep_eda_triples/audit_amount_holes.py)

**Các hướng âm.** Điểm khoảng cách thưa có dấu cùng chiều trên hai nửa nhưng p = 0,00690 vượt ngưỡng 0,001111. Chuyển ngày hoặc tuần × kênh và số dư trừ số tiền × loại thẻ làm AUC holdout giảm 0,00013785 và 0,00009981. Sàng 16 bộ ba sau khi trừ cả ba hiệu ứng cặp, ba bộ lặp lại trong A→B và B→A đều làm log-loss xấu đi trên phần C riêng (A = 237.986, B = C = 178.490 dòng). Ví dụ số giao dịch 24 giờ × giờ từ lần trước × kênh giảm log-loss 2,63 và 2,31 phần triệu trong hai chiều sàng lọc, rồi tăng 7,96 phần triệu trên C. [Mã bộ ba](../goal_0952/deep_eda_triples/triple_eda.py) · [Kết quả bộ ba](../goal_0952/deep_eda_triples/triple_eda_results.json) · [Báo cáo EDA sâu](../goal_0952/BAO_CAO_EDA_SAU.md)

**Xuất test.** Giữ nguyên 404 mức, học một hệ số trên đủ 699.960 dòng train bằng điểm Grandmaster phân cấp ngoài fold. Hệ số xuất là **+0,2872908173 log-odds**, tương ứng nhân tỷ lệ cược **1,3328117618** trên 4.448 dòng thuộc nhóm. Bản xuất có 299.984 dòng, cột `id,is_fraud`, đúng thứ tự mẫu. Manifest ghi SHA-256 `7068b053cb3e39a07b544c95d9f66b98958e0722d8669a82f465284062af1287`. Mốc 0,952 còn cách 0,0006841545 trong phép đo này. Đây là bản hiệu chỉnh số tiền trên Grandmaster EDA gốc được ưu tiên trong lượt EDA sâu. Bản bậc hai đạt AUC cao hơn trên cùng holdout nhưng dùng nền mật độ và tín hiệu khác. [Mã xuất và kiểm tra](../goal_0952/deep_eda_measurement/export_amount_hole.py) · [Manifest](../goal_0952/deep_eda_measurement/amount_hole_submission.json) · [CSV số tiền thưa](../goal_0952/deep_eda_measurement/submission_grandmaster_eda_amount_hole.csv)

## Phạm vi nguồn đã đọc

- Đọc toàn bộ `novel_model_experiments/BAO_CAO.md` và các báo cáo `goal_0952/BAO_CAO_0952.md`, `followup/BAO_CAO_FOLLOWUP.md`, `robust_validation/BAO_CAO_DO_ON_DINH.md`, `data_signals/BAO_CAO_DAC_DIEM.md`, `score_space/BAO_CAO_HUONG_KHAC.md`, `grandmaster_eda_refinement/BAO_CAO_GRANDMASTER_EDA.md`, `grandmaster_eda_risk/BAO_CAO_TOI_UU_EDA.md`, `BAO_CAO_THU_SONG_SONG.md`, `BAO_CAO_EDA_SAU.md` và `deep_eda_measurement/METHOD.md`.
- Đối chiếu trường kết quả OOF, ứng viên, nhánh độc lập, mẫu học/chấm và công thức chọn trong JSON được dẫn ở từng phần. Phần số tiền đọc trường `holdout.groups.amount_usd_hole`, các mục độ nhạy theo mức tiền, đóng góp dải điểm và manifest xuất. Đọc các đoạn tạo fold và gộp trong `grandmaster_inputs.py`, `evaluate_and_export.py`, cấu hình mô hình và các đoạn được dẫn trong từng runner. Tài liệu này dùng các kết quả đã lưu của dự án. Các cấu hình và phạm vi đọc sâu hơn nằm trong báo cáo gốc dẫn tại từng giai đoạn.
