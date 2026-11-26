# Lịch sử các cuộc chat và quyết định theo phase

[Về README](README.md) · [Baseline và Grandmaster](01_BASELINE_VA_GRANDMASTER.md) · [Mô hình và kết quả](04_MO_HINH_VA_THU_NGHIEM.md)

## Phạm vi tổng hợp

Tổng hợp các cuộc chat liên quan trực tiếp đến Predicting Credit Card Fraud trong dự án từ 19 đến 26/11/2026, theo giờ Việt Nam. Đã duyệt các trang lịch sử trả về cho **132 lượt**, gồm cả lượt tổng hợp hiện tại. Rà nội dung mở đầu, điểm và tên hiện vật của **125 câu trả lời cuối**, đồng thời đọc kỹ các đoạn quyết định và câu hỏi hiển thị liên quan. Các số liệu được đối chiếu với mã, báo cáo, JSON, CSV và manifest đã lưu. [Chỉ mục nguồn theo dòng](05_BANG_DIEM_VA_ARTEFACT.md)

| Tên chat | Khoảng ngày có lượt | Phần đã đọc |
|---|---|---|
| `Draft Predicting Credit Card Fraud rules` | 19/11 | 5 lượt, 5 câu trả lời cuối. Giai đoạn chốt dữ liệu hợp lệ, vai trò trợ lý và quy tắc project. |
| `Tóm lược hai file Markdown` | 19–22/11 | 81 lượt, 78 câu trả lời cuối. Từ plan, notebook, chạy smoke/GPU đến 37 kết quả, FINAL, bảng 68 cấu hình, lệch nhãn, tiền huấn luyện, EDA/FE, nhiều seed và refit. |
| `Cải thiện điểm dự đoán gian lận` | 22/11 | 4 lượt, 4 câu trả lời cuối. Ba hướng blend/calibration, chấm hai seed và xuất ba tệp. |
| `Tóm tắt model tham khảo` | 22–24/11 | 30 lượt, 28 câu trả lời cuối. So V1/V2/Grandmaster, tái lập bảy nhánh, thử phân cấp, mô hình chuyên biệt, mật độ và xếp hạng. |
| `Nâng cấp giải pháp Kaggle Fraud` | 24–26/11 | 12 lượt, 10 câu trả lời cuối. Bảy hướng, mục tiêu local 0.952, các nhánh EDA/độ ổn định và ứng viên số tiền thưa. |

Các mốc dưới đây ghi việc có kết quả hoặc hiện vật đối chiếu. Chat từng có lượt ngắt hoặc ý tưởng dừng ở đề xuất. Những lượt chạy lại được ghi ở mốc có số liệu hoàn tất. [Ví dụ nhật ký kết quả đã xuất bản](../../s6e11_gpu/plan_runs/published/full/comparison.md)

## Các mốc thực hiện

### Phase A · Chuẩn bị dữ liệu, quy tắc và baseline

- Dựng tên cột, audit, checksum, folder dữ liệu, fold phân tầng và hợp đồng đầu ra OOF/test. Kế hoạch đánh số bước 0–8 cho giai đoạn đầu. Baseline có các họ B00–B07. Từ đó so độ phức tạp của cây, phương pháp mã hóa và các cấu hình T01–T03 ở bước 8. [Kế hoạch và nhật ký](01_BASELINE_VA_GRANDMASTER.md) · [Runner nền](../../s6e11_project_run.py)
- Thử năm notebook trên dữ liệu giả lập ở giai đoạn kiểm tra hạ tầng và tự xác minh không trộn dữ liệu này với lượt thi. Kết quả giả lập có test riêng. Sau khi quy tắc được chốt, pipeline thi dùng dữ liệu S6E11 và mã ML truyền thống hoặc mạng nơ-ron cho dữ liệu bảng. [Kết quả giả lập](05_BANG_DIEM_VA_ARTEFACT.md) · [Quy tắc project](../README.md#quy-tắc-dự-án)
- Tổ chức lại repo con và đặt tên notebook theo phương pháp. Lịch sử báo cáo ghi mốc baseline ở `ba8a24d`, báo cáo GPU ở `5e3026b` và danh mục mã ở `a1ce856`. Các mã commit ghi lịch sử. Tài liệu này dùng cây kết quả hiện có trên máy. [Danh mục notebook](../../s6e11_gpu/model_notebooks/README.md) · [Nhật ký tổ chức](../../reports/reorganization/README.md)

### Phase B · GPU, đặc trưng, FINAL và refit

- Chạy năm notebook vòng 1 gồm Logistic + TE, LightGBM + TE, XGBoost + TE, MLP, FT-Transformer. Vòng 2 thêm CatBoost TE, TabM Mini, XGBoost pair TE và rank blend. Tạo tài liệu giải thích đường đi OOF và test. [GPU vòng 1](../s6e11_gpu/results/comparison.md) · [GPU vòng 2](../s6e11_gpu/round2/results/comparison.md) · [Sơ đồ OOF](../.visualizations/s6e11-oof-flow.html)
- Mở rộng từ mô hình thô sang TE, chữ số D01/D02/D03, count, GLM spline và cây học sai lệch. Chấm khoảng 30 bài chạy huấn luyện và lưu 37 kết quả tính cả tổ hợp. XGB-D03 dẫn nhóm đơn. Hai nhánh GLM-NEWTON + XGB-D03 tạo `FINAL` đạt AUC 0.95034521 trong xác nhận lồng 5×5. Tổ hợp ba nhánh được thử và xuất riêng. [Bảng 37 dòng](../../s6e11_gpu/plan_runs/published/full/comparison.csv) · [Tổ hợp đã khảo sát](../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv)
- Cân nhắc lệch nhãn bằng trọng số, lấy mẫu và bộ loss MLP. Thử masked reconstruction, SCARF và nối embedding. Đánh giá các đặc trưng theo EDA. Nhiều tập cột bổ sung làm giảm AUC so với D03 hoặc FINAL. Danh sách 68 mô hình đơn ghi xác nhận từ dự đoán và nguồn báo cáo. [Lệch nhãn](../s6e11_gpu/imbalance/ket_qua_thu_nghiem_lech_nhan.md) · [Tiền huấn luyện](../s6e11_gpu/pretraining/results/summary.md) · [Ablation](../s6e11_gpu/final_feature_optimization/results/ket_qua_toi_uu_dac_trung_final.md) · [68 mô hình đơn](../s6e11_gpu/xep_hang_mo_hinh_don.md)
- Thử FINAL cộng 20% LightGBM, cộng 10% MLP và hiệu chỉnh nhóm không cờ. Kiểm tra lại cả hai seed 1337/2718 bằng 5 fold ngoài, 3 fold trong. 80/20 đạt trung bình AUC 0.95043460 và tăng ở 9/10 fold. Lần tiếp theo có XGBoost SEED5-MEAN, refit sáu mô hình đơn, refit năm tổ hợp và LightGBM SEED5 riêng. Seed chia fold, seed encoder, seed mô hình và refit toàn bộ có hồ sơ khác nhau. [Hai seed](../s6e11_gpu/followup/results/full/seed_1337/metrics.json) · [XGBoost nhiều seed](../s6e11_gpu/seed3_improvement/ket_qua_seed3_mean.md) · [Sơ đồ SEED5](../.visualizations/seed5-mean-workflow.html) · [Refit blend](../s6e11_gpu/blend_full_refit/results/summary.json)

### Phase C · Tham khảo V1/V2/Grandmaster và cải tiến có đối chứng

- So kho tham khảo theo nguồn. V1 gộp rank 45/35/20. V2 (tên gốc V3) thêm TE lồng, offset boosting và calibration. Grandmaster dùng hai GLM, ba cây offset, XGBoost riêng và EBM. Các điểm 0.95069, 0.95092 và 0.95154 là Public LB ghi trong kho. Nhóm tham khảo dùng để đọc kiến trúc và thiết kế đối chiếu. [Bảng phiên bản](<../Model tham khảo/VERSION_COMPARISON.md>)
- Xem xét nguồn của một bản tham khảo Board Model + RealMLP có Public LB 0.95553. Chat ghi nhận tổ hợp đó dùng lại dự đoán của bên khác và GPT-2. Quy tắc dự án dựa vào dữ liệu cuộc thi và pipeline riêng, nên mốc đó chỉ được ghi trong cuộc đọc tham khảo. [Quy tắc dự án](../README.md#quy-tắc-dự-án)
- Tái lập Grandmaster trên máy, refit 7 mô hình và tầng gộp. Sau đó chủ nhân chọn mô hình 10 fold làm đối chứng để cải tiến. Tách holdout 15%, so residual TE, hierarchical amount và aligned offset. Nhánh hierarchical amount tạo AUC 0.95094893 trên holdout so với 0.95094666 của mốc. [Tái lập](../grandmaster_refit/full/metrics.json) · [Ablation](../grandmaster_10fold_ablation/validation/metrics.json)
- Tách bạch dự đoán holdout 104.994 dòng và submission đủ 299.984 dòng sau câu hỏi về tệp Kaggle. Chat ghi chủ nhân báo các lượt nộp lúc đó cùng hiển thị 0.95154. Đây là thông tin do chủ nhân báo, bên cạnh 0.95154 Public LB của kho tham khảo. Mục tiêu 0.953 được đặt trong cuộc chat đó. Vòng nghiên cứu kế tiếp đặt mốc 0.952 cho phép đo local. [Đúng tệp nộp](06_TAI_LAP_VA_SUBMISSION.md)
- Sau phân cấp, thử tầng meta theo rủi ro, hiệu chỉnh sau stack, bộ xếp hạng cho giao dịch không bật cờ, GLM phân cấp có tương tác, trộn mật độ theo lớp vào cây và bộ học ưu tiên điểm thấp. Thay đổi hữu ích nhất ở cây riêng ít chuyển sang ensemble. Các kết quả dùng cùng holdout hoặc giao thức riêng được ghi ở nguồn. [Thử nghiệm Grandmaster](01_BASELINE_VA_GRANDMASTER.md) · [Kết quả nhánh](05_BANG_DIEM_VA_ARTEFACT.md)

### Phase D · Bảy hướng, mục tiêu local 0.952 và chốt Grandmaster EDA

- Chạy bảy hướng độc lập và xuất 7 CSV ở lượt ban đầu. Tổ hợp 70% FINAL + 20% LightGBM D03 + 10% MLP-BCE đạt OOF 0.95045126 với seed 42. Spline, lân cận, mật độ, occupancy và phần dư thời gian được đối chiếu theo cùng thiết kế 5 fold. [Bảy hướng](../novel_model_experiments/BAO_CAO.md)
- Khi mục tiêu chuyển thành 0.952, làm EDA lại từ CSV, khảo sát 276 cặp biến, chọn 12 tương tác và thêm tín hiệu chữ số cuối của khoảng cách. Chấm 27 cấu hình, thử cây 96–672 vòng, bagging và loss xếp hạng. Grandmaster EDA 85/10/5 đạt holdout **0.95122445**. [EDA và 27 công thức](../goal_0952/BAO_CAO_0952.md) · [Kết quả](../goal_0952/evaluation.json)
- Bổ sung mật độ và hồ sơ lân cận, thử XGB/LightGBM/CatBoost offset và hiệu chỉnh cấu trúc. Expanded half-density đạt 0.95131224. Chủ nhân nhấn mạnh tránh quá khớp và tránh tối ưu cực đoan AUC. Thêm phép 10 fold đảo vai trò, các đối chứng đơn giản, AP, log loss và khoảng tin cậy ghép cặp. [Follow-up](../goal_0952/followup/BAO_CAO_FOLLOWUP.md) · [Độ ổn định](../goal_0952/robust_validation/BAO_CAO_DO_ON_DINH.md)
- Thử góc nhìn khác từ các đặc điểm bỏ sót và bất đồng mô hình. Đặc trưng đồng cảnh cải thiện chất lượng xác suất ở một số phép thử. Hiệu chỉnh bậc hai trên điểm bất đồng đạt 0.95134001 trên holdout, còn 10 fold đảo vai trò cho Δ AUC có trọng số âm. Nhãn mềm 80/20 được so với chỉ giảm biên cây. [Đặc điểm bỏ sót](../goal_0952/data_signals/BAO_CAO_DAC_DIEM.md) · [Không gian điểm](../goal_0952/score_space/BAO_CAO_HUONG_KHAC.md)

### Phase E · Bám Grandmaster EDA và EDA sâu

- Sau khi chủ nhân chọn bám bản Grandmaster EDA, kiểm tra đồng nhất cỡ mẫu của bảng EDA, làm mượt ô và trộn nhẹ. Ba CSV được giữ để đối chiếu. Thử phụ thuộc vào rủi ro và 10 cặp chưa dùng, xuất sáu biến thể. Hiệu chỉnh tuyến tính 10 cặp đạt 0.95123214. Phép đảo vai trò gần như ngang điểm. [Refinement](../goal_0952/grandmaster_eda_refinement/BAO_CAO_GRANDMASTER_EDA.md) · [Rủi ro và 10 cặp](../goal_0952/grandmaster_eda_risk/BAO_CAO_TOI_UU_EDA.md)
- Chạy song song chi tiết trong ô phân vị, tương quan đồng cảnh, LightGBM cây cộng và bản trộn cố định. Xuất và kiểm tra chín CSV, gồm đối chứng giảm biên. LightGBM một nút tăng ở 10/10 fold đảo vai trò, song đối chứng chỉ giảm biên tăng nhiều hơn ở phép kiểm mẫu nhỏ. Holdout của LightGBM đạt 0.95122558. [Ba hướng và chín CSV](../goal_0952/BAO_CAO_THU_SONG_SONG.md)
- EDA sâu kiểm tra 82 nhóm cách ghi số, 9 giả thuyết hành vi và 16 tổ hợp ba biến, cùng ma trận các cặp xếp sai. Hiệu chỉnh cho 404 mức tiền thưa là lựa chọn mới của nhánh đã chốt. Các giả thuyết thời gian, số dư và tương tác ba biến mất ưu thế ở phần chấm riêng. Nhóm khoảng cách thưa có tín hiệu cùng hướng nhưng vẫn ở dưới tiêu chí xác nhận. [EDA sâu](../goal_0952/BAO_CAO_EDA_SAU.md) · [Submission cuối](../goal_0952/deep_eda_measurement/submission_grandmaster_eda_amount_hole.csv)

## Những phân biệt đã chốt trong các chat

- **`FINAL` và ứng viên cuối.** FINAL là tổ hợp GPU 5 fold GLM-NEWTON + XGB-D03. Ứng viên cuối dùng Grandmaster EDA và hiệu chỉnh số tiền. [README tổng hợp](README.md)
- **Nhiều seed và full refit.** SEED5-MEAN ban đầu là 5 cấu hình XGBoost trung bình xác suất từ 5 fold mỗi cấu hình. Bản refit huấn luyện lại mỗi cấu hình một lần trên đủ train. LGBM-SEED5-MEAN gồm 5 LightGBM D03 tối đa 4 lá. Tệp mỗi tổ hợp có đúng một điểm trên mỗi ID test. [Seed XGBoost](../s6e11_gpu/seed3_improvement/ket_qua_seed3_mean.md) · [Seed LightGBM](../s6e11_gpu/lightgbm_seed5/results/LGBM-SEED5-MEAN/metrics.json)
- **Nguồn xác nhận và độ chắc chắn.** OOF, holdout, 10 fold đảo vai trò, Public LB trong kho và báo cáo của người dùng giữ nhãn nguồn riêng. Bản cuối 0.95131585 dùng holdout đã từng sử dụng. Log loss có bằng chứng lặp lại, còn khoảng tin cậy Δ AUC gồm 0. [Phép đo cuối](../goal_0952/deep_eda_measurement/measurement_metrics.json)
