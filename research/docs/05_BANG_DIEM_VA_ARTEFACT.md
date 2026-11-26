# Bảng điểm, cấu hình và tệp dự đoán

[Về README](README.md) · [Mô hình và mã chạy](04_MO_HINH_VA_THU_NGHIEM.md) · [Tái lập và submission](06_TAI_LAP_VA_SUBMISSION.md)

Bảng lấy 394 dòng có chỉ số từ 91 tệp kết quả cục bộ. Một đối chứng có thể xuất hiện trong nhiều phép kiểm. Tiêu đề bảng ghi phép đo và cột nguồn của mỗi hàng chỉ đến mô hình, tập chấm và công thức tương ứng. Các trường AP, log loss và Δ được chép theo tên trường trong tệp, dạng dấu chấm thập phân.

- **OOF** là điểm gộp trên dự đoán ngoài fold. Các dòng thăm dò tổ hợp trên kho OOF chung đi cùng tệp cấu hình của phép khảo sát. Mốc FINAL xác nhận lồng nhau có hồ sơ riêng.
- **Holdout** gồm 104.994 dòng cho nhánh Grandmaster. Các nguồn so trên mốc riêng liệt kê đối chứng trong chính tệp. **Đảo vai trò** đánh giá nhánh được thử với tập tham chiếu 104.994 dòng và chấm theo 10 fold của 594.966 dòng còn lại.
- **Khoảng tin cậy** được ghi theo phép so sánh của từng tệp. Delta có trọng số theo fold dùng số cặp dương–âm từng fold, khi nguồn có lưu chỉ số này. Cột AP gắn ghi chú PR AUC khi đó là tên đo tại nguồn. Dấu — biểu thị trường không lưu ở bản ghi đó.
- **Phạm vi đọc** gồm các trường chỉ số, tên mô hình và đường dẫn nguồn trong toàn bộ JSON/CSV được dẫn ở từng bảng. Tệp gốc chứa cấu hình, seed, điểm từng fold và kiểm tra tái dựng. Chỉ số smoke, khóa nhãn EDA không sinh mô hình và kết quả train/test thiếu nhãn nằm ở các chương liên quan.

## Mốc lịch sử ghi ngoài bảng máy đọc được

| Bản | Nguồn điểm | ROC AUC | Nguồn |
|---|---|---:|---|
| B06 | OOF trong báo cáo dự án | 0.946965 | [Bước 5](../../reports/steps_progress.md) |
| S6-L04 | OOF trong báo cáo dự án | 0.947365 | [Bước 6](../../reports/steps_progress.md) |
| S6-X02 | OOF trong báo cáo dự án | 0.947317 | [Bước 6](../../reports/steps_progress.md) |
| V1 tham khảo | OOF 5 fold / Public LB trong kho | 0.9508989 / 0.95069 | [Kho phiên bản](<../Model tham khảo/VERSION_COMPARISON.md>) |
| V2, tên gốc V3 | OOF 5 fold lồng / Public LB trong kho | 0.9510771 / 0.95092 | [Kho phiên bản](<../Model tham khảo/VERSION_COMPARISON.md>) |
| Grandmaster tham khảo | OOF bảo vệ 10 fold / Public LB trong kho | 0.951267 / 0.95154 | [Kho phiên bản](<../Model tham khảo/VERSION_COMPARISON.md>) |

Bảng xếp hạng 68 cấu hình đơn giữ điểm từ dự đoán đã lưu cho 52 cấu hình và nguồn báo cáo cho 16 cấu hình cũ. [Danh sách 68 cấu hình](../s6e11_gpu/xep_hang_mo_hinh_don.md)

## Dữ liệu giả lập ngoài cuộc thi

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| ft-transformer · validation | 0.65633423 | — | — | — | — | — | [ft-transformer-metrics.json](<../kaggle_practice/results/ft-transformer-metrics.json>) |
| ft-transformer · test giả lập | 0.68065847 | 0.09004682 | — | — | — | test_roc_auc | [ft-transformer-metrics.json](<../kaggle_practice/results/ft-transformer-metrics.json>) |
| lightgbm-te · validation | 0.74118173 | — | — | — | — | — | [lightgbm-te-metrics.json](<../kaggle_practice/results/lightgbm-te-metrics.json>) |
| lightgbm-te · test giả lập | 0.76868952 | 0.16990802 | — | — | — | test_roc_auc | [lightgbm-te-metrics.json](<../kaggle_practice/results/lightgbm-te-metrics.json>) |
| logistic-te · validation | 0.76001160 | — | — | — | — | — | [logistic-te-metrics.json](<../kaggle_practice/results/logistic-te-metrics.json>) |
| logistic-te · test giả lập | 0.78213277 | 0.20362297 | — | — | — | test_roc_auc | [logistic-te-metrics.json](<../kaggle_practice/results/logistic-te-metrics.json>) |
| mlp · validation | 0.76022446 | — | — | — | — | — | [mlp-metrics.json](<../kaggle_practice/results/mlp-metrics.json>) |
| mlp · test giả lập | 0.77044958 | 0.16225198 | — | — | — | test_roc_auc | [mlp-metrics.json](<../kaggle_practice/results/mlp-metrics.json>) |
| xgboost-te · validation | 0.73936712 | — | — | — | — | — | [xgboost-te-metrics.json](<../kaggle_practice/results/xgboost-te-metrics.json>) |
| xgboost-te · test giả lập | 0.77276174 | 0.16925539 | — | — | — | test_roc_auc | [xgboost-te-metrics.json](<../kaggle_practice/results/xgboost-te-metrics.json>) |

## RTX 4060 · OOF 5 fold, vòng 1

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| ft-transformer | 0.93833215 | 0.47478633 | — | — | — | — | [ft-transformer.json](<../s6e11_gpu/results/ft-transformer.json>) |
| lightgbm-te | 0.94832930 | 0.51640167 | — | — | — | — | [lightgbm-te.json](<../s6e11_gpu/results/lightgbm-te.json>) |
| logistic-te | 0.94729970 | 0.51378884 | — | — | — | — | [logistic-te.json](<../s6e11_gpu/results/logistic-te.json>) |
| mlp | 0.94145617 | 0.48687377 | — | — | — | — | [mlp.json](<../s6e11_gpu/results/mlp.json>) |
| xgboost-te | 0.94831147 | 0.51878475 | — | — | — | — | [xgboost-te.json](<../s6e11_gpu/results/xgboost-te.json>) |

## RTX 4060 · OOF 5 fold, vòng 2

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| catboost-te | 0.94715633 | 0.51742794 | — | — | — | — | [catboost-te.json](<../s6e11_gpu/round2/results/catboost-te.json>) |
| rank-blend | 0.94847399 | 0.51860590 | — | — | — | — | [rank-blend.json](<../s6e11_gpu/round2/results/rank-blend.json>) |
| tabm-mini | 0.94234968 | 0.48921439 | — | — | — | — | [tabm-mini.json](<../s6e11_gpu/round2/results/tabm-mini.json>) |
| xgboost-pair-te | 0.94819325 | 0.51824539 | — | — | — | — | [xgboost-pair-te.json](<../s6e11_gpu/round2/results/xgboost-pair-te.json>) |

## RTX 4060 · pipeline chính, OOF 5 fold và FINAL lồng 5×5

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| FINAL | 0.95034521 | 0.52937620 | 0.04434665 | +0.00016290 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| ALL3-MEAN | 0.95027765 | 0.52906315 | 0.04439979 | +0.00009534 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| TREES2-MEAN | 0.95027618 | 0.52893911 | 0.04440568 | +0.00009388 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| ALL3-RANK | 0.95027560 | 0.52849657 | 0.92644591 | +0.00009329 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| SEED3-MEAN | 0.95026928 | 0.52897540 | 0.04440926 | +0.00008697 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| TREES2-RANK | 0.95026731 | 0.52889267 | 0.93123656 | +0.00008501 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| SEED3-RANK | 0.95026302 | 0.52893911 | 0.93101458 | +0.00008072 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| XGB-D03 | 0.95018230 | 0.52876265 | 0.04442177 | +0.00168769 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| XGB-D03-R125 | 0.95014625 | 0.52932883 | 0.04438350 | -0.00003605 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| D03 | 0.95013086 | 0.52760385 | 0.04447513 | +0.00155673 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| D02 | 0.95012422 | 0.52820601 | 0.04443548 | +0.00155009 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| TUNE-NOSUB | 0.95010368 | 0.52859756 | 0.04444356 | -0.00007862 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| XGB-D03-S93 | 0.95009142 | 0.52792983 | 0.04446976 | -0.00009089 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| XGB-D02 | 0.95007439 | 0.52838160 | 0.04444421 | +0.00157978 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| XGB-D03-S71 | 0.95007415 | 0.52784829 | 0.04447929 | -0.00010815 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| TUNE-ETA005 | 0.95005989 | 0.52845614 | 0.04445771 | -0.00012242 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| TUNE-LAMBDA3 | 0.95004546 | 0.52851324 | 0.04444972 | -0.00013684 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| D01 | 0.94878365 | 0.52150665 | 0.04488951 | +0.00020952 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| XGB-D01 | 0.94862328 | 0.52163128 | 0.04491986 | +0.00012867 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| GLM-BOOST-LGB | 0.94862322 | 0.52211316 | 0.04487195 | +0.00009653 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| T04 | 0.94857413 | 0.52005519 | 0.04497261 | +0.00004744 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| GLM-BOOST-XGB | 0.94854414 | 0.52121213 | 0.04492384 | +0.00004953 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| DERIVED | 0.94854209 | 0.52031470 | 0.04497228 | -0.00003204 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| T02 | 0.94852669 | 0.51854241 | 0.04505654 | +0.00129454 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| T03 | 0.94849461 | 0.52101956 | 0.04496324 | +0.00135552 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| GLM-NEWTON | 0.94765641 | 0.51551225 | 0.04527547 | +0.00007550 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| GLM-C0.3 | 0.94760502 | 0.51556331 | 0.04528130 | +0.00002411 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| P1-P2 | 0.94758091 | 0.51549117 | 0.04528638 | +0.00009700 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| GLM-C0.1 | 0.94755777 | 0.51529369 | 0.04529953 | -0.00002315 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| P1 | 0.94748391 | 0.51507023 | 0.04531444 | +0.00022290 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| P1-P2-P3 | 0.94738532 | 0.51420751 | 0.04536292 | -0.00019560 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| T01 | 0.94726101 | 0.51358252 | 0.04540577 | +0.00537065 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| GPU-S6-L04 | 0.94723215 | 0.51326175 | 0.04540873 | — | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| T01-M50 | 0.94720801 | 0.51300658 | 0.04543620 | -0.00005301 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| GPU-S6-X02 | 0.94713909 | 0.51392964 | 0.04540771 | — | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| COUNT | 0.94682509 | 0.50582047 | 0.04628969 | -0.00174904 | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |
| GPU-B02 | 0.94189036 | 0.48971212 | 0.04703778 | — | — | oof_auc | [comparison.csv](<../../s6e11_gpu/plan_runs/published/full/comparison.csv>) |

## RTX 4060 · khảo sát tổ hợp trên kho OOF

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| GLM-NEWTON | 0.94765641 | 0.51551225 | 0.04527547 | -0.00252589 | — | fold ↑ 0 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| D03 | 0.95013086 | 0.52760385 | 0.04447513 | -0.00005144 | — | fold ↑ 2 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| XGB-D03 | 0.95018230 | 0.52876265 | 0.04442177 | +0.00000000 | — | fold ↑ 0 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| probability_mean | 0.95027765 | 0.52906315 | 0.04439979 | +0.00009534 | — | fold ↑ 4 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| rank_mean | 0.95027560 | 0.52849501 | — | +0.00009329 | — | fold ↑ 3 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| logistic_stack_seed7_c0.01 | 0.95038135 | 0.52949528 | 0.04433550 | +0.00019904 | — | fold ↑ 5 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| logistic_stack_seed7_c0.1 | 0.95038580 | 0.52954932 | 0.04433125 | +0.00020350 | — | fold ↑ 5 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| logistic_stack_seed7_c1 | 0.95038424 | 0.52955720 | 0.04433113 | +0.00020193 | — | fold ↑ 5 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| rank_logit_stack_seed7_c0.01 | 0.95037684 | 0.52944957 | 0.04438089 | +0.00019453 | — | fold ↑ 4 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| rank_logit_stack_seed7_c0.1 | 0.95038408 | 0.52954271 | 0.04437382 | +0.00020178 | — | fold ↑ 5 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| rank_logit_stack_seed7_c1 | 0.95038234 | 0.52956011 | 0.04437360 | +0.00020004 | — | fold ↑ 5 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| logistic_stack_seed8_c0.01 | 0.95039312 | 0.52950323 | 0.04433394 | +0.00021081 | — | fold ↑ 5 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| logistic_stack_seed8_c0.1 | 0.95039383 | 0.52955888 | 0.04433034 | +0.00021153 | — | fold ↑ 5 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| logistic_stack_seed8_c1 | 0.95039011 | 0.52956105 | 0.04433072 | +0.00020781 | — | fold ↑ 5 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| rank_logit_stack_seed8_c0.01 | 0.95038783 | 0.52942949 | 0.04438064 | +0.00020552 | — | fold ↑ 5 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| rank_logit_stack_seed8_c0.1 | 0.95039168 | 0.52951847 | 0.04437435 | +0.00020938 | — | fold ↑ 5 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| rank_logit_stack_seed8_c1 | 0.95038677 | 0.52952115 | 0.04437487 | +0.00020446 | — | fold ↑ 5 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| leave_out_GLM-NEWTON | 0.95026745 | 0.52900509 | 0.04436780 | +0.00008515 | — | fold ↑ 4 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| leave_out_D03 | 0.95033923 | 0.52939298 | 0.04434536 | +0.00015693 | — | fold ↑ 5 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |
| leave_out_XGB-D03 | 0.95030365 | 0.52863940 | 0.04437663 | +0.00012134 | — | fold ↑ 3 · auc | [ensemble_metrics.csv](<../../s6e11_gpu/plan_runs/published/full/ensemble/ensemble_metrics.csv>) |

## Xử lý lệch nhãn · OOF XGBoost

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| XGB-RANDOM | 0.95017683 | 0.52839691 | 0.04459042 | — | — | oof_auc | [comparison.csv](<../s6e11_gpu/imbalance/results/xgb/comparison.csv>) |
| XGB-HARD | 0.95014087 | 0.52900425 | 0.04824359 | — | — | oof_auc | [comparison.csv](<../s6e11_gpu/imbalance/results/xgb/comparison.csv>) |
| XGB-W2 | 0.95010351 | 0.52841269 | 0.04740870 | — | — | oof_auc | [comparison.csv](<../s6e11_gpu/imbalance/results/xgb/comparison.csv>) |
| XGB-W4 | 0.95007915 | 0.52790398 | 0.05824764 | — | — | oof_auc | [comparison.csv](<../s6e11_gpu/imbalance/results/xgb/comparison.csv>) |

## Xử lý lệch nhãn · OOF MLP

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| MLP-BCE | 0.94489872 | 0.50390506 | 0.04644432 | — | — | oof_auc | [metrics.json](<../s6e11_gpu/imbalance/results/mlp/MLP-BCE/metrics.json>) |
| MLP-FOCAL-G1 | 0.94426659 | 0.50275809 | 0.06608377 | -0.00063213 | — | oof_auc | [metrics.json](<../s6e11_gpu/imbalance/results/mlp/MLP-FOCAL-G1/metrics.json>) |
| MLP-FOCAL-G2 | 0.94400727 | 0.50178263 | 0.11450700 | -0.00089145 | — | oof_auc | [metrics.json](<../s6e11_gpu/imbalance/results/mlp/MLP-FOCAL-G2/metrics.json>) |
| MLP-PAIRWISE | 0.94121284 | 0.48722373 | 0.27616889 | -0.00368588 | — | oof_auc | [metrics.json](<../s6e11_gpu/imbalance/results/mlp/MLP-PAIRWISE/metrics.json>) |

## Tiền huấn luyện bảng · OOF trên cột giao dịch

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| ft-masked | 0.94224129 | 0.48550873 | 0.04753140 | — | — | oof_auc | [metrics.json](<../s6e11_gpu/pretraining/results/full/ft-masked/metrics.json>) |
| ft-scratch | 0.94159885 | 0.48387299 | 0.04816476 | — | — | oof_auc | [metrics.json](<../s6e11_gpu/pretraining/results/full/ft-scratch/metrics.json>) |
| mlp-masked | 0.94155513 | 0.48945917 | 0.04712077 | — | — | oof_auc | [metrics.json](<../s6e11_gpu/pretraining/results/full/mlp-masked/metrics.json>) |
| mlp-scarf | 0.94120676 | 0.48746209 | 0.04732170 | — | — | oof_auc | [metrics.json](<../s6e11_gpu/pretraining/results/full/mlp-scarf/metrics.json>) |
| mlp-scratch | 0.94124013 | 0.48846847 | 0.04723192 | — | — | oof_auc | [metrics.json](<../s6e11_gpu/pretraining/results/full/mlp-scratch/metrics.json>) |
| tabnet-masked | 0.75302268 | 0.09401351 | 0.07708657 | — | — | oof_auc | [metrics.json](<../s6e11_gpu/pretraining/results/full/tabnet-masked/metrics.json>) |
| tabnet-scratch | 0.73378241 | 0.08542450 | 0.07792142 | — | — | oof_auc | [metrics.json](<../s6e11_gpu/pretraining/results/full/tabnet-scratch/metrics.json>) |
| xgb-d03-mlp-masked-embeddings | 0.94966046 | 0.52534680 | 0.04463277 | — | — | oof_auc | [metrics.json](<../s6e11_gpu/pretraining/results/full/xgb-d03-mlp-masked-embeddings/metrics.json>) |

## Ablation đặc trưng từ EDA

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| XGB-D03-CHANNEL-6H | 0.95010814 | 0.52863882 | 0.04443123 | — | — | oof_auc | [comparison.csv](<../eda_review/feature_ablation/results/full/comparison.csv>) |
| XGB-D03-VELOCITY-RESIDUAL | 0.95008310 | 0.52857390 | 0.04443876 | — | — | oof_auc | [comparison.csv](<../eda_review/feature_ablation/results/full/comparison.csv>) |

## TE cặp, mức hiếm và hệ quả với FINAL

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| XGB-D03-RARE-TE | 0.95006260 | 0.52847718 | 0.04444709 | — | — | oof_auc | [comparison.csv](<../s6e11_gpu/final_feature_optimization/results/full/comparison.csv>) |
| XGB-D03-PAIR-TE | 0.94961810 | 0.52695742 | 0.04455674 | — | — | oof_auc | [comparison.csv](<../s6e11_gpu/final_feature_optimization/results/full/comparison.csv>) |
| XGB-D03-PAIR-RARE-TE | 0.94958154 | 0.52645091 | 0.04459277 | — | — | oof_auc | [comparison.csv](<../s6e11_gpu/final_feature_optimization/results/full/comparison.csv>) |

## XGB D03 nhiều seed · OOF 5 fold

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| SEED5-MEAN | 0.95033114 | 0.52929598 | 0.04438593 | — | — | oof_auc | [comparison.csv](<../s6e11_gpu/seed3_improvement/results/comparison.csv>) |
| SEED4-NEW-MEAN | 0.95030907 | 0.52926489 | 0.04439114 | — | — | oof_auc | [comparison.csv](<../s6e11_gpu/seed3_improvement/results/comparison.csv>) |
| SEED4-CROSS-MEAN | 0.95030311 | 0.52907618 | 0.04439926 | — | — | oof_auc | [comparison.csv](<../s6e11_gpu/seed3_improvement/results/comparison.csv>) |
| XGB-D03-S71-E2028 | 0.95020055 | 0.52834715 | 0.04443341 | — | — | oof_auc | [comparison.csv](<../s6e11_gpu/seed3_improvement/results/comparison.csv>) |
| XGB-D03-S103-E2032 | 0.95012148 | 0.52845389 | 0.04443709 | — | — | oof_auc | [comparison.csv](<../s6e11_gpu/seed3_improvement/results/comparison.csv>) |

## FINAL và ba hướng · seed_1337

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| FINAL | 0.95027012 | 0.52930632 | 0.04436633 | — | — | oof_auc | [metrics.json](<../s6e11_gpu/followup/results/full/seed_1337/metrics.json>) |
| FINAL_PLUS_LGBM20 | 0.95033889 | 0.52946776 | 0.04434703 | +0.00006876 | — | oof_auc | [metrics.json](<../s6e11_gpu/followup/results/full/seed_1337/metrics.json>) |
| FINAL_PLUS_MLP10 | 0.95037954 | 0.53011068 | 0.04432672 | +0.00010942 | — | oof_auc | [metrics.json](<../s6e11_gpu/followup/results/full/seed_1337/metrics.json>) |
| FINAL_NO_FLAGS | 0.95025868 | 0.52936032 | 0.04436501 | -0.00001144 | — | oof_auc | [metrics.json](<../s6e11_gpu/followup/results/full/seed_1337/metrics.json>) |

## FINAL và ba hướng · seed_2718

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| FINAL | 0.95042653 | 0.52996255 | 0.04430999 | — | — | oof_auc | [metrics.json](<../s6e11_gpu/followup/results/full/seed_2718/metrics.json>) |
| FINAL_PLUS_LGBM20 | 0.95053031 | 0.53049608 | 0.04426909 | +0.00010378 | — | oof_auc | [metrics.json](<../s6e11_gpu/followup/results/full/seed_2718/metrics.json>) |
| FINAL_PLUS_MLP10 | 0.95046511 | 0.53062442 | 0.04428232 | +0.00003858 | — | oof_auc | [metrics.json](<../s6e11_gpu/followup/results/full/seed_2718/metrics.json>) |
| FINAL_NO_FLAGS | 0.95041475 | 0.53004893 | 0.04430725 | -0.00001178 | — | oof_auc | [metrics.json](<../s6e11_gpu/followup/results/full/seed_2718/metrics.json>) |

## Grandmaster tái huấn luyện local · score OOF của đối chứng 10 fold

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| baseline_oof | 0.95128045 | 0.53561216 | 0.04395569 | — | — | CI [+0.94941910, +0.95314180] · nguồn ghi PR AUC · oof_auc | [metrics.json](<../grandmaster_refit/full/metrics.json>) |

## Grandmaster · density_offset_validation

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| holdout / current_grandmaster | 0.95094893 | 0.53818509 | 0.04387376 | — | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_offset_validation/metrics.json>) |
| holdout / offset_tree_control | 0.95047316 | 0.53787387 | 0.04397147 | — | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_offset_validation/metrics.json>) |
| holdout / offset_tree_density | 0.95059053 | 0.53817206 | 0.04394332 | — | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_offset_validation/metrics.json>) |
| holdout / grandmaster_density_replacement | 0.95091567 | 0.53851217 | 0.04386693 | — | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_offset_validation/metrics.json>) |
| holdout / grandmaster_density_delta | 0.95097483 | 0.53826322 | 0.04386741 | — | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_offset_validation/metrics.json>) |
| offset_density_vs_control | — | — | — | +0.00011737 | — | CI [+0.00000693, +0.00022780] | [metrics.json](<../grandmaster_10fold_ablation/density_offset_validation/metrics.json>) |
| stack_comparison / grandmaster_density_delta | — | — | — | +0.00002590 | — | CI [-0.00000935, +0.00006115] | [metrics.json](<../grandmaster_10fold_ablation/density_offset_validation/metrics.json>) |
| stack_comparison / grandmaster_density_replacement | — | — | — | -0.00003326 | — | CI [-0.00010944, +0.00004293] | [metrics.json](<../grandmaster_10fold_ablation/density_offset_validation/metrics.json>) |

## Grandmaster · density_tree_validation

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| development_pooled / control_xgb | 0.94822217 | 0.51891429 | 0.04511470 | — | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| development_pooled / density_xgb | 0.94863800 | 0.51958406 | 0.04502407 | — | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| holdout / current_grandmaster | 0.95094893 | 0.53818509 | 0.04387376 | — | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| holdout / control_xgb | 0.94839439 | 0.52371714 | 0.04485827 | — | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| holdout / density_xgb | 0.94884018 | 0.52572258 | 0.04473803 | — | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| holdout_density_vs_control | — | — | — | +0.00044579 | — | CI [-0.00008476, +0.00097634] | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| holdout_density_vs_grandmaster | — | — | — | -0.00210875 | — | CI [-0.00306910, -0.00114839] | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| development_folds / 1 | — | — | — | +0.00015701 | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| development_folds / 2 | — | — | — | +0.00159186 | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| development_folds / 3 | — | — | — | -0.00009424 | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| development_folds / 4 | — | — | — | -0.00002808 | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| development_folds / 5 | — | — | — | +0.00037905 | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| development_folds / 6 | — | — | — | +0.00020839 | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| development_folds / 7 | — | — | — | +0.00052119 | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| development_folds / 8 | — | — | — | +0.00055824 | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| development_folds / 9 | — | — | — | +0.00069070 | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |
| development_folds / 10 | — | — | — | +0.00026727 | — | — | [metrics.json](<../grandmaster_10fold_ablation/density_tree_validation/metrics.json>) |

## Grandmaster · hard_group_ranker

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| holdout / current_hierarchy | 0.95094893 | 0.53818509 | 0.04387376 | — | — | — | [metrics.json](<../grandmaster_10fold_ablation/hard_group_ranker/metrics.json>) |
| holdout / pointwise | 0.95093956 | 0.53824223 | 0.04387178 | -0.00000937 | — | CI [-0.00004097, +0.00002223] | [metrics.json](<../grandmaster_10fold_ablation/hard_group_ranker/metrics.json>) |
| holdout / uniform | 0.95093647 | 0.53833605 | 0.04387061 | -0.00001246 | — | CI [-0.00008496, +0.00006004] | [metrics.json](<../grandmaster_10fold_ablation/hard_group_ranker/metrics.json>) |
| holdout / matched | 0.95093844 | 0.53839612 | 0.04386775 | -0.00001049 | — | CI [-0.00008695, +0.00006598] | [metrics.json](<../grandmaster_10fold_ablation/hard_group_ranker/metrics.json>) |

## Grandmaster · hard_group_ranker_balanced

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| holdout / current_hierarchy | 0.95094893 | 0.53818509 | 0.04387376 | — | — | — | [metrics.json](<../grandmaster_10fold_ablation/hard_group_ranker_balanced/metrics.json>) |
| holdout / balanced | 0.95093880 | 0.53824589 | 0.04387178 | -0.00001013 | — | CI [-0.00004251, +0.00002224] | [metrics.json](<../grandmaster_10fold_ablation/hard_group_ranker_balanced/metrics.json>) |

## Grandmaster · hierarchical_glm_architecture

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| metrics / current_hierarchy | 0.95094893 | 0.53818509 | 0.04387376 | +0.00000000 | — | — | [metrics.json](<../grandmaster_10fold_ablation/hierarchical_glm_architecture/metrics.json>) |
| metrics / hierarchical_glm | 0.95094893 | 0.53818509 | 0.04387376 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [metrics.json](<../grandmaster_10fold_ablation/hierarchical_glm_architecture/metrics.json>) |
| metrics / hierarchical_glm_interactions | 0.95094893 | 0.53818509 | 0.04387376 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [metrics.json](<../grandmaster_10fold_ablation/hierarchical_glm_architecture/metrics.json>) |
| interactions_vs_additive | — | — | — | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [metrics.json](<../grandmaster_10fold_ablation/hierarchical_glm_architecture/metrics.json>) |

## Grandmaster · đối chứng loss ưu tiên ca có điểm dưới 2%

Phép này dùng 10 fold trên 594.966 dòng chấm, với 104.994 dòng tham chiếu. ROC AUC trong bảng được gộp theo trọng số số cặp dương–âm của từng fold.

| Cấu hình | ROC AUC trọng số | Δ AUC trọng số | Fold tăng AUC | Nguồn |
|---|---:|---:|---:|---|
| current_grandmaster | 0.95054073 | — | — | [metrics.json](../grandmaster_10fold_ablation/low_score_rank_validation/metrics.json) |
| pointwise | 0.95012515 | −0.00041559 | 0/10 | [metrics.json](../grandmaster_10fold_ablation/low_score_rank_validation/metrics.json) |
| pairwise | 0.95003391 | −0.00050683 | 0/10 | [metrics.json](../grandmaster_10fold_ablation/low_score_rank_validation/metrics.json) |
| pairwise so với pointwise | — | −0.00009124 | 3/10 | [metrics.json](../grandmaster_10fold_ablation/low_score_rank_validation/metrics.json) |

## Grandmaster · meta_improvements

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| variants / current_hierarchy | 0.95094893 | 0.53818509 | 0.04387376 | +0.00000000 | — | — | [metrics.json](<../grandmaster_10fold_ablation/meta_improvements/metrics.json>) |
| variants / pairwise_loss | 0.95092340 | 0.53817911 | 0.04387673 | -0.00002553 | — | CI [-0.00005046, -0.00000060] | [metrics.json](<../grandmaster_10fold_ablation/meta_improvements/metrics.json>) |
| variants / risk_gated | 0.95094413 | 0.53817553 | 0.04387559 | -0.00000480 | — | CI [-0.00001936, +0.00000976] | [metrics.json](<../grandmaster_10fold_ablation/meta_improvements/metrics.json>) |
| variants / paired_logits | 0.95094805 | 0.53815035 | 0.04387383 | -0.00000088 | — | CI [-0.00000440, +0.00000264] | [metrics.json](<../grandmaster_10fold_ablation/meta_improvements/metrics.json>) |
| variants / mean_probabilities | 0.95094528 | 0.53820941 | 0.04387454 | -0.00000365 | — | CI [-0.00001560, +0.00000830] | [metrics.json](<../grandmaster_10fold_ablation/meta_improvements/metrics.json>) |

## Grandmaster · post_stack_residual

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| post_stack_residual | — | — | — | -0.00007160 | — | CI [-0.00021253, +0.00006933] | [metrics.json](<../grandmaster_10fold_ablation/post_stack_residual/metrics.json>) |

## Grandmaster · validation

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | 0.95094666 | 0.53814886 | 0.04387461 | — | — | — | [metrics.json](<../grandmaster_10fold_ablation/validation/metrics.json>) |
| variants / residual_te | 0.95094453 | 0.53817918 | 0.04387453 | -0.00000213 | — | CI [-0.00000580, +0.00000155] | [metrics.json](<../grandmaster_10fold_ablation/validation/metrics.json>) |
| variants / hierarchical_amount | 0.95094893 | 0.53818509 | 0.04387376 | +0.00000227 | — | CI [-0.00000271, +0.00000725] | [metrics.json](<../grandmaster_10fold_ablation/validation/metrics.json>) |
| variants / aligned_offset | 0.95094565 | 0.53819317 | 0.04387372 | -0.00000101 | — | CI [-0.00000611, +0.00000409] | [metrics.json](<../grandmaster_10fold_ablation/validation/metrics.json>) |

## Bảy hướng mới · OOF 5 fold và kiểm theo seed

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| control_metrics | 0.95017296 | 0.52922561 | 0.04439629 | — | — | — | [control_metrics.json](<../novel_model_experiments/results/control_metrics.json>) |
| copula_metrics | 0.95022205 | 0.52865376 | 0.04442023 | — | — | — | [copula_metrics.json](<../novel_model_experiments/results/copula_metrics.json>) |
| seeds / 42 / metrics / FINAL | 0.95034521 | 0.52937620 | 0.04434665 | — | — | oof_auc | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 42 / metrics / FINAL_PLUS_LGBM20 | 0.95040009 | 0.52963001 | 0.04432869 | — | — | oof_auc | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 42 / metrics / FINAL_PLUS_MLP10 | 0.95039776 | 0.53001930 | 0.04432102 | — | — | oof_auc | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 42 / metrics / FIXED_BLEND | 0.95045126 | 0.53027823 | 0.04430444 | — | — | oof_auc | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 42 / contrasts / LGBM20_vs_FINAL | — | — | — | +0.00005488 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 42 / contrasts / MLP10_vs_FINAL | — | — | — | +0.00005255 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 42 / contrasts / FIXED_BLEND_vs_FINAL | — | — | — | +0.00010605 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 42 / contrasts / FIXED_BLEND_vs_LGBM20 | — | — | — | +0.00005117 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 42 / contrasts / FIXED_BLEND_vs_MLP10 | — | — | — | +0.00005350 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 1337 / metrics / FINAL | 0.95027012 | 0.52930632 | 0.04436633 | — | — | oof_auc | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 1337 / metrics / FINAL_PLUS_LGBM20 | 0.95033889 | 0.52946776 | 0.04434703 | — | — | oof_auc | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 1337 / metrics / FINAL_PLUS_MLP10 | 0.95037954 | 0.53011068 | 0.04432672 | — | — | oof_auc | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 1337 / metrics / FIXED_BLEND | 0.95044346 | 0.53030214 | 0.04431007 | — | — | oof_auc | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 1337 / contrasts / LGBM20_vs_FINAL | — | — | — | +0.00006876 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 1337 / contrasts / MLP10_vs_FINAL | — | — | — | +0.00010942 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 1337 / contrasts / FIXED_BLEND_vs_FINAL | — | — | — | +0.00017334 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 1337 / contrasts / FIXED_BLEND_vs_LGBM20 | — | — | — | +0.00010457 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 1337 / contrasts / FIXED_BLEND_vs_MLP10 | — | — | — | +0.00006392 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 2718 / metrics / FINAL | 0.95042653 | 0.52996255 | 0.04430999 | — | — | oof_auc | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 2718 / metrics / FINAL_PLUS_LGBM20 | 0.95053031 | 0.53049608 | 0.04426909 | — | — | oof_auc | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 2718 / metrics / FINAL_PLUS_MLP10 | 0.95046511 | 0.53062442 | 0.04428232 | — | — | oof_auc | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 2718 / metrics / FIXED_BLEND | 0.95056783 | 0.53116499 | 0.04424394 | — | — | oof_auc | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 2718 / contrasts / LGBM20_vs_FINAL | — | — | — | +0.00010378 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 2718 / contrasts / MLP10_vs_FINAL | — | — | — | +0.00003858 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 2718 / contrasts / FIXED_BLEND_vs_FINAL | — | — | — | +0.00014130 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 2718 / contrasts / FIXED_BLEND_vs_LGBM20 | — | — | — | +0.00003751 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| seeds / 2718 / contrasts / FIXED_BLEND_vs_MLP10 | — | — | — | +0.00010271 | — | — | [fixed_blend_metrics.json](<../novel_model_experiments/results/fixed_blend_metrics.json>) |
| hours_residual_metrics | 0.95035247 | 0.52934534 | 0.04434682 | — | — | — | [hours_residual_metrics.json](<../novel_model_experiments/results/hours_residual_metrics.json>) |
| baseline | 0.95034521 | 0.52937620 | 0.04434665 | — | — | — | [hours_residual_metrics.json](<../novel_model_experiments/results/hours_residual_metrics.json>) |
| neighborhood_metrics | 0.95029222 | 0.52909739 | 0.04440046 | — | — | — | [neighborhood_metrics.json](<../novel_model_experiments/results/neighborhood_metrics.json>) |
| spline_blend | — | — | — | +0.00010536 | — | — | [spline_blend_metrics.json](<../novel_model_experiments/results/spline_blend_metrics.json>) |
| historical_final | 0.95034521 | 0.52937620 | 0.04434665 | — | — | — | [spline_blend_metrics.json](<../novel_model_experiments/results/spline_blend_metrics.json>) |
| spline_blend | 0.95045057 | 0.53009202 | 0.04430608 | — | — | — | [spline_blend_metrics.json](<../novel_model_experiments/results/spline_blend_metrics.json>) |
| baseline | 0.94591771 | 0.51109120 | 0.04578482 | — | — | — | [spline_interactions_metrics.json](<../novel_model_experiments/spline_interactions_metrics.json>) |
| spline_interactions | 0.94576566 | 0.50953742 | 0.04590274 | — | — | — | [spline_interactions_metrics.json](<../novel_model_experiments/spline_interactions_metrics.json>) |

## Mục tiêu 0,952 · kết quả nhánh và thang đo ghi theo nguồn

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| latent_development_metrics | 0.93860801 | — | — | — | — | holdout_auc | [latent_development_metrics.json](<../goal_0952/latent_development_metrics.json>) |
| development | 0.95049680 | — | — | — | — | holdout_auc | [lightgbm_metrics.json](<../goal_0952/lightgbm_metrics.json>) |
| holdout_scores / grandmaster_original | 0.95094666 | 0.53814886 | 0.04387461 | — | — | — | [residual_bagged_development_metrics.json](<../goal_0952/residual_bagged_development_metrics.json>) |
| holdout_scores / grandmaster_hierarchical | 0.95094893 | 0.53818509 | 0.04387376 | — | — | — | [residual_bagged_development_metrics.json](<../goal_0952/residual_bagged_development_metrics.json>) |
| holdout_scores / screened_residual | 0.95092524 | 0.53768621 | 0.04389379 | — | — | — | [residual_bagged_development_metrics.json](<../goal_0952/residual_bagged_development_metrics.json>) |
| holdout_scores / bagged_offset_224 | 0.95106722 | 0.53803845 | 0.04386547 | — | — | — | [residual_bagged_development_metrics.json](<../goal_0952/residual_bagged_development_metrics.json>) |
| holdout_scores / grandmaster_original | 0.95094666 | 0.53814886 | 0.04387461 | — | — | — | [residual_development_metrics.json](<../goal_0952/residual_development_metrics.json>) |
| holdout_scores / grandmaster_hierarchical | 0.95094893 | 0.53818509 | 0.04387376 | — | — | — | [residual_development_metrics.json](<../goal_0952/residual_development_metrics.json>) |
| holdout_scores / screened_residual | 0.95092524 | 0.53768621 | 0.04389379 | — | — | — | [residual_development_metrics.json](<../goal_0952/residual_development_metrics.json>) |
| holdout_scores / offset_xgb_96 | 0.95103456 | 0.53822247 | 0.04385955 | — | — | — | [residual_development_metrics.json](<../goal_0952/residual_development_metrics.json>) |
| holdout_scores / offset_xgb_224 | 0.95112021 | 0.53803676 | 0.04385635 | — | — | — | [residual_development_metrics.json](<../goal_0952/residual_development_metrics.json>) |
| holdout_scores / grandmaster_original | 0.95094666 | 0.53814886 | 0.04387461 | — | — | — | [residual_extended_development_metrics.json](<../goal_0952/residual_extended_development_metrics.json>) |
| holdout_scores / grandmaster_hierarchical | 0.95094893 | 0.53818509 | 0.04387376 | — | — | — | [residual_extended_development_metrics.json](<../goal_0952/residual_extended_development_metrics.json>) |
| holdout_scores / screened_residual | 0.95092524 | 0.53768621 | 0.04389379 | — | — | — | [residual_extended_development_metrics.json](<../goal_0952/residual_extended_development_metrics.json>) |
| holdout_scores / offset_xgb_448 | 0.95110415 | 0.53754005 | 0.04387368 | — | — | — | [residual_extended_development_metrics.json](<../goal_0952/residual_extended_development_metrics.json>) |
| holdout_scores / offset_xgb_672 | 0.95111153 | 0.53696057 | 0.04388682 | — | — | — | [residual_extended_development_metrics.json](<../goal_0952/residual_extended_development_metrics.json>) |
| holdout_scores / raw_offset_xgb_224 | 0.95104590 | 0.53773762 | 0.04387143 | — | — | — | [residual_extended_development_metrics.json](<../goal_0952/residual_extended_development_metrics.json>) |
| holdout_scores / raw_offset_xgb_448 | 0.95103058 | 0.53743419 | 0.04388207 | — | — | — | [residual_extended_development_metrics.json](<../goal_0952/residual_extended_development_metrics.json>) |
| holdout_scores / grandmaster_original | 0.95094666 | 0.53814886 | 0.04387461 | — | — | — | [residual_ranked_development_metrics.json](<../goal_0952/residual_ranked_development_metrics.json>) |
| holdout_scores / grandmaster_hierarchical | 0.95094893 | 0.53818509 | 0.04387376 | — | — | — | [residual_ranked_development_metrics.json](<../goal_0952/residual_ranked_development_metrics.json>) |
| holdout_scores / screened_residual | 0.95092524 | 0.53768621 | 0.04389379 | — | — | — | [residual_ranked_development_metrics.json](<../goal_0952/residual_ranked_development_metrics.json>) |
| holdout_scores / ranked_offset_96 | 0.95081615 | 0.53817965 | 0.04388914 | — | — | — | [residual_ranked_development_metrics.json](<../goal_0952/residual_ranked_development_metrics.json>) |
| holdout_scores / ranked_offset_224 | 0.95071530 | 0.53809183 | 0.04390306 | — | — | — | [residual_ranked_development_metrics.json](<../goal_0952/residual_ranked_development_metrics.json>) |
| MLP-BCE | 0.94491654 | — | — | — | — | roc_auc_oof | [mlp_development.json](<../goal_0952/mlp_development.json>) |

## Mục tiêu 0,952 · so sánh 27 công thức trên holdout

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| candidates / grandmaster_original | 0.95094666 | 0.53814886 | 0.04387461 | -0.00000227 | — | CI [-0.00000725, +0.00000271] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / grandmaster_hierarchical | 0.95094893 | 0.53818509 | 0.04387376 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / screened_residual | 0.95092524 | 0.53768621 | 0.04389379 | -0.00002369 | — | CI [-0.00010728, +0.00005989] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / offset_xgb_96 | 0.95103456 | 0.53822247 | 0.04385955 | +0.00008563 | — | CI [-0.00008549, +0.00025676] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / offset_xgb_224 | 0.95112021 | 0.53803676 | 0.04385635 | +0.00017128 | — | CI [-0.00006588, +0.00040844] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / offset_xgb_448 | 0.95110415 | 0.53754005 | 0.04387368 | +0.00015522 | — | CI [-0.00016376, +0.00047421] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / offset_xgb_672 | 0.95111153 | 0.53696057 | 0.04388682 | +0.00016260 | — | CI [-0.00021204, +0.00053723] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / raw_offset_xgb_224 | 0.95104590 | 0.53773762 | 0.04387143 | +0.00009697 | — | CI [-0.00014553, +0.00033948] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / raw_offset_xgb_448 | 0.95103058 | 0.53743419 | 0.04388207 | +0.00008165 | — | CI [-0.00023690, +0.00040020] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / ranked_offset_96 | 0.95081615 | 0.53817965 | 0.04388914 | -0.00013278 | — | CI [-0.00054822, +0.00028267] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / ranked_offset_224 | 0.95071530 | 0.53809183 | 0.04390306 | -0.00023363 | — | CI [-0.00077374, +0.00030648] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / grandmaster_mlp10 | 0.95094700 | 0.53887112 | 0.04385103 | -0.00000193 | — | CI [-0.00012629, +0.00012243] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / grandmaster_spline10 | 0.95101207 | 0.53813173 | 0.04387100 | +0.00006314 | — | CI [-0.00009304, +0.00021933] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / grandmaster_latent03 | 0.95096754 | 0.53815048 | 0.04387700 | +0.00001861 | — | CI [-0.00004216, +0.00007938] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / grandmaster_mlp05_spline05 | 0.95100080 | 0.53865919 | 0.04385307 | +0.00005187 | — | CI [-0.00005618, +0.00015993] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / offset224_mlp10 | 0.95110569 | 0.53867135 | 0.04383686 | +0.00015676 | — | CI [-0.00009387, +0.00040740] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / offset224_spline05 | 0.95115951 | 0.53805571 | 0.04384911 | +0.00021058 | — | CI [-0.00002848, +0.00044964] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / offset224_mlp05_spline05 | 0.95115635 | 0.53842351 | 0.04383658 | +0.00020742 | — | CI [-0.00003460, +0.00044945] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / offset224_latent03 | 0.95114385 | 0.53799776 | 0.04385689 | +0.00019492 | — | CI [-0.00003797, +0.00042781] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / offset448_mlp05_spline05 | 0.95115081 | 0.53791657 | 0.04384845 | +0.00020188 | — | CI [-0.00011090, +0.00051466] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / grandmaster_lgb20_mlp10 | 0.95107135 | 0.53899987 | 0.04383687 | +0.00012242 | — | CI [-0.00011351, +0.00035836] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / offset224_lgb20_mlp10 | 0.95118135 | 0.53866075 | 0.04383246 | +0.00023242 | — | CI [-0.00008586, +0.00055069] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / offset224_lgb10_spline05 | 0.95122445 | 0.53823528 | 0.04383599 | +0.00027552 | — | CI [+0.00002221, +0.00052883] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / offset224_lgb10 | 0.95117600 | 0.53819290 | 0.04384434 | +0.00022707 | — | CI [-0.00002713, +0.00048127] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / bagged_offset_224 | 0.95106722 | 0.53803845 | 0.04386547 | +0.00011829 | — | CI [-0.00011064, +0.00034721] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / bagged224_lgb10_spline05 | 0.95117700 | 0.53817260 | 0.04384343 | +0.00022807 | — | CI [-0.00002107, +0.00047721] | [evaluation.json](<../goal_0952/evaluation.json>) |
| candidates / grandmaster_learned_mlp | 0.95097235 | 0.53883837 | 0.04384633 | +0.00002342 | — | CI [-0.00011120, +0.00015804] | [evaluation.json](<../goal_0952/evaluation.json>) |
| branch_holdout_scores / mlp | 0.94633580 | 0.52107990 | 0.04536046 | — | — | — | [evaluation.json](<../goal_0952/evaluation.json>) |
| branch_holdout_scores / spline | 0.94693008 | 0.51673536 | 0.04534873 | — | — | — | [evaluation.json](<../goal_0952/evaluation.json>) |
| branch_holdout_scores / latent | 0.93860801 | 0.43991406 | 0.04929144 | — | — | — | [evaluation.json](<../goal_0952/evaluation.json>) |
| branch_holdout_scores / lightgbm | 0.95049680 | 0.53084652 | 0.04423601 | — | — | — | [evaluation.json](<../goal_0952/evaluation.json>) |

## Mở rộng mật độ, booster và hiệu chỉnh cấu trúc · holdout hoặc theo nhánh ghi ở nguồn

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| metrics / targeted | 0.95121922 | 0.53828731 | 0.04383991 | -0.00000523 | — | CI [-0.00016336, +0.00015289] | [context_metrics.json](<../goal_0952/followup/context_metrics.json>) |
| metrics / expanded | 0.95131181 | 0.53834381 | 0.04382947 | +0.00008736 | — | CI [-0.00008870, +0.00026342] | [context_metrics.json](<../goal_0952/followup/context_metrics.json>) |
| metrics / expanded_half_density | 0.95131224 | 0.53828373 | 0.04382592 | +0.00008779 | — | CI [-0.00003230, +0.00020788] | [context_metrics.json](<../goal_0952/followup/context_metrics.json>) |
| candidates / incumbent | 0.95122445 | 0.53823528 | 0.04383599 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / xgb_density | 0.95129466 | 0.53812586 | 0.04382983 | +0.00007021 | — | CI [-0.00002903, +0.00016945] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / xgb_density_half | 0.95126470 | 0.53824000 | 0.04383068 | +0.00004025 | — | CI [-0.00000958, +0.00009009] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / xgb_neighborhood | 0.95118606 | 0.53807844 | 0.04385283 | -0.00003839 | — | CI [-0.00019488, +0.00011809] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / xgb_neighborhood_half | 0.95121584 | 0.53825871 | 0.04384014 | -0.00000861 | — | CI [-0.00008693, +0.00006970] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / xgb_both | 0.95126146 | 0.53816648 | 0.04384377 | +0.00003701 | — | CI [-0.00013615, +0.00021018] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / xgb_both_half | 0.95125641 | 0.53828291 | 0.04383448 | +0.00003196 | — | CI [-0.00005452, +0.00011843] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / xgb_weighted | 0.95115070 | 0.53798830 | 0.04385528 | -0.00007375 | — | CI [-0.00023422, +0.00008671] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / xgb_weighted_half | 0.95119742 | 0.53813845 | 0.04384259 | -0.00002702 | — | CI [-0.00010738, +0.00005333] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / xgb_both_weighted | 0.95119348 | 0.53801445 | 0.04386317 | -0.00003096 | — | CI [-0.00024477, +0.00018284] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / xgb_both_weighted_half | 0.95123040 | 0.53828231 | 0.04384221 | +0.00000595 | — | CI [-0.00010113, +0.00011303] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / lightgbm_offset | 0.95120510 | 0.53828728 | 0.04384132 | -0.00001934 | — | CI [-0.00012697, +0.00008828] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / lightgbm_offset_half | 0.95122026 | 0.53826947 | 0.04383638 | -0.00000419 | — | CI [-0.00005817, +0.00004979] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / catboost_offset | 0.95112573 | 0.53826918 | 0.04385863 | -0.00009872 | — | CI [-0.00029585, +0.00009840] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / catboost_offset_half | 0.95119944 | 0.53837693 | 0.04383598 | -0.00002501 | — | CI [-0.00012340, +0.00007337] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / three_tree_families | 0.95124289 | 0.53838588 | 0.04383315 | +0.00001844 | — | CI [-0.00006815, +0.00010502] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| candidates / density_weighting_mix | 0.95123610 | 0.53817420 | 0.04383615 | +0.00001165 | — | CI [-0.00005938, +0.00008268] | [holdout_metrics.json](<../goal_0952/followup/holdout_metrics.json>) |
| metrics / incumbent | 0.95122445 | 0.53823528 | 0.04383599 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [structured_metrics.json](<../goal_0952/followup/structured_metrics.json>) |
| metrics / shared_half | 0.95119591 | 0.53765572 | 0.04386336 | -0.00002854 | — | CI [-0.00017701, +0.00011993] | [structured_metrics.json](<../goal_0952/followup/structured_metrics.json>) |
| metrics / gated_half | 0.95101509 | 0.53755899 | 0.04388940 | -0.00020936 | — | CI [-0.00044675, +0.00002802] | [structured_metrics.json](<../goal_0952/followup/structured_metrics.json>) |
| metrics / gated_full | 0.95061792 | 0.53602438 | 0.04400071 | -0.00060653 | — | CI [-0.00107689, -0.00013616] | [structured_metrics.json](<../goal_0952/followup/structured_metrics.json>) |

## Kiểm tra độ ổn định và calibration · holdout hoặc theo nhánh ghi ở nguồn

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| crossfit_original | 0.95131224 | — | 0.04382592 | — | — | auc | [calibration_metrics.json](<../goal_0952/robust_validation/calibration_metrics.json>) |
| crossfit_calibrated | 0.95117459 | — | 0.04383683 | — | — | auc | [calibration_metrics.json](<../goal_0952/robust_validation/calibration_metrics.json>) |
| candidates / without_density | 0.95122445 | 0.53823528 | 0.04383599 | -0.00008779 | — | CI [-0.00020788, +0.00003230] | [forward_metrics.json](<../goal_0952/robust_validation/forward_metrics.json>) |
| candidates / current_density | 0.95131224 | 0.53828373 | 0.04382592 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [forward_metrics.json](<../goal_0952/robust_validation/forward_metrics.json>) |
| candidates / conservative_only | 0.95120543 | 0.53819149 | 0.04384557 | -0.00010681 | — | CI [-0.00021801, +0.00000440] | [forward_metrics.json](<../goal_0952/robust_validation/forward_metrics.json>) |
| candidates / regularized_blend | 0.95126523 | 0.53828048 | 0.04383366 | -0.00004701 | — | CI [-0.00010297, +0.00000895] | [forward_metrics.json](<../goal_0952/robust_validation/forward_metrics.json>) |
| candidates / without_density | 0.95122445 | 0.53823528 | 0.04383599 | -0.00008779 | — | CI [-0.00020788, +0.00003230] | [smoothed_forward_metrics.json](<../goal_0952/robust_validation/smoothed_forward_metrics.json>) |
| candidates / current_density | 0.95131224 | 0.53828373 | 0.04382592 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [smoothed_forward_metrics.json](<../goal_0952/robust_validation/smoothed_forward_metrics.json>) |
| candidates / conservative_only | 0.95126687 | 0.53816938 | 0.04383718 | -0.00004536 | — | CI [-0.00014645, +0.00005573] | [smoothed_forward_metrics.json](<../goal_0952/robust_validation/smoothed_forward_metrics.json>) |
| candidates / regularized_blend | 0.95129606 | 0.53831057 | 0.04382987 | -0.00001617 | — | CI [-0.00006690, +0.00003456] | [smoothed_forward_metrics.json](<../goal_0952/robust_validation/smoothed_forward_metrics.json>) |

## Kiểm tra độ ổn định và calibration · 10 fold đảo vai trò

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| candidates / base_only | 0.95022345 | 0.53067561 | 0.04432845 | +0.00004045 | — | CI [-0.00006573, +0.00014664] · fold ↑ 6 | [reversed_metrics.json](<../goal_0952/robust_validation/reversed_metrics.json>) |
| candidates / current_density | 0.95018299 | 0.53066892 | 0.04433998 | +0.00000000 | — | CI [-0.00000000, +0.00000000] · fold ↑ 0 | [reversed_metrics.json](<../goal_0952/robust_validation/reversed_metrics.json>) |
| candidates / conservative_only | 0.95025374 | 0.53076352 | 0.04431930 | +0.00007075 | — | CI [-0.00003402, +0.00017551] · fold ↑ 7 | [reversed_metrics.json](<../goal_0952/robust_validation/reversed_metrics.json>) |
| candidates / regularized_blend | 0.95025287 | 0.53088297 | 0.04431729 | +0.00006987 | — | CI [+0.00001752, +0.00012223] · fold ↑ 8 | [reversed_metrics.json](<../goal_0952/robust_validation/reversed_metrics.json>) |
| candidates / base_only | 0.95022345 | 0.53067561 | 0.04432845 | +0.00004045 | — | CI [-0.00006573, +0.00014664] · fold ↑ 6 | [smoothed_reversed_metrics.json](<../goal_0952/robust_validation/smoothed_reversed_metrics.json>) |
| candidates / current_density | 0.95018299 | 0.53066892 | 0.04433998 | +0.00000000 | — | CI [-0.00000000, +0.00000000] · fold ↑ 0 | [smoothed_reversed_metrics.json](<../goal_0952/robust_validation/smoothed_reversed_metrics.json>) |
| candidates / conservative_only | 0.95013666 | 0.53020555 | 0.04436149 | -0.00004633 | — | CI [-0.00013591, +0.00004325] · fold ↑ 5 | [smoothed_reversed_metrics.json](<../goal_0952/robust_validation/smoothed_reversed_metrics.json>) |
| candidates / regularized_blend | 0.95018592 | 0.53056133 | 0.04434121 | +0.00000292 | — | CI [-0.00004184, +0.00004768] · fold ↑ 6 | [smoothed_reversed_metrics.json](<../goal_0952/robust_validation/smoothed_reversed_metrics.json>) |

## Hồ sơ đồng cảnh, dấu chữ số và quan hệ số · holdout hoặc theo nhánh ghi ở nguồn

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| candidates / current | 0.95131224 | 0.53828373 | 0.04382592 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [holdout_metrics.json](<../goal_0952/data_signals/holdout_metrics.json>) |
| candidates / logic_time | 0.95129338 | 0.53831928 | 0.04382754 | -0.00001886 | — | CI [-0.00005708, +0.00001936] | [holdout_metrics.json](<../goal_0952/data_signals/holdout_metrics.json>) |
| candidates / peer_profiles | 0.95129283 | 0.53861055 | 0.04381643 | -0.00001941 | — | CI [-0.00009551, +0.00005670] | [holdout_metrics.json](<../goal_0952/data_signals/holdout_metrics.json>) |
| candidates / numeric_fingerprints | 0.95129325 | 0.53806553 | 0.04383836 | -0.00001899 | — | CI [-0.00008566, +0.00004769] | [holdout_metrics.json](<../goal_0952/data_signals/holdout_metrics.json>) |
| candidates / combined | 0.95125725 | 0.53824256 | 0.04383429 | -0.00005498 | — | CI [-0.00014611, +0.00003614] | [holdout_metrics.json](<../goal_0952/data_signals/holdout_metrics.json>) |

## Hồ sơ đồng cảnh, dấu chữ số và quan hệ số · 10 fold đảo vai trò

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| metrics / current | 0.95002686 | 0.52995291 | 0.04439411 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [reversed_metrics.json](<../goal_0952/data_signals/reversed_metrics.json>) |
| metrics / peer_profiles | 0.95002094 | 0.52998627 | 0.04439171 | -0.00000592 | — | CI [-0.00005792, +0.00004608] | [reversed_metrics.json](<../goal_0952/data_signals/reversed_metrics.json>) |

## Không gian điểm, bất đồng và nhãn mềm · holdout hoặc theo nhánh ghi ở nguồn

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| variants / current | 0.95131224 | 0.53828373 | 0.04382592 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [holdout_metrics.json](<../goal_0952/score_space/holdout_metrics.json>) |
| variants / quadratic | 0.95134001 | 0.53864400 | 0.04381021 | +0.00002777 | — | CI [-0.00004221, +0.00009775] | [holdout_metrics.json](<../goal_0952/score_space/holdout_metrics.json>) |
| variants / kernel_from_mean | 0.95130161 | 0.53844498 | 0.04382359 | -0.00001063 | — | CI [-0.00003153, +0.00001027] | [holdout_metrics.json](<../goal_0952/score_space/holdout_metrics.json>) |
| variants / kernel_fold_average | 0.95130037 | 0.53839768 | 0.04382380 | -0.00001186 | — | CI [-0.00003168, +0.00000796] | [holdout_metrics.json](<../goal_0952/score_space/holdout_metrics.json>) |
| variants / current | 0.95131224 | 0.53828373 | 0.04382592 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [soft_holdout_metrics.json](<../goal_0952/score_space/soft_holdout_metrics.json>) |
| variants / shrink_only | 0.95129698 | 0.53834061 | 0.04382538 | -0.00001526 | — | CI [-0.00003982, +0.00000930] | [soft_holdout_metrics.json](<../goal_0952/score_space/soft_holdout_metrics.json>) |
| variants / soft_targets | 0.95129512 | 0.53838016 | 0.04382499 | -0.00001712 | — | CI [-0.00004522, +0.00001098] | [soft_holdout_metrics.json](<../goal_0952/score_space/soft_holdout_metrics.json>) |

## Không gian điểm, bất đồng và nhãn mềm · 10 fold đảo vai trò

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| variants / current | 0.95018299 | 0.53066892 | 0.04433998 | +0.00000000 | +0.00000000 | CI [-0.00000000, +0.00000000] · fold ↑ 0 | [reversed_metrics.json](<../goal_0952/score_space/reversed_metrics.json>) |
| variants / quadratic | 0.95015218 | 0.53081274 | 0.04433802 | -0.00003081 | -0.00002963 | CI [-0.00008012, +0.00001850] · fold ↑ 4 | [reversed_metrics.json](<../goal_0952/score_space/reversed_metrics.json>) |
| variants / kernel | 0.95016362 | 0.53074182 | 0.04433907 | -0.00001938 | -0.00001818 | CI [-0.00003639, -0.00000237] · fold ↑ 3 | [reversed_metrics.json](<../goal_0952/score_space/reversed_metrics.json>) |
| variants / current | 0.95002686 | 0.52995291 | 0.04439411 | +0.00000000 | +0.00000000 | CI [-0.00000000, +0.00000000] · fold ↑ 0 | [soft_reversed_metrics.json](<../goal_0952/score_space/soft_reversed_metrics.json>) |
| variants / shrink_only | 0.95014700 | 0.53048335 | 0.04435237 | +0.00012014 | +0.00011889 | CI [+0.00009495, +0.00014533] · fold ↑ 10 | [soft_reversed_metrics.json](<../goal_0952/score_space/soft_reversed_metrics.json>) |
| variants / soft_targets | 0.95015374 | 0.53051489 | 0.04435048 | +0.00012689 | +0.00012738 | CI [+0.00009857, +0.00015521] · fold ↑ 10 | [soft_reversed_metrics.json](<../goal_0952/score_space/soft_reversed_metrics.json>) |

## Bảng EDA khớp cỡ mẫu và làm mượt · holdout hoặc theo nhánh ghi ở nguồn

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| variants / grandmaster_eda | 0.95122445 | 0.53823528 | 0.04383599 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [holdout_metrics.json](<../goal_0952/grandmaster_eda_refinement/holdout_metrics.json>) |
| variants / matched_encoding_size | 0.95121087 | 0.53815394 | 0.04383945 | -0.00001358 | — | CI [-0.00004510, +0.00001794] | [holdout_metrics.json](<../goal_0952/grandmaster_eda_refinement/holdout_metrics.json>) |
| variants / smoothed_pairs | 0.95121100 | 0.53810391 | 0.04384148 | -0.00001345 | — | CI [-0.00006784, +0.00004094] | [holdout_metrics.json](<../goal_0952/grandmaster_eda_refinement/holdout_metrics.json>) |
| variants / smoothed_pairs_half | 0.95121931 | 0.53815166 | 0.04383819 | -0.00000514 | — | CI [-0.00003236, +0.00002208] | [holdout_metrics.json](<../goal_0952/grandmaster_eda_refinement/holdout_metrics.json>) |

## Bảng EDA khớp cỡ mẫu và làm mượt · 10 fold đảo vai trò

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| variants / grandmaster_eda | 0.95018970 | 0.53079830 | 0.04433165 | +0.00000000 | +0.00000000 | CI [-0.00000000, +0.00000000] · fold ↑ 0 | [reverse_metrics.json](<../goal_0952/grandmaster_eda_refinement/reverse_metrics.json>) |
| variants / matched_encoding_size | 0.95024952 | 0.53092856 | 0.04431592 | +0.00005981 | +0.00005404 | CI [+0.00001529, +0.00010434] · fold ↑ 7 | [reverse_metrics.json](<../goal_0952/grandmaster_eda_refinement/reverse_metrics.json>) |
| variants / smoothed_pairs | 0.95012022 | 0.53057268 | 0.04434659 | -0.00006949 | -0.00006778 | CI [-0.00013013, -0.00000885] · fold ↑ 1 | [reverse_metrics.json](<../goal_0952/grandmaster_eda_refinement/reverse_metrics.json>) |
| variants / smoothed_pairs_half | 0.95016633 | 0.53074153 | 0.04433516 | -0.00002337 | -0.00002392 | CI [-0.00005364, +0.00000690] · fold ↑ 1 | [reverse_metrics.json](<../goal_0952/grandmaster_eda_refinement/reverse_metrics.json>) |

## Tương tác theo rủi ro và 10 cặp EDA bổ sung · holdout hoặc theo nhánh ghi ở nguồn

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| variants / grandmaster_eda | 0.95122445 | 0.53823528 | 0.04383599 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [development_metrics.json](<../goal_0952/grandmaster_eda_risk/development_metrics.json>) |
| variants / risk_conditioned_tree | 0.95112217 | 0.53804466 | 0.04385380 | -0.00010228 | — | CI [-0.00020253, -0.00000202] | [development_metrics.json](<../goal_0952/grandmaster_eda_risk/development_metrics.json>) |
| variants / risk_conditioned_half | 0.95117764 | 0.53815946 | 0.04384304 | -0.00004681 | — | CI [-0.00009709, +0.00000346] | [development_metrics.json](<../goal_0952/grandmaster_eda_risk/development_metrics.json>) |
| variants / risk_conditioned_linear | 0.95122158 | 0.53823341 | 0.04383549 | -0.00000287 | — | CI [-0.00001183, +0.00000608] | [development_metrics.json](<../goal_0952/grandmaster_eda_risk/development_metrics.json>) |
| variants / grandmaster_eda | 0.95122445 | 0.53823528 | 0.04383599 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [extra_development_metrics.json](<../goal_0952/grandmaster_eda_risk/extra_development_metrics.json>) |
| variants / extra_pairs_tree | 0.95123132 | 0.53847140 | 0.04382906 | +0.00000687 | — | CI [-0.00003383, +0.00004757] | [extra_development_metrics.json](<../goal_0952/grandmaster_eda_risk/extra_development_metrics.json>) |
| variants / extra_pairs_half | 0.95122902 | 0.53836860 | 0.04383213 | +0.00000458 | — | CI [-0.00001594, +0.00002509] | [extra_development_metrics.json](<../goal_0952/grandmaster_eda_risk/extra_development_metrics.json>) |
| variants / extra_pairs_linear | 0.95123214 | 0.53822991 | 0.04383585 | +0.00000769 | — | CI [-0.00000233, +0.00001770] | [extra_development_metrics.json](<../goal_0952/grandmaster_eda_risk/extra_development_metrics.json>) |
| placebo_vs_control | 0.95120693 | 0.53825368 | 0.04383928 | -0.00001752 | — | CI [-0.00004304, +0.00000800] | [extra_placebo_metrics.json](<../goal_0952/grandmaster_eda_risk/extra_placebo_metrics.json>) |
| real_vs_placebo | 0.95123132 | 0.53847140 | 0.04382906 | +0.00002439 | — | CI [-0.00001776, +0.00006654] | [extra_placebo_metrics.json](<../goal_0952/grandmaster_eda_risk/extra_placebo_metrics.json>) |
| real_vs_control | 0.95123132 | 0.53847140 | 0.04382906 | +0.00000687 | — | CI [-0.00003383, +0.00004757] | [extra_placebo_metrics.json](<../goal_0952/grandmaster_eda_risk/extra_placebo_metrics.json>) |

## Tương tác theo rủi ro và 10 cặp EDA bổ sung · 10 fold đảo vai trò

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| variants / grandmaster_eda | 0.95018970 | 0.53079830 | 0.04433165 | +0.00000000 | +0.00000000 | CI [-0.00000000, +0.00000000] · fold ↑ 0 | [extra_reverse_metrics.json](<../goal_0952/grandmaster_eda_risk/extra_reverse_metrics.json>) |
| variants / extra_pairs_tree | 0.95017885 | 0.53070916 | 0.04433639 | -0.00001085 | -0.00000988 | CI [-0.00003628, +0.00001458] · fold ↑ 4 | [extra_reverse_metrics.json](<../goal_0952/grandmaster_eda_risk/extra_reverse_metrics.json>) |
| variants / extra_pairs_half | 0.95018614 | 0.53076611 | 0.04433333 | -0.00000356 | -0.00000194 | CI [-0.00001628, +0.00000916] · fold ↑ 4 | [extra_reverse_metrics.json](<../goal_0952/grandmaster_eda_risk/extra_reverse_metrics.json>) |
| variants / extra_pairs_linear | 0.95018979 | 0.53079879 | 0.04433163 | +0.00000009 | +0.00000014 | CI [-0.00000004, +0.00000021] · fold ↑ 3 | [extra_reverse_metrics.json](<../goal_0952/grandmaster_eda_risk/extra_reverse_metrics.json>) |
| variants / grandmaster_eda | 0.95018970 | 0.53079830 | 0.04433165 | +0.00000000 | +0.00000000 | CI [-0.00000000, +0.00000000] · fold ↑ 0 | [reverse_metrics.json](<../goal_0952/grandmaster_eda_risk/reverse_metrics.json>) |
| variants / risk_conditioned_tree | 0.95014898 | 0.53086850 | 0.04433287 | -0.00004072 | -0.00004679 | CI [-0.00012391, +0.00004246] · fold ↑ 3 | [reverse_metrics.json](<../goal_0952/grandmaster_eda_risk/reverse_metrics.json>) |
| variants / risk_conditioned_half | 0.95018783 | 0.53091740 | 0.04432662 | -0.00000187 | -0.00000589 | CI [-0.00004346, +0.00003971] · fold ↑ 4 | [reverse_metrics.json](<../goal_0952/grandmaster_eda_risk/reverse_metrics.json>) |
| variants / risk_conditioned_linear | 0.95018910 | 0.53079899 | 0.04433175 | -0.00000060 | -0.00000083 | CI [-0.00000189, +0.00000069] · fold ↑ 4 | [reverse_metrics.json](<../goal_0952/grandmaster_eda_risk/reverse_metrics.json>) |

## Phân vị mịn, giảm biên và trộn ba hướng · holdout hoặc theo nhánh ghi ở nguồn

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| variants / grandmaster_eda | 0.95122445 | 0.53823528 | 0.04383599 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [development_metrics.json](<../goal_0952/parallel_eda_resolution/development_metrics.json>) |
| variants / resolution_tree | 0.95120508 | 0.53822095 | 0.04383828 | -0.00001937 | — | CI [-0.00004478, +0.00000604] | [development_metrics.json](<../goal_0952/parallel_eda_resolution/development_metrics.json>) |
| variants / resolution_half | 0.95121552 | 0.53826113 | 0.04383702 | -0.00000893 | — | CI [-0.00002174, +0.00000389] | [development_metrics.json](<../goal_0952/parallel_eda_resolution/development_metrics.json>) |
| variants / resolution_linear | 0.95122403 | 0.53822355 | 0.04383642 | -0.00000042 | — | CI [-0.00000297, +0.00000213] | [development_metrics.json](<../goal_0952/parallel_eda_resolution/development_metrics.json>) |

## Phân vị mịn, giảm biên và trộn ba hướng · holdout và 10 fold đảo vai trò

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| phases / holdout / variants / grandmaster_eda | 0.95122445 | 0.53823528 | 0.04383599 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [parallel_three_way_metrics.json](<../goal_0952/parallel_eda_resolution/parallel_three_way_metrics.json>) |
| phases / holdout / variants / parallel_three_way | 0.95121425 | 0.53838517 | 0.04382859 | -0.00001019 | — | CI [-0.00003288, +0.00001250] | [parallel_three_way_metrics.json](<../goal_0952/parallel_eda_resolution/parallel_three_way_metrics.json>) |
| phases / reverse / variants / grandmaster_eda | 0.95018970 | 0.53079830 | 0.04433165 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [parallel_three_way_metrics.json](<../goal_0952/parallel_eda_resolution/parallel_three_way_metrics.json>) |
| phases / reverse / variants / parallel_three_way | 0.95021322 | 0.53090651 | 0.04432373 | +0.00002352 | +0.00002271 | CI [+0.00001450, +0.00003254] · fold ↑ 9 | [parallel_three_way_metrics.json](<../goal_0952/parallel_eda_resolution/parallel_three_way_metrics.json>) |

## Phân vị mịn, giảm biên và trộn ba hướng · 10 fold đảo vai trò

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| variants / grandmaster_eda | 0.95018970 | 0.53079830 | 0.04433165 | +0.00000000 | +0.00000000 | CI [-0.00000000, +0.00000000] · fold ↑ 0 | [reverse_metrics.json](<../goal_0952/parallel_eda_resolution/reverse_metrics.json>) |
| variants / resolution_tree | 0.95017676 | 0.53072282 | 0.04433557 | -0.00001294 | -0.00001371 | CI [-0.00005072, +0.00002484] · fold ↑ 4 | [reverse_metrics.json](<../goal_0952/parallel_eda_resolution/reverse_metrics.json>) |
| variants / resolution_half | 0.95018749 | 0.53078749 | 0.04433193 | -0.00000221 | -0.00000320 | CI [-0.00002110, +0.00001669] · fold ↑ 5 | [reverse_metrics.json](<../goal_0952/parallel_eda_resolution/reverse_metrics.json>) |
| variants / resolution_linear | 0.95018962 | 0.53079742 | 0.04433167 | -0.00000008 | -0.00000016 | CI [-0.00000026, +0.00000010] · fold ↑ 3 | [reverse_metrics.json](<../goal_0952/parallel_eda_resolution/reverse_metrics.json>) |

## Tương quan có điều kiện và khoảng cách đồng cảnh · holdout hoặc theo nhánh ghi ở nguồn

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| variants / grandmaster_eda | 0.95122445 | 0.53823528 | 0.04383599 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [holdout_metrics.json](<../goal_0952/parallel_peer_context/holdout_metrics.json>) |
| variants / joint_context | 0.95113668 | 0.53908200 | 0.04380857 | -0.00008777 | — | CI [-0.00020552, +0.00002999] | [holdout_metrics.json](<../goal_0952/parallel_peer_context/holdout_metrics.json>) |
| variants / joint_context_half | 0.95118936 | 0.53870944 | 0.04381704 | -0.00003509 | — | CI [-0.00009512, +0.00002494] | [holdout_metrics.json](<../goal_0952/parallel_peer_context/holdout_metrics.json>) |

## Tương quan có điều kiện và khoảng cách đồng cảnh · 10 fold đảo vai trò

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| variants / grandmaster_eda | 0.95018970 | 0.53079830 | 0.04433165 | +0.00000000 | +0.00000000 | CI [-0.00000000, +0.00000000] · fold ↑ 0 | [reverse_metrics.json](<../goal_0952/parallel_peer_context/reverse_metrics.json>) |
| variants / joint_context | 0.95019927 | 0.53087124 | 0.04432691 | +0.00000957 | +0.00000914 | CI [-0.00002427, +0.00004341] · fold ↑ 4 | [reverse_metrics.json](<../goal_0952/parallel_peer_context/reverse_metrics.json>) |
| variants / joint_context_half | 0.95019794 | 0.53086089 | 0.04432804 | +0.00000823 | +0.00000743 | CI [-0.00000869, +0.00002516] · fold ↑ 6 | [reverse_metrics.json](<../goal_0952/parallel_peer_context/reverse_metrics.json>) |

## LightGBM một nút tách và cây tối đa năm lá · holdout hoặc theo nhánh ghi ở nguồn

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| variants / grandmaster_eda | 0.95122445 | 0.53823528 | 0.04383599 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [holdout_metrics.json](<../goal_0952/parallel_tree_calibration/holdout_metrics.json>) |
| variants / additive_stumps | 0.95122558 | 0.53823066 | 0.04383578 | +0.00000113 | — | CI [-0.00001580, +0.00001806] | [holdout_metrics.json](<../goal_0952/parallel_tree_calibration/holdout_metrics.json>) |
| variants / five_leaf_trees | 0.95122496 | 0.53821075 | 0.04383563 | +0.00000051 | — | CI [-0.00001581, +0.00001683] | [holdout_metrics.json](<../goal_0952/parallel_tree_calibration/holdout_metrics.json>) |

## LightGBM một nút tách và cây tối đa năm lá · 10 fold đảo vai trò

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| variants / grandmaster_eda | 0.95018970 | 0.53079830 | 0.04433165 | +0.00000000 | +0.00000000 | CI [-0.00000000, +0.00000000] · fold ↑ 0 | [reverse_metrics.json](<../goal_0952/parallel_tree_calibration/reverse_metrics.json>) |
| variants / additive_stumps | 0.95024763 | 0.53104082 | 0.04431307 | +0.00005793 | +0.00005627 | CI [+0.00003992, +0.00007593] · fold ↑ 10 | [reverse_metrics.json](<../goal_0952/parallel_tree_calibration/reverse_metrics.json>) |
| variants / five_leaf_trees | 0.95023438 | 0.53104669 | 0.04431400 | +0.00004468 | +0.00004212 | CI [+0.00002701, +0.00006236] · fold ↑ 10 | [reverse_metrics.json](<../goal_0952/parallel_tree_calibration/reverse_metrics.json>) |

## Đối chứng giảm biên 15% · holdout và 10 fold

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| holdout / grandmaster_eda | 0.95122445 | 0.53823528 | 0.04383599 | +0.00000000 | — | CI [-0.00000000, +0.00000000] | [stump_shrink_audit.json](<../goal_0952/parallel_eda_resolution/stump_shrink_audit.json>) |
| holdout / additive_stumps | 0.95122558 | 0.53823066 | 0.04383578 | +0.00000113 | — | CI [-0.00001580, +0.00001806] | [stump_shrink_audit.json](<../goal_0952/parallel_eda_resolution/stump_shrink_audit.json>) |
| holdout / shrink_toward_grandmaster | 0.95121511 | 0.53831307 | 0.04383394 | -0.00000934 | — | CI [-0.00004029, +0.00002162] | [stump_shrink_audit.json](<../goal_0952/parallel_eda_resolution/stump_shrink_audit.json>) |
| reverse / grandmaster_eda | 0.95018970 | 0.53079830 | 0.04433165 | +0.00000000 | +0.00000000 | CI [-0.00000000, +0.00000000] · fold ↑ 0 | [stump_shrink_audit.json](<../goal_0952/parallel_eda_resolution/stump_shrink_audit.json>) |
| reverse / additive_stumps | 0.95024763 | 0.53104082 | 0.04431307 | +0.00005793 | +0.00005627 | CI [+0.00003992, +0.00007593] · fold ↑ 10 | [stump_shrink_audit.json](<../goal_0952/parallel_eda_resolution/stump_shrink_audit.json>) |
| reverse / shrink_toward_grandmaster | 0.95032367 | 0.53134380 | 0.04428681 | +0.00013397 | +0.00013460 | CI [+0.00009858, +0.00016935] · fold ↑ 10 | [stump_shrink_audit.json](<../goal_0952/parallel_eda_resolution/stump_shrink_audit.json>) |
| holdout_stumps_vs_shrink | 0.95122558 | 0.53823066 | 0.04383578 | +0.00001047 | — | CI [-0.00001220, +0.00003313] | [stump_shrink_audit.json](<../goal_0952/parallel_eda_resolution/stump_shrink_audit.json>) |
| reverse_stumps_vs_shrink | 0.95024763 | 0.53104082 | 0.04431307 | -0.00007604 | -0.00007833 | CI [-0.00010150, -0.00005057] · fold ↑ 0 | [stump_shrink_audit.json](<../goal_0952/parallel_eda_resolution/stump_shrink_audit.json>) |

## EDA hành vi · các hiệu chỉnh ngoài fold đã khóa

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| Phát triển / clock_block_pair / dev_pooled | 0.95043929 | — | — | -0.00012311 | — | giảm log loss × 10⁶ = -59.902 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Phát triển / encoded_week_transition_channel / dev_pooled | 0.95059583 | — | — | +0.00003343 | — | giảm log loss × 10⁶ = +1.022 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Phát triển / night_continuity_count / dev_pooled | 0.95052586 | — | — | -0.00003654 | — | giảm log loss × 10⁶ = -10.885 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Phát triển / rapid_amount_new_merchant / dev_pooled | 0.95050352 | — | — | -0.00005888 | — | giảm log loss × 10⁶ = -42.406 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Phát triển / amount_count_relative_to_balance / dev_pooled | 0.95045145 | — | — | -0.00011094 | — | giảm log loss × 10⁶ = -49.503 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Phát triển / remaining_balance_card_type / dev_pooled | 0.95056510 | — | — | +0.00000270 | — | giảm log loss × 10⁶ = +1.583 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Phát triển / spend_share_auth_address / dev_pooled | 0.95043473 | — | — | -0.00012767 | — | giảm log loss × 10⁶ = -41.206 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Phát triển / short_gap_distance_channel / dev_pooled | 0.95046028 | — | — | -0.00010211 | — | giảm log loss × 10⁶ = -40.710 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Phát triển / spend_share_card_tenure / dev_pooled | 0.95051820 | — | — | -0.00004419 | — | giảm log loss × 10⁶ = -37.097 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Holdout / encoded_week_transition_channel / grandmaster_risk_band_control | 0.95081111 | — | — | -0.00013841 | — | giảm log loss × 10⁶ = -29.756 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Holdout / encoded_week_transition_channel / grandmaster_raw | 0.95081972 | — | — | -0.00012921 | — | giảm log loss × 10⁶ = -29.889 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Holdout / encoded_week_transition_channel / grandmaster_eda | 0.95108660 | — | — | -0.00013785 | — | giảm log loss × 10⁶ = -33.586 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Holdout / encoded_week_transition_channel / mixed_density | 0.95117459 | — | — | -0.00013765 | — | giảm log loss × 10⁶ = -33.236 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Holdout / remaining_balance_card_type / grandmaster_risk_band_control | 0.95085381 | — | — | -0.00009572 | — | giảm log loss × 10⁶ = -22.421 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Holdout / remaining_balance_card_type / grandmaster_raw | 0.95085418 | — | — | -0.00009475 | — | giảm log loss × 10⁶ = -21.951 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Holdout / remaining_balance_card_type / grandmaster_eda | 0.95112464 | — | — | -0.00009981 | — | giảm log loss × 10⁶ = -22.108 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |
| Holdout / remaining_balance_card_type / mixed_density | 0.95121264 | — | — | -0.00009960 | — | giảm log loss × 10⁶ = -22.447 | [behavior_audit.json](<../goal_0952/deep_eda_behavior/behavior_audit.json>) |

## EDA ba biến · phép chấm holdout mô tả

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| txn_count_last_24h × hours_since_last_txn × channel | 0.95122878 | 0.53814405 | 0.04383853 | +0.00000433 | — | holdout sau khóa A/B · CI [-0.00002490, +0.00003357] | [triple_eda_results.json](<../goal_0952/deep_eda_triples/triple_eda_results.json>) |
| card_type × auth_method × cvv_retry_count | 0.95120656 | 0.53812200 | 0.04384151 | -0.00001789 | — | holdout sau khóa A/B · CI [-0.00004428, +0.00000850] | [triple_eda_results.json](<../goal_0952/deep_eda_triples/triple_eda_results.json>) |
| cvv_retry_count × prior_disputes × auth_method | 0.95123906 | 0.53816804 | 0.04383394 | +0.00001461 | — | holdout sau khóa A/B · CI [+0.00000328, +0.00002595] | [triple_eda_results.json](<../goal_0952/deep_eda_triples/triple_eda_results.json>) |

## EDA giá trị thưa · holdout đúng Grandmaster EDA

| Cấu hình hoặc trường chỉ số | ROC AUC | AP / PR AUC | Log loss | Δ AUC | Δ AUC trọng số | Ghi chú | Nguồn |
|---|---:|---:|---:|---:|---:|---|---|
| amount_usd_hole · exact Grandmaster EDA | 0.95131585 | 0.53852282 | 0.04380602 | +0.00009140 | — | CI [-0.00002229, +0.00020508] | [measurement_metrics.json](<../goal_0952/deep_eda_measurement/measurement_metrics.json>) |

## Refit 100% và lần xuất test

Các lượt dưới đây dùng đủ 699.960 dòng train. AUC tham chiếu, khi có, thuộc phép chia đã đánh giá trước đó. Tương quan Spearman là độ giống nhau về thứ hạng của dự đoán test so với dự đoán gộp từ fold.

| Cấu hình | Số vòng / thành viên | AUC OOF tham chiếu | Spearman với bản gộp fold | Nguồn |
|---|---:|---:|---:|---|
| D03 | 1563 | 0.95013086 | 0.99754797 | [metrics.json](<../s6e11_gpu/single_full_refit/results/D03/metrics.json>) |
| XGB-D03 | 2131 | 0.95018230 | 0.99718590 | [metrics.json](<../s6e11_gpu/single_full_refit/results/XGB-D03/metrics.json>) |
| XGB-D03-R125 | 2664 | 0.95014625 | 0.99656668 | [metrics.json](<../s6e11_gpu/single_full_refit/results/XGB-D03-R125/metrics.json>) |
| XGB-D03-S71-E2028 | 1983 | 0.95020055 | 0.99710453 | [metrics.json](<../s6e11_gpu/single_full_refit/results/XGB-D03-S71-E2028/metrics.json>) |
| XGB-HARD | 2131 | 0.95014087 | 0.99636902 | [metrics.json](<../s6e11_gpu/single_full_refit/results/XGB-HARD/metrics.json>) |
| XGB-RANDOM | 2131 | 0.95017683 | 0.99693642 | [metrics.json](<../s6e11_gpu/single_full_refit/results/XGB-RANDOM/metrics.json>) |
| FINAL | 2 | 0.95034521 | 0.99812517 | [metrics.json](<../s6e11_gpu/blend_full_refit/results/FINAL/metrics.json>) |
| FINAL3 | 3 | 0.95039383 | 0.99850208 | [metrics.json](<../s6e11_gpu/blend_full_refit/results/FINAL3/metrics.json>) |
| FINAL_PLUS_LGBM20 | 3 | — | 0.99831831 | [metrics.json](<../s6e11_gpu/blend_full_refit/results/FINAL_PLUS_LGBM20/metrics.json>) |
| SEED3-MEAN | 3 | 0.95026928 | 0.99814269 | [metrics.json](<../s6e11_gpu/blend_full_refit/results/SEED3-MEAN/metrics.json>) |
| SEED5-MEAN | 5 | 0.95033114 | 0.99846105 | [metrics.json](<../s6e11_gpu/blend_full_refit/results/SEED5-MEAN/metrics.json>) |
| LGBM-SEED5-MEAN | 5 × 1.563 | — | — | [metrics.json](<../s6e11_gpu/lightgbm_seed5/results/LGBM-SEED5-MEAN/metrics.json>) |
| Grandmaster full refit | 7 nhánh | OOF đối chứng 0.95128045 | xem từng quy trình | [metrics.json](<../grandmaster_refit/full/metrics.json>) |

## Danh mục 71 CSV lịch sử

Khi lập danh mục cục bộ, đã đọc cả 71 CSV và đối chiếu hai cột `id,is_fraud`, 299.984 ID theo đúng thứ tự mẫu và giá trị trong [0,1]. Git lưu hai CSV có liên kết trong danh mục, gồm đối chứng Grandmaster EDA và ứng viên hiệu chỉnh số tiền thưa. Các đường dẫn dạng mã ghi vị trí của 69 CSV còn lại trong các lượt chạy lịch sử. Tệp dự đoán theo fold, checkpoint mô hình và CSV holdout được tạo cục bộ khi tái lập. [Cách kiểm tra hai CSV phát hành](06_TAI_LAP_VA_SUBMISSION.md#xác-minh-hai-csv-phát-hành)

### Grandmaster tại gốc

- `submission_grandmaster.csv`

### s6e11_gpu

- `s6e11_gpu/blend_full_refit/results/base_models/GLM-NEWTON/GLM-NEWTON_FULLREFIT_submission.csv`
- `s6e11_gpu/blend_full_refit/results/base_models/XGB-D03-S103-E2032/XGB-D03-S103-E2032_FULLREFIT_submission.csv`
- `s6e11_gpu/blend_full_refit/results/base_models/XGB-D03-S71/XGB-D03-S71_FULLREFIT_submission.csv`
- `s6e11_gpu/blend_full_refit/results/base_models/XGB-D03-S93/XGB-D03-S93_FULLREFIT_submission.csv`
- `s6e11_gpu/blend_full_refit/results/FINAL/FINAL_FULLREFIT_submission.csv`
- `s6e11_gpu/blend_full_refit/results/FINAL3/FINAL3_FULLREFIT_submission.csv`
- `s6e11_gpu/blend_full_refit/results/FINAL_PLUS_LGBM20/FINAL_PLUS_LGBM20_FULLREFIT_submission.csv`
- `s6e11_gpu/blend_full_refit/results/SEED3-MEAN/SEED3-MEAN_FULLREFIT_submission.csv`
- `s6e11_gpu/blend_full_refit/results/SEED5-MEAN/SEED5-MEAN_FULLREFIT_submission.csv`
- `s6e11_gpu/final_feature_optimization/results/submissions/01_FINAL-RARE-TE_submission.csv`
- `s6e11_gpu/final_feature_optimization/results/submissions/02_FINAL-PAIR-TE_submission.csv`
- `s6e11_gpu/final_feature_optimization/results/submissions/03_FINAL-PAIR-RARE-TE_submission.csv`
- `s6e11_gpu/final_three_stack/FINAL3_GLM-NEWTON_LGBM-D03_XGB-D03_submission.csv`
- `s6e11_gpu/followup/submissions/01_FINAL_LGBM_D03_20pct_submission.csv`
- `s6e11_gpu/followup/submissions/02_FINAL_MLP_BCE_10pct_submission.csv`
- `s6e11_gpu/followup/submissions/03_FINAL_NO_FLAGS_submission.csv`
- `s6e11_gpu/lightgbm_seed5/results/LGBM-SEED5-MEAN/LGBM-SEED5-MEAN_FULLREFIT_submission.csv`
- `s6e11_gpu/plan_runs/results/full/final/submission.csv`
- `s6e11_gpu/plan_runs/results/full/top5_submissions/01_FINAL_submission.csv`
- `s6e11_gpu/plan_runs/results/full/top5_submissions/02_ALL3-MEAN_submission.csv`
- `s6e11_gpu/plan_runs/results/full/top5_submissions/03_TREES2-MEAN_submission.csv`
- `s6e11_gpu/plan_runs/results/full/top5_submissions/04_ALL3-RANK_submission.csv`
- `s6e11_gpu/plan_runs/results/full/top5_submissions/05_SEED3-MEAN_submission.csv`
- `s6e11_gpu/seed3_improvement/results/SEED5-MEAN/SEED5-MEAN_submission.csv`
- `s6e11_gpu/single_full_refit/results/D03/D03_FULLREFIT_submission.csv`
- `s6e11_gpu/single_full_refit/results/XGB-D03-R125/XGB-D03-R125_FULLREFIT_submission.csv`
- `s6e11_gpu/single_full_refit/results/XGB-D03-S71-E2028/XGB-D03-S71-E2028_FULLREFIT_submission.csv`
- `s6e11_gpu/single_full_refit/results/XGB-D03/XGB-D03_FULLREFIT_submission.csv`
- `s6e11_gpu/single_full_refit/results/XGB-HARD/XGB-HARD_FULLREFIT_submission.csv`
- `s6e11_gpu/single_full_refit/results/XGB-RANDOM/XGB-RANDOM_FULLREFIT_submission.csv`

### grandmaster_refit

- `grandmaster_refit/full/submission_full_refit.csv`
- `grandmaster_refit/full/submission_refit_fixed_stack.csv`

### grandmaster_10fold_ablation

- `grandmaster_10fold_ablation/submission_hierarchical_amount/submission.csv`

### novel_model_experiments

- `novel_model_experiments/results/submission_control.csv`
- `novel_model_experiments/results/submission_copula.csv`
- `novel_model_experiments/results/submission_fixed_blend.csv`
- `novel_model_experiments/results/submission_hours_residual.csv`
- `novel_model_experiments/results/submission_neighborhood.csv`
- `novel_model_experiments/results/submission_spline_blend.csv`
- `novel_model_experiments/submission_occupancy.csv`
- `novel_model_experiments/submission_spline_interactions.csv`

### goal_0952

- `goal_0952/data_signals/submission_peer_profiles.csv`
- [goal_0952/deep_eda_measurement/submission_grandmaster_eda_amount_hole.csv](<../goal_0952/deep_eda_measurement/submission_grandmaster_eda_amount_hole.csv>)
- `goal_0952/followup/submission_challenger.csv`
- `goal_0952/followup/submission_context.csv`
- `goal_0952/grandmaster_eda_refinement/submission_grandmaster_eda_matched_encoding_size.csv`
- `goal_0952/grandmaster_eda_refinement/submission_grandmaster_eda_smoothed_pairs.csv`
- `goal_0952/grandmaster_eda_refinement/submission_grandmaster_eda_smoothed_pairs_half.csv`
- `goal_0952/grandmaster_eda_risk/submission_grandmaster_eda_extra_pairs_half.csv`
- `goal_0952/grandmaster_eda_risk/submission_grandmaster_eda_extra_pairs_linear.csv`
- `goal_0952/grandmaster_eda_risk/submission_grandmaster_eda_extra_pairs_tree.csv`
- `goal_0952/grandmaster_eda_risk/submission_grandmaster_eda_risk_conditioned_half.csv`
- `goal_0952/grandmaster_eda_risk/submission_grandmaster_eda_risk_conditioned_linear.csv`
- `goal_0952/grandmaster_eda_risk/submission_grandmaster_eda_risk_conditioned_tree.csv`
- `goal_0952/parallel_eda_resolution/submission_grandmaster_eda_parallel_three_way.csv`
- `goal_0952/parallel_eda_resolution/submission_grandmaster_eda_resolution_half.csv`
- `goal_0952/parallel_eda_resolution/submission_grandmaster_eda_resolution_linear.csv`
- `goal_0952/parallel_eda_resolution/submission_grandmaster_eda_resolution_tree.csv`
- `goal_0952/parallel_eda_resolution/submission_grandmaster_eda_shrink_toward_grandmaster.csv`
- `goal_0952/parallel_peer_context/submission_grandmaster_eda_joint_context.csv`
- `goal_0952/parallel_peer_context/submission_grandmaster_eda_joint_context_half.csv`
- `goal_0952/parallel_tree_calibration/submission_grandmaster_eda_additive_stumps.csv`
- `goal_0952/parallel_tree_calibration/submission_grandmaster_eda_five_leaf_trees.csv`
- `goal_0952/robust_validation/submission_regularized.csv`
- `goal_0952/score_space/submission_quadratic.csv`
- `goal_0952/score_space/submission_soft_targets.csv`
- [goal_0952/submission_grandmaster_eda.csv](<../goal_0952/submission_grandmaster_eda.csv>)

### Model tham khảo

- `Model tham khảo/Grandmaster/submission.csv`
- `Model tham khảo/V1/submission.csv`
- `Model tham khảo/V2/submission.csv`
