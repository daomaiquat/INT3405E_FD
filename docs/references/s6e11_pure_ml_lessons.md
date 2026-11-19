# S6E11 Credit Card Fraud: những điều hữu ích cho dự án ML thuần

Tài liệu này lọc từ báo cáo tiến độ của một người tham gia cuộc thi (CV từ 0.9469 lên 0.9526). Phạm vi dự án của bạn: **chỉ dùng mô hình tự huấn luyện trên dữ liệu cuộc thi, không dùng dữ liệu ngoài, không dùng mô hình huấn luyện sẵn, không dùng dự đoán của người khác.** Vì vậy phần lớn con đường của tác giả về sau (các "pool" dự đoán công khai, các file xếp hạng công khai) bị loại, chỉ giữ lại những gì áp dụng được.

Mọi con số dưới đây là **số của tác giả** trên 5 fold `StratifiedKFold(5, shuffle=True, random_state=42)`. Chúng là mốc để đối chiếu với số liệu của bạn, không phải số của bạn.

---

## 1. Mốc tham chiếu cho dự án thuần ML

Trần của các mô hình tự xây (không dùng nguồn ngoài) trong báo cáo là **CV 0.95111, public 0.95105**. Đó là mục tiêu thực tế cho dự án của bạn.

| Bước | Mô hình | CV AUC |
|---|---|---|
| 0 | LightGBM 15 lá trên cột gốc | 0.94689 |
| 1 | LightGBM 4 lá (3 seed) | 0.94756 |
| 1 | XGBoost depth 2 | 0.94753 |
| 1 | CatBoost depth 6 | 0.94583 |
| 1 | Logistic với spline cho cột số + one-hot | 0.94338 |
| 2 | XGBoost depth 2 + mã hóa theo nhãn từng giá trị | 0.95051 |
| 2 | LightGBM 4 lá + mã hóa | 0.95038 |
| 2 | Logistic trên log-odds của các mã hóa + one-hot | 0.95028 |
| 2 | Stack logistic của 3 mô hình trên | 0.95077 |
| 3 | LightGBM khởi động từ logit của GLM (`init_score`) | 0.95069 |
| 3 | Stack 4 mô hình | 0.95085 |
| 4 | Huấn luyện lại trên toàn fold + đếm tần suất + khóa kết hợp (3 mô hình mới) | 0.95103 / 0.95077 / 0.95060 |
| 4 | Stack 7 mô hình của riêng tác giả | **0.95111** |

Ba nhận xét nhanh:
- Từ bước 1 sang bước 2, mọi mô hình đều tăng ở cả 5 fold, và logistic tăng nhiều nhất (+0.007). Phần phi tuyến đã được mã hóa làm hộ, nên mô hình tuyến tính chỉ việc cộng lại.
- Một ý tưởng (mã hóa theo nhãn từng giá trị) cho +0.0026, nhiều hơn mọi thứ trước đó cộng lại (+0.0013). Kỹ thuật đặc trưng đúng đáng giá hơn tinh chỉnh mô hình.
- Sau bước 2, mỗi cải tiến chỉ còn vài phần mười nghìn. Đó là lý do cần quy trình đánh giá chặt (mục 3).

---

## 2. Các kỹ thuật chính (theo thứ tự đóng góp)

### 2.1 Cây rất nhỏ thay vì cây sâu
- Quét số lá cho LightGBM (4, 8, 15, 31) và độ sâu cho XGBoost (2, 3, 4) trên cùng các fold và cùng tập dừng sớm: **4 lá và depth 2 thắng**.
- Lý do: nhãn gần với một công thức logistic (cộng tính theo từng cột), nên cây sâu chủ yếu khớp nhiễu. Một thí nghiệm khác trong báo cáo xác nhận: LightGBM chỉ cho phép mỗi cây kết hợp các khóa của một cột đạt 0.95069, hơn LightGBM thường (0.95038) ở cả 5 fold.
- Trọng số lớp (5×, 20× cho lớp gian lận) **không giúp**. AUC chỉ cần thứ hạng, và đặt lại trọng số chỉ làm méo xác suất.
- Bài học cho báo cáo: vẽ đường cong AUC theo độ phức tạp của cây và giải thích bằng bias–variance.

### 2.2 Mã hóa theo nhãn cho từng giá trị (target encoding), làm đúng cách
Ý tưởng: coi **mỗi giá trị khác nhau của mỗi cột (kể cả cột số) như một hạng mục**, rồi thay nó bằng tỷ lệ gian lận đã làm mượt.

- Công thức: `(tổng_gian_lận + prior × m) / (số_dòng + m)`, với `m = 20`.
- Hiệu quả: `account_balance_usd` dạng số gần như vô dụng (AUC 0.506), nhưng sau khi mã hóa theo từng giá trị đạt 0.536.
- **Cạm bẫy lớn nhất:** mã hóa tính trên toàn bộ tập huấn luyện cho fold 0 AUC giả khoảng 0.98. Phải mã hóa lồng nhau:
  1. Trong mỗi fold ngoài, các dòng huấn luyện được mã hóa bằng cross-fit 5 inner fold (không dòng nào thấy nhãn của chính nó).
  2. Các dòng dừng sớm, fold chấm điểm và test được mã hóa bằng toàn bộ dòng huấn luyện.
  3. Trung bình 3 seed mã hóa để giảm nhiễu từ phép chia ngẫu nhiên bên trong.
- Thí nghiệm rò rỉ có chủ đích này rất hợp để đưa vào báo cáo.

### 2.3 Hồi quy logistic trên log-odds của các mã hóa
- Đầu vào: log-odds của các mã hóa (chuẩn hóa) + one-hot cho hạng mục + log của các cột số.
- Một mình nó đạt 0.95028, ngang các cây, trong khi logistic với spline cho cột số chỉ đạt 0.94338.
- Giá trị lớn nhất của nó là trong ensemble: sai số của mô hình tuyến tính khác sai số của cây.

### 2.4 Cây khởi động từ GLM (boosting from a GLM)
- Vì nhãn gần logistic, để GLM học phần cộng tính trước, rồi cây 4 lá chỉ khớp phần GLM còn sai. Trong LightGBM, truyền logit của GLM vào `init_score`.
- GLM chỉ fit trên các dòng huấn luyện của fold, nên các dòng dừng sớm, fold chấm điểm và test nhận logit theo cùng cách.
- Kết quả: 0.95069, hơn LightGBM thường (0.95038) ở cả 5 fold.
- Tác giả cũng thử GLM mạnh hơn (spline) rồi boost từ nó: không giúp thêm, vì GLM mạnh hơn thì để lại ít việc cho cây.

### 2.5 Huấn luyện lại trên toàn bộ fold (+0.0003)
- Cách cũ: dành 10% dòng huấn luyện để dừng sớm, nên mô hình và mã hóa chỉ học từ 72% dữ liệu thay vì 80%.
- Cách mới: 10% đó chỉ dùng để chọn số cây, sau đó **fit lại mã hóa và mô hình trên toàn bộ fold huấn luyện** với số cây nhân 1/0.9.
- Bài học tổng quát: lượng dữ liệu mô hình thực sự thấy ảnh hưởng đáng kể. Khoảng cách với các giải pháp công khai đến từ cách huấn luyện, không phải từ kiến trúc.
- Lưu ý: ba thay đổi của bước này (refit, đếm tần suất, khóa kết hợp) được đo gộp, nên không tách riêng được đóng góp của từng thay đổi.

### 2.6 Đặc trưng đếm tần suất và khóa kết hợp
- Mỗi mô hình cây được thêm số dòng (train + test) có cùng giá trị khóa. Nhờ đó cây tin một mã hóa từ 30,000 dòng hơn một mã hóa từ 3 dòng: LightGBM +0.00021 (5/5 fold).
- GLM được thêm 8 khóa kết hợp cho những ảnh hưởng không cộng tính (ví dụ `merchant_category × merchant_risk_score`, cờ ngoại quốc × lệch IP, hạng mục × thập phân vị số tiền) và log của số đếm: +0.00050 (4/5 fold).
- Ngược lại, thêm mã hóa cho **tất cả các cặp** (59 cặp, hoặc 45 cặp ở một notebook khác) không giúp: tổ hợp hiếm chỉ là nhiễu. Bài học: chọn khóa kết hợp có lý do, đừng sinh hàng loạt.

### 2.7 Khóa chữ số (đặc thù bộ dữ liệu, dùng như một thí nghiệm ablation)
Với 6 cột số thực, cắt mỗi số thành các phần (phần nguyên, phần nguyên bỏ chữ số cuối, cắt một chữ số thập phân, phần thập phân, độ dài + chữ số đầu) rồi mã hóa từng phần như một giá trị. Ở notebook tham chiếu khác (Part 2), chỉ mã hóa giá trị cho 0.94875, thêm khóa chữ số lên 0.95029.

Vì bạn không muốn dự án xoay quanh các mẹo kiểu này, nên đặt nó thành **một nhóm đặc trưng trong ablation**, báo cáo mức đóng góp và nói rõ đây là cấu trúc riêng của bộ dữ liệu tổng hợp. Mức đóng góp lớn nhất đến từ ý tưởng tổng quát (mã hóa từng giá trị), còn các khóa chữ số là phần mở rộng hẹp hơn. Cờ "số tròn" tác giả thử đầu tiên (+0.00044) thực chất là vài trường hợp đặc biệt của cùng ý tưởng này.

---

## 3. Quy trình đánh giá: phần đáng học nhất

### 3.1 Cùng một bộ fold cho mọi mô hình
Mọi OOF và dự đoán test đều lưu trên cùng 5 fold để xếp chồng không rò rỉ.

### 3.2 Dừng sớm không nhìn fold chấm điểm
Dừng sớm trên 10% dòng huấn luyện của chính fold đó, không dừng trên fold chấm điểm. Phiên bản nghiêm ngặt của tác giả cho CV thấp hơn khoảng 0.00007 so với cách dừng trên fold chấm điểm của notebook công khai, đúng như dự đoán.

### 3.3 Stack đánh giá lồng nhau
- Stack: hồi quy logistic (C = 1.0) trên log-odds đã chuẩn hóa của các mô hình, giới hạn xác suất trước khi lấy logit.
- Đánh giá bằng cross-validation trên một cách chia thứ hai (seed 7), để bộ stack không bao giờ chấm điểm các dòng nó đã học.
- Tác giả đo thêm trên cách chia thứ ba (seed 8) để kiểm tra độ ổn định.

### 3.4 Luật chấp nhận một thay đổi, và cách kiểm chứng luật đó
- Luật ban đầu: giữ một thay đổi nếu tăng ít nhất 0.00002 ở ít nhất 4/5 fold.
- **Kiểm thử bằng mẫu rỗng (null test):** thêm 40 "thành viên" không mang thông tin mới (một mô hình có sẵn cộng nhiễu Gauss, tương quan 0.993) vào stack, mỗi lần một cái. Với một cách chia meta, **42% số thành viên nhiễu vẫn vượt luật**, và mức tăng giả lớn nhất là +0.000049. Với hai cách chia thì không cái nào qua.
- Luật mới: tăng ít nhất 0.0001 trên cả hai cách chia meta. Nhưng lợi ích trên hai cách chia tương quan 86%, nên cách chia thứ hai không hoàn toàn độc lập.
- Bài học: một ngưỡng chấp nhận phải nằm trên mức tăng lớn nhất mà nhiễu thuần có thể tạo ra. Rất hợp để đưa vào báo cáo như một thí nghiệm kiểm chứng phương pháp.
- Ngược lại, ngưỡng chỉ là bộ lọc, không phải phép đo. Một mô hình tăng +0.00006, dưới ngưỡng, vẫn có thể là thật. Khi đó đo trực tiếp nhiễu bằng bootstrap ghép cặp (trong báo cáo: tốt hơn trong 96% lần lấy mẫu) thay vì chỉ nhìn ngưỡng.

### 3.5 Nhiễu của bảng xếp hạng công khai
- Tập công khai chỉ có khoảng 1,000 vụ gian lận, nên AUC của một file dao động khoảng ±0.003. Chênh lệch giữa hai file trên cùng các dòng nhỏ hơn nhiều (±0.0003).
- Mô phỏng 50 lần thử chọn theo điểm công khai cho kết quả: **+0.00027 trên phần công khai, −0.00010 trên phần riêng.** Nghĩa là leo điểm công khai bằng cách nộp nhiều lần chỉ là khớp nhiễu.
- Trong báo cáo, CV và điểm công khai đi cùng nhau ở mọi bước của các mô hình tự xây (chênh lệch ≤ 0.00007), nên **quyết định bằng CV**.

---

## 4. Những gì không giúp được (tiết kiệm thời gian thử)

| Ý tưởng | Kết quả | Ghi chú |
|---|---|---|
| Cây sâu hơn (8–31 lá, depth 3–4) | thấp hơn 4 lá / depth 2 | nhãn gần cộng tính |
| Trọng số lớp 5× / 20× | không giúp | AUC chỉ cần thứ hạng |
| Tỷ lệ và giá trị so với trung bình theo `merchant_category` | −0.00038, 0/5 fold | cây đã tự kết hợp được |
| Đếm tần suất trên cột thô (không có mã hóa) | +0.00008, 2/5 fold | tỷ lệ gian lận theo giá trị quan trọng hơn mức độ phổ biến |
| Mã hóa với `m` = 5 hoặc 100 làm thành viên thêm | +0.00005 / +0.00006, 2/5 | gần như cùng một mô hình |
| Mã hóa theo các cặp cột | −0.0006 khi đứng riêng, +0.00005 trong stack | tổ hợp hiếm là nhiễu |
| Mạng nơ-ron (RealMLP) trên các mã hóa | −0.00003 trong stack | cùng thông tin, cùng lỗi (công thức CPU ngắn, 8 epoch, 1 seed) |
| GLM với solver newton-cholesky | −0.00017 so với lbfgs | lbfgs đã hội tụ |
| CatBoost trong stack | −0.00001 đến −0.00007 | yếu hơn và tương quan cao với các mô hình khác, chỉ thêm nhiễu cho mô hình meta |
| GLM spline + boost từ nó | +0.00001–0.00002, 3/5 | GLM mạnh hơn để lại ít việc cho cây |
| Mô hình cộng tính (một cột mỗi cây) trong stack | +0.00002, 2/5 | GLM boost đã là mô hình cộng tính |
| Mạng transformer mức ký tự (một fold, T4) | 0.94461 một mình, không giúp stack | mô hình nhỏ huấn luyện từ đầu chưa đủ mạnh |

Hai cảnh báo thêm từ báo cáo:
- Một ý tưởng hiệu quả ở cuộc thi trước có thể có hại ở cuộc thi này (ví dụ dùng nhãn của dữ liệu gốc làm đặc trưng, ở đây −0.0006). Không áp dụng công thức cũ mà không kiểm chứng. Phần này nằm ngoài phạm vi dự án của bạn vì dùng dữ liệu ngoài.
- Một thành viên đứng riêng tốt chưa chắc giúp ensemble. Mạng nơ-ron và các biến thể mã hóa không thêm thông tin mới.

---

## 5. Bài học tổng quát cho báo cáo môn học

1. **Hiểu cách dữ liệu được sinh ra rồi mã hóa đúng điều đó** cho lợi ích lớn hơn việc đổi mô hình.
2. **Một đặc trưng thủ công hiệu quả thường là một trường hợp đặc biệt của một ý tưởng tổng quát hơn.** Khi một đặc trưng làm tay có tác dụng, hãy tìm phiên bản tổng quát.
3. **Một mô hình chỉ có ích nếu nó thêm thông tin mới.** CatBoost làm giảm mọi stack nó tham gia; các biến thể mã hóa không thêm gì.
4. **Mã hóa theo nhãn cần lồng nhau nghiêm ngặt**, nếu không CV vô nghĩa.
5. **Mô hình khác loại quan trọng hơn nhiều biến thể giống nhau.** Logistic yếu hơn cây khi đứng riêng nhưng thường nhận trọng số lớn trong stack.
6. **Lượng dữ liệu mô hình nhìn thấy quan trọng:** chọn số cây trên phần giữ lại rồi huấn luyện lại trên toàn bộ.
7. **Kiểm thử ngưỡng chấp nhận bằng mẫu rỗng** để biết nhiễu tạo ra mức tăng giả lớn đến đâu.
8. **Trung bình hai stack xây độc lập** (khác tập thành viên, khác seed) không đổi CV nhưng giảm phương sai phía test, vì một stack có hàng trăm trọng số dương và âm khuếch đại khác biệt giữa các lần refit. Chênh lệch giữa hai stack trên test lớn hơn rất nhiều so với trên dữ liệu huấn luyện (khoảng 78–91 × 10⁻⁴ so với 21 × 10⁻⁴, tính bằng 1 trừ tương quan hạng). Đây là kết quả của tác giả khác trên stack lớn; với stack nhỏ của bạn, hiệu ứng có thể yếu hơn, nên cần tự đo.
9. **Tính bão hòa phải đo so với mọi lựa chọn có thể, không chỉ so với những gì đã gom được.** Bài học này áp dụng cho ensemble của riêng bạn: đừng kết luận "đã bão hòa" chỉ vì thêm vài biến thể cùng họ không giúp.
10. **Đo thông lượng trước khi lập kế hoạch tính toán.** Ước lượng GPU của tác giả lệch gấp đôi; 15 phút đọc log là đủ để biết.

---

## 6. Đề xuất áp dụng vào dự án của bạn

### 6.1 Lộ trình tái hiện (đối chiếu số tham chiếu ở mục 1)
1. Khung CV với fold cố định và kiểm tra mô hình ngẫu nhiên (AUC ≈ 0.50).
2. Baseline: LightGBM 15 lá, rồi LightGBM 4 lá, XGBoost depth 2, logistic spline. Kiểm tra xem 4 lá có thắng như báo cáo không.
3. Mã hóa theo từng giá trị, lồng nhau, kèm thí nghiệm rò rỉ có chủ đích (mục 2.2).
4. Logistic trên log-odds của mã hóa.
5. LightGBM khởi động từ GLM.
6. Huấn luyện lại trên toàn fold, đếm tần suất, vài khóa kết hợp có lý do.
7. Stack lồng nhau và null test cho ngưỡng chấp nhận.
8. (Tùy chọn) Khóa chữ số như một nhóm ablation riêng.

Nếu mô hình nào của bạn lệch hơn khoảng 0.0003–0.0005 so với số tham chiếu, kiểm tra theo thứ tự: cách mã hóa và làm mượt, cách chọn số cây, seed, phiên bản thư viện.

### 6.2 Các hướng thử thêm (giả thuyết của tôi, báo cáo không đo)
Chỉ là gợi ý để khám phá, chưa có bằng chứng từ báo cáo:
- Trung bình nhiều seed cho các thành viên cây tốt nhất (báo cáo dùng 3 seed mã hóa và một số mô hình 3 seed, chưa đo riêng).
- Hai stack xây độc lập (khác tập khóa, khác seed) rồi lấy trung bình hạng, đo trực tiếp bằng bootstrap.
- Mạng nơ-ron học embedding cho từng giá trị trực tiếp (thử RealMLP của tác giả chỉ dùng log-odds đã mã hóa, nên chưa loại trừ hướng này).
- Phân tích lỗi: những vụ gian lận bị xếp thấp có điểm chung gì, để tìm đặc trưng còn thiếu.
- Quét `C` và tinh chỉnh ràng buộc dấu của trọng số stack nếu thấy trọng số âm.

---

## 7. Phần đã loại khỏi tài liệu này, và lý do

Các phần sau thuộc phạm vi ngoài dự án của bạn nên không được tổng hợp: dùng dữ liệu gốc làm nguồn thêm, dùng các mô hình huấn luyện sẵn, dùng dự đoán công khai của người khác để gộp thành pool, và các file xếp hạng công khai. Phần lớn điểm tăng sau CV 0.9511 của báo cáo (đến 0.9526) đến từ chính những nguồn đó, và báo cáo cũng tự nói rằng mô hình của tác giả không thêm gì vào pool này. Vì vậy trần thực tế cho dự án thuần ML của bạn là khoảng **0.951**.

## 8. Điều chưa được kiểm chứng

- Các con số CV là của tác giả, chưa được chạy lại trong dự án của bạn.
- Ba thay đổi ở mục 2.5–2.6 được đo gộp, không tách riêng được từng đóng góp.
- Hiệu ứng của việc trung bình hai stack (mục 5, điểm 8) đo trên các stack rất lớn với nhiều nguồn khác nhau, không chắc áp dụng cho stack nhỏ của bạn.
- Kết luận "mạng nơ-ron trên các mã hóa không giúp" dựa trên một công thức CPU ngắn (8 epoch, 1 seed), chưa phải kết luận chắc chắn.
