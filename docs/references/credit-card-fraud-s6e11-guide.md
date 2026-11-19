# Hướng Dẫn Chuyên Sâu Tối Ưu Mô Hình Phát Hiện Gian Lận Thẻ Tín Dụng (Kaggle S6E11) — Phương Pháp Tự Huấn Luyện (Self-Trained Pipeline)

Bản hướng dẫn này chắt lọc toàn bộ kiến thức, phân tích EDA, kỹ thuật tạo đặc trưng (Feature Engineering), kiến trúc mô hình, chiến lược kiểm chứng (Cross-Validation) và phương pháp kết hợp mô hình (Ensembling/Stacking) từ các nghiên cứu chuyên sâu trong cuộc thi Kaggle Playground Series S6E11. 

Toàn bộ quy trình được xây dựng tuân thủ nghiêm ngặt phạm vi dự án độc lập: **Chỉ sử dụng mô hình tự huấn luyện trên dữ liệu cuộc thi, không sử dụng dữ liệu bên ngoài, không sử dụng mô hình huấn luyện sẵn (pre-trained models), và không sử dụng bất kỳ file dự đoán công khai nào từ các đối thủ khác**.

---

## 1. Khung Đánh Giá CV & Phòng Chống Rò Rỉ Dữ Liệu (Validation & Anti-Leakage Framework)

Trong các bài toán dữ liệu tổng hợp (synthetic data) có tỷ lệ mất cân bằng lớp cao, việc xây dựng một hệ thống kiểm chứng (Cross-Validation) sạch, không rò rỉ là yếu tố tiên quyết quyết định sự thành bại trên tập Private Leaderboard.

### 1.1 Thiết lập Phân Chia Stratified 5-Fold CV

Tập dữ liệu huấn luyện bao gồm $699,960$ dòng giao dịch với tỷ lệ gian lận nền (base fraud rate) là **$1.67\%$** ($11,689$ ca gian lận). Tập kiểm thử (Test set) bao gồm $299,984$ dòng.

*   **Quy tắc Stratified 5-Fold**: Sử dụng `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` cố định cho toàn bộ các mô hình trong hệ thống. Stratification đảm bảo tỷ lệ nhãn gian lận $1.67\%$ được giữ nguyên vẹn trên tất cả $5$ folds, tránh hiện tượng lệch phân bố nhãn giữa các tập validation.
*   **Điểm Out-Of-Fold (OOF)**: Mọi mô hình đơn lẻ phải tạo ra dự đoán OOF trên đúng $5$ fold này. Điểm ROC AUC tính trên toàn bộ tập OOF là thước đo khách quan duy nhất để đánh giá cải tiến.

### 1.2 Kỹ Thuật Mã Hóa Mục Tiêu Lồng Ghép (Nested / In-Fold Target Encoding)

Các biến số thực trong bài toán này chứa nhiều giá trị trùng lặp (ví dụ `account_balance_usd` có $43,317$ giá trị duy nhất nhưng xuất hiện lặp lại hàng trăm nghìn lần). Mỗi giá trị mang một tỷ lệ gian lận riêng. Tuy nhiên, việc mã hóa mục tiêu (Target Encoding) nếu làm sai quy trình sẽ gây ra hiện tượng rò rỉ dữ liệu cực kỳ nghiêm trọng.

#### Công thức Smoothed Target Encoding (m-estimate)
Với mỗi giá trị $v$ của một biến, tỷ lệ gian lận làm mượt được tính theo công thức:

$$\text{TE}(v) = \frac{\text{Số ca gian lận có giá trị } v + m \cdot \text{Tỷ lệ gian lận tổng thể}}{\text{Tổng số dòng có giá trị } v + m}$$

Trong đó, trọng số làm mượt $m$ (m-estimate) thường đặt ở mức $m = 20.0$ (cho các mô hình cây) hoặc $m = 50.0$ (cho các mô hình GLM tuyến tính) để kéo các giá trị hiếm (rare values) về mức trung bình nền, tránh nhiễu.

#### Nguy cơ Rò Rỉ Dữ Liệu (Leaky TE vs. In-Fold TE)
*   **Mã hóa rò rỉ (Leaky TE)**: Nếu tính Target Encoding trên toàn bộ tập dữ liệu huấn luyện trước khi chia fold, điểm CV AUC sẽ vọt lên mức giả tạo là **$0.9828$**, nhưng khi đánh giá độc lập trên tập test chỉ đạt **$0.8675$**. Lý do là mỗi dòng validation đã tự "nhìn thấy" nhãn của chính nó trong chỉ số TE.
*   **Mã hóa lồng ghép chuẩn (Nested / In-Fold TE)**: 
    *   Trong mỗi fold thứ $k$ (chiếm $80\%$ dữ liệu train), thực hiện một vòng chia $5$-fold nội bộ (inner $5$-fold split).
    *   Mỗi dòng huấn luyện nội bộ được mã hóa bằng tỷ lệ tính từ $4/5$ phần còn lại của inner fold.
    *   Tập Validation ($20\%$) và tập Test được mã hóa bằng tỷ lệ tính trên toàn bộ $100\%$ dữ liệu huấn luyện của fold $k$.
    *   **Kiểm chứng sạch**: Khi áp dụng In-Fold TE, điểm CV AUC ($0.9492$) và điểm trên tập kiểm thử độc lập ($0.9480$) hoàn toàn trùng khớp. Bài kiểm thử đảo nhãn ngẫu nhiên (label-shuffle test) thu được AUC = **$0.4868$** (tiệm cận mức nhiễu $0.5$), khẳng định pipeline hoàn toàn sạch rò rỉ.

### 1.3 Cơ Chế Dừng Sớm (Early Stopping) Không Rò Rỉ

Việc theo dõi trực tiếp điểm số trên tập Validation của fold đang đánh giá để dừng sớm (early stopping) sẽ khiến mô hình bị overfit nhẹ vào tập validation đó.

*   **Giải pháp**: Trong mỗi fold huấn luyện ($80\%$ dữ liệu), trích ra $10\%$ dữ liệu làm tập Inner Validation dành riêng cho việc early stopping (`eval_set`). 
*   Mô hình được huấn luyện trên $72\%$ dữ liệu còn lại và dừng lại khi điểm AUC trên tập $10\%$ inner validation không tăng sau $300$ vòng (`early_stopping_rounds=300`).
*   Tập Validation chuẩn của fold ($20\%$) chỉ được dùng để chấm điểm OOF sau khi mô hình đã hoàn tất huấn luyện.

### 1.4 Seed Bagging Cho Target Encoder

Do việc chia inner fold trong Target Encoding mang tính ngẫu nhiên, các giá trị xuất hiện ít dòng sẽ nhận giá trị mã hóa hơi khác nhau giữa các seed.

*   **Thực thi**: Chạy lại cùng một cấu hình mô hình với $3 \dots 5$ seed ngẫu nhiên khác nhau (thay đổi inner fold split của Target Encoder và subsampling của cây).
*   **Kết quả**: Lấy trung bình cộng thứ hạng (rank average) giữa các seed giúp làm mượt nhiễu mã hóa, nâng điểm CV AUC của mô hình đơn lẻ từ **$0.95041$ lên $0.95069$**.

### 1.5 Chiến Thuật Refit Trên 100% Dữ Liệu Train Cho Tập Test

Các mô hình huấn luyện trên $5$ fold chỉ nhìn thấy $80\%$ dữ liệu (thực tế là $72\%$ khi dùng inner early stopping), dẫn đến các bảng mã hóa Target Encoding bị thiếu thông tin từ $20\%$ dữ liệu còn lại.

*   **Quy trình Refit**:
    1. Tính toán số lượng cây tối ưu $N_{\text{trees}}$ bằng trung bình số cây dừng sớm trên $5$ fold nhân với hệ số $1.25$ ($25\%$ dữ liệu nhiều hơn):
       $$N_{\text{trees\_refit}} = \text{round}\left(1.25 \times \frac{1}{5} \sum_{k=1}^5 \text{best\_iteration}_k\right)$$
    2. Mã hóa Target Encoding trên toàn bộ $100\%$ tập huấn luyện ($699,960$ dòng).
    3. Huấn luyện $1$ mô hình duy nhất trên $100\%$ tập train với đúng $N_{\text{trees\_refit}}$ cây, không sử dụng early stopping.
    4. Dùng mô hình refit này để dự đoán cho tập Test.
*   **Hiệu quả**: Kỹ thuật này nâng cao độ chính xác của bảng mã hóa Target Encoding trên tập Test (đặc biệt đối với các giá trị có tần suất xuất hiện thấp như `account_balance_usd`), giúp cải thiện điểm số thực tế.

---

## 2. Phân Tích EDA & "Dấu Vết" Dữ Liệu Tổng Hợp (Synthetic Data Fingerprints)

Tập dữ liệu Kaggle S6E11 được tạo ra bởi một thuật toán sinh dữ liệu tổng hợp (synthetic generator). Việc hiểu rõ các "dấu vết" (fingerprints) của bộ sinh dữ liệu này quan trọng hơn rất nhiều so với việc tinh chỉnh tham số (hyperparameter tuning) mù quáng.

### 2.1 Đặc Điểm Phân Bố & Các Biến Đơn Lẻ Sắc Báo

Tỷ lệ gian lận chung là $1.67\%$. Khi phân tích đơn biến, các đặc trưng chia thành hai nhóm rõ rệt:

#### Nhóm biến đơn lẻ mang tín hiệu cực mạnh:
*   **`cvv_retry_count` (Số lần thử lại CVV)**: Biến đơn lẻ mạnh nhất.
    *   $0$ lần thử: Tỷ lệ gian lận **$0.93\%$** ($642,088$ dòng).
    *   $1$ lần thử: Tỷ lệ gian lận **$8.75\%$** ($55,942$ dòng).
    *   $\ge 2$ lần thử: Tỷ lệ gian lận vọt lên **$41.73\%$** ($1,905$ dòng) — gấp $25$ lần tỷ lệ nền.
*   **`merchant_risk_score` (Điểm rủi ro thương nhân)**: Tỷ lệ gian lận tăng dốc từ **$0.16\%$** ở decile thấp nhất lên **$5.5\%$** ở decile cao nhất.
*   **`velocity_score` (Điểm tốc độ giao dịch)**: Tỷ lệ gian lận tăng dốc từ **$0.12\%$** ở decile thấp nhất lên **$5.3\%$** ở decile cao nhất.
*   **`txn_count_last_24h`**: Tăng từ **$0.23\%$** ($0$ giao dịch) lên **$7.0\%$** ($\ge 9$ giao dịch).
*   **`customer_age`**: Khách hàng lớn tuổi bị nhắm tới nhiều hơn ($18-25$ tuổi: $0.4\%$; $69-76$ tuổi: $2.9\%$).
*   **`auth_method` (Phương thức xác thực)**: Nhóm `Biometric` có tỷ lệ gian lận chỉ **$0.5\%$**, nhóm `PIN` $0.87\%$, `OTP` $1.41\%$, `3D Secure` $1.48\%$, trong khi nhóm `No Authentication` lên tới **$6.44\%$**.
*   **`merchant_category`**: Các ngành rủi ro cao gồm `Gift Cards` ($7.30\%$), `Crypto Exchange` ($6.53\%$), `Gaming` ($5.51\%$), so với `Groceries` ($1.11\%$) và `Online Retail` ($1.11\%$).

#### Nhóm biến đơn lẻ có phân bố "phẳng" (Flat Features):
*   Các cột: `time_of_day_hour`, `day_of_week`, `distance_from_home_km`, `card_age_months`, `account_balance_usd`, `amount_usd`, `card_type`, `channel`, `device_type`.
*   Tỷ lệ gian lận khi xét đơn lẻ trên các cột này hầu như nằm ngang xung quanh mức $1.67\%$. 
*   **Bài học**: Tuyệt đối không xóa các cột này! Chúng chứa tín hiệu ngầm dưới dạng tương tác đa biến và tần suất lặp giá trị.

### 2.2 Dấu Vết Chữ Số Thập Phân (Decimal Fingerprints)

Bộ sinh dữ liệu viết các số thực theo từng chữ số. Chữ số thập phân đầu tiên sau dấu phẩy và hai chữ số phần xu (`cents`) của các cột `amount_usd`, `hours_since_last_txn`, `distance_from_home_km`, `account_balance_usd`, `velocity_score`, `merchant_risk_score` vướng chặt vào nhãn gian lận.

*   Các phần thập phân `.0` và `.4` chạy "hot" (tỷ lệ gian lận cao hơn rõ rệt, khoảng $1.9\%$).
*   Các phần thập phân `.3` và `.6` chạy "cold" (tỷ lệ gian lận thấp hơn rõ rệt, khoảng $1.3\%-1.4\%$).
*   Ví dụ: Đối với `hours_since_last_txn`, phần thập phân đuôi `.03` qua thử nghiệm kiểm chứng độc lập (Discovery/Confirmation split) cho tỷ lệ gian lận $2.86\%$ so với $1.65\%$ ở các đuôi khác (Odds Ratio = $1.73$).

### 2.3 Sự Trùng Lặp Giá Trị Tuyệt Đối (Per-Value Repetition)

Bộ sinh dữ liệu tái sử dụng các giá trị số thực chính xác rất nhiều lần:
*   `account_balance_usd`: Có $43,317$ giá trị duy nhất trên $699,960$ dòng (trung vị $3$ dòng/giá trị).
*   `distance_from_home_km`: Chỉ có $4,365$ giá trị duy nhất (trung vị $66$ dòng/giá trị).
*   `velocity_score`: Chỉ có $666$ giá trị duy nhất (trung vị $690$ dòng/giá trị).

Mỗi giá trị trùng lặp mang một tỷ lệ gian lận riêng biệt. Ví dụ: Giá trị balance `511.73` xuất hiện $974$ lần có tỷ lệ gian lận $1.03\%$, trong khi `681.12` xuất hiện $690$ lần có tỷ lệ gian lận $2.46\%$. Cây quyết định thông thường dựa trên vết cắt khoảng (`amount > 100`) hoàn toàn không thể bắt được tín hiệu này nếu không có Target Encoding.

### 2.4 Tương Tác Dạng Nhân (Multiplicative & Compounding Interactions)

Các tín hiệu gian lận mạnh kết hợp với nhau theo **dạng nhân (multiplicative)** trên tỷ lệ rủi ro (tương đương dạng cộng tính trên thang log-odds):

1.  **Merchant Risk Quintile $\times$ Velocity Quintile**:
    *   Rủi ro rất thấp + Tốc độ rất thấp: Tỷ lệ gian lận **$0.03\%$** ($1/3,000$).
    *   Rủi ro rất cao + Tốc độ rất cao: Tỷ lệ gian lận **$9.10\%$** (gấp $>250$ lần).
2.  **Auth Method $\times$ CVV Retries**:
    *   `Biometric` + $0$ CVV retry: **$0.3\%$** gian lận.
    *   `No Authentication` + $\ge 2$ CVV retries: Vọt lên **$72.6\%$** gian lận.
3.  **Merchant Category $\times$ Auth Method**:
    *   `Gift Cards` + `No Authentication`: Tỷ lệ gian lận **$23.0\%$**.
    *   `Gaming` + `No Authentication`: Tỷ lệ gian lận **$19.1\%$**.
4.  **Cộng dồn 6 cờ cảnh báo Boolean** (`is_foreign_transaction`, `is_new_merchant`, `used_vpn`, `ip_country_mismatch`, `billing_shipping_mismatch`, `is_ai_generated_scam_attempt`):
    *   $0$ cờ: $0.71\%$ gian lận.
    *   $1$ cờ: $3.61\%$ gian lận.
    *   $2$ cờ: $18.18\%$ gian lận.
    *   $3$ cờ: $51.89\%$ gian lận.
    *   $4$ cờ: $83.93\%$ gian lận.
    *   $5$ cờ: $100.0\%$ gian lận.

#### ⚠️ Điểm mù của các cờ cảnh báo (No-Flag Fraud Blind Spot)
Có tới **$33.1\%$ tổng số vụ gian lận** ($3,870$ ca) xảy ra ở các giao dịch **không bật bất kỳ cờ cảnh báo Boolean nào** ($0$ cờ). Các ca gian lận ẩn này tập trung chủ yếu vào nhóm giao dịch `No Authentication` kết hợp với số lần thử CVV $\ge 1$.

---

## 3. Kỹ Thuật Tạo Đặc Trưng Nâng Cao (Feature Engineering Architecture)

Dựa trên các dấu vết của dữ liệu tổng hợp, quy trình tạo đặc trưng tập trung vào $3$ nhóm chính mang lại hiệu quả cao nhất:

### 3.1 Nhóm 1: Per-Value Target Encoding (Mã Hóa Mục Tiêu Theo Giá Trị Thô)
Coi tất cả 24 cột (bao gồm 13 cột số, 5 cột phân loại, 6 cột cờ Boolean) là các biến danh mục (categorical). Mã hóa từng giá trị duy nhất bằng công thức Smoothed Target Encoding In-Fold với $m=20.0$.

*   **Tác động**: Rút ra tín hiệu mạnh từ các biến số thực phẳng. Ví dụ: cột `account_balance_usd` trước khi mã hóa có AUC đơn lẻ là $0.5063$, sau khi mã hóa Target Encoding vọt lên **$0.5358$**.

### 3.2 Nhóm 2: Khóa Chữ Số (Digit Keys) — Cải Tiến Tăng Điểm Mạnh Nhất (+0.00147 AUC)
Cắt $6$ cột số thực có phần thập phân (`amount_usd`, `hours_since_last_txn`, `distance_from_home_km`, `account_balance_usd`, `velocity_score`, `merchant_risk_score`) thành **$6$ loại khóa chữ số (Digit Keys)** không chứa nhãn (label-free keys):

Với một số thực ví dụ $x = 1371.37$:
1.  **`cents`**: $2$ chữ số thập phân cuối $\rightarrow$ `37` (giữ thêm dạng số thực `37.0`).
2.  **`int`**: Phần nguyên của số $\rightarrow$ `1371`.
3.  **`int10`**: Phần nguyên bỏ chữ số hàng đơn vị $\rightarrow$ `137`.
4.  **`tenths`**: Số cắt lấy $1$ chữ số thập phân $\rightarrow$ `13713`.
5.  **`lead1`**: Độ dài phần nguyên + chữ số đầu tiên $\rightarrow$ `4|1`.
6.  **`lead2`**: Độ dài phần nguyên + $2$ chữ số đầu tiên $\rightarrow$ `4|13`.

👉 **Tạo ra $36$ khóa chữ số mới** ($6$ cột $\times$ $6$ khóa). Thực hiện Target Encoding In-Fold cho toàn bộ $36$ khóa này.

```python
# Minh họa thuật toán trích xuất Digit Keys trong Python
import numpy as np
import pandas as pd

def extract_digit_keys(df, float_cols):
    keys = {}
    for col in float_cols:
        # Loại bỏ sai số số thực và chuyển về số nguyên cents (1371.37 -> 137137)
        units = np.rint(df[col].to_numpy() * 100).astype(np.int64)
        integer = units // 100
        s = pd.Series(integer).astype(str)
        
        keys[f"{col}__int"] = integer
        keys[f"{col}__int10"] = units // 1000
        keys[f"{col}__tenths"] = units // 10
        keys[f"{col}__cents"] = units % 100
        keys[f"{col}__lead1"] = (s.str.len() * 10 + s.str.str[0].astype(int)).to_numpy()
        keys[f"{col}__lead2"] = (s.str.len() * 100 + s.str.str[:2].astype(int)).to_numpy()
    return pd.DataFrame(keys, index=df.index)
```

### 3.3 Nhóm 3: Các Cặp Khóa Tương Tác (13 Pair Keys Cho Mô Hình GLM)
Do mô hình Logistic Regression (GLM) là mô hình cộng tính tuyến tính, nó không thể tự phát hiện các tương tác giữa hai biến. Tác giả đã ghép cặp **$13$ cặp đặc trưng rủi ro cao nhất** để đưa vào GLM:

*   `j_category_risk`: `merchant_category` $\times$ `merchant_risk_score`
*   `j_foreign_ip`: `is_foreign_transaction` $\times$ `ip_country_mismatch`
*   `j_category_amount`: `merchant_category` $\times$ `decile(amount_usd)`
*   `j_txn_velocity`: `txn_count_last_24h` $\times$ `decile(velocity_score)`
*   `j_category_age`: `merchant_category` $\times$ `customer_age`
*   `j_amount_risk`: `decile(amount_usd)` $\times$ `decile(merchant_risk_score)`
*   `j_hours_velocity`: `decile(hours_since_last_txn)` $\times$ `decile(velocity_score)`
*   `j_age_risk`: `customer_age` $\times$ `decile(merchant_risk_score)`
*   `j_newmerchant_weekday`: `is_new_merchant` $\times$ `day_of_week`
*   `j_scam_velocity`: `is_ai_generated_scam_attempt` $\times$ `decile(velocity_score)`
*   `j_age_cardage`: `customer_age` $\times$ `card_age_months`
*   `j_hour_weekday`: `time_of_day_hour` $\times$ `day_of_week`
*   `j_txn_age`: `txn_count_last_24h` $\times$ `customer_age`

Mỗi cặp tương tác được biến đổi thành khóa danh mục và thực hiện Target Encoding In-Fold làm mượt với $m=50.0$ dành riêng cho GLM.

---

### 3.4 Tổng Kết Kỹ Thuật Tạo Đặc Trưng: HIỆU QUẢ vs KHÔNG HIỆU QUẢ

| Kỹ thuật Đặc trưng | Kết quả CV AUC | Đánh giá & Nguyên nhân |
| :--- | :---: | :--- |
| **Digit Keys + In-Fold TE** | **+0.00147** | **HIỆU QUẢ CỰC CAO**: Bắt chính xác dấu vết sinh số của generator. |
| **Per-Value TE (24 cột thô)** | **+0.00113** | **HIỆU QUẢ CỰC CAO**: Khai phá tín hiệu gian lận trùng lặp từng giá trị. |
| **Cents dạng số thực** | **+0.00117** | **HIỆU QUẢ**: Đưa 2 chữ số thập phân vào mô hình dạng số liên tục. |
| **13 Pair Keys (cho GLM)** | **+0.00035** | **HIỆU QUẢ**: Giúp GLM học các tương tác phi tuyến tính mạnh nhất. |
| **Biến đếm cờ (`flag_count`)** | **-0.00001** | **KHÔNG HIỆU QUẢ**: Nhiễu; các mô hình cây tự tách được cờ. |
| **Tạo đặc trưng tỷ lệ (amount/balance)**| **-0.00040** | **KHÔNG HIỆU QUẢ**: Mất tín hiệu gốc của từng giá trị. |
| **Thêm dữ liệu gốc (Original 20k)**| **-0.00028** | **KHÔNG HIỆU QUẢ**: Dữ liệu gốc khác phân bố với dữ liệu synthetic. |
| **Pair Target Encoding cho cây GBDT**| **-0.00060** | **KHÔNG HIỆU QUẢ**: Cây quyết định tự học cặp tương tác, đưa TE cặp vào làm quá khớp. |
| **Tần suất giá trị (`value_counts`)** | **+0.00004** | **KHÔNG HIỆU QUẢ (Nhiễu)**: Tín hiệu tần suất đã được bao hàm trong TE. |

---

## 4. Kiến Trúc Mô Hình Đơn Lẻ Tự Huấn Luyện (Self-Trained Model Ecosystem)

Để tạo ra một tập hợp mô hình phong phú cho bước Stacking cuối cùng, ta huấn luyện từ đầu $5$ họ mô hình hoàn toàn khác biệt trên tập dữ liệu cuộc thi:

### 4.1 Mô Hình Tuyến Tính GLM (Logistic Regression)
Dữ liệu tổng hợp tuân theo công thức gần như cộng tính (logistic formula), nên một mô hình tuyến tính chuẩn chỉnh đạt điểm số rất cao.

*   **Thiết kế đầu vào**:
    *   Logit của các đặc trưng Target Encoding: $\text{logit}(E) = \log\left(\frac{E}{1-E}\right)$, được chuẩn hóa `StandardScaler`.
    *   Biến số thực biến đổi log: $\log(1 + x)$, chuẩn hóa `StandardScaler`.
    *   Biến phân loại được mã hóa One-Hot.
    *   13 cặp khóa tương tác Target Encoded.
*   **Thuật toán & Solver**: Sử dụng `LogisticRegression(C=0.3, solver='newton-cholesky', tol=1e-8, max_iter=200)`.
*   **Điểm quan trọng**: Solver mặc định `lbfgs` dừng sớm sau $34$ vòng chưa hội tụ. Solver `newton-cholesky` hội tụ chính xác sau khoảng $10$ vòng, giúp AUC của GLM tăng thêm **+0.00008**.
*   **Kết quả CV AUC**: **$0.95041 - 0.95055$** (mô hình GLM đơn lẻ mạnh nhất).

### 4.2 Mô Hình Cây Quyết Định Nông (Shallow GBDT Trees)
Sau khi đã nạp các đặc trưng Target Encoding (Per-value và Digit keys), **cây nông chiến thắng hoàn toàn cây sâu**.

*   **Cấu hình LightGBM/XGBoost**:
    *   `num_leaves`: **$3 \dots 4$ lá** (độ sâu depth $2$).
    *   `learning_rate`: $0.02 \dots 0.03$.
    *   `colsample_bytree`: $0.5$ (lấy mẫu $50\%$ số cột để tránh đè các đặc trưng mạnh).
    *   `subsample`: $0.8$.
    *   `min_child_samples`: $100$.
*   **Giải thích thực nghiệm**: Khi chạy cây $31$ lá, điểm CV AUC chỉ đạt $0.94663$. Khi giảm xuống $4$ lá, điểm vọt lên **$0.94771$** (+0.00108) và kết hợp Digit Keys đạt **$0.95031$**. Do dữ liệu mang bản chất cộng tính, các cây sâu ($31-64$ lá) chỉ ép khớp vào nhiễu phi tuyến tính không tồn tại.

### 4.3 Cây Học Sai Số Từ GLM (GLM-Boosted Trees) — Mô Hình Đơn Lẻ Mạnh Nhất ($0.95106$ CV)
Đây là sự kết hợp hoàn hảo giữa mô hình tuyến tính và mô hình cây (ý tưởng của megayak):

1.  **Bước 1**: Huấn luyện mô hình GLM trên logit của các bảng Target Encoding để học toàn bộ phần cộng tính tuyến tính. Xuất ra điểm `logit` dự đoán $z_{\text{GLM}}$.
2.  **Bước 2**: Khởi tạo điểm bắt đầu cho cây LightGBM/XGBoost $4$ lá bằng điểm $z_{\text{GLM}}$:
    *   LightGBM: `init_score = z_GLM`
    *   XGBoost: `base_margin = z_GLM`
3.  **Bước 3**: Cây quyết định nông lúc này không cần lãng phí các vết cắt để học lại phần tuyến tính, mà chỉ tập trung học phần sai số (residual) và các tương tác nhỏ còn thiếu.

```python
# Minh họa khởi tạo GLM-Boosted LightGBM trong Python
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

# 1. Fit GLM trên tập train fold
glm = LogisticRegression(C=0.3, solver='newton-cholesky', tol=1e-8)
glm.fit(X_glm_train, y_train)
z_train = glm.decision_function(X_glm_train)
z_val = glm.decision_function(X_glm_val)

# 2. Fit 4-leaf LightGBM từ GLM logit init_score
d_train = lgb.Dataset(X_tree_train, y_train, init_score=z_train)
d_val = lgb.Dataset(X_tree_val, y_val, init_score=z_val, reference=d_train)

params = {
    'objective': 'binary',
    'num_leaves': 4,
    'learning_rate': 0.02,
    'colsample_bytree': 0.5,
    'subsample': 0.8,
    'metric': 'auc'
}
booster = lgb.train(params, d_train, num_boost_round=10000, valid_sets=[d_val], 
                    callbacks=[lgb.early_stopping(300, verbose=False)])

# 3. Dự đoán raw_score và cộng với z_val
p_val = 1 / (1 + np.exp(-(z_val + booster.predict(X_tree_val, raw_score=True))))
```
*   **Kết quả CV AUC**: Đạt **$0.95106$** (khi bag $3$ seed), là mô hình đơn lẻ tự huấn luyện có điểm cao nhất toàn bộ notebook.

### 4.4 Mạng Nơ-Ron Bảng Tự Huấn Luyện (RealMLP trên GPU)
Để tạo ra sự đa dạng về dạng sai số khi ensembling, ta tự huấn luyện một mạng nơ-ron RealMLP (từ thư viện `pytabkit`) trên GPU:

*   **Đặc trưng đầu vào cho RealMLP**:
    *   Các cột số thực, phần nguyên, và phần xu (`cents`) biến đổi thành danh mục để mạng tự học vector nhúng (embeddings).
    *   Thêm các cột Target Encoding In-Fold dưới dạng biến số thực (+0.0019 CV).
*   **Hyperparameters**: Mạng $3$ lớp ẩn `[512, 256, 128]`, hàm kích hoạt `SiLU`, `lr=0.04`, `batch_size=512`, `n_epochs=4`.
*   **Cơ chế không rò rỉ**: Đặt `stop_epoch=2` hoặc cố định epoch thay vì chọn epoch tốt nhất trên tập validation fold đang chấm.
*   **Kết quả**: Điểm đơn lẻ đạt **$0.95000-0.95015$**. Mặc dù thấp hơn cây, độ tương quan thứ hạng giữa RealMLP và các mô hình cây chỉ là **$0.977$** (trong khi các cây tương quan với nhau $0.987-0.998$). Sự khác biệt này giúp RealMLP chiếm tới $23-33\%$ trọng số trong mô hình Stacking.

### 4.5 Mô Hình Ngôn Ngữ Tự Huấn Luyện Từ Đầu (Target-First LM / GPT-2)
Cách tiếp cận độc đáo nhất: Đọc dữ liệu bảng dưới dạng một chuỗi văn bản (text string):

1.  **Format dòng văn bản**: Chuyển mỗi dòng giao dịch thành chuỗi ký tự theo thứ tự chữ số, **đặt nhãn `is_fraud` lên đầu chuỗi** (Target-First):
    `"is_fraud: 1 | amount_usd: 1371.37 | merchant_category: Gift Cards | ..."`
2.  **Huấn luyện từ đầu (Train from scratch)**: Huấn luyện một kiến trúc GPT-2 / Language Model nhỏ trên tập text này từ đầu (không dùng weights pre-trained ngoài).
3.  **Dự đoán theo định lý Bayes**: Với mỗi dòng kiểm thử, chấm điểm xác suất log-likelihood của chuỗi khi gán `is_fraud: 1` so với khi gán `is_fraud: 0`:
    $$P(\text{fraud} | \text{row}) = \frac{P(\text{row} | \text{fraud}) P(\text{fraud})}{P(\text{row} | \text{fraud}) P(\text{fraud}) + P(\text{row} | \text{legit}) P(\text{legit})}$$
*   **Kết quả**: Mô hình LM đạt điểm đơn lẻ **$0.95003-0.95043$** hoàn toàn từ text thô mà không cần bất kỳ cột Target Encoding thủ công nào. Độ tương quan thứ hạng giữa nó và mô hình GLM-Boost chỉ là **$0.9598$**, tạo ra sự bổ sung tín hiệu mạnh mẽ nhất cho bộ Stacker.

---

## 5. Chiến Thuật Ensemble & Stacking Tự Huấn Luyện (Self-Trained Blending & Stacking)

Khi đã xây dựng xong tập hợp các mô hình đơn lẻ tự huấn luyện ($8-15$ mô hình gồm GLM, LightGBM 3-4 lá, XGBoost, GLM-Boost, RealMLP, Target-First LM), bước cuối cùng là kết hợp chúng thành một đáp án tối ưu.

### 5.1 Chuyển Đổi Không Gian Logit-Rank (Logit-Rank Transformation)

Chỉ số đánh giá cuộc thi là ROC AUC, chỉ quan tâm đến thứ hạng tương đối chứ không quan tâm đến giá trị xác suất thô. Do đó, việc trộn xác suất thô trực tiếp sẽ bị ảnh hưởng bởi sự lệch thang đo giữa các mô hình.

*   **Công thức biến đổi Logit-Rank**: Với chuỗi dự đoán xác suất $p$ gồm $N$ dòng:
    1. Chuyển về tỷ lệ thứ hạng (fractional rank):
       $$U(p) = \frac{\text{rank}(p) - 0.5}{N}$$
    2. Chuyển sang không gian Logit để mở rộng hai phần đuôi phân bố (nơi chứa các ca gian lận cực đoan):
       $$L(p) = \text{logit}(U(p)) = \log\left(\frac{U(p)}{1 - U(p)}\right)$$

### 5.2 Logistic Regression Stacking Lồng Ghép (Nested Logistic Stacker)

Thực hiện huấn luyện một mô hình Logistic Regression trên tập các biến $L(p)$ của các mô hình thành phần:

*   **Mô hình Stacker**: `LogisticRegression(C=0.3, max_iter=1000, solver='lbfgs')` hoặc `solver='newton-cholesky'`.
*   **Nested Stacking**: Để đánh giá điểm CV của Stacker một cách trung thực, chính mô hình Stacker cũng phải được huấn luyện lồng ghép ($5$-fold CV trên tập OOF). Tức là trọng số Stacker được fit trên $4$ fold OOF và áp dụng dự đoán cho fold OOF thứ $5$.
*   **Ưu thế vượt trội của Logistic Stacker so với Hill Climbing (Rank Averaging)**:
    *   Hill Climbing chỉ có thể cộng dồn trọng số dương ($\ge 0$) và bị giới hạn khi số lượng mô hình tăng lên.
    *   Logistic Stacker học được các **trọng số âm (negative weights)** cho một số mô hình. Trọng số âm đóng vai trò hiệu chỉnh sai số (error correction) giữa các mô hình có độ tương quan cao, giúp tối ưu hóa AUC sâu hơn.

### 5.3 Bảng Thang Điểm Tiến Hóa Mô Hình Tự Huấn Luyện (Self-Trained Score Ladder)

Dưới đây là bảng tổng hợp mức tăng trưởng điểm số $5$-fold CV AUC qua từng giai đoạn nâng cấp của hệ thống tự huấn luyện:

| Step | Cấp độ | Cải tiến kỹ thuật chính | 5-Fold CV AUC | Mức tăng |
| :---: | :---: | :--- | :---: | :---: |
| **Step 0** | 🟢 Baseline | Logistic Regression trên dữ liệu thô (One-Hot + Standardized) | **0.94187** | Baseline |
| **Step 1** | 🟡 Intermediate | LightGBM mặc định ($31$ lá) trên dữ liệu thô | **0.94663** | +0.00476 |
| **Step 2** | 🟡 Intermediate | Giảm cây LightGBM xuống **$4$ lá** (`num_leaves=4`) | **0.94771** | +0.00108 |
| **Step 3** | 🟡 Intermediate | + Per-Value Target Encoding In-Fold ($m=20$) | **0.94884** | +0.00113 |
| **Step 4** | 🔴 Expert | **+ Khóa chữ số (Digit Keys)**: Mã hóa 36 khóa chữ số phần xu/nguyên/đầu | **0.95031** | **+0.00147** |
| **Step 5** | 🔴 Expert | + Seed Bagging cho Target Encoder ($3$ seeds) | **0.95051** | +0.00019 |
| **Step 6** | 🔴 Expert | **GLM-Boosted Trees**: Cây $4$ lá học từ logit của GLM hội tụ | **0.95106** | +0.00055 |
| **Step 7** | 🏆 Top Pipeline | **Nested Logistic Stack (Self-Trained)**: Stack GLM-Boost + Digit-Trees + RealMLP + Target-First LM | **0.95150 - 0.95206** | +0.00050 |

---

## 6. Lộ Trình Triển Khai Thực Thi Theo Từng Bước (Step-by-Step Implementation Roadmap)

Dưới đây là danh mục kiểm tra (Checklist) gồm $6$ bước cụ thể để lập trình viên triển khai thành công hệ thống này từ đầu:

### 📋 Bước 1: Khởi Tạo Kiểm Chứng (Setup & Folds)
- [ ] Load dữ liệu `train.csv` ($699,960$ dòng) và `test.csv` ($299,984$ dòng).
- [ ] Khai báo `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` cố định.
- [ ] Đặt tỷ lệ làm mượt Target Encoding mặc định $m=20.0$ cho cây và $m=50.0$ cho GLM.

### 📋 Bước 2: Trích Xuất Khóa Chữ Số & Cặp Tương Tác (Feature Engineering)
- [ ] Trích xuất $6$ loại Digit Keys cho $6$ biến số thực (`cents`, `int`, `int10`, `tenths`, `lead1`, `lead2`).
- [ ] Giữ nguyên biến `cents` làm $1$ cột số thực bổ sung.
- [ ] Ghép $13$ cặp tương tác rủi ro cao cho mô hình GLM (`merchant_category × merchant_risk_score`, `is_foreign_transaction × ip_country_mismatch`, v.v.).

### 📋 Bước 3: Xây Dựng Hàm Target Encoding Không Rò Rỉ (In-Fold TE Function)
- [ ] Viết hàm Target Encoding tính toán tỷ lệ gian lận bằng `np.bincount` lồng ghép bên trong inner fold.
- [ ] Đảm bảo không có dòng dữ liệu nào nhìn thấy nhãn của chính nó.
- [ ] Chạy thử bài kiểm thử label-shuffle test để đảm bảo điểm AUC thu được xấp xỉ $0.48-0.50$ (sạch rò rỉ).

### 📋 Bước 4: Huấn Luyện Các Mô Hình Đơn Lẻ (Model Training)
- [ ] **Mô hình 1 (GLM)**: Logistic Regression với `solver='newton-cholesky'`, $C=0.3$, nạp logit của các bảng TE và $13$ pair keys.
- [ ] **Mô hình 2 (Digit Trees)**: LightGBM $3-4$ lá (`num_leaves=4`, `colsample_bytree=0.5`) nạp đặc trưng thô + Digit Keys TE. Bagging qua $3$ encoder seeds.
- [ ] **Mô hình 3 (GLM-Boosted Trees)**: Khởi tạo `init_score` / `base_margin` bằng logit của GLM, cho LightGBM/XGBoost $4$ lá học sai số residual.
- [ ] **Mô hình 4 (RealMLP)**: Huấn luyện mạng RealMLP trên GPU với $4$ epoch (`stop_epoch=2`), nhúng biến số/danh mục và nạp TE logits.
- [ ] **Mô hình 5 (Target-First LM - Tuỳ chọn)**: Định dạng mỗi dòng thành chuỗi text đặt nhãn lên đầu, huấn luyện GPT-2 nhỏ từ đầu trên tập train và dự đoán xác suất bằng công thức Bayes.

### 📋 Bước 5: Refit Cho Tập Test (Test Refit Execution)
- [ ] Tính trung bình số cây dừng sớm $N_{\text{trees}}$ của $5$ fold cho các mô hình cây.
- [ ] Thực hiện refit từng mô hình trên $100\%$ dữ liệu huấn luyện với $1.25 \times N_{\text{trees}}$ cây.
- [ ] Tạo file dự đoán tập Test từ các mô hình refit.

### 📋 Bước 6: Logistic Stacking Trên Logit-Ranks (Ensembling)
- [ ] Biến đổi toàn bộ dự đoán OOF và Test của các mô hình tự huấn luyện sang không gian Logit-Rank: $L(p) = \text{logit}((\text{rank}(p)-0.5)/N)$.
- [ ] Chạy $5$-fold Nested Logistic Regression Stacker với phạt L2 ($C=0.3 \dots 1.0$) trên tập biến $L(p)$ OOF.
- [ ] Áp dụng trọng số coef thu được để tổng hợp dự đoán tập Test cuối cùng.
- [ ] Đảm bảo file kết quả `submission.csv` có đủ $299,984$ dòng và ID hợp lệ.

---

### 💡 Lời Kết & Các Nguyên Tắc Vàng:
1. **Đọc hiểu bộ tạo dữ liệu quan trọng hơn tinh chỉnh tham số (Generator > Tuning)**: Khai thác chữ số và mã hóa giá trị trùng lặp tăng điểm gấp $3$ lần so với chạy Optuna trên đặc trưng thô.
2. **Ưu tiên mô hình nông (2–4 lá)**: Do hàm mục tiêu mang bản chất cộng tính tuyến tính.
3. **GLM + Trees**: Để GLM đảm nhận phần tuyến tính, cây nông học phần sai số.
4. **Sự đa dạng mô hình là chìa khóa**: Sự kết hợp giữa Cây + GLM + RealMLP + Target-First LM đem lại bước nhảy điểm Stacking lớn nhất nhờ các dạng sai số không trùng lặp.
