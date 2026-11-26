# Từ điển 24 đặc trưng gốc của S6E11

Bộ dữ liệu có **13 cột số, 5 cột phân loại và 6 cờ Có/Không**. Cả 24 cột đều có trong `train.csv` và `test.csv`. Ngoài các đặc trưng này, `id` là mã dòng giao dịch và `is_fraud` là nhãn trong `train.csv`, với `1` là gian lận và `0` là hợp lệ. [Mô tả cuộc thi trên Kaggle](https://www.kaggle.com/competitions/playground-series-s6e11/overview)

## 13 cột số

| Tên cột | Ý nghĩa |
|---|---|
| `amount_usd` | Số tiền của giao dịch, tính bằng USD. |
| `hours_since_last_txn` | Số giờ kể từ giao dịch trước. |
| `txn_count_last_24h` | Số giao dịch ghi nhận trong 24 giờ gần nhất. |
| `distance_from_home_km` | Khoảng cách từ vị trí giao dịch đến nhà của khách hàng, tính bằng km. |
| `card_age_months` | Tuổi của thẻ, tính bằng tháng. |
| `customer_age` | Tuổi của khách hàng, tính bằng năm. |
| `account_balance_usd` | Số dư tài khoản, tính bằng USD. |
| `cvv_retry_count` | Số lần nhập lại mã bảo mật CVV. Giá trị quan sát từ `0` đến `3`. |
| `velocity_score` | Điểm phản ánh mức độ dồn dập của hoạt động giao dịch. |
| `time_of_day_hour` | Giờ thực hiện giao dịch, mã hóa từ `0` đến `23`. |
| `day_of_week` | Ngày trong tuần, mã hóa từ `0` đến `6`. |
| `merchant_risk_score` | Điểm rủi ro của đơn vị bán hàng. Giá trị quan sát từ `0` đến `100`. |
| `prior_disputes` | Số lần tranh chấp giao dịch trước đó. |

## 5 cột phân loại

| Tên cột | Ý nghĩa |
|---|---|
| `merchant_category` | Ngành hàng của đơn vị bán hàng, ví dụ `Groceries` (hàng tạp hóa), `Fuel` (nhiên liệu), `Online Retail` (bán lẻ trực tuyến). |
| `card_type` | Thương hiệu hoặc mạng lưới thẻ. Có `Visa`, `Mastercard`, `Amex`, `RuPay` và `Discover`. |
| `auth_method` | Phương thức xác thực. Có `3D Secure`, `OTP` (mã dùng một lần), `PIN`, `Biometric` (sinh trắc học) và `No Authentication` (không xác thực). |
| `channel` | Kênh thực hiện giao dịch. Có `Online`, `POS` (tại điểm bán), `Contactless` (không tiếp xúc), `In-App` (trong ứng dụng) và `ATM`. |
| `device_type` | Loại thiết bị thực hiện giao dịch, ví dụ `iPhone`, `Android Phone`, `Windows PC` hoặc `POS Terminal`. |

## 6 cờ Có/Không

`True` cho biết cờ được bật, còn `False` cho biết cờ được tắt.

| Tên cột | Ý nghĩa khi `True` |
|---|---|
| `is_foreign_transaction` | Giao dịch được đánh dấu là giao dịch nước ngoài. |
| `is_new_merchant` | Giao dịch diễn ra tại đơn vị bán hàng được đánh dấu là mới đối với khách hàng. |
| `used_vpn` | Kết nối dùng để giao dịch có sử dụng VPN. |
| `ip_country_mismatch` | Quốc gia suy ra từ địa chỉ IP khác với quốc gia dùng để đối chiếu. |
| `billing_shipping_mismatch` | Địa chỉ thanh toán và địa chỉ giao hàng khác nhau. |
| `is_ai_generated_scam_attempt` | Giao dịch được gắn cờ có dấu hiệu của một nỗ lực lừa đảo do AI tạo. |

## Phạm vi đối chiếu

Tên cột được đối chiếu với hàng tiêu đề của hai CSV. Miền giá trị và ví dụ lấy từ lần kiểm tra toàn bộ `train.csv` và `test.csv`, gồm 699.960 dòng train và 299.984 dòng test. Tài liệu truy cập được chưa xác nhận công thức của `velocity_score` và `merchant_risk_score`, ngày tương ứng với mã `0` của `day_of_week`, hay quốc gia tham chiếu của `ip_country_mismatch`. [Chi tiết kiểm tra dữ liệu](data_audit.json) · [Báo cáo EDA](ket_qua_eda.md)
