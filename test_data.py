import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from vnstock.api.quote import Quote
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

print("1. KÉO DỮ LIỆU ĐA CHIỀU (Giá và Khối lượng)...")
q = Quote(symbol='FPT', source='VCI')
df = q.history(start='2024-01-01', end='2026-08-08')

# CẢI TIẾN 1: Lấy 5 cột dữ liệu thay vì 1. Sắp xếp 'close' ở cột đầu tiên (index 0)
features = ['close', 'open', 'high', 'low', 'volume']
data = df.filter(features).values

print("2. CHUẨN HÓA DỮ LIỆU...")
# Scale chung cho cả 5 cột ma trận đầu vào (X)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Tạo một scaler riêng lẻ chỉ cho 'close' để xài cho kết quả đầu ra (Y) lúc vẽ đồ thị
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_close.fit(df.filter(['close']).values)

print("3. TẠO CỬA SỔ TRƯỢT ĐA ĐẶC TRƯNG (NÂNG CẤP T+3)...")
window_size = 60
target_offset = 3  # CHÌA KHÓA: Ép AI dự báo điểm rơi 3 ngày sau (T+3)

X, Y = [], []
# Phải trừ đi target_offset để vòng lặp không bị lỗi thiếu dữ liệu ở các ngày cuối
for i in range(window_size, len(scaled_data) - target_offset + 1):
    # Nạp 60 ngày vào X (không đổi)
    X.append(scaled_data[i-window_size:i, :]) 
    
    # CẢI TIẾN T+3: Cố tình dịch chuyển Y về tương lai 3 ngày
    Y.append(scaled_data[i + target_offset - 1, 0]) 

X, Y = np.array(X), np.array(Y)
print(f"Kích thước X mới: {X.shape} -> (Số mẫu, 60 ngày, 5 đặc trưng)")

print("\n4. XÂY DỰNG MÔ HÌNH MULTIVARIATE LSTM...")
model = Sequential()
# CẢI TIẾN 3: Khai báo input_shape nhận 5 đặc trưng (thay vì 1 như cũ)
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

print("\n5. BẮT ĐẦU HUẤN LUYỆN...")
model.fit(X, Y, epochs=20, batch_size=32)

print("\n6. DỰ ĐOÁN VÀ VẼ BIỂU ĐỒ...")
predicted_prices = model.predict(X)

# CẢI TIẾN 4: Dùng scaler_close để giải mã ngược ra tiền VNĐ chính xác
predicted_prices = scaler_close.inverse_transform(predicted_prices)
real_prices = scaler_close.inverse_transform(Y.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(real_prices, color='blue', label='Giá Thực Tế (FPT)')
plt.plot(predicted_prices, color='red', linestyle='dashed', label='Giá AI Dự Đoán (Có Volume)')
plt.title('Bản Nâng Cấp: Dự Đoán Giá FPT Kết Hợp Khối Lượng Giao Dịch')
plt.xlabel('Thời gian (Phiên giao dịch trượt)')
plt.ylabel('Giá Cổ Phiếu')
plt.legend()
plt.grid(True)
plt.show()

# Lưu đè bản nâng cấp đa chiều
model.save("fpt_price_predictor.keras")

plt.show()