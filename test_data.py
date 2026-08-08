import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from vnstock.api.quote import Quote
from sklearn.preprocessing import MinMaxScaler
import os

# Danh sách 10 mã siêu cổ phiếu
danh_sach_ma = ["ACV", "BVH", "GAS", "PLX", "POW", "SAB", "VJC"]

print("🚀 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN TỰ ĐỘNG 10 MÃ CỔ PHIẾU...")

for symbol in danh_sach_ma:
    print(f"\n{'='*50}")
    print(f"🤖 ĐANG HUẤN LUYỆN BỘ NÃO CHO MÃ: {symbol}")
    print(f"{'='*50}")
    
    try:
        # 1. Kéo dữ liệu
        print(f"Đang tải dữ liệu {symbol} từ VNStock...")
        q = Quote(symbol=symbol, source='VCI')
        df = q.history(start='2024-01-01', end='2026-08-08')
        
        if df.empty:
            print(f"⚠️ Lỗi: Không có dữ liệu cho {symbol}, bỏ qua mã này.")
            continue

        # 2. Tiền xử lý
        features = ['close', 'open', 'high', 'low', 'volume']
        data = df.filter(features).values
        
        scaler_all = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler_all.fit_transform(data)
        
        # 3. Tạo cửa sổ trượt T+3
        window_size = 60
        target_offset = 3 
        X, Y = [], []
        
        for i in range(window_size, len(scaled_data) - target_offset + 1):
            X.append(scaled_data[i-window_size:i, :]) 
            Y.append(scaled_data[i + target_offset - 1, 0]) 

        X, Y = np.array(X), np.array(Y)
        
        # Xóa mô hình cũ trong RAM để tránh rò rỉ bộ nhớ giữa các vòng lặp
        tf.keras.backend.clear_session()
        
        # 4. Xây dựng Mạng LSTM
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # 5. Bắt đầu ép máy học
        print(f"Đang ép AI học quy luật của {symbol}...")
        model.fit(X, Y, epochs=20, batch_size=32, verbose=1) # verbose=1 để xem tiến trình học
        
        # 6. Lưu file độc lập
        file_name = f"{symbol.lower()}_price_predictor.keras"
        model.save(file_name)
        print(f"✅ Đã lưu thành công bộ não: {file_name}")
        
    except Exception as e:
        print(f"❌ Xảy ra lỗi khi huấn luyện mã {symbol}: {e}")

print("\n🎉 CHÚC MỪNG! HỆ THỐNG ĐÃ HUẤN LUYỆN XONG TOÀN BỘ DANH SÁCH.")