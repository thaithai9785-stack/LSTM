from fastapi import FastAPI
import numpy as np
from tensorflow.keras.models import load_model
from vnstock.api.quote import Quote
from sklearn.preprocessing import MinMaxScaler
import os

app = FastAPI(title="AI Stock Prediction API")

@app.get("/")
def home():
    return {"message": "Hệ thống AI Đa Mã Cổ Phiếu đang hoạt động!"}

# ĐỔI THÀNH BIẾN ĐỘNG {symbol}
@app.get("/predict/{symbol}")
def predict_price(symbol: str):
    symbol = symbol.upper()
    
    # 1. Tự động tìm bộ não AI tương ứng với mã cổ phiếu
    model_path = f"{symbol.lower()}_price_predictor.keras"
    if not os.path.exists(model_path):
         return {"status": "error", "message": f"Chưa có bộ não AI cho mã {symbol}. Hãy chạy file huấn luyện trước!"}
    
    model = load_model(model_path)
    
    # 2. Kéo dữ liệu mã tương ứng từ VNStock
    q = Quote(symbol=symbol, source='VCI')
    df = q.history(start='2024-01-01', end='2026-08-08')
    
    features = ['close', 'open', 'high', 'low', 'volume']
    data = df.filter(features).values
    
    scaler_close = MinMaxScaler(feature_range=(0, 1))
    scaler_close.fit(df.filter(['close']).values)
    
    scaler_all = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_all.fit_transform(data)
    
    last_60_days = scaled_data[-60:]
    X_input = np.reshape(last_60_days, (1, 60, 5))
    
    predicted_scaled = model.predict(X_input)
    predicted_price = scaler_close.inverse_transform(predicted_scaled)
    
    return {
        "status": "success",
        "symbol": symbol,
        "predicted_price_vnd": round(float(predicted_price[0][0]), 2),
        "model_used": f"Multivariate LSTM T+3 ({symbol})"
    }