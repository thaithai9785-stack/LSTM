import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from vnstock.api.quote import Quote
from sklearn.preprocessing import MinMaxScaler
import os

st.set_page_config(page_title="AI Chứng Khoán", page_icon="📈", layout="centered")

st.title("📈 Hệ thống AI Dự báo Toàn Thị Trường")
st.markdown("---")

danh_sach_ma = ["FPT", "HPG", "TCB", "VCB", "SSI", "VND", "MWG", "VNM", "MSN", "VIC"]
ma_co_phieu = st.selectbox("🔍 Chọn mã cổ phiếu muốn phân tích:", danh_sach_ma)

gia_hien_tai = st.number_input(f"Nhập giá {ma_co_phieu} hiện tại trên bảng điện (VNĐ):", min_value=1000, value=68000, step=100)

if st.button(f"Dự đoán giá {ma_co_phieu} sau 3 ngày (T+3)", type="primary", use_container_width=True):
    with st.spinner(f"AI đang tải dữ liệu và phân tích mã {ma_co_phieu}..."):
        try:
            model_path = f"{ma_co_phieu.lower()}_price_predictor.keras"
            
            if not os.path.exists(model_path):
                st.error(f"Chưa có bộ não AI cho mã {ma_co_phieu}. Vui lòng huấn luyện trước!")
            else:
                # 1. Load AI và kéo dữ liệu trực tiếp trên Web
                model = load_model(model_path)
                q = Quote(symbol=ma_co_phieu, source='msn')
                df = q.history(start='2024-01-01', end='2026-08-08')
                
                features = ['close', 'open', 'high', 'low', 'volume']
                data = df.filter(features).values
                
                scaler_close = MinMaxScaler(feature_range=(0, 1))
                scaler_close.fit(df.filter(['close']).values)
                
                scaler_all = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler_all.fit_transform(data)
                
                last_60_days = scaled_data[-60:]
                X_input = np.reshape(last_60_days, (1, 60, 5))
                
                # 2. Suy luận giá
                predicted_scaled = model.predict(X_input)
                gia_du_doan = float(scaler_close.inverse_transform(predicted_scaled)[0][0]) * 1000
                
                st.success("Phân tích hoàn tất!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Mã Cổ Phiếu", value=ma_co_phieu)
                with col2:
                    st.metric(label="Mức Giá Dự Kiến (T+3)", value=f"{gia_du_doan:,.0f} VNĐ")
                
                st.markdown("---")
                
                ty_suat_thô = ((gia_du_doan - gia_hien_tai) / gia_hien_tai) * 100
                loi_nhuan_thuc_te = ty_suat_thô - 0.4 
                
                st.subheader("Hệ thống Cố vấn Khuyến nghị:")
                if loi_nhuan_thuc_te >= 1.5:
                    st.success(f"🔥 **TÍN HIỆU: MUA ĐẸP** (Lãi ròng dự kiến: **+{loi_nhuan_thuc_te:.2f}%**)")
                elif loi_nhuan_thuc_te > 0:
                    st.warning(f"⚠️ **TÍN HIỆU: ĐỨNG NGOÀI** (Lãi quá mỏng: **+{loi_nhuan_thuc_te:.2f}%**)")
                else:
                    st.error(f"❄️ **TÍN HIỆU: KHÔNG MUA / CẮT LỖ** (Dự kiến âm: **{loi_nhuan_thuc_te:.2f}%**)")
                    
        except Exception as e:
            st.error(f"Hệ thống gặp sự cố: {e}")