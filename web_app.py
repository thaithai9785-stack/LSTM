import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from vnstock.api.quote import Quote
from sklearn.preprocessing import MinMaxScaler
import os

st.set_page_config(page_title="AI Chứng Khoán", page_icon="📈", layout="centered")

st.title("📈 Hệ thống AI Dự báo Toàn Thị Trường")
st.markdown("---")

# Khai báo danh sách mã
danh_sach_ma = ["ACV", "BVH", "FPT", "GAS", "HPG", "MSN", "MWG", "PLX", "POW", "SAB", "TCB", "VCB", "VIC", "VJC", "VNM"]

# Giao diện Bảng nhập liệu
st.markdown("### 📊 Bảng Điều Khiển Danh Mục T+3")
st.write("Nhập giá hiện tại vào cột bên dưới. Bạn có thể nhập nhiều mã cùng lúc (những mã để giá trị 0 hệ thống sẽ tự động bỏ qua).")

# 1. Tạo bảng dữ liệu mặc định
df_mac_dinh = pd.DataFrame({
    "Mã Cổ Phiếu": danh_sach_ma,
    "Giá Nhập Tay (VNĐ)": [0] * len(danh_sach_ma)
})

# 2. Hiển thị bảng để nhập liệu
df_nhap_lieu = st.data_editor(df_mac_dinh, hide_index=True, use_container_width=True)

# 3. Nút quét hàng loạt
if st.button("⚡ QUÉT TOÀN BỘ DANH MỤC", type="primary", use_container_width=True):
    
    # Lọc ra các mã có giá trị nhập > 0
    df_can_du_bao = df_nhap_lieu[df_nhap_lieu["Giá Nhập Tay (VNĐ)"] > 0]
    
    if df_can_du_bao.empty:
        st.warning("⚠️ Vui lòng nhập giá cho ít nhất 1 mã cổ phiếu!")
    else:
        st.success(f"Đang phân tích {len(df_can_du_bao)} mã cổ phiếu...")
        st.markdown("---")
        
        # Bắt đầu vòng lặp chạy từng mã
        for index, row in df_can_du_bao.iterrows():
            ma_co_phieu = row["Mã Cổ Phiếu"]
            gia_hien_tai = row["Giá Nhập Tay (VNĐ)"]
            
            st.markdown(f"#### 🔍 Kết quả phân tích mã: **{ma_co_phieu}**")
            
            with st.spinner(f"Đang chạy AI cho {ma_co_phieu}..."):
                try:
                    model_path = f"{ma_co_phieu.lower()}_price_predictor.keras"
                    
                    if not os.path.exists(model_path):
                        st.error(f"Chưa có bộ não AI cho mã {ma_co_phieu}. Vui lòng huấn luyện trước!")
                    else:
                        # 1. Load AI và kéo dữ liệu
                        model = load_model(model_path)
                        q = Quote(symbol=ma_co_phieu, source='kbs')
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
                        predicted_scaled = model.predict(X_input, verbose=0)
                        gia_du_doan = float(scaler_close.inverse_transform(predicted_scaled)[0][0]) * 1000
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Mức Giá Bạn Nhập", value=f"{gia_hien_tai:,.0f} VNĐ")
                        with col2:
                            st.metric(label="Mức Giá Dự Kiến (T+3)", value=f"{gia_du_doan:,.0f} VNĐ")
                        
                        ty_suat_thô = ((gia_du_doan - gia_hien_tai) / gia_hien_tai) * 100
                        loi_nhuan_thuc_te = ty_suat_thô - 0.4 
                        
                        # 3. Đưa ra khuyến nghị
                        if loi_nhuan_thuc_te >= 1.5:
                            st.success(f"🔥 **TÍN HIỆU: MUA ĐẸP** (Lãi ròng dự kiến: **+{loi_nhuan_thuc_te:.2f}%**)")
                        elif loi_nhuan_thuc_te > 0:
                            st.warning(f"⚠️ **TÍN HIỆU: ĐỨNG NGOÀI** (Lãi quá mỏng: **+{loi_nhuan_thuc_te:.2f}%**)")
                        else:
                            st.error(f"❄️ **TÍN HIỆU: KHÔNG MUA / CẮT LỖ** (Dự kiến âm: **{loi_nhuan_thuc_te:.2f}%**)")
                            
                except Exception as e:
                    st.error(f"Hệ thống gặp sự cố với mã {ma_co_phieu}: {e}")
            
            st.markdown("---") # Kẻ đường phân cách giữa các mã
            
        st.balloons() # Bắn bóng bay khi quét xong toàn bộ