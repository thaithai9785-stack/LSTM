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

# ==========================================
# GIAI ĐOẠN 2: TỰ ĐỘNG LẤY GIÁ & QUẢN LÝ BẢNG
# ==========================================

# Khởi tạo bộ nhớ tạm (session_state) cho bảng dữ liệu nếu chưa có
if "df_gia" not in st.session_state:
    st.session_state.df_gia = pd.DataFrame({
        "Mã Cổ Phiếu": danh_sach_ma,
        "Giá Hiện Tại (VNĐ)": [0] * len(danh_sach_ma)
    })

st.markdown("### 📊 Bảng Điều Khiển Danh Mục T+3")
st.write("Bấm nút bên dưới để AI tự động cập nhật giá mới nhất từ bảng điện, hoặc bạn có thể tự nhập tay.")

# Nút tự động kéo giá
if st.button("🔄 TỰ ĐỘNG LẤY GIÁ THỊ TRƯỜNG"):
    with st.spinner("Đang kết nối API với bảng điện để lấy giá mới nhất..."):
        gia_moi = []
        for ma in danh_sach_ma:
            try:
                # Kéo dữ liệu giá mới nhất của từng mã
                q = Quote(symbol=ma, source='kbs')
                df_temp = q.history(start='2024-01-01', end='2026-08-08') 
                
                # Kiểm tra xem dữ liệu kéo về có bị rỗng/lỗi không
                if df_temp is not None and not df_temp.empty:
                    gia_chot = float(df_temp['close'].iloc[-1]) * 1000 # Lấy giá ngày cuối cùng x 1000
                    gia_moi.append(int(gia_chot))
                else:
                    gia_moi.append(0)
            except Exception as e:
                gia_moi.append(0) # Nếu lỗi (mã không tồn tại/lỗi mạng), để giá bằng 0
        
        # Cập nhật lại bộ nhớ tạm
        st.session_state.df_gia["Giá Hiện Tại (VNĐ)"] = gia_moi
        st.success("Đã cập nhật giá mới nhất thành công!")

# ==========================================

# Hiển thị bảng tương tác (được liên kết với bộ nhớ tạm)
df_nhap_lieu = st.data_editor(st.session_state.df_gia, hide_index=True, use_container_width=True)

# Nút quét hàng loạt
if st.button("⚡ QUÉT TOÀN BỘ DANH MỤC", type="primary", use_container_width=True):
    
    # Lọc ra các mã có giá trị nhập > 0
    df_can_du_bao = df_nhap_lieu[df_nhap_lieu["Giá Hiện Tại (VNĐ)"] > 0]
    
    if df_can_du_bao.empty:
        st.warning("⚠️ Vui lòng nhập giá hoặc bấm nút lấy giá tự động cho ít nhất 1 mã cổ phiếu!")
    else:
        st.success(f"Đang phân tích {len(df_can_du_bao)} mã cổ phiếu...")
        st.markdown("---")
        
        # Bắt đầu vòng lặp chạy từng mã
        for index, row in df_can_du_bao.iterrows():
            ma_co_phieu = row["Mã Cổ Phiếu"]
            gia_hien_tai = row["Giá Hiện Tại (VNĐ)"]
            
            st.markdown(f"#### 🔍 Kết quả phân tích mã: **{ma_co_phieu}**")
            
            with st.spinner(f"Đang chạy AI cho {ma_co_phieu}..."):
                try:
                    model_path = f"{ma_co_phieu.lower()}_price_predictor.keras"
                    
                    if not os.path.exists(model_path):
                        st.error(f"Chưa có bộ não AI cho mã {ma_co_phieu}. Vui lòng huấn luyện trước!")
                        continue # Bỏ qua mã này, chạy tiếp mã sau
                        
                    # Load AI và kéo dữ liệu
                    model = load_model(model_path)
                    q = Quote(symbol=ma_co_phieu, source='kbs')
                    df = q.history(start='2024-01-01', end='2026-08-08')
                    
                    # Bẫy lỗi: Nếu dữ liệu mạng trả về rỗng thì bỏ qua ngay lập tức
                    if df is None or df.empty:
                        st.error(f"Không thể tải được dữ liệu mạng cho mã {ma_co_phieu}. Bỏ qua mã này.")
                        continue
                        
                    features = ['close', 'open', 'high', 'low', 'volume']
                    data = df.filter(features).values
                    
                    scaler_close = MinMaxScaler(feature_range=(0, 1))
                    scaler_close.fit(df.filter(['close']).values)
                    
                    scaler_all = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler_all.fit_transform(data)
                    
                    last_60_days = scaled_data[-60:]
                    X_input = np.reshape(last_60_days, (1, 60, 5))
                    
                    # Suy luận giá
                    predicted_scaled = model.predict(X_input, verbose=0)
                    gia_du_doan = float(scaler_close.inverse_transform(predicted_scaled)[0][0]) * 1000
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Mức Giá Cập Nhật", value=f"{gia_hien_tai:,.0f} VNĐ")
                    with col2:
                        st.metric(label="Mức Giá Dự Kiến (T+3)", value=f"{gia_du_doan:,.0f} VNĐ")
                    
                    ty_suat_thô = ((gia_du_doan - gia_hien_tai) / gia_hien_tai) * 100
                    loi_nhuan_thuc_te = ty_suat_thô - 0.4 
                    
                    # Đưa ra khuyến nghị
                    if loi_nhuan_thuc_te >= 1.5:
                        st.success(f"🔥 **TÍN HIỆU: MUA ĐẸP** (Lãi ròng dự kiến: **+{loi_nhuan_thuc_te:.2f}%**)")
                    elif loi_nhuan_thuc_te > 0:
                        st.warning(f"⚠️ **TÍN HIỆU: ĐỨNG NGOÀI** (Lãi quá mỏng: **+{loi_nhuan_thuc_te:.2f}%**)")
                    else:
                        st.error(f"❄️ **TÍN HIỆU: KHÔNG MUA / CẮT LỖ** (Dự kiến âm: **{loi_nhuan_thuc_te:.2f}%**)")
                            
                except Exception as e:
                    # Bẫy lỗi: Đảm bảo app không sập nếu có bất cứ lỗi nào phát sinh
                    st.error(f"⚠️ Mã {ma_co_phieu} gặp lỗi kết nối API: {e}. Hệ thống đã tự động bỏ qua để chạy tiếp.")
            
            st.markdown("---") 
            
        st.balloons()