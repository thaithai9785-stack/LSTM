import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from vnstock.api.quote import Quote
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime

st.set_page_config(page_title="AI Chứng Khoán", page_icon="📈", layout="centered")

st.title("📈 Hệ thống AI Dự báo Toàn Thị Trường")
st.markdown("---")

danh_sach_ma = ["ACV", "BVH", "FPT", "GAS", "HPG", "MSN", "MWG", "PLX", "POW", "SAB", "TCB", "VCB", "VIC", "VJC", "VNM"]
FILE_LICH_SU = "trading_log.csv"

tab_quet, tab_lich_su = st.tabs(["📊 Bảng Điều Khiển T+3", "🕒 Lịch Sử Dự Báo"])

# ----------------- TAB 1: QUÉT TÍN HIỆU -----------------
with tab_quet:
    if "df_gia" not in st.session_state:
        st.session_state.df_gia = pd.DataFrame({
            "Mã Cổ Phiếu": danh_sach_ma,
            "Giá Hiện Tại (VNĐ)": [0] * len(danh_sach_ma)
        })

    st.write("Bấm nút bên dưới để AI tự động cập nhật giá, hoặc bạn có thể tự gõ tay.")

    if st.button("🔄 TỰ ĐỘNG LẤY GIÁ THỊ TRƯỜNG"):
        with st.spinner("Đang kết nối API với bảng điện..."):
            gia_moi = []
            for ma in danh_sach_ma:
                try:
                    # TRỞ LẠI NGUỒN 'kbs' CŨ
                    q = Quote(symbol=ma, source='kbs') 
                    df_temp = q.history(start='2024-01-01', end='2026-08-08') 
                    # BẪY LỖI: Chỉ lấy giá khi df có dữ liệu
                    if df_temp is not None and not df_temp.empty:
                        gia_chot = float(df_temp['close'].iloc[-1]) * 1000
                        gia_moi.append(int(gia_chot))
                    else:
                        gia_moi.append(0)
                except Exception:
                    gia_moi.append(0) 
            
            st.session_state.df_gia["Giá Hiện Tại (VNĐ)"] = gia_moi
            st.success("Đã cập nhật giá mới nhất thành công!")

    df_nhap_lieu = st.data_editor(st.session_state.df_gia, hide_index=True, use_container_width=True)

    if st.button("⚡ QUÉT TOÀN BỘ DANH MỤC", type="primary", use_container_width=True):
        df_can_du_bao = df_nhap_lieu[df_nhap_lieu["Giá Hiện Tại (VNĐ)"] > 0]
        
        if df_can_du_bao.empty:
            st.warning("⚠️ Vui lòng nhập giá hoặc bấm lấy giá tự động!")
        else:
            st.success(f"Đang phân tích {len(df_can_du_bao)} mã cổ phiếu...")
            st.markdown("---")
            
            du_lieu_luu_tru = []
            thoi_gian_quet = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for index, row in df_can_du_bao.iterrows():
                ma_co_phieu = row["Mã Cổ Phiếu"]
                gia_hien_tai = row["Giá Hiện Tại (VNĐ)"]
                
                st.markdown(f"#### 🔍 Mã: **{ma_co_phieu}**")
                
                with st.spinner(f"Đang chạy AI cho {ma_co_phieu}..."):
                    try:
                        model_path = f"{ma_co_phieu.lower()}_price_predictor.keras"
                        
                        if not os.path.exists(model_path):
                            st.error(f"Chưa có AI cho {ma_co_phieu}.")
                            continue
                            
                        model = load_model(model_path)
                        # TRỞ LẠI NGUỒN 'kbs' CŨ
                        q = Quote(symbol=ma_co_phieu, source='kbs')
                        df = q.history(start='2024-01-01', end='2026-08-08')
                        
                        # BẪY LỖI MSN CŨ Ở ĐÂY
                        if df is None or df.empty:
                            st.error(f"Không tải được dữ liệu mạng cho mã {ma_co_phieu}. Đã tự động bỏ qua để không bị treo.")
                            continue
                            
                        features = ['close', 'open', 'high', 'low', 'volume']
                        data = df.filter(features).values
                        
                        scaler_close = MinMaxScaler(feature_range=(0, 1))
                        scaler_close.fit(df.filter(['close']).values)
                        scaler_all = MinMaxScaler(feature_range=(0, 1))
                        scaled_data = scaler_all.fit_transform(data)
                        
                        last_60_days = scaled_data[-60:]
                        X_input = np.reshape(last_60_days, (1, 60, 5))
                        
                        predicted_scaled = model.predict(X_input, verbose=0)
                        gia_du_doan = float(scaler_close.inverse_transform(predicted_scaled)[0][0]) * 1000
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Mức Giá Cập Nhật", value=f"{gia_hien_tai:,.0f} VNĐ")
                        with col2:
                            st.metric(label="Mức Giá Dự Kiến (T+3)", value=f"{gia_du_doan:,.0f} VNĐ")
                        
                        ty_suat_thô = ((gia_du_doan - gia_hien_tai) / gia_hien_tai) * 100
                        loi_nhuan_thuc_te = ty_suat_thô - 0.4 
                        
                        tin_hieu_chu = ""
                        if loi_nhuan_thuc_te >= 1.5:
                            tin_hieu_chu = "MUA ĐẸP"
                            st.success(f"🔥 **TÍN HIỆU: {tin_hieu_chu}** (+{loi_nhuan_thuc_te:.2f}%)")
                        elif loi_nhuan_thuc_te > 0:
                            tin_hieu_chu = "ĐỨNG NGOÀI"
                            st.warning(f"⚠️ **TÍN HIỆU: {tin_hieu_chu}** (+{loi_nhuan_thuc_te:.2f}%)")
                        else:
                            tin_hieu_chu = "KHÔNG MUA / CẮT LỖ"
                            st.error(f"❄️ **TÍN HIỆU: {tin_hieu_chu}** ({loi_nhuan_thuc_te:.2f}%)")
                        
                        du_lieu_luu_tru.append({
                            "Thời gian": thoi_gian_quet,
                            "Mã CP": ma_co_phieu,
                            "Giá Cập Nhật": gia_hien_tai,
                            "Giá T+3 Dự Kiến": int(gia_du_doan),
                            "Lãi Dự Kiến (%)": round(loi_nhuan_thuc_te, 2),
                            "Tín Hiệu": tin_hieu_chu
                        })
                                
                    except Exception as e:
                        st.error(f"⚠️ Mã {ma_co_phieu} gặp lỗi kết nối: {e}. Đã tự động bỏ qua.")
                
                st.markdown("---") 
            
            if len(du_lieu_luu_tru) > 0:
                df_log = pd.DataFrame(du_lieu_luu_tru)
                if os.path.exists(FILE_LICH_SU):
                    df_log.to_csv(FILE_LICH_SU, mode='a', header=False, index=False, encoding='utf-8')
                else:
                    df_log.to_csv(FILE_LICH_SU, mode='w', header=True, index=False, encoding='utf-8')
                st.info("💾 Đã lưu phiên phân tích này vào Sổ Nhật Ký.")
            
            st.balloons()

# ----------------- TAB 2: LỊCH SỬ DỰ BÁO -----------------
with tab_lich_su:
    st.markdown("### 🕒 Sổ Nhật Ký Đối Chiếu")
    st.write("Tại đây lưu trữ toàn bộ các phiên bạn đã quét để thứ 5 tuần này đối chiếu lại.")
    
    if os.path.exists(FILE_LICH_SU):
        df_history = pd.read_csv(FILE_LICH_SU)
        df_history = df_history.iloc[::-1]
        st.dataframe(df_history, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        if st.button("🗑️ Xóa sạch lịch sử", type="secondary"):
            os.remove(FILE_LICH_SU)
            st.rerun()
    else:
        st.info("Chưa có dữ liệu nào. Dữ liệu sẽ xuất hiện ở đây sau khi bạn quét danh mục bên Tab 1.")