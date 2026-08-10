import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from vnstock.api.quote import Quote
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime
import gspread # Thư viện Google Sheets
import concurrent.futures # Thư viện làm cầu dao chống treo

st.set_page_config(page_title="AI Chứng Khoán", page_icon="📈", layout="centered")

st.title("📈 Hệ thống AI Dự báo Toàn Thị Trường")
st.markdown("---")

danh_sach_ma = ["ACV", "BVH", "FPT", "GAS", "HPG", "MSN", "MWG", "PLX", "POW", "SAB", "TCB", "VCB", "VIC", "VJC", "VNM"]

# --- HÀM KẾT NỐI GOOGLE SHEETS ---
@st.cache_resource
def get_gspread_client():
    credentials = st.secrets["gcp_service_account"]
    gc = gspread.service_account_from_dict(credentials)
    return gc

SHEET_URL = "https://docs.google.com/spreadsheets/d/1NveHlCyiFd4-tbVH-dV9K2vPqydD-jPgPL75aIOCCOA/edit"

# --- HÀM KÉO API CHO CẦU DAO ---
def lay_du_lieu_api(ma):
    # DÙNG LẠI NGUỒN KBS ĐÃ CHỨNG MINH LÀ CHẠY ĐƯỢC
    q = Quote(symbol=ma, source='kbs')
    return q.history(start='2024-01-01', end='2026-08-10')

tab_quet, tab_lich_su = st.tabs(["📊 Bảng Điều Khiển T+3", "☁️ Lịch Sử Trên Mây"])

# ----------------- TAB 1: QUÉT TÍN HIỆU -----------------
with tab_quet:
    if "df_gia" not in st.session_state:
        st.session_state.df_gia = pd.DataFrame({
            "Mã Cổ Phiếu": danh_sach_ma,
            "Giá Hiện Tại (VNĐ)": [0] * len(danh_sach_ma)
        })

    st.write("Bấm nút bên dưới để AI tự động cập nhật giá, hoặc bạn có thể tự gõ tay.")

    if st.button("🔄 TỰ ĐỘNG LẤY GIÁ THỊ TRƯỜNG"):
        with st.spinner("Đang kết nối API với bảng điện KBS..."):
            gia_moi = []
            # Bật cầu dao đa luồng
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for ma in danh_sach_ma:
                    try:
                        future = executor.submit(lay_du_lieu_api, ma)
                        # Ép thời gian chờ tối đa 8 giây
                        df_temp = future.result(timeout=8) 
                        
                        if df_temp is not None and not df_temp.empty:
                            gia_chot = float(df_temp['close'].iloc[-1]) * 1000
                            gia_moi.append(int(gia_chot))
                        else:
                            gia_moi.append(0)
                    except concurrent.futures.TimeoutError:
                        gia_moi.append(0) # Quá 8s -> Ngắt mạng, gán 0
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
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
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
                            
                            # CẦU DAO CHỐNG TREO CHO PHẦN CHẠY AI
                            future = executor.submit(lay_du_lieu_api, ma_co_phieu)
                            df = future.result(timeout=8)
                            
                            if df is None or df.empty:
                                st.error(f"Không tải được dữ liệu mạng cho mã {ma_co_phieu}. Đã tự động bỏ qua.")
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
                            
                            du_lieu_luu_tru.append([
                                thoi_gian_quet,
                                ma_co_phieu,
                                gia_hien_tai,
                                int(gia_du_doan),
                                round(loi_nhuan_thuc_te, 2),
                                tin_hieu_chu
                            ])
                                    
                        except concurrent.futures.TimeoutError:
                            st.error(f"⚠️ Máy chủ mạng treo quá 8 giây tại mã {ma_co_phieu}. Đã ép ngắt kết nối để chạy tiếp!")
                        except Exception as e:
                            st.error(f"⚠️ Mã {ma_co_phieu} gặp lỗi: {e}. Đã tự động bỏ qua.")
                    
                    st.markdown("---") 
        
        # BẮN DỮ LIỆU LÊN GOOGLE SHEETS
        if len(du_lieu_luu_tru) > 0:
            with st.spinner("Đang đồng bộ dữ liệu lên Google Sheets Đám Mây..."):
                try:
                    gc = get_gspread_client()
                    sh = gc.open_by_url(SHEET_URL)
                    worksheet = sh.sheet1
                    worksheet.append_rows(du_lieu_luu_tru)
                    st.info("☁️ Đã lưu phiên phân tích này lên Google Sheets Đám Mây vĩnh viễn!")
                except Exception as e:
                    st.error(f"Lỗi đồng bộ mây: {e}")
        
        st.balloons()

# ----------------- TAB 2: LỊCH SỬ DỰ BÁO -----------------
with tab_lich_su:
    st.markdown("### ☁️ Sổ Nhật Ký Đối Chiếu (Đồng bộ Google Sheets)")
    st.write("Dữ liệu được lưu vĩnh viễn trên mây, không bao giờ bị mất.")
    
    try:
        gc = get_gspread_client()
        sh = gc.open_by_url(SHEET_URL)
        worksheet = sh.sheet1
        records = worksheet.get_all_records()
        
        if len(records) > 0:
            df_history = pd.DataFrame(records)
            df_history = df_history.iloc[::-1] 
            st.dataframe(df_history, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            # --- THIẾT KẾ LẠI KHU VỰC NÚT BẤM ---
            col1, col2 = st.columns(2)
            with col1:
                # Nút nhảy sang Google Sheets
                st.link_button("🌐 MỞ TRỰC TIẾP FILE GOOGLE SHEETS", SHEET_URL, type="primary", use_container_width=True)
            with col2:
                # Nút xóa lịch sử
                if st.button("🗑️ Xóa sạch lịch sử trên mây", type="secondary", use_container_width=True):
                    worksheet.resize(1) 
                    st.rerun()
        else:
            st.info("Chưa có dữ liệu nào. Hãy quét danh mục ở Tab 1 để đồng bộ lên mây.")
            st.link_button("🌐 MỞ TRỰC TIẾP FILE GOOGLE SHEETS", SHEET_URL, type="primary")
            
    except Exception as e:
        st.error(f"Không thể tải lịch sử: Lỗi chi tiết: {e}")