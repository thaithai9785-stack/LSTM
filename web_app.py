import streamlit as st
import requests

st.set_page_config(page_title="AI Chứng Khoán", page_icon="📈", layout="centered")

st.title("📈 Hệ thống AI Dự báo Toàn Thị Trường")
st.markdown("---")

# 1. MENU CHỌN MÃ CỔ PHIẾU
# Danh sách 10 Siêu cổ phiếu đại diện cho các nhóm ngành cốt lõi
danh_sach_ma = [
    "FPT", # Công nghệ
    "HPG", # Thép / Công nghiệp nặng
    "TCB", # Ngân hàng (Techcombank)
    "VCB", # Ngân hàng (Vietcombank)
    "SSI", # Chứng khoán
    "VND", # Chứng khoán
    "MWG", # Bán lẻ (Thế Giới Di Động)
    "VNM", # Tiêu dùng (Vinamilk)
    "MSN", # Bán lẻ / Tiêu dùng (Masan)
    "VIC"  # Bất động sản (Vingroup)
]
ma_co_phieu = st.selectbox("🔍 Chọn mã cổ phiếu muốn phân tích:", danh_sach_ma)

gia_hien_tai = st.number_input(f"Nhập giá {ma_co_phieu} hiện tại trên bảng điện (VNĐ):", min_value=1000, value=68000, step=100)

# 2. NÚT BẤM CŨNG ĐỔI TÊN THEO MÃ
if st.button(f"Dự đoán giá {ma_co_phieu} sau 3 ngày (T+3)", type="primary", use_container_width=True):
    with st.spinner(f"AI đang tải dữ liệu và phân tích mã {ma_co_phieu}..."):
        try:
            # GỌI API ĐỘNG
            response = requests.get(f"http://127.0.0.1:8000/predict/{ma_co_phieu.lower()}")
            data = response.json()
            
            # Nếu API trả về thành công
            if data.get("status") == "success":
                gia_du_doan = data['predicted_price_vnd'] * 1000
                st.success("Phân tích hoàn tất!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Mã Cổ Phiếu", value=data['symbol'])
                with col2:
                    st.metric(label="Mức Giá Dự Kiến (T+3)", value=f"{gia_du_doan:,.0f} VNĐ")
                
                st.markdown("---")
                
                # Logic thuế phí 0.4%
                ty_suat_thô = ((gia_du_doan - gia_hien_tai) / gia_hien_tai) * 100
                loi_nhuan_thuc_te = ty_suat_thô - 0.4 
                
                st.subheader("Hệ thống Cố vấn Khuyến nghị:")
                if loi_nhuan_thuc_te >= 1.5:
                    st.success(f"🔥 **TÍN HIỆU: MUA ĐẸP** (Lãi ròng dự kiến: **+{loi_nhuan_thuc_te:.2f}%**)")
                elif loi_nhuan_thuc_te > 0:
                    st.warning(f"⚠️ **TÍN HIỆU: ĐỨNG NGOÀI** (Lãi quá mỏng: **+{loi_nhuan_thuc_te:.2f}%**)")
                else:
                    st.error(f"❄️ **TÍN HIỆU: KHÔNG MUA / CẮT LỖ** (Dự kiến âm: **{loi_nhuan_thuc_te:.2f}%**)")
                    
            # Nếu API báo lỗi (chưa có model)
            else:
                st.error(data.get("message"))
                
        except Exception as e:
            st.error("Không thể kết nối tới Backend. Hãy kiểm tra Terminal Uvicorn!")