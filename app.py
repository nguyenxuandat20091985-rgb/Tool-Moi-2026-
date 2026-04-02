import streamlit as st
import pandas as pd

# Thiết lập giao diện
st.set_page_config(page_title="5D Bet Master - Đối Soát & Soi Cầu", layout="wide")

# CSS tùy chỉnh để dễ nhìn trên điện thoại
st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .win { color: #2ecc71; }
    .loss { color: #e74c3c; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏆 5D BET - HỆ THỐNG ĐỐI SOÁT & DỰ ĐOÁN")

# --- PHẦN 1: NHẬP DỮ LIỆU & ĐỐI SOÁT ---
st.header("🔍 Đối Soát Kết Quả")
with st.expander("Nhấn để nhập kỳ vừa mở", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        ky_id = st.text_input("Mã kỳ (VD: 123):")
        so_mo = st.text_input("Nhập 5 số mở thưởng (VD: 56789):")
    with col2:
        cuoc_cua = st.selectbox("Anh đã đặt cửa nào?", ["Chưa đặt", "Tài", "Xỉu"])
        tien_cuoc = st.number_input("Số tiền cược (k):", min_value=0, value=0)

if st.button("✅ KIỂM TRA & LƯU LỊCH SỬ"):
    if len(so_mo) == 5 and so_mo.isdigit():
        digits = [int(d) for d in so_mo]
        tong = sum(digits)
        ket_qua = "Tài" if tong >= 23 else "Xỉu"
        
        st.subheader(f"Kết quả Tổng: {tong} ➡ {ket_qua.upper()}")
        
        # Kiểm tra thắng thua
        if cuoc_cua != "Chưa đặt":
            if cuoc_cua == ket_qua:
                st.balloons()
                st.success(f"CHÚC MỪNG! Anh thắng {tien_cuoc * 1.95}k")
            else:
                st.error(f"RẤT TIẾC! Kỳ này anh chưa may mắn.")
        
        # Lưu vào bộ nhớ tạm (Session State)
        if 'history' not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({"Kỳ": ky_id, "Số": so_mo, "Tổng": tong, "Kết Quả": ket_qua})
    else:
        st.warning("Vui lòng nhập đúng 5 chữ số!")

# --- PHẦN 2: LỊCH SỬ & SOI CẦU ---
if 'history' in st.session_state and len(st.session_state.history) > 0:
    st.divider()
    st.header("📊 Phân Tích Soi Cầu")
    df = pd.DataFrame(st.session_state.history)
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.write("Lịch sử 10 kỳ gần nhất:")
        st.table(df.tail(10))
    
    with col_r:
        # Thuật toán soi cầu đơn giản
        last_results = df['Kết Quả'].tail(5).tolist()
        tai_count = last_results.count("Tài")
        xiu_count = last_results.count("Xỉu")
        
        st.write("**Thống kê 5 kỳ cuối:**")
        st.write(f"- Tài: {tai_count}")
        st.write(f"- Xỉu: {xiu_count}")
        
        # Gợi ý lệnh tiếp theo
        st.subheader("🔮 GỢI Ý KỲ TỚI")
        if tai_count >= 4:
            st.warning("Cầu bệt TÀI - Cân nhắc theo bệt hoặc dừng.")
            st.button("VÀO LỆNH: TÀI", b_type="primary")
        elif xiu_count >= 4:
            st.warning("Cầu bệt XỈU - Cân nhắc theo bệt hoặc dừng.")
            st.button("VÀO LỆNH: XỈU", b_type="primary")
        else:
            st.info("Cầu đang biến động - Đánh nhẹ tay.")

# --- PHẦN 3: QUẢN LÝ VỐN (NÂNG CẤP ĐÁNG GIÁ) ---
st.divider()
st.header("💰 Quản Lý Vốn Smart-Bet")
vốn = st.number_input("Tổng vốn hiện tại (k):", value=1000)
st.write("Bảng đi tiền Gấp thếp (nếu anh muốn gỡ):")
steps = [vốn*0.01, vốn*0.02, vốn*0.04, vốn*0.08, vốn*0.16]
st.write(f"Lệnh 1: {steps[0]:.1f}k | Lệnh 2: {steps[1]:.1f}k | Lệnh 3: {steps[2]:.1f}k | Lệnh 4: {steps[3]:.1f}k")
st.caption("Lời khuyên: Nếu thua đến Lệnh 4 mà chưa về, hãy DỪNG LẠI. Đừng cháy túi vì một dây bệt!")

if st.button("🗑 XÓA DỮ LIỆU LÀM MỚI"):
    st.session_state.history = []
    st.rerun()
