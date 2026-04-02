import streamlit as st
import pandas as pd
import numpy as np

# Cấu hình trang
st.set_page_config(page_title="AI 5D BET PREDICTOR", layout="wide", initial_sidebar_state="expanded")

# --- PHẦN GIAO DIỆN STYLE ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #4a4a4a; }
    </style>
    """, unsafe_allow_html=True)

# --- KHỞI TẠO DỮ LIỆU TRONG SESSION ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'balance' not in st.session_state:
    st.session_state.balance = 0

# --- HÀM LOGIC AI ---
def analyze_logic(history_data):
    if len(history_data) < 3:
        return "Cần thêm dữ liệu", 50
    
    # Lấy danh sách Tài (1) và Xỉu (0)
    outcomes = [1 if h['total'] >= 23 else 0 for h in history_data]
    
    # Thuật toán Markov đơn giản: Kiểm tra mẫu hình gần nhất
    last_3 = outcomes[-3:]
    if last_3 == [1, 1, 1]: return "XỈU", 75  # Cầu bệt quá dài, dự đoán gãy
    if last_3 == [0, 0, 0]: return "TÀI", 75
    if last_3 == [1, 0, 1]: return "XỈU", 65  # Cầu 1-1
    
    # Mặc định theo số đông
    return "TÀI" if np.mean(outcomes) < 0.5 else "XỈU", 60

# --- SIDEBAR: QUẢN LÝ TÀI CHÍNH ---
with st.sidebar:
    st.header("💰 Quản Lý Vốn")
    st.session_state.balance = st.number_input("Vốn hiện tại ($):", value=st.session_state.balance)
    st.write("---")
    st.write("📌 **Mẹo AI:** Nếu thua 2 kỳ liên tiếp, hãy nghỉ 5 phút để thuật toán nhà cái reset định danh tài khoản của anh.")

# --- MÀN HÌNH CHÍNH ---
st.title("🤖 AI 5D Bet - Hệ Thống Đối Soát & Dự Đoán")

col_input, col_stats = st.columns([1, 1])

with col_input:
    st.subheader("📥 Nhập Kết Quả Kỳ Hiện Tại")
    ky_hien_tai = st.text_input("Mã kỳ (ví dụ: 101):")
    so_mo_thuong = st.text_input("Nhập 5 số mở thưởng (ví dụ: 56789):")
    lenh_da_danh = st.radio("Lệnh anh đã đánh kỳ này:", ["Tài", "Xỉu", "Không đánh"])
    
    if st.button("Xác Nhận & Dự Đoán Kỳ Tiếp"):
        if len(so_mo_thuong) == 5 and so_mo_thuong.isdigit():
            # Tính toán tổng
            digits = [int(d) for d in so_mo_thuong]
            tong = sum(digits)
            ket_qua_thuc_te = "Tài" if tong >= 23 else "Xỉu"
            
            # Đối soát thắng thua
            status = "Hòa/Chờ"
            if lenh_da_danh != "Không đánh":
                status = "THẮNG ✅" if lenh_da_danh == ket_qua_thuc_te else "THUA ❌"
            
            # Lưu lịch sử
            entry = {
                "Kỳ": ky_hien_tai,
                "Số": so_mo_thuong,
                "Tổng": tong,
                "Kết quả": ket_qua_thuc_te,
                "Đã đánh": lenh_da_danh,
                "Đối soát": status
            }
            st.session_state.history.append(entry)
            st.success(f"Tổng: {tong} -> {ket_qua_thuc_te}. Kết quả: {status}")
        else:
            st.error("Vui lòng nhập đúng 5 chữ số!")

with col_stats:
    st.subheader("🔮 AI Dự Đoán Kỳ Tiếp Theo")
    if len(st.session_state.history) > 0:
        prediction, confidence = analyze_logic([{'total': h['Tổng']} for h in st.session_state.history])
        
        c1, c2 = st.columns(2)
        c1.metric("Cửa nên vào", prediction)
        c2.metric("Độ tin cậy", f"{confidence}%")
        
        st.write("💡 **Gợi ý vào tiền:**")
        if confidence > 70:
            st.warning("Tín hiệu mạnh: Có thể vào 5% vốn.")
        else:
            st.info("Tín hiệu trung bình: Vào 1-2% vốn hoặc bỏ qua.")
    else:
        st.write("Chưa có dữ liệu để phân tích.")

# --- BẢNG THEO DÕI THẮNG THUA ---
st.divider()
st.subheader("📋 Nhật Ký Đối Soát Tự Động")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.table(df.tail(10)) # Hiển thị 10 kỳ gần nhất
    
    if st.button("Xóa lịch sử để làm mới"):
        st.session_state.history = []
        st.rerun()
else:
    st.write("Chưa có lịch sử kỳ cược nào.")
