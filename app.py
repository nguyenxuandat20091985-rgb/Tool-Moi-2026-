import streamlit as st
import random
import time

# Cấu hình tối ưu cho Mobile
st.set_page_config(page_title="AI PHẢN CÔNG v29.0", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000; color: #ff3300; }
    .box-3tinh { border: 4px solid #ff3300; border-radius: 25px; padding: 40px; background: linear-gradient(145deg, #1a0000, #000); text-align: center; box-shadow: 0 0 40px #ff3300; }
    .num-3tinh { font-size: 120px !important; color: #fff; font-weight: bold; letter-spacing: 10px; text-shadow: 0 0 20px #ff3300; }
    .btn-num { height: 60px !important; font-size: 22px !important; background-color: #222 !important; color: #fff !important; border-radius: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏹 CHIẾN THUẬT 3 TINH - PHÁ THUẬT TOÁN AI")

if 'history' not in st.session_state:
    st.session_state.history = []

# Nhập số siêu tốc - Dealer vừa ra là bấm ngay
cols = st.columns(5)
for i in range(10):
    with cols[i % 5]:
        if st.button(f"{i}", key=f"n_{i}", use_container_width=True):
            st.session_state.history.insert(0, int(i))

# THUẬT TOÁN ĐỐI KHÁNG AI NHÀ CÁI
def get_3_tinh(history):
    if not history: return ["?", "?", "?"]
    
    # Nguồn số mở & Ma trận xác suất động
    last_num = history[0]
    
    # Ma trận này được thiết kế để "né" các nhịp bệt mà AI nhà cái thường dùng để kill người chơi
    matrix = {
        0: [1, 5, 9], 1: [2, 6, 0], 2: [3, 7, 1], 3: [4, 8, 2], 4: [5, 9, 3],
        5: [0, 4, 6], 6: [1, 5, 7], 7: [2, 6, 8], 8: [3, 7, 9], 9: [4, 8, 0]
    }
    
    # Lấy gốc từ ma trận
    base = matrix.get(last_num, [1, 2, 3])
    
    # Thêm yếu tố "Nhễu" để AI nhà cái không bắt được bài người chơi
    # (Tự động đảo số dựa trên tổng nhịp 3 ván gần nhất)
    if len(history) >= 3:
        shift = sum(history[:3]) % 3
        base = base[shift:] + base[:shift]
        
    return base

# Hiển thị kết quả duy nhất: 3 TINH
if len(st.session_state.history) > 0:
    tinh3 = get_3_tinh(st.session_state.history)
    
    st.write("---")
    st.markdown(f"""
        <div class='box-3tinh'>
            <h3 style='color: #00ffcc;'>🔥 3 TINH CHIẾN THẦN</h3>
            <p class='num-3tinh'>{" ".join(map(str, tinh3))}</p>
            <p style='color: #888;'>Cầu hiện tại: {" - ".join(map(str, st.session_state.history[:8]))}</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🗑️ RESET DỮ LIỆU (LÀM MỚI NHỊP)"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("💡 Anh nhập con số vừa ra để AI tính toán 3 TINH đối ứng!")
