import streamlit as st

# Cấu hình giao diện thực chiến
st.set_page_config(page_title="3 TINH SIÊU CẤP v25.0", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #ffffff; }
    /* Nút số khổng lồ để bấm nhanh */
    div.stButton > button {
        height: 85px !important;
        font-size: 35px !important;
        font-weight: bold !important;
        background-color: #1a1a1a !important;
        color: #00ffcc !important;
        border: 2px solid #333 !important;
        border-radius: 12px !important;
    }
    div.stButton > button:hover { border-color: #00ffcc !important; color: #fff !important; }
    div.stButton > button:active { background-color: #ff0000 !important; }
    
    /* Hộp kết quả 3 TINH */
    .result-container {
        border: 5px solid #ffcc00;
        border-radius: 25px;
        padding: 30px;
        text-align: center;
        background: linear-gradient(145deg, #0f0f0f, #222);
        box-shadow: 0 0 30px rgba(255, 204, 0, 0.4);
        margin-top: 20px;
    }
    .label-3tinh { font-size: 28px; color: #ffcc00; font-weight: bold; text-transform: uppercase; }
    .number-3tinh { font-size: 130px !important; color: #ffffff; font-weight: bold; letter-spacing: 15px; text-shadow: 0 0 20px #ffcc00; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏹 CHIẾN THUẬT 3 TINH - PHẢN CÔNG AI")

if 'kq' not in st.session_state: st.session_state.kq = "- - -"

# Thuật toán ma trận nhịp nhảy (Né quét ID nhà cái)
matrix = {
    0: "1 5 8", 1: "2 6 9", 2: "0 3 7", 3: "1 4 8", 4: "0 5 9",
    5: "0 4 6", 6: "1 5 7", 7: "2 8 0", 8: "3 7 9", 9: "4 1 0"
}

st.subheader("📡 Dealer vừa ra số mấy? Bấm ngay:")

# Chia 10 nút thành 2 hàng cho dễ bấm trên điện thoại
row1 = st.columns(5)
for i in range(5):
    if row1[i].button(str(i)): st.session_state.kq = matrix[i]

row2 = st.columns(5)
for i in range(5, 10):
    if row2[i-5].button(str(i)): st.session_state.kq = matrix[i]

# VÙNG HIỂN THỊ DUY NHẤT: 3 TINH
st.markdown(f"""
    <div class='result-container'>
        <p class='label-3tinh'>🎯 DÀN 3 TINH TAY SAU</p>
        <p class='number-3tinh'>{st.session_state.kq}</p>
        <p style='color: #00ffcc; font-size: 18px;'>⚠️ Đánh đều tay - Không bẻ cầu khi đang thông</p>
    </div>
""", unsafe_allow_html=True)

if st.button("🗑️ RESET (LÀM MỚI NHỊP)"):
    st.session_state.kq = "- - -"
    st.rerun()
