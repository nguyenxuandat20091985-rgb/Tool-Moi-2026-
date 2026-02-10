import streamlit as st

st.set_page_config(page_title="VÉT SÀN v30.0", layout="wide")

# Giao diện tối giản hết mức để load cực nhanh
st.markdown("""
    <style>
    .stApp { background-color: #000; }
    .btn-big { height: 80px !important; font-size: 30px !important; font-weight: bold !important; border: 2px solid #ff0000 !important; }
    .res-box { border: 5px solid #00ffcc; border-radius: 20px; padding: 20px; text-align: center; margin-top: 10px; background: #111; }
    .tinh-3 { font-size: 90px !important; color: #ffff00; font-weight: bold; text-shadow: 0 0 15px #ff0000; }
    </style>
    """, unsafe_allow_html=True)

if 'last' not in st.session_state: st.session_state.last = "?"

st.title("🏹 3 TINH SIÊU TỐC (1 CHẠM)")

# Ma trận phản công nhanh (Đã nạp sẵn nguồn số mở)
matrix = {
    0: "1-5-8", 1: "2-6-9", 2: "0-3-7", 3: "1-4-8", 4: "0-5-9",
    5: "0-4-6", 6: "1-5-7", 7: "2-8-0", 8: "3-7-9", 9: "4-1-0"
}

# 10 nút bấm - Anh chỉ cần chạm vào số vừa ra
st.write("---")
c1, c2, c3, c4, c5 = st.columns(5)
for i in range(10):
    column = [c1, c2, c3, c4, c5][i % 5]
    if column.button(f"{i}", key=f"n_{i}", use_container_width=True):
        st.session_state.last = matrix.get(i)

# HIỆN KẾT QUẢ NGAY LẬP TỨC
st.markdown(f"""
    <div class='res-box'>
        <h2 style='color: #00ffcc; margin:0;'>VÀO TIỀN 3 TINH:</h2>
        <p class='tinh-3'>{st.session_state.last}</p>
    </div>
""", unsafe_allow_html=True)

st.button("🗑️ RESET", on_click=lambda: st.session_state.update(last="?"))
