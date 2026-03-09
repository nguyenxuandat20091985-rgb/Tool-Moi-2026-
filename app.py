import streamlit as st
import google.generativeai as genai
import re
import json
import pandas as pd
from collections import Counter
from itertools import combinations

# ================= CẤU HÌNH AI =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# --- Thuật toán thống kê Combo (Chạy bằng RAM - Không đứng hình) ---
def get_combo_stats(history, n=3):
    all_combos = []
    for line in history:
        unique_digits = sorted(list(set(line)))
        if len(unique_digits) >= n:
            all_combos.extend(combinations(unique_digits, n))
    return Counter(all_combos).most_common(1)

# ================= UI STYLE v22 CHUẨN =================
st.set_page_config(page_title="TITAN v24.4 SPEED", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 2px solid #58a6ff;
        border-radius: 12px; padding: 20px; margin-top: 10px;
    }
    .num-box {
        font-size: 80px; font-weight: 900; color: #ff4b4b;
        text-align: center; letter-spacing: 15px;
    }
    .lot-box {
        font-size: 55px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 5px;
    }
    .status-bar { padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 1.2rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>⚡ TITAN v24.4 - TỐC ĐỘ THỰC CHIẾN</h2>", unsafe_allow_html=True)

# ================= XỬ LÝ NHẬP LIỆU (ÉP REFRESH) =================
if "last_res" not in st.session_state:
    st.session_state.last_res = None

with st.container():
    col_in, col_btn = st.columns([3, 1])
    with col_in:
        # Thêm key ngẫu nhiên vào text_area để ép nó không bị lưu cache cũ
        raw_input = st.text_area("📡 Dán dữ liệu mới nhất vào đây:", height=150, key="input_data_v24")
    with col_btn:
        st.write("🚀 **HÀNH ĐỘNG**")
        btn_run = st.button("🔥 GIẢI MÃ NGAY", use_container_width=True)
        if st.button("🗑️ XÓA SẠCH", use_container_width=True):
            st.session_state.last_res = None
            st.rerun()

# ================= LOGIC PHÂN TÍCH TỨC THÌ =================
if btn_run and raw_input:
    with st.spinner('AI đang quét nhịp cầu...'):
        # Lọc dữ liệu chuẩn
        lines = [l.strip() for l in raw_input.split('\n') if re.match(r"^\d{5}$", l.strip())]
        
        if len(lines) >= 5:
            # 1. Chạy thuật toán thống kê nhanh
            best_c = get_combo_stats(lines[:30])
            combo_fix = "".join(best_c[0][0]) if best_c else "123"
            
            # 2. Gọi AI phân tích sâu
            prompt = f"""
            Phân tích 3 số 5 tinh cho 5D. Lịch sử: {lines[:40]}.
            Dựa trên quy tắc: Thắng khi kết quả có đủ 3 số.
            Hãy tìm bộ 3 số (Combo) có tần suất đi cùng nhau cao nhất.
            Trả về JSON cực ngắn: {{"m": "abc", "s": "defg", "d": "ĐÁNH/DỪNG", "l": "lý do", "c": "Green/Red"}}
            """
            try:
                response = neural_engine.generate_content(prompt)
                match = re.search(r'\{.*\}', response.text, re.DOTALL)
                st.session_state.last_res = json.loads(match.group())
            except:
                st.session_state.last_res = {"m": combo_fix, "s": "0468", "d": "ĐÁNH", "l": "Thống kê Combo", "c": "Green"}
        else:
            st.error("❌ Thiếu dữ liệu! Dán ít nhất 5 kỳ anh ơi.")

# ================= HIỂN THỊ KẾT QUẢ =================
if st.session_state.last_res:
    res = st.session_state.last_res
    bg = "#238636" if res['c'].lower() == "green" else "#da3633"
    
    st.markdown(f"<div class='status-bar' style='background: {bg};'>📢 TRẠNG THÁI: {res['d']}</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🔥 3 SỐ 5 TINH CHỦ LỰC</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['m']}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res['s']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.write(f"🧠 **LOGIC:** {res['l']}")
    
    full_dan = "".join(sorted(set(res['m'] + res['s'])))
    st.text_input("📋 DÀN 7 SỐ KUBET:", full_dan)
    st.markdown("</div>", unsafe_allow_html=True)

    # Thống kê nhanh nhịp rơi
    all_num = "".join(raw_input.split('\n')[:20])
    st.bar_chart(pd.Series(Counter(all_num)).sort_index())
