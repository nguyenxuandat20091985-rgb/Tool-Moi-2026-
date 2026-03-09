import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter
from itertools import combinations
from datetime import datetime

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_permanent_pro.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# --- Thuật toán xử lý dữ liệu thông minh ---
def clean_data(raw_text):
    # Regex cực mạnh: Chỉ lấy dãy đúng 5 số, bỏ qua số kỳ, bỏ qua chữ
    return re.findall(r"\b\d{5}\b", raw_text)

def get_matrix_combo(history, n=3):
    all_combos = []
    # Lấy 30 kỳ gần nhất để soi combo nổ cùng nhau
    for line in history[:30]:
        unique_digits = sorted(list(set(line)))
        if len(unique_digits) >= n:
            all_combos.extend(combinations(unique_digits, n))
    top = Counter(all_combos).most_common(1)
    return "".join(top[0][0]) if top else "123"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-3000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ================= GIAO DIỆN v22 STYLE PRO =================
st.set_page_config(page_title="TITAN v24.7 PRO", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 2px solid #58a6ff;
        border-radius: 15px; padding: 25px; margin-top: 10px;
        box-shadow: 0 4px 25px rgba(88, 166, 255, 0.25);
    }
    .num-box {
        font-size: 100px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 15px; text-shadow: 4px 4px #000;
    }
    .lot-box {
        font-size: 65px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 8px;
    }
    .status-bar { padding: 15px; border-radius: 12px; text-align: center; font-weight: 900; font-size: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>💎 TITAN v24.7 - SIÊU TRÍ TUỆ</h1>", unsafe_allow_html=True)

# ================= PHẦN 1: NHẬP LIỆU =================
with st.container():
    c_in, c_st = st.columns([2.5, 1])
    with c_in:
        raw_input = st.text_area("📥 Dán dữ liệu (Dán thẳng từ KU, tool tự lọc số kỳ):", height=150)
    with c_st:
        st.write(f"📊 Database: **{len(st.session_state.history)} kỳ**")
        btn_run = st.button("🚀 GIẢI MÃ NGAY", use_container_width=True)
        if st.button("🗑 RESET", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_prediction = None
            if os.path.exists(DB_FILE): os.remove(DB_FILE)
            st.rerun()

if btn_run and raw_input:
    cleaned = clean_data(raw_input)
    if len(cleaned) >= 5:
        st.session_state.history = cleaned + [x for x in st.session_state.history if x not in cleaned]
        save_db(st.session_state.history)
        
        # 1. Thuật toán Matrix dự phòng (Luôn chạy được)
        combo_matrix = get_matrix_combo(st.session_state.history)
        all_nums = "".join(st.session_state.history[:20])
        top_7 = [x[0] for x in Counter(all_nums).most_common(7)]
        supp_default = "".join(top_7[3:]) if len(top_7) > 3 else "0468"

        # 2. AI Hybrid
        prompt = f"Phân tích 3 số 5 tinh. Dữ liệu: {st.session_state.history[:40]}. Trả JSON: {{'main': 'abc', 'supp': 'defg', 'dec': 'ĐÁNH/DỪNG', 'log': '...', 'col': 'Green'}}"
        
        try:
            res_ai = neural_engine.generate_content(prompt)
            match = re.search(r'\{.*\}', res_ai.text, re.DOTALL)
            st.session_state.last_prediction = json.loads(match.group())
        except:
            # CHẾ ĐỘ DỰ PHÒNG KHI AI LỖI (Sửa lỗi báo đỏ)
            st.session_state.last_prediction = {
                "main": combo_matrix, "supp": supp_default, 
                "dec": "CHẾ ĐỘ DỰ PHÒNG", "log": "Dữ liệu Matrix Combo nổ mạnh nhất.", 
                "col": "Green"
            }
        st.rerun()

# ================= PHẦN 2: HIỂN THỊ =================
if st.session_state.last_prediction:
    res = st.session_state.last_prediction
    status_color = "#238636" if res['col'].lower() == "green" else "#da3633"
    
    st.markdown(f"<div class='status-bar' style='background: {status_color};'>📢 TRẠNG THÁI: {res['dec']}</div>", unsafe_allow_html=True)
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🔥 3 SỐ CHỦ LỰC (3 SỐ 5 TINH)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['main']}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🛡️ 4 SỐ LÓT (DÀN 7)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res['supp']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.write(f"🧠 **LOGIC:** {res['log']}")
    full_dan = "".join(sorted(set(res['main'] + res['supp'])))
    st.text_input("📋 DÀN 7 SỐ KUBET (Copy):", full_dan)
    st.markdown("</div>", unsafe_allow_html=True)

# Biểu đồ
if st.session_state.history:
    with st.expander("📊 Phân tích nhịp rơi"):
        st.bar_chart(pd.Series(Counter("".join(st.session_state.history[:25]))).sort_index())
