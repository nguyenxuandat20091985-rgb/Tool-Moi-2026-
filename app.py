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
    """Lọc bỏ số kỳ, chỉ giữ lại dãy 5 số kết quả"""
    return re.findall(r"\b\d{5}\b", raw_text)

def get_matrix_combo(history, n=3):
    """Thuật toán Matrix: Tìm bộ 3 số nổ cùng nhau dày nhất"""
    all_combos = []
    # Chỉ lấy 30 kỳ gần nhất để đảm bảo nhịp cầu mới
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

# Khởi tạo trạng thái
if "history" not in st.session_state:
    st.session_state.history = load_db()
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ================= UI CUSTOM v22 STYLE PRO =================
st.set_page_config(page_title="TITAN v24.6 OMNI PRO", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 2px solid #58a6ff;
        border-radius: 15px; padding: 25px; margin-top: 10px;
        box-shadow: 0 4px 25px rgba(88, 166, 255, 0.25);
    }
    .num-box {
        font-size: 95px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 15px; text-shadow: 4px 4px #000;
        line-height: 1.2;
    }
    .lot-box {
        font-size: 60px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 8px;
    }
    .status-bar { padding: 15px; border-radius: 12px; text-align: center; font-weight: 900; font-size: 1.5rem; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>💎 TITAN v24.6 - OMNI PRO</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #8b949e;'>Thời gian: {datetime.now().strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)

# ================= PHẦN 1: NHẬP LIỆU & ĐIỀU KHIỂN =================
with st.container():
    c_in, c_st = st.columns([2.2, 1])
    with c_in:
        raw_input = st.text_area("📥 Dán dữ liệu (Hỗ trợ dán cả bảng KU):", height=150, placeholder="Dán kết quả tại đây...")
    with c_st:
        st.write(f"📊 Kho dữ liệu: **{len(st.session_state.history)} kỳ**")
        btn_run = st.button("🚀 GIẢI MÃ MATRIX", use_container_width=True)
        btn_clr = st.button("🗑 RESET DATABASE", use_container_width=True)

if btn_clr:
    st.session_state.history = []
    st.session_state.last_prediction = None
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()

if btn_run:
    if raw_input:
        cleaned = clean_data(raw_input)
        if len(cleaned) >= 5:
            # Cập nhật database: Ưu tiên dữ liệu mới nhất
            st.session_state.history = cleaned + [x for x in st.session_state.history if x not in cleaned]
            save_db(st.session_state.history)
            
            # 1. Lấy Combo nổ mạnh nhất bằng Matrix
            combo_matrix = get_matrix_combo(st.session_state.history)
            
            # 2. AI Hybrid phân tích nhịp bệt
            prompt = f"""
            Hệ thống: TITAN v24.6 OMNI. 
            Dữ liệu: {st.session_state.history[:40]}
            Gợi ý Matrix: {combo_matrix}
            Nhiệm vụ:
            - Phân tích quy tắc 3 số 5 tinh (Thắng khi nổ đủ 3 số).
            - Tìm nhịp rơi đồng thời của các con số.
            - Trả về JSON: {{"main": "abc", "supp": "defg", "dec": "VÀO LỆNH/DỪNG", "log": "...", "col": "Green/Red", "conf": 99}}
            """
            try:
                res_ai = neural_engine.generate_content(prompt)
                match = re.search(r'\{.*\}', res_ai.text, re.DOTALL)
                st.session_state.last_prediction = json.loads(match.group())
            except:
                st.session_state.last_prediction = {
                    "main": combo_matrix, "supp": "0468", "dec": "ĐÁNH THEO MATRIX", 
                    "log": "Hệ thống AI bận, sử dụng thống kê Matrix Combo.", "col": "Green", "conf": 80
                }
            st.rerun()
        else:
            st.error("❌ Dữ liệu không hợp lệ! Cần ít nhất 5 dòng kết quả.")

# ================= PHẦN 2: HIỂN THỊ KẾT QUẢ PRO =================
if st.session_state.last_prediction:
    res = st.session_state.last_prediction
    
    status_color = "#238636" if res['col'].lower() == "green" else "#da3633"
    st.markdown(f"<div class='status-bar' style='background: {status_color};'>📢 LỆNH CHIẾN THUẬT: {res['dec']} ({res['conf']}%)</div>", unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    col_main, col_supp = st.columns([1.6, 1])
    with col_main:
        st.markdown("<p style='text-align:center; color:#8b949e; font-weight:bold;'>🔥 3 SỐ 5 TINH (CHỦ LỰC)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['main']}</div>", unsafe_allow_html=True)
    with col_supp:
        st.markdown("<p style='text-align:center; color:#8b949e; font-weight:bold;'>🛡️ 4 SỐ LÓT (DÀN 7)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res['supp']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.write(f"🧠 **PHÂN TÍCH THUẬT TOÁN:** {res['log']}")
    
    # Ghép dàn chuẩn 7 số
    full_dan = "".join(sorted(set(res['main'] + res['supp'])))
    st.text_input("📋 DÀN 7 SỐ KUBET (Copy tại đây):", full_dan)
    st.markdown("</div>", unsafe_allow_html=True)

# Thống kê trực quan
if st.session_state.history:
    with st.expander("📊 PHÂN TÍCH NHỊP RƠI (30 KỲ GẦN NHẤT)"):
        all_d = "".join(st.session_state.history[:30])
        st.bar_chart(pd.Series(Counter(all_d)).sort_index())
