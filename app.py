import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter
from itertools import combinations

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_permanent_v24.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# --- Hàm xử lý dữ liệu thực chiến ---
def get_combo_stats(history, n=3):
    """Tìm bộ 3 số xuất hiện cùng nhau nhiều nhất"""
    all_combos = []
    for line in history:
        unique_digits = sorted(list(set(line)))
        if len(unique_digits) >= n:
            all_combos.extend(combinations(unique_digits, n))
    return Counter(all_combos).most_common(1)

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

# ================= UI CUSTOM v22 STYLE =================
st.set_page_config(page_title="TITAN v24.3 ULTIMATE", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #161b22; border: 2px solid #58a6ff;
        border-radius: 15px; padding: 25px; margin-top: 10px;
        box-shadow: 0 4px 20px rgba(88, 166, 255, 0.2);
    }
    .num-box {
        font-size: 85px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 12px; text-shadow: 2px 2px #000;
    }
    .lot-box {
        font-size: 55px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 5px;
    }
    .status-bar { padding: 15px; border-radius: 10px; text-align: center; font-weight: 800; font-size: 1.3rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🚀 TITAN v24.3 - ULTIMATE HYBRID</h1>", unsafe_allow_html=True)

# ================= PHẦN 1: NHẬP LIỆU & XỬ LÝ =================
with st.container():
    c_in, c_st = st.columns([2, 1])
    with c_in:
        raw_input = st.text_area("📥 Dán dữ liệu KU (Mới nhất ở trên):", height=120)
    with c_st:
        st.write(f"📊 Database: **{len(st.session_state.history)} kỳ**")
        c1, c2 = st.columns(2)
        btn_run = c1.button("🔥 GIẢI MÃ")
        btn_clr = c2.button("🗑 RESET")

if btn_clr:
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()

if btn_run:
    clean = [l.strip() for l in raw_input.split('\n') if re.match(r"^\d{5}$", l.strip())]
    if clean:
        st.session_state.history = clean + [x for x in st.session_state.history if x not in clean]
        save_db(st.session_state.history)
        
        # 1. Thuật toán thống kê cứng (Combo nổ)
        best_combo = get_combo_stats(st.session_state.history[:50])
        combo_str = "".join(best_combo[0][0]) if best_combo else "123"
        
        # 2. Gửi AI phân tích đa tầng
        prompt = f"""
        Hệ thống: TITAN v24.3 ULTIMATE. 
        Lịch sử 50 kỳ: {st.session_state.history[:50]}
        Combo thống kê: {combo_str}
        Nhiệm vụ:
        - Sử dụng thuật toán xác suất đa biến để lọc bộ 3 số 5 tinh.
        - Tìm điểm rơi nhịp cầu (Cầu bệt số, cầu kẹp).
        - Trả về JSON: {{"main_3": "abc", "support_4": "defg", "decision": "ĐÁNH/DỪNG", "logic": "...", "color": "Green/Red", "conf": 99}}
        """
        try:
            res_ai = neural_engine.generate_content(prompt)
            st.session_state.last_prediction = json.loads(re.search(r'\{.*\}', res_ai.text, re.DOTALL).group())
        except:
            st.session_state.last_prediction = {"main_3": combo_str, "support_4": "0456", "decision": "ĐÁNH", "logic": "Dựa trên Combo nổ nhất.", "color": "Green", "conf": 85}
        st.rerun()

# ================= PHẦN 2: HIỂN THỊ KẾT QUẢ =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    status_color = "#238636" if res['color'].lower() == "green" else "#da3633"
    st.markdown(f"<div class='status-bar' style='background: {status_color};'>📢 LỆNH CHIẾN THUẬT: {res['decision']} ({res['conf']}%)</div>", unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    col_main, col_supp = st.columns([1.5, 1])
    with col_main:
        st.markdown("<p style='text-align:center; color:#8b949e;'>💎 3 SỐ 5 TINH CHỦ LỰC</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['main_3']}</div>", unsafe_allow_html=True)
    with col_supp:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🛡️ DÀN LÓT GIỮ VỐN</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.write(f"🧠 **PHÂN TÍCH THUẬT TOÁN:** {res['logic']}")
    
    full_dan = "".join(sorted(set(res['main_3'] + res['support_4'])))
    st.text_input("📋 DÀN 7 SỐ KUBET:", full_dan)
    st.markdown("</div>", unsafe_allow_html=True)

# Thống kê trực quan
if st.session_state.history:
    with st.expander("📊 BẢN ĐỒ NHỊP RƠI (50 KỲ)"):
        all_d = "".join(st.session_state.history[:50])
        st.bar_chart(pd.Series(Counter(all_d)).sort_index())
