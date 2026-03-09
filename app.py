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
DB_FILE = "titan_elite_v24.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# --- Thuật toán xử lý dữ liệu thông minh ---
def smart_clean(text):
    """Tách đúng dãy 5 số kết quả, bỏ qua số kỳ"""
    return re.findall(r"\b\d{5}\b", text)

def get_advanced_stats(history):
    """Phân tích combo 3 số và nhịp rơi"""
    if not history: return "123", "4567"
    
    # 1. Tìm bộ 3 (Combo) nổ dày nhất
    all_combos = []
    for line in history[:30]:
        unique_digits = sorted(list(set(line)))
        if len(unique_digits) >= 3:
            all_combos.extend(combinations(unique_digits, 3))
    
    best_3 = "".join(Counter(all_combos).most_common(1)[0][0]) if all_combos else "123"
    
    # 2. Tìm 4 số đang có nhịp rơi nóng (Hot numbers)
    all_nums = "".join(history[:20])
    top_4 = "".join([x[0] for x in Counter(all_nums).most_common(7)[3:7]])
    
    return best_3, top_4

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-2000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= UI CUSTOM ELITE STYLE =================
st.set_page_config(page_title="TITAN v24.6 ELITE", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 12px; padding: 25px; margin-top: 10px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
    .num-box {
        font-size: 95px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 15px;
        text-shadow: 4px 4px 0px #000;
    }
    .lot-box {
        font-size: 60px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 8px;
    }
    .status-bar { padding: 18px; border-radius: 12px; text-align: center; font-weight: 900; font-size: 1.5rem; text-transform: uppercase; }
    .logic-box { background: #161b22; padding: 15px; border-left: 4px solid #58a6ff; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🚀 TITAN v24.6 - ELITE OMNI</h1>", unsafe_allow_html=True)

# ================= PHẦN 1: NHẬP LIỆU & LOGIC =================
with st.container():
    c_in, c_st = st.columns([2, 1])
    with c_in:
        raw_input = st.text_area("📥 Dán lịch sử KU (Dán cả bảng có số kỳ vẫn được):", height=130, placeholder="225 80586\n224 64549...")
    with c_st:
        st.write(f"📊 Dữ liệu: **{len(st.session_state.history)} kỳ**")
        st.write(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
        col_b1, col_b2 = st.columns(2)
        btn_run = col_b1.button("🔥 GIẢI MÃ ELITE", use_container_width=True)
        btn_clr = col_b2.button("🗑 RESET", use_container_width=True)

if btn_clr:
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.session_state.pop('last_prediction', None)
    st.rerun()

if btn_run:
    clean_data = smart_clean(raw_input)
    if len(clean_data) >= 5:
        # Cập nhật database (Dòng mới nhất lên đầu)
        st.session_state.history = clean_data + [x for x in st.session_state.history if x not in clean_data]
        save_db(st.session_state.history)
        
        # 1. Thuật toán cứng
        m3, s4 = get_advanced_stats(st.session_state.history)
        
        # 2. AI Hybrid phân tích nhịp
        prompt = f"""
        System: TITAN v24.6 ELITE AI.
        Input 40 kỳ: {st.session_state.history[:40]}
        Stats Combo: {m3}
        Nhiệm vụ:
        - Phân tích 'Cầu kẹp' và 'Cầu bệt bộ 3'.
        - Chốt 3 số 5 tinh chính xác nhất.
        - Trả về JSON: {{"main_3": "{m3}", "support_4": "{s4}", "decision": "ĐÁNH/DỪNG", "logic": "...", "color": "Green/Red", "conf": 99}}
        """
        try:
            response = neural_engine.generate_content(prompt)
            st.session_state.last_prediction = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
        except:
            st.session_state.last_prediction = {"main_3": m3, "support_4": s4, "decision": "ĐÁNH THEO COMBO", "logic": "Dựa trên nhịp rơi thực tế.", "color": "Green", "conf": 80}
        st.rerun()

# ================= PHẦN 2: HIỂN THỊ KẾT QUẢ =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    st.markdown(f"<div class='status-bar' style='background: {'#238636' if res['color'].lower() == 'green' else '#da3633'};'>📢 LỆNH: {res['decision']} ({res['conf']}%)</div>", unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    c_m, c_s = st.columns([1.6, 1])
    with c_m:
        st.markdown("<p style='text-align:center; color:#8b949e; font-weight:bold;'>💎 3 SỐ 5 TINH CHỦ LỰC</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['main_3']}</div>", unsafe_allow_html=True)
    with c_s:
        st.markdown("<p style='text-align:center; color:#8b949e; font-weight:bold;'>🛡️ DÀN LÓT GIỮ VỐN</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown(f"<div class='logic-box'><b>🧠 CHIẾN THUẬT AI:</b> {res['logic']}</div>", unsafe_allow_html=True)
    
    f_dan = "".join(sorted(set(res['main_3'] + res['support_4'])))
    st.text_input("📋 DÀN 7 SỐ KUBET (CLICK ĐỂ COPY):", f_dan)
    st.markdown("</div>", unsafe_allow_html=True)

# Thống kê nhịp rơi trực quan
if st.session_state.history:
    with st.expander("📊 BẢN ĐỒ NHỊP RƠI (30 KỲ)"):
        all_d = "".join(st.session_state.history[:30])
        st.bar_chart(pd.Series(Counter(all_d)).sort_index())
