import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_permanent_v24.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

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

# ================= THIẾT KẾ GIAO DIỆN v22.0 STYLE =================
st.set_page_config(page_title="TITAN v24.1 OMNI", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 12px; padding: 20px; margin-top: 20px;
    }
    .num-box {
        font-size: 70px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 10px; border-right: 2px solid #30363d;
    }
    .lot-box {
        font-size: 50px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 5px; padding-left: 20px;
    }
    .status-bar { padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🎯 TITAN v24.1 - SIÊU TRÍ TUỆ (GIAO DIỆN v22)</h2>", unsafe_allow_html=True)

# ================= PHẦN 1: NHẬP LIỆU (NẰM TRÊN) =================
with st.container():
    col_in, col_st = st.columns([2, 1])
    with col_in:
        raw_input = st.text_area("📡 Dán dữ liệu (5 số mỗi kỳ):", height=100, placeholder="32880\n21808...")
    with col_st:
        st.write(f"📊 Dữ liệu: **{len(st.session_state.history)} kỳ**")
        c1, c2 = st.columns(2)
        btn_save = c1.button("🚀 GIẢI MÃ")
        btn_reset = c2.button("🗑️ RESET")

if btn_reset:
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()

if btn_save:
    clean = re.findall(r"\d{5}", raw_input)
    if clean:
        st.session_state.history.extend(clean)
        st.session_state.history = list(dict.fromkeys(st.session_state.history))
        save_db(st.session_state.history)
        
        # Gửi AI Phân tích
        prompt = f"""
        Hệ thống: TITAN v24.1 ELITE. 
        Phân tích 100 kỳ gần đây: {st.session_state.history[-100:]}
        Nhiệm vụ: 
        1. Nhận diện cầu Bệt/Đảo. 
        2. Chốt 3 số chủ lực + 4 số lót. 
        3. Phân tích rõ 'NÊN ĐÁNH' hay 'DỪNG'.
        Trả về JSON: {{"main_3": "abc", "support_4": "defg", "decision": "ĐÁNH/DỪNG", "logic": "...", "color": "Green/Red", "conf": 98}}
        """
        try:
            response = neural_engine.generate_content(prompt)
            st.session_state.last_prediction = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
        except:
            all_n = "".join(st.session_state.history[-40:])
            top = [x[0] for x in Counter(all_n).most_common(7)]
            st.session_state.last_prediction = {"main_3": "".join(top[:3]), "support_4": "".join(top[3:]), "decision": "ĐÁNH", "logic": "Dùng thống kê tần suất.", "color": "Green", "conf": 75}
        st.rerun()

# ================= PHẦN 2: KẾT QUẢ (DÀN HÀNG NGANG - DỄ NHÌN) =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    # Thanh trạng thái Đánh/Dừng
    bg_color = "#238636" if res['color'].lower() == "green" else "#da3633"
    st.markdown(f"<div class='status-bar' style='background: {bg_color};'>📢 TRẠNG THÁI: {res['decision']} ({res['conf']}%)</div>", unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Hiển thị 3 số chính và 4 số lót trên cùng 1 hàng
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown(f"<p style='color:#8b949e; margin-bottom:0;'>🔥 3 SỐ CHÍNH (VÀO TIỀN)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['main_3']}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<p style='color:#8b949e; margin-bottom:0;'>🛡️ 4 SỐ LÓT (GIỮ VỐN)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.write(f"💡 **LOGIC AI:** {res['logic']}")
    
    # Copy dàn cho Kubet
    full_dan = "".join(sorted(set(res['main_3'] + res['support_4'])))
    st.text_input("📋 DÀN 7 SỐ KUBET (Copy tại đây):", full_dan)
    st.markdown("</div>", unsafe_allow_html=True)

# Thống kê nhịp rơi dưới cùng
if st.session_state.history:
    with st.expander("📊 Thống kê tần suất số đơn (0-9)"):
        all_d = "".join(st.session_state.history[-50:])
        st.bar_chart(pd.Series(Counter(all_d)).sort_index())



streamlit
google-generativeai
pandas
numpy
