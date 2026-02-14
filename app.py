import streamlit as st
import google.generativeai as genai
import re
import json
import os
import numpy as np
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_quantum_v22.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

model = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU ĐA CHIỀU =================
def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: return json.load(f)
    return []

def save_db(data):
    with open(DB_FILE, "w") as f: json.dump(data[-1500:], f)

if "db" not in st.session_state:
    st.session_state.db = load_db()

# ================= UI LUXURY - CHỐNG SAI SỐ =================
st.set_page_config(page_title="TITAN v22.0 QUANTUM", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #00050a; color: #00d4ff; }
    .status-ok { color: #00ff88; font-weight: bold; font-size: 13px; text-shadow: 0 0 5px #00ff88; }
    .main-card {
        background: rgba(0, 20, 40, 0.8); border: 1px solid #00d4ff;
        border-radius: 20px; padding: 30px; margin-top: 10px;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.2);
    }
    .num-main { 
        font-size: 75px; font-weight: 900; color: #ffffff; 
        text-align: center; letter-spacing: 15px; 
        background: linear-gradient(to bottom, #ffffff, #00d4ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .num-sub { font-size: 45px; font-weight: 700; color: #ff8800; text-align: center; letter-spacing: 8px; opacity: 0.8; }
    .logic-box { background: #001a33; border-left: 4px solid #00d4ff; padding: 15px; border-radius: 8px; margin-bottom: 25px; font-size: 14px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>💠 TITAN v22.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>HỆ THỐNG DỰ ĐOÁN Đón Đầu (Quantum Prediction)</p>", unsafe_allow_html=True)

if model:
    st.markdown(f"<p class='status-ok'>● KẾT NỐI QUANTUM CORE: SẴN SÀNG | DỮ LIỆU: {len(st.session_state.db)} KỲ</p>", unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU & AI =================
raw_data = st.text_area("📡 NẠP DỮ LIỆU (Copy chuỗi kỳ):", height=100)

col1, col2 = st.columns(2)
with col1:
    if st.button("🌀 GIẢI MÃ CẦU"):
        ky_vua_ve = re.findall(r"\d{5}", raw_data)
        if ky_vua_ve:
            st.session_state.db.extend(ky_vua_ve)
            save_db(st.session_state.db)
            
            # PROMPT ÉP AI BẮT CẦU HỒI (CHỐNG CHẾT CHỦ LỰC)
            prompt = f"""
            Bạn là siêu máy tính Quantum phân tích 5D. 
            Dữ liệu gần đây: {st.session_state.db[-60:]}.
            Yêu cầu:
            1. Bỏ qua tần suất đơn giản. Hãy tìm quy luật "Hồi số" (Số sắp nổ sau chuỗi gan).
            2. Phân tích nhịp nhảy của nhà cái (ví dụ: đang bệt thì sắp gãy chưa?).
            3. Chốt 4 số chủ lực đón đầu và 3 số lót giữ vốn.
            TRẢ VỀ JSON: {{"main": [], "sub": [], "tu_duy": "giải thích nhịp cầu ngắn gọn"}}
            """
            try:
                response = model.generate_content(prompt)
                data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                st.session_state.quantum_res = data
            except:
                # Thuật toán dự phòng (Quantum Fallback)
                all_nums = "".join(st.session_state.db[-40:])
                counts = Counter(all_nums).most_common(7)
                res = [str(x[0]) for x in counts]
                st.session_state.quantum_res = {"main": res[:4], "sub": res[4:], "tu_duy": "Dùng thuật toán xác suất hồi số."}
            st.rerun()

with col2:
    if st.button("🗑️ RESET"):
        st.session_state.db = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if "quantum_res" in st.session_state:
    res = st.session_state.quantum_res
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='logic-box'><b>💎 Nhịp cầu:</b> {res['tu_duy']}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; color:#00d4ff; font-weight:bold;'>🎯 4 CHỦ LỰC (ĐÓN ĐẦU)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-main'>{''.join(map(str, res['main']))}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; color:#ff8800; font-weight:bold; margin-top:30px;'>🛡️ 3 LÓT (GIỮ VỐN)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-sub'>{''.join(map(str, res['sub']))}</div>", unsafe_allow_html=True)
    
    copy_str = "".join(map(str, res['main'])) + "".join(map(str, res['sub']))
    st.text_input("📋 COPY DÀN 7 SỐ:", copy_str)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><p style='text-align:center; font-size:10px; color:#333;'>Quantum Core - Giải mã mọi thuật toán nhà cái</p>", unsafe_allow_html=True)
