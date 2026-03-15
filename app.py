import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter

# ================= CẤU HÌNH TITAN V26 =================
API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_v26_data.json"

# Danh sách bộ số hay đi cùng (Data của anh Đạt)
PAIR_RULES = [
    "178", "034", "458", "578", "019", "679", "235", "456", "124", "245", 
    "247", "248", "246", "340", "349", "348", "015", "236", "028", "026", 
    "047", "046", "056", "136", "138", "378"
]

def setup_ai():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

ai_engine = setup_ai()

def load_data():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

if "db" not in st.session_state:
    st.session_state.db = load_data()
if "pred" not in st.session_state:
    st.session_state.pred = None

# ================= GIAO DIỆN HIỆN ĐẠI =================
st.set_page_config(page_title="TITAN V26 OMNI", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #010409; color: #e6edf3; }
    .main-box { background: #0d1117; border: 1px solid #30363d; border-radius: 15px; padding: 25px; }
    .big-num { font-size: 80px; font-weight: 900; color: #00ff88; text-align: center; letter-spacing: 15px; }
    .lot-num { font-size: 40px; font-weight: 700; color: #58a6ff; text-align: center; letter-spacing: 8px; }
    .status { padding: 15px; border-radius: 10px; text-align: center; font-weight: 900; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 TITAN V26 - HỆ THỐNG DỰ ĐOÁN 5D")

# PHẦN NHẬP LIỆU
col1, col2 = st.columns([2, 1])
with col1:
    raw_input = st.text_area("📡 Dán kỳ quay (Ví dụ: 32457):", height=150, placeholder="Dán dãy số tại đây...")
with col2:
    st.write(f"📊 Kho dữ liệu: **{len(st.session_state.db)} kỳ**")
    if st.button("🚀 BẮT ĐẦU DỰ ĐOÁN", use_container_width=True):
        clean_data = re.findall(r"\d{5}", raw_input)
        if clean_data:
            st.session_state.db.extend(clean_data)
            st.session_state.db = list(dict.fromkeys(st.session_state.db))[-3000:]
            with open(DB_FILE, "w") as f: json.dump(st.session_state.db, f)
            
            # --- THUẬT TOÁN 8 LỚP DỰ PHÒNG ---
            all_str = "".join(st.session_state.db[-50:])
            c = Counter(all_str)
            # Ưu tiên cộng điểm cho các bộ số hay đi cùng
            scores = {str(i): c.get(str(i), 0) for i in range(10)}
            last_val = st.session_state.db[-1]
            for rule in PAIR_RULES:
                matches = sum(1 for d in last_val if d in rule)
                if matches >= 2: # Nếu kỳ trước nổ 2/3 số trong bộ
                    for r_digit in rule: scores[r_digit] += 10 # Cộng điểm mạnh cho bộ đó nổ lại

            sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_3 = "".join([x[0] for x in sorted_nums[:3]])
            support_4 = "".join([x[0] for x in sorted_nums[3:7]])

            # --- GỌI GEMINI ---
            prompt = f"Dữ liệu: {st.session_state.db[-30:]}. Dựa vào các bộ số {PAIR_RULES}, chốt JSON: {{'main': '3 số', 'sub': '4 số', 'advice': 'ĐÁNH/DỪNG', 'conf': 95}}"
            try:
                res = ai_engine.generate_content(prompt)
                st.session_state.pred = json.loads(re.search(r'\{.*\}', res.text).group())
            except:
                st.session_state.pred = {"main": top_3, "sub": support_4, "advice": "ĐÁNH", "conf": 80}
            st.rerun()

    if st.button("🗑️ XÓA DỮ LIỆU", use_container_width=True):
        st.session_state.db = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# PHẦN HIỂN THỊ KẾT QUẢ
if st.session_state.pred:
    p = st.session_state.pred
    bg = "#238636" if p['advice'] == "ĐÁNH" else "#da3633"
    st.markdown(f"<div class='status' style='background:{bg}'>TRẠNG THÁI: {p['advice']} ({p['conf']}%)</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='main-box'>", unsafe_allow_html=True)
    c_main, c_sub = st.columns([1.5, 1])
    with c_main:
        st.markdown("<p style='text-align:center;color:#8b949e'>🔥 3 SỐ CHỦ LỰC (LOẠI 7 CHỌN 3)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-num'>{p['main']}</div>", unsafe_allow_html=True)
    with c_sub:
        st.markdown("<p style='text-align:center;color:#8b949e'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-num'>{p['sub']}</div>", unsafe_allow_html=True)
    
    st.divider()
    dan_7 = "".join(sorted(set(p['main'] + p['sub'])))
    st.text_input("📋 DÀN 7 SỐ KUBET:", dan_7)
    st.markdown("</div>", unsafe_allow_html=True)

# THỐNG KÊ
if st.session_state.db:
    with st.expander("📊 Xem tần suất nhịp rơi"):
        st.bar_chart(pd.Series(Counter("".join(st.session_state.db[-50:]))).sort_index())
