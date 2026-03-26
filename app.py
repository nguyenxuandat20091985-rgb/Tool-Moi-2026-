import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter
from openai import OpenAI  # Dùng để gọi NVIDIA API

# ================= CẤU HÌNH HỆ THỐNG TITAN V27 =================
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_v27_permanent.json"

# Bảng bộ số hay đi cùng của anh Đạt
PAIR_RULES = [
    "178", "034", "458", "578", "019", "679", "235", "456", "124", "245", 
    "247", "248", "246", "340", "349", "348", "015", "236", "028", "026", 
    "047", "046", "056", "136", "138", "378"
]

# Khởi tạo Engines
def setup_engines():
    nv_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
    gm_model = genai.GenerativeModel('gemini-1.5-flash')
    return nv_client, gm_model

nv_ai, gm_ai = setup_engines()

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

if "db" not in st.session_state: st.session_state.db = load_db()
if "pred" not in st.session_state: st.session_state.pred = None

# ================= GIAO DIỆN CYBER V27 =================
st.set_page_config(page_title="TITAN V27 NVIDIA PRO", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #05050a; color: #ffffff; }
    .main-box { background: #0d1117; border: 2px solid #76b900; border-radius: 20px; padding: 30px; box-shadow: 0 0 20px rgba(118,185,0,0.2); }
    .big-num { font-size: 90px; font-weight: 900; color: #76b900; text-align: center; letter-spacing: 15px; text-shadow: 0 0 15px #76b900; }
    .lot-num { font-size: 45px; font-weight: 700; color: #00d4ff; text-align: center; letter-spacing: 10px; }
    .status-bar { padding: 15px; border-radius: 50px; text-align: center; font-weight: 900; margin-bottom: 20px; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #76b900;'>🚀 TITAN V27 - NVIDIA AI EDITION</h1>", unsafe_allow_html=True)

# PHẦN NHẬP LIỆU
col1, col2 = st.columns([2, 1])
with col1:
    raw_input = st.text_area("📡 DÁN DỮ LIỆU KỲ QUAY:", height=150, placeholder="32457\n83465...")
with col2:
    st.write(f"📊 Kho dữ liệu: **{len(st.session_state.db)} kỳ**")
    if st.button("⚡ CHỐT SỐ (NVIDIA AI)", use_container_width=True):
        clean_data = re.findall(r"\d{5}", raw_input)
        if clean_data:
            st.session_state.db.extend(clean_data)
            st.session_state.db = list(dict.fromkeys(st.session_state.db))[-5000:]
            with open(DB_FILE, "w") as f: json.dump(st.session_state.db, f)
            
            # --- THUẬT TOÁN DỰ PHÒNG (LOẠI 7 LẤY 3) ---
            all_str = "".join(st.session_state.db[-60:])
            scores = {str(i): Counter(all_str).get(str(i), 0) for i in range(10)}
            last_kỳ = st.session_state.db[-1]
            for rule in PAIR_RULES:
                if sum(1 for d in last_kỳ if d in rule) >= 2:
                    for digit in rule: scores[digit] += 15
            
            top_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            t3 = "".join([x[0] for x in top_res[:3]])
            s4 = "".join([x[0] for x in top_res[3:7]])

            # --- GỌI SIÊU TRÍ TUỆ NVIDIA ---
            prompt = f"Data: {st.session_state.db[-50:]}. Rules: {PAIR_RULES}. Chốt JSON: {{'main': '3 số chính', 'sub': '4 số lót', 'adv': 'ĐÁNH/DỪNG', 'logic': '...', 'conf': 98}}"
            
            try:
                # Thử NVIDIA trước
                completion = nv_ai.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2, response_format={"type": "json_object"}
                )
                st.session_state.pred = json.loads(completion.choices[0].message.content)
            except:
                try:
                    # NVIDIA lỗi thì gọi Gemini
                    res = gm_ai.generate_content(prompt)
                    st.session_state.pred = json.loads(re.search(r'\{.*\}', res.text).group())
                except:
                    # Cả 2 lỗi dùng thuật toán nội bộ
                    st.session_state.pred = {"main": t3, "sub": s4, "adv": "ĐÁNH", "logic": "Cầu thuận thuật toán lớp 8.", "conf": 85}
            st.rerun()

# HIỂN THỊ KẾT QUẢ
if st.session_state.pred:
    p = st.session_state.pred
    color = "#76b900" if p['adv'] == "ĐÁNH" else "#ff4444"
    st.markdown(f"<div class='status-bar' style='background:{color}; color:black'>KHUYÊN DÙNG: {p['adv']} | TIN CẬY: {p['conf']}%</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='main-box'>", unsafe_allow_html=True)
    c_left, c_right = st.columns([1.5, 1])
    with c_left:
        st.markdown("<p style='text-align:center; color:#888'>🔥 3 SỐ CHỦ LỰC (LOẠI 7 CHỌN 3)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-num'>{p['main']}</div>", unsafe_allow_html=True)
    with c_right:
        st.markdown("<p style='text-align:center; color:#888'>🛡️ 4 SỐ LÓT BẢO VỆ</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-num'>{p['sub']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.write(f"🧠 **LOGIC NVIDIA AI:** {p.get('logic', p.get('logic', 'Phân tích nhịp sóng bộ số.'))}")
    dan_7 = "".join(sorted(set(p['main'] + p['sub'])))
    st.text_input("📋 DÀN 7 SỐ KUBET (COPY TẠI ĐÂY):", dan_7)
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.db:
    with st.expander("📊 PHÂN TÍCH TẦN SUẤT"):
        st.bar_chart(pd.Series(Counter("".join(st.session_state.db[-100:]))).sort_index())
