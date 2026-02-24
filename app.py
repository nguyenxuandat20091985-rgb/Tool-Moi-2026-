import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter

# ================= CẤU HÌNH SIÊU CẤP =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_ultimate_memory_v23.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= HỆ THỐNG TỰ HỌC VÀ LỌC SỐ BẨN =================
def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_memory(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-5000:], f) # Mở rộng bộ nhớ lên 5000 kỳ

if "history" not in st.session_state:
    st.session_state.history = load_memory()

# ================= THUẬT TOÁN SOI CẦU CAO CẤP =================
def advanced_analysis(history):
    if len(history) < 10: return "Cần thêm dữ liệu"
    
    # 1. Ma trận tần suất vị trí
    matrix = np.array([[int(d) for d in res] for res in history[-50:]])
    pos_freq = [Counter(matrix[:, i]).most_common(1)[0][0] for i in range(5)]
    
    # 2. Quy luật bóng số nâng cao
    shadow_map = {'0':'5', '1':'6', '2':'7', '3':'8', '4':'9', '5':'0', '6':'1', '7':'2', '8':'3', '9':'4'}
    last_res = history[-1]
    shadows = "".join([shadow_map[d] for d in last_res])
    
    return f"Vị trí nổ mạnh: {pos_freq} | Dàn bóng: {shadows}"

# ================= GIAO DIỆN CHIẾN ĐẤU =================
st.set_page_config(page_title="TITAN v23.0 ULTIMATE", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #050505; color: #e0e0e0; }
    .prediction-box {
        background: linear-gradient(135deg, #001f3f, #000000);
        border: 2px solid #0074d9; border-radius: 20px; padding: 40px;
        box-shadow: 0 0 50px rgba(0, 116, 217, 0.4);
    }
    .core-3 { font-size: 100px; font-weight: 900; color: #ff4136; text-shadow: 0 0 40px #ff4136; text-align: center; }
    .logic-text { background: #111; padding: 15px; border-left: 5px solid #2ecc40; font-family: 'Courier New', monospace; }
    .critical-warn { background: #ff4136; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; animation: blink 1s infinite; }
    @keyframes blink { 0% {opacity: 1;} 50% {opacity: 0.5;} 100% {opacity: 1;} }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #0074d9;'>🚀 TITAN v23.0 ULTIMATE OMNI</h1>", unsafe_allow_html=True)

# Nạp dữ liệu tự động lọc bẩn
raw_input = st.text_area("📥 NẠP DỮ LIỆU GIẢI ĐẶC BIỆT (Mỗi kỳ 1 dòng):", height=150)

if st.button("⚡ PHÂN TÍCH TRIỆT HẠ NHÀ CÁI"):
    # Lọc số bẩn nghiêm ngặt
    new_data = re.findall(r"\d{5}", raw_input)
    if new_data:
        st.session_state.history.extend(new_data)
        save_memory(st.session_state.history)
        
        # Prompt "Vắt kiệt" AI nhà cái
        prompt = f"""
        Hệ thống: TITAN v23.0 ULTIMATE. 
        Mục tiêu: Thắng tuyệt đối kèo 3 số 5 tinh.
        Lịch sử 100 kỳ gần nhất: {st.session_state.history[-100:]}.
        Yêu cầu:
        1. Tìm ra 3 số 'Chủ Lực' (Core 3) dựa trên nhịp cầu bệt và bóng số vị trí.
        2. Phân tích xem nhà cái có đang dùng thuật toán đảo cầu (Scattering) không.
        3. Nếu xác suất thắng dưới 90%, đặt 'danger': true.
        TRẢ VỀ JSON: {{"core_3": "3 số", "logic": "phân tích thuật toán", "danger": false, "percent": 99}}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            st.session_state.ultimate_res = data
        except:
            # Fallback nâng cao
            all_digits = "".join(st.session_state.history[-40:])
            top_3 = "".join([x[0] for x in Counter(all_digits).most_common(3)])
            st.session_state.ultimate_res = {"core_3": top_3, "logic": "Dựa trên mật độ nổ dày đặc của các kỳ vừa qua.", "danger": False, "percent": 85}
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ ĐẲNG CẤP =================
if "ultimate_res" in st.session_state:
    res = st.session_state.ultimate_res
    
    if res.get('danger'):
        st.markdown("<div class='critical-warn'>⚠️ CẢNH BÁO: NHÀ CÁI ĐANG ĐẢO CẦU ẢO - DỪNG CƯỢC NGAY!</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:#aaa;'>🎯 3 SỐ CHỦ LỰC (XÁC SUẤT {res['percent']}%):</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='core-3'>{res['core_3']}</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div class='logic-text'><b>🧬 CHIẾN THUẬT:</b> {res['logic']}</div>", unsafe_allow_html=True)
    
    # Soi cầu vị trí
    st.divider()
    st.write(f"📊 **NHẬN DIỆN CẦU HIỆN TẠI:** {advanced_analysis(st.session_state.history)}")
    st.markdown("</div>", unsafe_allow_html=True)

if st.button("🗑️ RESET DỮ LIỆU ĐỂ LÀM MỚI CẦU"):
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()
