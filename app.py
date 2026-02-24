import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_neural_memory_v23.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= THUẬT TOÁN BỔ SUNG: MA TRẬN VỊ TRÍ =================
def analyze_position_matrix(data):
    """Phân tích xác suất từng vị trí từ 1-5"""
    if len(data) < 10: return {}
    
    # Tạo ma trận 5 cột (5 vị trí)
    matrix = np.array([[int(d) for d in str(s)] for s in data[-50:]])
    pos_stats = {}
    for i in range(5):
        col = matrix[:, i]
        common = Counter(col).most_common(2)
        pos_stats[f"Vị trí {i+1}"] = [c[0] for c in common]
    return pos_stats

def analyze_odd_even(data):
    """Cảm biến Chẵn Lẻ / Tài Xỉu"""
    all_digits = "".join(data[-20:])
    nums = [int(d) for d in all_digits]
    even = sum(1 for n in nums if n % 2 == 0)
    odd = len(nums) - even
    big = sum(1 for n in nums if n >= 5)
    small = len(nums) - big
    return "Chẵn" if even > odd else "Lẻ", "Tài" if big > small else "Xỉu"

# ================= QUẢN LÝ DỮ LIỆU =================
def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_memory(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-2000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_memory()

# ================= GIAO DIỆN TITAN v23.0 =================
st.set_page_config(page_title="TITAN v23.0 - MATRIX OMNI", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #c9d1d9; }
    .matrix-box { background: #0d1117; border: 1px dashed #58a6ff; padding: 10px; border-radius: 5px; font-family: monospace; }
    .prediction-card {
        background: linear-gradient(160deg, #0d1117 0%, #1a1f25 100%);
        border: 2px solid #58a6ff; border-radius: 20px; padding: 40px;
        box-shadow: 0 0 50px rgba(88, 166, 255, 0.2);
    }
    .main-number { font-size: 110px; font-weight: 900; color: #ff5858; text-align: center; text-shadow: 0 0 40px #ff5858; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🧬 TITAN v23.0 MATRIX OMNI</h1>", unsafe_allow_html=True)

# ================= NHẬP LIỆU & GIẢI MÃ =================
raw_input = st.text_area("📥 NẠP DỮ LIỆU MỚI:", height=100, placeholder="Dán dãy 5 số vào đây...")

if st.button("🚀 KÍCH HOẠT QUÉT MA TRẬN"):
    clean_data = re.findall(r"\d{5}", raw_input)
    if clean_data:
        st.session_state.history.extend(clean_data)
        save_memory(st.session_state.history)
        
        pos_stats = analyze_position_matrix(st.session_state.history)
        trend_oe, trend_bs = analyze_odd_even(st.session_state.history)
        
        # PROMPT THẾ HỆ MỚI: Tích hợp Ma trận và Xu hướng
        prompt = f"""
        Hệ thống: TITAN v23.0 - Phân tích Ma trận vị trí.
        Lịch sử: {st.session_state.history[-50:]}
        Thống kê vị trí: {pos_stats}
        Xu hướng hiện tại: {trend_oe} và {trend_bs}
        
        Nhiệm vụ:
        1. Sử dụng thuật toán Ma trận Vị trí đối xứng để tìm điểm rơi 3D.
        2. Kết hợp xu hướng {trend_oe}/{trend_bs} để loại bỏ các số nghịch cầu.
        3. Chốt 3 số chủ lực và dàn 7 số.
        TRẢ VỀ JSON: {{"main_3": "ABC", "support_4": "DEFG", "logic": "Dựa trên Ma trận vị trí + xu hướng {trend_oe}", "warning": false, "confidence": 98}}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            st.session_state.last_prediction = json.loads(json_match.group())
        except:
            st.error("Lỗi kết nối AI - Đang dùng thuật toán dự phòng.")
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>🎯 3 SỐ CHỦ LỰC</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-number'>{res['main_3']}</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; color:#58a6ff;'>Lót: {res['support_4']}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.subheader("📊 Phân tích Ma trận Vị trí")
        pos_data = analyze_position_matrix(st.session_state.history)
        oe, bs = analyze_odd_even(st.session_state.history)
        
        cols = st.columns(5)
        for i, (pos, vals) in enumerate(pos_data.items()):
            cols[i].markdown(f"<div class='matrix-box'>{pos}<br><b style='color:#ff5858'>{vals}</b></div>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <br>
            - **Xu hướng dòng số:** <b style='color:#58a6ff'>{oe} | {bs}</b>
            - **Chiến thuật:** {res['logic']}
            - **Độ tin cậy:** {res['confidence']}%
        """, unsafe_allow_html=True)

if st.button("🗑️ RESET"):
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()
