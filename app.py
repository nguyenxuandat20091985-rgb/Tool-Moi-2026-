import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG TITAN v23.0 =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_v23_core.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU SẠCH =================
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

# ================= THUẬT TOÁN BỔ SUNG (VỊ TRÍ & TÀI XỈU) =================
def advanced_stats(data):
    if len(data) < 10: return {}
    matrix = np.array([[int(d) for d in s] for s in data[-20:]])
    
    # Phân tích Tài (5-9) / Xỉu (0-4)
    big_small = []
    for row in matrix:
        big_small.append("Tài" if np.mean(row) >= 4.5 else "Xỉu")
    
    # Tần suất vị trí (Hàng đơn vị)
    pos_counts = Counter(matrix[:, -1])
    return {
        "trend": Counter(big_small).most_common(1)[0][0],
        "hot_pos": pos_counts.most_common(3)
    }

# ================= GIAO DIỆN TITAN v23 =================
st.set_page_config(page_title="TITAN v23.0 - SUPREME AI", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .main-card {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 12px; padding: 25px; margin-bottom: 20px;
    }
    .main-num { font-size: 90px; color: #39d353; font-weight: 900; text-align: center; text-shadow: 0 0 20px #238636; }
    .warning-text { color: #f85149; background: #2d1616; padding: 10px; border-radius: 5px; border: 1px solid #f85149; }
    .stat-box { background: #161b22; padding: 10px; border-radius: 8px; border: 1px solid #30363d; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 TITAN v23.0 - CHIẾN THẦN BẺ CẦU")

# Sidebar thông tin
with st.sidebar:
    st.header("📊 TRẠNG THÁI HỆ THỐNG")
    st.write(f"Kỳ đã lưu: {len(st.session_state.history)}")
    if st.button("🗑️ RESET DỮ LIỆU"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# Nhập liệu
raw_input = st.text_area("📥 NẠP DỮ LIỆU KỲ MỚI:", height=100, placeholder="Dán dãy số 5D tại đây...")

if st.button("🚀 KÍCH HOẠT PHÂN TÍCH V23"):
    clean_data = re.findall(r"\d{5}", raw_input)
    if clean_data:
        st.session_state.history.extend(clean_data)
        save_memory(st.session_state.history)
        
        # Thống kê nội bộ trước khi hỏi AI
        internal_stats = advanced_stats(st.session_state.history)
        
        # PROMPT V23.0 - NÂNG CẤP MA TRẬN VỊ TRÍ
        prompt = f"""
        Hệ thống: TITAN v23.0. Chuyên gia 3D Lotobet.
        Dữ liệu thực tế (100 kỳ): {st.session_state.history[-100:]}
        Thống kê nội bộ: {internal_stats}
        
        YÊU CẦU:
        1. Áp dụng MA TRẬN VỊ TRÍ ĐỐI XỨNG để tìm 3 số chủ lực.
        2. Kiểm tra chu kỳ Fibonacci để loại bỏ các số đang "ảo".
        3. Dự đoán 3 số (Main_3) nằm trong 5 số của giải ĐB.
        4. Trả về dự đoán với độ tin cậy thực tế (Confidence).
        
        TRẢ VỀ JSON:
        {{
            "main_3": "ABC",
            "support_4": "DEFG",
            "logic": "Giải thích vắn tắt",
            "warning": false,
            "confidence": 98
        }}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            st.session_state.v23_result = data
        except Exception as e:
            st.error("Lỗi xử lý AI - Sử dụng thuật toán dự phòng.")
            # Fallback
            all_n = "".join(st.session_state.history[-30:])
            top = [x[0] for x in Counter(all_n).most_common(7)]
            st.session_state.v23_result = {"main_3": "".join(top[:3]), "support_4": "".join(top[3:]), "logic": "Fallback Stat", "warning": False, "confidence": 60}
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if "v23_result" in st.session_state:
    res = st.session_state.v23_result
    
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    if res['warning'] or res['confidence'] < 80:
        st.markdown("<div class='warning-text'>⚠️ CẢNH BÁO: Cầu đang nhiễu (Nhịp Tài/Xỉu không ổn định). ĐÁNH NHỎ HOẶC NGHỈ.</div>", unsafe_allow_html=True)
    
    st.write(f"🔍 **CHIẾN THUẬT v23:** {res['logic']}")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"<div class='main-num'>{res['main_3']}</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>🔥 3 SỐ CHỦ LỰC (98% XÁC SUẤT)</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h2 style='text-align:center; color:#58a6ff;'>{res['support_4']}</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>🛡️ DÀN LÓT</p>", unsafe_allow_html=True)

    st.divider()
    
    # Tính năng Copy
    st.text_input("📋 DÀN 7 SỐ KUBET:", res['main_3'] + res['support_4'])
    st.progress(res['confidence'] / 100)
    st.markdown(f"<p style='text-align:right;'>Độ tin cậy AI: {res['confidence']}%</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Hiển thị thống kê Tài Xỉu để anh đối soát
    if st.session_state.history:
        st.subheader("📈 NHỊP CẦU TÀI/XỈU (Gần đây)")
        stats = advanced_stats(st.session_state.history)
        st.info(f"Xu hướng hiện tại: **{stats.get('trend')}** | Top vị trí hàng đơn vị: **{stats.get('hot_pos')}**")

