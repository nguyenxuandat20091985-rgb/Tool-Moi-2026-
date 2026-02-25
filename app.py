import streamlit as st
import google.generativeai as genai
import re
import json
import os
import numpy as np
from collections import Counter

# ================= CẤU HÌNH SIÊU CẤP =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_elite_v24.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= THUẬT TOÁN ĐỐI KHÁNG NHÀ CÁI =================
def detect_bet_bridge(data):
    """Phát hiện cầu bệt và cảnh báo bẫy"""
    if len(data) < 15: return "Dữ liệu mỏng", 0
    
    all_nums = "".join(data[-15:])
    counts = Counter(all_nums)
    most_common = counts.most_common(1)[0] # (số, số lần)
    
    # Nếu 1 số xuất hiện > 8 lần trong 15 kỳ -> Cầu bệt cực nặng
    if most_common[1] >= 8:
        return f"CẦU BỆT SỐ {most_common[0]} (Rủi ro bẻ cầu cao)", 2 
    elif most_common[1] >= 5:
        return f"Cầu đang nhen nhóm số {most_common[0]}", 1
    return "Cầu nhảy (Biến động)", 0

def calculate_smart_money(confidence):
    """Tính toán tỷ lệ vào tiền để bảo toàn vốn"""
    if confidence >= 95: return "100% Vốn định mức (Đánh mạnh)"
    if confidence >= 85: return "50% Vốn định mức (Đánh vừa)"
    return "10% Vốn (Đánh văn nghệ hoặc BỎ)"

# ================= GIAO DIỆN TINH HOA =================
st.set_page_config(page_title="TITAN v24.0 ELITE", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #050505; color: #d1d1d1; }
    .elite-card {
        background: linear-gradient(145deg, #0f1115, #1a1d23);
        border: 1px solid #d4af37; border-radius: 20px; padding: 35px;
        box-shadow: 0 0 50px rgba(212, 175, 55, 0.1);
    }
    .main-number { font-size: 110px; font-weight: 900; color: #d4af37; text-align: center; text-shadow: 0 0 40px #d4af37; }
    .warning-glow { color: #ff4b4b; text-shadow: 0 0 10px #ff4b4b; font-weight: bold; animation: blinker 1.5s linear infinite; }
    @keyframes blinker { 50% { opacity: 0; } }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #d4af37;'>🔱 TITAN v24.0 ELITE 🔱</h1>", unsafe_allow_html=True)

# Nhập liệu thông minh
raw_input = st.text_area("📡 HỆ THỐNG NHẬN DIỆN DỮ LIỆU:", height=120, placeholder="Dán kết quả tại đây...")

if st.button("⚜️ GIẢI MÃ TINH HOA ⚜️"):
    clean_data = re.findall(r"\d{5}", raw_input)
    if clean_data:
        if "history" not in st.session_state: st.session_state.history = []
        st.session_state.history.extend(clean_data)
        
        bridge_status, risk_level = detect_bet_bridge(st.session_state.history)
        
        # PROMPT TINH HOA - TỔNG HỢP MỌI THUẬT TOÁN
        prompt = f"""
        Bạn là kiến trúc sư trưởng về xác suất 5D. 
        Lịch sử: {st.session_state.history[-100:]}
        Trạng thái cầu: {bridge_status}
        
        Yêu cầu tối mật:
        1. Sử dụng thuật toán Đối xứng Ma trận và Nhịp rơi Fibonacci.
        2. Loại bỏ các số "ảo" nhà cái đang kìm.
        3. Phân tích "Bóng âm dương" của 3 kỳ gần nhất.
        4. Chốt 3 số CHỦ LỰC có tỷ lệ nổ chung giải ĐB cao nhất.
        
        TRẢ VỀ JSON DUY NHẤT:
        {{
            "main_3": "ABC",
            "support_4": "DEFG",
            "logic": "Giải thích sắc bén",
            "confidence": 99,
            "action": "VÀO TIỀN/DỪNG"
        }}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            res = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            
            st.markdown("<div class='elite-card'>", unsafe_allow_html=True)
            
            # Cảnh báo rủi ro cầu bệt
            if risk_level == 2:
                st.markdown("<p class='warning-glow'>⚠️ CẢNH BÁO: PHÁT HIỆN CẦU BỆT ẢO - CỰC KỲ NGUY HIỂM</p>", unsafe_allow_html=True)
            
            st.write(f"🔍 **CHIẾN THUẬT:** {res['logic']}")
            st.markdown(f"<div class='main-number'>{res['main_3']}</div>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align:center; color:#58a6ff;'>Lót: {res['support_4']}</h3>", unsafe_allow_html=True)
            
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("ĐỘ TỰ TIN", f"{res['confidence']}%")
            c2.metric("KHUYẾN NGHỊ", calculate_smart_money(res['confidence']))
            
            st.markdown("</div>", unsafe_allow_html=True)
        except:
            st.error("Hệ thống đang điều chỉnh nhịp cầu, vui lòng thử lại sau 30 giây.")

