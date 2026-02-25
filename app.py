import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter

# ================= CẤU HÌNH TITAN V4 SUPREME =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_v4_supreme.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= HỆ THỐNG XỬ LÝ DỮ LIỆU THÔNG MINH =================
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

# ================= THUẬT TOÁN NHẬN DIỆN "BẪY NHÀ CÁI" =================
def detect_house_trap(data):
    if len(data) < 15: return "Dữ liệu mỏng", 0
    
    last_10 = data[-10:]
    all_digits = "".join(last_10)
    counts = Counter(all_digits)
    
    # 1. Kiểm tra cầu bệt (Streak)
    streak_found = False
    for i in range(10):
        if counts[str(i)] >= 6: # Một số xuất hiện quá 60% trong 10 kỳ
            streak_found = True
            break
            
    # 2. Kiểm tra nhịp đảo (Zigzag)
    is_messy = len(counts) > 8 # Quá nhiều số xuất hiện rời rạc
    
    if streak_found: return "CẦU BỆT DỮ DỘI (RỦI RO CAO)", 85
    if is_messy: return "NHỊP ĐẢO LIÊN TỤC (ẢO)", 40
    return "NHỊP CẦU ỔN ĐỊNH", 100

# ================= GIAO DIỆN TITAN V4 SUPREME =================
st.set_page_config(page_title="TITAN V4 - SUPREME AI", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #050505; color: #d1d1d1; }
    .supreme-card {
        background: linear-gradient(145deg, #0f0f0f, #1a1a1a);
        border: 1px solid #d4af37; border-radius: 20px; padding: 40px;
        box-shadow: 0 0 50px rgba(212, 175, 55, 0.1);
    }
    .main-number { font-size: 110px; font-weight: 900; color: #d4af37; text-align: center; text-shadow: 0 0 30px #d4af37; }
    .status-badge { padding: 5px 15px; border-radius: 20px; font-size: 12px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #d4af37;'>🔱 TITAN V4 - TINH HOA SUPREME</h1>", unsafe_allow_html=True)

# Input
raw_input = st.text_area("📡 NẠP DỮ LIỆU TỔNG HỢP:", height=100, placeholder="Dán kết quả 5D tại đây...")

if st.button("🔥 KÍCH HOẠT TRÍ TUỆ TINH HOA"):
    clean_data = re.findall(r"\d{5}", raw_input)
    if clean_data:
        st.session_state.history.extend(clean_data)
        save_memory(st.session_state.history)
        
        trap_msg, safety_score = detect_house_trap(st.session_state.history)
        
        # PROMPT SIÊU CẤP - TỔNG HỢP TINH HOA
        prompt = f"""
        Bạn là kiến trúc sư trưởng của hệ thống TITAN V4 Supreme.
        Dữ liệu 100 kỳ: {st.session_state.history[-100:]}
        Trạng thái nhà cái: {trap_msg} | Điểm an toàn: {safety_score}
        
        Nhiệm vụ: 
        1. Phân tích "Bóng chồng" và "Nhịp gãy Fibonacci".
        2. Nếu đang gặp 'CẦU BỆT', hãy đưa ra dự đoán dựa trên logic 'Đu dây' hoặc 'Bẻ cầu' tùy theo độ dài chuỗi.
        3. Chốt 3 số CHỦ LỰC (Main_3) có xác suất nổ cao nhất trong 5 số giải ĐB.
        4. Trả về JSON chuẩn.
        
        TRẢ VỀ JSON:
        {{
            "main_3": "ABC",
            "support_4": "DEFG",
            "logic_supreme": "Phân tích cực sâu về nhịp cầu",
            "action": "VÀO TIỀN / CHỜ ĐỢI / ĐÁNH NHỎ",
            "confidence": 99
        }}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            res = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            st.session_state.v4_result = res
            st.session_state.v4_safety = {"msg": trap_msg, "score": safety_score}
        except:
            st.error("Hệ thống đang điều chỉnh thuật toán, vui lòng thử lại kỳ sau.")
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ TINH HOA =================
if "v4_result" in st.session_state:
    res = st.session_state.v4_result
    safety = st.session_state.v4_safety
    
    st.markdown("<div class='supreme-card'>", unsafe_allow_html=True)
    
    # Hiển thị trạng thái an toàn
    c1, c2 = st.columns([3, 1])
    with c1:
        st.write(f"🛡️ **TRẠNG THÁI:** {safety['msg']}")
    with c2:
        st.write(f"⭐ **ĐỘ TIN CẬY:** {res['confidence']}%")

    st.divider()
    
    st.markdown(f"<div class='main-number'>{res['main_3']}</div>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align:center; color:#888;'>DÀN LÓT: {res['support_4']}</h3>", unsafe_allow_html=True)
    
    st.markdown(f"**💡 CHIẾN THUẬT SUPREME:** {res['logic_supreme']}")
    
    # Khuyến nghị hành động cực kỳ quan trọng
    action_color = "#39d353" if res['action'] == "VÀO TIỀN" else "#f85149"
    st.markdown(f"<h2 style='text-align:center; color:{action_color};'>👉 HÀNH ĐỘNG: {res['action']}</h2>", unsafe_allow_html=True)
    
    st.divider()
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ:", "".join(sorted(res['main_3'] + res['support_4'])))
    st.markdown("</div>", unsafe_allow_html=True)

# Biểu đồ trực quan để anh thấy "Bẫy"
if len(st.session_state.history) > 10:
    st.subheader("📊 PHÂN TÍCH NHỊP CẦU THỰC TẾ")
    
    last_draws = st.session_state.history[-20:]
    st.write("20 kỳ gần nhất: " + " | ".join(last_draws))
