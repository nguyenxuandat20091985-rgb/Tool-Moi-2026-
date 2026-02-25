import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter
import scipy.stats as stats

# ================= SIÊU CẤU HÌNH TITAN v24.0 =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_v24_supreme.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-pro') # Nâng cấp lên bản Pro nếu có thể
    except: return None

neural_engine = setup_neural()

# ================= HỆ THỐNG QUẢN TRỊ DỮ LIỆU =================
def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_memory(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-2500:], f) # Tăng lên 2500 kỳ để soi cầu trường kỳ

if "history" not in st.session_state:
    st.session_state.history = load_memory()

# ================= THUẬT TOÁN TINH HOA (AGI LOGIC) =================
def agi_analysis(data):
    if len(data) < 20: return "Cần thêm dữ liệu"
    
    # 1. Chuyển đổi dữ liệu sang ma trận số học
    matrix = np.array([[int(d) for d in s] for s in data[-50:]])
    
    # 2. Thuật toán phát hiện Bệt (Streak) - Cực quan trọng
    streaks = {}
    for n in range(10):
        count = 0
        for i in range(1, 11): # Kiểm tra 10 kỳ gần nhất
            if n in [int(d) for d in data[-i]]: count += 1
            else: break
        streaks[n] = count
    
    # 3. Phân tích độ lệch chuẩn (Chống cầu ảo)
    all_digits = "".join(data[-100:])
    freq = Counter(all_digits)
    counts = list(freq.values())
    z_scores = stats.zscore(counts) if len(counts) > 1 else [0]*10
    
    return {
        "streaks": streaks,
        "anomalies": [i for i, z in enumerate(z_scores) if abs(z) > 1.5],
        "last_5": data[-5:]
    }

# ================= GIAO DIỆN CHIẾN THẦN =================
st.set_page_config(page_title="TITAN v24.0 OMNIPOTENT", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #00050a; color: #00ffcc; }
    .supreme-card {
        background: rgba(0, 20, 40, 0.8);
        border: 2px solid #00ffcc;
        border-radius: 20px; padding: 40px;
        box-shadow: 0 0 50px rgba(0, 255, 204, 0.2);
    }
    .gold-num { font-size: 110px; font-weight: 900; color: #ffcc00; text-align: center; text-shadow: 0 0 40px #ffcc00; }
    .danger-zone { background: #400; color: #ff4444; border: 1px solid #ff4444; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    .info-text { color: #8899aa; font-family: 'Courier New', monospace; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🔱 TITAN v24.0: SIÊU TRÍ TUỆ OMNIPOTENT</h1>", unsafe_allow_html=True)

# Side-info
st.sidebar.markdown("### 📊 CHỈ SỐ NEURAL")
st.sidebar.write(f"Độ sâu dữ liệu: {len(st.session_state.history)} kỳ")
if st.sidebar.button("🗑️ RESET TOÀN BỘ"):
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()

# Nhập liệu thông minh
raw_input = st.text_area("📡 NẠP DỮ LIỆU TỔNG HỢP (Dán mọi định dạng):", height=120)

if st.button("🧠 KÍCH HOẠT SIÊU TRÍ TUỆ"):
    clean_data = re.findall(r"\d{5}", raw_input)
    if clean_data:
        st.session_state.history.extend(clean_data)
        save_memory(st.session_state.history)
        
        # Phân tích nội bộ cấp cao
        intel = agi_analysis(st.session_state.history)
        
        # PROMPT SIÊU TINH HOA CHO GEMINI
        prompt = f"""
        Hệ thống: TITAN v24.0 Omnipotent AGI. 
        Mục tiêu: Chiến thắng tuyệt đối Lotobet 3D Không cố định.
        Dữ liệu lịch sử: {st.session_state.history[-120:]}
        Phân tích bệt (Streaks): {intel['streaks']}
        Cảnh báo bất thường (Anomalies): {intel['anomalies']}

        YÊU CẦU CHIẾN THUẬT:
        1. PHÂN TÍCH BỆT: Nếu một số bệt > 3 kỳ, tính xác suất gãy. KHÔNG đưa số sắp gãy vào Main_3.
        2. BÓNG SỐ & ĐIỂM RƠI: Áp dụng bóng âm dương (0-5, 1-6, 2-7, 3-8, 4-9) kết hợp nhịp rơi Fibonacci.
        3. DÀN 7 SỐ TINH HOA: Chốt 3 số chủ lực (Main_3) và 4 số lót (Support_4).
        4. CẢNH BÁO NHÀ CÁI: Chỉ ra cụ thể nhà cái đang dùng chiêu trò gì (kìm số, đảo cầu, hay thả cầu).

        TRẢ VỀ JSON DUY NHẤT:
        {{
            "main_3": "ABC",
            "support_4": "DEFG",
            "house_trap": "Mô tả bẫy nhà cái",
            "strategy": "Cách vào tiền kỳ này",
            "danger_level": "Thấp/Trung bình/Cao",
            "confidence": 99
        }}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            st.session_state.v24_prediction = data
        except:
            st.error("Neural Link gián đoạn. Đang dùng thuật toán dự phòng tối cao...")
            # Fallback AGI
            all_n = "".join(st.session_state.history[-40:])
            counts = Counter(all_n).most_common(7)
            res = [x[0] for x in counts]
            st.session_state.v24_prediction = {
                "main_3": "".join(res[:3]), "support_4": "".join(res[3:]),
                "house_trap": "Dữ liệu nhiễu, nhà cái đang đảo nhịp.",
                "strategy": "Đánh nhỏ giữ vốn.", "danger_level": "Cao", "confidence": 65
            }
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ TINH HOA =================
if "v24_prediction" in st.session_state:
    res = st.session_state.v24_prediction
    
    st.markdown("<div class='supreme-card'>", unsafe_allow_html=True)
    
    if res['danger_level'] == "Cao" or res['confidence'] < 85:
        st.markdown(f"<div class='danger-zone'>⚠️ CẢNH BÁO NGUY HIỂM: {res['house_trap']}</div>", unsafe_allow_html=True)
    else:
        st.success(f"✅ NHỊP CẦU ĐẸP: {res['house_trap']}")

    st.markdown(f"<p class='info-text'>🛡️ CHIẾN THUẬT: {res['strategy']}</p>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"<div class='gold-num'>{res['main_3']}</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; font-size:20px;'>💎 3 SỐ CHỦ LỰC (SIÊU CẤP)</p>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<h1 style='text-align:center; color:#00ffcc; font-size:60px;'>{res['support_4']}</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>🛡️ DÀN LÓT AN TOÀN</p>", unsafe_allow_html=True)

    st.divider()
    
    full_dan = "".join(sorted(set(res['main_3'] + res['support_4'])))
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ TINH HOA:", full_dan)
    
    st.progress(res['confidence'] / 100)
    st.write(f"Độ tin cậy hệ thống: {res['confidence']}%")
    st.markdown("</div>", unsafe_allow_html=True)

# Phân tích kỹ thuật sâu (Dành cho anh theo dõi)
if st.session_state.history:
    with st.expander("🔍 PHÂN TÍCH NHỊP BỆT & ĐIỂM RƠI"):
        intel = agi_analysis(st.session_state.history)
        st.write("Tần suất bệt kỳ gần nhất:", intel['streaks'])
        if intel['anomalies']:
            st.warning(f"Phát hiện số có dấu hiệu bị nhà cái 'kìm': {intel['anomalies']}")
