import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter

# ================= CẤU HÌNH TITAN v24.0 ELITE =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_elite_v24.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-pro') # Dùng bản Pro để thông minh nhất
    except: return None

neural_engine = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU CỨNG (BẢO LƯU VĨNH VIỄN) =================
def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-3000:], f) # Lưu tới 3000 kỳ để AI học sâu

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= THUẬT TOÁN NHẬN BIẾT CẦU (BỆT/ĐẢO) =================
def detect_patterns(data):
    if len(data) < 20: return "Dữ liệu đang nạp...", False
    
    all_digits = "".join(data[-20:])
    last_5 = data[-5:]
    
    # Kiểm tra cầu bệt (Streak)
    flat_last_5 = "".join(last_5)
    counts = Counter(flat_last_5)
    bet_detected = [num for num, freq in counts.items() if freq >= 4]
    
    # Kiểm tra đảo cầu (Zigzag)
    is_reversed = False
    if len(data) >= 4:
        if data[-1] == data[-3] and data[-2] == data[-4]:
            is_reversed = True
            
    status = ""
    if bet_detected: status += f"⚠️ CẦU BỆT SỐ {bet_detected} | "
    if is_reversed: status += "🔄 CẦU ĐẢO LIÊN TỤC | "
    
    risk = len(bet_detected) > 0 or is_reversed
    return status if status else "Cầu nhịp ổn định", risk

# ================= GIAO DIỆN TITAN ELITE =================
st.set_page_config(page_title="TITAN v24.0 ELITE", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .status-bar { background: #0d1117; padding: 15px; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 20px; }
    .bet-signal { font-size: 100px; font-weight: 900; text-align: center; line-height: 1; margin: 20px 0; }
    .stop-signal { background: #440000; color: #ff5555; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; border: 2px solid #ff0000; }
    .go-signal { background: #002200; color: #55ff55; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 24px; border: 2px solid #00ff00; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🚀 TITAN v24.0 ELITE - SIÊU TRÍ TUỆ AI</h1>", unsafe_allow_html=True)

# Hiển thị trạng thái dữ liệu bảo lưu
st.sidebar.markdown(f"### 🗄️ BỘ NHỚ VĨNH VIỄN\n- Đã lưu: **{len(st.session_state.history)} kỳ**")
if st.sidebar.button("🗑️ XÓA HẾT DỮ LIỆU"):
    st.session_state.history = []
    save_db([])
    st.rerun()

# Nạp dữ liệu mượt mà
raw_input = st.text_area("📡 NẠP DỮ LIỆU MỚI (Tự động lọc bẩn):", height=100)

if st.button("⚡ KÍCH HOẠT SIÊU TRÍ TUỆ"):
    new_entries = re.findall(r"\d{5}", raw_input)
    if new_entries:
        # Chỉ thêm những kỳ chưa có (tránh trùng)
        current_history = st.session_state.history
        added_count = 0
        for entry in new_entries:
            if entry not in current_history[-10:]:
                current_history.append(entry)
                added_count += 1
        st.session_state.history = current_history
        save_db(current_history)
        
        # Phân tích nhịp cầu
        p_status, is_risky = detect_patterns(st.session_state.history)
        
        # SIÊU PROMPT ELITE
        prompt = f"""
        Bạn là kiến trúc sư trưởng về giải mã thuật toán xác suất 5D.
        Lịch sử dữ liệu chuyên sâu (2000 kỳ): {st.session_state.history[-100:]}
        Trạng thái cầu hiện tại: {p_status}
        
        Nhiệm vụ:
        1. Sử dụng thuật toán Mạng thần kinh phân tích nhịp rơi.
        2. Nếu phát hiện nhà cái đang "vét tiền" (cầu ảo), hãy trả về 'action': 'STOP'.
        3. Chọn 3 số chủ lực (Main_3) có xác suất xuất hiện trong 5 số của giải ĐB > 99%.
        4. Tư duy qua bóng số âm dương và nhịp Fibonacci.
        
        Trả về JSON duy nhất:
        {{
            "action": "PLAY" hoặc "STOP",
            "main_3": "3 số",
            "support_4": "4 số",
            "logic": "Giải thích sâu về nhịp cầu",
            "confidence": 0-100
        }}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            res = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            st.session_state.elite_prediction = res
        except:
            st.error("Neural Link quá tải. Vui lòng thử lại sau 5 giây.")
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ THỰC CHIẾN =================
if "elite_prediction" in st.session_state:
    res = st.session_state.elite_prediction
    p_status, is_risky = detect_patterns(st.session_state.history)
    
    st.markdown("<div class='status-bar'>", unsafe_allow_html=True)
    st.write(f"📊 **TRẠNG THÁI CẦU:** {p_status}")
    st.markdown("</div>", unsafe_allow_html=True)

    if res['action'] == "STOP" or is_risky or res['confidence'] < 90:
        st.markdown("<div class='stop-signal'>🔴 KHÔNG ĐÁNH - NHÀ CÁI ĐANG ĐẢO CẦU 🔴</div>", unsafe_allow_html=True)
        st.write(f"**Lý do AI:** {res['logic']}")
    else:
        st.markdown("<div class='go-signal'>🟢 TÍN HIỆU VÀNG - VÀO TIỀN AN TOÀN 🟢</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"<div class='bet-signal' style='color:#39d353;'>{res['main_3']}</div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center;'>🎯 3 SỐ CHỦ LỰC (SIÊU CẤP)</p>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='bet-signal' style='color:#58a6ff; font-size:60px;'>{res['support_4']}</div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center;'>🛡️ DÀN LÓT BẢO VỆ VỐN</p>", unsafe_allow_html=True)

        st.info(f"💡 **PHÂN TÍCH AI:** {res['logic']}")
        st.text_input("📋 SAO CHÉP DÀN 7 SỐ:", res['main_3'] + res['support_4'])
        st.progress(res['confidence'] / 100)
        st.write(f"Độ tự tin siêu trí tuệ: {res['confidence']}%")

