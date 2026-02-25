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
DB_FILE = "titan_elite_v24.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-pro') # Dùng bản Pro để tư duy mạnh hơn
    except: return None

neural_engine = setup_neural()

# ================= HỆ THỐNG LƯU TRỮ VĨNH VIỄN =================
def load_data():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_data(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-3000:], f) # Lưu tối đa 3000 kỳ để học sâu

if "history" not in st.session_state:
    st.session_state.history = load_data()

# ================= THUẬT TOÁN NHẬN BIẾT BỆT & ĐẢO =================
def detect_market_behavior(data):
    if len(data) < 10: return "Dữ liệu mỏng", "Chờ"
    
    last_5 = data[-5:]
    all_digits = "".join(last_5)
    counts = Counter(all_digits)
    
    # Kiểm tra Bệt (1 hoặc 2 số xuất hiện quá dày trong 5 kỳ)
    is_streak = any(v >= 4 for v in counts.values())
    
    # Kiểm tra Đảo (Các số ra không lặp lại, thay đổi liên tục)
    is_choppy = len(set(all_digits)) > 8
    
    if is_streak: return "CẦU BỆT NGUY HIỂM", "DỪNG"
    if is_choppy: return "CẦU ĐẢO LOẠN", "DỪNG"
    return "NHỊP CẦU ỔN ĐỊNH", "ĐÁNH"

# ================= GIAO DIỆN HIỆN ĐẠI =================
st.set_page_config(page_title="TITAN v24.0 ELITE", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; font-family: 'Segoe UI'; }
    .action-stop { background: #490a0a; border: 2px solid #f85149; padding: 20px; border-radius: 15px; text-align: center; color: #ff7b72; font-size: 24px; font-weight: bold; }
    .action-go { background: #052309; border: 2px solid #39d353; padding: 20px; border-radius: 15px; text-align: center; color: #7ee787; font-size: 24px; font-weight: bold; }
    .number-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin: 10px 0; }
    .big-num { font-size: 80px; font-weight: 900; color: #58a6ff; text-align: center; letter-spacing: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🧬 TITAN v24.0 ELITE: SIÊU TRÍ TUỆ</h1>", unsafe_allow_html=True)

# Layout chính
col_input, col_display = st.columns([1, 2])

with col_input:
    st.subheader("📡 CẬP NHẬT DỮ LIỆU")
    raw_input = st.text_area("Dán số vào đây (Tự động lưu):", height=200)
    
    if st.button("🚀 GIẢI MÃ TINH HOA"):
        new_nums = re.findall(r"\d{5}", raw_input)
        if new_nums:
            # Gộp và loại trùng nhưng giữ thứ tự
            updated_history = st.session_state.history + new_nums
            st.session_state.history = updated_history[-3000:]
            save_data(st.session_state.history)
            
            # Phân tích hành vi cầu
            behavior, action = detect_market_behavior(st.session_state.history)
            
            # Gửi Prompt "Siêu trí tuệ" cho AI
            prompt = f"""
            Bạn là TITAN v24.0 - Hệ thống dự đoán 3D Lotobet tinh hoa nhất.
            Lịch sử kỳ: {st.session_state.history[-150:]}
            Hành vi cầu hiện tại: {behavior}
            
            NHIỆM VỤ:
            1. Sử dụng thuật toán Xác suất Bayes và Chu kỳ Fibonacci để tìm 3 số (Main_3).
            2. Phân tích xem nhà cái có đang dùng thuật toán kìm số không.
            3. Nếu hành vi là 'DỪNG', hãy giải thích cực kỳ chi tiết tại sao.
            
            TRẢ VỀ JSON:
            {{
                "action": "{action}",
                "main_3": "ABC",
                "support_4": "DEFG",
                "analysis": "Phân tích sâu về nhịp cầu và bẫy nhà cái",
                "risk_level": "High/Medium/Low"
            }}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                res_data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                st.session_state.prediction = res_data
            except:
                st.session_state.prediction = {"action": "DỪNG", "main_3": "---", "support_4": "----", "analysis": "Lỗi kết nối Neural. Hãy kiểm tra API.", "risk_level": "High"}
            st.rerun()

    if st.button("🗑️ RESET TOÀN BỘ"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

with col_display:
    if "prediction" in st.session_state:
        res = st.session_state.prediction
        
        # HIỂN THỊ LỆNH CHIẾN THUẬT
        if res['action'] == "ĐÁNH" and res['risk_level'] != "High":
            st.markdown(f"<div class='action-go'>✅ LỆNH: VÀO TIỀN (Rủi ro: {res['risk_level']})</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='action-stop'>🚫 LỆNH: DỪNG CƯỢC - CHỜ NHỊP MỚI</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='number-card'>", unsafe_allow_html=True)
        st.write(f"🔬 **PHÂN TÍCH TỪ AI:** {res['analysis']}")
        
        if res['action'] == "ĐÁNH":
            st.markdown(f"<div class='big-num'>{res['main_3']}</div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center;'>🎯 3 SỐ CHỦ LỰC VÀNG</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align:center; color:#58a6ff;'>Lót: {res['support_4']}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Thống kê trực quan
    if st.session_state.history:
        st.subheader("📊 BIỂU ĐỒ NHỊP CẦU (30 kỳ gần nhất)")
        last_30 = "".join(st.session_state.history[-30:])
        chart_data = pd.DataFrame.from_dict(Counter(last_30), orient='index', columns=['Tần suất'])
        st.bar_chart(chart_data)

st.markdown(f"<p style='text-align:center; color:#444;'>Dữ liệu bảo lưu: {len(st.session_state.history)} kỳ | TITAN v24.0 ELITE</p>", unsafe_allow_html=True)
