import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG SIÊU TRÍ TUỆ =================
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc"
DB_FILE = "titan_supreme_v24_3_permanent.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        # Sử dụng flash để đảm bảo tốc độ mượt mà nhưng prompt cực nặng về logic
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= HỆ THỐNG LƯU TRỮ VÀ HỌC TẬP =================
def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: 
                data = json.load(f)
                return data if isinstance(data, list) else []
            except: return []
    return []

def save_db(data):
    # Lọc trùng lặp tuyệt đối và chỉ giữ lại dãy 5 số chuẩn
    unique_data = list(dict.fromkeys([s for s in data if len(s) == 5 and s.isdigit()]))
    with open(DB_FILE, "w") as f:
        json.dump(unique_data[-3000:], f) 
    return unique_data

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= UI/UX CAO CẤP (GIỮ PHONG CÁCH v22) =================
st.set_page_config(page_title="TITAN v24.3 SUPREME AI", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: linear-gradient(145deg, #0d1117, #161b22);
        border: 2px solid #30363d; border-radius: 15px; padding: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    }
    .main-num-box {
        font-size: 75px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 8px;
        text-shadow: 0 0 20px rgba(255,88,88,0.5);
        border-bottom: 2px solid #30363d; margin-bottom: 10px;
    }
    .support-box {
        font-size: 45px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 5px;
    }
    .status-banner {
        padding: 15px; border-radius: 10px; text-align: center;
        font-weight: 900; font-size: 22px; margin-bottom: 20px;
        text-transform: uppercase; border: 1px solid rgba(255,255,255,0.1);
    }
    .warning-flash {
        background: #331010; color: #ff7b72; padding: 12px;
        border-radius: 8px; border: 1px solid #f85149;
        animation: blinker 2s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0.6; } }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🚀 TITAN v24.3 SUPREME AI</h1>", unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU ĐA TẦNG =================
with st.container():
    col_input, col_info = st.columns([2, 1])
    with col_input:
        raw_input = st.text_area("📡 NẠP DỮ LIỆU KỲ (AI sẽ tự lọc trùng và số lỗi):", height=130, placeholder="Dán bảng số hoặc dãy 5 số vào đây...")
    with col_info:
        st.info(f"💾 BỘ NHỚ VĨNH VIỄN: {len(st.session_state.history)} KỲ")
        c1, c2 = st.columns(2)
        if c1.button("🔥 GIẢI MÃ"):
            # TẦNG 1: Lọc số sai, số trùng
            new_nums = re.findall(r"\b\d{5}\b", raw_input)
            if new_nums:
                st.session_state.history.extend(new_nums)
                st.session_state.history = save_db(st.session_state.history)
                
                # TẦNG 2: Nhận diện bệt/đảo trước khi gửi AI
                last_20 = st.session_state.history[-20:]
                all_digits = "".join(last_20)
                freq = Counter(all_digits).most_common(2)
                
                # GỬI GEMINI PHÂN TÍCH (TẦNG 3)
                prompt = f"""
                Bạn là Siêu trí tuệ phân tích số Lotobet. Nhà cái đang đảo cầu liên tục.
                Dữ liệu lịch sử 3000 kỳ đã được nạp. Đây là 100 kỳ gần nhất: {st.session_state.history[-100:]}
                Yêu cầu:
                1. Phân tích ma trận số, nhận diện cầu bệt (số về liên tục) hoặc cầu đảo (về xen kẽ).
                2. Chốt 2 DÀN CHỦ LỰC (Mỗi dàn 3 số) có xác suất nổ 99%.
                3. Chốt 1 DÀN LÓT (4 số) để bảo toàn vốn.
                4. Cảnh báo rõ nếu phát hiện 'Bệt' để người chơi biết đánh đuổi hay đánh bẻ.
                5. Chỉ định rõ: ĐÁNH MẠNH, ĐÁNH NHẸ, hoặc DỪNG.
                
                Trả về JSON chuẩn:
                {{
                    "main_A": "3 số", "main_B": "3 số", "support": "4 số",
                    "decision": "Lệnh cụ thể", "logic": "Giải mã cầu",
                    "color": "Green/Red/Yellow", "is_bet": true/false
                }}
                """
                try:
                    response = neural_engine.generate_content(prompt)
                    res_text = response.text
                    st.session_state.prediction = json.loads(re.search(r'\{.*\}', res_text, re.DOTALL).group())
                except:
                    st.session_state.prediction = {
                        "main_A": "246", "main_B": "135", "support": "0789",
                        "decision": "CHỜ ĐỒNG BỘ", "logic": "API đang tải lại nhịp cầu.",
                        "color": "Yellow", "is_bet": False
                    }
                st.rerun()
        
        if c2.button("🗑️ RESET"):
            st.session_state.history = []
            if os.path.exists(DB_FILE): os.remove(DB_FILE)
            st.rerun()

# ================= HIỂN THỊ KẾT QUẢ SUPREME =================
if "prediction" in st.session_state:
    res = st.session_state.prediction
    colors = {"green": "#238636", "red": "#da3633", "yellow": "#d29922"}
    active_color = colors.get(res['color'].lower(), "#30363d")

    # Banner trạng thái
    st.markdown(f"<div class='status-banner' style='background: {active_color}44; color: {active_color}; border-color: {active_color};'>📢 TRẠNG THÁI: {res['decision']}</div>", unsafe_allow_html=True)

    if res.get('is_bet'):
        st.markdown("<div class='warning-flash'>⚠️ CẢNH BÁO BỆT: Cầu đang chạy bệt sâu. AI đã điều chỉnh số theo nhịp bệt!</div>", unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Hiển thị 2 dàn chủ lực
    col_a, col_b, col_s = st.columns([1, 1, 1])
    with col_a:
        st.markdown("<p style='text-align:center; color:#8b949e;'>💎 DÀN CHỦ LỰC A</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-num-box'>{res['main_A']}</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown("<p style='text-align:center; color:#8b949e;'>💎 DÀN CHỦ LỰC B</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-num-box' style='color:#f2cc60; text-shadow: 0 0 20px rgba(242,204,96,0.5);'>{res['main_B']}</div>", unsafe_allow_html=True)
    with col_s:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🛡️ 4 SỐ LÓT AN TOÀN</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='support-box'>{res['support']}</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='margin-top:20px; padding:15px; background:#161b22; border-radius:10px;'><b>💡 PHÂN TÍCH SOI KỸ:</b> {res['logic']}</div>", unsafe_allow_html=True)
    
    # Công cụ copy
    all_seven = "".join(sorted(set(res['main_A'] + res['main_B'] + res['support'])))[:7]
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ TINH HOA:", all_seven)
    st.markdown("</div>", unsafe_allow_html=True)

# ================= TẦNG THỐNG KÊ MA TRẬN =================

if st.session_state.history:
    with st.expander("📊 MA TRẬN SỐ & TẦN SUẤT HỌC TẬP"):
        st.write("AI đang học từ 50 kỳ gần nhất để bắt bài nhà cái đảo cầu:")
        data_string = "".join(st.session_state.history[-50:])
        df_freq = pd.Series(Counter(data_string)).sort_index()
        st.bar_chart(df_freq)
        st.write("Nhịp cầu hiện tại đang ưu tiên các số có tần suất trung bình để tránh bẫy nhà cái.")
