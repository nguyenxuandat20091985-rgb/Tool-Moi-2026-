import streamlit as st
import re
import json
import numpy as np
import google.generativeai as genai
from collections import Counter
from pathlib import Path

# ================= CẤU HÌNH GEMINI AI =================
# Anh dán API Key của anh vào đây nhé
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" 
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ================= DATA MANAGEMENT =================
DATA_FILE = "titan_v13_neural.json"

def load_db():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_db(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data[-5000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= UI & STYLE =================
st.set_page_config(page_title="TITAN v13.0 NEURAL", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #050a10; color: #00ffcc; }
    .stButton > button {
        background: linear-gradient(135deg, #00ffcc 0%, #0055ff 100%);
        color: black; border: none; font-weight: 900; border-radius: 8px; height: 50px;
    }
    .gemini-analysis {
        background: rgba(0, 85, 255, 0.1); border-left: 5px solid #0055ff;
        padding: 15px; margin: 15px 0; border-radius: 5px; font-style: italic;
    }
    .number-box {
        font-size: 35px; font-weight: 900; color: #fff; text-align: center;
        background: #111b27; border: 1px solid #00ffcc; border-radius: 10px; padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ================= THUẬT TOÁN GEMINI NEURAL =================
def gemini_brain(history):
    if len(history) < 10: return None
    
    # Chuẩn bị dữ liệu gửi cho Gemini
    data_str = " | ".join(history[-30:]) # Gửi 30 kỳ gần nhất
    prompt = f"""
    Bạn là một chuyên gia phân tích dữ liệu xác suất cho trò chơi 5D. 
    Dữ liệu 30 kỳ gần nhất: {data_str}.
    Nhiệm vụ: 
    1. Nhận diện các số có xu hướng bệt (lặp lại).
    2. Nhận diện quy luật bước nhảy của 5 vị trí.
    3. Chọn ra dàn 7 số an toàn nhất.
    4. Trả về kết quả theo định dạng JSON: 
    {{"dan7": [7 số], "ly_do": "phân tích ngắn gọn", "do_tin_cay": %}}
    """
    
    try:
        response = model.generate_content(prompt)
        # Tìm và trích xuất JSON từ phản hồi của Gemini
        res_text = response.text
        json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data
    except Exception as e:
        # Nếu lỗi API, dùng thuật toán fallback (Dự phòng)
        return {"dan7": ["0","1","2","3","5","6","8"], "ly_do": "API Error - Using Fallback", "do_tin_cay": 50}

# ================= GIAO DIỆN CHÍNH =================
st.markdown("<h2 style='text-align: center;'>🧠 TITAN v13.0 GEMINI-NEURAL</h2>", unsafe_allow_html=True)

input_data = st.text_area("📡 NẠP DỮ LIỆU KỲ MỚI (5D):", height=80)

c1, c2 = st.columns(2)
with c1:
    if st.button("🔥 KÍCH HOẠT GEMINI"):
        if input_data:
            new_recs = re.findall(r"\d{5}", input_data)
            st.session_state.history.extend(new_recs)
            save_db(st.session_state.history)
            
            with st.spinner('Gemini đang tư duy cầu bệt...'):
                result = gemini_brain(st.session_state.history)
                st.session_state.last_result = result
            st.rerun()
with c2:
    if st.button("🗑️ RESET"):
        st.session_state.history = []
        st.session_state.last_result = None
        save_db([])
        st.rerun()

if "last_result" in st.session_state and st.session_state.last_result:
    res = st.session_state.last_result
    dan7 = [str(x) for x in res['dan7']]
    
    # Hiển thị phân tích của Gemini
    st.markdown(f"<div class='gemini-analysis'><b>Tư duy AI:</b> {res['ly_do']}</div>", unsafe_allow_html=True)
    
    # Chia dàn 4 và dàn 3 như anh muốn
    st.markdown("<div style='display: flex; justify-content: space-around;'>", unsafe_allow_html=True)
    st.write("### Dàn 4 (Chủ lực)")
    st.write("### Dàn 3 (Lót)")
    st.markdown("</div>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"<div class='number-box'>{' - '.join(dan7[:4])}</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"<div class='number-box' style='border-color: #ffaa00;'>{' - '.join(dan7[4:7])}</div>", unsafe_allow_html=True)

    st.text_input("📋 COPY DÀN 7 SỐ NHANH:", "".join(dan7))
    st.progress(res['do_tin_cay'] / 100)
    st.write(f"Độ tin cậy: {res['do_tin_cay']}% | Dữ liệu học: {len(st.session_state.history)} kỳ")
