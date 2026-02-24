import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter 

# ================= CẤU HÌNH HỆ THỐNG =================
# API Key Gemini của anh
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_deep_memory_v22.json" 

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None 

neural_engine = setup_neural() 

# ================= HỆ THỐNG XỬ LÝ DỮ LIỆU & BỘ NHỚ =================
def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: 
            try: return json.load(f)
            except: return []
    return [] 

def save_memory(data):
    # Lưu 2000 kỳ gần nhất để AI có cái nhìn tổng thể hơn về quy luật
    with open(DB_FILE, "w") as f: 
        json.dump(data[-2000:], f) 

if "history" not in st.session_state:
    st.session_state.history = load_memory() 

# ================= GIAO DIỆN DARK-MODE CHUYÊN NGHIỆP =================
st.set_page_config(page_title="TITAN v22.0 PRO", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #c9d1d9; }
    .status-active { color: #238636; font-weight: bold; padding: 10px; border: 1px solid #238636; border-radius: 8px; }
    .prediction-card {
        background: #0d1117; border: 2px solid #58a6ff;
        border-radius: 15px; padding: 20px; margin-top: 15px;
    }
    .main-num { 
        font-size: 80px; font-weight: 900; color: #ff5a5f; 
        text-align: center; text-shadow: 0 0 30px #ff5a5f;
    }
    .secondary-num { 
        font-size: 45px; font-weight: 700; color: #58a6ff; 
        text-align: center; opacity: 0.8;
    }
    .logic-text { font-style: italic; color: #8b949e; border-left: 3px solid #58a6ff; padding-left: 15px; }
    </style>
""", unsafe_allow_html=True) 

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🧬 TITAN v22.0 OMNI PRO</h1>", unsafe_allow_html=True)

# Hiển thị trạng thái
if neural_engine:
    st.markdown(f"<div class='status-active'>CONNECTED: AI GEMINI | DATABASE: {len(st.session_state.history)} KỲ</div>", unsafe_allow_html=True)
else:
    st.error("LỖI KẾT NỐI API") 

# ================= NHẬP LIỆU & LỌC SẠCH =================
st.subheader("📡 Nạp dữ liệu lịch sử")
raw_input = st.text_area("Dán kết quả (AI tự lọc số bẩn):", height=120, placeholder="Dán dãy số từ web nhà cái tại đây...") 

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 PHÂN TÍCH CHUYÊN SÂU"):
        # Lọc sạch số: Chỉ lấy các dãy 5 chữ số
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            save_memory(st.session_state.history)
            
            # PROMPT PHÂN TÍCH QUY LUẬT NHÀ CÁI
            prompt = f"""
            Dữ liệu kết quả 5D/Lotobet: {st.session_state.history[-150:]}.
            Nhiệm vụ:
            1. Tìm ra "Bóng số" (ví dụ 0 đi với 5, 1 đi với 6).
            2. Phân tích nhịp cầu bệt của nhà cái trong 20 kỳ gần nhất.
            3. Chọn ra 3 số "CHỦ LỰC" có xác suất xuất hiện 100% ở kỳ tiếp theo.
            4. Chọn thêm 4 số lót an toàn.
            TRẢ VỀ JSON DUY NHẤT: {{"chu_luc": [3 số], "lot": [4 số], "quy_luat": "mô tả nhịp quay của nhà cái"}}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                st.session_state.last_prediction = data
            except:
                # Thuật toán dự phòng dựa trên tần suất thực tế nếu AI lỗi
                all_nums = "".join(st.session_state.history[-50:])
                counts = Counter(all_nums).most_common(7)
                res = [str(x[0]) for x in counts]
                st.session_state.last_prediction = {
                    "chu_luc": res[:3], 
                    "lot": res[3:], 
                    "quy_luat": "Dựa trên thuật toán thống kê tần suất cao điểm."
                }
            st.rerun()

with col2:
    if st.button("🗑️ DỌN DẸP BỘ NHỚ"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ DỰ ĐOÁN =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    st.markdown("### 🎯 3 SỐ CHỦ LỰC (VÀO TIỀN MẠNH)")
    st.markdown(f"<div class='main-num'>{' '.join(map(str, res['chu_luc']))}</div>", unsafe_allow_html=True)
    
    st.markdown("### 🛡️ 4 SỐ LÓT (BẢO VỆ VỐN)")
    st.markdown(f<div class='secondary-num'>{' '.join(map(str, res['lot']))}</div>, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"<div class='logic-text'><b>💡 Quy luật nhà cái:</b> {res['quy_luat']}</div>", unsafe_allow_html=True)
    
    full_dan = "".join(map(str, res['chu_luc'])) + "".join(map(str, res['lot']))
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ:", full_dan)
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Cảnh báo: Dữ liệu dựa trên xác suất AI, anh hãy cân đối vốn hợp lý.")
