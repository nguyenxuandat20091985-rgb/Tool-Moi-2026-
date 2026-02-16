import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter 

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v21.json" 

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None 

neural_engine = setup_neural() 

def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: 
            try: return json.load(f)
            except: return []
    return [] 

def save_memory(data):
    with open(DB_FILE, "w") as f: 
        json.dump(data[-1000:], f) 

if "history" not in st.session_state:
    st.session_state.history = load_memory() 

# ================= UI DESIGN (Giữ nguyên kết cấu anh yêu cầu) =================
st.set_page_config(page_title="TITAN v21.0 PRO", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #c9d1d9; }
    .status-active { color: #238636; font-weight: bold; border-left: 3px solid #238636; padding-left: 10px; }
    .prediction-card {
        background: #0d1117; border: 2px solid #30363d;
        border-radius: 12px; padding: 25px; margin-top: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .num-display { 
        font-size: 60px; font-weight: 900; color: #58a6ff; 
        text-align: center; letter-spacing: 10px; text-shadow: 0 0 25px #58a6ff;
    }
    .logic-box { font-size: 14px; color: #8b949e; background: #161b22; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True) 

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🧬 TITAN v21.0 OMNI</h2>", unsafe_allow_html=True)
if neural_engine:
    st.markdown(f"<p class='status-active'>● KẾT NỐI NEURAL-LINK: OK | DỮ LIỆU: {len(st.session_state.history)} KỲ</p>", unsafe_allow_html=True)
else:
    st.error("LỖI KẾT NỐI API - KIỂM TRA LẠI KEY") 

# ================= XỬ LÝ DỮ LIỆU =================
raw_input = st.text_area("📡 NẠP DỮ LIỆU (Dán các dãy 5 số):", height=100, placeholder="32880\n21808\n...") 

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 GIẢI MÃ THUẬT TOÁN"):
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            save_memory(st.session_state.history)
            
            prompt = f"""
            Bạn là AI chuyên gia xác suất 5D. 
            Lịch sử lưu trữ: {st.session_state.history[-100:]}.
            Yêu cầu:
            1. Phân tích các số đang bệt (Streak) và các số "bóng" sắp nổ.
            2. Phát hiện nếu nhà cái đang đảo cầu để né các số hay về.
            3. Chốt dàn 7 số an toàn nhất cho sảnh 3 số 5 tinh (Không cố định).
            TRẢ VỀ JSON: {{"dan4": ["1","2","3","4"], "dan3": ["5","6","7"], "logic": "viết ngắn gọn"}}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                st.session_state.last_result = data
            except:
                all_nums = "".join(st.session_state.history[-30:])
                counts = Counter(all_nums).most_common(7)
                res = [str(x[0]) for x in counts]
                st.session_state.last_result = {"dan4": res[:4], "dan3": res[4:], "logic": "Dùng thống kê tần suất thực tế."}
            st.rerun() 

with col2:
    if st.button("🗑️ RESET BỘ NHỚ"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        if "last_result" in st.session_state: del st.session_state.last_result
        st.rerun() 

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='logic-box'><b>💡 Phân tích:</b> {res['logic']}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; font-size:12px; color:#888;'>🎯 4 SỐ CHỦ LỰC (VÀO TIỀN)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display'>{''.join(map(str, res['dan4']))}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; font-size:12px; color:#888; margin-top:20px;'>🛡️ 3 SỐ LÓT (GIỮ VỐN)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display' style='color:#f2cc60; text-shadow: 0 0 25px #f2cc60;'>{''.join(map(str, res['dan3']))}</div>", unsafe_allow_html=True)
    
    copy_val = "".join(map(str, res['dan4'])) + "".join(map(str, res['dan3']))
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ:", copy_val)
    st.markdown("</div>", unsafe_allow_html=True) 
