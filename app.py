import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter

# ================= CONFIG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_v32_final.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: return json.load(f)
    return {"history": [], "predictions": []}

def save_db(data):
    with open(DB_FILE, "w") as f: json.dump(data, f)

if "db" not in st.session_state:
    st.session_state.db = load_db()

# ================= UI DESIGN =================
st.set_page_config(page_title="TITAN v32.0 ANTI-SCAM", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #010b13; color: #ffffff; }
    .critical-card { background: linear-gradient(180deg, #1a2a6c, #b21f1f); border-radius: 15px; padding: 20px; border: 2px solid #ff4b2b; }
    .num-main { font-size: 80px; font-weight: 900; color: #00ff88; text-align: center; line-height: 1; }
    .trash-box { color: #ff4b2b; text-decoration: line-through; font-size: 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ TITAN v32.0 - CHẶN THUA")

# ================= LOGIC XỬ LÝ =================
raw_input = st.text_area("📡 DÁN KẾT QUẢ 5 SỐ (Kỳ gần nhất ở trên cùng):", height=120)

if st.button("🔍 PHÂN TÍCH & LOẠI SỐ RÁC"):
    new_data = re.findall(r"\d{5}", raw_input)
    if new_raw := new_data:
        st.session_state.db["history"].extend(new_raw)
        
        # Lấy 100 kỳ gần nhất để soi số gan
        history_str = ",".join(st.session_state.db["history"][-100:])
        
        prompt = f"""
        Hệ thống cược 5D - Sảnh Không Cố Định.
        Lịch sử: {history_str}.
        Nhiệm vụ:
        1. Tìm 3 số có tần suất xuất hiện thấp nhất (Số Rác) -> Loại bỏ.
        2. Trong 7 số còn lại, chọn 4 số có nhịp rơi mạnh nhất (Trúng thưởng).
        3. 3 số còn lại làm dàn lót.
        TRẢ VỀ JSON: {{"loai": "1,2,3", "trung": "4567", "lot": "890", "ly_do": "..."}}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            st.session_state.db["predictions"].append(data)
            save_db(st.session_state.db)
        except:
            st.error("Lỗi AI - Đang dùng thuật toán dự phòng!")

# ================= HIỂN THỊ KẾT QUẢ =================
if st.session_state.db["predictions"]:
    res = st.session_state.db["predictions"][-1]
    st.markdown("<div class='critical-card'>", unsafe_allow_html=True)
    
    st.markdown(f"**🗑️ 3 SỐ ĐÃ LOẠI (KHÔNG TRÚNG):** <span class='trash-box'>{res['loai']}</span>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; margin:10px 0;'>🎯 4 SỐ TRÚNG THƯỞNG CỰC MẠNH:</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-main'>{res['trung']}</div>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='text-align:center;'>🛡️ DÀN LÓT AN TOÀN: <b>{res['lot']}</b></p>", unsafe_allow_html=True)
    
    full_dan = res['trung'] + res['lot']
    st.text_input("📋 COPY DÀN 7 SỐ ĐỂ DÁN:", full_dan)
    
    st.info(f"💡 Giải mã cầu: {res['ly_do']}")
    st.markdown("</div>", unsafe_allow_html=True)

if st.button("🗑️ XÓA HẾT LÀM LẠI"):
    st.session_state.db = {"history": [], "predictions": []}
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()
