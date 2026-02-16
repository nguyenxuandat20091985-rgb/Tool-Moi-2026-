import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_v31_elite.json"

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
            except: return {"history": [], "last_pred": None}
    return {"history": [], "last_pred": None}

def save_memory(data):
    with open(DB_FILE, "w") as f: 
        json.dump(data, f)

if "db" not in st.session_state:
    st.session_state.db = load_memory()

# ================= UI ELITE DESIGN =================
st.set_page_config(page_title="TITAN v31.0 ELITE", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #000814; color: #e0e1dd; }
    .main-card { background: #0d1b2a; border: 1px solid #415a77; border-radius: 15px; padding: 20px; box-shadow: 0 4px 30px rgba(0,255,136,0.1); }
    .num-target { font-size: 70px; font-weight: 900; color: #00ff88; text-align: center; text-shadow: 0 0 20px #00ff88; }
    .num-sub { font-size: 30px; font-weight: 700; color: #f2cc60; text-align: center; }
    .logic-box { font-size: 14px; color: #8d99ae; background: #1b263b; padding: 12px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #00ff88; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #00ff88;'>🛡️ TITAN v31.0 ELITE</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 12px;'>Dành riêng cho: 3 Tinh Không cố định (7 Loại 3)</p>", unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU =================
raw_input = st.text_area("📡 NẠP DỮ LIỆU (Dán 5 số mỗi kỳ):", height=100)

col1, col2 = st.columns(2)
with col1:
    if st.button("🔥 GIẢI MÃ KHÔNG CỐ ĐỊNH"):
        # Lấy 5 số nhưng chỉ phân tích 3 số cuối (Hàng Trăm - Chục - Đơn vị)
        new_raw = re.findall(r"\d{5}", raw_input)
        if new_raw:
            clean_data = [s[2:] for s in new_raw] # Cắt lấy 3 số cuối
            st.session_state.db["history"].extend(clean_data)
            
            # PROMPT ÉP AI LOẠI 3 SỐ XẤU
            prompt = f"""
            Bạn là chuyên gia phân tích 3 Tinh Không cố định.
            Dữ liệu 3 số cuối (Trăm-Chục-Đơn): {st.session_state.db["history"][-50:]}.
            Yêu cầu:
            1. Loại bỏ 3 con số có xác suất về thấp nhất (dựa trên cầu bệt và gan).
            2. Trong 7 số còn lại, chọn ra 4 SỐ CHỦ LỰC trúng thưởng cao nhất.
            3. 3 SỐ LÓT để bọc lót.
            Trả về JSON: {{"dan4": ["x","x","x","x"], "dan3": ["x","x","x"], "logic": "phân tích ngắn gọn"}}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                st.session_state.db["last_pred"] = data
                save_memory(st.session_state.db)
            except:
                # Dự phòng nếu AI lỗi: Thống kê 3 số cuối
                all_3 = "".join(st.session_state.db["history"][-20:])
                counts = [x[0] for x in Counter(all_3).most_common(7)]
                st.session_state.db["last_pred"] = {"dan4": counts[:4], "dan3": counts[4:], "logic": "Dùng tần suất 3 số cuối."}
            st.rerun()

with col2:
    if st.button("🗑️ RESET DỮ LIỆU"):
        st.session_state.db = {"history": [], "last_pred": None}
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if st.session_state.db["last_pred"]:
    res = st.session_state.db["last_pred"]
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='logic-box'><b>💎 Chiến thuật:</b> {res['logic']}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; color:#888;'>🎯 4 SỐ CHỦ LỰC (VÀO TIỀN)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-target'>{''.join(map(str, res['dan4']))}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; color:#888; margin-top:20px;'>🛡️ 3 SỐ LÓT (GIỮ VỐN)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-sub'>{''.join(map(str, res['dan3']))}</div>", unsafe_allow_html=True)
    
    copy_val = "".join(map(str, res['dan4'])) + "".join(map(str, res['dan3']))
    st.text_input("📋 COPY DÀN 7 SỐ:", copy_val)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><p style='text-align:center; font-size:10px; color:#444;'>Tự động lọc nhiễu 2 số đầu - Tập trung 3 số cuối</p>", unsafe_allow_html=True)
