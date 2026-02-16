import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter
import itertools

# ================= CONFIG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v30.json"

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
        json.dump(data[-800:], f)

if "history" not in st.session_state:
    st.session_state.history = load_memory()

# ================= UI MATRIX DESIGN =================
st.set_page_config(page_title="TITAN v30.0 MATRIX", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #000b00; color: #00ff41; }
    .prediction-card {
        background: rgba(0, 40, 0, 0.9); border: 2px solid #00ff41;
        border-radius: 15px; padding: 20px; box-shadow: 0 0 20px #00ff41;
    }
    .num-display { 
        font-family: 'Courier New', monospace; font-size: 20px; 
        color: #00ff41; background: #000; padding: 15px; border-radius: 10px;
        line-height: 1.6; border: 1px solid #00ff41;
    }
    .highlight-label { color: #ffffff; font-weight: bold; font-size: 14px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>📟 TITAN v30.0 OMNI-MATRIX</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa;'>HẬU TỨ NHÓM 24 - CHIẾN THUẬT VÂY BẮT 20 TỔ HỢP</p>", unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU =================
raw_input = st.text_area("📥 DÁN LỊCH SỬ KỲ (5 SỐ):", height=100)

if st.button("📡 GIẢI MÃ MA TRẬN"):
    new_data = re.findall(r"\d{5}", raw_input)
    if new_data:
        st.session_state.history.extend(new_data)
        save_memory(st.session_state.history)
        
        prompt = f"""
        Hệ thống phân tích Nhóm 24. Dữ liệu: {st.session_state.history[-100:]}.
        Nhiệm vụ:
        1. Tìm ra 8 con số gốc (8-digit core) xuất hiện nhiều và có nhịp hồi đẹp.
        2. Từ 8 số đó, lọc ra 20 tổ hợp 4 số (không lặp) có khả năng nổ cao nhất.
        3. Định dạng: Danh sách các tổ hợp cách nhau bằng dấu phẩy.
        TRẢ VỀ JSON: {{"combos": [], "core8": "12345678", "logic": "..."}}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            st.session_state.matrix_res = data
        except:
            # Thuật toán dự phòng Matrix
            last_nums = "".join([s[1:] for s in st.session_state.history[-40:]])
            core8 = [x[0] for x in Counter(last_nums).most_common(8)]
            combos = ["".join(p) for p in itertools.combinations(core8, 4)][:20]
            st.session_state.matrix_res = {"combos": combos, "core8": "".join(core8), "logic": "Dữ liệu Matrix dự phòng."}
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if "matrix_res" in st.session_state:
    res = st.session_state.matrix_res
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#fff;'><b>💡 Logic:</b> {res['logic']}</p>", unsafe_allow_html=True)
    
    st.markdown("<p class='highlight-label'>🎯 DÀN 20 TỔ HỢP (CHẠM ĐỂ COPY):</p>", unsafe_allow_html=True)
    copy_string = ", ".join(res['combos'])
    st.text_area("", value=copy_string, height=150, key="copy_area")
    
    st.markdown(f"<p class='highlight-label'>🛡️ 8 SỐ GỐC: <span style='color:#00ff41;'>{res['core8']}</span></p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if st.button("🗑️ RESET"):
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()
