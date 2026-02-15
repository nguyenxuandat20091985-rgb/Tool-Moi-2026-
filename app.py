import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter
import itertools

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v26.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU SẠCH =================
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

# ================= UI DESIGN (GIỮ NGUYÊN UI) =================
st.set_page_config(page_title="TITAN v26.0 PRO", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #020617; color: #f8fafc; }
    .status-ok { color: #10b981; font-weight: bold; border-bottom: 2px solid #10b981; }
    .prediction-card {
        background: #0f172a; border: 1px solid #1e293b;
        border-radius: 16px; padding: 25px; margin-top: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    }
    .num-main { 
        font-size: 40px; font-weight: 800; color: #38bdf8; 
        text-align: center; letter-spacing: 3px;
    }
    .logic-box { font-size: 14px; color: #94a3b8; background: #1e293b; padding: 12px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid #38bdf8; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #38bdf8;'>🧬 TITAN v26.0 NEURAL-LOGIC</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 12px;'>CHUYÊN BIỆT HẬU TỨ NHÓM 24 - CHỐNG CẦU ẢO</p>", unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU & AI =================
raw_input = st.text_area("📡 NẠP DỮ LIỆU (Dán các dãy 5 số):", height=100)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 PHÂN TÍCH TỔ HỢP"):
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            save_memory(st.session_state.history)
            
            # PROMPT ÉP AI LỌC SỐ SẠCH (KHÔNG LẶP)
            prompt = f"""
            Bạn là hệ thống phân tích xác suất Nhóm 24.
            Dữ liệu Hậu Tứ (4 số cuối): {st.session_state.history[-50:]}.
            Luật Nhóm 24: 4 số mở thưởng phải khác nhau hoàn toàn.
            Yêu cầu:
            1. Loại bỏ các kỳ có số lặp trong 4 số cuối khỏi phân tích.
            2. Tìm 7 số đơn lẻ có nhịp về ổn định nhất.
            3. Ghép thành 5 tổ hợp 4 số khác nhau (ví dụ: 1234, 2345...).
            TRẢ VỀ JSON: {{"combos": ["1234", "2345", "3456", "4567", "5678"], "dan7": "1234567", "logic": "Giải thích nhịp cầu"}}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                st.session_state.last_result = data
            except:
                # Thuật toán dự phòng nếu AI bận
                all_raw = "".join([s[1:] for s in st.session_state.history[-30:]])
                counts = [x[0] for x in Counter(all_raw).most_common(7)]
                # Tự ghép tổ hợp thủ công từ 7 số mạnh nhất
                combos = ["".join(p) for p in itertools.combinations(counts, 4)][:5]
                st.session_state.last_result = {"combos": combos, "dan7": "".join(counts), "logic": "Thống kê tần suất tổ hợp sạch."}
            st.rerun()

with col2:
    if st.button("🗑️ RESET"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='logic-box'><b>💡 Chiến thuật:</b> {res['logic']}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; font-size:12px; color:#64748b;'>🎯 5 TỔ HỢP NHÓM 24 (VÀO TIỀN)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-main'>{', '.join(res['combos'])}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; font-size:12px; color:#64748b; margin-top:20px;'>🛡️ DÀN 7 SỐ GỐC</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-main' style='color:#facc15;'>{res['dan7']}</div>", unsafe_allow_html=True)
    
    st.text_input("📋 COPY DÁN VÀO WEB:", ", ".join(res['combos']))
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Khuyên dùng: Theo dõi 3-5 kỳ trước khi vào tiền để khớp nhịp AI.")
