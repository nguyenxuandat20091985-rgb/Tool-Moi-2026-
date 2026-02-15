import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter
import itertools

# ================= CONFIG HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v28.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= QUẢN LÝ BỘ NHỚ THÔNG MINH =================
def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: 
            try: return json.load(f)
            except: return []
    return []

def save_memory(data):
    # Giữ 500 kỳ để AI không bị loãng dữ liệu
    with open(DB_FILE, "w") as f: 
        json.dump(data[-500:], f)

if "history" not in st.session_state:
    st.session_state.history = load_memory()

# ================= GIAO DIỆN TITAN OMNI-FLOW =================
st.set_page_config(page_title="TITAN v28.0 OMNI-FLOW", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #050505; color: #e2e8f0; }
    .status-panel { background: #1e293b; padding: 10px; border-radius: 8px; border-left: 5px solid #10b981; margin-bottom: 15px; }
    .prediction-card {
        background: linear-gradient(145deg, #0f172a, #1e293b);
        border: 1px solid #334155; border-radius: 20px; padding: 25px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.7);
    }
    .num-main { 
        font-size: 38px; font-weight: 800; color: #60a5fa; 
        text-align: center; letter-spacing: 2px; text-shadow: 0 0 15px rgba(96, 165, 250, 0.5);
    }
    .copy-box { background: #000; color: #10b981; padding: 15px; border-radius: 10px; border: 1px dashed #10b981; font-family: monospace; font-size: 18px; text-align: center; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #60a5fa;'>🧬 TITAN v28.0 OMNI-FLOW</h2>", unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU & THUẬT TOÁN FLOW =================
raw_input = st.text_area("📡 DÁN DỮ LIỆU (5 số mỗi dòng):", height=100, placeholder="Dán kết quả tại đây...")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 GIẢI MÃ OMNI-FLOW"):
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            save_memory(st.session_state.history)
            
            # PROMPT ÉP AI TRUY HỒI SAI SỐ
            prompt = f"""
            Hệ thống phân tích Hậu Tứ Nhóm 24.
            Dữ liệu gần đây: {st.session_state.history[-60:]}.
            Yêu cầu chuyên sâu:
            1. Phân tích nhịp rơi của 4 số cuối, loại bỏ các kỳ có số lặp (kép).
            2. Tìm 7 số gốc có xác suất nổ cao nhất trong 3 kỳ tới.
            3. Ghép thành 6 tổ hợp 4 số khác nhau hoàn toàn.
            TRẢ VỀ JSON: {{"combos": ["1234", "2345", "3456", "4567", "5678", "6789"], "dan7": "1234567", "logic": "Giải thích hướng đi của cầu"}}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                st.session_state.last_result = data
            except:
                # Thuật toán dự phòng OMNI-FLOW
                last_30 = "".join([s[1:] for s in st.session_state.history[-30:]])
                top_7 = [x[0] for x in Counter(last_30).most_common(7)]
                combos = ["".join(p) for p in itertools.combinations(top_7, 4)][:6]
                st.session_state.last_result = {"combos": combos, "dan7": "".join(top_7), "logic": "Sử dụng bộ lọc Flow-Logic dự phòng."}
            st.rerun()

with col2:
    if st.button("🗑️ RESET TOOL"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ ĐỂ COPY =================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #94a3b8; font-size: 14px;'>💡 <b>Phân tích:</b> {res['logic']}</p>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; font-size:12px; color:#64748b;'>🎯 6 TỔ HỢP NHÓM 24 CỰC MẠNH</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-main'>{', '.join(res['combos'])}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; font-size:12px; color:#64748b; margin-top:15px;'>🛡️ DÀN 7 SỐ GỐC (DỰ PHÒNG):</p>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center; color:#facc15; font-size:24px; font-weight:bold;'>{res['dan7']}</div>", unsafe_allow_html=True)
    
    # Ô copy tối ưu cho mobile
    copy_text = ", ".join(res['combos'])
    st.markdown("<p style='font-size:12px; margin-top:20px;'>📋 CHẠM ĐỂ COPY DÀN (DÁN VÀO MỤC NHẬP SỐ):</p>", unsafe_allow_html=True)
    st.text_area("Copy tại đây:", value=copy_text, height=70)
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><p style='text-align:center; font-size:10px; color:#444;'>Phiên bản v28.0 - Tối ưu hóa cho Copy-Paste nhanh</p>", unsafe_allow_html=True)
