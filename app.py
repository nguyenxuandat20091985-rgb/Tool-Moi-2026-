import streamlit as st
import google.generativeai as genai
import re
import json
import random
from collections import Counter
from itertools import combinations

# ================= CẤU HÌNH AI =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= UI STYLE v22 CHUẨN =================
st.set_page_config(page_title="TITAN v24.5 FORCE", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 2px solid #58a6ff;
        border-radius: 12px; padding: 20px; margin-top: 10px;
    }
    .num-box { font-size: 80px; font-weight: 900; color: #ff4b4b; text-align: center; letter-spacing: 15px; }
    .lot-box { font-size: 55px; font-weight: 700; color: #58a6ff; text-align: center; letter-spacing: 5px; }
    .status-bar { padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 1.2rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>⚡ TITAN v24.5 - BẢN ÉP NHẢY SỐ</h2>", unsafe_allow_html=True)

# Khởi tạo ID ngẫu nhiên để đánh lừa Cache
if "input_id" not in st.session_state:
    st.session_state.input_id = random.randint(1, 1000)
if "last_res" not in st.session_state:
    st.session_state.last_res = None

# ================= GIAO DIỆN NHẬP LIỆU =================
with st.container():
    # Tạo key động cho text_area dựa trên input_id
    raw_input = st.text_area(
        "📡 Dán dữ liệu mới nhất (Xóa sạch cũ rồi dán mới):", 
        height=150, 
        key=f"input_{st.session_state.input_id}"
    )
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔥 GIẢI MÃ & ÉP LÀM MỚI", use_container_width=True):
            if raw_input:
                with st.spinner('Đang ép AI chạy lại...'):
                    # 1. Lọc dữ liệu
                    lines = [l.strip() for l in raw_input.split('\n') if re.match(r"^\d{5}$", l.strip())]
                    
                    if len(lines) >= 5:
                        # 2. Thuật toán thống kê Combo nhanh
                        all_combos = []
                        for line in lines[:30]:
                            u = sorted(list(set(line)))
                            if len(u) >= 3: all_combos.extend(combinations(u, 3))
                        top_c = "".join(Counter(all_combos).most_common(1)[0][0]) if all_combos else "123"

                        # 3. Ép AI phân tích
                        prompt = f"Phân tích bộ 3 số 5 tinh (Combo hay về cùng nhau) cho 5D từ: {lines[:40]}. Trả về JSON: {{\"m\": \"abc\", \"s\": \"defg\", \"d\": \"ĐÁNH\", \"l\": \"...\", \"c\": \"Green\"}}"
                        try:
                            resp = neural_engine.generate_content(prompt)
                            st.session_state.last_res = json.loads(re.search(r'\{.*\}', resp.text, re.DOTALL).group())
                        except:
                            st.session_state.last_res = {"m": top_c, "s": "0456", "d": "ĐÁNH", "l": "Thống kê Combo trực tiếp", "c": "Green"}
                        
                        # ĐỔI ID ĐỂ KỲ SAU KHÔNG BỊ TRÙNG CACHE
                        st.session_state.input_id += 1
                        st.rerun()
            else:
                st.error("Chưa dán số anh ơi!")
                
    with c2:
        if st.button("🗑️ XÓA SẠCH DỮ LIỆU", use_container_width=True):
            st.session_state.last_res = None
            st.session_state.input_id += 1
            st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if st.session_state.last_res:
    res = st.session_state.last_res
    bg = "#238636" if res['c'].lower() == "green" else "#da3633"
    
    st.markdown(f"<div class='status-bar' style='background: {bg};'>📢 TRẠNG THÁI: {res['d']}</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    col_main, col_supp = st.columns([1.5, 1])
    with col_main:
        st.markdown("<p style='text-align:center; color:#8b949e;'>💎 3 SỐ 5 TINH CHỦ LỰC</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['m']}</div>", unsafe_allow_html=True)
    with col_supp:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res['s']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.write(f"🧠 **LOGIC:** {res['l']}")
    full_dan = "".join(sorted(set(res['m'] + res['s'])))
    st.text_input("📋 DÀN 7 SỐ KUBET:", full_dan)
    st.markdown("</div>", unsafe_allow_html=True)
