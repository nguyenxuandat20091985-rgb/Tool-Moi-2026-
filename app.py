import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter
import itertools

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v21.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= HỆ THỐNG GHI NHỚ VĨNH VIỄN =================
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

# ================= UI DESIGN (GIỮ NGUYÊN UI v21) =================
st.set_page_config(page_title="TITAN v21.0 PRO - NHÓM 24", layout="centered")
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
        font-size: 45px; font-weight: 900; color: #58a6ff; 
        text-align: center; letter-spacing: 5px; text-shadow: 0 0 25px #58a6ff;
    }
    .logic-box { font-size: 14px; color: #8b949e; background: #161b22; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🧬 TITAN v21.0 OMNI - NHÓM 24</h2>", unsafe_allow_html=True)
if neural_engine:
    st.markdown(f"<p class='status-active'>● KẾT NỐI NEURAL-LINK: OK | CHẾ ĐỘ: NHÓM 24 TỔ HỢP</p>", unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU & THUẬT TOÁN MỚI =================
raw_input = st.text_area("📡 NẠP DỮ LIỆU (Dán các dãy 5 số):", height=100, placeholder="32880\n21808\n...")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 GIẢI MÃ THUẬT TOÁN"):
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            save_memory(st.session_state.history)
            
            # NÂNG CẤP PROMPT: CHUYÊN BIỆT CHO NHÓM 24
            prompt = f"""
            Bạn là AI chuyên gia xác suất Hậu Tứ Nhóm 24. 
            Lịch sử 100 kỳ: {st.session_state.history[-100:]}.
            Yêu cầu:
            1. Phân tích 4 số cuối (Ngàn, Trăm, Chục, Đơn vị).
            2. Tìm 7 số có xác suất xuất hiện cùng nhau cao nhất nhưng không lặp lại trong 1 bộ.
            3. Trả về 4 tổ hợp mạnh nhất (mỗi tổ hợp 4 số khác nhau) và dàn 7 số tổng.
            TRẢ VỀ JSON: {{"tohop": ["1234", "5678", "1357", "2468"], "dan7": "1234567", "logic": "Dữ liệu bệt tổ hợp 4 số không lặp"}}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                st.session_state.last_result = data
            except:
                # Thuật toán dự phòng (Lọc 7 số rồi ghép tổ hợp ngẫu nhiên không lặp)
                all_nums = "".join([s[1:] for s in st.session_state.history[-30:]]) # Chỉ lấy 4 số cuối
                counts = [x[0] for x in Counter(all_nums).most_common(7)]
                combos = ["".join(p) for p in itertools.combinations(counts, 4)][:4]
                st.session_state.last_result = {"tohop": combos, "dan7": "".join(counts), "logic": "Thống kê tổ hợp xác suất thực tế."}
            st.rerun()

with col2:
    if st.button("🗑️ RESET BỘ NHỚ"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='logic-box'><b>💡 Phân tích Nhóm 24:</b> {res['logic']}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; font-size:12px; color:#888;'>🎯 4 TỔ HỢP MẠNH NHẤT (VÀO TIỀN)</p>", unsafe_allow_html=True)
    # Hiển thị các tổ hợp cách nhau bằng dấu phẩy để anh dễ nhìn
    st.markdown(f"<div class='num-display'>{', '.join(res['tohop'])}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; font-size:12px; color:#888; margin-top:20px;'>🛡️ DÀN 7 SỐ TỔNG (ĐỂ ANH TỰ GHÉP THÊM)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display' style='color:#f2cc60; text-shadow: 0 0 25px #f2cc60;'>{res['dan7']}</div>", unsafe_allow_html=True)
    
    # Mục copy dán thẳng vào web
    st.text_input("📋 SAO CHÉP DÀN TỔ HỢP (Dán vào mục Nhập Số):", ", ".join(res['tohop']))
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><p style='text-align:center; font-size:10px; color:#444;'>Thiết kế nâng cấp riêng cho Nhóm 24 - Không lặp số</p>", unsafe_allow_html=True)
