import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter

# ================= CẤU HÌNH API MỚI =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DATA_FILE = "titan_history_v18.json"

# Khởi tạo AI
def init_ai():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

model = init_ai()

# ================= QUẢN LÝ DỮ LIỆU VĨNH VIỄN =================
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f: return json.load(f)
    return []

def save_data(history):
    with open(DATA_FILE, "w") as f:
        json.dump(history[-5000:], f) # Lưu tối đa 5000 kỳ gần nhất

if "db" not in st.session_state:
    st.session_state.db = load_data()

# ================= GIAO DIỆN PREMIUM =================
st.set_page_config(page_title="TITAN v18.0 GOLD", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #02040a; color: #ffd700; }
    .gold-card {
        background: linear-gradient(145deg, #0f172a, #1e293b);
        border: 1px solid #ffd700; border-radius: 15px; padding: 20px;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
    }
    .big-num { font-size: 48px; font-weight: 900; color: #ffffff; text-shadow: 0 0 15px #ffd700; text-align: center; }
    .stButton > button {
        background: linear-gradient(90deg, #ffd700, #b8860b);
        color: #000; border: none; font-weight: bold; border-radius: 8px; width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔱 TITAN v18.0 OMNI-GOLD")
status = "🟢 AI LIVE" if model else "🔴 API ERROR"
st.markdown(f"<p style='text-align: center;'>Trạng thái: <b>{status}</b> | Dữ liệu: <b>{len(st.session_state.db)} kỳ</b></p>", unsafe_allow_html=True)

# ================= XỬ LÝ CHÍNH =================
input_raw = st.text_area("📡 NẠP KỲ MỚI (Dán hàng loạt):", height=100)

c1, c2 = st.columns(2)
with c1:
    if st.button("🔥 PHÂN TÍCH & LƯU"):
        new_recs = re.findall(r"\d{5}", input_raw)
        if new_recs:
            st.session_state.db.extend(new_recs)
            save_data(st.session_state.db)
            
            # Gửi Prompt chuyên sâu cho AI
            prompt = f"""
            Bạn là hệ thống Neural xử lý dữ liệu 5D. 
            Lịch sử: {st.session_state.db[-50:]}.
            Yêu cầu:
            1. Phân tích chu kỳ lặp (Bệt) và chu kỳ đảo của 5 vị trí.
            2. Chốt dàn 7 số an toàn nhất (4 chính, 3 lót).
            3. Trả về JSON: {{"chinh": [4 số], "lot": [3 số], "logic": "tóm tắt chiến thuật"}}
            """
            try:
                response = model.generate_content(prompt)
                data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                st.session_state.result = data
            except:
                # Thuật toán dự phòng (Probability Fallback)
                all_nums = "".join(st.session_state.db[-20:])
                counts = Counter(all_nums).most_common(7)
                res = [str(x[0]) for x in counts]
                st.session_state.result = {"chinh": res[:4], "lot": res[4:], "logic": "Cầu nhiễu - Dùng xác suất thống kê."}
            st.rerun()

with c2:
    if st.button("🗑️ RESET DỮ LIỆU"):
        st.session_state.db = []
        save_data([])
        st.rerun()

# HIỂN THỊ KẾT QUẢ
if "result" in st.session_state:
    res = st.session_state.result
    st.markdown("<div class='gold-card'>", unsafe_allow_html=True)
    st.write(f"💡 **Tư duy:** {res['logic']}")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("🎯 4 CHỦ LỰC")
        st.markdown(f"<div class='big-num'>{''.join(map(str, res['chinh']))}</div>", unsafe_allow_html=True)
    with col_b:
        st.warning("🛡️ 3 LÓT")
        st.markdown(f"<div class='big-num' style='color:#ffd700;'>{''.join(map(str, res['lot']))}</div>", unsafe_allow_html=True)
    
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ:", "".join(map(str, res['chinh'])) + "".join(map(str, res['lot'])))
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Khuyên dùng: Nạp ít nhất 20 kỳ để AI đạt độ chính xác cao nhất.")
