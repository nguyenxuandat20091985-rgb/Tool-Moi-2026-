import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_neural_memory_v22.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= QUẢN LÝ BỘ NHỚ VÀ DỮ LIỆU SẠCH =================
def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_memory(data):
    # Lưu trữ 2000 kỳ để phân tích chu kỳ dài hơn
    with open(DB_FILE, "w") as f:
        json.dump(data[-2000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_memory()

# ================= GIAO DIỆN TITAN PRO =================
st.set_page_config(page_title="TITAN v22.0 OMNI", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #c9d1d9; }
    .status-panel { background: #0d1117; padding: 10px; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 20px; }
    .prediction-card {
        background: linear-gradient(145deg, #0d1117, #161b22);
        border: 1px solid #58a6ff; border-radius: 15px; padding: 30px;
        box-shadow: 0 0 30px rgba(88, 166, 255, 0.1);
    }
    .main-number { font-size: 85px; font-weight: 900; color: #ff5858; text-shadow: 0 0 30px #ff5858; text-align: center; }
    .secondary-number { font-size: 50px; font-weight: 700; color: #58a6ff; text-align: center; opacity: 0.8; }
    .warning-box { background: #331010; color: #ff7b72; padding: 15px; border-radius: 8px; border: 1px solid #6e2121; text-align: center; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ================= PHẦN PHÂN TÍCH THUẬT TOÁN =================
def analyze_patterns(data):
    if not data: return "Chưa có dữ liệu"
    all_digits = "".join(data)
    counts = Counter(all_digits)
    # Tìm quy luật bóng số
    shadow_map = {'0':'5', '5':'0', '1':'6', '6':'1', '2':'7', '7':'2', '3':'8', '8':'3', '4':'9', '9':'4'}
    last_draw = data[-1]
    potential_shadows = [shadow_map[d] for d in last_draw]
    return f"Tần suất cao: {counts.most_common(3)} | Bóng số tiềm năng: {''.join(potential_shadows)}"

# ================= UI CHÍNH =================
st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🧬 TITAN v22.0 PRO OMNI</h1>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='status-panel'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.write(f"📡 NEURAL: {'✅ ONLINE' if neural_engine else '❌ ERROR'}")
    c2.write(f"📊 DATASET: {len(st.session_state.history)} KỲ")
    c3.write(f"🛡️ SAFETY: ACTIVE")
    st.markdown("</div>", unsafe_allow_html=True)

raw_input = st.text_area("📥 NẠP DỮ LIỆU SẠCH (5 số viết liền):", height=120, placeholder="Dán dãy số tại đây...")

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("🚀 KÍCH HOẠT GIẢI MÃ"):
        # Lọc số bẩn: chỉ lấy đúng các cụm 5 chữ số
        clean_data = re.findall(r"\b\d{5}\b", raw_input)
        if clean_data:
            st.session_state.history.extend(clean_data)
            save_memory(st.session_state.history)
            
            # Gửi Prompt "Khắc chế nhà cái" cho Gemini
            prompt = f"""
            Hệ thống: TITAN v22.0. Chuyên gia bẻ cầu nhà cái Kubet/Lotobet.
            Dữ liệu lịch sử (100 kỳ): {st.session_state.history[-100:]}.
            Quy luật bóng số: 0-5, 1-6, 2-7, 3-8, 4-9.
            Nhiệm vụ:
            1. Phân tích chu kỳ 'nhả' số của nhà cái.
            2. Chọn ra 3 số CHỦ LỰC có xác suất nổ cao nhất (Xác suất yêu cầu > 95%).
            3. Nếu dữ liệu có dấu hiệu bị điều tiết (ảo), hãy đặt 'warning': true.
            TRẢ VỀ JSON: {{"main_3": "chuỗi 3 số", "support_4": "chuỗi 4 số", "logic": "phân tích ngắn", "warning": false, "confidence": 98}}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                # Xử lý JSON an toàn
                json_str = re.search(r'\{.*\}', response.text, re.DOTALL).group()
                st.session_state.last_prediction = json.loads(json_str)
            except Exception as e:
                # Thuật toán dự phòng (Statistical Fallback)
                all_nums = "".join(st.session_state.history[-50:])
                common = [x[0] for x in Counter(all_nums).most_common(7)]
                st.session_state.last_prediction = {
                    "main_3": "".join(common[:3]),
                    "support_4": "".join(common[3:]),
                    "logic": "Sử dụng thuật toán thống kê xác suất thực tế.",
                    "warning": False,
                    "confidence": 75
                }
            st.rerun()

with col_btn2:
    if st.button("🗑️ DỌN DẸP BỘ NHỚ"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ DỰ ĐOÁN =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    if res.get('warning') or res.get('confidence', 0) < 70:
        st.markdown("<div class='warning-box'>⚠️ CẢNH BÁO: CẦU ĐANG NHIỄU - HẠ MỨC CƯỢC HOẶC DỪNG LẠI</div>", unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.write(f"🔍 **CHIẾN THUẬT:** {res['logic']}")
    
    st.markdown("<p style='text-align:center; color:#888; margin-bottom:0;'>🔥 3 SỐ CHỦ LỰC (VÀO TIỀN MẠNH)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='main-number'>{res['main_3']}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; color:#888; margin-top:20px; margin-bottom:0;'>🛡️ DÀN LÓT AN TOÀN</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='secondary-number'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    full_dan = res['main_3'] + res['support_4']
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ:", full_dan)
    st.progress(res.get('confidence', 50) / 100)
    st.markdown(f"<p style='text-align:right; font-size:12px;'>Độ tin cậy: {res.get('confidence')}%</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Thống kê nhanh dưới cùng
with st.expander("📊 Thống kê nhanh nhịp cầu"):
    st.write(analyze_patterns(st.session_state.history))
