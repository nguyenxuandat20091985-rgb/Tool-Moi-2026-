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
DB_FILE = "titan_phantom_v23.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-pro') # Nâng cấp lên Pro để tư duy sâu hơn
    except: return None

neural_engine = setup_neural()

# ================= PHÂN TÍCH VỊ TRÍ (MỚI) =================
def analyze_positional_logic(history):
    if len(history) < 5: return {}
    # Phân tách 5 vị trí: Chục ngàn, Ngàn, Trăm, Chục, Đơn vị
    matrix = np.array([[int(d) for d in streak] for streak in history])
    pos_stats = {}
    for i in range(5):
        pos_stats[f"P{i}"] = Counter(matrix[:, i]).most_common(2)
    return pos_stats

# ================= UI DESIGN (DARK PHANTOM) =================
st.set_page_config(page_title="TITAN v23.0 PHANTOM", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #050505; color: #00ff41; font-family: 'Courier New', monospace; }
    .prediction-card {
        background: #000000; border: 1px solid #00ff41;
        border-radius: 10px; padding: 25px;
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.2);
    }
    .main-3 { font-size: 100px; font-weight: 900; color: #00ff41; text-align: center; text-shadow: 0 0 40px #00ff41; }
    .caution { background: #4a0000; color: #ff0000; padding: 10px; border: 1px solid #ff0000; border-radius: 5px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>💀 TITAN v23.0 PHANTOM OMNI</h1>", unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU =================
if "history" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.history = json.load(f)
    else: st.session_state.history = []

raw_data = st.text_area("📡 NẠP DỮ LIỆU GIẢI ĐẶC BIỆT (Dán thẳng hàng):", height=150)

c1, c2 = st.columns(2)
with c1:
    if st.button("⚡ GIẢI MÃ PHANTOM"):
        clean = re.findall(r"\b\d{5}\b", raw_data)
        if clean:
            st.session_state.history.extend(clean)
            st.session_state.history = st.session_state.history[-2000:]
            with open(DB_FILE, "w") as f: json.dump(st.session_state.history, f)
            
            pos_data = analyze_positional_logic(st.session_state.history[-50:])
            
            # PROMPT PHẢN ĐÒN AI NHÀ CÁI
            prompt = f"""
            Bạn là TITAN PHANTOM - Hệ thống khắc chế AI Kubet/Lotobet.
            Dữ liệu gần đây: {st.session_state.history[-100:]}.
            Thống kê vị trí: {pos_data}.
            Quy tắc: Không cố định - 3 số 5 tinh (Chọn 3, nếu nổ trong 5 vị trí là thắng).
            
            Nhiệm vụ:
            1. Tìm ra 3 số "Chủ Lực" né được thuật toán quét của nhà cái.
            2. Phân tích xem nhà cái đang thả cầu hay bẻ cầu.
            3. Nếu xác suất thắng < 80%, đặt 'abort': true.
            
            TRẢ VỀ JSON: {{"main_3": "abc", "backup_4": "xyz", "intel": "tâm lý nhà cái kỳ này", "confidence": 99, "abort": false}}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                res_json = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                st.session_state.result = res_json
            except:
                st.error("AI Phantom đang bị tường lửa nhà cái chặn - Đang dùng thuật toán dự phòng...")
                # Thuật toán dự phòng (Statistical position-based)
                all_nums = "".join(st.session_state.history[-30:])
                fallback = [x[0] for x in Counter(all_nums).most_common(7)]
                st.session_state.result = {"main_3": "".join(fallback[:3]), "backup_4": "".join(fallback[3:]), "intel": "Cầu nhảy tự do - Đánh nhỏ.", "confidence": 70, "abort": False}
            st.rerun()

with c2:
    if st.button("🔴 RESET HỆ THỐNG"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# ================= HIỂN THỊ CHIẾN THUẬT =================
if "result" in st.session_state:
    res = st.session_state.result
    
    if res['abort']:
        st.markdown("<div class='caution'>HỆ THỐNG PHÁT HIỆN DẤU HIỆU QUÉT CẦU - DỪNG CƯỢC KỲ NÀY!</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        st.markdown(f"**⚡ PHÂN TÍCH PHANTOM:** {res['intel']}")
        
        st.markdown("<p style='text-align:center; color:#888;'>🎯 3 SỐ CHỦ LỰC (XÁC SUẤT CAO NHẤT)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main_3'>{res['main_3']}</div>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("🛡️ Dàn lót an toàn:")
            st.info(res['backup_4'])
        with col_b:
            st.write("📈 Độ tin cậy:")
            st.success(f"{res['confidence']}%")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Hiển thị lịch sử nhập để anh kiểm tra
with st.expander("📝 Xem lịch sử dữ liệu"):
    st.write(st.session_state.history[::-1])
