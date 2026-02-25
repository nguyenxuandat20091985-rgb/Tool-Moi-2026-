import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter

# ================= CONFIG V23.0 =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_v23_core.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= THUẬT TOÁN MA TRẬN VỊ TRÍ (MỚI) =================
def matrix_position_analysis(data):
    if len(data) < 10: return {}
    # Tạo ma trận 5 cột (tương ứng 5 vị trí giải ĐB)
    matrix = np.array([[int(d) for d in str(k)] for k in data[-100:]])
    pos_stats = {}
    for i in range(5):
        col = matrix[:, i]
        most_common = Counter(col).most_common(2)
        pos_stats[f"Vị trí {i+1}"] = [str(x[0]) for x in most_common]
    return pos_stats

# ================= GIAO DIỆN TITAN v23.0 =================
st.set_page_config(page_title="TITAN v23.0 - MATRIX AI", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #c9d1d9; }
    .main-card { background: #0d1117; border: 1px solid #238636; border-radius: 15px; padding: 25px; }
    .matrix-box { background: #161b22; border: 1px dashed #58a6ff; padding: 10px; border-radius: 8px; font-family: monospace; }
    .confidence-high { color: #238636; font-weight: bold; font-size: 24px; }
    .bet-advice { background: #1b1100; border-left: 5px solid #d29922; padding: 15px; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

# Khởi tạo bộ nhớ
if "history" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.history = json.load(f)
    else: st.session_state.history = []

# ================= UI CHÍNH =================
st.markdown("<h1 style='text-align: center; color: #238636;'>🧬 TITAN v23.0 - MATRIX AI SYSTEM</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Hệ thống")
    if st.button("🗑️ Xóa sạch dữ liệu"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()
    st.divider()
    st.write("📊 **Phân tích Ma Trận Vị Trí:**")
    pos_data = matrix_position_analysis(st.session_state.history)
    for pos, vals in pos_data.items():
        st.write(f"{pos}: **{', '.join(vals)}**")

# Nhập liệu
raw_input = st.text_area("📥 NHẬP DỮ LIỆU KỲ MỚI:", height=100, placeholder="Dán dãy 5 số vào đây...")

if st.button("🚀 GIẢI MÃ MA TRẬN & SOI CẦU"):
    new_data = re.findall(r"\b\d{5}\b", raw_input)
    if new_data:
        st.session_state.history.extend(new_data)
        with open(DB_FILE, "w") as f: json.dump(st.session_state.history[-2000:], f)
        
        # PHÂN TÍCH MA TRẬN TRƯỚC KHI GỬI AI
        pos_summary = str(matrix_position_analysis(st.session_state.history))
        
        prompt = f"""
        Hệ thống: TITAN v23.0 (Matrix-Neural Hybrid).
        Mục tiêu: 3 Càng không cố định (3D 5 tinh).
        Lịch sử: {st.session_state.history[-50:]}
        Thống kê vị trí (Ma trận): {pos_summary}
        
        Yêu cầu:
        1. Sử dụng thuật toán Bóng số âm dương để lọc 3 số chủ lực.
        2. Dựa vào ma trận vị trí để tìm điểm rơi (Hot spots).
        3. Kết quả là dàn 7 số KHÔNG TRÙNG (3 chính + 4 lót).
        4. Gợi ý mức tiền vào (Bet size) dựa trên độ tự tin.

        TRẢ VỀ JSON:
        {{
            "main_3": "ABC",
            "support_4": "DEFG",
            "logic": "ngắn gọn",
            "warning": false,
            "confidence": 98,
            "bet_advice": "Mô tả cách vào tiền"
        }}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            json_str = re.search(r'\{.*\}', response.text, re.DOTALL).group()
            st.session_state.last_prediction = json.loads(json_str)
        except Exception as e:
            st.error("Lỗi AI: Không thể giải mã JSON. Thử lại sau 30 giây.")
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    if res['warning']:
        st.markdown("<div style='color:#ff4b4b; border:1px solid red; padding:10px; text-align:center;'>⚠️ CẢNH BÁO: PHÁT HIỆN CẦU ẢO - DỪNG CƯỢC</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🎯 3 SỐ CHỦ LỰC")
        st.markdown(f"<h1 style='font-size:80px; color:#58a6ff; letter-spacing:10px;'>{res['main_3']}</h1>", unsafe_allow_html=True)
        st.write(f"**💡 Logic:** {res['logic']}")
    
    with col2:
        st.subheader("🛡️ 4 SỐ LÓT")
        st.markdown(f"<h1 style='font-size:40px; color:#8b949e;'>{res['support_4']}</h1>", unsafe_allow_html=True)
        st.markdown(f"Độ tin cậy: <span class='confidence-high'>{res['confidence']}%</span>", unsafe_allow_html=True)

    st.markdown(f"<div class='bet-advice'>💰 **GỢI Ý VÀO TIỀN:** {res['bet_advice']}</div>", unsafe_allow_html=True)
    
    st.divider()
    full_7 = "".join(sorted(set(res['main_3'] + res['support_4'])))
    st.text_input("📋 DÀN 7 SỐ KUBET (CHỌN TRÊN WEB):", full_7)
    st.markdown("</div>", unsafe_allow_html=True)

# Biểu đồ Ma Trận
if st.session_state.history:
    st.divider()
    st.subheader("📈 Bản đồ Ma Trận Nhiệt (Heatmap)")
    
    # (Phần này anh có thể xem qua sidebar đã có thống kê chi tiết)
