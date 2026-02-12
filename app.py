import streamlit as st
import re
import json
import numpy as np
from collections import Counter

# ================= CONFIG SIÊU DI ĐỘNG (X-MOBILE) =================
st.set_page_config(page_title="TITAN v7000", layout="centered")

st.markdown("""
    <style>
    /* Ép giao diện về dạng Mobile App */
    .main { background-color: #000; color: #fff; padding: 5px; }
    [data-testid="stHeader"] {display: none;}
    .stNumberInput, .stButton, .stTextArea { margin-bottom: 5px; }
    .stButton > button {
        background: linear-gradient(90deg, #ff0055, #ff5500);
        color: white; border: none; border-radius: 5px; width: 100%; height: 35px; font-weight: bold;
    }
    .result-box {
        background: #111; border: 2px solid #ff0055; border-radius: 10px;
        padding: 10px; text-align: center; margin-top: 5px;
    }
    .prediction { font-size: 38px; font-weight: 900; color: #00ff00; margin: 0; }
    .mini-text { font-size: 10px; color: #888; }
    </style>
""", unsafe_allow_html=True)

# Khởi tạo Database gọn nhẹ
if "db_x" not in st.session_state: st.session_state.db_x = []

def analyze_x_mobile(data):
    if len(data) < 10: return None
    
    # Lấy 21 kỳ gần nhất (Số vàng trong xác suất 5D)
    recent = data[-21:]
    matrix = np.array([[int(d) for d in list(ky)] for ky in recent])
    
    # 1. Bắt số (Weighting by recency)
    flat_data = "".join(recent)
    counts = Counter(flat_data)
    # Tăng trọng số cho 3 kỳ gần nhất (Bắt bệt)
    last_3 = "".join(recent[-3:])
    for s in last_3: counts[s] += 2
    
    p1 = sorted(counts, key=counts.get, reverse=True)[:3]
    
    # 2. Bắt Tổng 5 (Logic Trend Following)
    totals = np.sum(matrix, axis=1)
    current_total = totals[-1]
    avg_total = np.mean(totals)
    
    t5_tx = "TÀI" if avg_total < 22 else "XỈU"
    # Logic đảo cầu
    if abs(current_total - avg_total) > 10: t5_tx = "TÀI" if current_total < 22 else "XỈU"
    
    # 3. Độ tự tin (Dựa trên độ lặp lại mẫu)
    confidence = 60 + (len(set(p1) & set(last_3)) * 10)
    return p1, t5_tx, min(confidence, 95)

# ================= GIAO DIỆN CHÍNH (COMPACT) =================
st.markdown("<h6 style='text-align: center; color: #ff0055; margin-bottom:5px;'>🛰️ TITAN v7000 X-MOBILE</h6>", unsafe_allow_html=True)

# Ô nhập liệu tối giản
raw_input = st.text_area("Dán kết quả:", height=70, placeholder="Ví dụ: 82134...", label_visibility="collapsed")

col_btn1, col_btn2 = st.columns(2)
if col_btn1.button("🚀 PHÂN TÍCH"):
    if raw_input:
        new_data = re.findall(r"\d{5}", raw_input)
        st.session_state.db_x.extend(new_data)
        st.rerun()

if col_btn2.button("🧹 RESET"):
    st.session_state.db_x = []
    st.rerun()

# Khu vực hiển thị kết quả "Nén"
if len(st.session_state.db_x) >= 10:
    p1, t5, conf = analyze_x_mobile(st.session_state.db_x)
    
    st.markdown(f"""
        <div class='result-box'>
            <p class='mini-text'>TAY TIẾP THEO (TỰ TIN {conf}%)</p>
            <p class='prediction'>{"-".join(p1)}</p>
            <p style='color:#ffd700; font-weight:bold; font-size:14px; margin:0;'>TỔNG 5: {t5}</p>
        </div>
    """, unsafe_allow_html=True)
    
    if conf < 70:
        st.markdown("<p style='color:red; font-size:10px; text-align:center;'>⚠️ Cầu yếu - Nên chờ thêm 1-2 tay</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#00ff00; font-size:10px; text-align:center;'>✅ Nhịp đẹp - Vào đều tay</p>", unsafe_allow_html=True)

    st.markdown(f"<p class='mini-text' style='text-align:right;'>Data size: {len(st.session_state.db_x)}</p>", unsafe_allow_html=True)
else:
    st.warning("Nạp 10 kỳ để soi")
