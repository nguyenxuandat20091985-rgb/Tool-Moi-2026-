import streamlit as st
import re
import json
import numpy as np
from collections import Counter
from pathlib import Path

# ================= CONFIG VĨNH VIỄN =================
DATA_FILE = "titan_v12_fast.json"

def load_db():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_db(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data[-5000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= UI FAST-COMBAT =================
st.set_page_config(page_title="TITAN v12.0 FAST", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #04090f; color: #00ffcc; }
    .stButton > button {
        background: linear-gradient(135deg, #00ffcc 0%, #0055ff 100%);
        color: black; border: none; font-weight: 900; border-radius: 5px; height: 45px; width: 100%;
    }
    .main-card {
        background: rgba(0, 255, 204, 0.05); border: 2px solid #00ffcc;
        border-radius: 15px; padding: 20px; margin-bottom: 15px;
    }
    .group-card {
        background: #111b27; border-left: 5px solid #0055ff;
        padding: 15px; margin-top: 10px; border-radius: 5px;
    }
    .number-display { font-size: 30px; font-weight: 900; color: #fff; letter-spacing: 3px; }
    .label { font-size: 12px; color: #888; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

# ================= ENGINE CHIA DÀN TỐI ƯU =================
def fast_engine(data):
    if len(data) < 10: return None
    
    matrix = np.array([[int(d) for d in list(ky)] for ky in data])
    
    # Phân tích tần suất và bước nhảy (giữ lõi v11)
    all_nums = "".join(data[-50:])
    freq = Counter(all_nums)
    
    # Lấy 7 số mạnh nhất (Safe 7)
    safe_7 = [x[0] for x in freq.most_common(7)]
    
    # Chia làm 2 cụm theo trọng số
    dan_4_strong = safe_7[:4] # 4 số mạnh nhất
    dan_3_support = safe_7[4:7] # 3 số lót
    
    # Tính rủi ro dựa trên độ biến động kỳ cuối
    volatility = np.std(matrix[-5:], axis=0).mean()
    risk = "CAO" if volatility > 2.5 else "THẤP"
    
    return {
        "dan4": dan_4_strong,
        "dan3": dan_3_support,
        "full7": safe_7,
        "risk": risk,
        "count": len(data)
    }

# ================= GIAO DIỆN CHÍNH =================
st.markdown("<h3 style='text-align: center; color: #00ffcc;'>⚡ TITAN v12.0 FAST-COMBAT</h3>", unsafe_allow_html=True)

input_data = st.text_area("📡 NẠP DỮ LIỆU:", height=70, placeholder="Dán chuỗi số tại đây...")

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 QUÉT NHANH"):
        if input_data:
            new_recs = re.findall(r"\d{5}", input_data)
            st.session_state.history.extend(new_recs)
            save_db(st.session_state.history)
            st.rerun()
with c2:
    if st.button("🗑️ XÓA"):
        st.session_state.history = []
        save_db([])
        st.rerun()

if len(st.session_state.history) >= 10:
    res = fast_engine(st.session_state.history)
    
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    # HIỂN THỊ DÀN 4 (CHỦ LỰC)
    st.markdown(f"""
    <div class='group-card'>
        <p class='label'>🎯 DÀN 4 SỐ (CHỦ LỰC - VÀO TIỀN MẠNH)</p>
        <p class='number-display'>{" - ".join(res['dan4'])}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # HIỂN THỊ DÀN 3 (LÓT)
    st.markdown(f"""
    <div class='group-card' style='border-left-color: #ffaa00;'>
        <p class='label'>🛡️ DÀN 3 SỐ (LÓT - BẢO TOÀN VỐN)</p>
        <p class='number-display' style='color: #ffaa00;'>{" - ".join(res['dan3'])}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Dàn 7 số tổng hợp để copy nhanh
    full_7_str = "".join(res['full7'])
    st.text_input("📋 COPY DÀN 7 SỐ NHANH:", full_7_str)
    
    # Cảnh báo rủi ro
    color = "#ff0055" if res['risk'] == "CAO" else "#00ffcc"
    st.markdown(f"<p style='text-align:center;'>RỦI RO: <b style='color:{color};'>{res['risk']}</b> | DỮ LIỆU: {res['count']} KỲ</p>", unsafe_allow_html=True)

else:
    st.info("Nạp tối thiểu 10 kỳ để AI chia dàn.")

st.markdown("<p style='font-size:10px; color:#444; text-align:center;'>Chiến thuật: Đánh dàn 4 làm gốc, dàn 3 làm ngọn. Không đánh lẻ 1-2 số.</p>", unsafe_allow_html=True)
