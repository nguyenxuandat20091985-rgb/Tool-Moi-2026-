import streamlit as st
import re
import json
import numpy as np
from collections import Counter
from pathlib import Path

# ================= CONFIG VĨNH VIỄN =================
DATA_FILE = "titan_v11_omni.json"

def load_db():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_db(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data[-10000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= GIAO DIỆN TITAN RECOVERY =================
st.set_page_config(page_title="TITAN v11000 OMNI", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #050a10; color: #00ffcc; }
    .stButton > button {
        background: linear-gradient(135deg, #ff0055 0%, #ff5500 100%);
        color: white; border: none; font-weight: 900; border-radius: 8px; height: 45px; width: 100%;
    }
    .prediction-card {
        background: rgba(0, 255, 204, 0.03); border: 1px solid #334455;
        border-radius: 15px; padding: 20px; margin-top: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .big-val { font-size: 42px; font-weight: 900; color: #00ffcc; text-align: center; text-shadow: 0 0 10px #00ffcc; }
    .alert-box { padding: 10px; background: rgba(255,0,0,0.1); border-left: 5px solid #ff0055; margin-top: 10px; font-size: 13px; }
    </style>
""", unsafe_allow_html=True)

# ================= THUẬT TOÁN ĐỐI KHÁNG V11 =================
def omni_engine(data):
    if len(data) < 10: return None
    
    # Chuyển dữ liệu sang mảng
    matrix = np.array([[int(d) for d in list(ky)] for ky in data])
    
    # 1. Phân tích Bước Nhảy (Interval Analysis)
    # Tìm xem sau con số vừa về, con số nào thường xuất hiện nhất ở chu kỳ sau
    last_val = matrix[-1]
    potential_next = []
    
    for pos in range(5):
        current_val = last_val[pos]
        next_vals = []
        for i in range(len(matrix)-1):
            if matrix[i, pos] == current_val:
                next_vals.append(matrix[i+1, pos])
        
        if next_vals:
            potential_next.append(Counter(next_vals).most_common(1)[0][0])
        else:
            # Nếu chưa có dữ liệu lặp, dùng thuật toán hồi số
            potential_next.append((current_val + 3) % 10)

    # 2. Lọc TOP 3 "TINH AN TOÀN"
    # Kết hợp giữa số hay về và số dự đoán theo bước nhảy
    freq_overall = Counter("".join(data[-30:]))
    candidates = [str(x) for x in potential_next]
    # Thêm 2 số có tần suất cao nhất vào danh sách cân nhắc
    top_freq = [x[0] for x in freq_overall.most_common(2)]
    candidates.extend(top_freq)
    
    final_p3 = [x[0] for x in Counter(candidates).most_common(3)]

    # 3. Tính độ rủi ro (Risk Detection)
    # Nếu 5 kỳ gần nhất có tổng biến thiên quá lớn -> Cầu đang ảo
    volatility = np.std(matrix[-5:], axis=0).mean()
    risk_level = "CAO" if volatility > 2.8 else "THẤP"
    confidence = max(60, 95 - (volatility * 10))

    return {
        "p3": final_p3,
        "conf": round(confidence, 1),
        "risk": risk_level,
        "count": len(data)
    }

# ================= UI CHÍNH =================
st.markdown("<h3 style='text-align: center;'>🛡️ TITAN v11000 OMNI</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 11px; color: #ff0055;'>ANTI-LOSS & RECOVERY MODE ACTIVE</p>", unsafe_allow_html=True)

input_data = st.text_area("📡 NẠP DỮ LIỆU THỰC CHIẾN:", height=80, placeholder="Nhập chuỗi 5 số mỗi kỳ...")

c1, c2 = st.columns(2)
with c1:
    if st.button("⚡ PHÂN TÍCH & LƯU"):
        if input_data:
            new_recs = re.findall(r"\d{5}", input_data)
            if new_recs:
                st.session_state.history.extend(new_recs)
                save_db(st.session_state.history)
                st.rerun()
with c2:
    if st.button("🗑️ RESET"):
        st.session_state.history = []
        save_db([])
        st.rerun()

if len(st.session_state.history) >= 10:
    res = omni_engine(st.session_state.history)
    
    st.markdown(f"""
    <div class='prediction-card'>
        <p style='color: #888; font-size: 12px;'>🎯 TOP 3 SIÊU TINH (KHUYÊN DÙNG)</p>
        <p class='big-val'>{" - ".join(res['p3'])}</p>
        <div style='display: flex; justify-content: space-between; border-top: 1px solid #334; pt-10;'>
            <span>Độ tin cậy: <b style='color:#ffd700;'>{res['conf']}%</b></span>
            <span>Rủi ro: <b style='color:{"#ff0055" if res['risk']=="CAO" else "#00ffcc"};'>{res['risk']}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if res['risk'] == "CAO":
        st.markdown("<div class='alert-box'>⚠️ Cảnh báo: Cầu đang biến động mạnh (Cầu ảo). Khuyến nghị vào tiền nhẹ tay hoặc tạm dừng quan sát.</div>", unsafe_allow_html=True)

    st.info(f"Hệ thống đã học từ {res['count']} kỳ. Trạng thái: Ổn định.")
else:
    st.warning("Cần nạp tối thiểu 10 kỳ để thuật toán Omni bắt đầu quét bước nhảy.")
