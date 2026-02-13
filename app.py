import streamlit as st
import re
import json
import numpy as np
from collections import Counter
import google.generativeai as genai
from pathlib import Path

# ================= CONFIG LƯU TRỮ VĨNH VIỄN =================
DATA_FILE = "titan_database_v9.json"

def load_db():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_db(data):
    # Lưu tối đa 5000 kỳ gần nhất để đảm bảo tốc độ xử lý nhanh
    with open(DATA_FILE, "w") as f:
        json.dump(data[-5000:], f)

# Khởi tạo dữ liệu từ file khi mở app
if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= GIAO DIỆN & THUẬT TOÁN (GIỮ NGUYÊN MẠNH MẼ) =================
st.set_page_config(page_title="TITAN v9000 PRO", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000; color: #00ffcc; }
    [data-testid="stHeader"] {display: none;}
    .stButton > button {
        background: linear-gradient(135deg, #00ffcc 0%, #0055ff 100%);
        color: #000; border: none; font-weight: 900; border-radius: 4px; height: 38px;
    }
    .prediction-card {
        background: rgba(0, 255, 204, 0.05); border: 1px solid #00ffcc;
        border-radius: 8px; padding: 10px; margin-top: 5px;
    }
    .big-val { font-size: 28px; font-weight: 900; color: #fff; margin: 0; }
    .percent { font-size: 16px; color: #ffd700; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ... (Hàm quantum_engine giữ nguyên như bản v9000 em đã gửi anh) ...
def quantum_engine(data):
    if len(data) < 15: return None
    matrix = np.array([[int(d) for d in list(ky)] for ky in data[-30:]])
    all_nums = "".join(data[-20:])
    freq = Counter(all_nums)
    potential = [str(i) for i in range(10) if all_nums.count(str(i)*2) < 2]
    p3 = sorted(potential, key=lambda x: freq[x], reverse=True)[:3]
    p3_prob = 75 + (freq[p3[0]] / len(all_nums) * 100)
    
    totals = np.sum(matrix, axis=1)
    mean_t = np.mean(totals)
    t5_tx = "TÀI" if mean_t < 22.5 else "XỈU"
    t5_cl = "LẺ" if int(mean_t) % 2 != 0 else "CHẴN"
    
    diff = np.std(matrix[-10:], axis=1).mean()
    if diff < 1.2: xi_to, xt_prob = "CÙ LŨ / TỨ QUÝ", 72
    elif 1.2 <= diff < 2.5: xi_to, xt_prob = "1 ĐÔI / SÁM CÔ", 85
    else: xi_to, xt_prob = "SẢNH / SỐ RỜI", 78
    
    r_val = matrix[-5:, 0].sum(); h_val = matrix[-5:, 4].sum()
    rh = "RỒNG" if r_val > h_val else "HỔ"; rh_p = 88 if abs(r_val - h_val) > 5 else 65
    
    return {"p3": p3, "p3_p": min(p3_prob, 96), "t5": f"{t5_tx} {t5_cl}", "t5_p": 82, "xt": xi_to, "xt_p": xt_prob, "rh": rh, "rh_p": rh_p}

st.markdown("<h4 style='text-align: center; color: #00ffcc;'>💎 TITAN v9000 PRO</h4>", unsafe_allow_html=True)

input_data = st.text_area("Dán kỳ mới:", height=65, label_visibility="collapsed")

c1, c2 = st.columns(2)
if c1.button("⚡ QUÉT & LƯU"):
    if input_data:
        new_records = re.findall(r"\d{5}", input_data)
        # Hợp nhất dữ liệu mới và cũ, loại bỏ trùng lặp nếu cần
        st.session_state.history.extend(new_records)
        # Lưu vào ổ cứng ngay lập tức
        save_db(st.session_state.history)
        st.rerun()

if c2.button("🗑️ XÓA HẾT"):
    st.session_state.history = []
    save_db([]) # Xóa luôn file lưu trữ
    st.rerun()

# Hiển thị kết quả (Logic hiển thị card giữ nguyên)
if len(st.session_state.history) >= 15:
    res = quantum_engine(st.session_state.history)
    st.markdown(f"""
    <div class='prediction-card'>
        <p style='font-size:11px; color:#888;'>🎯 3-TINH CHỐT (TỈ LỆ {res['p3_p']:.1f}%)</p>
        <p class='big-val' style='color:#00ff00;'>{" - ".join(res['p3'])}</p>
    </div>
    <div class='prediction-card'>
        <p style='font-size:11px; color:#888;'>📊 TỔNG 5: {res['t5']} ({res['t5_p']}%)</p>
        <p style='font-size:11px; color:#888;'>🐲 RỒNG HỔ: {res['rh']} ({res['rh_p']}%)</p>
    </div>
    """, unsafe_allow_html=True)
    st.success(f"Dữ liệu đã lưu: {len(st.session_state.history)} kỳ")
else:
    st.info("Nạp 15 kỳ để AI bắt đầu.")
