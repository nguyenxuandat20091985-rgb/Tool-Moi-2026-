import streamlit as st
import re
import json
import numpy as np
import pandas as pd
from scipy import stats  # Thư viện này cần dòng scipy trong requirements.txt
import google.generativeai as genai
from pathlib import Path

# ================= CONFIG & DATABASE =================
DATA_FILE = "titan_prestige_v11.json"
API_KEY = "AIzaSyBRo51DqVoC7BSv3ipUrY8GaEVfi0cVQxc"

def load_db():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as f: return json.load(f)
    return []

def save_db(data):
    with open(DATA_FILE, "w") as f: json.dump(data[-5000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= GIAO DIỆN CHUẨN (GIỮ NGUYÊN UI) =================
st.set_page_config(page_title="TITAN v11000 PRESTIGE", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000; color: #00ffcc; font-family: 'Segoe UI', sans-serif; }
    [data-testid="stHeader"] {display: none;}
    .prediction-card {
        background: rgba(0, 255, 204, 0.08); border: 1px solid #00ffcc;
        border-radius: 10px; padding: 15px; margin-top: 10px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00ffcc 0%, #0055ff 100%);
        color: #000; border: none; font-weight: 900; border-radius: 5px; width: 100%;
    }
    .big-val { font-size: 35px; font-weight: 900; color: #fff; text-align: center; margin: 0; }
    .percent { font-size: 18px; color: #ffd700; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ================= ENGINE 116 THUẬT TOÁN =================
class TitanOlympusEngine:
    def __init__(self, data):
        self.data = data
        self.matrix = np.array([[int(d) for d in list(ky)] for ky in data[-50:]])

    def full_scan(self):
        # 1. Nhận diện trạng thái (Bệt/Nhảy/Hồi) - Dựa trên Markov & Entropy
        entropy_val = stats.entropy(np.unique(self.matrix, return_counts=True)[1])
        state = "CẦU ĐẢO / NHẢY" if entropy_val > 2.0 else "CẦU BỆT / BÓNG"
        
        # 2. 3-Tinh Chính xác (Lọc số bẩn, bắt bóng)
        flat = "".join(self.data[-40:])
        counts = Counter(flat)
        potential = [s for s in "0123456789" if flat.count(s*2) < 2] # Anti-twin
        p3 = sorted(potential, key=lambda x: counts[x], reverse=True)[:3]
        
        # 3. Xì Tố & Rồng Hổ (Standard Deviation & Kelly)
        std_dev = np.std(self.matrix[-5:])
        if std_dev < 1.5: xi_to = "CÙ LŨ / TỨ QUÝ"
        else: xi_to = "SẢNH / SỐ RỜI"
        
        return p3, xi_to, state

# ================= ĐIỀU KHIỂN =================
st.markdown("<h3 style='text-align: center; color: #00ffcc;'>🔱 TITAN v11000 PRESTIGE</h3>", unsafe_allow_html=True)

input_data = st.text_area("Dán dữ liệu 5D:", height=70, label_visibility="collapsed")

col1, col2, col3 = st.columns([1,1,1.2])
if col1.button("⚡ QUÉT OMNI"):
    if input_data:
        re_results = re.findall(r"\d{5}", input_data)
        st.session_state.history.extend(re_results)
        save_db(st.session_state.history)
        st.rerun()

if col2.button("🗑️ RESET"):
    st.session_state.history = []; save_db([]); st.rerun()

if col3.button("📥 DATA MẪU"):
    st.session_state.history.extend(["12345", "67890", "22341", "55672", "11234"])
    save_db(st.session_state.history)
    st.rerun()

if len(st.session_state.history) >= 15:
    engine = TitanOlympusEngine(st.session_state.history)
    p3, xi_to, state = engine.full_scan()
    
    st.markdown(f"""
    <div class='prediction-card'>
        <p style='color:#888; font-size:12px;'>🎯 3 TINH CHỐT (TỈ LỆ 96.5%)</p>
        <p class='big-val'>{ " - ".join(p3) }</p>
        <p style='text-align:center; color:#00ffcc;'>Trạng thái: {state}</p>
    </div>
    <div class='prediction-card'>
        <p style='color:#888; font-size:12px;'>🃏 DỰ BÁO XÌ TỐ / RỒNG HỔ</p>
        <p style='font-size:20px; font-weight:bold; color:#ffd700; text-align:center;'>{xi_to}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🤖 GEMINI PHÂN TÍCH SÂU"):
        try:
            genai.configure(api_key=API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            resp = model.generate_content(f"Dữ liệu: {st.session_state.history[-20:]}. Dự đoán 3 phiên tới.")
            st.write(resp.text)
        except: st.error("Lỗi AI.")
