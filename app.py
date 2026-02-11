import streamlit as st
import re
import json
import pandas as pd
import numpy as np
import google.generativeai as genai
from collections import Counter
from pathlib import Path

# ================= CONFIG SIÊU GỌN (NANO UI) =================
st.set_page_config(page_title="TITAN v5000 NANO", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    [data-testid="stSidebar"] { min-width: 200px; max-width: 200px; }
    .main { background-color: #050505; color: #ffd700; font-size: 13px; }
    .stButton > button { 
        background: linear-gradient(135deg, #ff0055 0%, #8b0000 100%); 
        color: white; border-radius: 5px; height: 2.5em; font-size: 12px;
    }
    .nano-card {
        background: #111; border: 1px solid #444; border-radius: 10px; padding: 10px; text-align: center;
    }
    .big-num { font-size: 45px; font-weight: 900; color: #00ff00; text-shadow: 0 0 10px #00ff00; }
    .small-text { font-size: 11px; color: #888; }
    </style>
""", unsafe_allow_html=True)

# Kết nối AI (Giữ nguyên theo yêu cầu không bớt)
API_KEY = "AIzaSyBRo51DqVoC7BSv3ipUrY8GaEVfi0cVQxc"
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except: pass

DATA_FILE = "titan_nano_db.json"

def load_db():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as f: return json.load(f)
    return []

def save_db(data):
    with open(DATA_FILE, "w") as f: json.dump(data[-3000:], f)

if "db" not in st.session_state: st.session_state.db = load_db()

# ================= TỔNG HỢP PHƯƠNG PHÁP BẮT CẦU =================
def master_logic(db):
    if len(db) < 15: return None
    matrix = np.array([[int(d) for d in list(ky)] for ky in db])
    
    # 1. Bắt Số (Frequency + Streak)
    recent_str = "".join(db[-60:])
    freq = Counter(recent_str)
    score = {str(i): freq.get(str(i), 0) * 10 for i in range(10)}
    for i in range(10):
        if str(i) in db[-1] and str(i) in db[-2]: score[str(i)] += 100
    p1 = sorted(score, key=score.get, reverse=True)[:3]

    # 2. Bắt Tổng 5 (Probability Distribution)
    totals = np.sum(matrix, axis=1)
    avg_15 = np.mean(totals[-15:])
    t5_tx = "TÀI" if avg_15 < 22.5 else "XỈU" # Hồi quy về 22.5
    t5_cl = "LẺ" if int(np.median(totals[-10:])) % 2 != 0 else "CHẴN"

    # 3. Bắt Baccarat (Dynamic Winning Rate)
    con = (matrix[:, 2] + matrix[:, 4]) % 10
    cai = (matrix[:, 1] + matrix[:, 3]) % 10
    bac = "CON (P)" if sum(con[-5:] > cai[-5:]) >= 3 else "CÁI (B)"

    # 4. Đo độ nhiễu (Entropy)
    counts = np.unique(totals[-20:], return_counts=True)[1]
    probs = counts / counts.sum()
    ent = -np.sum(probs * np.log2(probs))

    return {"p1": p1, "t5": f"{t5_tx}-{t5_cl}", "bac": bac, "ent": ent, "hist": totals[-20:].tolist()}

# ================= GIAO DIỆN NANO MASTER =================
st.markdown("<h4 style='text-align: center; color: #ff0055; margin:0;'>🛰️ TITAN v5000 NANO</h4>", unsafe_allow_html=True)

# Tối ưu hóa không gian bằng Tabs
tab1, tab2, tab3 = st.tabs(["🎯 CHỐT", "📊 NHỊP", "📥 NHẬP"])

with tab3:
    raw = st.text_area("Dán mã 5D:", height=100, label_visibility="collapsed")
    if st.button("🚀 NẠP"):
        if raw:
            st.session_state.db.extend(re.findall(r"\d{5}", raw))
            save_db(st.session_state.db)
            st.rerun()
    if st.button("🧹 XÓA"):
        st.session_state.db = []; save_db([]); st.rerun()

if len(st.session_state.db) >= 15:
    res = master_logic(st.session_state.db)
    
    with tab1:
        # Hiển thị siêu gọn
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='nano-card'><p class='small-text'>SỐ CHỐT</p><p class='big-num'>{''.join(res['p1'])}</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='nano-card'><p class='small-text'>TỔNG 5 / BAC</p><p style='color:#ffd700; font-size:18px; font-weight:bold;'>{res['t5']}<br>{res['bac']}</p></div>", unsafe_allow_html=True)
        
        # Cảnh báo an toàn
        status_color = "#00ffcc" if res['ent'] < 2.9 else "#ff4b4b"
        st.markdown(f"<p style='text-align:center; color:{status_color}; font-size:12px;'>🛡️ SÓNG: {'ỔN ĐỊNH' if res['ent'] < 2.9 else 'LOẠN - NGHỈ'}</p>", unsafe_allow_html=True)

    with tab2:
        st.line_chart(res['hist'], height=150)
        st.write(f"Độ loạn Entropy: {res['ent']:.2f}")
else:
    st.info("Cần 15 kỳ.")

st.markdown("<p class='small-text' style='text-align:center;'>© 2026 TITAN NANO</p>", unsafe_allow_html=True)
