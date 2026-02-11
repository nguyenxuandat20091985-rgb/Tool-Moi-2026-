import streamlit as st
import re
import json
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ================= CONFIG SIÊU GỌN - CHUYÊN SÂU =================
st.set_page_config(page_title="TITAN v6000 GHOST", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #050505; color: #00ff00; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px; background-color: #111; border-radius: 5px; color: #888;
    }
    .stTabs [aria-selected="true"] { background-color: #ff0055; color: white; }
    .nano-card {
        background: #000; border: 1px solid #ff0055; border-radius: 8px; padding: 12px; text-align: center;
    }
    .big-num { font-size: 50px; font-weight: 900; color: #ff0055; text-shadow: 0 0 15px rgba(255, 0, 85, 0.5); }
    .bot-status { font-size: 11px; font-family: monospace; color: #00ffcc; }
    </style>
""", unsafe_allow_html=True)

DATA_FILE = "titan_v6_db.json"

def load_db():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as f: return json.load(f)
    return []

def save_db(data):
    # Lưu tối đa 5000 kỳ để máy học nhịp sâu
    with open(DATA_FILE, "w") as f: json.dump(data[-5000:], f)

if "db" not in st.session_state: st.session_state.db = load_db()

# ================= BỘ NÃO v6000 GHOST PROTOCOL =================
def ghost_brain(db):
    if len(db) < 30: return None
    
    # 1. Xử lý ma trận dữ liệu
    matrix = np.array([[int(d) for d in list(ky)] for ky in db])
    
    # 2. Thuật toán Ma trận Trận thế (Pattern Matching)
    # Tìm 2 kỳ gần nhất để so khớp lịch sử
    last_pattern = "".join(db[-2:])
    matches = []
    for i in range(len(db)-3):
        if "".join(db[i:i+2]) == last_pattern:
            matches.append(db[i+2])
    
    # 3. Tính toán nhịp Tổng 5 (Markov Chain)
    totals = np.sum(matrix, axis=1)
    diffs = np.diff(totals[-20:]) # Sự biến thiên giữa các kỳ
    next_diff_est = np.mean(diffs)
    est_total = totals[-1] + next_diff_est
    
    t5_tx = "TÀI" if est_total > 22.5 else "XỈU"
    t5_cl = "CHẴN" if int(est_total) % 2 == 0 else "LẺ"

    # 4. Bắt số rời (Deep Frequency)
    # Lọc ra các số có "độ rơi" ổn định nhất
    score = {str(i): 0 for i in range(10)}
    if matches: # Nếu tìm thấy mẫu giống trong quá khứ
        match_counts = Counter("".join(matches))
        for k, v in match_counts.items(): score[k] += v * 50
    
    # Cộng điểm xu hướng 20 kỳ gần nhất
    recent_freq = Counter("".join(db[-20:]))
    for k, v in recent_freq.items(): score[k] += v * 5
    
    p1 = sorted(score, key=score.get, reverse=True)[:3]

    # 5. Hệ số tin cậy (Confidence)
    confidence = 50 + (len(matches) * 10) if matches else 45
    confidence = min(confidence, 98)

    return {"p1": p1, "t5": f"{t5_tx}-{t5_cl}", "conf": confidence, "hist": totals[-20:].tolist()}

# ================= GIAO DIỆN NANO MASTER v6 =================
st.markdown("<h5 style='text-align: center; color: #ff0055; margin:0;'>🛰️ TITAN v6000 GHOST</h5>", unsafe_allow_html=True)

tab_play, tab_input = st.tabs(["🎯 SOI CẦU", "📥 NẠP DATA"])

with tab_input:
    raw = st.text_area("Dán mã 5D:", height=100, label_visibility="collapsed", placeholder="Dán dãy số mở thưởng...")
    c1, c2 = st.columns(2)
    if c1.button("🚀 NẠP"):
        if raw:
            st.session_state.db.extend(re.findall(r"\d{5}", raw))
            save_db(st.session_state.db)
            st.rerun()
    if c2.button("🧹 XÓA"):
        st.session_state.db = []; save_db([]); st.rerun()
    st.markdown(f"<p class='bot-status'>DATABASE: {len(st.session_state.db)} KỲ</p>", unsafe_allow_html=True)

if len(st.session_state.db) >= 30:
    res = ghost_brain(st.session_state.db)
    
    with tab_play:
        # Khu vực số chốt - Ép cực gọn
        st.markdown(f"""
            <div class='nano-card'>
                <p style='color: #888; font-size: 10px; margin:0;'>DỰ ĐOÁN TAY TIẾP</p>
                <p class='big-num'>{''.join(res['p1'])}</p>
                <div style='display: flex; justify-content: space-around; border-top: 1px solid #222; padding-top: 5px;'>
                    <span style='color: #00ffcc; font-size: 12px;'>TỔNG 5: <b>{res['t5']}</b></span>
                    <span style='color: #ffd700; font-size: 12px;'>TỰ TIN: <b>{res['conf']}%</b></span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Biểu đồ nhịp nén
        st.line_chart(res['hist'], height=120)
        
        if res['conf'] < 60:
            st.warning("⚠️ Cầu đang nhiễu, nên chờ nhịp mới.")
else:
    st.info("Cần nạp 30 kỳ để AI học nhịp cầu.")

st.markdown("<p style='text-align:center; color:#333; font-size:10px;'>GHOST PROTOCOL ACTIVATED</p>", unsafe_allow_html=True)
