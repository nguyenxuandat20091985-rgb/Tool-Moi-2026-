import streamlit as st
import re
import json
import pandas as pd
import numpy as np
import google.generativeai as genai
from collections import Counter
from pathlib import Path

# ================= CONFIG HỆ THỐNG =================
st.set_page_config(page_title="TITAN v4000 MASTER", layout="wide")

# CSS Cao cấp: Black & Gold kết hợp Red-Neon
st.markdown("""
    <style>
    .main { background-color: #050505; color: white; }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #ff0055 0%, #8b0000 100%);
        color: white; font-weight: bold; border: none; border-radius: 10px; height: 3.5em; width: 100%;
    }
    .stTextArea textarea { background-color: #111; color: #00ff00; border: 1px solid #ff0055; border-radius: 10px; }
    .prediction-card {
        background: linear-gradient(180deg, #121212 0%, #000 100%);
        border: 2px solid #ff0055; border-radius: 20px; padding: 25px; text-align: center;
        box-shadow: 0 0 20px rgba(255, 0, 85, 0.3);
    }
    .status-bar { padding: 10px; border-radius: 10px; text-align: center; font-weight: bold; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

API_KEY = "AIzaSyBRo51DqVoC7BSv3ipUrY8GaEVfi0cVQxc"
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except: st.error("Lỗi kết nối AI.")

DATA_FILE = "titan_master_v4.json"

def load_data():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as f: return list(dict.fromkeys(json.load(f)))
    return []

def save_data(data):
    clean = list(dict.fromkeys(data))
    with open(DATA_FILE, "w") as f: json.dump(clean[-5000:], f)
    return clean

if "dataset" not in st.session_state: st.session_state.dataset = load_data()

# ================= THUẬT TOÁN ĐA TẦNG V4000 =================
def analyze_v4000(dataset):
    if len(dataset) < 20: return None
    
    # 1. Chuyển đổi ma trận số
    matrix = np.array([[int(d) for d in list(ky)] for ky in dataset])
    recent_100 = dataset[-100:]
    recent_str = "".join(recent_100)
    
    # 2. Phân tích Số Rời (Scoring nâng cao)
    freq_100 = Counter(recent_str)
    freq_last_10 = Counter("".join(dataset[-10:]))
    streaks = [str(i) for i in range(10) if freq_last_10.get(str(i), 0) >= 4]
    
    score = {str(i): 0 for i in range(10)}
    for i in score:
        score[i] += freq_100.get(i, 0) * 3
        score[i] += freq_last_10.get(i, 0) * 25
        if i in streaks: score[i] += 200
        if i in dataset[-1] and i in dataset[-2]: score[i] += 100 # Bắt nhịp rơi điểm

    # 3. Phân tích Tổng 5 Banh (Mean Reversion)
    totals = np.sum(matrix, axis=1)
    avg_20 = np.mean(totals[-20:])
    pred_t5_tx = "TÀI" if avg_20 < 22.5 else "XỈU"
    pred_t5_cl = "CHẴN" if int(avg_20) % 2 == 0 else "LẺ"

    # 4. Phân tích Baccarat 5D
    con_scores = (matrix[:, 2] + matrix[:, 4]) % 10
    cai_scores = (matrix[:, 1] + matrix[:, 3]) % 10
    con_win_rate = sum(1 for i in range(-10, 0) if con_scores[i] > cai_scores[i]) / 10
    pred_bac = "CON (P)" if con_win_rate >= 0.5 else "CÁI (B)"

    # 5. Đo Entropy (Độ loạn cầu)
    counts = np.unique(totals[-20:], return_counts=True)[1]
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))

    ranked = sorted(score, key=score.get, reverse=True)
    return ranked, score, streaks, pred_t5_tx, pred_t5_cl, pred_bac, entropy, totals[-30:]

# ================= GIAO DIỆN CHỐT HẠ =================
st.markdown("<h1 style='text-align: center; color: #ff0055;'>🛰️ TITAN v4000 MASTER CORE</h1>", unsafe_allow_html=True)

col_in, col_out = st.columns([1, 2.2])

with col_in:
    st.markdown("<h3 style='color: #ff0055;'>📥 INPUT DATA</h3>", unsafe_allow_html=True)
    raw = st.text_area("Dán mã 5D vào đây:", height=250, placeholder="82164\n35012\n...")
    if st.button("🔥 PHÂN TÍCH TỔNG LỰC", use_container_width=True):
        if raw:
            new = re.findall(r"\d{5}", raw)
            st.session_state.dataset = save_data(st.session_state.dataset + new)
            st.rerun()
    if st.button("🧹 RESET DATABASE"):
        st.session_state.dataset = []
        save_data([])
        st.rerun()

with col_out:
    if len(st.session_state.dataset) >= 20:
        ranked, scores, streaks, t5_tx, t5_cl, bac, ent, h_totals = analyze_v4000(st.session_state.dataset)
        p1 = ranked[:3]

        # Trạng thái sóng
        if ent < 2.8:
            st.markdown("<div class='status-bar' style='background: rgba(0, 255, 0, 0.1); color: #00ff00; border: 1px solid #00ff00;'>✅ SÓNG ĐẸP - NHỊP CẦU KHỚP</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-bar' style='background: rgba(255, 0, 0, 0.1); color: #ff4b4b; border: 1px solid #ff4b4b;'>⚠️ SÓNG LOẠN - ĐI VỐN NHỎ</div>", unsafe_allow_html=True)

        # CARD DỰ ĐOÁN CHÍNH
        st.markdown(f"""
            <div class='prediction-card'>
                <p style='color: #888; letter-spacing: 3px;'>TAY TIẾP THEO</p>
                <h1 style='color: #00ff00; font-size: 80px; margin: 10px;'>{"-".join(p1)}</h1>
                <div style='display: flex; justify-content: space-around; border-top: 1px solid #333; padding-top: 15px;'>
                    <div><p style='color: #888;'>TỔNG 5</p><p style='color: #ffd700; font-weight: bold;'>{t5_tx} - {t5_cl}</p></div>
                    <div><p style='color: #888;'>BACCARAT</p><p style='color: #ffd700; font-weight: bold;'>{bac}</p></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # BIỂU ĐỒ
        st.subheader("📊 Nhịp sóng Tổng 5 (30 kỳ)")
        st.line_chart(h_totals)
        
        with st.expander("🧠 XÁC NHẬN TỪ AI GEMINI"):
            try:
                prompt = f"Data: {st.session_state.dataset[-10:]}. Dự đoán: {p1}, Tổng 5: {t5_tx}. Phân tích nhịp cầu ngắn gọn."
                res = model.generate_content(prompt)
                st.info(res.text)
            except: st.warning("AI đang bận quét sóng.")
    else:
        st.info("Anh dán thêm kỳ 5D (đủ 5 số) để em bắt đầu 'vét' nhà cái nhé!")
