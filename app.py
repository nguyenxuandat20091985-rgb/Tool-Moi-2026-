import streamlit as st
import re
import json
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# ================= CONFIG GIAO DIỆN QUÂN SỰ HI-TECH =================
st.set_page_config(page_title="TITAN V3000 ULTIMATE PLUS", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #050505; color: white; }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #ffd700 0%, #b8860b 100%);
        color: black; font-weight: bold; border: none; border-radius: 10px; height: 3.5em; width: 100%;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
    }
    .stTextArea textarea {
        background-color: #121212; color: #ffd700; border: 1px solid #ffd700; border-radius: 10px;
    }
    .prediction-card {
        background: linear-gradient(180deg, #1b1e23 0%, #000 100%);
        border: 2px solid #ffd700; border-radius: 20px; padding: 25px; text-align: center;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.15);
    }
    .number-display {
        font-family: 'JetBrains Mono', monospace;
        font-size: 70px; font-weight: 900; color: #ffd700;
        text-shadow: 0 0 15px rgba(255, 215, 0, 0.6);
    }
    .status-alert {
        padding: 12px; border-radius: 10px; font-weight: bold; text-align: center; margin-bottom: 15px;
    }
    .metric-box {
        background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 15px; text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

DATA_FILE = "titan_master_db.json"

def load_data():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as f: return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f: json.dump(data[-5000:], f)
    return data

if "dataset" not in st.session_state: st.session_state.dataset = load_data()

# ================= CORE LOGIC - AI COUNTER =================
def analyze_ultimate(dataset):
    if len(dataset) < 15: return None
    
    # 1. Phân tích Số rời (Scoring V2000)
    matrix = np.array([[int(d) for d in list(ky)] for ky in dataset])
    recent_50 = "".join(dataset[-50:])
    freq = Counter(recent_50)
    scores = {str(i): freq.get(str(i), 0) * 15 for i in range(10)}
    
    # Cộng điểm bệt
    for i in range(10):
        if str(i) in dataset[-1] and str(i) in dataset[-2]: scores[str(i)] += 150

    ranked_nums = sorted(scores, key=scores.get, reverse=True)
    
    # 2. Phân tích Tổng 5 Banh (Mean Reversion)
    totals = np.sum(matrix, axis=1)
    avg_short = np.mean(totals[-10:])
    pred_t5_tx = "TÀI" if avg_short < 22.5 else "XỈU"
    pred_t5_cl = "CHẴN" if int(avg_short) % 2 != 0 else "LẺ"
    
    # 3. Phân tích Baccarat 5D
    con_scores = (matrix[:, 2] + matrix[:, 4]) % 10
    cai_scores = (matrix[:, 1] + matrix[:, 3]) % 10
    con_win_streak = sum(1 for i in range(-3, 0) if con_scores[i] > cai_scores[i])
    pred_bac = "CON (PLAYER)" if con_win_streak >= 2 else "CÁI (BANKER)"
    
    # 4. Chỉ số Entropy (Độ loạn)
    counts = np.unique(totals[-20:], return_counts=True)[1]
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    
    return {
        "nums": ranked_nums[:3],
        "t5": f"{pred_t5_tx} - {pred_t5_cl}",
        "bac": pred_bac,
        "entropy": entropy,
        "history_totals": totals[-30:].tolist()
    }

# ================= GIAO DIỆN CHÍNH =================
st.markdown("<h3 style='text-align: center; color: #888; letter-spacing: 5px;'>SYSTEM V3000 ULTIMATE PLUS</h3>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #ffd700; margin-bottom: 40px;'>🛡️ TITAN CORE MASTER</h1>", unsafe_allow_html=True)

col_in, col_out = st.columns([1, 2.5])

with col_in:
    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    raw = st.text_area("📡 NHẬN TÍN HIỆU 5D", height=250, placeholder="Dán dãy số mở thưởng...")
    if st.button("🚀 QUÉT SÓNG AI"):
        if raw:
            new_data = re.findall(r"\d{5}", raw)
            st.session_state.dataset = save_data(st.session_state.dataset + new_data)
            st.rerun()
    if st.button("🧹 LÀM SẠCH"):
        st.session_state.dataset = []
        save_data([])
        st.rerun()
    st.markdown(f"<p style='color: #666; font-size: 13px;'>DATA SIZE: {len(st.session_state.dataset)} KỲ</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_out:
    if len(st.session_state.dataset) >= 15:
        res = analyze_ultimate(st.session_state.dataset)
        
        # Cảnh báo sóng
        if res['entropy'] < 2.8:
            st.markdown("<div class='status-alert' style='background: rgba(0, 255, 204, 0.15); color: #00ffcc;'>✅ SÓNG ỔN ĐỊNH - TỈ LỆ THẮNG CAO</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-alert' style='background: rgba(255, 75, 75, 0.15); color: #ff4b4b;'>⚠️ SÓNG NHIỄU - GIẢM VỐN HOẶC QUAN SÁT</div>", unsafe_allow_html=True)
            
        # Dashboard chính
        st.markdown(f"""
            <div class='prediction-card'>
                <p style='color: #888; letter-spacing: 3px;'>DỰ ĐOÁN TỔNG 5 & XIÊN</p>
                <div class='number-display'>{res['t5']}</div>
                <hr style='border-color: #333;'>
                <div style='display: flex; justify-content: space-around;'>
                    <div>
                        <p style='color: #888;'>BACCARAT</p>
                        <p style='color: #ffd700; font-size: 20px; font-weight: bold;'>{res['bac']}</p>
                    </div>
                    <div>
                        <p style='color: #888;'>SỐ ƯU TIÊN</p>
                        <p style='color: #00ffcc; font-size: 20px; font-weight: bold;'>{" - ".join(res['nums'])}</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Biểu đồ nhịp
        with st.expander("📊 PHÂN TÍCH NHỊP SÓNG CHI TIẾT", expanded=True):
            st.line_chart(res['history_totals'])
            st.write(f"Độ loạn Entropy: **{res['entropy']:.2f}** (Dưới 2.5 là đẹp nhất)")
    else:
        st.info("Vui lòng nạp tối thiểu 15 kỳ để kích hoạt bộ não AI.")

st.markdown("<p style='text-align: center; color: #444; margin-top: 50px;'>© 2026 TITAN ULTIMATE SYSTEM - TRANG BỊ TỐT NHẤT ĐỐI ĐẦU AI NHÀ CÁI</p>", unsafe_allow_html=True)
