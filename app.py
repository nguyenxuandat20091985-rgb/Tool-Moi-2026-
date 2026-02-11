import streamlit as st
import re
import json
import pandas as pd
from collections import Counter
from pathlib import Path

# ================= CONFIG GIAO DIỆN CAO CẤP =================
st.set_page_config(page_title="TITAN V2000 ULTIMATE", layout="wide", initial_sidebar_state="collapsed")

# CSS để biến Streamlit thành app chuyên nghiệp
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #ffd700 0%, #b8860b 100%);
        color: black; font-weight: bold; border: none; border-radius: 10px; height: 3em; width: 100%;
    }
    .stTextArea textarea {
        background-color: #1b1e23; color: #ffd700; border: 1px solid #444; border-radius: 10px;
    }
    .metric-card {
        background: #1b1e23; border: 1px solid #333; border-radius: 15px; padding: 15px; text-align: center;
    }
    .number-display {
        font-family: 'Courier New', Courier, monospace;
        font-size: 80px; font-weight: 900; color: #ffd700;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
        margin: 20px 0;
    }
    .status-bar {
        padding: 10px; border-radius: 50px; font-size: 14px; font-weight: bold; text-align: center; margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

DATA_FILE = "titan_dataset.json"

def load_data():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as f: return list(dict.fromkeys(json.load(f)))
    return []

def save_data(data):
    clean = list(dict.fromkeys(data))
    with open(DATA_FILE, "w") as f: json.dump(clean, f)
    return clean

if "dataset" not in st.session_state: st.session_state.dataset = load_data()

# ================= CORE LOGIC (V1800 PRECISION) =================
def analyze_v2000(dataset):
    recent_50 = dataset[-50:]
    recent_str = "".join(recent_50)
    freq_50 = Counter(recent_str)
    last_5 = dataset[-5:]
    
    score = {str(i): 0 for i in range(10)}
    real_streaks = []

    for i in range(10):
        s_digit = str(i)
        count_in_5 = sum(1 for k in last_5 if s_digit in k)
        if count_in_5 >= 4:
            real_streaks.append(s_digit)
            score[s_digit] += 300 
        score[s_digit] += freq_50.get(s_digit, 0) * 10
        if len(dataset) >= 2 and s_digit in dataset[-1] and s_digit in dataset[-2]:
            score[s_digit] += 70

    ranked = sorted(score, key=score.get, reverse=True)
    return ranked, score, real_streaks

# ================= GIAO DIỆN CHÍNH =================
st.markdown("<h3 style='text-align: center; color: #888;'>PREMIUM PREDICTION TOOL</h3>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #ffd700; margin-bottom: 30px;'>TITAN V2000 ULTIMATE</h1>", unsafe_allow_html=True)

# Bố cục 2 cột
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    raw_input = st.text_area("NHẬP KỲ MỚI", height=150, placeholder="Dán kết quả tại đây...")
    if st.button("🚀 PHÂN TÍCH"):
        if raw_input:
            new_nums = re.findall(r"\d{1,5}", raw_input)
            st.session_state.dataset = save_data(st.session_state.dataset + new_nums)
            st.rerun()
    
    if st.button("🧹 LÀM SẠCH"):
        st.session_state.dataset = []
        save_data([])
        st.rerun()
    st.markdown(f"<p style='color: #666; font-size: 12px; margin-top: 10px;'>DATABASE: {len(st.session_state.dataset)} KỲ</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    if len(st.session_state.dataset) >= 5:
        ranked, scores, streaks = analyze_v2000(st.session_state.dataset)
        p1 = ranked[:3]
        
        # Trạng thái cầu
        if streaks:
            st.markdown(f"<div class='status-bar' style='background: rgba(255, 0, 0, 0.2); color: #ff4b4b; border: 1px solid #ff4b4b;'>⚠️ CẢNH BÁO BỆT: {', '.join(streaks)}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-bar' style='background: rgba(0, 255, 0, 0.1); color: #00ffcc; border: 1px solid #00ffcc;'>✅ NHỊP CẦU ĐANG ỔN ĐỊNH</div>", unsafe_allow_html=True)

        # Hiển thị số chốt chính
        st.markdown(f"""
            <div style='background: linear-gradient(180deg, #1b1e23 0%, #0e1117 100%); border: 2px solid #ffd700; border-radius: 20px; padding: 20px; text-align: center;'>
                <p style='color: #ffd700; letter-spacing: 5px; font-weight: bold;'>🎯 DỰ ĐOÁN TAY TIẾP</p>
                <div class='number-display'>{"-".join(p1)}</div>
                <p style='color: #888;'>Ưu tiên: <span style='color: #00ffcc; font-size: 24px; font-weight: bold;'>{p1[0]}</span></p>
            </div>
        """, unsafe_allow_html=True)

        # Biểu đồ sức mạnh (Gọn lại)
        with st.expander("📊 BIỂU ĐỒ SỨC MẠNH", expanded=True):
            chart_data = pd.DataFrame({'Điểm': scores.values()}, index=scores.keys())
            st.bar_chart(chart_data, height=200)
    else:
        st.info("Vui lòng nạp thêm kết quả để kích hoạt hệ thống phân tích.")

st.markdown("<p style='text-align: center; color: #444; margin-top: 50px;'>© 2026 TITAN CORE SYSTEM - V2000 PRO</p>", unsafe_allow_html=True)
