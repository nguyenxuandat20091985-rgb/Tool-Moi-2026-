import streamlit as st
import re
import json
import pandas as pd
import google.generativeai as genai
from collections import Counter
from pathlib import Path

# ================= CONFIG =================
st.set_page_config(page_title="TITAN v1600 PRO STABLE", layout="wide")

API_KEY = "AIzaSyBRo51DqVoC7BSv3ipUrY8GaEVfi0cVQxc"
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except:
    st.error("API Error")

DATA_FILE = "titan_dataset.json"

# ================= DATA CORE =================
def load_data():
    if Path(DATA_FILE).exists():
        try:
            with open(DATA_FILE, "r") as f:
                return list(dict.fromkeys(json.load(f)))
        except: return []
    return []

def save_data(data):
    clean = list(dict.fromkeys(data))
    with open(DATA_FILE, "w") as f: json.dump(clean, f)
    return clean

if "dataset" not in st.session_state:
    st.session_state.dataset = load_data()

# ================= THUẬT TOÁN CÂN BẰNG (STABLE) =================
def analyze_v1600_pro(dataset):
    # Lấy dữ liệu nền (toàn bộ 4152 kỳ)
    all_digits = "".join(dataset)
    freq_total = Counter(all_digits)
    
    # Lấy dữ liệu nhịp (50 kỳ gần nhất)
    recent_50 = dataset[-50:]
    freq_recent = Counter("".join(recent_50))
    
    # Kiểm tra bệt thực sự (Phải xuất hiện 3/5 kỳ cuối mới gọi là bệt)
    last_5 = dataset[-5:]
    real_streaks = [str(i) for i in range(10) if sum(1 for k in last_5 if str(i) in k) >= 3]

    score = {str(i): 0 for i in range(10)}
    for i in score:
        # 1. Điểm nền tảng (Lấy từ 4152 kỳ - Giúp ổn định)
        score[i] += freq_total.get(i, 0) * 0.5
        
        # 2. Điểm xu hướng (Lấy từ 50 kỳ - Giúp nhảy số)
        score[i] += freq_recent.get(i, 0) * 15.0
        
        # 3. Điểm bệt (Chỉ cộng khi bệt thực sự rõ nét)
        if i in real_streaks:
            score[i] += 100 
            
    ranked = sorted(score, key=score.get, reverse=True)
    return ranked, score, real_streaks

# ================= GIAO DIỆN =================
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>🛡️ TITAN v1600 PRO</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📥 DỮ LIỆU")
    raw = st.text_area("Dán kỳ mới:", height=200)
    if st.button("🚀 PHÂN TÍCH CHUẨN", use_container_width=True):
        if raw:
            new = re.findall(r"\d{1,5}", raw)
            st.session_state.dataset = save_data(st.session_state.dataset + new)
            st.rerun()

if len(st.session_state.dataset) > 10:
    ranked, scores, streaks = analyze_v1600_pro(st.session_state.dataset)
    p1 = ranked[:3]

    # Dashboard chỉ số
    c1, c2, c3 = st.columns(3)
    c1.metric("TRẠNG THÁI", "ỔN ĐỊNH" if not streaks else "CẦU BỆT")
    c2.metric("SỐ KẾT", ", ".join(p1))
    c3.metric("TỔNG DỮ LIỆU", len(st.session_state.dataset))

    # KẾT QUẢ DỰ ĐOÁN
    st.markdown(f"""
    <div style='background: #000; padding: 25px; border-radius: 20px; border: 3px solid #ff4b4b; text-align: center;'>
        <h2 style='color: white; margin:0;'>🎯 DỰ ĐOÁN TAY TIẾP</h2>
        <h1 style='color: yellow; font-size: 85px; margin: 10px 0;'>{" - ".join(p1)}</h1>
        <p style='color: #00ffcc;'>Dự phòng: {", ".join(ranked[3:6])}</p>
    </div>
    """, unsafe_allow_html=True)

    st.bar_chart(pd.Series(scores))
else:
    st.warning("Cần thêm dữ liệu.")
