import streamlit as st
import re
import json
import pandas as pd
import google.generativeai as genai
from collections import Counter
from pathlib import Path

# ================= CONFIG =================
st.set_page_config(page_title="TITAN v1700 THE KILLER", layout="wide")
API_KEY = "AIzaSyBRo51DqVoC7BSv3ipUrY8GaEVfi0cVQxc"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

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

# ================= THUẬT TOÁN V1700 (CHỈ SOI NHỊP SỐNG) =================
def analyze_v1700(dataset):
    # CHỈ LẤY 100 KỲ GẦN NHẤT ĐỂ SOI - ĐÂY LÀ CHÌA KHÓA ỔN ĐỊNH
    recent_context = dataset[-100:]
    recent_str = "".join(recent_context)
    
    # 1. Tính tần suất trong khung 100 kỳ (Trend ngắn hạn)
    freq_100 = Counter(recent_str)
    
    # 2. Tính độ nhạy cực kỳ (10 kỳ gần nhất)
    last_10 = dataset[-10:]
    freq_last_10 = Counter("".join(last_10))
    
    # 3. Nhận diện bệt chuẩn (Xuất hiện >= 4 lần trong 10 kỳ)
    streaks = [str(i) for i in range(10) if freq_last_10.get(str(i), 0) >= 4]

    score = {str(i): 0 for i in range(10)}
    for i in score:
        # Trọng số nhịp trend (100 kỳ)
        score[i] += freq_100.get(i, 0) * 2
        # Trọng số bùng nổ (10 kỳ gần nhất) - Ưu tiên cực cao
        score[i] += freq_last_10.get(i, 0) * 20
        # Điểm thưởng bệt
        if i in streaks: score[i] += 150
            
    ranked = sorted(score, key=score.get, reverse=True)
    return ranked, score, streaks

# ================= GIAO DIỆN =================
st.markdown("<h1 style='text-align: center; color: #ff0055;'>🔥 TITAN v1700 THE KILLER</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📥 CẬP NHẬT KỲ MỚI")
    raw = st.text_area("Dán kết quả Ku:", height=200)
    if st.button("🚀 CHỐT HẠ", use_container_width=True):
        if raw:
            new = re.findall(r"\d{1,5}", raw)
            st.session_state.dataset = save_data(st.session_state.dataset + new)
            st.rerun()

if len(st.session_state.dataset) >= 10:
    ranked, scores, streaks = analyze_v1600_pro(st.session_state.dataset) if 'analyze_v1600_pro' in globals() else analyze_v1700(st.session_state.dataset)
    # Ghi đè để dùng v1700
    ranked, scores, streaks = analyze_v1700(st.session_state.dataset)
    p1 = ranked[:3]

    st.markdown(f"""
    <div style='background: #000; padding: 20px; border-radius: 15px; border: 4px solid #ff0055; text-align: center;'>
        <h2 style='color: white;'>🎯 TAY TIẾP THEO</h2>
        <h1 style='color: #00ff00; font-size: 90px; margin: 10px;'>{" - ".join(p1)}</h1>
        <p style='color: #fff;'>Dòng tiền đề xuất: <b>1-2-4-8-16</b> hoặc <b>Đều tay</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 Nhịp cầu thực tế (100 kỳ gần nhất)")
    st.bar_chart(pd.Series(scores))
    
    if streaks:
        st.warning(f"⚠️ CẢNH BÁO BỆT: Các số {', '.join(streaks)} đang nổ rất dày!")
else:
    st.info("Anh dán thêm vài kỳ để em bắt đầu bắt nhịp nhé!")
