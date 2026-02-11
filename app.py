import streamlit as st
import re
import json
import pandas as pd
import google.generativeai as genai
from collections import Counter
from pathlib import Path

# ================= CONFIG & API =================
st.set_page_config(page_title="TITAN v1600 ULTRA STABLE", layout="wide")

API_KEY = "AIzaSyBRo51DqVoC7BSv3ipUrY8GaEVfi0cVQxc"
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except:
    st.error("Lỗi API.")

DATA_FILE = "titan_dataset.json"
BACKUP_FILE = "titan_backup.json"

# ================= DATA CORE =================
def load_data():
    for f_path in [DATA_FILE, BACKUP_FILE]:
        if Path(f_path).exists():
            try:
                with open(f_path, "r") as f:
                    return list(dict.fromkeys(json.load(f)))
            except: continue
    return []

def save_data(data):
    clean = list(dict.fromkeys(data))
    with open(DATA_FILE, "w") as f: json.dump(clean, f)
    with open(BACKUP_FILE, "w") as f: json.dump(clean, f)
    return clean

if "dataset" not in st.session_state:
    st.session_state.dataset = load_data()

# ================= THUẬT TOÁN CAO CẤP V1600 =================
def analyze_v1600(dataset):
    # Lấy 50 kỳ gần nhất để phân tích nhịp (Trend)
    recent_kỳ = dataset[-50:]
    all_digits = "".join(recent_kỳ)
    digits_list = list(all_digits)
    
    # 1. Nhận diện bệt (Streaks)
    # Kiểm tra xem con gì đang nổ liên tục trong 5 kỳ cuối
    last_5_kỳ = dataset[-5:]
    streaks = []
    for num in range(10):
        s_count = sum(1 for kỳ in last_5_kỳ if str(num) in kỳ)
        if s_count >= 2: # Nếu số xuất hiện từ 2/5 kỳ gần nhất -> Đang vào bệt
            streaks.append(str(num))

    # 2. Tính điểm nhạy (Dynamic Scoring)
    score = {str(i): 0 for i in range(10)}
    for i in score:
        # Tần suất gần (30 kỳ)
        freq_recent = digits_list[-30:].count(i)
        score[i] += freq_recent * 5.0
        
        # Điểm bệt: Nếu nằm trong danh sách bệt, cộng cực mạnh
        if i in streaks:
            score[i] += 50 
            
        # Điểm rơi: Nếu kỳ vừa rồi có mặt, cộng thêm điểm nhịp
        if i in "".join(dataset[-1:]):
            score[i] += 15

    ranked = sorted(score, key=score.get, reverse=True)
    return ranked, score, streaks

# ================= GIAO DIỆN =================
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>🛡️ TITAN v1600 ULTRA</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📥 NHẬP KỲ MỚI")
    raw = st.text_area("Dán kết quả (Vừa nổ):", height=200)
    if st.button("🔥 CHỐT SỐ & BẮT BỆT", use_container_width=True):
        if raw:
            new_nums = re.findall(r"\d{1,5}", raw)
            st.session_state.dataset = save_data(st.session_state.dataset + new_nums)
            st.rerun()

if len(st.session_state.dataset) > 5:
    ranked, scores, streaks = analyze_v1600(st.session_state.dataset)
    p1 = ranked[:3]

    # Hiển thị Trạng thái Cầu
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("NHỊP CẦU", "BỆT" if streaks else "NHẢY")
    with c2:
        st.metric("SỐ ĐANG BỆT", ", ".join(streaks) if streaks else "N/A")
    with c3:
        st.metric("DATA SIZE", len(st.session_state.dataset))

    # Kết quả dự đoán
    st.markdown(f"""
    <div style='background: #111; padding: 25px; border-radius: 20px; border: 3px solid #00ffcc; text-align: center;'>
        <h2 style='color: white; margin:0;'>🎯 TAY TIẾP THEO (ƯU TIÊN BỆT)</h2>
        <h1 style='color: yellow; font-size: 85px; margin: 10px 0;'>{" - ".join(p1)}</h1>
    </div>
    """, unsafe_allow_html=True)

    # Biểu đồ sức mạnh
    st.subheader("📊 Sức mạnh nhịp cầu hiện tại")
    st.bar_chart(pd.Series(scores))
    
    # AI Gemini chốt hạ
    with st.expander("🧠 XÁC NHẬN TỪ AI GEMINI"):
        prompt = f"Lịch sử: {st.session_state.dataset[-10:]}. Dự đoán: {p1}. Số đang bệt: {streaks}. Phân tích ngắn."
        try:
            res = model.generate_content(prompt)
            st.info(res.text)
        except: st.warning("AI bận.")
else:
    st.info("Hãy nạp ít nhất 5 kỳ để bắt đầu soi cầu bệt.")
