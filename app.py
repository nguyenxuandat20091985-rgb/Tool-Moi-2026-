import streamlit as st
import re
import json
import pandas as pd
from collections import Counter
from pathlib import Path

# ================= CONFIG =================
st.set_page_config(page_title="TITAN v1800 PRECISION", layout="wide")
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

# ================= THUẬT TOÁN ĐIỀU CHỈNH ĐỘ NHẠY =================
def analyze_v1800(dataset):
    # CHỈ SOI 50 KỲ GẦN NHẤT ĐỂ TRÁNH NHIỄU DỮ LIỆU CŨ
    recent_50 = dataset[-50:]
    recent_str = "".join(recent_50)
    
    # 1. Tần suất 50 kỳ
    freq_50 = Counter(recent_str)
    
    # 2. Soi nhịp cực ngắn (5 kỳ cuối) để bắt bệt thực sự
    last_5 = dataset[-5:]
    
    score = {str(i): 0 for i in range(10)}
    real_streaks = []

    for i in range(10):
        s_digit = str(i)
        # Đếm số kỳ xuất hiện trong 5 kỳ gần nhất
        count_in_5 = sum(1 for k in last_5 if s_digit in k)
        
        # CHỈ TÍNH LÀ BỆT NẾU XUẤT HIỆN TỪ 4/5 KỲ (Cực kỳ khắt khe)
        if count_in_5 >= 4:
            real_streaks.append(s_digit)
            score[s_digit] += 200 # Điểm thưởng bệt cực cao
        
        # Điểm tần suất nền
        score[s_digit] += freq_50.get(s_digit, 0) * 10
        
        # Thưởng điểm cho nhịp rơi (xuất hiện 2 kỳ liên tiếp cuối cùng)
        if len(dataset) >= 2:
            if s_digit in dataset[-1] and s_digit in dataset[-2]:
                score[s_digit] += 50

    ranked = sorted(score, key=score.get, reverse=True)
    return ranked, score, real_streaks

# ================= GIAO DIỆN CHUẨN =================
st.markdown("<h1 style='text-align: center; color: #00ffcc;'>🎯 TITAN v1800 PRECISION</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📥 CẬP NHẬT KỲ MỚI")
    raw = st.text_area("Dán kết quả Ku:", height=200, placeholder="Dán dãy số vừa nổ...")
    if st.button("🚀 CHỐT SỐ NGAY", use_container_width=True):
        if raw:
            new = re.findall(r"\d{1,5}", raw)
            st.session_state.dataset = save_data(st.session_state.dataset + new)
            st.rerun()
    if st.button("Reset Dữ Liệu"):
        save_data([])
        st.session_state.dataset = []
        st.rerun()

if len(st.session_state.dataset) >= 5:
    ranked, scores, streaks = analyze_v1800(st.session_state.dataset)
    p1 = ranked[:3]

    # Hiển thị bộ số chốt
    st.markdown(f"""
    <div style='background: #111; padding: 25px; border-radius: 20px; border: 4px solid #00ffcc; text-align: center;'>
        <h2 style='color: white; margin: 0;'>🎯 TAY TIẾP THEO</h2>
        <h1 style='color: yellow; font-size: 100px; margin: 15px 0;'>{" - ".join(p1)}</h1>
        <p style='color: #00ffcc;'>Tỉ lệ nổ ưu tiên: <b>{p1[0]}</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Biểu đồ phân tích nhịp
    st.subheader("📊 Phân tích nhịp cầu (50 kỳ)")
    st.bar_chart(pd.Series(scores))
    
    # Cảnh báo bệt chỉ hiện khi thực sự rõ ràng
    if streaks:
        st.error(f"🔥 CẢNH BÁO BỆT THỰC SỰ: Số {', '.join(streaks)} đang vào dây!")
    else:
        st.success("✅ Nhịp cầu đang ổn định, không có dấu hiệu bệt ảo.")
else:
    st.info("Anh dán thêm kỳ vừa nổ để em bắt đầu tính toán chính xác.")
