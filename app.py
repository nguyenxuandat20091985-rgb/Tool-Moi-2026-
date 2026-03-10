import streamlit as st
import re
from collections import Counter
import math
import json
import os

# --- PHẦN 1: BỘ NÃO AI (TRINITY ENGINE) ---
class TitanTrinityAI:
    def __init__(self):
        self.shadow_map = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}

    def parse_data(self, raw):
        cleaned = []
        if not raw: return cleaned
        lines = str(raw).split('\n')
        for item in lines:
            match = re.search(r'\d{5}', str(item))
            if match: cleaned.append([int(d) for d in match.group()])
        return cleaned

    def analyze(self, data):
        if len(data) < 3: return ["-"] * 5, 0
        recent = data[:15]
        all_nums = [d for row in recent for d in row]
        freq = Counter(all_nums)
        gaps = {i: 30 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row: gaps[i] = idx; break
        scores = {}
        for i in range(10):
            f_score = freq.get(i, 0) * 4
            g_score = 30 if gaps[i] == 1 else 15 if gaps[i] == 0 else 0
            s_score = 20 if gaps[self.shadow_map[i]] == 0 else 0
            scores[str(i)] = f_score + g_score + s_score
        top_5 = [n for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:5]
        try:
            counts = [all_nums.count(i) for i in set(all_nums)]
            total = sum(counts)
            entropy = -sum((c/total)*math.log2(c/total) for c in counts if c > 0)
            accuracy = int(max(0, min(100, (3.32 - entropy) * 280)))
        except: accuracy = 50
        return top_5, accuracy

# --- PHẦN 2: GIAO DIỆN HIỆN ĐẠI (UI/UX) ---
st.set_page_config(page_title="TITAN AI GOLD", layout="centered")

# CSS để làm đẹp giao diện
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTextArea textarea { background-color: #1e1e1e; color: #ffd700; border: 1px solid #ffd700; }
    .gold-box {
        background: linear-gradient(145deg, #2c2c2c, #1a1a1a);
        border: 2px solid #ffd700;
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #ffd700;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
        margin: 5px;
    }
    .accuracy-text { font-size: 20px; font-weight: bold; color: #00ff00; text-align: center; }
    .suggest-box { background-color: #ffd700; color: black; padding: 10px; border-radius: 8px; font-weight: bold; text-align: center; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

DB_FILE = "history_v82.json"
def load_data():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_data(data):
    with open(DB_FILE, "w") as f: json.dump(data[:100], f)

history = load_data()
ai = TitanTrinityAI()

# Tiêu đề gọn
st.markdown("<h2 style='text-align: center; color: #ffd700; margin-bottom: 0;'>🏆 TITAN AI GOLD</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'>Premium Trinity System v8.2</p>", unsafe_allow_html=True)

# Ô nhập liệu nhỏ gọn
raw_input = st.text_area("", placeholder="Dán kỳ quay tại đây...", height=70, label_visibility="collapsed")

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 GIẢI MÃ", use_container_width=True):
        new_records = ai.parse_data(raw_input)
        if new_records:
            history = (new_records + history)[:100]
            save_data(history)
            st.rerun()
with c2:
    if st.button("🗑️ RESET", use_container_width=True):
        save_data([])
        st.rerun()

# Khu vực hiển thị kết quả
if history:
    top_5, acc = ai.analyze(history)
    
    st.markdown(f"<p class='accuracy-text'>Độ tin cậy: {acc}%</p>", unsafe_allow_html=True)
    
    # Hiển thị 5 số vàng trên một hàng ngang
    st.markdown("<p style='text-align: center; color: white; margin-bottom: 5px;'>💎 5 SỐ VÀNG CHỐT 3 TINH 💎</p>", unsafe_allow_html=True)
    cols = st.columns(5)
    for i in range(5):
        cols[i].markdown(f"<div class='gold-box'>{top_5[i]}</div>", unsafe_allow_html=True)
    
    # Gợi ý bộ 3 tinh nổi bật
    st.markdown(f"<div class='suggest-box'>🔥 BỘ 3 TINH CHỦ LỰC: {top_5[0]} - {top_5[1]} - {top_5[2]}</div>", unsafe_allow_html=True)
    
    with st.expander("📜 Xem lịch sử"):
        for row in history[:5]: st.code("".join(map(str, row)))
else:
    st.info("💡 Hãy dán ít nhất 1 kỳ quay để AI bắt đầu.")
