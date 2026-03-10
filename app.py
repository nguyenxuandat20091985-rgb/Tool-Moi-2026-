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

# --- PHẦN 2: GIAO DIỆN TỐI ƯU DÀN NGANG ---
st.set_page_config(page_title="TITAN AI COMPACT", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .gold-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .gold-item {
        background: linear-gradient(145deg, #333, #111);
        border: 1px solid #ffd700;
        border-radius: 8px;
        width: 18%; /* Ép 5 ô nằm ngang */
        aspect-ratio: 1 / 1;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: bold;
        color: #ffd700;
        box-shadow: 0 2px 8px rgba(255, 215, 0, 0.3);
    }
    .stButton button { background-color: #333; color: white; border: 1px solid #555; height: 35px; }
    .accuracy-label { text-align: center; color: #00ff00; font-size: 18px; font-weight: bold; margin-bottom: 10px; }
    .suggest-label { background: #ffd700; color: black; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

DB_FILE = "history_v83.json"
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

st.markdown("<h3 style='text-align: center; color: #ffd700; margin-top: -30px;'>🏆 TITAN GOLD v8.3</h3>", unsafe_allow_html=True)

# Nhập liệu thu gọn
raw_input = st.text_area("", placeholder="Dán kỳ quay...", height=60, label_visibility="collapsed")

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

if history:
    top_5, acc = ai.analyze(history)
    
    st.markdown(f"<div class='accuracy-label'>Độ tin cậy: {acc}%</div>", unsafe_allow_html=True)
    
    # Hiển thị 5 số vàng dàn ngang
    gold_html = f"""
    <div class="gold-container">
        <div class="gold-item">{top_5[0]}</div>
        <div class="gold-item">{top_5[1]}</div>
        <div class="gold-item">{top_5[2]}</div>
        <div class="gold-item">{top_5[3]}</div>
        <div class="gold-item">{top_5[4]}</div>
    </div>
    """
    st.markdown(gold_html, unsafe_allow_html=True)
    
    st.markdown(f"<div class='suggest-label'>🔥 CHỐT 3 TINH: {top_5[0]} - {top_5[1]} - {top_5[2]}</div>", unsafe_allow_html=True)
    
    with st.expander("📜 Lịch sử"):
        for row in history[:5]: st.write("".join(map(str, row)))
else:
    st.info("Hãy dán kỳ quay để AI phân tích.")
