import streamlit as st
import re
from collections import Counter
import math
import json
import os

# --- PHẦN 1: BỘ NÃO AI (CẢI TIẾN KHẢ NĂNG ĐỌC DỮ LIỆU) ---
class TitanTrinityAI:
    def __init__(self):
        self.shadow_map = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}

    def parse_data(self, raw):
        """Hàm này đã được nâng cấp để đọc được mọi kiểu dán số của anh"""
        cleaned = []
        if not raw: return cleaned
        # Tìm tất cả các dãy có ít nhất 5 chữ số trong văn bản anh dán vào
        found = re.findall(r'\d{5}', str(raw))
        for item in found:
            cleaned.append([int(d) for d in item])
        return cleaned

    def analyze(self, data):
        if len(data) < 1: return ["-"] * 5, 0
        recent = data[:20] # Lấy 20 kỳ gần nhất để phân tích sâu hơn
        all_nums = [d for row in recent for d in row]
        freq = Counter(all_nums)
        
        gaps = {i: 30 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row: gaps[i] = idx; break

        scores = {}
        for i in range(10):
            f_score = freq.get(i, 0) * 5
            g_score = 40 if gaps[i] == 1 else 20 if gaps[i] == 0 else 0
            s_score = 25 if gaps[self.shadow_map[i]] == 0 else 0
            scores[str(i)] = f_score + g_score + s_score
            
        top_5 = [n for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:5]
        
        try:
            counts = [all_nums.count(i) for i in set(all_nums)]
            total = sum(counts)
            entropy = -sum((c/total)*math.log2(c/total) for c in counts if c > 0)
            accuracy = int(max(18, min(99, (3.32 - entropy) * 300)))
        except: accuracy = 18
        
        return top_5, accuracy

# --- PHẦN 2: GIAO DIỆN SIÊU GỌN ---
st.set_page_config(page_title="TITAN GOLD v8.4", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .gold-container { display: flex; justify-content: space-between; margin: 15px 0; }
    .gold-item {
        background: linear-gradient(145deg, #333, #111);
        border: 2px solid #ffd700; border-radius: 10px;
        width: 18%; aspect-ratio: 1/1;
        display: flex; align-items: center; justify-content: center;
        font-size: 24px; font-weight: bold; color: #ffd700;
        box-shadow: 0 4px 10px rgba(255, 215, 0, 0.4);
    }
    .stButton button { background-color: #ffd700; color: black; font-weight: bold; border-radius: 8px; border: none; }
    .stTextArea textarea { border: 1px solid #ffd700; }
    .suggest-label { background: #ffd700; color: black; padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

DB_FILE = "history_v84.json"
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

st.markdown("<h2 style='text-align: center; color: #ffd700;'>🏆 TITAN GOLD v8.4</h2>", unsafe_allow_html=True)

raw_input = st.text_area("", placeholder="Dán dãy số vào đây (Ví dụ: 24373)...", height=80, label_visibility="collapsed")

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 GIẢI MÃ NGAY", use_container_width=True):
        new_records = ai.parse_data(raw_input)
        if new_records:
            history = (new_records + history)[:100]
            save_data(history)
            st.rerun()
        else:
            st.error("Không tìm thấy dãy 5 số. Anh kiểm tra lại nhé!")
with c2:
    if st.button("🗑️ RESET", use_container_width=True):
        save_data([])
        st.rerun()

if history:
    top_5, acc = ai.analyze(history)
    st.markdown(f"<p style='text-align: center; color: #00ff00; font-weight: bold;'>Độ tin cậy: {acc}%</p>", unsafe_allow_html=True)
    
    # Dàn ngang 5 số
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
    
    with st.expander("📜 Lịch sử dữ liệu"):
        for row in history[:10]: st.write("".join(map(str, row)))
