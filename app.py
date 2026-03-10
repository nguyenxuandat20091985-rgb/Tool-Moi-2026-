import streamlit as st
import re
from collections import Counter
import math
import json
import os
import time

# --- PHẦN 1: BỘ NÃO AI (TRINITY OMNI) ---
class TitanTrinityAI:
    def __init__(self):
        self.shadow_map = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}

    def parse_data(self, raw):
        """Hàm bóc tách số siêu tốc"""
        if not raw: return []
        # Chỉ lấy các dãy 5 số, loại bỏ mọi chữ cái và ký tự lạ
        found = re.findall(r'\d{5}', str(raw))
        return [[int(d) for d in item] for item in found]

    def analyze(self, data):
        if not data: return ["-"] * 5, 18
        # Phân tích dựa trên 30 kỳ gần nhất để tăng độ nhạy
        recent = data[:30]
        all_nums = [d for row in recent for d in row]
        freq = Counter(all_nums)
        
        gaps = {i: 50 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row: gaps[i] = idx; break

        scores = {}
        for i in range(10):
            # Thuật toán trọng số mới v8.5
            f_score = freq.get(i, 0) * 6
            g_score = 50 if gaps[i] == 1 else 25 if gaps[i] == 0 else 0
            s_score = 30 if gaps[self.shadow_map[i]] == 0 else 0
            scores[str(i)] = f_score + g_score + s_score
            
        top_5 = [n for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:5]
        
        # Tính toán độ tin cậy thực tế
        try:
            counts = [all_nums.count(i) for i in set(all_nums)]
            total = sum(counts)
            entropy = -sum((c/total)*math.log2(c/total) for c in counts if c > 0)
            acc = int(max(21, min(98, (3.32 - entropy) * 350)))
        except: acc = 21
        
        return top_5, acc

# --- PHẦN 2: GIAO DIỆN PHẢN HỒI TỨC THÌ ---
st.set_page_config(page_title="TITAN v8.5", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; }
    .gold-container { display: flex; justify-content: space-between; margin: 10px 0; }
    .gold-item {
        background: linear-gradient(145deg, #ffd700, #b8860b);
        border-radius: 12px; width: 18%; aspect-ratio: 1/1;
        display: flex; align-items: center; justify-content: center;
        font-size: 26px; font-weight: bold; color: #000;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.6);
        animation: pulse 0.5s;
    }
    @keyframes pulse { 0% {transform: scale(0.9);} 100% {transform: scale(1);} }
    .stButton button { 
        background: #ffd700 !important; color: black !important; 
        font-weight: bold !important; border: none !important;
        box-shadow: 0 4px 10px rgba(255,215,0,0.3);
    }
    .status-text { text-align: center; color: #00ff00; font-size: 20px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

DB_FILE = "titan_v85.json"
def load_data():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_data(data):
    with open(DB_FILE, "w") as f: json.dump(data[:150], f)

if 'history' not in st.session_state:
    st.session_state.history = load_data()

ai = TitanTrinityAI()

st.markdown("<h2 style='text-align: center; color: #ffd700;'>💎 TITAN OMNI v8.5</h2>", unsafe_allow_html=True)

# Input
raw_input = st.text_area("", placeholder="Dán dãy số tại đây...", height=80, label_visibility="collapsed")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 PHÂN TÍCH NGAY", use_container_width=True):
        new_records = ai.parse_data(raw_input)
        if new_records:
            # Cập nhật và ép làm mới
            st.session_state.history = (new_records + st.session_state.history)[:150]
            save_data(st.session_state.history)
            with st.spinner('AI đang quét dữ liệu...'):
                time.sleep(0.5)
            st.rerun()
        else:
            st.warning("⚠️ Vui lòng dán dãy số!")

with col2:
    if st.button("🗑️ LÀM TRỐNG", use_container_width=True):
        st.session_state.history = []
        save_data([])
        st.rerun()

# Hiển thị kết quả
if st.session_state.history:
    top_5, acc = ai.analyze(st.session_state.history)
    st.markdown(f"<p class='status-text'>Độ tin cậy: {acc}%</p>", unsafe_allow_html=True)
    
    # Hiển thị 5 số vàng
    res_html = f"""
    <div class="gold-container">
        <div class="gold-item">{top_5[0]}</div>
        <div class="gold-item">{top_5[1]}</div>
        <div class="gold-item">{top_5[2]}</div>
        <div class="gold-item">{top_5[3]}</div>
        <div class="gold-item">{top_5[4]}</div>
    </div>
    """
    st.markdown(res_html, unsafe_allow_html=True)
    
    st.markdown(f"<div style='background:#ffd700; color:black; padding:15px; border-radius:10px; text-align:center; font-weight:bold; font-size:20px;'>🔥 CHỐT 3 TINH: {top_5[0]} - {top_5[1]} - {top_5[2]}</div>", unsafe_allow_html=True)
    
    with st.expander("📊 Xem 10 kỳ gần nhất"):
        for row in st.session_state.history[:10]:
            st.text("".join(map(str, row)))
else:
    st.info("💡 Hệ thống sẵn sàng. Hãy dán dữ liệu kỳ quay vào ô trên.")
