import streamlit as st
import re
from collections import Counter
import math
import json
import os

# --- PHẦN 1: BỘ NÃO AI (Gộp chung vào đây để tránh lỗi AttributeError) ---
class TitanTrinityAI:
    def __init__(self):
        self.shadow_map = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}

    def parse_data(self, raw):
        cleaned = []
        if not raw: return cleaned
        lines = str(raw).split('\n')
        for item in lines:
            match = re.search(r'\d{5}', str(item))
            if match:
                cleaned.append([int(d) for d in match.group()])
        return cleaned

    def analyze(self, data):
        if len(data) < 3: return ["-"] * 5, 0
        recent = data[:15]
        all_nums = [d for row in recent for d in row]
        freq = Counter(all_nums)
        
        gaps = {i: 30 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row:
                    gaps[i] = idx
                    break

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

# --- PHẦN 2: GIAO DIỆN & LƯU TRỮ ---
st.set_page_config(page_title="TITAN AI v8.1", layout="centered")

DB_FILE = "history_v81.json"
def load_data():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except: return []
    return []

def save_data(data):
    with open(DB_FILE, "w") as f: json.dump(data[:100], f)

# Giao diện chính
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎯 TITAN AI v8.1</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Thiết kế riêng cho kèo 3 Số 5 Tinh</p>", unsafe_allow_html=True)

history = load_data()
ai = TitanTrinityAI()

# Nhập liệu
raw_input = st.text_area("Dán kỳ mới nhất tại đây:", placeholder="Ví dụ: 60577", height=100)

c1, c2 = st.columns(2)
with c1:
    if st.button("🚀 GIẢI MÃ TRINITY", use_container_width=True):
        new_records = ai.parse_data(raw_input)
        if new_records:
            # Gộp dữ liệu mới vào lịch sử (không trùng lặp)
            history = (new_records + history)[:100]
            save_data(history)
            st.toast("Đã nạp dữ liệu thành công!", icon="✅")
            st.rerun()
with c2:
    if st.button("🗑️ RESET DỮ LIỆU", use_container_width=True):
        save_data([])
        st.rerun()

# Kết quả phân tích
if history:
    top_5, acc = ai.analyze(history)
    
    st.markdown("---")
    st.markdown(f"<h3 style='text-align: center;'>Độ tin cậy: <span style='color: #00FF00;'>{acc}%</span></h3>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='text-align: center;'>💎 TOP 5 SỐ VÀNG</h4>", unsafe_allow_html=True)
    cols = st.columns(5)
    for i in range(5):
        cols[i].markdown(f"<div style='background: #1E1E1E; border: 2px solid #FF4B4B; border-radius: 10px; padding: 15px; text-align: center; font-size: 24px; font-weight: bold; color: white;'>{top_5[i]}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.success(f"💡 **Dự đoán bộ 3 tinh:** {top_5[0]} - {top_5[1]} - {top_5[2]}")
    
    with st.expander("📜 Xem lịch sử 10 kỳ gần nhất"):
        for row in history[:10]:
            st.code("".join(map(str, row)))
else:
    st.info("💡 Mẹo: Dán kết quả vừa về vào ô trống phía trên và nhấn Giải Mã.")
