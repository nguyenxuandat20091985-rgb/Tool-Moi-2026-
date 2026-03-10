import streamlit as st
import pandas as pd
import time
import re
import json
import os
from collections import Counter
import math

# ==============================================================================
# CONFIGURATION & UI STYLE
# ==============================================================================
st.set_page_config(page_title="TITAN OMNI v9.0", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background: #010409; color: #e6edf3; }
    .header-card {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        padding: 20px; border-radius: 15px; border-bottom: 4px solid #ffd700;
        margin-bottom: 20px; text-align: center;
    }
    .num-box {
        font-size: calc(40px + 4vw); font-weight: 900; color: #ffd700;
        text-align: center; letter-spacing: 10px;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.4);
        line-height: 1.1; margin: 10px 0;
    }
    .status-bar {
        padding: 12px; border-radius: 10px; text-align: center;
        font-weight: 800; font-size: 1.2rem; margin-bottom: 10px;
    }
    .prediction-card {
        background: #0d1117; border: 2px solid #30363d;
        border-radius: 20px; padding: 25px; margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CORE LOGIC (Database & AI Engine)
# ==============================================================================
class DatabaseManager:
    def __init__(self):
        self.file_path = "history_v9.json"
        self.data = self.load()

    def load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f: return json.load(f)
        return []

    def save(self):
        with open(self.file_path, "w") as f: json.dump(self.data[:200], f)

    def clean_data(self, raw):
        found = re.findall(r'\d{5}', str(raw))
        return [[int(d) for d in item] for item in found]

    def add_numbers(self, new_nums):
        # Tránh trùng lặp kỳ quay
        for num in reversed(new_nums):
            if num not in self.data:
                self.data.insert(0, num)
        self.save()

class TitanAI:
    def analyze(self, data):
        if len(data) < 3: return None
        recent = data[:30]
        all_nums = [d for row in recent for d in row]
        freq = Counter(all_nums)
        
        gaps = {i: 30 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row: gaps[i] = idx; break

        scores = {}
        for i in range(10):
            # Thuật toán nhịp 3 tinh
            f_score = freq.get(i, 0) * 5
            g_score = 45 if gaps[i] == 1 else 20 if gaps[i] == 0 else 0
            scores[str(i)] = f_score + g_score
            
        top_nums = [n for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        
        return {
            "m3": "".join(top_nums[:3]),
            "l4": "".join(top_nums[3:7]),
            "win_rate": int(max(45, min(98, 100 - (len(set(all_nums)) * 2)))),
            "logic": "Ưu tiên nhịp Gaps 1 và tần suất rơi bệt."
        }

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    db = DatabaseManager()
    ai = TitanAI()

    st.markdown("""<div class="header-card"><div style="font-size:30px; font-weight:900; color:#ffd700;">🛡️ TITAN OMNI v9.0</div>
                <div style="color: #8b949e;">CHUYÊN GIA 3 SỐ 5 TINH | PRO EDITION</div></div>""", unsafe_allow_html=True)

    # Stats
    c1, c2, c3 = st.columns(3)
    c1.metric("📦 Tổng kỳ", len(db.data))
    
    # Input Area
    raw_input = st.text_area("Dán kết quả tại đây:", height=100, placeholder="Ví dụ: 60577...")
    
    col_b1, col_b2, col_b3 = st.columns([2,1,1])
    if col_b1.button("🚀 GIẢI MÃ MATRIX AI", type="primary", use_container_width=True):
        new_nums = db.clean_data(raw_input)
        if new_nums:
            db.add_numbers(new_nums)
            st.rerun()
        else:
            st.error("Không tìm thấy dãy 5 số hợp lệ!")

    if col_b2.button("🔄 Làm mới", use_container_width=True): st.rerun()
    if col_b3.button("🗑️ Reset", use_container_width=True):
        db.data = []
        db.save()
        st.rerun()

    # Phân tích và Hiển thị
    res = ai.analyze(db.data)
    if res:
        st.markdown(f'<div class="status-bar" style="background: #238636;">📢 TRẠNG THÁI: SẴN SÀNG VÀO TIỀN</div>', unsafe_allow_html=True)
        
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#8b949e; margin:0;'>💎 3 SỐ 5 TINH CHỦ LỰC</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res['m3']}</div>", unsafe_allow_html=True)
        
        st.markdown("<p style='text-align:center; color:#8b949e; margin:0;'>🛡️ DÀN LÓT (4 SỐ)</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; font-size:35px; color:#58a6ff; font-weight:bold;'>{res['l4']}</p>", unsafe_allow_html=True)
        
        st.divider()
        st.write(f"🧠 **LOGIC AI:** {res['logic']}")
        st.text_input("📋 DÀN 7 SỐ (Copy):", "".join(sorted(set(res['m3'] + res['l4']))))
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Hiển thị lịch sử để kiểm tra
    with st.expander("📜 Lịch sử dữ liệu đã nạp"):
        for row in db.data[:10]:
            st.write("".join(map(str, row)))

if __name__ == "__main__":
    main()
