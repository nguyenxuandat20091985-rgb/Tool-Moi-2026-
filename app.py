import streamlit as st
from algorithms import TitanAI
import json
import os

# 1. Quản lý dữ liệu lưu trữ
DB_FILE = "history_v8.json"
def load_data():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: return json.load(f)
    return []

def save_data(data):
    with open(DB_FILE, "w") as f: json.dump(data[:100], f)

# 2. Giao diện
st.set_page_config(page_title="TITAN AI v8.0", layout="centered")
st.markdown("<h1 style='text-align: center; color: red;'>🎯 TITAN AI v8.0</h1>", unsafe_allow_html=True)

history = load_data()
ai = TitanAI()

# Ô nhập liệu
raw_input = st.text_area("Dán kỳ mới nhất (Ví dụ: 12345):", height=100)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 GIẢI MÃ TRINITY", use_container_width=True):
        new_kỳ = ai._parse_data(raw_input) # Gọi đúng hàm đã sửa lỗi
        if new_kỳ:
            history = (new_kỳ + history)[:100]
            save_data(history)
            st.rerun()
with col2:
    if st.button("🗑️ RESET", use_container_width=True):
        save_data([])
        st.rerun()

# 3. Hiển thị kết quả
if history:
    top_5, acc = ai.analyze_trinity(history)
    st.markdown(f"<h3 style='text-align: center; color: yellow;'>Độ tin cậy: {acc}%</h3>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.write("💎 **TOP 5 SỐ VÀNG:**")
    cols = st.columns(5)
    for i in range(5):
        cols[i].success(f"**{top_5[i]}**")
    
    st.info(f"💡 Bộ 3 tinh chủ lực: **{top_5[0]} - {top_5[1]} - {top_5[2]}**")
    
    with st.expander("📜 Lịch sử dữ liệu"):
        for row in history[:10]:
            st.write(f"Kỳ mở thưởng: {''.join(map(str, row))}")
else:
    st.info("Hãy nhập ít nhất 1 kỳ để AI bắt đầu phân tích.")
