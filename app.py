import streamlit as st
from algorithms import TitanAI
import json
import os

# Cấu hình trang
st.set_page_config(page_title="TITAN AI v8.0 - TRINITY", layout="centered")

# Khởi tạo DB đơn giản
DB_FILE = "history_db.json"
if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as f: json.dump([], f)

def load_db():
    with open(DB_FILE, "r") as f: return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f: json.dump(data[:100], f)

# Giao diện chính
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🎯 TITAN AI v8.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Chuyên biệt: 3 Số 5 Tinh</p>", unsafe_allow_html=True)

# Phần nhập dữ liệu
history = load_db()
raw_input = st.text_area("Dán kỳ mới nhất tại đây (Ví dụ: 12345):", height=100)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 GIẢI MÃ TRINITY", use_container_width=True):
        ai = TitanAI()
        new_kỳ = ai._parse_data(raw_input)
        if new_kỳ:
            history = (new_kỳ + history)[:100]
            save_db(history)
            st.rerun()
with col2:
    if st.button("🗑️ RESET DỮ LIỆU", use_container_width=True):
        save_db([])
        st.rerun()

# Hiển thị kết quả
if history:
    ai = TitanAI()
    top_5, acc = ai.analyze_trinity(history)
    
    # Hiển thị độ tin cậy
    color = "green" if acc > 70 else "orange" if acc > 40 else "red"
    st.markdown(f"<h3 style='text-align: center; color: {color};'>Độ tin cậy: {acc}%</h3>", unsafe_allow_html=True)
    
    # Hiển thị 5 số Vàng
    st.markdown("---")
    st.markdown("<h4 style='text-align: center;'>💎 TOP 5 SỐ VÀNG 5 TINH</h4>", unsafe_allow_html=True)
    cols = st.columns(5)
    for i in range(5):
        cols[i].markdown(f"<div style='background: #1e1e1e; padding: 20px; border-radius: 10px; text-align: center; font-size: 30px; font-weight: bold; color: #ff4b4b; border: 2px solid #ff4b4b;'>{top_5[i]}</div>", unsafe_allow_html=True)
    
    # Gợi ý bộ 3 tinh
    st.info(f"💡 Gợi ý bộ 3 tinh chủ lực: **{top_5[0]}, {top_5[1]}, {top_5[2]}**")
    
    # Bảng lịch sử
    with st.expander("📜 Xem lịch sử 20 kỳ gần nhất"):
        for row in history[:20]:
            st.write(f"Kỳ: {''.join(map(str, row))}")
else:
    st.warning("Vui lòng nhập dữ liệu để bắt đầu soi cầu.")
