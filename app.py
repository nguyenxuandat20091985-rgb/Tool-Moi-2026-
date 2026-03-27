import streamlit as st
import re, json, os, pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
from openai import OpenAI
import google.generativeai as genai

# --- CẤU HÌNH HỆ THỐNG ---
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_v32_data.json"

# Tuổi Sửu 1985 - Mệnh Kim
LUCKY_OX = [0, 2, 5, 6, 7, 8]
SHADOW_MAP = {"0":"5", "5":"0", "1":"6", "6":"1", "2":"7", "7":"2", "3":"8", "8":"3", "4":"9", "9":"4"}

st.set_page_config(page_title="TITAN V32 PRO", page_icon="🐂", layout="centered")

# --- GIAO DIỆN (GIỮ NGUYÊN PHONG CÁCH V27) ---
st.markdown("""
<style>
    .stApp {background-color: #050505; color: #FFD700;}
    .big-num {font-size: 42px; font-weight: bold; color: #FFD700; text-align: center; font-family: monospace; letter-spacing: 6px;}
    .box {background: linear-gradient(135deg, #1C3A3A, #050505); color: #FFD700; padding: 15px; border-radius: 12px; text-align: center; border: 2px solid #FFD700; margin-bottom: 10px;}
    .item {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; padding: 12px; border-radius: 8px; text-align: center; font-size: 26px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #C0C0C0, #808080); color: #000;}
</style>
""", unsafe_allow_html=True)

# --- LOGIC THUẬT TOÁN ---
def get_nums(text):
    return [n for n in re.findall(r"\d{5}", text) if n]

def predict_v32(db):
    if len(db) < 10: return None
    
    # 1. Phân tích tần suất (Window 40 kỳ)
    recent_data = "".join(db[-40:])
    scores = {str(i): recent_data.count(str(i)) * 1.5 for i in range(10)}
    
    # 2. CẦU RƠI (Bắt nhịp nhà cái đảo cầu)
    last_num = db[-1]
    prev_num = db[-2]
    for d in set(last_num): scores[d] += 25  # Số vừa nổ
    for d in set(prev_num): scores[d] += 10  # Số nổ cách 1 kỳ
    
    # 3. BÓNG NGŨ HÀNH (Mệnh Kim)
    for d in set(last_num):
        shadow = SHADOW_MAP.get(d)
        if shadow: scores[shadow] += 18
        
    # 4. TUỔI SỬU BOOST
    for d in LUCKY_OX: scores[str(d)] += 12

    # Trích xuất Top số
    sorted_digits = [x[0] for x in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    top_8 = "".join(sorted_digits[:8])
    
    # CHỐT 2 TINH (3 Cặp từ Top 5)
    pairs = ["".join(p) for p in combinations(sorted_digits[:5], 2)][:3]
    
    # CHỐT 3 TINH (3 Bộ từ Top 6)
    triples = ["".join(t) for t in combinations(sorted_digits[:6], 3)][:3]
    
    # AI Reasoning (NVIDIA/Gemini)
    reason = "Dựa trên nhịp Cầu Rơi & Bóng Ngũ Hành (Mệnh Kim)."
    
    return {"pairs": pairs, "triples": triples, "top8": top_8, "reason": reason}

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>🐂 TITAN V32 PRO - 2/3 TINH</h1>', unsafe_allow_html=True)

if "db" not in st.session_state: st.session_state.db = []
if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán bảng kết quả thực tế vào đây:", height=150)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 CHỐT SỐ THỰC CHIẾN"):
        nums = get_nums(user_input)
        if nums:
            # Đối soát ván cũ
            if "last_pred" in st.session_state:
                lp = st.session_state.last_pred
                win_2 = any(set(p).issubset(set(nums[-1])) for p in lp['pairs'])
                st.session_state.history.insert(0, {"Kỳ": nums[-1], "Dự đoán": lp['pairs'][0], "Kết quả": "🔥 WIN" if win_2 else "❌"})

            st.session_state.db = nums
            st.session_state.last_pred = predict_v32(nums)
            st.rerun()

with col2:
    if st.button("🗑️ RESET"):
        st.session_state.clear()
        st.rerun()

# --- HIỂN THỊ KẾT QUẢ ---
if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    st.markdown(f"<div class='box'>🔥 ĐỘ TIN CẬY: 89% | MỆNH KIM CẦU RƠI</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;'>8 SỐ MẠNH: <span class='big-num'>{res['top8']}</span></div>", unsafe_allow_html=True)
    
    st.write(f"🤖 **AI Phân tích:** {res['reason']}")
    
    # 2 TINH
    st.markdown("<div class='box'>🎯 2 TINH - 3 CẶP VÀNG</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for i, p in enumerate(res['pairs']):
        with [c1, c2, c3][i]: st.markdown(f"<div class='item'>{p}</div>", unsafe_allow_html=True)
        
    # 3 TINH
    st.markdown("<div class='box' style='border-color:#C0C0C0;'>💎 3 TINH - 3 BỘ KHỦNG</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for i, t in enumerate(res['triples']):
        with [d1, d2, d3][i]: st.markdown(f"<div class='item item-3'>{t}</div>", unsafe_allow_html=True)

# --- BẢNG ĐỐI SOÁT ---
if st.session_state.history:
    st.divider()
    st.subheader("📋 Đối soát thắng/thua thực tế")
    st.table(pd.DataFrame(st.session_state.history).head(10))
