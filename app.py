import streamlit as st
import re, json, os, pandas as pd
from collections import Counter
from itertools import combinations
from openai import OpenAI
import google.generativeai as genai

# --- CẤU HÌNH HỆ THỐNG ---
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

# Tuổi Sửu 1985 - Mệnh Kim
LUCKY_OX = [0, 2, 5, 6, 7, 8]
SHADOW_MAP = {"0":"5", "5":"0", "1":"6", "6":"1", "2":"7", "7":"2", "3":"8", "8":"3", "4":"9", "9":"4"}

st.set_page_config(page_title="TITAN V37 - 5 TINH PRO", page_icon="🐂", layout="centered")

# --- GIAO DIỆN (GIỮ PHONG CÁCH ANH THÍCH) ---
st.markdown("""
<style>
    .stApp {background-color: #050505; color: #FFD700;}
    .big-num {font-size: 42px; font-weight: bold; color: #00FFCC; text-align: center; font-family: monospace; letter-spacing: 6px;}
    .box {background: linear-gradient(135deg, #1C3A3A, #050505); color: #FFD700; padding: 15px; border-radius: 12px; text-align: center; border: 2px solid #FFD700; margin-bottom: 10px;}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 12px; border-radius: 8px; text-align: center; font-size: 26px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000;}
</style>
""", unsafe_allow_html=True)

# --- LOGIC THUẬT TOÁN 5 TINH ---
def get_nums(text):
    return [n for n in re.findall(r"\d{5}", text) if n]

def predict_5tinh_v37(db):
    if len(db) < 10: return None
    
    # 1. Soi tần suất cặp (2 số) và bộ (3 số) hay về cùng nhau
    pair_counts = Counter()
    triple_counts = Counter()
    
    # Chỉ lấy 60 kỳ gần nhất để bắt nhịp hiện tại
    for num in db[-60:]:
        unique_digits = sorted(list(set(num))) # Luật 5 tinh chỉ cần xuất hiện, không cần số lượng
        if len(unique_digits) >= 2:
            for p in combinations(unique_digits, 2): pair_counts["".join(p)] += 1
        if len(unique_digits) >= 3:
            for t in combinations(unique_digits, 3): triple_counts["".join(t)] += 1

    # 2. Điểm thưởng cho nhịp "Cầu Rơi" (Số kỳ trước nổ lại)
    last_num_set = set(db[-1])
    
    # Lọc 2 TINH (Top 3 cặp mạnh nhất)
    scored_pairs = []
    for p, count in pair_counts.items():
        score = count * 2.5
        if any(d in last_num_set for d in p): score += 15 # Thưởng cầu rơi
        if any(int(d) in LUCKY_OX for d in p): score += 8 # Thưởng tuổi Sửu
        scored_pairs.append((p, score))
    
    # Lọc 3 TINH (Top 3 bộ mạnh nhất)
    scored_triples = []
    for t, count in triple_counts.items():
        score = count * 3.0
        if any(d in last_num_set for d in t): score += 20 
        if any(int(d) in LUCKY_OX for d in t): score += 10
        scored_triples.append((t, score))

    final_pairs = [x[0] for x in sorted(scored_pairs, key=lambda x: x[1], reverse=True)][:3]
    final_triples = [x[0] for x in sorted(scored_triples, key=lambda x: x[1], reverse=True)][:3]
    
    # Tần suất đơn lẻ cho Top 8
    all_digits = "".join(db[-40:])
    top_8 = "".join([d for d, c in Counter(all_digits).most_common(8)])
    
    return {"pairs": final_pairs, "triples": final_triples, "top8": top_8}

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>🐂 TITAN V37 - CHUYÊN BIỆT 5 TINH</h1>', unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán bảng kết quả thực tế vào đây:", height=150)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 CHỐT TỔ HỢP 5 TINH"):
        nums = get_nums(user_input)
        if len(nums) >= 5:
            # Đối soát thắng thua luật Bao sảnh
            if "last_pred" in st.session_state:
                lp = st.session_state.last_pred
                last_actual = nums[-1]
                # Thắng nếu TẤT CẢ các số trong cặp đều có mặt trong kết quả
                win_check = all(d in last_actual for d in lp['pairs'][0])
                st.session_state.history.insert(0, {"Kỳ": last_actual, "Cặp soi": lp['pairs'][0], "Kết quả": "🔥 WIN" if win_check else "❌"})

            st.session_state.last_pred = predict_5tinh_v37(nums)
            st.rerun()
        else:
            st.warning("Dán thêm số đi anh Đạt ơi!")

with col2:
    if st.button("🗑️ RESET"):
        st.session_state.clear()
        st.rerun()

# --- HIỂN THỊ KẾT QUẢ ---
if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    st.markdown(f"<div class='box'>🔥 ĐỘ PHỦ 5 TINH: {res['top8']}</div>", unsafe_allow_html=True)
    
    # 2 TINH
    st.markdown("<div class='box'>🎯 2 SỐ 5 TINH (Cược 2 số bất kỳ)</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for i, p in enumerate(res['pairs']):
        with [c1, c2, c3][i]: st.markdown(f"<div class='item'>{p[0]} , {p[1]}</div>", unsafe_allow_html=True)
        
    # 3 TINH
    st.markdown("<div class='box' style='border-color:#FFD700;'>💎 3 SỐ 5 TINH (Cược 3 số bất kỳ)</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for i, t in enumerate(res['triples']):
        with [d1, d2, d3][i]: st.markdown(f"<div class='item item-3'>{t[0]},{t[1]},{t[2]}</div>", unsafe_allow_html=True)

# --- BẢNG ĐỐI SOÁT ---
if st.session_state.history:
    st.divider()
    st.subheader("📋 Đối soát thắng/thua (Bao sảnh)")
    st.table(pd.DataFrame(st.session_state.history).head(10))
