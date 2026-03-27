import streamlit as st
import re, json, os, pandas as pd
from collections import Counter
from itertools import combinations
from openai import OpenAI
import google.generativeai as genai

# --- CẤU HÌNH API ---
NV_KEY = "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"
GM_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_v28_final.json"

# --- PHONG THỦY TUỔI SỬU 1985 (MỆNH KIM) ---
LUCKY_OX = [0, 2, 5, 6, 7, 8]
SHADOW_MAP = {"0":"5", "5":"0", "1":"6", "6":"1", "2":"7", "7":"2", "3":"8", "8":"3", "4":"9", "9":"4"}

st.set_page_config(page_title="TITAN V28 | 2-3 TINH ELITE", layout="centered")

# --- GIAO DIỆN DARK GOLD ---
st.markdown("""
<style>
    .stApp {background-color: #050505; color: #FFD700;}
    .main-box {background: #111; border: 2px solid #FFD700; border-radius: 15px; padding: 20px; box-shadow: 0 0 15px #FFD700;}
    .item-2 {background: linear-gradient(135deg, #FFD700, #B8860B); color: #000; padding: 15px; border-radius: 10px; font-size: 28px; font-weight: 900; text-align: center; margin: 5px;}
    .item-3 {background: linear-gradient(135deg, #C0C0C0, #708090); color: #000; padding: 15px; border-radius: 10px; font-size: 28px; font-weight: 900; text-align: center; margin: 5px;}
    .status {font-size: 14px; color: #C0C0C0; text-align: center; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

def analyze_logic(db):
    if len(db) < 10: return None
    
    all_digits = "".join(db[-60:])
    last_num = db[-1]
    
    # 1. Tính điểm tần suất cơ bản
    scores = {str(i): all_digits.count(str(i)) * 1.5 for i in range(10)}
    
    # 2. Cộng điểm Cầu Rơi (Số vừa về)
    for digit in set(last_num):
        scores[digit] += 20 
        
    # 3. Cộng điểm Bóng Ngũ Hành (Kim - Thổ)
    for digit in set(last_num):
        shadow = SHADOW_MAP.get(digit)
        if shadow: scores[shadow] += 15
        
    # 4. Cộng điểm Tuổi Sửu
    for d in LUCKY_OX:
        scores[str(d)] += 12

    # Trích xuất Top số mạnh nhất
    sorted_nums = [x[0] for x in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    
    # GHÉP 2 TINH (Chọn 3 cặp từ Top 5 số điểm cao nhất)
    pool_2 = sorted_nums[:5]
    pairs = ["".join(p) for p in combinations(pool_2, 2)][:3]
    
    # GHÉP 3 TINH (Chọn 3 bộ từ Top 6 số điểm cao nhất)
    pool_3 = sorted_nums[:6]
    triples = ["".join(t) for t in combinations(pool_3, 3)][:3]
    
    return {"pairs": pairs, "triples": triples, "top8": "".join(sorted_nums[:8])}

# --- NỘI DUNG APP ---
st.markdown("<h1 style='text-align: center; color: #FFD700;'>🐂 TITAN V28 - CHUYÊN 2/3 TINH</h1>", unsafe_allow_html=True)

if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: st.session_state.db = json.load(f)
    else: st.session_state.db = []

input_data = st.text_area("📡 Dán dữ liệu 5D (30-50 kỳ):", height=120, placeholder="12345\n67890...")

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("🚀 CHỐT SỐ NGAY", use_container_width=True):
        nums = re.findall(r"\d{5}", input_data)
        if nums:
            st.session_state.db = nums[-100:]
            with open(DB_FILE, "w") as f: json.dump(st.session_state.db, f)
            st.session_state.res = analyze_logic(st.session_state.db)
            st.rerun()
with col_btn2:
    if st.button("🗑️ XÓA DỮ LIỆU", use_container_width=True):
        st.session_state.db = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

if "res" in st.session_state and st.session_state.res:
    res = st.session_state.res
    st.markdown("<div class='status'>PHÂN TÍCH BIAS + BÓNG NGŨ HÀNH + TUỔI SỬU</div>", unsafe_allow_html=True)
    
    # HIỂN THỊ 2 TINH
    st.markdown("<div class='main-box'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>🎯 2 TINH (3 CẶP VÀNG)</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='item-2'>{res['pairs'][0]}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='item-2'>{res['pairs'][1]}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='item-2'>{res['pairs'][2]}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # HIỂN THỊ 3 TINH
    st.markdown("<div class='main-box' style='margin-top:15px; border-color:#C0C0C0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#C0C0C0;'>💎 3 TINH (3 BỘ KHỦNG)</h3>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    d1.markdown(f<div class='item-3'>{res['triples'][0]}</div>", unsafe_allow_html=True)
    d2.markdown(f"<div class='item-3'>{res['triples'][1]}</div>", unsafe_allow_html=True)
    d3.markdown(f"<div class='item-3'>{res['triples'][2]}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<p style='text-align:center; color:#888; margin-top:10px;'>Dàn 8 số hỗ trợ: <b>{res['top8']}</b></p>", unsafe_allow_html=True)
