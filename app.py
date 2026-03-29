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

st.set_page_config(page_title="TITAN V40 - ELITE MATRIX", page_icon="🐂", layout="centered")

# --- GIAO DIỆN DARK GOLDEN (GIỮ PHONG CÁCH ANH THÍCH) ---
st.markdown("""
<style>
    .stApp {background-color: #050505; color: #FFD700;}
    .big-num {font-size: 42px; font-weight: bold; color: #00FFCC; text-align: center; font-family: monospace; letter-spacing: 6px;}
    .box {background: linear-gradient(135deg, #1C3A3A, #050505); color: #FFD700; padding: 15px; border-radius: 12px; text-align: center; border: 2px solid #FFD700; margin-bottom: 10px;}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 12px; border-radius: 8px; text-align: center; font-size: 26px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000;}
    .gan-label {font-size: 13px; color: #FF3131; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- LOGIC THUẬT TOÁN MATRIX NÂNG CẤP ---
def get_nums(text):
    return [n for n in re.findall(r"\d{5}", text) if n]

def check_gan(db, combo):
    """Tính nhịp gan thực tế của tổ hợp số"""
    count = 0
    combo_set = set(combo)
    for num in reversed(db):
        if not combo_set.issubset(set(num)):
            count += 1
        else:
            break
    return count

def predict_v40_elite(db):
    if len(db) < 20: return None
    
    # Phân tích sâu 100 kỳ để bắt nhịp Delta
    recent_db = db[-100:]
    pair_pool = Counter()
    triple_pool = Counter()
    
    for num in recent_db:
        unique_digits = sorted(list(set(num)))
        if len(unique_digits) >= 2:
            for p in combinations(unique_digits, 2): pair_pool[p] += 1
        if len(unique_digits) >= 3:
            for t in combinations(unique_digits, 3): triple_pool[t] += 1

    last_num_set = set(db[-1])
    
    # 🎯 XỬ LÝ 2 TINH (CẶP)
    scored_pairs = []
    for p, freq in pair_pool.items():
        gan = check_gan(db, p)
        # Điểm Matrix: Tần suất + Thưởng nhịp rơi Golden (2-5 kỳ)
        score = freq * 2.8
        if 2 <= gan <= 5: score += 35 
        if any(d in last_num_set for d in p): score += 15 # Cầu rơi
        if any(int(d) in LUCKY_OX for d in p): score += 10 # Tuổi Sửu
        if gan > 12: score -= 50 # Né số giam quá lâu
        
        scored_pairs.append(("".join(p), score, gan))
        
    # 💎 XỬ LÝ 3 TINH (BỘ)
    scored_triples = []
    for t, freq in triple_pool.items():
        gan = check_gan(db, t)
        score = freq * 3.5
        if 3 <= gan <= 7: score += 40
        if any(d in last_num_set for d in t): score += 20
        if gan > 18: score -= 60
        scored_triples.append(("".join(t), score, gan))

    res_p = sorted(scored_pairs, key=lambda x: x[1], reverse=True)[:3]
    res_t = sorted(scored_triples, key=lambda x: x[1], reverse=True)[:3]
    
    # Top 8 số bao sảnh mạnh nhất
    top_8 = "".join([d for d, c in Counter("".join(db[-60:])).most_common(8)])
    
    return {"pairs": res_p, "triples": res_t, "top8": top_8}

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>🐂 TITAN V40 - ELITE MATRIX</h1>', unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán bảng kết quả thực tế vào đây (Càng nhiều càng chuẩn):", height=150)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 PHÂN TÍCH MATRIX"):
        nums = get_nums(user_input)
        if len(nums) >= 15:
            if "last_pred" in st.session_state:
                lp = st.session_state.last_pred
                last_actual = nums[-1]
                # Kiểm tra thắng 5 tinh cho cặp đầu tiên
                win_check = all(d in last_actual for d in lp['pairs'][0][0])
                st.session_state.history.insert(0, {
                    "Kỳ": last_actual, 
                    "Cặp": lp['pairs'][0][0], 
                    "Gan": lp['pairs'][0][2],
                    "KQ": "🔥 WIN" if win_check else "❌"
                })

            st.session_state.last_pred = predict_v40_elite(nums)
            st.rerun()
        else:
            st.warning("Anh Đạt dán thêm kết quả đi, ít nhất 15 kỳ mới soi được nhịp Matrix!")

with col2:
    if st.button("🗑️ RESET"):
        st.session_state.clear()
        st.rerun()

# --- HIỂN THỊ KẾT QUẢ ---
if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    st.markdown(f"<div class='box'>🔥 TỶ LỆ PHỦ SẢNH 8 SỐ: <span class='big-num'>{res['top8']}</span></div>", unsafe_allow_html=True)
    
    # 2 TINH
    st.markdown("<div class='box'>🎯 2 SỐ 5 TINH - ĐIỂM RƠI MATRIX</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for i, (p, score, gan) in enumerate(res['pairs']):
        with [c1, c2, c3][i]: 
            st.markdown(f"<div class='item'>{p[0]} , {p[1]}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='gan-label'>Gan: {gan} kỳ</div>", unsafe_allow_html=True)
        
    # 3 TINH
    st.markdown("<div class='box' style='border-color:#FFD700;'>💎 3 SỐ 5 TINH - BỘ BA TINH TÚY</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for i, (t, score, gan) in enumerate(res['triples']):
        with [d1, d2, d3][i]: 
            st.markdown(f"<div class='item item-3'>{t[0]},{t[1]},{t[2]}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='gan-label'>Gan: {gan} kỳ</div>", unsafe_allow_html=True)

# --- BẢNG ĐỐI SOÁT ---
if st.session_state.history:
    st.divider()
    st.subheader("📋 Nhật ký thực chiến (Bao sảnh 5 Tinh)")
    st.table(pd.DataFrame(st.session_state.history).head(10))
