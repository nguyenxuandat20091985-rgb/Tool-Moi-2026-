import streamlit as st
import re, json, os, pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
from openai import OpenAI
import google.generativeai as genai

# --- CẤU HÌNH HỆ THỐNG (GIỮ NGUYÊN) ---
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

# Tuổi Sửu 1985 - Mệnh Kim
LUCKY_OX = [0, 2, 5, 6, 7, 8]
SHADOW_MAP = {"0":"5", "5":"0", "1":"6", "6":"1", "2":"7", "7":"2", "3":"8", "8":"3", "4":"9", "9":"4"}

st.set_page_config(page_title="TITAN V45 - OMNI MATRIX", page_icon="🐂", layout="centered")

# --- GIAO DIỆN PHONG CÁCH ANH ĐẠT ---
st.markdown("""
<style>
    .stApp {background-color: #050505; color: #FFD700;}
    .big-num {font-size: 42px; font-weight: bold; color: #00FFCC; text-align: center; font-family: monospace; letter-spacing: 6px;}
    .box {background: linear-gradient(135deg, #1C3A3A, #050505); color: #FFD700; padding: 15px; border-radius: 12px; text-align: center; border: 2px solid #FFD700; margin-bottom: 10px;}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 12px; border-radius: 8px; text-align: center; font-size: 26px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000;}
    .bet-label {color: #FF3131; font-weight: bold; font-size: 12px; animation: blinker 1.5s linear infinite;}
    @keyframes blinker { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# --- LOGIC THUẬT TOÁN V45 ---
def get_nums(text):
    return [n for n in re.findall(r"\d{5}", text) if n]

def check_gan_v45(db, combo):
    count = 0
    combo_set = set(combo)
    for num in reversed(db):
        if not combo_set.issubset(set(num)): count += 1
        else: break
    return count

def analyze_streaks(db, size=2):
    """Nhận diện các cặp/bộ đang bệt cực mạnh"""
    all_combos = []
    for num in db[-15:]: # Soi bệt trong 15 kỳ gần nhất
        unique_digits = sorted(list(set(num)))
        if len(unique_digits) >= size:
            all_combos.extend(["".join(c) for c in combinations(unique_digits, size)])
    counts = Counter(all_combos)
    return {k: v for k, v in counts.items() if v >= 3} # Chỉ lấy những con xuất hiện >= 3 lần

def predict_v45_omni(db):
    if len(db) < 25: return None
    
    recent_db = db[-120:]
    pair_pool = Counter()
    triple_pool = Counter()
    
    # Nhận diện bệt sớm
    pair_streaks = analyze_streaks(db, 2)
    triple_streaks = analyze_streaks(db, 3)
    
    for num in recent_db:
        digits = sorted(list(set(num)))
        if len(digits) >= 2:
            for p in combinations(digits, 2): pair_pool[p] += 1
        if len(digits) >= 3:
            for t in combinations(digits, 3): triple_pool[t] += 1

    last_num_set = set(db[-1])
    
    # 🎯 XỬ LÝ 2 TINH (MATRIX + STREAK)
    scored_pairs = []
    for p, freq in pair_pool.items():
        p_str = "".join(p)
        gan = check_gan_v45(db, p)
        score = freq * 3.0
        
        # Thưởng bệt (Chống nhà cái lừa cầu ngắt nhịp)
        if p_str in pair_streaks: score += 50 
        # Thưởng nhịp rơi Matrix Golden
        if 1 <= gan <= 4: score += 40
        # Thưởng Tuổi Sửu & Cầu rơi
        if any(d in last_num_set for d in p): score += 20
        if any(int(d) in LUCKY_OX for d in p): score += 10
        # Né số giam
        if gan > 10: score -= 60
        
        scored_pairs.append((p_str, score, gan, p_str in pair_streaks))
        
    # 💎 XỬ LÝ 3 TINH
    scored_triples = []
    for t, freq in triple_pool.items():
        t_str = "".join(t)
        gan = check_gan_v45(db, t)
        score = freq * 4.0
        if t_str in triple_streaks: score += 60
        if 2 <= gan <= 6: score += 45
        if gan > 15: score -= 70
        scored_triples.append((t_str, score, gan, t_str in triple_streaks))

    res_p = sorted(scored_pairs, key=lambda x: x[1], reverse=True)[:3]
    res_t = sorted(scored_triples, key=lambda x: x[1], reverse=True)[:3]
    
    # Phủ sảnh 8 số
    top_8 = "".join([d for d, c in Counter("".join(db[-80:])).most_common(8)])
    
    return {"pairs": res_p, "triples": res_t, "top8": top_8}

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>🐂 TITAN V45 - OMNI MATRIX</h1>', unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán dữ liệu thực tế (Mới nhất ở cuối):", height=150)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 PHÂN TÍCH OMNI"):
        nums = get_nums(user_input)
        if len(nums) >= 20:
            if "last_pred" in st.session_state:
                lp = st.session_state.last_pred
                last_actual = nums[-1]
                best_pair = lp['pairs'][0][0]
                win_check = all(d in last_actual for d in best_pair)
                st.session_state.history.insert(0, {
                    "Kỳ": last_actual, "Cặp": best_pair, "Kết quả": "🔥 WIN" if win_check else "❌"
                })
            st.session_state.last_pred = predict_v40_elite(nums) if 'predict_v40_elite' in globals() else predict_v45_omni(nums)
            st.rerun()
        else:
            st.warning("Dán thêm dữ liệu đi anh Đạt ơi (ít nhất 25 kỳ)!")

with col2:
    if st.button("🗑️ RESET"):
        st.session_state.clear()
        st.rerun()

# --- HIỂN THỊ KẾT QUẢ ---
if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    st.markdown(f"<div class='box'>🔥 ĐỘ PHỦ 8 SỐ: <span class='big-num'>{res['top8']}</span></div>", unsafe_allow_html=True)
    
    # 2 TINH
    st.markdown("<div class='box'>🎯 2 SỐ 5 TINH - CHỐNG LỪA & BỆT</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (p, score, gan, is_streak) in enumerate(res['pairs']):
        with cols[i]:
            st.markdown(f"<div class='item'>{p[0]} , {p[1]}</div>", unsafe_allow_html=True)
            if is_streak: st.markdown("<div class='bet-label'>⚡ ĐANG BỆT mạnh</div>", unsafe_allow_html=True)
            else: st.markdown(f"<div style='color:#aaa;'>Gan: {gan} kỳ</div>", unsafe_allow_html=True)
        
    # 3 TINH
    st.markdown("<div class='box' style='border-color:#FFD700;'>💎 3 SỐ 5 TINH - MATRIX CAO CẤP</div>", unsafe_allow_html=True)
    cols3 = st.columns(3)
    for i, (t, score, gan, is_streak) in enumerate(res['triples']):
        with cols3[i]:
            st.markdown(f"<div class='item item-3'>{t[0]},{t[1]},{t[2]}</div>", unsafe_allow_html=True)
            if is_streak: st.markdown("<div class='bet-label'>⚡ BỆT CỰC ĐẬM</div>", unsafe_allow_html=True)
            else: st.markdown(f"<div style='color:#aaa;'>Gan: {gan} kỳ</div>", unsafe_allow_html=True)

if st.session_state.history:
    st.divider()
    st.subheader("📋 Đối soát thực chiến (Luật Bao Sảnh)")
    st.table(pd.DataFrame(st.session_state.history).head(10))
