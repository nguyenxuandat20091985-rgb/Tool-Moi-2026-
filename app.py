import streamlit as st
import re, json, os, pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
from openai import OpenAI
import google.generativeai as genai

# --- CẤU HÌNH HỆ THỐNG (GIỮ NGUYÊN BẢO MẬT) ---
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

# Tuổi Sửu 1985 - Mệnh Kim
LUCKY_OX = [0, 2, 5, 6, 7, 8]

st.set_page_config(page_title="TITAN V45 - ANTI-CHEAT", page_icon="🐂", layout="centered")

# --- GIAO DIỆN DARK MODE PRO ---
st.markdown("""
<style>
    .stApp {background-color: #030303; color: #FFD700;}
    .big-num {font-size: 38px; font-weight: bold; color: #00FFCC; text-align: center; font-family: 'Courier New', monospace; letter-spacing: 4px;}
    .box {background: linear-gradient(135deg, #0d1b1b, #030303); color: #FFD700; padding: 18px; border-radius: 15px; text-align: center; border: 1px solid #FFD700; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(255, 215, 0, 0.1);}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 12px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000;}
    .status-win {color: #00FF00; font-weight: bold;}
    .status-lose {color: #FF3131; font-weight: bold;}
    .streak-badge {background-color: #FF3131; color: white; padding: 2px 8px; border-radius: 5px; font-size: 10px;}
</style>
""", unsafe_allow_html=True)

# --- THUẬT TOÁN PHÂN TÍCH CHUYÊN SÂU ---
def get_nums(text):
    return [n for n in re.findall(r"\d{5}", text) if n]

def check_gan_and_streak(db, combo):
    """Tính toán nhịp Gan và nhịp Bệt của tổ hợp"""
    gan = 0
    streak = 0
    combo_set = set(combo)
    
    # Tính Gan
    for num in reversed(db):
        if not combo_set.issubset(set(num)): gan += 1
        else: break
            
    # Tính Bệt (Streak) - Kiểm tra 5 kỳ gần nhất xem nổ mấy kỳ liên tục
    for num in reversed(db):
        if combo_set.issubset(set(num)): streak += 1
        else: break
            
    return gan, streak

def predict_v45_anti_cheat(db):
    if len(db) < 20: return None
    
    recent_db = db[-120:] # Phân tích mẫu rộng hơn để chống lừa cầu
    pair_pool = Counter()
    triple_pool = Counter()
    single_pool = Counter("".join(recent_db))
    
    # 1. Thu thập dữ liệu tần suất
    for num in recent_db:
        u = sorted(list(set(num)))
        if len(u) >= 2:
            for p in combinations(u, 2): pair_pool[p] += 1
        if len(u) >= 3:
            for t in combinations(u, 3): triple_pool[t] += 1

    last_num_set = set(db[-1])
    
    # 2. Xử lý 2 TINH - Điểm Bayes & Chống Lừa
    scored_pairs = []
    for p, freq in pair_pool.items():
        gan, streak = check_gan_and_streak(db, p)
        
        # Chỉ số "Chống Lừa": Nếu 2 số đơn nổ nhiều nhưng ít đi cùng nhau -> Giảm điểm
        co_occurrence_rate = freq / ((single_pool[p[0]] + single_pool[p[1]]) / 2)
        
        score = freq * 3.0
        if streak >= 1: score += 50  # Thưởng nhịp Bệt (Đang bệt là phải bám)
        if 1 <= gan <= 4: score += 40 # Thưởng nhịp rơi chuẩn
        if co_occurrence_rate > 0.6: score += 25 # Thưởng cặp số "trung thành"
        if any(int(d) in LUCKY_OX for d in p): score += 10 # Tuổi Sửu 1985
        if gan > 15: score -= 70 # Loại bỏ số bị nhà cái giam
        
        scored_pairs.append(("".join(p), score, gan, streak))
        
    # 3. Xử lý 3 TINH
    scored_triples = []
    for t, freq in triple_pool.items():
        gan, streak = check_gan_and_streak(db, t)
        score = freq * 4.0
        if streak >= 1: score += 60
        if 2 <= gan <= 6: score += 45
        if gan > 20: score -= 80
        scored_triples.append(("".join(t), score, gan, streak))

    res_p = sorted(scored_pairs, key=lambda x: x[1], reverse=True)[:3]
    res_t = sorted(scored_triples, key=lambda x: x[1], reverse=True)[:3]
    
    # Phủ sảnh 8 số dùng Bayes đơn lẻ
    top_8 = "".join([d for d, c in Counter("".join(db[-50:])).most_common(8)])
    
    return {"pairs": res_p, "triples": res_t, "top8": top_8}

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>🐂 TITAN V45 - ANTI-CHEAT MATRIX</h1>', unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán bảng kết quả thực tế (Kỳ mới nhất ở dưới cùng):", height=150)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 SOI CẦU CHỐNG LỪA"):
        nums = get_nums(user_input)
        if len(nums) >= 15:
            if "last_pred" in st.session_state:
                lp = st.session_state.last_pred
                last_actual = nums[-1]
                # Kiểm tra thắng cho cặp VIP nhất
                best_pair = lp['pairs'][0][0]
                win_check = all(d in last_actual for d in best_pair)
                st.session_state.history.insert(0, {
                    "Kỳ": last_actual, 
                    "Cặp": best_pair, 
                    "Nhịp": f"G:{lp['pairs'][0][2]} B:{lp['pairs'][0][3]}",
                    "KQ": "🔥 WIN" if win_check else "❌"
                })

            st.session_state.last_pred = predict_v45_anti_cheat(nums)
            st.rerun()
        else:
            st.warning("Dán thêm dữ liệu đi anh Đạt ơi (ít nhất 15 kỳ)!")

with col2:
    if st.button("🗑️ LÀM MỚI"):
        st.session_state.clear()
        st.rerun()

# --- HIỂN THỊ KẾT QUẢ TỐI ƯU ---
if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    st.markdown(f"<div class='box'>🔥 ĐỘ PHỦ SẢNH (8 SỐ): <span class='big-num'>{res['top8']}</span></div>", unsafe_allow_html=True)
    
    # 2 TINH
    st.markdown("<div class='box'>🎯 2 TINH MATRIX (ƯU TIÊN BỆT & NHỊP RƠI)</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for i, (p, score, gan, streak) in enumerate(res['pairs']):
        with [c1, c2, c3][i]: 
            st.markdown(f"<div class='item'>{p[0]} , {p[1]}</div>", unsafe_allow_html=True)
            streak_html = f"<span class='streak-badge'>BỆT {streak}</span>" if streak > 0 else ""
            st.markdown(f"<div style='font-size:12px; margin-top:5px;'>Gan: {gan} kỳ {streak_html}</div>", unsafe_allow_html=True)
        
    # 3 TINH
    st.markdown("<div class='box' style='border-color:#FFD700;'>💎 3 TINH MATRIX (SIÊU CẤP CHUYÊN BIỆT)</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for i, (t, score, gan, streak) in enumerate(res['triples']):
        with [d1, d2, d3][i]: 
            st.markdown(f"<div class='item item-3'>{t[0]},{t[1]},{t[2]}</div>", unsafe_allow_html=True)
            streak_html = f"<span class='streak-badge'>BỆT {streak}</span>" if streak > 0 else ""
            st.markdown(f"<div style='font-size:12px; margin-top:5px;'>Gan: {gan} kỳ {streak_html}</div>", unsafe_allow_html=True)

# --- BẢNG ĐỐI SOÁT THỰC CHIẾN ---
if st.session_state.history:
    st.divider()
    st.subheader("📋 Nhật ký đối soát (Bao sảnh 5 Tinh)")
    df_history = pd.DataFrame(st.session_state.history).head(15)
    st.table(df_history)
