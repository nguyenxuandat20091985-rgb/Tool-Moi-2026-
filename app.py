import streamlit as st
import re, json, os, pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
from openai import OpenAI
import google.generativeai as genai

# --- CẤU HÌNH HỆ THỐNG ---
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

# Tuổi Sửu 1985 - Mệnh Kim
LUCKY_OX = [0, 2, 5, 6, 7, 8]
SHADOW_MAP = {"0":"5", "5":"0", "1":"6", "6":"1", "2":"7", "7":"2", "3":"8", "8":"3", "4":"9", "9":"4"}

st.set_page_config(page_title="TITAN V34 PRO", page_icon="🐂", layout="centered")

# --- GIAO DIỆN DARK GOLD ---
st.markdown("""
<style>
    .stApp {background-color: #050505; color: #FFD700;}
    .big-num {font-size: 42px; font-weight: bold; color: #FFD700; text-align: center; font-family: monospace; letter-spacing: 6px;}
    .box {background: linear-gradient(135deg, #1C3A3A, #050505); color: #FFD700; padding: 15px; border-radius: 12px; text-align: center; border: 2px solid #FFD700; margin-bottom: 10px;}
    .item {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; padding: 12px; border-radius: 8px; text-align: center; font-size: 26px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #C0C0C0, #808080); color: #000;}
    .alert-cycle {background: rgba(255, 49, 49, 0.1); border-left: 5px solid #FF3131; padding: 10px; color: #FF3131; font-weight: bold; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# --- THUẬT TOÁN ĐẮP THÊM (V27 + V33) ---
def get_nums(text):
    return [n for n in re.findall(r"\d{5}", text) if n]

def analyze_bias_v34(db):
    """Phân tích bias vị trí từ bản V27 gốc"""
    if len(db) < 20: return {}
    pos_freq = {i: Counter() for i in range(5)}
    for num in db[-50:]:
        for i in range(5): pos_freq[i][num[i]] += 1
    
    biases = {}
    for i in range(5):
        digit, count = pos_freq[i].most_common(1)[0]
        if count / len(db[-50:]) > 0.22: # Threshold 22%
            biases[i] = digit
    return biases

def detect_cycles(db):
    """Bắt chu trình số gánh và bệt bộ của nhà cái"""
    if len(db) < 5: return []
    alerts = []
    last_5 = db[-5:]
    # Soi số gánh (Ví dụ: 97878)
    for n in last_5:
        if n[0] == n[2] or n[1] == n[3] or n[2] == n[4]:
            alerts.append(f"Cảnh báo: Chu trình SỐ GÁNH ({n})")
            break
    # Soi bệt bộ/đuôi
    tails = [n[-2:] for n in last_5]
    if len(set(tails)) <= 3:
        alerts.append("Cảnh báo: Chu trình BỆT BỘ ĐUÔI")
    return alerts

def predict_v34(db):
    if len(db) < 10: return None
    
    # 1. Điểm tần suất nền
    all_digits = "".join(db[-50:])
    scores = {str(i): all_digits.count(str(i)) * 1.3 for i in range(10)}
    
    # 2. ĐỐI CHIẾU BIAS VỊ TRÍ (V27)
    biases = analyze_bias_v34(db)
    for pos, digit in biases.items():
        scores[digit] += 15
    
    # 3. CẦU RƠI & BÓNG (V32)
    last_num = db[-1]
    for d in set(last_num):
        scores[d] += 25 # Ưu tiên cực cao cầu bệt
        shadow = SHADOW_MAP.get(d)
        if shadow: scores[shadow] += 12 # Soi bóng mệnh Kim
        
    # 4. TUỔI SỬU BOOST
    for d in LUCKY_OX: scores[str(d)] += 10

    # Sắp xếp lấy Top
    sorted_digits = [x[0] for x in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    top_8 = "".join(sorted_digits[:8])
    
    # Ghép 2 Tinh (3 cặp từ Top 5)
    pairs = ["".join(p) for p in combinations(sorted_digits[:5], 2)][:3]
    # Ghép 3 Tinh (3 bộ từ Top 6)
    triples = ["".join(t) for t in combinations(sorted_digits[:6], 3)][:3]
    
    # AI Reasoning
    ai_msg = "Phát hiện chu trình đảo nhịp. Đang ưu tiên bộ số Bóng Ngũ Hành."
    
    return {
        "pairs": pairs, 
        "triples": triples, 
        "top8": top_8, 
        "biases": biases, 
        "alerts": detect_cycles(db),
        "ai": ai_msg
    }

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>🐂 TITAN V34 - CYCLE MASTER</h1>', unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán bảng kết quả (Kỳ mới nhất ở dưới cùng):", height=150)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 QUÉT CHU TRÌNH & CHỐT"):
        nums = get_nums(user_input)
        if nums:
            # Đối soát thắng thua
            if "last_pred" in st.session_state:
                lp = st.session_state.last_pred
                is_win = any(set(p).issubset(set(nums[-1])) for p in lp['pairs'])
                st.session_state.history.insert(0, {"Kỳ": nums[-1], "Dự đoán": lp['pairs'][0], "KQ": "🔥 WIN" if is_win else "❌"})
            
            st.session_state.last_pred = predict_v34(nums)
            st.rerun()
with col2:
    if st.button("🗑️ RESET"):
        st.session_state.clear()
        st.rerun()

# --- HIỂN THỊ KẾT QUẢ ---
if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    # Hiện cảnh báo chu trình
    for alert in res['alerts']:
        st.markdown(f"<div class='alert-cycle'>{alert}</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div class='box'>🎯 8 SỐ MẠNH: <span class='big-num'>{res['top8']}</span></div>", unsafe_allow_html=True)
    
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

    st.write(f"🤖 **AI:** {res['ai']}")

# --- LỊCH SỬ ---
if st.session_state.history:
    st.divider()
    st.table(pd.DataFrame(st.session_state.history).head(10))
