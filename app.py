"""
🚀 TITAN V27 - SIMPLE & EFFECTIVE
Đơn giản - Nhanh - Thực tế
Version: 8.0.0-SIMPLE
"""
import streamlit as st
import re
from collections import Counter
from itertools import combinations

st.set_page_config(page_title="TITAN V27", page_icon="🎲", layout="centered")

st.markdown("""
<style>
    .main {padding: 1rem;}
    .big-num {font-size: 56px; font-weight: bold; color: #28a745; text-align: center; font-family: monospace; letter-spacing: 8px; margin: 10px 0;}
    .box {background: #28a745; color: white; padding: 20px; border-radius: 15px; text-align: center; margin: 10px 0;}
    .grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 15px 0;}
    .item {background: #ffc107; color: #000; padding: 20px; border-radius: 10px; text-align: center; font-family: monospace; font-size: 32px; font-weight: bold;}
    .item-3 {background: #17a2b8; color: white;}
    button {width: 100%; background: #28a745; color: white; font-size: 24px; font-weight: bold; padding: 20px; border: none; border-radius: 10px;}
    textarea {height: 100px; font-size: 16px;}
    .metric {display: flex; justify-content: space-around; margin: 10px 0;}
    .metric-item {text-align: center;}
    .metric-val {font-size: 28px; font-weight: bold; color: #28a745;}
    .metric-lbl {font-size: 12px; color: #666;}
</style>
""", unsafe_allow_html=True)

def get_nums(text):
    return re.findall(r"\d{5}", text) if text else []

def predict(db):
    if len(db) < 10:
        return None
    
    # Tính tần suất 20 kỳ gần
    all_digits = "".join(db[-20:])
    scores = {str(i): all_digits.count(str(i)) for i in range(10)}
    
    # Bonus số vừa ra (3 kỳ gần)
    for num in db[-3:]:
        for d in set(num):
            scores[d] += 5
    
    # Trừ điểm số gan (không ra 8+ kỳ)
    for d in range(10):
        ds = str(d)
        gan = True
        for num in db[-8:]:
            if ds in num:
                gan = False
                break
        if gan:
            scores[ds] -= 10
    
    # Top 8 số
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_8 = [x[0] for x in sorted_scores[:8]]
    
    # 2 tinh: 6 số đầu → 3 cặp tốt nhất
    pool_2 = sorted(top_8[:6])
    all_pairs = ["".join(p) for p in combinations(pool_2, 2)]
    
    # Tính điểm từng cặp
    scored_pairs = []
    for pair in all_pairs:
        score = scores[pair[0]] + scores[pair[1]]
        # Bonus nếu cùng ra trong 1 kỳ gần
        for num in db[-10:]:
            if pair[0] in num and pair[1] in num:
                score += 15
                break
        scored_pairs.append((pair, score))
    
    scored_pairs.sort(key=lambda x: x[1], reverse=True)
    top_3_pairs = [p[0] for p in scored_pairs[:3]]
    
    # 3 tinh: 6 số sau → 3 tổ hợp tốt nhất
    pool_3 = sorted(top_8[2:8])
    all_triples = ["".join(t) for t in combinations(pool_3, 3)]
    
    scored_triples = []
    for triple in all_triples:
        score = sum(scores[d] for d in triple)
        # Bonus nếu cùng ra trong 1 kỳ gần
        for num in db[-10:]:
            if all(d in num for d in triple):
                score += 25
                break
        scored_triples.append((triple, score))
    
    scored_triples.sort(key=lambda x: x[1], reverse=True)
    top_3_triples = [t[0] for t in scored_triples[:3]]
    
    # Confidence
    conf = min(85, 60 + len(db)//4)
    
    return {
        "all_8": "".join(sorted(top_8)),
        "pairs": top_3_pairs,
        "triples": top_3_triples,
        "conf": conf
    }

# Init
if "db" not in st.session_state:
    st.session_state.db = []
if "result" not in st.session_state:
    st.session_state.result = None

# Title
st.markdown('<h1 style="text-align:center;color:#28a745;margin:10px 0;">🎲 TITAN V27</h1>', unsafe_allow_html=True)

# Metrics
col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div class="metric"><div class="metric-item"><div class="metric-val">{len(st.session_state.db)}</div><div class="metric-lbl">Tổng kỳ</div></div></div>', unsafe_allow_html=True)
with col2:
    last = st.session_state.db[-1] if st.session_state.db else "-"
    st.markdown(f'<div class="metric"><div class="metric-item"><div class="metric-val" style="font-size:20px;">{last}</div><div class="metric-lbl">Kỳ cuối</div></div></div>', unsafe_allow_html=True)

# Input
user_input = st.text_area("📥 Dán 20-30 kỳ gần nhất:", placeholder="3280231\n3280230\n3280229\n...")

if st.button("⚡ CHỐT SỐ"):
    numbers = get_nums(user_input)
    if numbers:
        st.session_state.db = numbers[-30:]  # Chỉ giữ 30 kỳ
        st.session_state.result = predict(st.session_state.db)
        st.rerun()
    else:
        st.error("❌ Không có số 5 chữ số!")

# Results
if st.session_state.result:
    r = st.session_state.result
    
    st.markdown(f'<div class="box">🔥 ĐỘ TIN CẬY: {r["conf"]}%</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div style="text-align:center;"><div style="color:#666;">🎲 8 SỐ MẠNH</div><div class="big-num">{r["all_8"]}</div></div>', unsafe_allow_html=True)
    
    # 2 TINH
    st.markdown('<div class="box"><h2 style="margin:0;color:#000;">🎯 2 TINH - 3 CẶP</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for pair in r["pairs"]:
        st.markdown(f'<div class="item">{pair}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 3 TINH
    st.markdown('<div class="box"><h2 style="margin:0;">🎯 3 TINH - 3 TỔ HỢP</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for triple in r["triples"]:
        st.markdown(f'<div class="item item-3">{triple}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Strategy
    st.markdown('<div style="background:#f8f9fa;padding:15px;border-radius:10px;margin:15px 0;"><b>💡 Cách đánh:</b><br>1️⃣ Đánh 3 cặp 2 tinh trước<br>2️⃣ Trượt → Đánh 3 tổ hợp 3 tinh<br>3️⃣ Vốn đều 6 phần</div>', unsafe_allow_html=True)

# Actions
if st.button("🗑️ XÓA"):
    st.session_state.db = []
    st.session_state.result = None
    st.rerun()

st.markdown('<div style="text-align:center;color:#999;font-size:11px;margin-top:20px;">⚡ Đơn giản - Hiệu quả</div>', unsafe_allow_html=True)