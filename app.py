"""
🚀 TITAN V27 AI - HARD FIX
Version: 13.0.0-HARDFIX
"""
import streamlit as st
import re
from collections import Counter
from itertools import combinations

st.set_page_config(page_title="TITAN V27 AI", page_icon="🤖", layout="centered")

st.markdown("""
<style>
    .main {padding: 0.5rem;}
    .big-num {font-size: 42px; font-weight: bold; color: #28a745; text-align: center; font-family: monospace; letter-spacing: 6px; margin: 5px 0;}
    .box {background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 10px; border-radius: 10px; text-align: center; margin: 5px 0;}
    .grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; margin: 10px 0;}
    .item {background: #ffc107; color: #000; padding: 12px; border-radius: 8px; text-align: center; font-family: monospace; font-size: 28px; font-weight: bold;}
    .item-3 {background: #17a2b8; color: white;}
    button {width: 100%; background: #28a745; color: white; font-size: 18px; font-weight: bold; padding: 12px; border: none; border-radius: 8px;}
    textarea {height: 80px; font-size: 14px;}
    .metric {display: flex; justify-content: space-around; margin: 5px 0;}
    .metric-val {font-size: 20px; font-weight: bold; color: #28a745;}
    .metric-lbl {font-size: 11px; color: #666;}
    h1 {font-size: 24px; margin: 5px 0;}
    h2 {font-size: 18px; margin: 5px 0;}
</style>
""", unsafe_allow_html=True)

def get_nums(text):
    return re.findall(r"\d{5}", text) if text else []

def predict(db):
    if len(db) < 10:
        return None
    
    # Tính điểm
    all_digits = "".join(db[-20:])
    scores = {str(i): all_digits.count(str(i)) for i in range(10)}
    
    for num in db[-3:]:
        for d in set(num):
            scores[d] += 5
    
    # Top 8 số - LOẠI BỎ TRÙNG
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_8 = []
    seen = set()
    for num, score in sorted_scores:
        if num not in seen and len(top_8) < 8:
            top_8.append(num)
            seen.add(num)
    
    # 2 TINH: Lấy 6 số ĐẦU TIÊN → Tạo cặp
    pool_2 = sorted(top_8[:6])
    
    # Tạo TẤT CẢ cặp 2 số
    all_pairs = []
    for i in range(len(pool_2)):
        for j in range(i+1, len(pool_2)):
            pair = pool_2[i] + pool_2[j]  # ✅ NỐI 2 SỐ: "0"+"9" = "09"
            all_pairs.append(pair)
    
    # Tính điểm và chọn TOP 3
    scored_pairs = []
    for pair in all_pairs:
        score = scores[pair[0]] + scores[pair[1]]
        for num in db[-10:]:
            if pair[0] in num and pair[1] in num:
                score += 20
                break
        scored_pairs.append((pair, score))
    
    scored_pairs.sort(key=lambda x: x[1], reverse=True)
    top_3_pairs = [p[0] for p in scored_pairs[:3]]  # ✅ ["09", "69", "79"]
    
    # 3 TINH: Lấy 6 số TIẾP THEO → Tạo tổ hợp 3 số
    pool_3 = sorted(top_8[2:8])
    
    # Tạo TẤT CẢ tổ hợp 3 số
    all_triples = []
    for i in range(len(pool_3)):
        for j in range(i+1, len(pool_3)):
            for k in range(j+1, len(pool_3)):
                triple = pool_3[i] + pool_3[j] + pool_3[k]  # ✅ NỐI 3 SỐ: "0"+"1"+"7" = "017"
                all_triples.append(triple)
    
    # Tính điểm và chọn TOP 3
    scored_triples = []
    for triple in all_triples:
        score = sum(scores[d] for d in triple)
        for num in db[-10:]:
            if all(d in num for d in triple):
                score += 30
                break
        scored_triples.append((triple, score))
    
    scored_triples.sort(key=lambda x: x[1], reverse=True)
    top_3_triples = [t[0] for t in scored_triples[:3]]  # ✅ ["017", "047", "147"]
    
    conf = min(85, 60 + len(db)//4)
    
    return {
        "all_8": "".join(sorted(top_8)),
        "pairs": top_3_pairs,       # ✅ 3 cặp 2 chữ số
        "triples": top_3_triples,   # ✅ 3 tổ hợp 3 chữ số
        "conf": conf
    }

if "db" not in st.session_state:
    st.session_state.db = []
if "result" not in st.session_state:
    st.session_state.result = None

st.markdown('<h1 style="text-align:center;color:#28a745;">🤖 TITAN V27 AI</h1>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div class="metric"><div class="metric-val">{len(st.session_state.db)}</div><div class="metric-lbl">Kỳ</div></div>', unsafe_allow_html=True)
with col2:
    last = st.session_state.db[-1] if st.session_state.db else "-"
    st.markdown(f'<div class="metric"><div class="metric-val" style="font-size:16px;">{last}</div><div class="metric-lbl">Cuối</div></div>', unsafe_allow_html=True)

user_input = st.text_area("📥 Dán kết quả:", placeholder="3280231\n3280230\n...")

if st.button("⚡ CHỐT SỐ"):
    numbers = get_nums(user_input)
    if numbers:
        st.session_state.db = numbers[-30:]
        st.session_state.result = predict(st.session_state.db)
        st.rerun()
    else:
        st.error("❌ Không có số!")

if st.session_state.result:
    r = st.session_state.result
    
    st.markdown(f'<div class="box">🔥 ĐỘ TIN CẬY: {r["conf"]}%</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;"><div style="color:#666;font-size:12px;">8 SỐ MẠNH</div><div class="big-num">{r["all_8"]}</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="box"><h2>🎯 2 TINH - 3 CẶP</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for pair in r["pairs"]:
        st.markdown(f'<div class="item">{pair}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="box"><h2>🎯 3 TINH - 3 TỔ HỢP</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for triple in r["triples"]:
        st.markdown(f'<div class="item item-3">{triple}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🗑️ XÓA"):
    st.session_state.db = []
    st.session_state.result = None
    st.rerun()

st.markdown('<div style="text-align:center;color:#999;font-size:10px;">TITAN V27 AI</div>', unsafe_allow_html=True)