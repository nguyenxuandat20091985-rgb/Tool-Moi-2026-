"""
🚀 TITAN V27 AI - ULTIMATE v17.0.0
✅ Độ chính xác: Ensemble 5 thuật toán + AI
✅ Tốc độ: <30 giây cho 5D
✅ Format: 2-tinh: 3 cặp (2 số) | 3-tinh: 3 tổ hợp (3 số)
✅ Tuổi Sửu mệnh Kim + Bias detection + Markov chains
"""
import streamlit as st
import re
from collections import Counter, defaultdict, deque
from itertools import combinations
from openai import OpenAI
import google.generativeai as genai
import json
import time

# ⚡ API Keys
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

# 🐂 Tuổi Sửu - Mệnh Kim
LUCKY_OX = {0, 2, 5, 6, 7, 8}
METAL = {1, 6}
EARTH = {2, 5, 8}

st.set_page_config(page_title="TITAN V27 AI", page_icon="⚡", layout="centered")

# ⚡ CSS tối ưu tốc độ render
st.markdown("""
<style>
    .main {padding: 0.3rem;}
    .big-num {font-size: 40px; font-weight: 900; color: #FFD700; text-align: center; font-family: monospace; letter-spacing: 5px; margin: 3px 0;}
    .box {background: linear-gradient(135deg, #1a1a2e, #16213e); color: #FFD700; padding: 8px; border-radius: 8px; text-align: center; margin: 3px 0; border: 1px solid #FFD700;}
    .grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; margin: 8px 0;}
    .item {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; padding: 10px; border-radius: 6px; text-align: center; font-family: monospace; font-size: 26px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #C0C0C0, #808080); color: #000;}
    button {width: 100%; background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; font-size: 16px; font-weight: bold; padding: 10px; border: none; border-radius: 6px;}
    textarea {height: 70px; font-size: 13px;}
    .metric {display: flex; justify-content: space-around; margin: 3px 0; background: rgba(255,215,0,0.08); padding: 6px; border-radius: 6px;}
    .metric-val {font-size: 18px; font-weight: bold; color: #FFD700;}
    .metric-lbl {font-size: 10px; color: #AAA;}
    h1 {font-size: 22px; margin: 3px 0; color: #FFD700; text-align: center;}
    h2 {font-size: 16px; margin: 3px 0; color: #FFD700;}
    .alert {padding: 6px; border-radius: 6px; margin: 6px 0; font-size: 11px; border-left: 3px solid #FFD700; background: rgba(255,215,0,0.08);}
    .speed-tag {background: #00ff88; color: #000; padding: 2px 5px; border-radius: 3px; font-size: 9px; margin-left: 5px;}
</style>
""", unsafe_allow_html=True)

# ⚡ Hàm tối ưu tốc độ
def get_nums(text):
    return re.findall(r"\d{5}", text)

# 🎯 5 THUẬT TOÁN ENSEMBLE
def algo_frequency(db, window=30):
    """1. Frequency Analysis - Nhanh, cơ bản"""
    if not db: return {}
    freq = Counter("".join(db[-window:]))
    return {str(i): freq.get(str(i), 0) * 2 for i in range(10)}

def algo_position_bias(db):
    """2. Position Bias - Phát hiện số hay ra ở vị trí cụ thể"""
    if len(db) < 15: return {}
    pos_scores = defaultdict(lambda: Counter())
    for num in db[-30:]:
        for pos, d in enumerate(num[:5]):
            pos_scores[pos][d] += 1
    
    scores = {}
    for pos in range(5):
        total = sum(pos_scores[pos].values()) or 1
        for d, c in pos_scores[pos].most_common(2):
            if c / total > 0.18:
                scores[d] = scores.get(d, 0) + (c / total) * 25
    return scores

def algo_markov(db, order=2):
    """3. Markov Chain - Dự đoán dựa trên chuỗi trước đó"""
    if len(db) < 20: return {}
    chains = defaultdict(lambda: Counter())
    
    for num in db[-40:]:
        digits = list(num[:5])
        for i in range(len(digits) - order):
            key = tuple(digits[i:i+order])
            chains[key][digits[i+order]] += 1
    
    # Dự đoán từ kỳ cuối
    last = db[-1][:5] if db else ""
    scores = {}
    if len(last) >= order:
        key = tuple(last[-order:])
        if key in chains:
            total = sum(chains[key].values())
            for d, c in chains[key].most_common(3):
                scores[d] = scores.get(d, 0) + (c / total) * 30
    return scores

def algo_momentum(db, window=10):
    """4. Momentum/Reversion - Số đang lên/xuống"""
    if len(db) < window: return {}
    recent = db[-window:]
    scores = {}
    
    for d in range(10):
        ds = str(d)
        appearances = [i for i, num in enumerate(recent) if ds in num]
        if len(appearances) >= 2:
            # Momentum: xuất hiện ngày càng nhiều
            if appearances[-1] - appearances[-2] <= 3:
                scores[ds] = scores.get(ds, 0) + 15
        elif len(appearances) == 1 and appearances[0] >= window - 3:
            # Reversion: số gan vừa về
            scores[ds] = scores.get(ds, 0) + 20
    return scores

def algo_zodiac(db):
    """5. Zodiac Boost - Tuổi Sửu mệnh Kim"""
    scores = {}
    for d in range(10):
        if d in LUCKY_OX:
            scores[str(d)] = 10
        elif d in METAL | EARTH:
            scores[str(d)] = 6
    return scores

def ensemble_predict(db):
    """🎯 Ensemble: Kết hợp 5 thuật toán + weighting"""
    if len(db) < 10: return None
    
    start = time.time()
    
    # Chạy 5 thuật toán song song (thực tế tuần tự nhưng nhanh)
    scores = defaultdict(float)
    
    # Weight cho từng thuật toán
    weights = {
        "frequency": 1.0,
        "position": 1.3,
        "markov": 1.5,
        "momentum": 1.2,
        "zodiac": 0.8
    }
    
    for s in algo_frequency(db).items(): scores[s[0]] += s[1] * weights["frequency"]
    for s in algo_position_bias(db).items(): scores[s[0]] += s[1] * weights["position"]
    for s in algo_markov(db).items(): scores[s[0]] += s[1] * weights["markov"]
    for s in algo_momentum(db).items(): scores[s[0]] += s[1] * weights["momentum"]
    for s in algo_zodiac(db).items(): scores[s[0]] += s[1] * weights["zodiac"]
    
    # Bonus số vừa ra (3 kỳ gần)
    for num in db[-3:]:
        for d in set(num):
            scores[d] += 12
    
    # Top 8 số (không trùng)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_8 = []
    seen = set()
    for num, score in sorted_scores:
        if num not in seen and len(top_8) < 8:
            top_8.append(num)
            seen.add(num)
    
    # Tạo 2-tinh: 3 cặp từ 6 số đầu
    pool_2 = sorted(top_8[:6])
    pairs = ["".join(p) for p in combinations(pool_2, 2)]
    
    scored_pairs = []
    for pair in pairs:
        score = scores[pair[0]] + scores[pair[1]]
        # Bonus tandem
        for num in db[-12:]:
            if pair[0] in num and pair[1] in num:
                score += 28
                break
        scored_pairs.append((pair, score))
    
    scored_pairs.sort(key=lambda x: x[1], reverse=True)
    top_3_pairs = [p[0] for p in scored_pairs[:3]]
    
    # Tạo 3-tinh: 3 tổ hợp từ 6 số sau
    pool_3 = sorted(top_8[2:8])
    triples = ["".join(t) for t in combinations(pool_3, 3)]
    
    scored_triples = []
    for triple in triples:
        score = sum(scores[d] for d in triple)
        for num in db[-12:]:
            if all(d in num for d in triple):
                score += 38
                break
        scored_triples.append((triple, score))
    
    scored_triples.sort(key=lambda x: x[1], reverse=True)
    top_3_triples = [t[0] for t in scored_triples[:3]]
    
    # ⚡ AI nhanh (chỉ gọi nếu cần, timeout 5s)
    ai_reasoning = ""
    try:
        genai.configure(api_key=GEMINI_API_KEY, request_options={"timeout": 5000})
        gm_model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"5D: {len(db)} kỳ. Top8:{''.join(sorted(top_8))}. Pairs:{top_3_pairs}. Triples:{top_3_triples}. Tuổi Sửu. Đề xuất 1 câu <50 ký tự."
        res = gm_model.generate_content(prompt)
        ai_reasoning = res.text.strip()[:50]
    except:
        ai_reasoning = "Ensemble 5 thuật toán + Sửu"
    
    elapsed = time.time() - start
    
    # Confidence dựa trên độ đồng thuận của ensemble
    top_scores = [s for _, s in sorted_scores[:8]]
    consensus = 1 - (max(top_scores) - min(top_scores)) / (max(top_scores) + 1)
    conf = min(95, int(65 + consensus * 25 + len(db)//8))
    
    return {
        "all_8": "".join(sorted(top_8)),
        "pairs": top_3_pairs,
        "triples": top_3_triples,
        "conf": conf,
        "ai_reasoning": ai_reasoning,
        "time": f"{elapsed:.2f}s"
    }

# Init session
if "db" not in st.session_state: st.session_state.db = []
if "result" not in st.session_state: st.session_state.result = None

# UI
st.markdown('<h1>⚡ TITAN V27 AI - ULTIMATE</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#AAA;font-size:11px;">Sửu•Kim•Ensemble 5 algo•<30s</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div class="metric"><div class="metric-val">{len(st.session_state.db)}</div><div class="metric-lbl">Kỳ</div></div>', unsafe_allow_html=True)
with col2:
    last = st.session_state.db[-1] if st.session_state.db else "-"
    st.markdown(f'<div class="metric"><div class="metric-val" style="font-size:14px;">{last}</div><div class="metric-lbl">Cuối</div></div>', unsafe_allow_html=True)

user_input = st.text_area("📥 Dán 25-40 kỳ:", placeholder="16923\n51475\n...", height=70)

if st.button("⚡ CHỐT NHANH"):
    numbers = get_nums(user_input)
    if numbers:
        st.session_state.db = numbers[-40:]  # Tối ưu: chỉ giữ 40 kỳ
        with st.spinner("⚡ Tính..."):
            st.session_state.result = ensemble_predict(st.session_state.db)
            st.rerun()
    else:
        st.error("❌ Không có số!")

if st.session_state.result:
    r = st.session_state.result
    
    st.markdown(f'<div class="box">🔥 {r["conf"]}% <span class="speed-tag">{r["time"]}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;"><div style="color:#AAA;font-size:11px;">8 SỐ</div><div class="big-num">{r["all_8"]}</div></div>', unsafe_allow_html=True)
    
    if r.get("ai_reasoning"):
        st.markdown(f'<div style="background:rgba(255,215,0,0.08);padding:6px;border-radius:6px;margin:6px 0;font-size:11px;"><b>🤖</b> {r["ai_reasoning"]}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="box"><h2>🎯 2 TINH</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for pair in r["pairs"]:
        st.markdown(f'<div class="item">{pair}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="box"><h2>🎯 3 TINH</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for triple in r["triples"]:
        st.markdown(f'<div class="item item-3">{triple}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🗑️ XÓA"):
    st.session_state.db = []
    st.session_state.result = None
    st.rerun()

st.markdown('<div style="text-align:center;color:#666;font-size:9px;">⚡ v17.0.0</div>', unsafe_allow_html=True)