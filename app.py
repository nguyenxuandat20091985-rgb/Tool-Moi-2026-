"""
🚀 TITAN V27 AI - BIAS DETECTOR VERSION
Phát hiện bias RNG của nhà cái + Tuổi Sửu mệnh Kim
2 tinh: 3 cặp (2 chữ số) ✅
3 tinh: 3 tổ hợp (3 chữ số) ✅
Version: 16.0.0-BIAS
"""
import streamlit as st
import re
from collections import Counter, defaultdict
from itertools import combinations
from openai import OpenAI
import google.generativeai as genai
import json
import math

# API Keys
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

# Tuổi Sửu - Mệnh Kim: Lucky numbers
LUCKY_OX = [0, 2, 5, 6, 7, 8]
METAL_NUMS = [1, 6]  # Kim
EARTH_NUMS = [2, 5, 8]  # Thổ sinh Kim

st.set_page_config(page_title="TITAN V27 AI", page_icon="🐂", layout="centered")

st.markdown("""
<style>
    .main {padding: 0.5rem;}
    .big-num {font-size: 42px; font-weight: bold; color: #FFD700; text-align: center; font-family: monospace; letter-spacing: 6px; margin: 5px 0;}
    .box {background: linear-gradient(135deg, #2F4F4F, #1C3A3A); color: #FFD700; padding: 10px; border-radius: 10px; text-align: center; margin: 5px 0; border: 2px solid #FFD700;}
    .grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; margin: 10px 0;}
    .item {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; padding: 12px; border-radius: 8px; text-align: center; font-family: monospace; font-size: 28px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #C0C0C0, #808080); color: #000;}
    button {width: 100%; background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; font-size: 18px; font-weight: bold; padding: 12px; border: none; border-radius: 8px;}
    textarea {height: 80px; font-size: 14px;}
    .metric {display: flex; justify-content: space-around; margin: 5px 0; background: rgba(255,215,0,0.1); padding: 8px; border-radius: 8px;}
    .metric-val {font-size: 20px; font-weight: bold; color: #FFD700;}
    .metric-lbl {font-size: 11px; color: #C0C0C0;}
    h1 {font-size: 24px; margin: 5px 0; color: #FFD700; text-align: center;}
    h2 {font-size: 18px; margin: 5px 0; color: #FFD700;}
    .alert {padding: 8px; border-radius: 8px; margin: 8px 0; font-size: 12px; border-left: 4px solid #FFD700; background: rgba(255,215,0,0.1);}
    .bias-tag {background: #6f42c1; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-left: 5px;}
</style>
""", unsafe_allow_html=True)

def get_nums(text):
    return [n for n in re.findall(r"\d{5}", text) if n]

def analyze_position_bias(db, positions=5):
    """Phân tích bias theo từng vị trí (chục ngàn, ngàn, trăm, chục, đơn vị)"""
    if len(db) < 20:
        return {}
    
    position_freq = {pos: Counter() for pos in range(positions)}
    
    for num in db[-50:]:
        for pos in range(positions):
            if pos < len(num):
                position_freq[pos][num[pos]] += 1
    
    bias_result = {}
    for pos in range(positions):
        if position_freq[pos]:
            total = sum(position_freq[pos].values())
            # Tìm số có tần suất > 25% ở vị trí này (bias)
            for digit, count in position_freq[pos].most_common(3):
                ratio = count / total
                if ratio > 0.20:  # Bias threshold
                    bias_result[f"{pos}_{digit}"] = ratio
    
    return bias_result

def detect_repeating_patterns(db):
    """Phát hiện pattern lặp: số trùng, số kép, số liền"""
    if len(db) < 15:
        return {}
    
    patterns = defaultdict(int)
    
    for num in db[-30:]:
        # Số trùng (ví dụ: 00xxx, xx000)
        if num[0] == num[1]: patterns["repeat_first2"] += 1
        if num[-2] == num[-1]: patterns["repeat_last2"] += 1
        if len(set(num)) <= 2: patterns["low_entropy"] += 1
        
        # Số liền kề (012xx, x3456)
        for i in range(len(num)-2):
            if ord(num[i+1]) == ord(num[i])+1 and ord(num[i+2]) == ord(num[i])+2:
                patterns["sequential"] += 1
                break
    
    return dict(patterns)

def detect_cold_hot(db, window=30):
    """Phát hiện số nóng/lạnh với sliding window"""
    if len(db) < window:
        return {}, {}
    
    recent = db[-window:]
    all_digits = "".join(recent)
    freq = Counter(all_digits)
    
    hot = [d for d, c in freq.most_common(4)]
    cold = [str(i) for i in range(10) if str(i) not in freq or freq[str(i)] < 3]
    
    return {"hot": hot}, {"cold": cold}

def calculate_zodiac_boost(digit, zodiac="ox"):
    """Tính bonus điểm theo tuổi"""
    if zodiac == "ox":
        if int(digit) in LUCKY_OX:
            return 8
        if int(digit) in METAL_NUMS + EARTH_NUMS:
            return 5
    return 0

def predict_with_bias_detection(db):
    """Thuật toán chính: Phát hiện bias + thống kê + AI"""
    if len(db) < 15:
        return None
    
    # 1. Phát hiện bias vị trí
    pos_bias = analyze_position_bias(db)
    
    # 2. Phát hiện pattern lặp
    patterns = detect_repeating_patterns(db)
    
    # 3. Số nóng/lạnh
    hot_info, cold_info = detect_cold_hot(db)
    
    # 4. Tính điểm cơ bản
    all_digits = "".join(db[-40:])
    scores = {str(i): all_digits.count(str(i)) * 1.5 for i in range(10)}
    
    # 5. Bonus theo bias vị trí
    for key, ratio in pos_bias.items():
        pos, digit = key.split("_")
        scores[digit] += ratio * 30
    
    # 6. Bonus pattern
    if patterns.get("repeat_last2", 0) > 5:
        for num in db[-5:]:
            scores[num[-1]] += 10
            scores[num[-2]] += 10
    
    # 7. Bonus số nóng
    for digit in hot_info.get("hot", []):
        scores[digit] += 12
    
    # 8. Bonus số gan sắp về (cold nhưng có dấu hiệu)
    for digit in cold_info.get("cold", []):
        # Nếu số gan vừa xuất hiện 1 lần trong 3 kỳ gần
        if any(digit in num for num in db[-3:]):
            scores[digit] += 15
    
    # 9. Tuổi Sửu boost
    for d in range(10):
        scores[str(d)] += calculate_zodiac_boost(str(d))
    
    # 10. Chọn top 8 số (loại trùng)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_8 = []
    seen = set()
    for num, score in sorted_scores:
        if num not in seen and len(top_8) < 8:
            top_8.append(num)
            seen.add(num)
    
    # 11. Tạo 2-tinh: 3 cặp từ 6 số đầu
    pool_2 = sorted(top_8[:6])
    all_pairs = ["".join(p) for p in combinations(pool_2, 2)]
    
    scored_pairs = []
    for pair in all_pairs:
        score = scores[pair[0]] + scores[pair[1]]
        # Bonus nếu cùng xuất hiện trong 1 kỳ
        for num in db[-15:]:
            if pair[0] in num and pair[1] in num:
                score += 25
                break
        scored_pairs.append((pair, score))
    
    scored_pairs.sort(key=lambda x: x[1], reverse=True)
    top_3_pairs = [p[0] for p in scored_pairs[:3]]
    
    # 12. Tạo 3-tinh: 3 tổ hợp từ 6 số sau
    pool_3 = sorted(top_8[2:8])
    all_triples = ["".join(t) for t in combinations(pool_3, 3)]
    
    scored_triples = []
    for triple in all_triples:
        score = sum(scores[d] for d in triple)
        for num in db[-15:]:
            if all(d in num for d in triple):
                score += 35
                break
        scored_triples.append((triple, score))
    
    scored_triples.sort(key=lambda x: x[1], reverse=True)
    top_3_triples = [t[0] for t in scored_triples[:3]]
    
    # 13. AI Enhancement (NVIDIA + Gemini)
    ai_reasoning = ""
    try:
        nv_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
        genai.configure(api_key=GEMINI_API_KEY)
        gm_model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
PHÂN TÍCH 5D LOTTERY - BIAS DETECTION
Dữ liệu: {len(db)} kỳ gần nhất
Bias vị trí: {pos_bias}
Pattern: {patterns}
Số nóng: {hot_info.get("hot", [])}
Số lạnh: {cold_info.get("cold", [])}
8 số mạnh: {"".join(sorted(top_8))}
2-tinh đề xuất: {top_3_pairs}
3-tinh đề xuất: {top_3_triples}

Cho người tuổi Sửu mệnh Kim, hãy đề xuất chiến lược đánh ngắn gọn.
Trả về 1 câu duy nhất dưới 100 ký tự.
"""
        try:
            res = gm_model.generate_content(prompt)
            ai_reasoning = res.text.strip()[:100]
        except:
            ai_reasoning = "Bias analysis + Tuổi Sửu"
    except:
        ai_reasoning = "Statistical + Pattern analysis"
    
    # 14. Tính confidence
    bias_count = len(pos_bias) + len([p for p in patterns.values() if p > 3])
    conf = min(92, 60 + bias_count * 3 + len(db)//10)
    
    return {
        "all_8": "".join(sorted(top_8)),
        "pairs": top_3_pairs,
        "triples": top_3_triples,
        "conf": conf,
        "pos_bias": pos_bias,
        "patterns": patterns,
        "hot": hot_info.get("hot", []),
        "cold": cold_info.get("cold", []),
        "ai_reasoning": ai_reasoning
    }

if "db" not in st.session_state:
    st.session_state.db = []
if "result" not in st.session_state:
    st.session_state.result = None

# UI
st.markdown('<h1>🐂 TITAN V27 AI - BIAS DETECTOR</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#C0C0C0;font-size:12px;">Tuổi Sửu • Mệnh Kim • Bắt bias nhà cái</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div class="metric"><div class="metric-val">{len(st.session_state.db)}</div><div class="metric-lbl">Kỳ</div></div>', unsafe_allow_html=True)
with col2:
    last = st.session_state.db[-1] if st.session_state.db else "-"
    st.markdown(f'<div class="metric"><div class="metric-val" style="font-size:16px;">{last}</div><div class="metric-lbl">Cuối</div></div>', unsafe_allow_html=True)

user_input = st.text_area("📥 Dán 30-50 kỳ gần nhất:", placeholder="16923\n51475\n31410\n...", height=100)

if st.button("⚡ BẮT BIAS & CHỐT SỐ"):
    numbers = get_nums(user_input)
    if numbers:
        st.session_state.db = numbers[-50:]  # Giữ 50 kỳ để phân tích bias tốt hơn
        with st.spinner("🔍 AI đang quét bias..."):
            st.session_state.result = predict_with_bias_detection(st.session_state.db)
            st.rerun()
    else:
        st.error("❌ Không có số 5 chữ số!")

if st.session_state.result:
    r = st.session_state.result
    
    # Bias alerts
    if r.get("pos_bias"):
        biases = [f"{k.split('_')[1]}(vị trí {k.split('_')[0]})" for k in r["pos_bias"].keys()]
        st.markdown(f'<div class="alert" style="border-color:#00FF00;">✅ <b>BIAS VỊ TRÍ:</b> {", ".join(biases)}</div>', unsafe_allow_html=True)
    
    if r.get("patterns"):
        pats = [f"{k}:{v}" for k,v in r["patterns"].items() if v > 3]
        if pats:
            st.markdown(f'<div class="alert">🔄 <b>PATTERN:</b> {", ".join(pats)}</div>', unsafe_allow_html=True)
    
    # Main result
    st.markdown(f'<div class="box">🔥 ĐỘ TIN CẬY: {r["conf"]}% <span class="bias-tag">BIAS+AI</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;"><div style="color:#C0C0C0;font-size:12px;">8 SỐ MẠNH</div><div class="big-num">{r["all_8"]}</div></div>', unsafe_allow_html=True)
    
    if r.get("ai_reasoning"):
        st.markdown(f'<div style="background:rgba(255,215,0,0.1);padding:8px;border-radius:8px;margin:8px 0;font-size:12px;"><b>🤖 AI:</b> {r["ai_reasoning"]}</div>', unsafe_allow_html=True)
    
    # 2 TINH - 3 CẶP (2 chữ số)
    st.markdown('<div class="box"><h2>🎯 2 TINH - 3 CẶP</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for pair in r["pairs"]:
        st.markdown(f'<div class="item">{pair}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 3 TINH - 3 TỔ HỢP (3 chữ số)
    st.markdown('<div class="box"><h2>🎯 3 TINH - 3 TỔ HỢP</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for triple in r["triples"]:
        st.markdown(f'<div class="item item-3">{triple}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Strategy guide
    st.markdown("""
    <div style="background:rgba(255,215,0,0.05);padding:10px;border-radius:8px;margin:10px 0;font-size:12px;">
    <b>💡 Chiến lược:</b><br>
    • Số có bias vị trí → Ưu tiên đánh<br>
    • Pattern lặp → Theo cầu<br>
    • Số nóng + Tuổi Sửu → Kết hợp<br>
    • Chia vốn: 2-tinh 60%, 3-tinh 40%
    </div>
    """, unsafe_allow_html=True)

if st.button("🗑️ XÓA"):
    st.session_state.db = []
    st.session_state.result = None
    st.rerun()

st.markdown('<div style="text-align:center;color:#666;font-size:10px;">🐂 TITAN V27 AI - Bắt bias nhà cái</div>', unsafe_allow_html=True)