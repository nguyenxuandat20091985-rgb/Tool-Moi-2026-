"""
🚀 TITAN V27 AI - ULTIMATE VERSION
Dành cho tuổi Sửu (1985, 1997, 2009)
Mệnh: Kim - Hợp màu Trắng, Vàng, Nâu
Thuật toán: Cầu bệt + Cầu lừa + AI kép
Version: 14.0.0-ULTIMATE
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

# Tuổi Sửu - Mệnh Kim
LUCKY_NUMBERS_OX = [0, 2, 5, 6, 7, 8]  # Số hợp tuổi Sửu
METAL_ELEMENTS = [1, 6]  # Kim: 1, 6
EARTH_ELEMENTS = [2, 5, 8]  # Thổ sinh Kim: 2, 5, 8

st.set_page_config(page_title="TITAN V27 AI - Tuổi Sửu", page_icon="🐂", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    .main {padding: 0.5rem; font-family: 'Inter', sans-serif;}
    .big-num {font-size: 48px; font-weight: 900; background: linear-gradient(135deg, #FFD700, #FFA500); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; font-family: monospace; letter-spacing: 8px; margin: 5px 0;}
    .box {background: linear-gradient(135deg, #2F4F4F, #1C3A3A); color: #FFD700; padding: 12px; border-radius: 12px; text-align: center; margin: 8px 0; border: 2px solid #FFD700;}
    .grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin: 10px 0;}
    .item {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; padding: 15px; border-radius: 10px; text-align: center; font-family: monospace; font-size: 28px; font-weight: bold; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .item-3 {background: linear-gradient(135deg, #C0C0C0, #808080); color: #000;}
    button {width: 100%; background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; font-size: 18px; font-weight: bold; padding: 15px; border: none; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    button:hover {transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.4);}
    textarea {height: 80px; font-size: 14px; border: 2px solid #FFD700; border-radius: 8px;}
    .metric {display: flex; justify-content: space-around; margin: 5px 0; background: rgba(255,215,0,0.1); padding: 8px; border-radius: 8px;}
    .metric-val {font-size: 22px; font-weight: bold; color: #FFD700;}
    .metric-lbl {font-size: 11px; color: #C0C0C0;}
    h1 {font-size: 26px; margin: 5px 0; color: #FFD700; text-align: center;}
    h2 {font-size: 18px; margin: 5px 0; color: #FFD700;}
    .alert {padding: 10px; border-radius: 8px; margin: 8px 0; font-size: 13px; border-left: 4px solid #FFD700;}
    .alert-success {background: rgba(0,255,0,0.1); border-color: #00FF00;}
    .alert-warning {background: rgba(255,215,0,0.1); border-color: #FFD700;}
    .pattern-box {background: rgba(255,215,0,0.05); border: 1px solid #FFD700; padding: 8px; border-radius: 8px; margin: 5px 0;}
</style>
""", unsafe_allow_html=True)

def get_nums(text):
    return re.findall(r"\d{5}", text) if text else []

def detect_cau_bet(db, num_recent=10):
    """Phát hiện cầu bệt - số ra liên tiếp"""
    if len(db) < num_recent:
        return {}
    
    recent = db[-num_recent:]
    cau_bet = defaultdict(int)
    
    for pos in range(5):
        digits_at_pos = [num[pos] for num in recent]
        current_digit = digits_at_pos[0]
        count = 1
        
        for i in range(1, len(digits_at_pos)):
            if digits_at_pos[i] == current_digit:
                count += 1
            else:
                if count >= 2:
                    cau_bet[current_digit] += count
                current_digit = digits_at_pos[i]
                count = 1
        
        if count >= 2:
            cau_bet[current_digit] += count
    
    return dict(cau_bet)

def detect_cau_lua(db, num_recent=15):
    """Phát hiện cầu lừa - pattern ra-nghỉ-ra"""
    if len(db) < num_recent:
        return []
    
    recent = db[-num_recent:]
    cau_lua = []
    
    for digit in range(10):
        ds = str(digit)
        pattern_count = 0
        
        for i in range(len(recent) - 2):
            has_i = ds in recent[i]
            has_i1 = ds in recent[i+1] if i+1 < len(recent) else False
            has_i2 = ds in recent[i+2] if i+2 < len(recent) else False
            
            if has_i and not has_i1 and has_i2:
                pattern_count += 1
        
        if pattern_count >= 2:
            cau_lua.append(ds)
    
    return cau_lua

def detect_gan(db, num_recent=20):
    """Phát hiện số gan"""
    if not db:
        return {}
    
    so_gan = {}
    for digit in range(10):
        ds = str(digit)
        gan_count = 0
        
        for num in reversed(db[-num_recent:]):
            if ds not in num:
                gan_count += 1
            else:
                break
        
        if gan_count >= 5:
            so_gan[ds] = gan_count
    
    return so_gan

def ox_zodiac_boost(scores):
    """Tăng điểm số hợp tuổi Sửu mệnh Kim"""
    boosted = scores.copy()
    
    # Bonus số hợp tuổi Sửu
    for num in LUCKY_NUMBERS_OX:
        ds = str(num)
        if ds in boosted:
            boosted[ds] += 8
    
    # Bonus Kim và Thổ (sinh Kim)
    for num in METAL_ELEMENTS + EARTH_ELEMENTS:
        ds = str(num)
        if ds in boosted:
            boosted[ds] += 5
    
    return boosted

def statistical_analysis(db):
    """AI 1: Statistical Analysis với tuổi Sửu"""
    if len(db) < 10:
        return None
    
    all_digits = "".join(db[-30:])
    scores = {str(i): all_digits.count(str(i)) * 2 for i in range(10)}
    
    # Bonus số vừa ra
    for num in db[-3:]:
        for d in set(num):
            scores[d] += 10
    
    # Phát hiện pattern
    cau_bet = detect_cau_bet(db)
    cau_lua = detect_cau_lua(db)
    so_gan = detect_gan(db)
    
    # Bonus cầu bệt
    for digit, count in cau_bet.items():
        scores[digit] += count * 3
    
    # Bonus cầu lừa
    for digit in cau_lua:
        scores[digit] += 15
    
    # Bonus số gan (sắp về)
    for digit, gan_count in so_gan.items():
        scores[digit] += gan_count * 2
    
    # Tuổi Sửu boost
    scores = ox_zodiac_boost(scores)
    
    # Top 8 số
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_8 = [x[0] for x in sorted_scores[:8]]
    
    # 2 TINH
    pool_2 = sorted(top_8[:6])
    all_pairs = ["".join(p) for p in combinations(pool_2, 2)]
    
    scored_pairs = []
    for pair in all_pairs:
        score = scores[pair[0]] + scores[pair[1]]
        for num in db[-10:]:
            if pair[0] in num and pair[1] in num:
                score += 25
                break
        scored_pairs.append((pair, score))
    
    scored_pairs.sort(key=lambda x: x[1], reverse=True)
    top_3_pairs = [p[0] for p in scored_pairs[:3]]
    
    # 3 TINH
    pool_3 = sorted(top_8[2:8])
    all_triples = ["".join(t) for t in combinations(pool_3, 3)]
    
    scored_triples = []
    for triple in all_triples:
        score = sum(scores[d] for d in triple)
        for num in db[-10:]:
            if all(d in num for d in triple):
                score += 35
                break
        scored_triples.append((triple, score))
    
    scored_triples.sort(key=lambda x: x[1], reverse=True)
    top_3_triples = [t[0] for t in scored_triples[:3]]
    
    return {
        "all_8": "".join(sorted(top_8)),
        "pairs": top_3_pairs,
        "triples": top_3_triples,
        "scores": scores,
        "cau_bet": cau_bet,
        "cau_lua": cau_lua,
        "so_gan": so_gan
    }

def ai_enhance(db, stat_result):
    """AI 2: NVIDIA/Gemini LLM"""
    try:
        nv_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
        genai.configure(api_key=GEMINI_API_KEY)
        gm_model = genai.GenerativeModel('gemini-1.5-flash')
        
        recent = db[-25:] if len(db) >= 25 else db
        
        prompt = f"""
Phân tích xổ số 5D cho người tuổi Sửu mệnh Kim.

{len(recent)} kỳ gần nhất:
{recent}

Thống kê:
- 8 số mạnh: {stat_result['all_8']}
- Cầu bệt: {stat_result['cau_bet']}
- Cầu lừa: {stat_result['cau_lua']}
- Số gan: {stat_result['so_gan']}

YÊU CẦU: Chọn 6 số (3 cho 2-tinh, 3 cho 3-tinh, KHÔNG trùng).
Ưu tiên số hợp tuổi Sửu: 0,2,5,6,7,8 và mệnh Kim: 1,6

JSON format:
{{
  "two_digit_3nums": ["x","y","z"],
  "three_digit_3nums": ["a","b","c"],
  "confidence": 80,
  "reasoning": "Lý do ngắn gọn"
}}

Lưu ý:
- 2-tinh: 3 cặp từ 3 số
- 3-tinh: 3 tổ hợp từ 3 số  
- 6 số khác nhau
- Confidence 70-90%
"""
        
        try:
            completion = nv_client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            ai_result = json.loads(completion.choices[0].message.content)
        except:
            res = gm_model.generate_content(prompt)
            json_match = re.search(r'\{[\s\S]*\}', res.text)
            ai_result = json.loads(json_match.group()) if json_match else None
        
        if ai_result:
            if "two_digit_3nums" in ai_result and len(ai_result["two_digit_3nums"]) >= 2:
                nums = ai_result["two_digit_3nums"][:3]
                ai_pairs = ["".join(p) for p in combinations(sorted(nums), 2)][:3]
            else:
                ai_pairs = stat_result["pairs"]
            
            if "three_digit_3nums" in ai_result and len(ai_result["three_digit_3nums"]) >= 3:
                nums = ai_result["three_digit_3nums"][:3]
                ai_triples = ["".join(sorted(nums))]
                if len(ai_triples) < 3:
                    other_nums = [n for n in stat_result["all_8"] if n not in nums][:2]
                    for i, num in enumerate(other_nums):
                        if i < 2:
                            triple_nums = sorted(nums[:2] + [num])
                            ai_triples.append("".join(triple_nums))
            else:
                ai_triples = stat_result["triples"]
            
            return {
                "pairs": ai_pairs,
                "triples": ai_triples[:3],
                "confidence": ai_result.get("confidence", 80),
                "reasoning": ai_result.get("reasoning", "AI analysis")
            }
        
        return None
    
    except Exception as e:
        return None

if "db" not in st.session_state:
    st.session_state.db = []
if "result" not in st.session_state:
    st.session_state.result = None

# UI
st.markdown('<h1>🐂 TITAN V27 AI - TUỔI SỬU</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#C0C0C0;font-size:12px;">Mệnh Kim • Hợp: Trắng, Vàng, Nâu</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div class="metric"><div class="metric-val">{len(st.session_state.db)}</div><div class="metric-lbl">Tổng kỳ</div></div>', unsafe_allow_html=True)
with col2:
    last = st.session_state.db[-1] if st.session_state.db else "-"
    st.markdown(f'<div class="metric"><div class="metric-val" style="font-size:16px;">{last}</div><div class="metric-lbl">Kỳ cuối</div></div>', unsafe_allow_html=True)

user_input = st.text_area("📥 Dán 25-30 kỳ gần nhất:", placeholder="3280231\n3280230\n3280229\n...")

if st.button("⚡ AI PHÂN TÍCH"):
    numbers = get_nums(user_input)
    if numbers:
        st.session_state.db = numbers[-30:]
        
        with st.spinner("🤖 AI đang phân tích..."):
            stat_result = statistical_analysis(st.session_state.db)
            
            if stat_result:
                ai_result = ai_enhance(st.session_state.db, stat_result)
                
                if ai_result:
                    final = {
                        "all_8": stat_result["all_8"],
                        "pairs": ai_result["pairs"],
                        "triples": ai_result["triples"],
                        "conf": ai_result["confidence"],
                        "ai_reasoning": ai_result["reasoning"],
                        "cau_bet": stat_result["cau_bet"],
                        "cau_lua": stat_result["cau_lua"],
                        "so_gan": stat_result["so_gan"],
                        "using_ai": True
                    }
                else:
                    final = {
                        "all_8": stat_result["all_8"],
                        "pairs": stat_result["pairs"],
                        "triples": stat_result["triples"],
                        "conf": 80,
                        "ai_reasoning": "Statistical analysis + Tuổi Sửu",
                        "cau_bet": stat_result["cau_bet"],
                        "cau_lua": stat_result["cau_lua"],
                        "so_gan": stat_result["so_gan"],
                        "using_ai": False
                    }
                
                st.session_state.result = final
                st.rerun()
            else:
                st.error("❌ Cần ít nhất 10 kỳ")
    else:
        st.error("❌ Không có số!")

if st.session_state.result:
    r = st.session_state.result
    
    # Pattern alerts
    if r.get("cau_bet"):
        st.markdown(f'<div class="alert alert-success">✅ <b>CẦU BỆT:</b> Số {", ".join(r["cau_bet"].keys())} đang bệt!</div>', unsafe_allow_html=True)
    
    if r.get("cau_lua"):
        st.markdown(f'<div class="alert alert-warning">⚠️ <b>CẦU LỪA:</b> Số {", ".join(r["cau_lua"])} sắp về!</div>', unsafe_allow_html=True)
    
    if r.get("so_gan"):
        gan_str = ", ".join([f"{k}({v})" for k,v in r["so_gan"].items()])
        st.markdown(f'<div class="alert" style="background:rgba(255,0,0,0.1);border-color:#FF0000;">🔥 <b>SỐ GAN:</b> {gan_str}</div>', unsafe_allow_html=True)
    
    # Main result
    ai_badge = "🤖 AI KÉP" if r.get("using_ai") else "📊 Stats"
    st.markdown(f'<div class="box">🔥 ĐỘ TIN CẬY: {r["conf"]}% | {ai_badge}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;"><div style="color:#C0C0C0;font-size:12px;">8 SỐ MẠNH</div><div class="big-num">{r["all_8"]}</div></div>', unsafe_allow_html=True)
    
    if "ai_reasoning" in r:
        st.markdown(f'<div class="pattern-box">🧠 <b>AI:</b> {r["ai_reasoning"]}</div>', unsafe_allow_html=True)
    
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
    
    # Strategy
    st.markdown("""
    <div class="pattern-box">
    <b>💡 Chiến lược cho tuổi Sửu:</b><br>
    • Ưu tiên số: 0,2,5,6,7,8 (hợp tuổi)<br>
    • Mệnh Kim: 1,6 (Kim) + 2,5,8 (Thổ sinh Kim)<br>
    • Đánh 2 tinh trước → Trượt đánh 3 tinh<br>
    • Vốn chia 6 phần đều nhau
    </div>
    """, unsafe_allow_html=True)

if st.button("🗑️ XÓA"):
    st.session_state.db = []
    st.session_state.result = None
    st.rerun()

st.markdown('<div style="text-align:center;color:#666;font-size:10px;margin-top:10px;">🐂 TITAN V27 AI - Tuổi Sửu mệnh Kim</div>', unsafe_allow_html=True)