"""
🐂 TITAN V27 AI - AGE OF OX EDITION
Dành cho tuổi Sửu - Hợp mệnh Thổ/Kim
Features:
- AI: NVIDIA Llama-3.1 + Gemini-1.5
- Pattern: Cầu bệt, cầu lừa, chu kỳ, tandem
- Stats: Frequency, hot/cold, gap analysis
Version: 14.0.0-COMPLETE
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

# Tuổi Sửu - Mệnh Thổ/Kim - Màu hợp: Vàng, Nâu, Xanh lá
STYLING = {
    "primary": "#28a745",    # Xanh lá (Mộc sinh Thổ)
    "secondary": "#ffc107",  # Vàng (Thổ)
    "accent": "#8B4513",     # Nâu (Thổ)
    "bg": "#FFFDD0",         # Kem (Thổ)
}

st.set_page_config(page_title="🐂 TITAN V27 - Tuổi Sửu", page_icon="🐂", layout="centered")

st.markdown(f"""
<style>
    .main {{padding: 0.5rem; background: {STYLING['bg']};}}
    .big-num {{font-size: 42px; font-weight: bold; color: {STYLING['primary']}; text-align: center; font-family: monospace; letter-spacing: 6px; margin: 5px 0;}}
    .box {{background: linear-gradient(135deg, {STYLING['primary']}, #20c997); color: white; padding: 10px; border-radius: 10px; text-align: center; margin: 5px 0;}}
    .grid {{display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; margin: 10px 0;}}
    .item {{background: {STYLING['secondary']}; color: #000; padding: 12px; border-radius: 8px; text-align: center; font-family: monospace; font-size: 28px; font-weight: bold;}}
    .item-3 {{background: #17a2b8; color: white;}}
    button {{width: 100%; background: {STYLING['primary']}; color: white; font-size: 18px; font-weight: bold; padding: 12px; border: none; border-radius: 8px;}}
    textarea {{height: 80px; font-size: 14px;}}
    .metric {{display: flex; justify-content: space-around; margin: 5px 0;}}
    .metric-val {{font-size: 20px; font-weight: bold; color: {STYLING['primary']};}}
    .metric-lbl {{font-size: 11px; color: #666;}}
    h1 {{font-size: 24px; margin: 5px 0; color: {STYLING['accent']};}}
    h2 {{font-size: 18px; margin: 5px 0;}}
    .pattern-box {{background: #fff3cd; border-left: 4px solid {STYLING['secondary']}; padding: 8px; margin: 5px 0; font-size: 12px;}}
    .ai-box {{background: #f3e5f5; border-left: 4px solid #6f42c1; padding: 8px; margin: 5px 0; font-size: 12px;}}
</style>
""", unsafe_allow_html=True)

def get_nums(text):
    return re.findall(r"\d{5}", text) if text else []

# ========== PATTERN DETECTION ==========
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

def detect_cycle(db, digit, max_cycle=12):
    """Phát hiện chu kỳ ra của 1 số"""
    if len(db) < 10:
        return None
    
    positions = []
    for i, num in enumerate(db):
        if str(digit) in num:
            positions.append(i)
    
    if len(positions) < 3:
        return None
    
    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    if not gaps:
        return None
    
    gap_counter = Counter(gaps)
    most_common = gap_counter.most_common(1)
    
    if most_common:
        cycle_length = most_common[0][0]
        frequency = most_common[0][1]
        
        if frequency >= 2 and cycle_length <= max_cycle:
            last_pos = positions[-1]
            next_expected = last_pos + cycle_length
            current_pos = len(db) - 1
            distance = next_expected - current_pos
            
            return {"cycle": cycle_length, "confidence": frequency/len(gaps), "next_in": distance}
    
    return None

def detect_tandem(db, num_recent=20):
    """Phát hiện cặp số hay đi cùng"""
    if len(db) < 5:
        return []
    
    recent = db[-num_recent:]
    tandem_count = defaultdict(int)
    
    for num in recent:
        digits = set(num)
        for d1 in digits:
            for d2 in digits:
                if d1 < d2:
                    tandem_count[(d1, d2)] += 1
    
    sorted_tandem = sorted(tandem_count.items(), key=lambda x: x[1], reverse=True)
    return [(p[0], p[1], c) for p, c in sorted_tandem[:10]]

# ========== STATISTICAL ANALYSIS ==========
def statistical_analysis(db):
    """AI 1: Statistical Analysis với pattern detection"""
    if len(db) < 10:
        return None
    
    # Patterns
    cau_bet = detect_cau_bet(db)
    cau_lua = detect_cau_lua(db)
    tandem = detect_tandem(db)
    
    # Base scores
    all_digits = "".join(db[-30:])
    scores = {str(i): all_digits.count(str(i)) * 2 for i in range(10)}
    
    # Bonus cầu bệt
    for digit, count in cau_bet.items():
        scores[digit] += count * 5
    
    # Bonus cầu lừa
    for digit in cau_lua:
        scores[digit] += 15
    
    # Bonus tandem
    for d1, d2, count in tandem[:5]:
        scores[d1] += count
        scores[d2] += count
    
    # Chu kỳ
    cycles = {}
    for digit in range(10):
        cycle_info = detect_cycle(db, digit)
        if cycle_info and cycle_info["next_in"] <= 3:
            scores[str(digit)] += 30 * cycle_info["confidence"]
            cycles[str(digit)] = cycle_info
    
    # Top 8 - Loại bỏ trùng
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_8 = []
    seen = set()
    for num, score in sorted_scores:
        if num not in seen and len(top_8) < 8:
            top_8.append(num)
            seen.add(num)
    
    # 2 TINH
    pool_2 = sorted(top_8[:6])
    all_pairs = ["".join(p) for p in combinations(pool_2, 2)]
    
    scored_pairs = []
    for pair in all_pairs:
        score = scores[pair[0]] + scores[pair[1]]
        # Bonus tandem
        for d1, d2, count in tandem:
            if (pair[0] == d1 and pair[1] == d2) or (pair[0] == d2 and pair[1] == d1):
                score += count * 3
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
        scored_triples.append((triple, score))
    
    scored_triples.sort(key=lambda x: x[1], reverse=True)
    top_3_triples = [t[0] for t in scored_triples[:3]]
    
    conf = min(90, 55 + len(db)//5 + len(cau_bet)*2)
    
    return {
        "all_8": "".join(sorted(top_8)),
        "pairs": top_3_pairs,
        "triples": top_3_triples,
        "scores": scores,
        "cau_bet": cau_bet,
        "cau_lua": cau_lua,
        "cycles": cycles,
        "tandem": tandem[:5],
        "conf": conf
    }

# ========== AI ENHANCEMENT ==========
def ai_enhance(db, stat_result):
    """AI 2: LLM Enhancement với NVIDIA/Gemini"""
    try:
        nv_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
        genai.configure(api_key=GEMINI_API_KEY)
        gm_model = genai.GenerativeModel('gemini-1.5-flash')
        
        recent = db[-25:] if len(db) >= 25 else db
        
        prompt = f"""
Phân tích xổ số 5D cho người tuổi Sửu (mệnh Thổ).

DỮ LIỆU {len(recent)} KỲ GẦN:
{recent}

THỐNG KÊ:
- 8 số mạnh: {stat_result['all_8']}
- Cầu bệt: {stat_result.get('cau_bet', {})}
- Cầu lừa: {stat_result.get('cau_lua', [])}
- Tandem: {stat_result.get('tandem', [])}
- Chu kỳ: {stat_result.get('cycles', {})}

YÊU CẦU:
Chọn 6 số TỐT NHẤT (3 cho 2-tinh, 3 cho 3-tinh, KHÔNG trùng).

TRẢ VỀ JSON:
{{
  "two_digit_3nums": ["x","y","z"],
  "three_digit_3nums": ["a","b","c"],
  "confidence": 75-90,
  "reasoning": "Giải thích ngắn"
}}

LƯU Ý:
- Ưu tiên số có cầu bệt, chu kỳ
- Tránh số gan (>8 kỳ)
- 2-tinh: 3 cặp từ 3 số
- 3-tinh: 3 tổ hợp từ 3 số
- Tuổi Sửu hợp: 0,2,5,6,8,9
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
            # Create pairs
            if "two_digit_3nums" in ai_result and len(ai_result["two_digit_3nums"]) >= 2:
                nums = ai_result["two_digit_3nums"][:3]
                ai_pairs = ["".join(p) for p in combinations(sorted(nums), 2)][:3]
            else:
                ai_pairs = stat_result["pairs"]
            
            # Create triples
            if "three_digit_3nums" in ai_result and len(ai_result["three_digit_3nums"]) >= 3:
                nums = ai_result["three_digit_3nums"][:3]
                ai_triples = ["".join(sorted(nums))]
                # Generate more
                other_nums = [n for n in stat_result["all_8"] if n not in nums][:2]
                for i, num in enumerate(other_nums):
                    if i < 2 and len(nums) >= 2:
                        triple_nums = sorted(nums[:2] + [num])
                        ai_triples.append("".join(triple_nums))
            else:
                ai_triples = stat_result["triples"]
            
            return {
                "pairs": ai_pairs,
                "triples": ai_triples[:3],
                "confidence": ai_result.get("confidence", 75),
                "reasoning": ai_result.get("reasoning", "AI analysis")
            }
        
        return None
    
    except Exception as e:
        return None

# ========== UI ==========
if "db" not in st.session_state:
    st.session_state.db = []
if "result" not in st.session_state:
    st.session_state.result = None

st.markdown('<h1 style="text-align:center;">🐂 TITAN V27 AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#666;font-size:12px;">Tuổi Sửu - Mệnh Thổ/Kim</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div class="metric"><div class="metric-val">{len(st.session_state.db)}</div><div class="metric-lbl">Kỳ</div></div>', unsafe_allow_html=True)
with col2:
    last = st.session_state.db[-1] if st.session_state.db else "-"
    st.markdown(f'<div class="metric"><div class="metric-val" style="font-size:16px;">{last}</div><div class="metric-lbl">Cuối</div></div>', unsafe_allow_html=True)

user_input = st.text_area("📥 Dán 20-30 kỳ:", placeholder="3280231\n3280230\n...")

if st.button("⚡ AI PHÂN TÍCH"):
    numbers = get_nums(user_input)
    if numbers:
        st.session_state.db = numbers[-30:]
        
        with st.spinner("🤖 AI đang phân tích..."):
            stat = statistical_analysis(st.session_state.db)
            
            if stat:
                ai = ai_enhance(st.session_state.db, stat)
                
                if ai:
                    final = {
                        "all_8": stat["all_8"],
                        "pairs": ai["pairs"],
                        "triples": ai["triples"],
                        "conf": ai["confidence"],
                        "reasoning": ai["reasoning"],
                        "cau_bet": stat.get("cau_bet", {}),
                        "cau_lua": stat.get("cau_lua", []),
                        "cycles": stat.get("cycles", {}),
                        "using_ai": True
                    }
                else:
                    final = {
                        "all_8": stat["all_8"],
                        "pairs": stat["pairs"],
                        "triples": stat["triples"],
                        "conf": stat["conf"],
                        "reasoning": "Statistical analysis",
                        "cau_bet": stat.get("cau_bet", {}),
                        "cau_lua": stat.get("cau_lua", []),
                        "cycles": stat.get("cycles", {}),
                        "using_ai": False
                    }
                
                st.session_state.result = final
                st.rerun()
            else:
                st.error("❌ Cần 10+ kỳ")
    else:
        st.error("❌ Không có số!")

if st.session_state.result:
    r = st.session_state.result
    
    ai_badge = "🤖 AI" if r.get("using_ai") else "📊 Stats"
    st.markdown(f'<div class="box">🔥 TIN CẬY: {r["conf"]}% {ai_badge}</div>', unsafe_allow_html=True)
    
    # Patterns
    if r.get("cau_bet"):
        st.markdown(f'<div class="pattern-box">✅ <b>Cầu bệt:</b> Số {", ".join(r["cau_bet"].keys())}</div>', unsafe_allow_html=True)
    if r.get("cau_lua"):
        st.markdown(f'<div class="pattern-box">⚠️ <b>Cầu lừa:</b> Số {", ".join(r["cau_lua"])}</div>', unsafe_allow_html=True)
    if r.get("cycles"):
        st.markdown(f'<div class="pattern-box">🔄 <b>Chu kỳ:</b> {len(r["cycles"])} số</div>', unsafe_allow_html=True)
    
    if "reasoning" in r:
        st.markdown(f'<div class="ai-box">🧠 <b>AI:</b> {r["reasoning"]}</div>', unsafe_allow_html=True)
    
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

st.markdown(f'<div style="text-align:center;color:#999;font-size:10px;">🐂 TITAN V27 AI - Tuổi Sửu</div>', unsafe_allow_html=True)