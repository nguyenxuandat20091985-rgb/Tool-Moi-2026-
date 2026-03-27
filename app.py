"""
🚀 TITAN V27 AI - FIXED FORMAT
Version: 10.1.0-FIXED
"""
import streamlit as st
import re
from collections import Counter
from itertools import combinations
from openai import OpenAI
import google.generativeai as genai
import json

NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

st.set_page_config(page_title="TITAN V27 AI", page_icon="🤖", layout="centered")

st.markdown("""
<style>
    .main {padding: 0.5rem;}
    .big-num {font-size: 42px; font-weight: bold; color: #28a745; text-align: center; font-family: monospace; letter-spacing: 6px; margin: 5px 0;}
    .box {background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 10px; border-radius: 10px; text-align: center; margin: 5px 0;}
    .grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px; margin: 10px 0;}
    .item {background: #ffc107; color: #000; padding: 12px; border-radius: 8px; text-align: center; font-family: monospace; font-size: 24px; font-weight: bold;}
    .item-3 {background: #17a2b8; color: white;}
    button {width: 100%; background: #28a745; color: white; font-size: 18px; font-weight: bold; padding: 12px; border: none; border-radius: 8px;}
    textarea {height: 80px; font-size: 14px;}
    .metric {display: flex; justify-content: space-around; margin: 5px 0;}
    .metric-val {font-size: 20px; font-weight: bold; color: #28a745;}
    .metric-lbl {font-size: 11px; color: #666;}
    h1 {font-size: 24px; margin: 5px 0;}
    h2 {font-size: 18px; margin: 5px 0;}
    .ai-logic {background: #f3e5f5; padding: 10px; border-radius: 8px; margin: 10px 0; font-size: 12px;}
</style>
""", unsafe_allow_html=True)

def get_nums(text):
    return re.findall(r"\d{5}", text) if text else []

def validate_pair(pair_str):
    """Ensure 2-digit format"""
    if not pair_str:
        return None
    # Extract only digits
    digits = re.findall(r'\d', str(pair_str))
    if len(digits) >= 2:
        return digits[0] + digits[1]
    return None

def validate_triple(triple_str):
    """Ensure 3-digit format"""
    if not triple_str:
        return None
    digits = re.findall(r'\d', str(triple_str))
    if len(digits) >= 3:
        return digits[0] + digits[1] + digits[2]
    return None

def statistical_predict(db):
    """AI 1: Statistical Analysis"""
    if len(db) < 10:
        return None
    
    all_digits = "".join(db[-20:])
    scores = {str(i): all_digits.count(str(i)) for i in range(10)}
    
    for num in db[-3:]:
        for d in set(num):
            scores[d] += 5
    
    for d in range(10):
        ds = str(d)
        gan = True
        for num in db[-8:]:
            if ds in num:
                gan = False
                break
        if gan:
            scores[ds] -= 10
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_8 = [x[0] for x in sorted_scores[:8]]
    
    pool_2 = sorted(top_8[:6])
    all_pairs = ["".join(p) for p in combinations(pool_2, 2)]
    
    scored_pairs = []
    for pair in all_pairs:
        score = scores[pair[0]] + scores[pair[1]]
        for num in db[-10:]:
            if pair[0] in num and pair[1] in num:
                score += 15
                break
        scored_pairs.append((pair, score))
    
    scored_pairs.sort(key=lambda x: x[1], reverse=True)
    top_3_pairs = [p[0] for p in scored_pairs[:3]]
    
    pool_3 = sorted(top_8[2:8])
    all_triples = ["".join(t) for t in combinations(pool_3, 3)]
    
    scored_triples = []
    for triple in all_triples:
        score = sum(scores[d] for d in triple)
        for num in db[-10:]:
            if all(d in num for d in triple):
                score += 25
                break
        scored_triples.append((triple, score))
    
    scored_triples.sort(key=lambda x: x[1], reverse=True)
    top_3_triples = [t[0] for t in scored_triples[:3]]
    
    return {
        "all_8": "".join(sorted(top_8)),
        "pairs": top_3_pairs,
        "triples": top_3_triples,
        "scores": scores
    }

def ai_predict(db, stat_result):
    """AI 2: LLM Analysis with strict validation"""
    try:
        nv_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
        genai.configure(api_key=GEMINI_API_KEY)
        gm_model = genai.GenerativeModel('gemini-1.5-flash')
        
        recent = db[-20:] if len(db) >= 20 else db
        
        prompt = f"""
Phân tích xổ số 5D. Dữ liệu {len(recent)} kỳ gần nhất:
{recent}

Thống kê: 8 số mạnh: {stat_result['all_8']}

NHIỆM VỤ: Chọn 6 số đơn (0-9) KHÔNG trùng nhau:
- 3 số cho 2 tinh (2-digit pairs)
- 3 số cho 3 tinh (3-digit triple)

Trả về JSON STRICT:
{{
  "two_digit_nums": ["0", "6", "9"],
  "three_digit_nums": ["1", "4", "7"],
  "confidence": 75,
  "reasoning": "Giải thích ngắn"
}}

LƯU Ý:
- Mỗi số là 1 digit (0-9)
- 6 số KHÁC NHAU hoàn toàn
- Confidence 60-90%
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
            if json_match:
                ai_result = json.loads(json_match.group())
            else:
                raise Exception("No JSON")
        
        # VALIDATE & FORMAT 2-DIGIT PAIRS
        ai_pairs = []
        if "two_digit_nums" in ai_result:
            nums = ai_result["two_digit_nums"]
            # Ensure we have at least 2 digits
            if len(nums) >= 2:
                # Take first 3 and create pairs
                selected = nums[:3]
                pair_list = ["".join(p) for p in combinations(sorted(selected), 2)]
                ai_pairs = pair_list[:3]
        
        # VALIDATE & FORMAT 3-DIGIT TRIPLE
        ai_triples = []
        if "three_digit_nums" in ai_result:
            nums = ai_result["three_digit_nums"]
            if len(nums) >= 3:
                triple = "".join(sorted(nums[:3]))
                ai_triples = [triple]
        
        # Fallback to statistical if AI fails
        if not ai_pairs:
            ai_pairs = stat_result["pairs"]
        if not ai_triples:
            ai_triples = stat_result["triples"]
        
        return {
            "pairs": ai_pairs,
            "triples": ai_triples,
            "confidence": ai_result.get("confidence", 75),
            "reasoning": ai_result.get("reasoning", "AI analysis")
        }
    
    except Exception as e:
        st.error(f"⚠️ AI lỗi: {str(e)[:50]}")
        return None

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

if st.button("⚡ AI PHÂN TÍCH"):
    numbers = get_nums(user_input)
    if numbers:
        st.session_state.db = numbers[-30:]
        
        with st.spinner("🤖 AI đang phân tích..."):
            stat_result = statistical_predict(st.session_state.db)
            
            if stat_result:
                ai_result = ai_predict(st.session_state.db, stat_result)
                
                if ai_result:
                    final_result = {
                        "all_8": stat_result["all_8"],
                        "pairs": ai_result["pairs"],
                        "triples": ai_result["triples"],
                        "conf": ai_result["confidence"],
                        "ai_reasoning": ai_result["reasoning"]
                    }
                    st.session_state.result = final_result
                    st.rerun()
                else:
                    st.session_state.result = stat_result
                    st.session_state.result["conf"] = 70
            else:
                st.error("❌ Cần ít nhất 10 kỳ")
    else:
        st.error("❌ Không có số!")

if st.session_state.result:
    r = st.session_state.result
    
    st.markdown(f'<div class="box">🔥 AI {r.get("conf", 70)}%</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;"><div style="color:#666;font-size:12px;">8 SỐ</div><div class="big-num">{r["all_8"]}</div></div>', unsafe_allow_html=True)
    
    if "ai_reasoning" in r:
        st.markdown(f'<div class="ai-logic">🧠 <b>AI:</b> {r["ai_reasoning"]}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="box"><h2>🎯 2 TINH</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for pair in r["pairs"]:
        # VALIDATE: Must be exactly 2 digits
        validated_pair = validate_pair(pair)
        if validated_pair:
            st.markdown(f'<div class="item">{validated_pair}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="box"><h2>🎯 3 TINH</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for triple in r["triples"]:
        # VALIDATE: Must be exactly 3 digits
        validated_triple = validate_triple(triple)
        if validated_triple:
            st.markdown(f'<div class="item item-3">{validated_triple}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🗑️ XÓA"):
    st.session_state.db = []
    st.session_state.result = None
    st.rerun()

st.markdown('<div style="text-align:center;color:#999;font-size:10px;">🤖 AI-Powered TITAN V27</div>', unsafe_allow_html=True)