"""
🚀 TITAN V27 AI - FIXED
Version: 11.0.0-FIXED
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
    .item {background: #ffc107; color: #000; padding: 12px; border-radius: 8px; text-align: center; font-family: monospace; font-size: 28px; font-weight: bold;}
    .item-3 {background: #17a2b8; color: white;}
    button {width: 100%; background: #28a745; color: white; font-size: 18px; font-weight: bold; padding: 12px; border: none; border-radius: 8px;}
    textarea {height: 80px; font-size: 14px;}
    .metric {display: flex; justify-content: space-around; margin: 5px 0;}
    .metric-val {font-size: 20px; font-weight: bold; color: #28a745;}
    .metric-lbl {font-size: 11px; color: #666;}
    h1 {font-size: 24px; margin: 5px 0;}
    h2 {font-size: 18px; margin: 5px 0;}
    .ai-logic {background: #f3e5f5; padding: 8px; border-radius: 8px; margin: 8px 0; font-size: 11px;}
</style>
""", unsafe_allow_html=True)

def get_nums(text):
    return re.findall(r"\d{5}", text) if text else []

def statistical_predict(db):
    if len(db) < 10:
        return None
    
    all_digits = "".join(db[-20:])
    scores = {str(i): all_digits.count(str(i)) for i in range(10)}
    
    for num in db[-3:]:
        for d in set(num):
            scores[d] += 5
    
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
        "triples": top_3_triples
    }

def ai_enhance(db, stat_result):
    """AI phân tích và điều chỉnh"""
    try:
        nv_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
        genai.configure(api_key=GEMINI_API_KEY)
        gm_model = genai.GenerativeModel('gemini-1.5-flash')
        
        recent = db[-20:] if len(db) >= 20 else db
        
        prompt = f"""
Dữ liệu 20 kỳ gần: {recent}

Thống kê:
- 8 số nóng: {stat_result['all_8']}
- Pairs đề xuất: {stat_result['pairs']}
- Triples đề xuất: {stat_result['triples']}

PHÂN TÍCH:
1. Tìm 3 số mạnh nhất cho 2 tinh (từ 0-9)
2. Tìm 3 số mạnh nhất cho 3 tinh (từ 0-9, KHÁC với 3 số trên)
3. Tổng 6 số không trùng nhau

JSON output:
{{
  "two_digit_3nums": ["1","2","3"],
  "three_digit_3nums": ["4","5","6"],
  "confidence": 75,
  "reasoning": "Giải thích ngắn"
}}
"""
        
        try:
            completion = nv_client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            ai_data = json.loads(completion.choices[0].message.content)
        except:
            gm_result = gm_model.generate_content(prompt)
            json_match = re.search(r'\{[\s\S]*\}', gm_result.text)
            ai_data = json.loads(json_match.group()) if json_match else {}
        
        # ✅ FIX: Tạo pairs đúng từ 3 số
        two_nums = ai_data.get("two_digit_3nums", [])
        if len(two_nums) >= 2:
            two_nums = sorted(two_nums[:3])
            final_pairs = ["".join(p) for p in combinations(two_nums, 2)][:3]
        else:
            final_pairs = stat_result["pairs"]
        
        # ✅ FIX: Tạo triples đúng từ 3 số
        three_nums = ai_data.get("three_digit_3nums", [])
        if len(three_nums) >= 3:
            three_nums = sorted(three_nums[:3])
            final_triples = ["".join(three_nums)]
        else:
            final_triples = stat_result["triples"]
        
        return {
            "pairs": final_pairs,
            "triples": final_triples,
            "confidence": ai_data.get("confidence", 75),
            "reasoning": ai_data.get("reasoning", "AI analysis")
        }
    
    except Exception as e:
        st.error(f"AI error: {str(e)[:50]}")
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
        
        with st.spinner("🤖 AI đang tính..."):
            stat_result = statistical_predict(st.session_state.db)
            
            if stat_result:
                ai_result = ai_enhance(st.session_state.db, stat_result)
                
                if ai_result:
                    st.session_state.result = {
                        "all_8": stat_result["all_8"],
                        "pairs": ai_result["pairs"],
                        "triples": ai_result["triples"],
                        "conf": ai_result["confidence"],
                        "reasoning": ai_result["reasoning"]
                    }
                else:
                    st.session_state.result = stat_result
                    st.session_state.result["conf"] = 70
                st.rerun()
            else:
                st.error("❌ Cần 10+ kỳ")
    else:
        st.error("❌ Không có số!")

if st.session_state.result:
    r = st.session_state.result
    
    st.markdown(f'<div class="box">🔥 {r.get("conf", 70)}% <span style="background:#6f42c1;padding:2px 6px;border-radius:4px;font-size:10px;">AI</span></div>', unsafe_allow_html=True)
    
    st.markdown(f'<div style="text-align:center;"><div style="color:#666;font-size:12px;">8 SỐ</div><div class="big-num">{r["all_8"]}</div></div>', unsafe_allow_html=True)
    
    if "reasoning" in r:
        st.markdown(f'<div class="ai-logic">🧠 {r["reasoning"]}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="box"><h2>🎯 2 TINH (2 chữ số)</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for pair in r["pairs"]:
        st.markdown(f'<div class="item">{pair}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="box"><h2>🎯 3 TINH (3 chữ số)</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for triple in r["triples"]:
        st.markdown(f'<div class="item item-3">{triple}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🗑️ XÓA"):
    st.session_state.db = []
    st.session_state.result = None
    st.rerun()

st.markdown('<div style="text-align:center;color:#999;font-size:10px;">AI-Powered</div>', unsafe_allow_html=True)