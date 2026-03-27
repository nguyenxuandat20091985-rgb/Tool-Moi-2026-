"""
🐂 TITAN V27 AI - ULTIMATE EDITION
Dành riêng cho tuổi Sửu (Thổ/Kim)
- Cosine Similarity Pattern Detection
- AI: NVIDIA Llama-3.1 + Gemini
- Phát hiện cầu lừa nhà cái
- Pascal, 1-1, nhịp cầu
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

st.set_page_config(page_title="🐂 TITAN V27 - TUỔI SỬU", page_icon="🐂", layout="centered")

# Màu hợp mệnh Thổ/Kim (tuổi Sửu)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
    
    .main {
        padding: 0.5rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .big-num {
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(135deg, #f4d03f, #b7950b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-family: monospace;
        letter-spacing: 8px;
        margin: 10px 0;
        text-shadow: 0 0 20px rgba(244, 208, 63, 0.5);
    }
    
    .box {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        border: 3px solid #f4d03f;
        color: #f4d03f;
        padding: 12px;
        border-radius: 15px;
        text-align: center;
        margin: 8px 0;
        box-shadow: 0 4px 15px rgba(244, 208, 63, 0.3);
    }
    
    .grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin: 10px 0;
    }
    
    .item {
        background: linear-gradient(135deg, #34495e, #2c3e50);
        border: 2px solid #f4d03f;
        color: #f4d03f;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-family: monospace;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 2px 10px rgba(244, 208, 63, 0.2);
    }
    
    .item-3 {
        background: linear-gradient(135deg, #8e44ad, #7d3c98);
        border: 2px solid #f4d03f;
        color: white;
    }
    
    button {
        width: 100%;
        background: linear-gradient(135deg, #f4d03f, #b7950b);
        color: #1a1a2e;
        font-size: 20px;
        font-weight: 900;
        padding: 15px;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(244, 208, 63, 0.4);
    }
    
    textarea {
        height: 80px;
        font-size: 14px;
        background: #2c3e50;
        color: #f4d03f;
        border: 2px solid #34495e;
        border-radius: 8px;
    }
    
    .metric {
        display: flex;
        justify-content: space-around;
        margin: 5px 0;
        background: #2c3e50;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid #34495e;
    }
    
    .metric-val {
        font-size: 22px;
        font-weight: bold;
        color: #f4d03f;
    }
    
    .metric-lbl {
        font-size: 11px;
        color: #95a5a6;
    }
    
    h1 {
        font-size: 28px;
        margin: 5px 0;
        color: #f4d03f;
        text-align: center;
        text-shadow: 0 0 10px rgba(244, 208, 63, 0.5);
    }
    
    h2 {
        font-size: 20px;
        margin: 5px 0;
        color: #f4d03f;
    }
    
    .alert {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #f4d03f;
    }
    
    .warning {
        background: linear-gradient(135deg, #f39c12, #d68910);
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .success {
        background: linear-gradient(135deg, #27ae60, #229954);
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .pattern-box {
        background: #2c3e50;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 3px solid #f4d03f;
        font-size: 12px;
    }
    
    .cosine-sim {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        padding: 8px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

def get_nums(text):
    return re.findall(r"\d{5}", text) if text else []

def cosine_similarity(vec1, vec2):
    """Tính cosine similarity giữa 2 vector"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a ** 2 for a in vec1))
    norm2 = math.sqrt(sum(b ** 2 for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def detect_cau_lua(db):
    """Phát hiện cầu lừa nhà cái"""
    if len(db) < 15:
        return []
    
    lua_patterns = []
    
    # Pattern 1: Ra 2-3 kỳ rồi mất 1-2 kỳ (lừa đánh theo cầu)
    for digit in range(10):
        ds = str(digit)
        recent = [ds in num for num in db[-15:]]
        
        for i in range(len(recent) - 4):
            if recent[i] and recent[i+1] and not recent[i+2] and recent[i+3]:
                lua_patterns.append(f"Số {ds}: Ra 2 kỳ → Nghỉ 1 kỳ → Ra lại (CẦU LỪA)")
                break
    
    # Pattern 2: Bệt giả (ra liên tiếp 2-3 kỳ rồi cắt)
    for pos in range(5):
        recent_pos = [num[pos] for num in db[-10:]]
        for i in range(len(recent_pos) - 3):
            if recent_pos[i] == recent_pos[i+1] != recent_pos[i+2]:
                lua_patterns.append(f"Vị trí {pos+1}: Bệt {recent_pos[i]} 2 kỳ rồi cắt (BỆT GIẢ)")
                break
    
    return lua_patterns

def pascal_triangle_prediction(db):
    """Tam giác Pascal dự đoán"""
    if len(db) < 5:
        return None
    
    # Lấy số cuối cùng
    last_num = db[-1]
    digits = [int(d) for d in last_num]
    
    # Tạo tam giác Pascal
    triangle = [digits]
    while len(triangle[-1]) > 1:
        prev_row = triangle[-1]
        new_row = [(prev_row[i] + prev_row[i+1]) % 10 for i in range(len(prev_row)-1)]
        triangle.append(new_row)
    
    # Dự đoán số tiếp theo từ đỉnh tam giác
    predicted = triangle[-1][0] if triangle[-1] else 0
    
    return {
        "triangle": triangle,
        "predicted_digit": predicted,
        "confidence": len(db) // 10
    }

def detect_nhip_cau(db):
    """Phát hiện nhịp cầu (1-1, 2-1, 1-2)"""
    if len(db) < 10:
        return None
    
    nhịp_info = {}
    
    for digit in range(10):
        ds = str(digit)
        appearances = []
        
        for i, num in enumerate(db[-20:]):
            if ds in num:
                appearances.append(i)
        
        if len(appearances) >= 3:
            gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                nhịp_info[ds] = {
                    "avg_gap": avg_gap,
                    "last_seen": len(db) - 1 - appearances[-1],
                    "pattern": f"Nhịp {int(avg_gap)} kỳ"
                }
    
    return nhịp_info

def advanced_statistical_analysis(db):
    """Phân tích thống kê nâng cao với Cosine Similarity"""
    if len(db) < 10:
        return None
    
    # Tạo vector tần suất cho từng số
    vectors = {}
    for digit in range(10):
        ds = str(digit)
        vector = [1 if ds in num else 0 for num in db[-20:]]
        vectors[ds] = vector
    
    # Tính cosine similarity giữa các số
    similarity_matrix = {}
    for d1 in range(10):
        for d2 in range(d1+1, 10):
            sim = cosine_similarity(vectors[str(d1)], vectors[str(d2)])
            similarity_matrix[(str(d1), str(d2))] = sim
    
    # Tìm cặp có similarity cao nhất (hay đi cùng nhau)
    top_pairs = sorted(similarity_matrix.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Tính điểm tổng hợp
    all_digits = "".join(db[-20:])
    scores = {str(i): all_digits.count(str(i)) * 2 for i in range(10)}
    
    # Bonus từ similarity
    for (d1, d2), sim in top_pairs[:3]:
        if sim > 0.6:
            scores[d1] += 10
            scores[d2] += 10
    
    # Sort và lấy top 8
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_8 = [x[0] for x in sorted_scores[:8]]
    
    # Tạo 3 cặp 2 tinh
    pool_2 = sorted(top_8[:6])
    all_pairs = ["".join(p) for p in combinations(pool_2, 2)]
    
    scored_pairs = []
    for pair in all_pairs:
        score = scores[pair[0]] + scores[pair[1]]
        # Bonus nếu có similarity cao
        if (pair[0], pair[1]) in similarity_matrix:
            score += similarity_matrix[(pair[0], pair[1])] * 20
        scored_pairs.append((pair, score))
    
    scored_pairs.sort(key=lambda x: x[1], reverse=True)
    top_3_pairs = [p[0] for p in scored_pairs[:3]]
    
    # Tạo 3 tổ hợp 3 tinh
    pool_3 = sorted(top_8[2:8])
    all_triples = ["".join(t) for t in combinations(pool_3, 3)]
    
    scored_triples = []
    for triple in all_triples:
        score = sum(scores[d] for d in triple)
        scored_triples.append((triple, score))
    
    scored_triples.sort(key=lambda x: x[1], reverse=True)
    top_3_triples = [t[0] for t in scored_triples[:3]]
    
    return {
        "all_8": "".join(sorted(top_8)),
        "pairs": top_3_pairs,
        "triples": top_3_triples,
        "scores": scores,
        "top_similar_pairs": top_pairs,
        "cosine_analysis": similarity_matrix
    }

def ai_enhancement(db, stat_result, pascal_result, nhịp_cau):
    """AI Enhancement với NVIDIA/Gemini"""
    try:
        nv_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
        genai.configure(api_key=GEMINI_API_KEY)
        gm_model = genai.GenerativeModel('gemini-1.5-flash')
        
        recent = db[-20:] if len(db) >= 20 else db
        
        prompt = f"""
🐂 PHÂN TÍCH XỔ SỐ 5D CHO TUỔI SỬU

📊 DỮ LIỆU {len(recent)} KỲ GẦN:
{recent}

📈 THỐNG KÊ NÂNG CAO:
- 8 số mạnh: {stat_result['all_8']}
- Cosine Similarity Pairs: {stat_result['top_similar_pairs'][:3]}
- Pascal dự đoán: {pascal_result['predicted_digit'] if pascal_result else 'N/A'}
- Nhịp cầu: {nhịp_cau}

🎯 YÊU CẦU:
Chọn 6 số TỐT NHẤT (3 cho 2-tinh, 3 cho 3-tinh, KHÔNG trùng nhau).
Ưu tiên số hợp mệnh Thổ/Kim (tuổi Sửu): 0, 2, 5, 7, 8

TRẢ VỀ JSON:
{{
  "two_digit_3nums": ["x","y","z"],
  "three_digit_3nums": ["a","b","c"],
  "confidence": 80,
  "reasoning": "Giải thích ngắn gọn tại sao chọn",
  "lucky_for_suu": true/false
}}

LƯU Ý:
- 2-tinh: Tạo 3 cặp từ 3 số (1,2,3 → 12,13,23)
- 3-tinh: Tạo 3 tổ hợp từ 3 số (4,5,6 → 456)
- Confidence: 70-90%
- Ưu tiên số có cosine similarity cao
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
                # Generate more triples
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
                "confidence": ai_result.get("confidence", 80),
                "reasoning": ai_result.get("reasoning", "AI analysis"),
                "lucky_for_suu": ai_result.get("lucky_for_suu", False)
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
st.markdown('<p style="text-align:center;color:#f4d03f;font-size:12px;">Hợp mệnh Thổ/Kim | Cosine Similarity + AI</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div class="metric"><div class="metric-val">{len(st.session_state.db)}</div><div class="metric-lbl">Tổng kỳ</div></div>', unsafe_allow_html=True)
with col2:
    last = st.session_state.db[-1] if st.session_state.db else "-"
    st.markdown(f'<div class="metric"><div class="metric-val" style="font-size:16px;">{last}</div><div class="metric-lbl">Kỳ cuối</div></div>', unsafe_allow_html=True)

user_input = st.text_area("📥 Dán 20-30 kỳ gần nhất:", placeholder="3280231\n3280230\n3280229\n...")

if st.button("⚡ PHÂN TÍCH CHUYÊN SÂU"):
    numbers = get_nums(user_input)
    if numbers:
        st.session_state.db = numbers[-30:]
        
        with st.spinner("🤖 AI đang phân tích sâu..."):
            # 1. Statistical Analysis với Cosine Similarity
            stat_result = advanced_statistical_analysis(st.session_state.db)
            
            # 2. Pascal Triangle
            pascal_result = pascal_triangle_prediction(st.session_state.db)
            
            # 3. Detect nhịp cầu
            nhịp_cau = detect_nhip_cau(st.session_state.db)
            
            # 4. Detect cầu lừa
            cau_lua = detect_cau_lua(st.session_state.db)
            
            if stat_result:
                # 5. AI Enhancement
                ai_result = ai_enhancement(st.session_state.db, stat_result, pascal_result, nhịp_cau)
                
                if ai_result:
                    final = {
                        "all_8": stat_result["all_8"],
                        "pairs": ai_result["pairs"],
                        "triples": ai_result["triples"],
                        "conf": ai_result["confidence"],
                        "reasoning": ai_result["reasoning"],
                        "lucky_for_suu": ai_result.get("lucky_for_suu", False),
                        "cau_lua": cau_lua,
                        "pascal": pascal_result,
                        "nhip_cau": nhịp_cau,
                        "cosine_pairs": stat_result["top_similar_pairs"][:3]
                    }
                else:
                    final = {
                        "all_8": stat_result["all_8"],
                        "pairs": stat_result["pairs"],
                        "triples": stat_result["triples"],
                        "conf": 75,
                        "reasoning": "Statistical analysis with Cosine Similarity",
                        "lucky_for_suu": False,
                        "cau_lua": cau_lua,
                        "pascal": pascal_result,
                        "nhip_cau": nhịp_cau,
                        "cosine_pairs": stat_result["top_similar_pairs"][:3]
                    }
                
                st.session_state.result = final
                st.rerun()
            else:
                st.error("❌ Cần ít nhất 10 kỳ")
    else:
        st.error("❌ Không có số!")

# Display Results
if st.session_state.result:
    r = st.session_state.result
    
    # Lucky badge for Suu
    lucky_badge = "🐂 HỢP MỆNH SỬU" if r.get("lucky_for_suu") else ""
    
    st.markdown(f'<div class="box">🔥 ĐỘ TIN CẬY: {r["conf"]}% {lucky_badge}</div>', unsafe_allow_html=True)
    
    # Cảnh báo cầu lừa
    if r.get("cau_lua"):
        st.markdown('<div class="alert">⚠️ <b>CẦU LỪA PHÁT HIỆN:</b></div>', unsafe_allow_html=True)
        for warning in r["cau_lua"][:3]:
            st.markdown(f'<div class="pattern-box">{warning}</div>', unsafe_allow_html=True)
    
    # Pascal prediction
    if r.get("pascal"):
        pascal = r["pascal"]
        st.markdown(f'<div class="success">🔺 <b>PASCAL:</b> Dự đoán số {pascal["predicted_digit"]} (độ tin cậy {pascal["confidence"]*10}%)</div>', unsafe_allow_html=True)
    
    # Cosine Similarity pairs
    if r.get("cosine_pairs"):
        st.markdown('<div class="cosine-sim"><b>COSINE SIMILARITY:</b> Các cặp hay đi cùng:</div>', unsafe_allow_html=True)
        for (d1, d2), sim in r["cosine_pairs"]:
            st.markdown(f'<div class="pattern-box">{d1}{d2}: {sim*100:.1f}% similarity</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div style="text-align:center;"><div style="color:#95a5a6;font-size:12px;">8 SỐ MẠNH</div><div class="big-num">{r["all_8"]}</div></div>', unsafe_allow_html=True)
    
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
    
    # AI Reasoning
    if r.get("reasoning"):
        st.markdown(f'<div class="box" style="background:#2c3e50;"><h3>🧠 AI PHÂN TÍCH:</h3><p style="color:#ecf0f1;font-size:13px;">{r["reasoning"]}</p></div>', unsafe_allow_html=True)

if st.button("🗑️ XÓA HẾT"):
    st.session_state.db = []
    st.session_state.result = None
    st.rerun()

st.markdown('<div style="text-align:center;color:#95a5a6;font-size:10px;margin-top:20px;">🐂 TITAN V27 AI - Designed for Tuổi Sửu</div>', unsafe_allow_html=True)