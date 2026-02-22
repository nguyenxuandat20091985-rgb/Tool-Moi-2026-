import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter
import numpy as np
from itertools import combinations

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v21.json" 

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None 

neural_engine = setup_neural() 

# ================= HỆ THỐNG GHI NHỚ VĨNH VIỄN =================
def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: 
            try: return json.load(f)
            except: return []
    return [] 

def save_memory(data):
    # Giữ lại 1000 kỳ gần nhất để AI có dữ liệu sâu
    with open(DB_FILE, "w") as f: 
        json.dump(data[-1000:], f) 

if "history" not in st.session_state:
    st.session_state.history = load_memory() 

# ================= THUẬT TOÁN DỰ ĐOÁN 3 SỐ 5 TÍNH =================
def analyze_3so5tinh(history_data):
    """
    Phân tích và dự đoán cho trò chơi 3 số 5 tính
    Quy tắc: Chọn 3 số bất kỳ từ 0-9, trúng thưởng nếu cả 3 số xuất hiện trong 5 vị trí
    """
    if len(history_data) < 10:
        return None
    
    # Lấy 50 kỳ gần nhất để phân tích
    recent = history_data[-50:]
    
    # 1. Phân tích tần suất xuất hiện của từng số
    all_digits = ''.join(recent)
    freq = Counter(all_digits)
    
    # 2. Phân tích số lần xuất hiện trong từng vị trí
    pos_freq = [{str(i):0 for i in range(10)} for _ in range(5)]
    for draw in recent:
        for pos, digit in enumerate(draw):
            pos_freq[pos][digit] = pos_freq[pos].get(digit, 0) + 1
    
    # 3. Phân tích cặp số thường xuất hiện cùng nhau
    pair_freq = Counter()
    for draw in recent:
        digits = set(draw)
        for pair in combinations(digits, 2):
            pair_freq[tuple(sorted(pair))] += 1
    
    # 4. Phân tích xu hướng "bệt" (số xuất hiện nhiều kỳ liên tiếp)
    streak_pattern = []
    last_draw = recent[-1]
    for digit in '0123456789':
        count = sum(1 for d in last_draw if d == digit)
        if count > 0:
            streak_pattern.append(digit)
    
    # 5. Thuật toán dự đoán
    scores = {str(i): 0 for i in range(10)}
    
    # Factor 1: Tần suất tổng thể (30%)
    total_draws = len(recent) * 5
    for digit, count in freq.items():
        scores[digit] += (count / total_draws) * 30
    
    # Factor 2: Tần suất gần đây (25%)
    recent_draws = ''.join(recent[-10:])
    recent_freq = Counter(recent_draws)
    for digit, count in recent_freq.items():
        scores[digit] += (count / (10 * 5)) * 25
    
    # Factor 3: Xu hướng bệt (20%)
    for digit in streak_pattern:
        scores[digit] += 20
    
    # Factor 4: Số "lạnh" cần nổ (15%)
    all_digits_set = set('0123456789')
    cold_digits = all_digits_set - set(freq.keys())
    for digit in cold_digits:
        scores[digit] += 15
    
    # Factor 5: Cặp số tiềm năng (10%)
    top_pairs = pair_freq.most_common(5)
    for pair, _ in top_pairs:
        for digit in pair:
            scores[digit] += 5
    
    # Sắp xếp và chọn top 7 số
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_7 = [digit for digit, _ in sorted_scores[:7]]
    
    # Tạo logic giải thích
    logic = f"🎯 PHÂN TÍCH 3 SỐ 5 TÍNH:\n"
    logic += f"- Top 3 số có điểm cao nhất: {', '.join(top_7[:3])}\n"
    logic += f"- Số đang bệt: {', '.join(streak_pattern[:3])}\n"
    logic += f"- Cặp số thường về cùng: "
    for pair, count in top_pairs[:3]:
        logic += f"{pair[0]}-{pair[1]}({count} lần) "
    
    return {
        "dan4": top_7[:4],      # 4 số chủ lực
        "dan3": top_7[4:7],     # 3 số lót
        "logic": logic,
        "top_pairs": [list(pair) for pair, _ in top_pairs[:3]],
        "streak": streak_pattern[:3]
    }

def predict_with_ai(history):
    """Sử dụng AI để phân tích nâng cao"""
    prompt = f"""
    Bạn là AI chuyên gia phân tích trò chơi "3 số 5 tính" (chọn 3 số từ 0-9, 
    trúng thưởng nếu cả 3 số xuất hiện trong 5 vị trí).
    
    Lịch sử 50 kỳ gần nhất: {history[-50:]}
    
    Phân tích:
    1. Xác định các số có khả năng xuất hiện cao nhất trong kỳ tiếp theo
    2. Phát hiện các cặp số thường về cùng nhau
    3. Dự đoán xu hướng "bệt" và "số lạnh"
    
    TRẢ VỀ JSON:
    {{
        "dan4": ["số1", "số2", "số3", "số4"],
        "dan3": ["số5", "số6", "số7"],
        "logic": "Giải thích ngắn gọn lý do chọn các số này",
        "confidence": 85
    }}
    """
    
    try:
        response = neural_engine.generate_content(prompt)
        res_text = response.text
        data = json.loads(re.search(r'\{.*\}', res_text, re.DOTALL).group())
        return data
    except:
        return None

# ================= UI DESIGN (Giữ nguyên cấu trúc) =================
st.set_page_config(page_title="TITAN v21.0 PRO", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #c9d1d9; }
    .status-active { color: #238636; font-weight: bold; border-left: 3px solid #238636; padding-left: 10px; }
    .prediction-card {
        background: #0d1117; border: 2px solid #30363d;
        border-radius: 12px; padding: 25px; margin-top: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .num-display { 
        font-size: 60px; font-weight: 900; color: #58a6ff; 
        text-align: center; letter-spacing: 10px; text-shadow: 0 0 25px #58a6ff;
    }
    .logic-box { font-size: 14px; color: #8b949e; background: #161b22; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
    .pair-analysis { font-size: 13px; color: #f2cc60; background: #1a1f2a; padding: 8px; border-radius: 5px; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True) 

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🧬 TITAN v21.0 OMNI - 3 SỐ 5 TÍNH</h2>", unsafe_allow_html=True)
if neural_engine:
    st.markdown(f"<p class='status-active'>● KẾT NỐI NEURAL-LINK: OK | DỮ LIỆU: {len(st.session_state.history)} KỲ</p>", unsafe_allow_html=True)
else:
    st.error("LỖI KẾT NỐI API - KIỂM TRA LẠI KEY") 

# ================= XỬ LÝ DỮ LIỆU =================
raw_input = st.text_area("📡 NẠP DỮ LIỆU (Dán các dãy 5 số):", height=100, placeholder="32880\n21808\n...") 

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 DỰ ĐOÁN 3 SỐ 5 TÍNH"):
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            save_memory(st.session_state.history)
            
            # Thử dùng AI trước, nếu không được thì dùng thuật toán
            ai_result = predict_with_ai(st.session_state.history[-50:]) if neural_engine else None
            
            if ai_result:
                st.session_state.last_result = ai_result
            else:
                # Dùng thuật toán phân tích
                st.session_state.last_result = analyze_3so5tinh(st.session_state.history[-50:])
            
            st.rerun() 

with col2:
    if st.button("🗑️ RESET BỘ NHỚ"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun() 

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_result" in st.session_state and st.session_state.last_result:
    res = st.session_state.last_result
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='logic-box'><b>💡 PHÂN TÍCH 3 SỐ 5 TÍNH:</b><br>{res['logic']}</div>", unsafe_allow_html=True)
    
    # Hiển thị phân tích cặp số nếu có
    if 'top_pairs' in res:
        pairs_text = " | ".join([f"{p[0]}-{p[1]}" for p in res['top_pairs']])
        st.markdown(f"<div class='pair-analysis'>📊 CẶP SỐ TIỀM NĂNG: {pairs_text}</div>", unsafe_allow_html=True)
    
    if 'streak' in res:
        st.markdown(f"<div class='pair-analysis'>🔥 SỐ ĐANG BỆT: {', '.join(res['streak'])}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; font-size:12px; color:#888;'>🎯 4 SỐ CHỦ LỰC (VÀO TIỀN)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display'>{''.join(map(str, res['dan4']))}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; font-size:12px; color:#888; margin-top:20px;'>🛡️ 3 SỐ LÓT (GIỮ VỐN)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display' style='color:#f2cc60; text-shadow: 0 0 25px #f2cc60;'>{''.join(map(str, res['dan3']))}</div>", unsafe_allow_html=True)
    
    copy_val = "".join(map(str, res['dan4'])) + "".join(map(str, res['dan3']))
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ:", copy_val)
    
    if 'confidence' in res:
        st.progress(res['confidence']/100, text=f"Độ tin cậy: {res['confidence']}%")
    
    st.markdown("</div>", unsafe_allow_html=True) 

# Thêm phần hướng dẫn
with st.expander("📘 HƯỚNG DẪN 3 SỐ 5 TÍNH"):
    st.markdown("""
    **QUY TẮC:**
    - Chọn 3 con số bất kỳ từ 0-9
    - Trúng thưởng nếu cả 3 số đều xuất hiện trong kết quả 5 số (không cần đúng thứ tự)
    - Mỗi số chỉ tính 1 lần dù xuất hiện nhiều lần
    
    **VÍ DỤ:**
    - Chọn: 1,2,6
    - Kết quả: 12864 → ✅ TRÚNG (có 1,2,6)
    - Kết quả: 12662 → ❌ KHÔNG TRÚNG (thiếu số 3)
    
    **THUẬT TOÁN DỰ ĐOÁN:**
    - Phân tích tần suất tổng thể
    - Phân tích tần suất gần đây
    - Xu hướng bệt số
    - Cặp số thường về cùng nhau
    - Số lạnh cần nổ
    """)

st.markdown("<br><p style='text-align:center; font-size:10px; color:#444;'>Hệ thống tự học từ dữ liệu lịch sử - Tối ưu cho 3 số 5 tính</p>", unsafe_allow_html=True)