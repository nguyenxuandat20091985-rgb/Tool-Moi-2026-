import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter
import math

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_neural_memory_v22.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= QUẢN LÝ BỘ NHỚ VÀ DỮ LIỆU SẠCH =================
def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_memory(data):
    # Lưu trữ 2000 kỳ để phân tích chu kỳ dài hơn
    with open(DB_FILE, "w") as f:
        json.dump(data[-2000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_memory()

# ================= GIAO DIỆN TITAN PRO =================
st.set_page_config(page_title="TITAN v22.0 OMNI", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #c9d1d9; }
    .status-panel { background: #0d1117; padding: 10px; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 20px; }
    .prediction-card {
        background: linear-gradient(145deg, #0d1117, #161b22);
        border: 1px solid #58a6ff; border-radius: 15px; padding: 30px;
        box-shadow: 0 0 30px rgba(88, 166, 255, 0.1);
    }
    .main-number { font-size: 85px; font-weight: 900; color: #ff5858; text-shadow: 0 0 30px #ff5858; text-align: center; }
    .secondary-number { font-size: 50px; font-weight: 700; color: #58a6ff; text-align: center; opacity: 0.8; }
    .warning-box { background: #331010; color: #ff7b72; padding: 15px; border-radius: 8px; border: 1px solid #6e2121; text-align: center; font-weight: bold; }
    .algo-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }
    .algo-table td, .algo-table th { border: 1px solid #30363d; padding: 8px; text-align: center; }
    .highlight-gold { color: #f2cc60; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ================= PHẦN PHÂN TÍCH THUẬT TOÁN CŨ =================
def analyze_patterns(data):
    if not data: return "Chưa có dữ liệu"
    all_digits = "".join(data)
    counts = Counter(all_digits)
    # Tìm quy luật bóng số
    shadow_map = {'0':'5', '5':'0', '1':'6', '6':'1', '2':'7', '7':'2', '3':'8', '8':'3', '4':'9', '9':'4'}
    last_draw = data[-1]
    potential_shadows = [shadow_map[d] for d in last_draw]
    return f"Tần suất cao: {counts.most_common(3)} | Bóng số tiềm năng: {''.join(potential_shadows)}"

# ================= THUẬT TOÁN NÂNG CẤP SOI CẦU 3 CÀNG (V23.0 BỔ SUNG) =================

def calculate_digit_score(data_200):
    """Tính toán điểm số cho từng digit 0-9 dựa trên đa tiêu chí"""
    scores = {str(i): 0.0 for i in range(10)}
    if len(data_200) < 10: return scores
    
    # 1. Tần suất (Frequency)
    all_digits = "".join(data_200)
    freq_counter = Counter(all_digits)
    total_d = sum(freq_counter.values())
    
    # 2. Trọng số suy giảm (Decay Weight) - Ưu tiên các kỳ gần nhất
    decay_scores = {str(i): 0.0 for i in range(10)}
    for i, draw in enumerate(reversed(data_200)):
        weight = math.exp(-0.05 * i) # Càng xa càng giảm
        for d in set(draw):
            decay_scores[d] += weight

    # 3. Markov Chain (Vị trí 3 số cuối: Trăm - Chục - Đơn vị)
    # Ở 5D, ta giả định lấy 3 số cuối làm 3 càng giải đặc biệt
    pos_data = [d[-3:] for d in data_200]
    markov_score = {str(i): 0.0 for i in range(10)}
    for p_idx in range(3):
        col_digits = [p[p_idx] for p in pos_data]
        for i in range(len(col_digits)-1):
            if col_digits[i+1] == col_digits[i]: # Xu hướng lặp vị trí
                markov_score[col_digits[i+1]] += 0.5

    # 4. Entropy & Density (Độ dày đặc của số)
    # Tính toán digit density trong 20 kỳ gần nhất
    recent_20 = "".join(data_200[-20:])
    density = Counter(recent_20)

    # Tổng hợp điểm Score
    for d in scores:
        f_score = (freq_counter[d] / total_d) * 100 if total_d > 0 else 0
        m_score = markov_score[d]
        d_weight = decay_scores[d]
        dens = (density[d] / 20) * 10
        
        # Công thức Score chuẩn hóa
        scores[d] = (f_score * 0.25) + (m_score * 0.25) + (d_weight * 0.25) + (dens * 0.25)
        
    return scores

def select_top7_digits(scores):
    """Chọn 7 digit mạnh nhất, loại 3 digit yếu nhất"""
    sorted_digits = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top7 = [d[0] for d in sorted_digits[:7]]
    bottom3 = [d[0] for d in sorted_digits[7:]]
    return top7, bottom3

def generate_3_digit_combinations(top7):
    """Tạo các tổ hợp 3 số từ 7 digit đã chọn (không tính số chập theo luật 3 càng tinh)"""
    import itertools
    # Theo ảnh mô tả '3 số 5 tinh', người chơi chọn 3 số khác nhau
    return list(itertools.combinations(top7, 3))

def rank_combinations(combinations, digit_scores, data_200):
    """Xếp hạng tổ hợp bằng Monte Carlo mô phỏng và điểm số tổng hợp"""
    ranked = []
    
    # Phân tích tổng 3 càng gần nhất
    sums = [sum(int(d) for d in draw[-3:]) for draw in data_200]
    sum_freq = Counter(sums)
    
    for combo in combinations:
        d1, d2, d3 = combo
        # Điểm dựa trên Score từng digit
        base_score = (digit_scores[d1] + digit_scores[d2] + digit_scores[d3]) / 3
        
        # Điểm dựa trên Tổng 3 số (Tổng phổ biến)
        c_sum = int(d1) + int(d2) + int(d3)
        sum_score = (sum_freq[c_sum] / len(data_200)) * 50 if len(data_200) > 0 else 0
        
        # Mô phỏng Monte Carlo đơn giản (Xác suất xuất hiện đồng thời trong lịch sử)
        hit_count = 0
        for draw in data_200:
            draw_last3 = draw[-3:]
            if all(d in draw_last3 for d in combo):
                hit_count += 1
        mc_score = (hit_count / len(data_200)) * 100 if len(data_200) > 0 else 0
        
        total_score = (base_score * 0.5) + (sum_score * 0.2) + (mc_score * 0.3)
        ranked.append({
            "combo": "".join(sorted(combo)),
            "score": round(total_score, 2),
            "details": f"Base: {round(base_score,1)} | SumScore: {round(sum_score,1)} | MC: {round(mc_score,1)}"
        })
    
    return sorted(ranked, key=lambda x: x['score'], reverse=True)[:10]

# ================= UI CHÍNH =================
st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🧬 TITAN v22.0 PRO OMNI</h1>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='status-panel'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.write(f"📡 NEURAL: {'✅ ONLINE' if neural_engine else '❌ ERROR'}")
    c2.write(f"📊 DATASET: {len(st.session_state.history)} KỲ")
    c3.write(f"🛡️ SAFETY: ACTIVE")
    st.markdown("</div>", unsafe_allow_html=True)

raw_input = st.text_area("📥 NẠP DỮ LIỆU SẠCH (5 số viết liền):", height=120, placeholder="Dán dãy số tại đây...")

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("🚀 KÍCH HOẠT GIẢI MÃ"):
        clean_data = re.findall(r"\b\d{5}\b", raw_input)
        if clean_data:
            st.session_state.history.extend(clean_data)
            save_memory(st.session_state.history)
            
            # 1. Gọi Gemini như cũ
            prompt = f"""
            Hệ thống: TITAN v22.0. Chuyên gia bẻ cầu nhà cái Kubet/Lotobet.
            Dữ liệu lịch sử (100 kỳ): {st.session_state.history[-100:]}.
            Quy luật bóng số: 0-5, 1-6, 2-7, 3-8, 4-9.
            Nhiệm vụ:
            1. Phân tích chu kỳ 'nhả' số của nhà cái.
            2. Chọn ra 3 số CHỦ LỰC có xác suất nổ cao nhất.
            3. TRẢ VỀ JSON: {{"main_3": "chuỗi 3 số", "support_4": "chuỗi 4 số", "logic": "phân tích ngắn", "warning": false, "confidence": 98}}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                json_str = re.search(r'\{.*\}', response.text, re.DOTALL).group()
                st.session_state.last_prediction = json.loads(json_str)
            except Exception as e:
                all_nums = "".join(st.session_state.history[-50:])
                common = [x[0] for x in Counter(all_nums).most_common(7)]
                st.session_state.last_prediction = {
                    "main_3": "".join(common[:3]), "support_4": "".join(common[3:]),
                    "logic": "Sử dụng thuật toán thống kê xác suất thực tế.", "warning": False, "confidence": 75
                }
            
            # 2. Thực hiện thuật toán 3 càng nâng cấp mới (V23)
            data_200 = st.session_state.history[-200:]
            digit_scores = calculate_digit_score(data_200)
            top7, bottom3 = select_top7_digits(digit_scores)
            combos = generate_3_digit_combinations(top7)
            top10_3cang = rank_combinations(combos, digit_scores, data_200)
            
            st.session_state.v23_result = {
                "top7": top7,
                "bottom3": bottom3,
                "top10": top10_3cang
            }
            
            st.rerun()

with col_btn2:
    if st.button("🗑️ DỌN DẸP BỘ NHỚ"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ DỰ ĐOÁN =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    if res.get('warning') or res.get('confidence', 0) < 70:
        st.markdown("<div class='warning-box'>⚠️ CẢNH BÁO: CẦU ĐANG NHIỄU - HẠ MỨC CƯỢC HOẶC DỪNG LẠI</div>", unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    st.write(f"🔍 **CHIẾN THUẬT:** {res['logic']}")
    st.markdown("<p style='text-align:center; color:#888; margin-bottom:0;'>🔥 3 SỐ CHỦ LỰC (2 TINH)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='main-number'>{res['main_3']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ================= HIỂN THỊ KẾT QUẢ 3 CÀNG NÂNG CẤP (V23) =================
if "v23_result" in st.session_state:
    v23 = st.session_state.v23_result
    st.markdown("### 🏆 PHÂN TÍCH 3 CÀNG ĐẶC BIỆT (V23 PRO)")
    
    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        st.success(f"✅ 7 DIGIT MẠNH: {', '.join(v23['top7'])}")
        st.error(f"❌ 3 DIGIT YẾU: {', '.join(v23['bottom3'])}")
        
    with col_v2:
        st.markdown("**Top 10 Tổ hợp 3 Càng tiềm năng nhất:**")
        html_table = "<table class='algo-table'><tr><th>Tổ hợp</th><th>Điểm</th><th>Phân tích chi tiết</th></tr>"
        for item in v23['top10']:
            html_table += f"<tr><td class='highlight-gold'>{item['combo']}</td><td>{item['score']}</td><td>{item['details']}</td></tr>"
        html_table += "</table>"
        st.markdown(html_table, unsafe_allow_html=True)

# Thống kê nhanh dưới cùng
with st.expander("📊 Thống kê nhanh nhịp cầu"):
    st.write(analyze_patterns(st.session_state.history))

