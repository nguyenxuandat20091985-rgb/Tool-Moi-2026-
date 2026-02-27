import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter, defaultdict
import time

# ================= CẤU HÌNH HỆ THỐNG TITAN v25.0 QUANTUM =================
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc" 
DB_FILE = "titan_quantum_v25.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        return None

neural_engine = setup_neural()

# ================= THUẬT TOÁN XỬ LÝ DỮ LIỆU NÂNG CAO =================

class QuantumAnalyzer:
    def __init__(self, history):
        self.history = history
        self.digits_history = []
        self._preprocess()

    def _preprocess(self):
        for num in self.history:
            if len(num) == 5:
                self.digits_history.extend([int(d) for d in num])

    def get_frequency_analysis(self, limit=100):
        recent_data = "".join(self.history[-limit:])
        counts = Counter(recent_data)
        total = sum(counts.values())
        return {k: round(v/total * 100, 2) for k, v in counts.items()}

    def get_gap_analysis(self):
        last_indices = {}
        for i, num in enumerate(self.history):
            for d in num:
                last_indices[d] = i
        
        current_idx = len(self.history)
        gaps = {d: current_idx - idx for d, idx in last_indices.items()}
        return sorted(gaps.items(), key=lambda x: x[1], reverse=True)

    def calculate_weighted_score(self):
        freq = self.get_frequency_analysis(200)
        gaps = dict(self.get_gap_analysis())
        
        scores = {}
        for d in "0123456789":
            f_score = freq.get(d, 0) * 1.5
            g_score = min(gaps.get(d, 0), 20) * 2
            scores[d] = f_score + g_score
            
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ================= QUẢN LÝ DỮ LIỆU =================

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            try: 
                data = json.load(f)
                return data if isinstance(data, list) else []
            except: return []
    return []

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data[-5000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()

# Khởi tạo biến dự phòng ngay từ đầu
if "math_backup" not in st.session_state:
    st.session_state.math_backup = {
        "main_3": "000",
        "support_4": "0000",
        "confidence": 75,
        "reasoning": "Dựa trên thuật toán thống kê thuần túy.",
        "warning": ""
    }

# ================= GIAO DIỆN TITAN v25.0 =================

st.set_page_config(page_title="TITAN v25.0 QUANTUM", layout="wide", page_icon="🔮")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    
    body { font-family: 'JetBrains Mono', monospace; }
    .stApp { background: #050505; color: #c9d1d9; }
    
    .main-card {
        background: linear-gradient(145deg, #0d1117, #161b22);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 0 30px rgba(88, 166, 255, 0.1);
        margin-bottom: 20px;
    }
    
    .digit-display {
        font-size: 60px; font-weight: 800; 
        background: -webkit-linear-gradient(#ff7b72, #ff5858);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: 15px;
        text-shadow: 0 0 20px rgba(255, 88, 88, 0.3);
    }
    
    .sub-digit {
        font-size: 40px; font-weight: 600; color: #58a6ff;
        text-align: center; letter-spacing: 10px;
    }

    .status-badge {
        display: inline-block; padding: 5px 15px; border-radius: 20px;
        font-size: 14px; font-weight: bold; text-transform: uppercase;
        margin-right: 10px;
    }
    .bg-high { background: rgba(46, 160, 67, 0.2); color: #3fb950; border: 1px solid #3fb950; }
    .bg-med { background: rgba(210, 153, 34, 0.2); color: #d29922; border: 1px solid #d29922; }
    .bg-low { background: rgba(218, 54, 51, 0.2); color: #f85149; border: 1px solid #f85149; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff; margin-bottom: 5px;'>🔮 TITAN v25.0 QUANTUM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; margin-top: 0;'>Hybrid AI & Statistical Probability Engine</p>", unsafe_allow_html=True)

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Cấu hình Thuật toán")
    algo_mode = st.selectbox("Chế độ phân tích", ["Balanced (Cân bằng)", "Aggressive (Tấn công)", "Conservative (An toàn)"])
    st.info("Chế độ Aggressive ưu tiên các số có Gap cao (lâu chưa về).")
    st.divider()
    st.write(f"📦 Database: **{len(st.session_state.history)}** kỳ")
    if st.button("🗑️ Xóa toàn bộ dữ liệu", type="primary"):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# --- MAIN INPUT AREA ---
with st.container():
    col_in, col_act = st.columns([2, 1])
    with col_in:
        raw_input = st.text_area("📡 Nạp dữ liệu lịch sử (Dán bảng kết quả):", height=100, placeholder="Ví dụ:\n32880\n21808\n99213...")
    with col_act:
        st.write("") 
        st.write("") 
        c1, c2 = st.columns(2)
        with c1:
            btn_analyze = st.button("🚀 KÍCH HOẠT TITAN", type="primary", use_container_width=True)
        with c2:
            btn_clear_input = st.button("Xóa khung nhập", use_container_width=True)

if btn_clear_input:
    st.rerun()

# --- LOGIC XỬ LÝ ---
if btn_analyze:
    with st.spinner('🔄 Đang đồng bộ dữ liệu & Chạy thuật toán lượng tử...'):
        # 1. Làm sạch dữ liệu
        new_data = re.findall(r"\b\d{5}\b", raw_input)
        
        if len(new_data) > 0:  # ✅ ĐÃ SỬA: Thêm điều kiện đầy đủ
            # Update history
            current_set = set(st.session_state.history)
            for item in new_data:
                if item not in current_set:
                    st.session_state.history.append(item)
            
            save_db(st.session_state.history)
            
            # 2. Chạy thuật toán thống kê nội bộ (Python)
            analyzer = QuantumAnalyzer(st.session_state.history)
            weighted_scores = analyzer.calculate_weighted_score()
            freq_data = analyzer.get_frequency_analysis(100)
            gap_data = analyzer.get_gap_analysis()
            
            # Top 5 số nóng nhất theo tính toán
            top_5_math = [x[0] for x in weighted_scores[:5]]
            
            # Tạo backup prediction từ thuật toán
            math_prediction = {
                "main_3": "".join(top_5_math[:3]) if len(top_5_math) >= 3 else "000",
                "support_4": "".join(top_5_math[3:7]) if len(top_5_math) > 3 else "0000",
                "confidence": 75,
                "reasoning": "Dựa hoàn toàn trên thuật toán Weighted Score.",
                "warning": "Không có cảnh báo đặc biệt."
            }
            
            # 3. Gửi dữ liệu đã xử lý cho AI (Gemini)
            prompt_data = f"""
            Dữ liệu thống kê Lotobet (100 kỳ gần):
            - Tần suất cao nhất: {freq_data}
            - Số lâu chưa về (Gap): {gap_data[:5]}
            - Top 5 số tiềm năng (Toán học): {top_5_math}
            
            Nhiệm vụ: Dự đoán 3 số chính (Main) và 4 số lót (Support) cho kỳ tiếp theo.
            
            Trả về JSON thuần túy (không markdown):
            {{
                "main_3": "xyz",
                "support_4": "abcd",
                "confidence": 85,
                "reasoning": "Lý do chọn...",
                "warning": "Cảnh báo nếu có"
            }}
            """
            
            try:
                if neural_engine:
                    response = neural_engine.generate_content(prompt_data)
                    text_res = response.text
                    
                    # Clean markdown code blocks
                    if "```json" in text_res:
                        text_res = text_res.split("```json")[1].split("```")[0]
                    elif "```" in text_res:
                        text_res = text_res.split("```")[1].split("```")[0]
                    
                    prediction = json.loads(text_res.strip())
                    st.session_state.last_prediction = prediction
                else:
                    st.session_state.last_prediction = math_prediction
                    st.session_state.last_prediction['reasoning'] += " (AI không khả dụng, dùng backup)"
                    
            except Exception as e:
                # ✅ ĐÃ SỬA: Dùng biến local thay vì session_state chưa khởi tạo
                st.session_state.last_prediction = math_prediction
                st.session_state.last_prediction['reasoning'] += f" (AI Error: {str(e)[:50]})"
            
            st.rerun()
        else:
            st.error("⚠️ Vui lòng nhập dữ liệu hợp lệ (dãy 5 chữ số)")

# --- HIỂN THỊ KẾT QUẢ ---
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    conf = int(res.get('confidence', 50))
    if conf >= 90: color_class = "bg-high"
    elif conf >= 75: color_class = "bg-med"
    else: color_class = "bg-low"
    
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    c_head1, c_head2 = st.columns([3, 1])
    with c_head1:
        st.markdown(f"<span class='status-badge {color_class}'>Độ tin cậy: {conf}%</span>", unsafe_allow_html=True)
        st.markdown(f"🧠 **Logic:** {res.get('reasoning', 'Đang phân tích...')}")
    with c_head2:
        if res.get('warning'):
            st.warning(f"⚠️ {res['warning']}")

    st.divider()
    
    c_num1, c_num2 = st.columns([1, 1])
    with c_num1:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🎯 3 SỐ CHỦ LỰC (MAIN)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='digit-display'>{res.get('main_3', '???')}</div>", unsafe_allow_html=True)
    with c_num2:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🛡️ 4 SỐ LÓT (SUPPORT)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='sub-digit'>{res.get('support_4', '???')}</div>", unsafe_allow_html=True)
        
    st.divider()
    
    full_set = sorted(set(res.get('main_3', '') + res.get('support_4', '')))
    full_str = "".join(full_set)
    
    c_copy, c_chart = st.columns([1, 2])
    with c_copy:
        st.text_input("📋 Dàn 7 số tối ưu:", full_str, label_visibility="collapsed")
    
    with c_chart:
        temp_analyzer = QuantumAnalyzer(st.session_state.history)
        w_scores = temp_analyzer.calculate_weighted_score()
        df_viz = pd.DataFrame(w_scores[:5], columns=['Số', 'Điểm'])
        df_viz['Số'] = df_viz['Số'].astype(str)
        st.bar_chart(df_viz.set_index('Số'), color="#58a6ff")

    st.markdown("</div>", unsafe_allow_html=True)

# --- FOOTER STATISTICS ---
with st.expander("📊 Chi tiết thống kê sâu (Dành cho Pro)"):
    if st.session_state.history:
        temp_analyzer = QuantumAnalyzer(st.session_state.history)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.write("**🔥 Top 5 Số Nóng Nhất (Tần suất cao):**")
            freq = temp_analyzer.get_frequency_analysis(50)
            sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
            for k, v in sorted_freq:
                st.progress(min(v/100, 1.0))
                st.caption(f"Số {k}: {v}%")
        with col_s2:
            st.write("**❄️ Top 5 Số Lạnh Nhất (Gap cao - Sắp về):**")
            gaps = temp_analyzer.get_gap_analysis()[:5]
            for k, v in gaps:
                st.progress(min(v/30, 1.0))
                st.caption(f"Số {k}: Chưa về {v} kỳ")