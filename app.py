import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# ================= CẤU HÌNH HỆ THỐNG TITAN v26.0 OMNI =================
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc" 
DB_FILE = "titan_omni_v26.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= THUẬT TOÁN PHÂN TÍCH SIÊU CẤP =================

class OmniAnalyzer:
    def __init__(self, history):
        self.history = history
        self.matrix = self._build_transition_matrix()

    def _build_transition_matrix(self):
        # Thuật toán Markov: Dự đoán số tiếp theo dựa trên chuỗi lịch sử
        transitions = defaultdict(lambda: defaultdict(int))
        all_digits = "".join(self.history)
        for i in range(len(all_digits) - 1):
            transitions[all_digits[i]][all_digits[i+1]] += 1
        return transitions

    def predict_next_digits(self):
        if not self.history: return []
        last_digit = self.history[-1][-1]
        next_possible = self.matrix[last_digit]
        sorted_next = sorted(next_possible.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_next]

    def analyze_patterns(self):
        # Nhận diện cầu bệt và cầu đảo
        all_nums = "".join(self.history[-15:])
        counts = Counter(all_nums)
        is_bet = any(v >= 6 for v in counts.values())
        
        sums = [sum(int(d) for d in s) for s in self.history[-10:]]
        is_shuffling = np.std(sums) > 8
        
        return is_bet, is_shuffling

# ================= QUẢN LÝ DỮ LIỆU BỀN VỮNG =================

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            try: return json.load(f)
            except: return []
    return []

def save_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data[-5000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= GIAO DIỆN SUPREME UI =================

st.set_page_config(page_title="TITAN v26.0 OMNI", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: linear-gradient(145deg, #0d1117, #161b22);
        border: 2px solid #30363d; border-radius: 20px; padding: 35px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    }
    .main-num {
        font-size: 100px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 15px;
        text-shadow: 0 0 30px rgba(255,88,88,0.4);
    }
    .supp-num {
        font-size: 60px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 10px;
    }
    .status-msg { padding: 15px; border-radius: 12px; font-weight: bold; text-align: center; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🚀 TITAN v26.0 OMNI-REVOLUTION</h1>", unsafe_allow_html=True)

# --- KHU VỰC NHẬP LIỆU ---
col_in, col_st = st.columns([2, 1])
with col_in:
    raw_input = st.text_area("📡 NẠP DỮ LIỆU (v26 Tự động lọc trùng/sai):", height=120)
with col_st:
    st.write(f"📂 Cơ sở dữ liệu: **{len(st.session_state.history)}** kỳ")
    c1, c2 = st.columns(2)
    btn_run = c1.button("🚀 GIẢI MÃ", type="primary", use_container_width=True)
    if c2.button("🗑️ RESET", use_container_width=True):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

# --- LÕI XỬ LÝ SIÊU TRÍ TUỆ ---
if btn_run:
    new_data = re.findall(r"\b\d{5}\b", raw_input)
    if new_data:
        st.session_state.history.extend(new_data)
        st.session_state.history = list(dict.fromkeys(st.session_state.history))
        save_db(st.session_state.history)

        analyzer = OmniAnalyzer(st.session_state.history)
        is_bet, is_shuffling = analyzer.analyze_patterns()
        markov_preds = analyzer.predict_next_digits()

        # Chuẩn bị Prompt siêu cấp cho Gemini
        prompt = f"""
        Bạn là TITAN v26.0 OMNI. Phân tích dữ liệu 5D Bet.
        Lịch sử: {st.session_state.history[-100:]}
        Gợi ý từ ma trận Markov: {markov_preds[:5]}
        Tình trạng: Bệt={is_bet}, Đảo cầu={is_shuffling}
        
        NHIỆM VỤ:
        1. Dự đoán 3 số chính (main_3) và 4 số lót (support_4).
        2. Phân tích rõ 'NÊN ĐÁNH' (Green) hay 'DỪNG' (Red) dựa trên nhịp cầu.
        
        TRẢ VỀ JSON:
        {{
            "main_3": "xyz", "support_4": "abcd", "logic": "...", "status": "Green/Red/Yellow", "conf": 98
        }}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            res_json = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            st.session_state.last_res = res_json
        except:
            # Thuật toán dự phòng Markov + Frequency
            st.session_state.last_res = {
                "main_3": "".join(markov_preds[:3]),
                "support_4": "".join(markov_preds[3:7]),
                "logic": "Sử dụng Ma trận Markov dự phòng.",
                "status": "Yellow", "conf": 75
            }
        st.rerun()

# --- HIỂN THỊ KẾT QUẢ ĐẲNG CẤP ---
if "last_res" in st.session_state:
    res = st.session_state.last_res
    
    color_map = {"Green": "#238636", "Red": "#da3633", "Yellow": "#d29922"}
    bg_color = color_map.get(res['status'], "#30363d")
    
    st.markdown(f"<div class='status-msg' style='background: {bg_color}33; border: 1px solid {bg_color}; color: {bg_color};'>CHỈ THỊ: {res['status']} | ĐỘ TIN CẬY: {res['conf']}%</div>", unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns([1.5, 1])
    with col_a:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🔥 3 SỐ CHỦ LỰC (OMNI-MAIN)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-num'>{res['main_3']}</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown("<p style='text-align:center; color:#8b949e;'>🛡️ 4 SỐ LÓT (OMNI-SUPPORT)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='supp-num'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.write(f"🧠 **PHÂN TÍCH CHUYÊN SÂU:** {res['logic']}")
    
    full_dan = "".join(sorted(set(res['main_3'] + res['support_4'])))
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ:", full_dan)
    st.markdown("</div>", unsafe_allow_html=True)

# Thống kê trực quan
if st.session_state.history:
    with st.expander("📊 Biểu đồ Ma trận Tần suất Lượng tử"):
        st.write("Dưới đây là xác suất di chuyển của các con số dựa trên Ma trận Markov:")
        
        all_digits = "".join(st.session_state.history[-100:])
        st.bar_chart(pd.Series(Counter(all_digits)).sort_index())
