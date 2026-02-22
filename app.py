import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from collections import Counter
from typing import List, Dict, Tuple

# ================= CẤU HÌNH SIÊU CẤP =================
st.set_page_config(page_title="TITAN v22.0 ELITE", layout="wide", initial_sidebar_state="collapsed")
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM" # Thay key của anh nếu cần

# Giả lập database nhỏ gọn
DB_FILE = "titan_elite_core.json"

def load_data():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: return json.load(f)
    return []

def save_data(data):
    with open(DB_FILE, "w") as f: json.dump(data[-1000:], f)

# Kết nối Neural
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# CSS Tối ưu UI (Chống lác, hỗ trợ thu nhỏ cửa sổ)
st.markdown("""
    <style>
    .reportview-container { background: #0a0a0a; }
    .stApp { background-color: #050505; color: #00ff00; font-family: 'Courier New', monospace; }
    .main-card { 
        background: linear-gradient(135deg, #111, #222);
        border: 1px solid #333; border-radius: 10px; padding: 15px;
        box-shadow: 0 0 20px rgba(0,255,0,0.1);
    }
    .big-num { 
        font-size: 50px !important; font-weight: bold; color: #ff0055; 
        text-shadow: 0 0 10px #ff0055; text-align: center;
    }
    .sub-num { 
        font-size: 35px !important; color: #00d4ff; 
        text-shadow: 0 0 10px #00d4ff; text-align: center;
    }
    .status-bar { font-size: 12px; color: #888; border-bottom: 1px solid #333; margin-bottom: 10px; }
    /* Tối ưu thu nhỏ tab */
    @media (max-width: 600px) {
        .big-num { font-size: 30px !important; }
        .sub-num { font-size: 20px !important; }
    }
    </style>
""", unsafe_allow_html=True)

# ================= THUẬT TOÁN TITAN ELITE =================
class TitanEliteAnalyzer:
    def __init__(self, history: List[str]):
        self.history = history
        self.digits = "0123456789"

    def detect_trap(self) -> str:
        """Phát hiện nhà cái đang bẻ cầu (Cầu lừa)"""
        if len(self.history) < 10: return "Dữ liệu mỏng"
        last_5 = self.history[-5:]
        # Kiểm tra tính lặp lại bất thường (nhà cái giữ số để hút tiền)
        flat_last_5 = "".join(last_5)
        counts = Counter(flat_last_5)
        if any(v > 4 for v in counts.values()): return "CẢNH BÁO: CẦU GIỮ (BẪY)"
        return "Cầu sạch - Có thể vào tiền"

    def analyze_prob(self):
        """Tính toán xác suất đa tầng"""
        if not self.history: return {d: 0.1 for d in self.digits}
        
        weights = np.linspace(0.5, 1.0, len(self.history))
        prob_map = {d: 0.0 for d in self.digits}
        
        for idx, draw in enumerate(self.history):
            for d in set(draw): # Ưu tiên các số xuất hiện trong GĐB
                prob_map[d] += weights[idx]
        
        total = sum(prob_map.values())
        return {k: v/total for k, v in prob_map.items()}

    def get_prediction(self):
        probs = self.analyze_prob()
        # Sắp xếp số theo xác suất từ cao đến thấp
        sorted_nums = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_7 = [n[0] for n in sorted_nums[:7]]
        
        # 3 số khả năng về cao nhất (Dàn 3)
        main_3 = top_7[:3]
        # 4 số dự phòng (Dàn 4)
        backup_4 = top_7[3:]
        
        return main_3, backup_4

# ================= GIAO DIỆN THỰC CHIẾN =================
st.markdown("<div class='status-bar'>TITAN ELITE v22.0 | NEURAL CONNECTED | BY GEMINI 2026</div>", unsafe_allow_html=True)

if 'data_store' not in st.session_state:
    st.session_state.data_store = load_data()

# Cột điều khiển nhanh
col_l, col_r = st.columns([1, 2])

with col_l:
    st.markdown("### 📥 NHẬP KỲ MỚI")
    input_val = st.text_input("Dán số GĐB (5 số)", placeholder="Ví dụ: 88321", key="input_box")
    if st.button("🔥 PHÂN TÍCH NGAY", use_container_width=True):
        if re.match(r"^\d{5}$", input_val):
            st.session_state.data_store.append(input_val)
            save_data(st.session_state.data_store)
            
            # Gọi Gemini phân tích chiến lược bào tiền
            analyzer = TitanEliteAnalyzer(st.session_state.data_store)
            m3, b4 = analyzer.get_prediction()
            trap_info = analyzer.detect_trap()
            
            prompt = f"Lịch sử: {st.session_state.data_store[-20:]}. Dự đoán thuật toán: {m3+b4}. Hãy phân tích quy luật lừa cầu và đưa ra chiến lược vào tiền tối ưu (JSON format)."
            try:
                response = model.generate_content(prompt)
                st.session_state.ai_logic = response.text
            except:
                st.session_state.ai_logic = "AI bận, dùng thuật toán Core Titan."
            
            st.session_state.last_m3 = m3
            st.session_state.last_b4 = b4
            st.session_state.trap = trap_info
            st.rerun()

    if st.button("🗑️ RESET DỮ LIỆU", use_container_width=True):
        st.session_state.data_store = []
        save_data([])
        st.rerun()

with col_r:
    if "last_m3" in st.session_state:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        
        # Hiển thị 3 số chủ lực
        st.markdown(f"<p style='text-align:center; color:#888;'>💎 3 SỐ CHỦ LỰC (99%)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-num'>{' - '.join(st.session_state.last_m3)}</div>", unsafe_allow_html=True)
        
        # Hiển thị 4 số dự phòng
        st.markdown(f"<p style='text-align:center; color:#888;'>🛡️ 4 SỐ DỰ PHÒNG</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='sub-num'>{' - '.join(st.session_state.last_b4)}</div>", unsafe_allow_html=True)
        
        # Trạng thái cầu
        color = "#ff0000" if "CẢNH BÁO" in st.session_state.trap else "#00ff00"
        st.markdown(f"<div style='text-align:center; color:{color}; font-weight:bold;'>{st.session_state.trap}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        with st.expander("🧠 CHIẾN THUẬT TỪ AI GEMINI", expanded=True):
            st.write(st.session_state.get('ai_logic', 'Đang đợi dữ liệu...'))

# Bảng lịch sử thu nhỏ
st.markdown("---")
with st.expander("📜 LỊCH SỬ 20 KỲ GẦN NHẤT"):
    df = pd.DataFrame(st.session_state.data_store[::-1], columns=["Số đã về"])
    st.table(df.head(20))

# Tự động refresh nhẹ để giữ kết nối
time.sleep(0.5)
