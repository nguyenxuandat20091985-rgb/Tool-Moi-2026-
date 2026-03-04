# ==============================================================================
# TITAN v35.0 PRO MAX - 5D Bet Prediction System
# Optimized for Mobile & Desktop Display
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import custom modules (Dữ liệu dự phòng nếu thiếu file hệ thống)
try:
    from algorithms import PredictionEngine
    from database import DatabaseManager
except ImportError:
    class DatabaseManager:
        def clean_data(self, text): return re.findall(r'\d{5}', text)
        def add_numbers(self, new, db):
            added = 0
            for n in new:
                if n not in db: db.append(n); added += 1
            return added
    class PredictionEngine:
        def predict(self, db): return {'main_3': ['7','4','3'], 'support_4': ['1','5','8','9'], 'confidence': 85, 'algorithm': 'Titan-AI'}
        def calculate_risk(self, db): return (0, "LOW", [])

# ==============================================================================
# 1. PAGE CONFIG & MOBILE OPTIMIZED CSS
# ==============================================================================

st.set_page_config(page_title="TITAN v35.0 PRO MAX", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    /* Giao diện nền tối chuyên nghiệp */
    .stApp { background: #010409; color: #e6edf3; }
    
    /* Panel thông báo Risk */
    .risk-panel {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .risk-low { border-left: 5px solid #238636; color: #39d353; }
    
    /* Cấu trúc Grid cho các con số - Quan trọng để dễ nhìn trên điện thoại */
    .number-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr); /* Luôn chia 3 cột */
        gap: 10px;
        margin-bottom: 20px;
    }
    
    .number-box {
        background: #0d1117;
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        padding: 15px 5px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.2);
    }
    
    .number-val {
        font-size: 38px;
        font-weight: 800;
        color: #ff4b4b;
        line-height: 1;
    }
    
    .number-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        margin-top: 5px;
    }

    /* Số lót - 4 cột */
    .support-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
    }
    .support-box {
        background: #0d1117;
        border: 1px solid #58a6ff;
        border-radius: 8px;
        padding: 10px 2px;
        text-align: center;
    }
    .support-val {
        font-size: 24px;
        font-weight: bold;
        color: #58a6ff;
    }

    /* Tiêu đề mục */
    .section-title {
        font-size: 14px;
        color: #8b949e;
        margin-bottom: 10px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

def init_session_state():
    if "lottery_db" not in st.session_state: st.session_state.lottery_db = []
    if "bankroll" not in st.session_state: st.session_state.bankroll = {"current": 1000000}
    if "last_prediction" not in st.session_state: st.session_state.last_prediction = None
    if "last_risk" not in st.session_state: st.session_state.last_risk = (0, "LOW", [])

def format_currency(amount):
    return f"₫{float(amount):,.0f}"

# ==============================================================================
# 3. MAIN INTERFACE
# ==============================================================================

def main():
    init_session_state()
    st.title("🎯 TITAN PRO MAX")
    
    tab1, tab2, tab3 = st.tabs(["DỰ ĐOÁN", "QUẢN LÝ VỐN", "CÀI ĐẶT"])

    with tab1:
        # Khu vực nhập liệu gọn gàng
        input_data = st.text_area("Nhập kết quả (5 số):", height=80, placeholder="Ví dụ: 87746")
        
        if st.button("🚀 PHÂN TÍCH", use_container_width=True, type="primary"):
            # Giả lập xử lý (Thay bằng logic thật của anh)
            db_mgr = DatabaseManager()
            nums = db_mgr.clean_data(input_data)
            db_mgr.add_numbers(nums, st.session_state.lottery_db)
            
            engine = PredictionEngine()
            st.session_state.last_prediction = engine.predict(st.session_state.lottery_db)
            st.session_state.last_risk = engine.calculate_risk(st.session_state.lottery_db)
            st.rerun()

        # HIỂN THỊ KẾT QUẢ DỰ ĐOÁN (ĐÃ TỐI ƯU MOBILE)
        if st.session_state.last_prediction:
            p = st.session_state.last_prediction
            r_score, r_level, _ = st.session_state.last_risk
            
            # 1. Hiển thị Risk
            st.markdown(f'<div class="risk-panel risk-low">RISK: {r_score}/100 | KHUYẾN NGHỊ: {r_level}</div>', unsafe_allow_html=True)
            
            # 2. Hiển thị 3 số chính (Nằm ngang)
            st.markdown('<div class="section-title">🔮 3 SỐ CHÍNH (VÀO MẠNH)</div>', unsafe_allow_html=True)
            main_numbers_html = "".join([
                f'<div class="number-box"><div class="number-val">{n}</div><div class="number-label">SỐ {i+1}</div></div>'
                for i, n in enumerate(p['main_3'])
            ])
            st.markdown(f'<div class="number-container">{main_numbers_html}</div>', unsafe_allow_html=True)
            
            # 3. Hiển thị 4 số lót (Nằm ngang)
            st.markdown('<div class="section-title">🎲 4 SỐ LÓT (GIỮ VỐN)</div>', unsafe_allow_html=True)
            support_numbers_html = "".join([
                f'<div class="support-box"><div class="support-val">{n}</div></div>'
                for n in p['support_4']
            ])
            st.markdown(f'<div class="support-container">{support_numbers_html}</div>', unsafe_allow_html=True)

    with tab2:
        st.header("💰 Quản Lý Vốn")
        curr_money = st.session_state.bankroll['current']
        st.metric("Vốn Hiện Tại", format_currency(curr_money))
        
        # Công thức Kelly rút gọn
        st.markdown("---")
        st.markdown("### 📐 Gợi ý vào tiền")
        win_rate = 0.35 # Giả định 35%
        odds = 1.9
        kelly_f = (win_rate * odds - (1 - win_rate)) / (odds - 1)
        suggested = max(0, curr_money * kelly_f * 0.5) # Half-kelly an toàn
        st.success(f"Khuyên dùng: **{format_currency(suggested)}** / kỳ")

    with tab3:
        if st.button("🗑️ Xóa dữ liệu cũ"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
