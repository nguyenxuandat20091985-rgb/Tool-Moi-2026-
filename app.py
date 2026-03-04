# ==============================================================================
# TITAN v35.0 PRO MAX - 5D Bet Prediction System
# Professional Real-Time Lottery Analysis
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

# Import custom modules
try:
    from algorithms import PredictionEngine
    from database import DatabaseManager
    from config import Config
except ImportError:
    # Fallback class nếu thiếu file hệ thống
    class DatabaseManager:
        def clean_data(self, text): return re.findall(r'\d{5}', text)
        def add_numbers(self, new, db):
            added = 0
            for n in new:
                if n not in db: db.append(n); added += 1
            return added
    class PredictionEngine:
        def __init__(self): self.weights = {}
        def predict(self, db): return {'main_3': ['1','2','3'], 'support_4': ['4','5','6','7'], 'confidence': 70, 'algorithm': 'Ensemble', 'logic': 'Dữ liệu mẫu'}
        def calculate_risk(self, db): return (30, "LOW", ["Dữ liệu ổn định"])
        def detect_patterns(self, db): return {'detected': []}

# ==============================================================================
# 1. PAGE CONFIG & CSS
# ==============================================================================

st.set_page_config(page_title="TITAN v35.0 PRO MAX", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #010409 0%, #0d1117 100%); color: #e6edf3; }
    .status-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin: 10px 0; }
    .status-green { border-left: 4px solid #238636; }
    .status-red { border-left: 4px solid #da3633; }
    .status-yellow { border-left: 4px solid #d29922; }
    .num-card { background: #0d1117; border: 2px solid #ff5858; border-radius: 15px; padding: 20px; text-align: center; }
    .num-value { font-size: 50px; font-weight: 900; color: #ff5858; }
    .metric-box { background: #0d1117; border: 1px solid #30363d; border-radius: 10px; padding: 15px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CORE FUNCTIONS
# ==============================================================================

def init_session_state():
    if "db_manager" not in st.session_state: st.session_state.db_manager = DatabaseManager()
    if "prediction_engine" not in st.session_state: st.session_state.prediction_engine = PredictionEngine()
    if "lottery_db" not in st.session_state: st.session_state.lottery_db = []
    if "predictions_log" not in st.session_state: st.session_state.predictions_log = []
    if "bankroll" not in st.session_state:
        st.session_state.bankroll = {"initial": 1000000, "current": 1000000, "bet_per_round": 10000}
    if "last_prediction" not in st.session_state: st.session_state.last_prediction = None
    if "last_risk" not in st.session_state: st.session_state.last_risk = (0, "NEUTRAL", [])
    if "last_update" not in st.session_state: st.session_state.last_update = None

def format_currency(amount):
    try:
        val = float(amount)
        return f"₫{val:,.0f}"
    except:
        return "₫0"

def calculate_win_rate():
    logs = st.session_state.predictions_log
    decided = [l for l in logs if l.get('won') is not None]
    if not decided: return 0, 0, 0
    wins = sum(1 for l in decided if l.get('won') is True)
    total = len(decided)
    return wins, total, round((wins / total * 100), 2)

# ==============================================================================
# 3. MAIN APPLICATION
# ==============================================================================

def main():
    init_session_state()
    st.title("🎯 TITAN v35.0 PRO MAX")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Dự Đoán", "📊 Phân Tích", "💰 Quản Lý Vốn", "⚙️ Cài Đặt"])

    # ---------------- TAB 1: PREDICTION ----------------
    with tab1:
        st.header("🎯 Hệ Thống Dự Đoán")
        input_text = st.text_area("Nhập dữ liệu (5 số mỗi dòng):", height=150, placeholder="87746\n56421...")
        
        if st.button("🚀 PHÂN TÍCH NGAY", type="primary"):
            if input_text:
                nums = st.session_state.db_manager.clean_data(input_text)
                st.session_state.db_manager.add_numbers(nums, st.session_state.lottery_db)
                
                if len(st.session_state.lottery_db) >= 5:
                    pred = st.session_state.prediction_engine.predict(st.session_state.lottery_db)
                    risk = st.session_state.prediction_engine.calculate_risk(st.session_state.lottery_db)
                    
                    st.session_state.last_prediction = pred
                    st.session_state.last_risk = risk
                    st.session_state.last_update = datetime.now()
                    st.session_state.predictions_log.append({'timestamp': datetime.now().isoformat(), 'prediction': pred, 'won': None})
                    st.rerun()
                else:
                    st.warning("Cần tối thiểu 5 kỳ dữ liệu.")

        if st.session_state.last_prediction:
            p = st.session_state.last_prediction
            r_score, r_level, _ = st.session_state.last_risk
            
            color = "green" if r_level == "LOW" else "red" if r_level == "HIGH" else "yellow"
            st.markdown(f'<div class="status-card status-{color}"><b>RISK: {r_score}/100 | KHUYẾN NGHỊ: {r_level}</b></div>', unsafe_allow_html=True)
            
            cols = st.columns(3)
            for i, num in enumerate(p.get('main_3', ['?','?','?'])):
                cols[i].markdown(f'<div class="num-card"><div class="num-value">{num}</div><div>SỐ {i+1}</div></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            res_check = st.text_input("Nhập kết quả thực tế để kiểm tra (5 số):")
            if st.button("✅ Xác nhận kết quả"):
                if len(res_check) == 5:
                    is_win = any(digit in res_check for digit in p['main_3'])
                    st.session_state.predictions_log[-1]['won'] = is_win
                    if is_win:
                        st.session_state.bankroll['current'] += st.session_state.bankroll['bet_per_round'] * 0.9 # Ví dụ lãi 90%
                        st.success("CHÚC MỪNG! BẠN ĐÃ THẮNG.")
                    else:
                        st.session_state.bankroll['current'] -= st.session_state.bankroll['bet_per_round']
                        st.error("RẤT TIẾC! HÃY THỬ LẠI KỲ SAU.")
                    st.rerun()

    # ---------------- TAB 3: BANKROLL (ĐÃ FIX LỖI 803) ----------------
    with tab3:
        st.header("💰 Quản Lý Vốn Chiến Thuật")
        
        # Hiển thị Metric vốn
        curr_bankroll = float(st.session_state.bankroll.get('current', 0))
        c1, c2, c3 = st.columns(3)
        c1.metric("Vốn Hiện Tại", format_currency(curr_bankroll))
        wins, total, rate = calculate_win_rate()
        c2.metric("Tỷ Lệ Thắng", f"{rate}%")
        c3.metric("Số Kỳ Đã Chơi", total)

        st.markdown("---")
        st.markdown("### 📐 Công Thức Kelly Optimizer")
        
        k_win_rate = st.slider("Xác suất thắng dự đoán (%)", 5, 95, 35)
        k_odds = st.number_input("Tỷ lệ trả thưởng (Odds)", value=1.9, min_value=1.1, step=0.1)
        
        # XỬ LÝ LỖI TOÁN HỌC & FORMAT TẠI ĐÂY
        try:
            p = float(k_win_rate) / 100.0
            b = float(k_odds) - 1.0
            
            if b > 0:
                q = 1.0 - p
                kelly_f = (b * p - q) / b
                kelly_percent = max(0.0, kelly_f * 100.0)
                suggested_bet = curr_bankroll * (kelly_percent / 100.0)
            else:
                kelly_percent = 0.0
                suggested_bet = 0.0

            # Tạo chuỗi hiển thị an toàn
            txt_kelly = f"{kelly_percent:.2f}%"
            txt_bet = format_currency(suggested_bet)

            st.info(f"""
            **Tỷ lệ cược Kelly tối ưu:** {txt_kelly} vốn
            
            **Tiền cược khuyến nghị:** {txt_bet} mỗi kỳ
            
            *Lưu ý: Nếu số âm hoặc 0%, hệ thống khuyên bạn nên dừng lại.*
            """)
        except Exception as e:
            st.error("Không thể tính toán Kelly. Vui lòng kiểm tra lại tỷ lệ cược.")

    # ---------------- TAB 4: SETTINGS ----------------
    with tab4:
        st.header("⚙️ Cài Đặt")
        if st.button("🗑️ XÓA TOÀN BỘ DỮ LIỆU"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
