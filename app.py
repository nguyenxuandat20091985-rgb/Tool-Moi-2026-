# ==============================================================================
# TITAN v35.0 PRO MAX - 5D Bet Prediction System
# Professional Real-Time Lottery Analysis
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import custom modules (Đảm bảo anh đã có các file này trong cùng thư mục)
try:
    from algorithms import PredictionEngine
    from database import DatabaseManager
    from config import Config
except ImportError:
    # Fallback nếu thiếu file để tránh sập app hoàn toàn
    class DatabaseManager:
        def clean_data(self, text): return re.findall(r'\d{5}', text)
        def add_numbers(self, new, db):
            added = 0
            for n in new:
                if n not in db: db.append(n); added += 1
            return added
    class PredictionEngine:
        def __init__(self): self.weights = {}
        def predict(self, db): return {'main_3': ['1','2','3'], 'support_4': ['4','5','6','7'], 'confidence': 70, 'algorithm': 'Demo', 'logic': 'Dữ liệu mẫu'}
        def calculate_risk(self, db): return (30, "LOW", [])
        def detect_patterns(self, db): return {'detected': []}

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="TITAN v35.0 PRO MAX | 5D Bet",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CUSTOM CSS - PROFESSIONAL THEME
# ==============================================================================

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #010409 0%, #0d1117 100%); color: #e6edf3; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .live-indicator { display: inline-block; width: 12px; height: 12px; background: #238636; border-radius: 50%; margin-right: 8px; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { opacity: 1; box-shadow: 0 0 0 0 rgba(35,134,54,0.7); } 70% { opacity: 0.5; box-shadow: 0 0 0 10px rgba(35,134,54,0); } 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(35,134,54,0); } }
    .status-card { background: linear-gradient(135deg, #0d1117, #161b22); border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
    .status-green { border-left: 4px solid #238636; background: linear-gradient(135deg, rgba(35,134,54,0.1), #0d1117); }
    .status-red { border-left: 4px solid #da3633; background: linear-gradient(135deg, rgba(218,54,51,0.1), #0d1117); }
    .status-yellow { border-left: 4px solid #d29922; background: linear-gradient(135deg, rgba(210,153,34,0.1), #0d1117); }
    .numbers-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
    .num-card { background: linear-gradient(135deg, #161b22, #0d1117); border: 3px solid #ff5858; border-radius: 15px; padding: 25px; text-align: center; box-shadow: 0 8px 30px rgba(255,88,88,0.3); }
    .num-value { font-size: 65px; font-weight: 900; color: #ff5858; text-shadow: 0 0 20px rgba(255,88,88,0.8); }
    .num-label { font-size: 14px; color: #8b949e; margin-top: 10px; text-transform: uppercase; }
    .lot-card { background: linear-gradient(135deg, #161b22, #0d1117); border: 2px solid #58a6ff; border-radius: 12px; padding: 18px; text-align: center; }
    .lot-value { font-size: 45px; font-weight: 800; color: #58a6ff; }
    .metric-box { background: #0d1117; border: 1px solid #30363d; border-radius: 10px; padding: 15px; text-align: center; }
    .metric-value { font-size: 32px; font-weight: 700; color: #58a6ff; }
    .metric-label { font-size: 12px; color: #8b949e; text-transform: uppercase; margin-top: 5px; }
    .stButton > button { background: linear-gradient(135deg, #238636, #2ea043); color: white; border: none; border-radius: 10px; font-weight: 700; padding: 15px 35px; width: 100%; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. UTILITIES & SESSION STATE
# ==============================================================================

def init_session_state():
    if "db_manager" not in st.session_state: st.session_state.db_manager = DatabaseManager()
    if "prediction_engine" not in st.session_state: st.session_state.prediction_engine = PredictionEngine()
    if "lottery_db" not in st.session_state: st.session_state.lottery_db = []
    if "predictions_log" not in st.session_state: st.session_state.predictions_log = []
    if "bankroll" not in st.session_state:
        st.session_state.bankroll = {"initial": 1000000, "current": 1000000, "bet_per_round": 10000, "sessions": []}
    if "last_prediction" not in st.session_state: st.session_state.last_prediction = None
    if "last_risk" not in st.session_state: st.session_state.last_risk = (0, "NEUTRAL", [])
    if "last_update" not in st.session_state: st.session_state.last_update = None
    if "auto_refresh" not in st.session_state: st.session_state.auto_refresh = False

def format_currency(amount):
    try:
        return f"₫{float(amount):,.0f}"
    except:
        return "₫0"

def format_datetime(dt):
    if not dt: return "N/A"
    if isinstance(dt, str): dt = datetime.fromisoformat(dt)
    return dt.strftime("%H:%M:%S %d/%m/%Y")

def calculate_win_rate(logs):
    decided = [l for l in logs if l.get('won') is not None]
    if not decided: return 0, 0, 0
    wins = sum(1 for l in decided if l.get('won') is True)
    total = len(decided)
    return wins, total, round((wins / total * 100), 2)

def check_prediction_win(prediction_list, result_str):
    if not prediction_list or not result_str: return False
    res_digits = list(result_str)
    matches = sum(1 for d in prediction_list if d in res_digits)
    return matches >= 3

# ==============================================================================
# 4. UI COMPONENTS (SỬA LỖI FORMAT TẠI ĐÂY)
# ==============================================================================

def render_prediction_display(prediction, risk_info):
    if not prediction: return
    risk_score, risk_level, _ = risk_info
    main_3 = prediction.get('main_3', ['?', '?', '?'])
    support_4 = prediction.get('support_4', ['?', '?', '?', '?'])
    
    color = "green" if risk_level == "LOW" else "red" if risk_level == "HIGH" else "yellow"
    st.markdown(f'<div class="status-card status-{color}" style="text-align:center"><b>RISK: {risk_score}/100 | KHUYẾN NGHỊ: {risk_level}</b></div>', unsafe_allow_html=True)
    
    cols = st.columns(3)
    for i in range(3):
        with cols[i]:
            st.markdown(f'<div class="num-card"><div class="num-value">{main_3[i]}</div><div class="num-label">SỐ {i+1}</div></div>', unsafe_allow_html=True)

# ==============================================================================
# 5. MAIN APP
# ==============================================================================

def main():
    init_session_state()
    st.title("🎯 TITAN v35.0 PRO MAX")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Dự Đoán", "📊 Phân Tích", "💰 Vốn", "⚙️ Cài Đặt"])

    with tab1:
        st.header("🎯 Dự Đoán 5D")
        input_data = st.text_area("Nhập dữ liệu 5 số (mỗi dòng 1 kỳ):", height=150)
        
        if st.button("🚀 PHÂN TÍCH HỆ THỐNG"):
            if input_data:
                nums = st.session_state.db_manager.clean_data(input_data)
                st.session_state.db_manager.add_numbers(nums, st.session_state.lottery_db)
                
                pred = st.session_state.prediction_engine.predict(st.session_state.lottery_db)
                risk = st.session_state.prediction_engine.calculate_risk(st.session_state.lottery_db)
                
                st.session_state.last_prediction = pred
                st.session_state.last_risk = risk
                st.session_state.last_update = datetime.now()
                
                # Log sơ bộ
                st.session_state.predictions_log.append({'timestamp': datetime.now().isoformat(), 'prediction': pred, 'won': None})
                st.rerun()

        if st.session_state.last_prediction:
            render_prediction_display(st.session_state.last_prediction, st.session_state.last_risk)
            
            st.markdown("---")
            res_input = st.text_input("Xác nhận kết quả kỳ vừa rồi (5 số):")
            if st.button("✅ Lưu kết quả"):
                if len(res_input) == 5:
                    is_win = check_prediction_win(st.session_state.last_prediction['main_3'], res_input)
                    st.session_state.predictions_log[-1]['won'] = is_win
                    if is_win:
                        st.session_state.bankroll['current'] += st.session_state.bankroll['bet_per_round'] * 1.9
                        st.balloons()
                    else:
                        st.session_state.bankroll['current'] -= st.session_state.bankroll['bet_per_round']
                    st.rerun()

    with tab3:
        st.header("💰 Quản Lý Vốn")
        c1, c2, c3 = st.columns(3)
        c1.metric("Vốn hiện tại", format_currency(st.session_state.bankroll['current']))
        wins, total, rate = calculate_win_rate(st.session_state.predictions_log)
        c2.metric("Win Rate", f"{rate}%")
        c3.metric("Tổng cược", total)

        st.markdown("### 📐 Kelly Criterion (Công thức tối ưu)")
        k_win = st.slider("Tỉ lệ thắng dự kiến (%)", 10, 90, 35)
        k_odds = st.number_input("Tỉ lệ ăn (ví dụ 1 ăn 2.0 thì nhập 2.0)", value=1.9)
        
        # --- ĐOẠN SỬA LỖI CHÍNH ---
        p = k_win / 100
        b = k_odds - 1
        kelly_f = (p * b - (1 - p)) / b if b > 0 else 0
        kelly_perc = max(0, kelly_f * 100)
        
        # Tính số tiền trước khi đưa vào f-string
        suggested_bet = st.session_state.bankroll['current'] * (kelly_perc / 100)
        
        st.info(f"""
        **Tỷ lệ cược Kelly:** {kelly_perc:.2f}% vốn.
        **Số tiền khuyến nghị:** {format_currency(suggested_bet)} mỗi kỳ.
        """)
        # --------------------------

    with tab4:
        if st.button("🗑️ Reset Dữ Liệu"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
