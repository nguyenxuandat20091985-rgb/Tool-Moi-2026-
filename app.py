# ==============================================================================
# TITAN v37.0 ULTRA AI - Main Application
# Synchronized with Multi-Layer Engine & Monte Carlo
# ==============================================================================

import streamlit as st
import pandas as pd
from datetime import datetime
import re
import time
import json
import random

# Import the AI Engine v37.0
from algorithms import PredictionEngine

# ==============================================================================
# 1. PAGE CONFIG & ENHANCED CSS
# ==============================================================================

st.set_page_config(
    page_title="🎯 TITAN v37.0 ULTRA AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #010409 0%, #0d1117 100%);
        color: #e6edf3;
    }
    
    /* Number Grid Optimization */
    .number-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 15px 0;
    }
    
    .number-box {
        background: rgba(22, 27, 34, 0.8);
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        padding: 15px 5px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255,75,75,0.2);
    }
    
    .number-val {
        font-size: 38px;
        font-weight: 900;
        color: #ff4b4b;
    }
    
    .support-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
    }
    
    .support-box {
        background: #0d1117;
        border: 1px solid #58a6ff;
        border-radius: 8px;
        padding: 10px 5px;
        text-align: center;
        color: #58a6ff;
        font-weight: bold;
        font-size: 24px;
    }

    /* Risk Metrics Styling */
    .risk-box {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
    }
    .risk-low { background: rgba(35, 134, 54, 0.2); border: 1px solid #238636; color: #39d353; }
    .risk-medium { background: rgba(210, 153, 34, 0.2); border: 1px solid #d29922; color: #f0b429; }
    .risk-high { background: rgba(218, 54, 51, 0.2); border: 1px solid #da3633; color: #f85149; }

    /* AI Stats Sidebar */
    .weight-item {
        background: #161b22;
        border-radius: 5px;
        padding: 5px 10px;
        margin-bottom: 5px;
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION & MEMORY SYNC
# ==============================================================================

def init_session():
    if "ai_engine" not in st.session_state:
        st.session_state.ai_engine = PredictionEngine()
    if "lottery_db" not in st.session_state:
        st.session_state.lottery_db = []
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "predictions_log" not in st.session_state:
        st.session_state.predictions_log = []
    if "bankroll" not in st.session_state:
        st.session_state.bankroll = {"initial": 1000000, "current": 1000000}

def format_money(val):
    return f"₫{val:,.0f}"

# ==============================================================================
# 3. MAIN INTERFACE
# ==============================================================================

def main():
    init_session()
    
    # Sidebar Info
    with st.sidebar:
        st.title("🧠 AI Status")
        weights = st.session_state.ai_engine.weights
        for algo, w in weights.items():
            st.markdown(f"""<div class="weight-item"><span>{algo.upper()}</span><b>{w}%</b></div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        if st.button("🗑️ Reset Hệ Thống"):
            st.session_state.clear()
            st.rerun()

    st.title("🎯 TITAN v37.0 ULTRA")
    
    tab1, tab2 = st.tabs(["🚀 PHÂN TÍCH AI", "📊 NHẬT KÝ & VỐN"])

    with tab1:
        # Input Area
        input_data = st.text_area("Nhập kết quả (5 số/dòng):", height=100, placeholder="87746\n56421...")
        
        if st.button("🚀 KÍCH HOẠT AI", use_container_width=True, type="primary"):
            new_nums = re.findall(r'\d{5}', input_data)
            if new_nums:
                # Cập nhật DB (loại bỏ trùng lặp)
                for n in new_nums:
                    if n not in st.session_state.lottery_db:
                        st.session_state.lottery_db.append(n)
                
                # Chạy AI v37
                with st.spinner("🧠 Đang giả lập Monte Carlo..."):
                    pred = st.session_state.ai_engine.predict(st.session_state.lottery_db)
                    st.session_state.last_prediction = pred
                    st.rerun()
            else:
                st.warning("Vui lòng nhập đúng định dạng 5 chữ số.")

        # Hiển thị Kết quả đồng bộ v37
        if st.session_state.last_prediction:
            p = st.session_state.last_prediction
            risk = p['risk_metrics']
            
            # 1. Risk Display
            r_class = f"risk-{risk['level'].lower()}"
            st.markdown(f"""
            <div class="risk-box {r_class}">
                MỨC ĐỘ RỦI RO: {risk['score']}/100 - {risk['level']}<br>
                <small>{' | '.join(risk['reasons']) if risk['reasons'] else 'Hệ thống ổn định'}</small>
            </div>
            """, unsafe_allow_html=True)

            # 
            
            # 2. Main 3 Numbers
            st.write("🔮 **3 SỐ CHÍNH (XÁC SUẤT CAO)**")
            main_html = "".join([f'<div class="number-box"><div class="number-val">{n}</div></div>' for n in p['main_3']])
            st.markdown(f'<div class="number-container">{main_html}</div>', unsafe_allow_html=True)

            # 3. Support 4 Numbers
            st.write("🎲 **4 SỐ LÓT (GIỮ VỐN)**")
            sup_html = "".join([f'<div class="support-box">{n}</div>' for n in p['support_4']])
            st.markdown(f'<div class="support-container">{sup_html}</div>', unsafe_allow_html=True)
            
            st.caption(f"💡 AI Logic: {p['logic']} | Confidence: {p['confidence']}%")

            # 4. Xác nhận kết quả (Dạy AI)
            st.markdown("---")
            col_a, col_b = st.columns([2, 1])
            actual = col_a.text_input("Nhập kết quả thực tế để dạy AI:", max_chars=5)
            if col_b.button("✅ XÁC NHẬN", use_container_width=True):
                if len(actual) == 5:
                    # Kiểm tra thắng (trúng ít nhất 1 trong 3 số chính)
                    is_win = any(d in actual for d in p['main_3'])
                    
                    # Cập nhật bộ nhớ AI
                    st.session_state.ai_engine.update_weights(is_win)
                    
                    # Cập nhật Vốn
                    bet = 10000 # Mặc định
                    if is_win:
                        st.session_state.bankroll['current'] += bet * 2
                        st.balloons()
                    else:
                        st.session_state.bankroll['current'] -= bet
                    
                    # Lưu Log
                    st.session_state.predictions_log.insert(0, {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "pred": ",".join(p['main_3']),
                        "actual": actual,
                        "status": "THẮNG" if is_win else "THUA"
                    })
                    st.rerun()

    with tab2:
        c1, c2 = st.columns(2)
        c1.metric("Vốn Hiện Tại", format_money(st.session_state.bankroll['current']))
        
        # Thống kê thắng thua
        if st.session_state.predictions_log:
            df = pd.DataFrame(st.session_state.predictions_log)
            st.table(df)
        else:
            st.info("Chưa có lịch sử dự đoán.")

if __name__ == "__main__":
    main()
