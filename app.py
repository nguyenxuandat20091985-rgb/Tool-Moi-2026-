# ==============================================================================
# TITAN v35.0 PRO MAX - 5D Bet Prediction System
# FIXED VERSION - Error handling improved
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

# Import custom modules
try:
    from algorithms import PredictionEngine
    from database import DatabaseManager
    from config import Config
except ImportError:
    st.error("❌ Lỗi import modules. Kiểm tra lại algorithms.py, database.py, config.py")
    st.stop()

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
# 2. CUSTOM CSS
# ==============================================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #010409 0%, #0d1117 100%);
        color: #e6edf3;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .live-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #238636;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; box-shadow: 0 0 0 0 rgba(35,134,54,0.7); }
        70% { opacity: 0.5; box-shadow: 0 0 0 10px rgba(35,134,54,0); }
        100% { opacity: 1; box-shadow: 0 0 0 0 rgba(35,134,54,0); }
    }
    
    .status-card {
        background: linear-gradient(135deg, #0d1117, #161b22);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    
    .status-green {
        border-left: 4px solid #238636;
    }
    
    .status-red {
        border-left: 4px solid #da3633;
    }
    
    .status-yellow {
        border-left: 4px solid #d29922;
    }
    
    .numbers-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin: 20px 0;
    }
    
    .num-card {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 3px solid #ff5858;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
    }
    
    .num-value {
        font-size: 65px;
        font-weight: 900;
        color: #ff5858;
    }
    
    .num-label {
        font-size: 14px;
        color: #8b949e;
        margin-top: 10px;
    }
    
    .lot-card {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 2px solid #58a6ff;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    
    .lot-value {
        font-size: 45px;
        font-weight: 800;
        color: #58a6ff;
    }
    
    .metric-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #58a6ff;
    }
    
    .metric-label {
        font-size: 12px;
        color: #8b949e;
        margin-top: 5px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 15px 35px;
    }
    
    @media (max-width: 600px) {
        .numbers-grid {
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }
        .num-value {
            font-size: 45px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. UTILITY FUNCTIONS - FIXED
# ==============================================================================

def format_currency(amount):
    """Format number as Vietnamese currency - FIXED."""
    try:
        if amount is None:
            return "₫0"
        amount = float(amount)
        return f"₫{amount:,.0f}"
    except (ValueError, TypeError):
        return "₫0"

def format_datetime(dt):
    """Format datetime for display."""
    try:
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)
        return dt.strftime("%H:%M:%S %d/%m/%Y")
    except:
        return "N/A"

def calculate_win_rate(logs):
    """Calculate win rate from prediction logs."""
    if not logs:
        return 0, 0, 0
    
    decided = [l for l in logs if l.get('result') is not None]
    if not decided:
        return 0, 0, 0
    
    wins = sum(1 for l in decided if l.get('won', False))
    total = len(decided)
    rate = (wins / total * 100) if total > 0 else 0
    
    return wins, total, round(rate, 2)

def check_prediction_win(prediction, result):
    """Check if prediction won (3 số 5 tinh)."""
    if not prediction or not result or len(result) != 5:
        return False
    
    pred_set = set(prediction)
    result_set = set(result)
    
    return len(pred_set.intersection(result_set)) >= 3

# ==============================================================================
# 4. SESSION STATE INITIALIZATION - FIXED
# ==============================================================================

def init_session_state():
    """Initialize all session state variables - FIXED."""
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    if "prediction_engine" not in st.session_state:
        st.session_state.prediction_engine = PredictionEngine()
    
    if "lottery_db" not in st.session_state:
        st.session_state.lottery_db = []
    
    if "predictions_log" not in st.session_state:
        st.session_state.predictions_log = []
    
    if "bankroll" not in st.session_state:
        st.session_state.bankroll = {
            "initial": 1000000,
            "current": 1000000,
            "bet_per_round": 10000,
            "sessions": []
        }
    
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    
    if "last_risk" not in st.session_state:
        st.session_state.last_risk = (0, "LOW", [])
    
    if "last_update" not in st.session_state:
        st.session_state.last_update = None
    
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False

# ==============================================================================
# 5. MAIN APPLICATION - SIMPLIFIED & FIXED
# ==============================================================================

def main():
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🎯 TITAN v35.0 PRO MAX")
    st.caption("5D Bet Prediction System")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Điều Khiển")
        
        st.session_state.auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
        
        st.markdown("### 💰 Vốn")
        st.metric("Tổng kỳ", len(st.session_state.lottery_db))
        st.metric("Dự đoán", len(st.session_state.predictions_log))
        
        if st.button("🗑️ Xóa dữ liệu"):
            st.session_state.lottery_db = []
            st.session_state.predictions_log = []
            st.success("✅ Đã xóa!")
            time.sleep(1)
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Dự Đoán", "📊 Phân Tích", "💰 Vốn"])
    
    # ==================== TAB 1: PREDICTION ====================
    with tab1:
        st.header("🎯 Dự Đoán 5D Bet")
        
        # Quick stats - FIXED
        try:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📦 Tổng kỳ", len(st.session_state.lottery_db))
            with col2:
                wins, total, rate = calculate_win_rate(st.session_state.predictions_log)
                st.metric("🎯 Win Rate", f"{rate}%")
            with col3:
                if st.session_state.last_update:
                    st.metric("⏰ Cập Nhật", format_datetime(st.session_state.last_update))
                else:
                    st.metric("⏰ Cập Nhật", "Chưa có")
        except Exception as e:
            st.error(f"❌ Lỗi hiển thị stats: {str(e)}")
        
        # Input area
        st.markdown("### 📥 Nhập Kết Quả")
        
        input_text = st.text_area(
            "📋 Dữ liệu (5 số/dòng)",
            height=200,
            placeholder="87746\n56421\n69137\n...",
            key="input_area"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            analyze_btn = st.button("🚀 PHÂN TÍCH", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Demo Data", use_container_width=True):
                demo_data = "\n".join([
                    "87746", "56421", "69137", "00443", "04475",
                    "64472", "16755", "58569", "62640", "99723"
                ])
                st.session_state.input_area = demo_data
                st.rerun()
        
        # Process input
        if analyze_btn and input_text.strip():
            try:
                with st.spinner("🧠 Đang phân tích..."):
                    # Clean data
                    numbers = st.session_state.db_manager.clean_data(input_text)
                    new_count = st.session_state.db_manager.add_numbers(
                        numbers, 
                        st.session_state.lottery_db
                    )
                    
                    if new_count > 0:
                        st.success(f"✅ Thêm {new_count} số mới")
                    else:
                        st.info("ℹ️ Không có số mới")
                    
                    # Generate prediction
                    if len(st.session_state.lottery_db) >= 20:
                        prediction = st.session_state.prediction_engine.predict(
                            st.session_state.lottery_db
                        )
                        risk_info = st.session_state.prediction_engine.calculate_risk(
                            st.session_state.lottery_db
                        )
                        
                        # Log prediction
                        st.session_state.predictions_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'prediction': prediction,
                            'risk_score': risk_info[0],
                            'result': None,
                            'won': None
                        })
                        
                        st.session_state.last_prediction = prediction
                        st.session_state.last_risk = risk_info
                        st.session_state.last_update = datetime.now()
                        
                        # Display prediction - FIXED
                        st.markdown("### 🎯 Kết Quả")
                        
                        risk_score, risk_level, risk_reasons = risk_info
                        main_3 = prediction.get('main_3', ['?', '?', '?'])
                        support_4 = prediction.get('support_4', ['?', '?', '?', '?'])
                        confidence = prediction.get('confidence', 0)
                        
                        # Status
                        if risk_level == "LOW":
                            status_color = "green"
                            status_text = "✅ ĐÁNH"
                        elif risk_level == "HIGH":
                            status_color = "red"
                            status_text = "🛑 DỪNG"
                        else:
                            status_color = "yellow"
                            status_text = "⚠️ THEO DÕI"
                        
                        st.markdown(f"""
                        <div class="status-card status-{status_color}">
                            <div style="text-align: center; font-size: 18px; font-weight: bold;">
                                {status_text} | Risk: {risk_score}/100
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Numbers display
                        st.markdown(f"""
                        <div style="text-align: center; margin: 20px 0;">
                            <div style="color: #8b949e; margin-bottom: 15px;">
                                🔮 3 SỐ CHÍNH ({confidence}%)
                            </div>
                            <div class="numbers-grid">
                                <div class="num-card"><div class="num-value">{main_3[0]}</div></div>
                                <div class="num-card"><div class="num-value">{main_3[1]}</div></div>
                                <div class="num-card"><div class="num-value">{main_3[2]}</div></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Support numbers
                        st.markdown(f"""
                        <div style="text-align: center; margin: 20px 0;">
                            <div style="color: #8b949e; margin-bottom: 15px;">🎲 4 SỐ LÓT</div>
                            <div style="display: flex; justify-content: center; gap: 10px;">
                                <div class="lot-card"><div class="lot-value">{support_4[0]}</div></div>
                                <div class="lot-card"><div class="lot-value">{support_4[1]}</div></div>
                                <div class="lot-card"><div class="lot-value">{support_4[2]}</div></div>
                                <div class="lot-card"><div class="lot-value">{support_4[3]}</div></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Logic
                        if prediction.get('logic'):
                            st.info(f"💡 **Logic:** {prediction['logic']}")
                        
                        # Risk warnings
                        if risk_reasons:
                            st.warning("⚠️ **Cảnh báo:**\n" + "\n".join([f"• {r}" for r in risk_reasons]))
                        
                        # Copy code
                        st.code(','.join(main_3 + support_4), language=None)
                        
                    else:
                        st.warning(f"⚠️ Cần ít nhất 20 kỳ (hiện có: {len(st.session_state.lottery_db)})")
                        
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                st.exception(e)
        
        elif st.session_state.last_prediction:
            st.info("👆 Nhập dữ liệu mới và bấm PHÂN TÍCH để có kết quả mới")
    
    # ==================== TAB 2: ANALYTICS ====================
    with tab2:
        st.header("📊 Phân Tích")
        
        if len(st.session_state.lottery_db) < 20:
            st.info("📊 Cần ít nhất 20 kỳ để phân tích")
        else:
            # Frequency
            st.markdown("### 🔥 Tần Suất")
            all_digits = ''.join(st.session_state.lottery_db[-100:])
            freq = Counter(all_digits)
            
            df_freq = pd.DataFrame(
                [(str(d), c) for d, c in sorted(freq.items())],
                columns=['Số', 'Tần Suất']
            )
            st.bar_chart(df_freq.set_index('Số'))
            
            # Top 5
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top 5 số nóng:**")
                for num, count in freq.most_common(5):
                    st.metric(f"Số {num}", f"{count} lần")
    
    # ==================== TAB 3: BANKROLL ====================
    with tab3:
        st.header("💰 Quản Lý Vốn")
        
        # Metrics - FIXED
        try:
            bankroll = st.session_state.bankroll
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Vốn ban đầu", format_currency(bankroll.get('initial', 0)))
            with col2:
                st.metric("Vốn hiện tại", format_currency(bankroll.get('current', 0)))
            with col3:
                profit = bankroll.get('current', 0) - bankroll.get('initial', 0)
                st.metric("Lợi nhuận", format_currency(profit))
        except Exception as e:
            st.error(f"❌ Lỗi hiển thị vốn: {str(e)}")
        
        # History
        st.markdown("### 📊 Lịch Sử")
        if st.session_state.predictions_log:
            df = pd.DataFrame(st.session_state.predictions_log)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8b949e; padding: 20px;">
        TITAN v35.0 PRO MAX | 5D Bet System<br>
        ⚠️ Chơi có trách nhiệm
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()