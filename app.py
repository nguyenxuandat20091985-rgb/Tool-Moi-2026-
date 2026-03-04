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

# Import custom modules
from algorithms import PredictionEngine
from database import DatabaseManager
from config import Config

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
    /* Global */
    .stApp {
        background: linear-gradient(135deg, #010409 0%, #0d1117 100%);
        color: #e6edf3;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Live indicator */
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
    
    /* Status cards */
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
        background: linear-gradient(135deg, rgba(35,134,54,0.1), #0d1117);
    }
    
    .status-red {
        border-left: 4px solid #da3633;
        background: linear-gradient(135deg, rgba(218,54,51,0.1), #0d1117);
    }
    
    .status-yellow {
        border-left: 4px solid #d29922;
        background: linear-gradient(135deg, rgba(210,153,34,0.1), #0d1117);
    }
    
    /* Number display */
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
        box-shadow: 0 8px 30px rgba(255,88,88,0.3);
    }
    
    .num-value {
        font-size: 65px;
        font-weight: 900;
        color: #ff5858;
        text-shadow: 0 0 20px rgba(255,88,88,0.8);
    }
    
    .num-label {
        font-size: 14px;
        color: #8b949e;
        margin-top: 10px;
        text-transform: uppercase;
    }
    
    .lot-card {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 2px solid #58a6ff;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 6px 25px rgba(88,166,255,0.2);
    }
    
    .lot-value {
        font-size: 45px;
        font-weight: 800;
        color: #58a6ff;
    }
    
    /* Metrics */
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
        text-transform: uppercase;
        margin-top: 5px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 15px 35px;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(35,134,54,0.5);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        gap: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #58a6ff, #1f6feb);
        color: white;
    }
    
    /* Progress */
    .stProgress > div > div {
        background: linear-gradient(90deg, #58a6ff, #1f6feb);
    }
    
    /* Tables */
    .dataframe {
        background: #0d1117 !important;
        color: #e6edf3 !important;
    }
    
    /* Mobile responsive */
    @media (max-width: 600px) {
        .numbers-grid {
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }
        .num-value {
            font-size: 45px;
        }
        .lot-value {
            font-size: 35px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. SESSION STATE INITIALIZATION
# ==============================================================================

def init_session_state():
    """Initialize all session state variables."""
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
    
    if "last_update" not in st.session_state:
        st.session_state.last_update = None
    
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False
    
    if "refresh_count" not in st.session_state:
        st.session_state.refresh_count = 0

# ==============================================================================
# 4. UTILITY FUNCTIONS
# ==============================================================================

def format_currency(amount):
    """Format number as Vietnamese currency."""
    return f"₫{amount:,.0f}"

def format_datetime(dt):
    """Format datetime for display."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return dt.strftime("%H:%M:%S %d/%m/%Y")

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
# 5. UI COMPONENTS
# ==============================================================================

def render_live_indicator():
    """Render live status indicator."""
    if st.session_state.auto_refresh:
        st.markdown("""
        <div style="display: flex; align-items: center; margin: 10px 0;">
            <span class="live-indicator"></span>
            <span style="color: #238636; font-weight: bold;">● LIVE</span>
            <span style="color: #8b949e; margin-left: 10px;">Auto-refresh: 60s</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: flex; align-items: center; margin: 10px 0;">
            <span style="color: #8b949e; font-weight: bold;">○ OFFLINE</span>
            <span style="color: #8b949e; margin-left: 10px;">Manual refresh</span>
        </div>
        """, unsafe_allow_html=True)

def render_status_card(title, value, status="neutral", icon="📊"):
    """Render a status card."""
    status_class = f"status-{status}"
    
    st.markdown(f"""
    <div class="status-card {status_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #8b949e; font-size: 14px; margin-bottom: 5px;">{title}</div>
                <div style="color: #e6edf3; font-size: 28px; font-weight: bold;">{value}</div>
            </div>
            <div style="font-size: 40px;">{icon}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_prediction_display(prediction, risk_info):
    """Render main prediction display."""
    if not prediction:
        st.warning("⚠️ Chưa có dự đoán. Vui lòng nhập dữ liệu.")
        return
    
    risk_score, risk_level, risk_reasons = risk_info
    main_3 = prediction.get('main_3', ['?', '?', '?'])
    support_4 = prediction.get('support_4', ['?', '?', '?', '?'])
    confidence = prediction.get('confidence', 0)
    algorithm = prediction.get('algorithm', 'Ensemble')
    avoid = prediction.get('avoid', [])
    
    # Status color
    if risk_level == "LOW":
        status = "green"
        status_icon = "✅"
        decision = "ĐÁNH"
    elif risk_level == "HIGH":
        status = "red"
        status_icon = "🛑"
        decision = "DỪNG"
    else:
        status = "yellow"
        status_icon = "⚠️"
        decision = "THEO DÕI"
    
    # Status bar
    st.markdown(f"""
    <div class="status-card status-{status}">
        <div style="text-align: center; font-size: 18px; font-weight: bold;">
            {status_icon} RISK: {risk_score}/100 | KHUYẾN NGHỊ: {decision}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 3 số chính
    st.markdown(f"""
    <div style="text-align: center; margin: 30px 0;">
        <div style="color: #8b949e; font-size: 15px; margin-bottom: 20px; font-weight: bold;">
            🔮 3 SỐ CHÍNH (Độ tin cậy: {confidence}%) | Algorithm: {algorithm}
        </div>
        <div class="numbers-grid">
            <div class="num-card">
                <div class="num-value">{main_3[0]}</div>
                <div class="num-label">Số 1</div>
            </div>
            <div class="num-card">
                <div class="num-value">{main_3[1]}</div>
                <div class="num-label">Số 2</div>
            </div>
            <div class="num-card">
                <div class="num-value">{main_3[2]}</div>
                <div class="num-label">Số 3</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Avoid numbers
    if avoid:
        st.markdown(f"""
        <div class="status-card status-red">
            <div style="text-align: center; color: #f85149; font-weight: bold; font-size: 16px;">
                🚫 TRÁNH: {', '.join(avoid)} (Nhà cái đang bẫy)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 4 số lót
    st.markdown(f"""
    <div style="text-align: center; margin: 30px 0;">
        <div style="color: #8b949e; font-size: 14px; margin-bottom: 20px; font-weight: bold;">
            🎲 4 SỐ LÓT
        </div>
        <div class="numbers-grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="lot-card">
                <div class="lot-value">{support_4[0]}</div>
            </div>
            <div class="lot-card">
                <div class="lot-value">{support_4[1]}</div>
            </div>
            <div class="lot-card">
                <div class="lot-value">{support_4[2]}</div>
            </div>
            <div class="lot-card">
                <div class="lot-value">{support_4[3]}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Copy button
    numbers_to_copy = ','.join(main_3 + support_4)
    st.code(numbers_to_copy, language=None)
    st.caption("📋 Bấm vào code để copy dàn 7 số")
    
    # Logic explanation
    if prediction.get('logic'):
        st.info(f"💡 **Logic:** {prediction['logic']}")
    
    # Risk warnings
    if risk_reasons:
        warning_text = "⚠️ **Cảnh báo:**\n"
        for reason in risk_reasons:
            warning_text += f"• {reason}\n"
        st.warning(warning_text)

def render_bankroll_display():
    """Render bankroll management display."""
    bankroll = st.session_state.bankroll
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{format_currency(bankroll['initial'])}</div>
            <div class="metric-label">Vốn Ban Đầu</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#238636" if bankroll['current'] >= bankroll['initial'] else "#da3633"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: {color}">{format_currency(bankroll['current'])}</div>
            <div class="metric-label">Vốn Hiện Tại</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        profit = bankroll['current'] - bankroll['initial']
        color = "#238636" if profit >= 0 else "#da3633"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: {color}">{format_currency(profit)}</div>
            <div class="metric-label">Lợi Nhuận</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        wins, total, rate = calculate_win_rate(st.session_state.predictions_log)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{rate}%</div>
            <div class="metric-label">Win Rate ({wins}/{total})</div>
        </div>
        """, unsafe_allow_html=True)

def render_analytics_chart():
    """Render analytics chart."""
    logs = st.session_state.predictions_log
    
    if len(logs) < 5:
        st.info("📊 Cần ít nhất 5 lần dự đoán để hiển thị biểu đồ")
        return
    
    # Prepare data
    df = pd.DataFrame(logs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Win/Loss over time
    df['cumulative_wins'] = df['won'].cumsum()
    df['cumulative_total'] = df.index + 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Win Rate Theo Thời Gian")
        if 'won' in df.columns:
            win_rate = df['won'].expanding().mean() * 100
            st.line_chart(win_rate)
    
    with col2:
        st.markdown("### 📊 Confidence vs Actual")
        if 'confidence' in df.columns and 'won' in df.columns:
            df_plot = df[['confidence', 'won']].copy()
            df_plot['won'] = df_plot['won'].astype(int)
            st.scatter_chart(df_plot)

# ==============================================================================
# 6. MAIN APPLICATION
# ==============================================================================

def main():
    # Initialize
    init_session_state()
    
    # Header
    st.title("🎯 TITAN v35.0 PRO MAX")
    st.caption("5D Bet Professional Prediction System | Real-Time Analysis")
    
    # Live indicator
    render_live_indicator()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Điều Khiển")
        
        # Auto-refresh toggle
        st.session_state.auto_refresh = st.checkbox("🔄 Auto-refresh (60s)", value=False)
        
        # Bankroll settings
        st.markdown("### 💰 Quản Lý Vốn")
        initial = st.number_input("Vốn ban đầu", value=st.session_state.bankroll['initial'], step=100000)
        bet_amount = st.number_input("Cược/kỳ", value=st.session_state.bankroll['bet_per_round'], step=10000)
        
        if st.button("💾 Cập nhật vốn"):
            st.session_state.bankroll['initial'] = initial
            st.session_state.bankroll['bet_per_round'] = bet_amount
            st.success("✅ Đã cập nhật!")
        
        st.markdown("---")
        
        # Data management
        st.markdown("### 📁 Dữ Liệu")
        st.metric("Tổng kỳ", len(st.session_state.lottery_db))
        st.metric("Dự đoán", len(st.session_state.predictions_log))
        
        if st.button("🗑️ Xóa tất cả dữ liệu"):
            st.session_state.lottery_db = []
            st.session_state.predictions_log = []
            st.success("✅ Đã xóa!")
            time.sleep(1)
            st.rerun()
        
        st.markdown("---")
        
        # Export
        if st.button("📥 Export Data (JSON)"):
            data = {
                'lottery_db': st.session_state.lottery_db,
                'predictions_log': st.session_state.predictions_log,
                'bankroll': st.session_state.bankroll,
                'exported_at': datetime.now().isoformat()
            }
            st.download_button(
                label="📥 Download",
                data=json.dumps(data, indent=2, ensure_ascii=False),
                file_name=f"titan_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        st.markdown("---")
        st.markdown("### ⚠️ Lưu Ý Quan Trọng")
        st.warning("""
        • Không có tool nào chính xác 100%
        • Luôn quản lý vốn cẩn thận
        • Dừng khi thua liên tiếp 5 kỳ
        • Chỉ chơi với tiền có thể mất
        """)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Dự Đoán", "📊 Phân Tích", "💰 Vốn", "⚙️ Cài Đặt"])
    
    # ==================== TAB 1: PREDICTION ====================
    with tab1:
        st.header("🎯 Dự Đoán 5D Bet")
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            render_status_card("Tổng Kỳ", len(st.session_state.lottery_db), "neutral", "📦")
        with col2:
            wins, total, rate = calculate_win_rate(st.session_state.predictions_log)
            render_status_card("Win Rate", f"{rate}%", "green" if rate >= 40 else "yellow" if rate >= 20 else "red", "🎯")
        with col3:
            last_update = st.session_state.last_update
            if last_update:
                render_status_card("Cập Nhật", format_datetime(last_update), "green", "⏰")
            else:
                render_status_card("Cập Nhật", "Chưa có", "yellow", "⏰")
        
        # Input area
        st.markdown("### 📥 Nhập Kết Quả")
        st.markdown("""
        **💡 Hướng dẫn:**
        - Nhập kết quả 5D bet (5 chữ số)
        - Mỗi kỳ cách nhau bằng xuống dòng
        - Tool tự động làm sạch dữ liệu
        - Khuyến nghị: Nhập 100+ kỳ để có kết quả tốt nhất
        """)
        
        input_text = st.text_area(
            "📋 Dữ liệu thô",
            height=200,
            placeholder="Ví dụ:\n87746\n56421\n69137\n00443\n...",
            key="input_area"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            analyze_btn = st.button("🚀 PHÂN TÍCH", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Làm Mới", use_container_width=True):
                st.rerun()
        with col3:
            if st.button("📋 Demo Data", use_container_width=True):
                demo_data = "\n".join([
                    "87746", "56421", "69137", "00443", "04475",
                    "64472", "16755", "58569", "62640", "99723",
                    "33769", "14671", "92002", "65449", "26073",
                    "93388", "31215", "51206", "41291", "24993"
                ])
                st.session_state.input_area = demo_data
                st.rerun()
        
        # Process input
        if analyze_btn and input_text.strip():
            with st.spinner("🧠 Đang phân tích thông minh..."):
                start_time = time.time()
                
                # Clean data
                numbers = st.session_state.db_manager.clean_data(input_text)
                new_count = st.session_state.db_manager.add_numbers(numbers, st.session_state.lottery_db)
                
                elapsed = time.time() - start_time
                
                if new_count > 0:
                    st.success(f"✅ Xử lý trong {elapsed:.2f}s | Thêm {new_count} số mới")
                else:
                    st.info(f"ℹ️ Không có số mới (data đã có trong hệ thống)")
                
                # Generate prediction
                if len(st.session_state.lottery_db) >= 20:
                    prediction = st.session_state.prediction_engine.predict(st.session_state.lottery_db)
                    risk_info = st.session_state.prediction_engine.calculate_risk(st.session_state.lottery_db)
                    
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
                    
                    st.markdown("### 🎯 Kết Quả Dự Đoán")
                    render_prediction_display(prediction, risk_info)
                    
                    # Update bankroll session
                    st.session_state.bankroll['sessions'].append({
                        'timestamp': datetime.now().isoformat(),
                        'prediction': prediction,
                        'bet': st.session_state.bankroll['bet_per_round']
                    })
                else:
                    st.warning(f"⚠️ Cần ít nhất 20 kỳ dữ liệu (hiện có: {len(st.session_state.lottery_db)})")
        
        elif st.session_state.last_prediction:
            st.markdown("### 🎯 Kết Quả Gần Nhất")
            render_prediction_display(st.session_state.last_prediction, st.session_state.last_risk)
            
            # Result verification
            st.markdown("---")
            st.markdown("### ✅ Xác Nhận Kết Quả")
            col1, col2 = st.columns([3, 1])
            with col1:
                actual_result = st.text_input("Kết quả thực tế (5 số)", key="actual_result", placeholder="Ví dụ: 12864")
            with col2:
                if st.button("✅ Xác nhận", key="confirm_result"):
                    if actual_result and len(actual_result) == 5 and actual_result.isdigit():
                        if st.session_state.predictions_log:
                            last_log = st.session_state.predictions_log[-1]
                            pred_numbers = last_log['prediction'].get('main_3', [])
                            won = check_prediction_win(pred_numbers, actual_result)
                            
                            last_log['result'] = actual_result
                            last_log['won'] = won
                            
                            # Update bankroll
                            if won:
                                # Win: 1.9x bet (typical 5D bet payout)
                                profit = st.session_state.bankroll['bet_per_round'] * 1.9
                                st.session_state.bankroll['current'] += profit
                                st.success(f"🎉 TRÚNG! +{format_currency(profit)}")
                            else:
                                loss = st.session_state.bankroll['bet_per_round']
                                st.session_state.bankroll['current'] -= loss
                                st.error(f"❌ Trượt! -{format_currency(loss)}")
                            
                            st.rerun()
                    else:
                        st.error("Nhập đúng 5 chữ số!")
    
    # ==================== TAB 2: ANALYTICS ====================
    with tab2:
        st.header("📊 Phân Tích Chi Tiết")
        
        if len(st.session_state.lottery_db) < 20:
            st.info("📊 Cần ít nhất 20 kỳ dữ liệu để phân tích")
        else:
            # Frequency analysis
            st.markdown("### 🔥 Tần Suất Số")
            all_digits = ''.join(st.session_state.lottery_db[-100:])
            freq = Counter(all_digits)
            
            col1, col2 = st.columns(2)
            
            with col1:
                df_freq = pd.DataFrame(
                    [(str(d), c) for d, c in sorted(freq.items())],
                    columns=['Số', 'Tần Suất']
                )
                st.bar_chart(df_freq.set_index('Số'))
            
            with col2:
                top_5 = freq.most_common(5)
                for num, count in top_5:
                    st.metric(f"Số {num}", f"{count} lần")
            
            # Pattern analysis
            st.markdown("### 🔄 Pattern Phát Hiện")
            patterns = st.session_state.prediction_engine.detect_patterns(st.session_state.lottery_db)
            
            if patterns.get('detected'):
                for p in patterns['detected'][:10]:
                    st.markdown(f"• {p}")
            else:
                st.caption("Không phát hiện pattern rõ ràng")
            
            # Analytics chart
            st.markdown("---")
            render_analytics_chart()
    
    # ==================== TAB 3: BANKROLL ====================
    with tab3:
        st.header("💰 Quản Lý Vốn")
        
        render_bankroll_display()
        
        st.markdown("---")
        st.markdown("### 📊 Lịch Sử Giao Dịch")
        
        if st.session_state.predictions_log:
            df_logs = pd.DataFrame(st.session_state.predictions_log)
            
            if 'result' in df_logs.columns:
                df_display = df_logs[['timestamp', 'prediction', 'result', 'won']].copy()
                df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%H:%M %d/%m')
                df_display['prediction'] = df_display['prediction'].apply(lambda x: ','.join(x.get('main_3', [])) if x else '?')
                df_display['won'] = df_display['won'].apply(lambda x: '✅' if x else '❌' if x is False else '⏳')
                
                st.dataframe(df_display, hide_index=True, use_container_width=True)
            else:
                st.info("Chưa có kết quả thực tế")
        else:
            st.info("Chưa có dữ liệu")
        
        # Kelly Criterion calculator
        st.markdown("---")
        st.markdown("### 📐 Kelly Criterion Calculator")
        
        win_rate = st.number_input("Win Rate (%)", value=35, min_value=0, max_value=100)
        payout = st.number_input("Tỷ lệ trả thưởng", value=1.9, min_value=1.0, max_value=10.0)
        
        if win_rate > 0 and payout > 1:
            p = win_rate / 100
            q = 1 - p
            b = payout - 1
            
            kelly = (p * b - q) / b
            kelly_percent = max(0, kelly * 100)
            
            st.info(f"""
            **Kelly Optimal Bet:** {kelly_percent:.2f}% vốn
            
            **Khuyến nghị:** {format_currency(st.session_state.bankroll['current'] * kelly_percent / 100):.0f} mỗi kỳ
            
            ⚠️ Không nên quá 5% vốn mỗi kỳ
            """)
    
    # ==================== TAB 4: SETTINGS ====================
    with tab4:
        st.header("⚙️ Cài Đặt Hệ Thống")
        
        st.markdown("### 🎯 Cấu Hình Dự Đoán")
        
        # Algorithm weights
        st.markdown("**Trọng số algorithms:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            freq_weight = st.slider("Frequency Analysis", 0, 100, 40)
            pattern_weight = st.slider("Pattern Recognition", 0, 100, 30)
        
        with col2:
            hotcold_weight = st.slider("Hot/Cold Analysis", 0, 100, 20)
            markov_weight = st.slider("Markov Chain", 0, 100, 10)
        
        if st.button("💾 Lưu cấu hình"):
            st.session_state.prediction_engine.weights = {
                'frequency': freq_weight,
                'pattern': pattern_weight,
                'hotcold': hotcold_weight,
                'markov': markov_weight
            }
            st.success("✅ Đã lưu!")
        
        st.markdown("---")
        st.markdown("### 🚨 Cảnh Báo Rủi Ro")
        
        risk_threshold = st.slider("Risk Threshold (Dừng khi >=)", 50, 100, 70)
        
        if st.button("💾 Lưu ngưỡng rủi ro"):
            st.session_state.prediction_engine.risk_threshold = risk_threshold
            st.success("✅ Đã lưu!")
        
        st.markdown("---")
        st.markdown("### 📱 Thông Báo")
        
        st.checkbox("🔔 Alert khi có kết quả mới", value=False)
        st.checkbox("🔔 Alert khi win rate < 20%", value=True)
        st.checkbox("🔔 Alert khi risk >= 70", value=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ Thông Tin Hệ Thống")
        
        st.info(f"""
        **Version:** 35.0 PRO MAX
        
        **Algorithms:** 5 (Ensemble)
        
        **Database:** {len(st.session_state.lottery_db)} kỳ
        
        **Last Update:** {st.session_state.last_update if st.session_state.last_update else 'Chưa có'}
        
        **Session:** {len(st.session_state.bankroll['sessions'])} lần cược
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8b949e; font-size: 12px; padding: 20px;">
        🎯 TITAN v35.0 PRO MAX | 5D Bet Professional System<br>
        ⚠️ Công cụ hỗ trợ - Không đảm bảo 100% - Chơi có trách nhiệm<br>
        📞 Support: titan-support@example.com
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(60)
        st.session_state.refresh_count += 1
        st.rerun()

# ==============================================================================
# 7. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()