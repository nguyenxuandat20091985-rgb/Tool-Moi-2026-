# ==============================================================================
# TITAN v36.0 AI SELF-LEARNING - Main Application
# Multi-Layer AI with Mobile-Optimized Grid UI
# ==============================================================================

import streamlit as st
import pandas as pd
from datetime import datetime
import re
import time
import json

# Import the AI Engine
from algorithms import PredictionEngine

# ==============================================================================
# 1. PAGE CONFIG & CSS
# ==============================================================================

st.set_page_config(
    page_title="🎯 TITAN v36.0 AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Mobile-Optimized Grid CSS
st.markdown("""
<style>
    /* Global Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #010409 0%, #0d1117 100%);
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Mobile Grid Container for Numbers */
    .number-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 20px 0;
    }
    
    .number-box {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 3px solid #ff4b4b;
        border-radius: 15px;
        padding: 20px 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(255,75,75,0.3);
    }
    
    .number-val {
        font-size: 42px;
        font-weight: 900;
        color: #ff4b4b;
        text-shadow: 0 0 15px rgba(255,75,75,0.6);
    }
    
    .number-label {
        font-size: 12px;
        color: #8b949e;
        margin-top: 8px;
        text-transform: uppercase;
    }
    
    /* Support Numbers Grid */
    .support-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin: 20px 0;
    }
    
    .support-box {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 2px solid #58a6ff;
        border-radius: 12px;
        padding: 15px 10px;
        text-align: center;
        color: #58a6ff;
        font-weight: 800;
        font-size: 32px;
    }
    
    /* Status Cards */
    .status-card {
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        margin: 15px 0;
    }
    .status-green {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
    }
    .status-red {
        background: linear-gradient(135deg, #da3633, #f85149);
        color: white;
    }
    .status-yellow {
        background: linear-gradient(135deg, #d29922, #f0b429);
        color: #0d1117;
    }
    
    /* AI Weights Display */
    .weights-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        margin: 10px 0;
    }
    .weight-item {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .weight-val {
        font-size: 20px;
        font-weight: bold;
        color: #58a6ff;
    }
    .weight-label {
        font-size: 11px;
        color: #8b949e;
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
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(35,134,54,0.5);
    }
    
    /* Mobile Responsive */
    @media (max-width: 600px) {
        .number-container {
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }
        .number-box {
            padding: 15px 10px;
        }
        .number-val {
            font-size: 35px;
        }
        .support-container {
            grid-template-columns: repeat(4, 1fr);
            gap: 6px;
        }
        .support-box {
            font-size: 28px;
            padding: 12px 8px;
        }
        .weights-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE INITIALIZATION
# ==============================================================================

def init_session():
    """Initialize all session state variables."""
    if "ai_engine" not in st.session_state:
        st.session_state.ai_engine = PredictionEngine()
    
    if "lottery_db" not in st.session_state:
        st.session_state.lottery_db = []
    
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    
    if "last_risk" not in st.session_state:
        st.session_state.last_risk = (0, "LOW", [])
    
    if "bankroll" not in st.session_state:
        st.session_state.bankroll = {
            "initial": 1000000,
            "current": 1000000,
            "bet_per_round": 10000,
            "history": []
        }

# ==============================================================================
# 3. UI COMPONENTS
# ==============================================================================

def render_number_grid(numbers, title, box_class, val_class, label_class=None, labels=None):
    """Render responsive number grid."""
    st.markdown(f"**{title}**")
    
    html = '<div class="number-container">'
    for i, num in enumerate(numbers):
        label_html = f'<div class="{label_class}">{labels[i]}</div>' if labels and label_class else ''
        html += f'''
        <div class="{box_class}">
            <div class="{val_class}">{num}</div>
            {label_html}
        </div>
        '''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_status_card(risk_score, risk_level):
    """Render risk status card."""
    if risk_level == "LOW":
        status_class = "status-green"
        icon = "✅"
        action = "ĐÁNH"
    elif risk_level == "HIGH":
        status_class = "status-red"
        icon = "🛑"
        action = "DỪNG"
    else:
        status_class = "status-yellow"
        icon = "⚠️"
        action = "THEO DÕI"
    
    st.markdown(f"""
    <div class="status-card {status_class}">
        {icon} RISK: {risk_score}/100 | KHUYẾN NGHỊ: {action}
    </div>
    """, unsafe_allow_html=True)

def render_ai_weights(weights):
    """Display AI algorithm weights."""
    st.markdown("**🧠 Trọng số thuật toán:**")
    
    html = '<div class="weights-grid">'
    for algo, weight in weights.items():
        html += f'''
        <div class="weight-item">
            <div class="weight-val">{weight}%</div>
            <div class="weight-label">{algo}</div>
        </div>
        '''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ==============================================================================
# 4. MAIN APPLICATION
# ==============================================================================

def main():
    # Initialize
    init_session()
    
    # Header
    st.title("🎯 TITAN v36.0 AI")
    st.caption("Multi-Layer Self-Learning Prediction System")
    
    # Sidebar: AI Status
    with st.sidebar:
        st.markdown("### 🧠 Trạng thái AI")
        
        ai_status = st.session_state.ai_engine.get_ai_status()
        
        st.metric("🎯 Win Rate", f"{ai_status['recent_win_rate']}%")
        st.metric("📊 Predictions", ai_status['predictions_tracked'])
        st.metric("🧠 Pattern Memory", ai_status['pattern_memory_size'])
        
        st.markdown("---")
        render_ai_weights(ai_status['weights'])
        
        st.markdown("---")
        st.markdown("### 💰 Quản lý vốn")
        st.metric("Vốn hiện tại", f"₫{st.session_state.bankroll['current']:,.0f}")
        
        if st.button("🗑️ Reset AI & Data"):
            st.session_state.clear()
            st.success("✅ Đã reset!")
            time.sleep(1)
            st.rerun()
        
        st.markdown("---")
        st.warning("⚠️ Risk >= 70: DỪNG ngay\n⚠️ Không có AI nào chính xác 100%")
    
    # Main Tabs
    tab1, tab2 = st.tabs(["🚀 DỰ ĐOÁN", "💰 VỐN & AI LOG"])
    
    # ==================== TAB 1: PREDICTION ====================
    with tab1:
        st.header("🚀 Nhập & Phân Tích AI")
        
        # Quick Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📦 Tổng kỳ", len(st.session_state.lottery_db))
        with col2:
            if st.session_state.predictions_log if hasattr(st.session_state, 'predictions_log') else []:
                logs = [l for l in st.session_state.predictions_log if l.get('result')]
                if logs:
                    rate = sum(1 for l in logs if l.get('won')) / len(logs) * 100
                    st.metric("🎯 Win Rate", f"{rate:.1f}%")
                else:
                    st.metric("🎯 Win Rate", "Chưa có")
            else:
                st.metric("🎯 Win Rate", "Chưa có")
        with col3:
            profit = st.session_state.bankroll['current'] - st.session_state.bankroll['initial']
            color = "🟢" if profit >= 0 else "🔴"
            st.metric("💰 Lợi nhuận", f"{color} ₫{profit:,.0f}")
        
        # Input Area
        st.markdown("### 📥 Nhập kết quả kỳ trước")
        raw_input = st.text_area(
            "Nhập 5 chữ số (có thể nhiều kỳ, mỗi kỳ 1 dòng):",
            height=100,
            placeholder="87746\n56421\n69137\n...",
            key="raw_input"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            analyze_btn = st.button("🚀 PHÂN TÍCH AI", type="primary", use_container_width=True)
        with col2:
            if st.button("📋 Demo Data", use_container_width=True):
                demo_data = "\n".join([
                    "87746", "56421", "69137", "00443", "04475",
                    "64472", "16755", "58569", "62640", "99723"
                ] * 2)
                st.session_state.raw_input = demo_data
                st.rerun()
        
        # Process Analysis
        if analyze_btn and raw_input.strip():
            with st.spinner("🧠 AI đang phân tích đa tầng..."):
                try:
                    # Clean and add data
                    new_nums = re.findall(r'\d{5}', raw_input)
                    added = 0
                    db_set = set(st.session_state.lottery_db)
                    
                    for n in new_nums:
                        if n not in db_set:
                            st.session_state.lottery_db.insert(0, n)
                            db_set.add(n)
                            added += 1
                    
                    if added > 0:
                        st.success(f"✅ Đã thêm {added} số mới")
                    else:
                        st.info("ℹ️ Dữ liệu đã có trong hệ thống")
                    
                    # AI Prediction
                    if len(st.session_state.lottery_db) >= 20:
                        pred = st.session_state.ai_engine.predict(st.session_state.lottery_db)
                        risk = st.session_state.ai_engine.calculate_risk(st.session_state.lottery_db)
                        
                        st.session_state.last_prediction = pred
                        st.session_state.last_risk = risk
                        
                        st.rerun()
                    else:
                        st.warning(f"⚠️ Cần ít nhất 20 kỳ (hiện có: {len(st.session_state.lottery_db)})")
                        
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
        
        # Display Prediction
        if st.session_state.last_prediction:
            p = st.session_state.last_prediction
            r_score, r_level, r_reasons = st.session_state.last_risk
            
            # Status Card
            render_status_card(r_score, r_level)
            
            # 3 Main Numbers - GRID DISPLAY
            render_number_grid(
                numbers=p['main_3'],
                title="🔮 3 SỐ CHÍNH (VÀO MẠNH)",
                box_class="number-box",
                val_class="number-val",
                label_class="number-label",
                labels=["Số 1", "Số 2", "Số 3"]
            )
            
            # Avoid Warning
            if p.get('avoid'):
                st.markdown(f"""
                <div style="background: rgba(218,54,51,0.15); border-left: 4px solid #da3633; 
                           padding: 12px; border-radius: 8px; margin: 15px 0;">
                    <strong style="color: #f85149;">🚫 TRÁNH:</strong> {', '.join(p['avoid'])}
                </div>
                """, unsafe_allow_html=True)
            
            # 4 Support Numbers - GRID DISPLAY
            render_number_grid(
                numbers=p['support_4'],
                title="🎲 4 SỐ LÓT",
                box_class="support-box",
                val_class=""  # Color handled by CSS
            )
            
            # Logic Explanation
            if p.get('logic'):
                st.info(f"💡 **AI Logic:** {p['logic']}")
            
            # Risk Reasons
            if r_reasons:
                st.warning("⚠️ **Cảnh báo:**\n" + "\n".join([f"• {r}" for r in r_reasons]))
            
            # Copy Code
            st.code(','.join(p['main_3'] + p['support_4']), language=None)
            st.caption("📋 Bấm để copy dàn 7 số")
            
            # Result Confirmation & AI Learning
            st.markdown("---")
            st.markdown("### ✅ Xác nhận kết quả & Dạy AI")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                actual = st.text_input("Kết quả thực tế (5 số):", key="actual_result", placeholder="12864")
            with col2:
                learn_btn = st.button("✅ DẠY AI", type="primary", use_container_width=True)
            
            if learn_btn and actual and len(actual) == 5 and actual.isdigit():
                # Check win condition (3 số 5 tinh)
                pred_set = set(p['main_3'])
                result_set = set(actual)
                is_win = len(pred_set.intersection(result_set)) >= 3
                
                # Determine which method to reward (simplified)
                layer_scores = p.get('layer_scores', {})
                if layer_scores:
                    best_method = max(layer_scores, key=layer_scores.get)
                else:
                    best_method = 'frequency'
                
                # Update AI weights (SELF-LEARNING)
                st.session_state.ai_engine.update_weights(is_win, best_method)
                
                # Update bankroll
                bet = st.session_state.bankroll['bet_per_round']
                if is_win:
                    profit = bet * 1.9  # Typical 5D payout
                    st.session_state.bankroll['current'] += profit
                    st.success(f"🎉 TRÚNG! +₫{profit:,.0f} | AI đã ghi nhớ pattern này")
                else:
                    st.session_state.bankroll['current'] -= bet
                    st.warning(f"❌ Trượt! -₫{bet:,.0f} | AI đang điều chỉnh trọng số")
                
                # Log result
                if "predictions_log" not in st.session_state:
                    st.session_state.predictions_log = []
                
                st.session_state.predictions_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'prediction': p['main_3'],
                    'actual': actual,
                    'won': is_win,
                    'method_rewarded': best_method
                })
                
                time.sleep(2)
                st.rerun()
    
    # ==================== TAB 2: BANKROLL & AI LOG ====================
    with tab2:
        st.header("💰 Vốn & AI Learning Log")
        
        # Bankroll Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vốn ban đầu", f"₫{st.session_state.bankroll['initial']:,.0f}")
        with col2:
            st.metric("Vốn hiện tại", f"₫{st.session_state.bankroll['current']:,.0f}")
        with col3:
            profit = st.session_state.bankroll['current'] - st.session_state.bankroll['initial']
            color = "🟢" if profit >= 0 else "🔴"
            st.metric("Lợi nhuận", f"{color} ₫{profit:,.0f}")
        
        # AI Status
        st.markdown("### 🧠 Trạng thái AI")
        ai_status = st.session_state.ai_engine.get_ai_status()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Trọng số hiện tại:**")
            render_ai_weights(ai_status['weights'])
        with col2:
            st.markdown("**Hiệu suất:**")
            st.metric("Win Rate", f"{ai_status['recent_win_rate']}%")
            st.metric("Predictions", ai_status['predictions_tracked'])
            st.metric("Pattern Memory", ai_status['pattern_memory_size'])
        
        # History Log
        st.markdown("### 📜 Lịch sử dự đoán")
        if "predictions_log" in st.session_state and st.session_state.predictions_log:
            df = pd.DataFrame(st.session_state.predictions_log)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%H:%M %d/%m')
            df['result'] = df.apply(lambda r: f"{r['prediction']} → {r['actual']}", axis=1)
            df['status'] = df['won'].apply(lambda w: '✅' if w else '❌')
            
            display_cols = ['timestamp', 'result', 'status', 'method_rewarded']
            st.dataframe(df[display_cols], hide_index=True, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu lịch sử")
        
        # Export Data
        st.markdown("---")
        st.markdown("### 💾 Export Data")
        
        if st.button("📥 Export JSON"):
            export_data = {
                'lottery_db': st.session_state.lottery_db,
                'bankroll': st.session_state.bankroll,
                'ai_weights': st.session_state.ai_engine.weights,
                'predictions_log': st.session_state.predictions_log if "predictions_log" in st.session_state else [],
                'exported_at': datetime.now().isoformat()
            }
            st.download_button(
                label="📥 Download",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"titan_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# ==============================================================================
# 5. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()