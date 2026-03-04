# ==============================================================================
# TITAN v37.5 PRO MAX - Main Application
# Mobile-Optimized Grid UI with Enhanced AI Integration
# ==============================================================================

import streamlit as st
import pandas as pd
from datetime import datetime
import re
import time
import json

# Import the enhanced AI Engine
from algorithms import PredictionEngine

# ==============================================================================
# 1. PAGE CONFIG & ENHANCED CSS
# ==============================================================================

st.set_page_config(
    page_title="🎯 TITAN v37.5 PRO MAX",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Mobile-Optimized CSS
st.markdown("""
<style>
    /* Global Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Number Boxes - Enhanced */
    .main-box {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 3px solid #f85149;
        border-radius: 18px;
        padding: 25px 20px;
        text-align: center;
        margin: 8px 0;
        box-shadow: 0 8px 25px rgba(248,81,73,0.3);
        transition: transform 0.2s;
    }
    .main-box:hover {
        transform: translateY(-3px);
    }
    
    .main-val {
        font-size: 55px;
        font-weight: 900;
        color: #f85149;
        text-shadow: 0 0 20px rgba(248,81,73,0.6);
        line-height: 1;
    }
    
    .main-label {
        font-size: 12px;
        color: #8b949e;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Support Numbers Grid */
    .sup-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        margin: 15px 0;
    }
    
    .sup-box {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 2px solid #58a6ff;
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
        color: #58a6ff;
        font-weight: 800;
        font-size: 32px;
        box-shadow: 0 4px 15px rgba(88,166,255,0.2);
    }
    
    /* Risk Tag/Banner */
    .risk-tag {
        padding: 12px 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 16px;
        margin: 15px 0;
        border: 2px solid;
    }
    
    /* Status Colors */
    .status-low {
        background: rgba(35,134,54,0.15);
        border-color: #238636;
        color: #3fb950;
    }
    .status-medium {
        background: rgba(210,153,34,0.15);
        border-color: #d29922;
        color: #f0b429;
    }
    .status-high {
        background: rgba(218,54,51,0.15);
        border-color: #da3633;
        color: #f85149;
    }
    
    /* Info Boxes */
    .info-enhanced {
        background: rgba(88,166,255,0.1);
        border-left: 4px solid #58a6ff;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
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
    
    /* Text Areas & Inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    
    /* Mobile Responsive */
    @media (max-width: 600px) {
        .main-box {
            padding: 20px 15px;
        }
        .main-val {
            font-size: 45px;
        }
        .sup-container {
            grid-template-columns: repeat(4, 1fr);
            gap: 6px;
        }
        .sup-box {
            font-size: 28px;
            padding: 15px 10px;
        }
        .risk-tag {
            font-size: 14px;
            padding: 10px 15px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE INITIALIZATION
# ==============================================================================

def init_session():
    """Initialize all session state variables."""
    if "ai" not in st.session_state:
        st.session_state.ai = PredictionEngine()
    
    if "db" not in st.session_state:
        st.session_state.db = []
    
    if "pred" not in st.session_state:
        st.session_state.pred = None
    
    if "bankroll" not in st.session_state:
        st.session_state.bankroll = {
            "initial": 1000000,
            "current": 1000000,
            "bet_per_round": 10000,
            "history": []
        }
    
    if "predictions_log" not in st.session_state:
        st.session_state.predictions_log = []

# ==============================================================================
# 3. UI HELPER FUNCTIONS
# ==============================================================================

def render_risk_banner(risk_metrics):
    """Render enhanced risk status banner."""
    score = risk_metrics.get('score', 0)
    level = risk_metrics.get('level', 'LOW')
    reasons = risk_metrics.get('reasons', [])
    
    if level == "LOW":
        status_class = "status-low"
        icon = "✅"
        action = "ĐÁNH"
    elif level == "HIGH":
        status_class = "status-high"
        icon = "🛑"
        action = "DỪNG"
    else:
        status_class = "status-medium"
        icon = "⚠️"
        action = "THEO DÕI"
    
    st.markdown(f"""
    <div class="risk-tag {status_class}">
        {icon} RISK: {score}/100 | KHUYẾN NGHỊ: {action}
    </div>
    """, unsafe_allow_html=True)
    
    # Show reasons if any
    if reasons and level != "LOW":
        with st.expander("📋 Chi tiết cảnh báo", expanded=False):
            for reason in reasons:
                st.markdown(f"• {reason}")

def render_main_numbers(numbers):
    """Render 3 main numbers in responsive grid."""
    st.markdown("🔮 **3 SỐ CHÍNH (VÀO MẠNH)**")
    
    html = '<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 15px 0;">'
    labels = ["Số 1", "Số 2", "Số 3"]
    
    for i, num in enumerate(numbers):
        html += f'''
        <div class="main-box">
            <div class="main-val">{num}</div>
            <div class="main-label">{labels[i]}</div>
        </div>
        '''
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_support_numbers(numbers):
    """Render 4 support numbers in responsive grid."""
    st.markdown("🎲 **4 SỐ LÓT (GIỮ VỐN)**")
    
    html = '<div class="sup-container">'
    for num in numbers:
        html += f'<div class="sup-box">{num}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ==============================================================================
# 4. MAIN APPLICATION
# ==============================================================================

def main():
    # Initialize session
    init_session()
    
    # Header
    st.title("🎯 TITAN v37.5 PRO MAX")
    st.caption("Multi-Layer AI Prediction | Self-Learning Engine")
    
    # Quick Stats Bar
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📦 Tổng kỳ", len(st.session_state.db))
    with col2:
        if st.session_state.predictions_log:
            logs = [l for l in st.session_state.predictions_log if l.get('actual')]
            if logs:
                wins = sum(1 for l in logs if l.get('won'))
                rate = wins / len(logs) * 100
                st.metric("🎯 Win Rate", f"{rate:.1f}%")
            else:
                st.metric("🎯 Win Rate", "Chưa có")
        else:
            st.metric("🎯 Win Rate", "Chưa có")
    with col3:
        profit = st.session_state.bankroll['current'] - st.session_state.bankroll['initial']
        color = "🟢" if profit >= 0 else "🔴"
        st.metric("💰 Lợi nhuận", f"{color} ₫{profit:,.0f}")
    
    # Input Section
    st.markdown("### 📥 Nhập kết quả")
    st.markdown("""
    **💡 Hướng dẫn:** Nhập kết quả 5D bet (5 chữ số), mỗi kỳ 1 dòng. 
    Càng nhiều dữ liệu, AI càng thông minh!
    """)
    
    raw = st.text_area(
        "Kết quả (5 số/dòng):",
        height=120,
        placeholder="71757\n81750\n92002\n...",
        key="raw_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🚀 KÍCH HOẠT PHÂN TÍCH AI", type="primary", use_container_width=True)
    with col2:
        if st.button("📋 Demo Data", use_container_width=True):
            demo = "\n".join([
                "87746", "56421", "69137", "00443", "04475",
                "64472", "16755", "58569", "62640", "99723",
                "33769", "14671", "92002", "65449", "26073"
            ])
            st.session_state.raw_input = demo
            st.rerun()
    
    # Process Analysis
    if analyze_btn and raw.strip():
        with st.spinner("🧠 AI đang phân tích đa tầng..."):
            try:
                # Clean and validate input
                nums = re.findall(r'\d{5}', raw)
                
                if not nums:
                    st.error("❌ Dữ liệu không đúng định dạng 5 chữ số!")
                else:
                    # Update database
                    st.session_state.db = nums
                    st.success(f"✅ Đã nạp {len(nums)} kỳ dữ liệu")
                    
                    # Run AI prediction
                    st.session_state.pred = st.session_state.ai.predict(st.session_state.db)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
    
    # Display Prediction Results
    if st.session_state.pred:
        p = st.session_state.pred
        risk = p.get('risk_metrics', {'score': 0, 'level': 'LOW', 'reasons': []})
        
        # Risk Banner
        render_risk_banner(risk)
        
        # 3 Main Numbers
        render_main_numbers(p['main_3'])
        
        # Show avoid numbers if any
        layer_details = p.get('layer_details', {})
        if 'pattern' in layer_details:
            avoid = layer_details['pattern'].get('details', {}).get('avoid', [])
            if avoid:
                st.markdown(f"""
                <div style="background: rgba(218,54,51,0.1); border-left: 4px solid #da3633; 
                           padding: 12px; border-radius: 8px; margin: 15px 0;">
                    <strong style="color: #f85149;">🚫 TRÁNH:</strong> {', '.join(avoid)}
                </div>
                """, unsafe_allow_html=True)
        
        # 4 Support Numbers
        render_support_numbers(p['support_4'])
        
        # Logic & Confidence
        confidence = p.get('confidence', 0)
        logic = p.get('logic', 'N/A')
        st.info(f"💡 **Logic:** {logic} | **Tin cậy:** {confidence}%")
        
        # Copy Code
        st.markdown("---")
        st.code(','.join(p['main_3'] + p['support_4']), language=None)
        st.caption("📋 Bấm vào code để copy dàn 7 số")
        
        # Result Confirmation & AI Learning
        st.markdown("### ✅ Xác nhận kết quả & Dạy AI")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            actual = st.text_input("Kết quả thực tế (5 số):", key="actual_input", placeholder="12864")
        with col2:
            learn_btn = st.button("✅ GHI NHẬN & TỐI ƯU AI", type="primary", use_container_width=True)
        
        if learn_btn and actual and len(actual) == 5 and actual.isdigit():
            # Check win condition (3 số 5 tinh)
            pred_set = set(p['main_3'])
            result_set = set(actual)
            is_win = len(pred_set.intersection(result_set)) >= 3
            
            # Determine which layer to reward (simplified heuristic)
            layer_details = p.get('layer_details', {})
            best_layer = 'frequency'  # Default
            
            if layer_details:
                # Pick layer with highest detail score (simplified)
                scores = {}
                for layer, details in layer_details.items():
                    if isinstance(details, dict) and 'details' in details:
                        d = details['details']
                        if 'top_score' in d:
                            scores[layer] = d['top_score']
                        elif 'patterns_detected' in d:
                            scores[layer] = d['patterns_detected'] * 2
                if scores:
                    best_layer = max(scores, key=scores.get)
            
            # Update AI weights (SELF-LEARNING)
            st.session_state.ai.update_weights(is_win, best_layer)
            
            # Update bankroll
            bet = st.session_state.bankroll['bet_per_round']
            if is_win:
                profit = bet * 1.9  # Typical 5D payout
                st.session_state.bankroll['current'] += profit
                st.success(f"🎉 TRÚNG! +₫{profit:,.0f} | AI đã ghi nhớ pattern")
            else:
                st.session_state.bankroll['current'] -= bet
                st.warning(f"❌ Trượt! -₫{bet:,.0f} | AI đang điều chỉnh trọng số")
            
            # Log result for analytics
            st.session_state.predictions_log.append({
                'timestamp': datetime.now().isoformat(),
                'prediction': p['main_3'],
                'actual': actual,
                'won': is_win,
                'confidence': p.get('confidence', 0),
                'method_rewarded': best_layer,
                'risk_score': risk.get('score', 0)
            })
            
            time.sleep(2)
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8b949e; font-size: 12px; padding: 20px;">
        🎯 TITAN v37.5 PRO MAX | Multi-Layer Self-Learning AI<br>
        ⚠️ Không có AI nào chính xác 100% - Chơi có trách nhiệm
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 5. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()