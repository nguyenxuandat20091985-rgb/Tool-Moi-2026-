# ==============================================================================
# TITAN v37.5 ULTRA - Main Application
# FIXED: All errors, enhanced UI, proper win condition, error handling
# ==============================================================================

import streamlit as st
import pandas as pd
from datetime import datetime
import re
import json
import time
from algorithms import PredictionEngine

# ==============================================================================
# 1. PAGE CONFIG & ENHANCED CSS
# ==============================================================================

st.set_page_config(
    page_title="🎯 TITAN v37.5 ULTRA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Mobile-Optimized CSS
st.markdown("""
<style>
    /* Global Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #010409 100%);
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Number Boxes */
    .main-box {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 3px solid #f85149;
        border-radius: 15px;
        padding: 20px 15px;
        text-align: center;
        margin: 8px 0;
        box-shadow: 0 6px 20px rgba(248,81,73,0.3);
    }
    
    .main-val {
        font-size: 50px;
        font-weight: 900;
        color: #f85149;
        text-shadow: 0 0 15px rgba(248,81,73,0.6);
    }
    
    /* Support Number Grid */
    .sup-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        margin: 15px 0;
    }
    
    .sup-box {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 2px solid #58a6ff;
        border-radius: 10px;
        padding: 15px 10px;
        text-align: center;
        color: #58a6ff;
        font-weight: 800;
        font-size: 28px;
    }
    
    /* Risk Banner */
    .risk-tag {
        padding: 12px 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 16px;
        margin: 15px 0;
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
        .main-val {
            font-size: 42px;
        }
        .sup-box {
            font-size: 24px;
            padding: 12px 8px;
        }
        .sup-container {
            gap: 6px;
        }
    }
    
    /* Info/Warning boxes */
    .stInfo, .stWarning, .stError, .stSuccess {
        border-radius: 10px;
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
            "bet_per_round": 10000
        }
    
    if "predictions_log" not in st.session_state:
        st.session_state.predictions_log = []

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def check_win_condition(main_3, actual_result):
    """
    Check win condition for 3 số 5 tinh:
    TRÚNG if actual_result contains ALL 3 numbers from main_3 (any position)
    """
    if not main_3 or not actual_result or len(actual_result) != 5:
        return False
    
    # Convert to sets for easy comparison
    pred_set = set(str(d) for d in main_3 if str(d).isdigit())
    result_set = set(actual_result)
    
    # Win if ALL 3 predicted numbers are in the result
    return pred_set.issubset(result_set)

def format_currency(amount):
    """Format number as Vietnamese currency."""
    try:
        return f"₫{float(amount):,.0f}"
    except:
        return "₫0"

# ==============================================================================
# 4. MAIN APPLICATION
# ==============================================================================

def main():
    # Initialize session state
    init_session()
    
    # Header
    st.title("🎯 TITAN v37.5 ULTRA")
    st.caption("Multi-Layer AI Prediction | Self-Learning")
    
    # Sidebar: Quick Stats
    with st.sidebar:
        st.markdown("### 📊 Trạng thái")
        
        # AI Status
        ai_status = st.session_state.ai.get_ai_status()
        st.metric("🎯 Win Rate", f"{ai_status['recent_win_rate']}%")
        st.metric("📈 Predictions", ai_status['predictions_tracked'])
        
        st.markdown("---")
        st.markdown("### 💰 Vốn")
        st.metric("Hiện tại", format_currency(st.session_state.bankroll['current']))
        
        profit = st.session_state.bankroll['current'] - st.session_state.bankroll['initial']
        color = "🟢" if profit >= 0 else "🔴"
        st.metric("Lợi nhuận", f"{color} {format_currency(profit)}")
        
        st.markdown("---")
        if st.button("🗑️ Reset All"):
            st.session_state.clear()
            st.success("✅ Đã reset!")
            time.sleep(1)
            st.rerun()
        
        st.markdown("---")
        st.warning("⚠️ Risk HIGH: Dừng ngay\n⚠️ Không AI nào chính xác 100%")
    
    # Main Content
    st.markdown("### 📥 Nhập kết quả (Mỗi kỳ 1 dòng, 5 chữ số)")
    
    raw = st.text_area(
        "📋 Dữ liệu:",
        height=120,
        placeholder="71757\n81750\n92002\n...",
        key="raw_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🚀 KÍCH HOẠT PHÂN TÍCH AI", type="primary", use_container_width=True)
    with col2:
        if st.button("📋 Demo Data", use_container_width=True):
            demo_data = "\n".join([
                "87746", "56421", "69137", "00443", "04475",
                "64472", "16755", "58569", "62640", "99723",
                "33769", "14671", "92002", "65449", "26073"
            ])
            st.session_state.raw_input = demo_data
            st.rerun()
    
    # Process Analysis
    if analyze_btn and raw.strip():
        try:
            with st.spinner("🧠 AI đang phân tích đa tầng..."):
                # Clean and validate input
                nums = re.findall(r'\d{5}', raw)
                
                if not nums:
                    st.error("❌ Không tìm thấy số 5 chữ số hợp lệ!")
                else:
                    # Update database (deduplicate)
                    db_set = set(st.session_state.db)
                    added = 0
                    for n in nums:
                        if n not in db_set:
                            st.session_state.db.insert(0, n)
                            db_set.add(n)
                            added += 1
                    
                    if added > 0:
                        st.success(f"✅ Đã thêm {added} số mới")
                    else:
                        st.info("ℹ️ Dữ liệu đã có trong hệ thống")
                    
                    # Generate prediction
                    if len(st.session_state.db) >= 10:
                        st.session_state.pred = st.session_state.ai.predict(st.session_state.db)
                        st.rerun()
                    else:
                        st.warning(f"⚠️ Cần ít nhất 10 kỳ (hiện có: {len(st.session_state.db)})")
                        
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)[:100]}")
    
    # Display Prediction
    if st.session_state.pred:
        p = st.session_state.pred
        risk = p.get('risk_metrics', {'score': 0, 'level': 'LOW'})
        
        # Risk Banner - FIXED: Proper color handling
        risk_color = "#238636" if risk.get('level') == "LOW" else "#da3633"
        risk_alpha = "33"  # 20% opacity in hex
        
        st.markdown(f"""
        <div class="risk-tag" style="background: {risk_color}{risk_alpha}; 
                   border: 1px solid {risk_color}; color: {risk_color}">
            RISK: {risk.get('score', 0)}/100 | KHUYẾN NGHỊ: {risk.get('level', 'LOW')}
        </div>
        """, unsafe_allow_html=True)
        
        # 3 Main Numbers - FIXED: Proper HTML rendering
        st.markdown("🔮 **3 SỐ CHÍNH (VÀO MẠNH)**")
        cols = st.columns(3)
        for i, num in enumerate(p.get('main_3', ['?', '?', '?'])):
            cols[i].markdown(f"""
            <div class="main-box">
                <div class="main-val">{num}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Avoid Warning
        if p.get('avoid'):
            st.markdown(f"""
            <div style="background: rgba(218,54,51,0.15); border-left: 4px solid #da3633; 
                       padding: 12px; border-radius: 8px; margin: 15px 0;">
                <strong style="color: #f85149;">🚫 TRÁNH:</strong> {', '.join(p['avoid'])}
            </div>
            """, unsafe_allow_html=True)
        
        # 4 Support Numbers - FIXED: Proper grid rendering
        st.markdown("🎲 **4 SỐ LÓT (GIỮ VỐN)**")
        support_nums = p.get('support_4', ['?', '?', '?', '?'])
        s_html = "".join([f'<div class="sup-box">{n}</div>' for n in support_nums])
        st.markdown(f'<div class="sup-container">{s_html}</div>', unsafe_allow_html=True)
        
        # Logic & Confidence
        confidence = p.get('confidence', 0)
        logic = p.get('logic', 'N/A')
        st.info(f"💡 Logic: {logic} | Tin cậy: {confidence}%")
        
        # Risk Reasons
        if risk.get('reasons') and risk.get('reasons') != ['Nhịp số tự nhiên']:
            st.warning("⚠️ **Cảnh báo:**\n" + "\n".join([f"• {r}" for r in risk['reasons']]))
        
        # Copy Code
        st.code(','.join(p.get('main_3', []) + p.get('support_4', [])), language=None)
        st.caption("📋 Bấm vào code để copy dàn 7 số")
        
        # Result Confirmation & AI Learning - FIXED: Proper win condition
        st.markdown("---")
        st.markdown("### ✅ Xác nhận kết quả & Dạy AI")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            actual = st.text_input("Kết quả thực tế (5 số):", key="actual_result", placeholder="12864")
        with col2:
            learn_btn = st.button("✅ GHI NHẬN", type="primary", use_container_width=True)
        
        if learn_btn and actual:
            if len(actual) == 5 and actual.isdigit():
                # FIXED: Proper win condition for 3 số 5 tinh
                is_win = check_win_condition(p.get('main_3', []), actual)
                
                # Determine which method to reward (simplified)
                layer_scores = p.get('layer_scores', {})
                best_method = max(layer_scores, key=layer_scores.get) if layer_scores else 'frequency'
                
                # Update AI weights (SELF-LEARNING)
                st.session_state.ai.update_weights(is_win, best_method)
                
                # Update bankroll
                bet = st.session_state.bankroll['bet_per_round']
                if is_win:
                    profit = bet * 1.9  # Typical 5D payout
                    st.session_state.bankroll['current'] += profit
                    st.success(f"🎉 TRÚNG! +{format_currency(profit)}")
                else:
                    st.session_state.bankroll['current'] -= bet
                    st.warning(f"❌ Trượt! -{format_currency(bet)}")
                
                # Log result for analytics
                st.session_state.predictions_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'prediction': p.get('main_3', []),
                    'actual': actual,
                    'won': is_win,
                    'confidence': confidence,
                    'method': best_method
                })
                
                # Keep log manageable
                if len(st.session_state.predictions_log) > 50:
                    st.session_state.predictions_log = st.session_state.predictions_log[-50:]
                
                time.sleep(1.5)
                st.rerun()
            else:
                st.error("❌ Nhập đúng 5 chữ số!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8b949e; padding: 20px; font-size: 12px;">
        🎯 TITAN v37.5 ULTRA | Multi-Layer Self-Learning AI<br>
        ⚠️ Công cụ hỗ trợ - Không đảm bảo 100% - Chơi có trách nhiệm
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 5. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()