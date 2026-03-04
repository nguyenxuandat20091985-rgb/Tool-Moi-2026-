# ==============================================================================
# TITAN v37.5 PRO MAX - Main Application (FIXED & UPGRADED)
# Mobile-Optimized UI | Error-Handled | Self-Learning AI
# ==============================================================================

import streamlit as st
import pandas as pd
from datetime import datetime
import re
import json
import time
from algorithms import PredictionEngine

# ==============================================================================
# 1. PAGE CONFIG & CSS
# ==============================================================================

st.set_page_config(
    page_title="🎯 TITAN v37.5 PRO MAX",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Mobile-Optimized CSS (Kept original structure, enhanced)
st.markdown("""
<style>
    /* Global Dark Theme */
    .stApp { 
        background: linear-gradient(135deg, #0d1117 0%, #010409 100%); 
        color: white; 
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
        padding: 20px; 
        text-align: center; 
        margin: 10px 0;
        box-shadow: 0 6px 20px rgba(248,81,73,0.3);
    }
    .main-val { 
        font-size: 50px; 
        font-weight: 900; 
        color: #f85149;
        text-shadow: 0 0 15px rgba(248,81,73,0.6);
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
        border-radius: 10px; 
        padding: 15px; 
        text-align: center; 
        color: #58a6ff; 
        font-weight: bold; 
        font-size: 28px;
    }
    
    /* Risk Tag */
    .risk-tag { 
        padding: 12px; 
        border-radius: 10px; 
        text-align: center; 
        font-weight: bold; 
        font-size: 16px;
        margin: 15px 0;
        border: 2px solid;
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
    
    /* Text inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #161b22;
        color: white;
        border: 1px solid #30363d;
    }
    
    /* Mobile Responsive */
    @media (max-width: 600px) {
        .main-val { font-size: 40px; }
        .sup-box { font-size: 24px; padding: 12px; }
        .sup-container { grid-template-columns: repeat(4, 1fr); gap: 5px; }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE INITIALIZATION
# ==============================================================================

def init_session():
    """Initialize all session state variables safely."""
    if "ai" not in st.session_state:
        st.session_state.ai = PredictionEngine()
    
    if "db" not in st.session_state:
        st.session_state.db = []
    
    if "pred" not in st.session_state:
        st.session_state.pred = None
    
    if "last_risk" not in st.session_state:
        st.session_state.last_risk = {'score': 0, 'level': 'LOW', 'reasons': []}
    
    if "predictions_log" not in st.session_state:
        st.session_state.predictions_log = []
    
    if "bankroll" not in st.session_state:
        st.session_state.bankroll = {
            "initial": 1000000,
            "current": 1000000,
            "bet_per_round": 10000
        }

# ==============================================================================
# 3. UTILITY FUNCTIONS
# ==============================================================================

def clean_and_add_numbers(raw_text, existing_db):
    """
    Clean raw input and add new numbers to database.
    Returns: (added_count, stats_dict)
    """
    if not raw_text or not raw_text.strip():
        return 0, {"found": 0, "new": 0, "duplicate": 0}
    
    # Extract all 5-digit numbers
    nums = re.findall(r'\d{5}', raw_text)
    
    stats = {"found": len(nums), "new": 0, "duplicate": 0}
    db_set = set(existing_db)
    
    for n in nums:
        if n not in db_set:
            existing_db.insert(0, n)  # Add to front (newest first)
            db_set.add(n)
            stats["new"] += 1
        else:
            stats["duplicate"] += 1
    
    # Limit database size
    if len(existing_db) > 3000:
        existing_db[:] = existing_db[:3000]
    
    return stats["new"], stats

def check_win_3so5tinh(prediction_3, result_5):
    """
    Check win condition for "3 số 5 tinh":
    Win if ALL 3 predicted numbers appear in the 5-digit result (any position).
    """
    if not prediction_3 or not result_5 or len(result_5) != 5:
        return False
    
    pred_set = set(prediction_3)
    result_set = set(result_5)
    
    # Win if all 3 predicted numbers are in the result
    return pred_set.issubset(result_set)

# ==============================================================================
# 4. MAIN APPLICATION
# ==============================================================================

def main():
    # Initialize session state
    init_session()
    
    # Header
    st.title("🎯 TITAN v37.5 PRO MAX")
    st.caption("Multi-Layer AI Prediction | Self-Learning")
    
    # Quick Stats Bar
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📦 Tổng kỳ", len(st.session_state.db))
    with col2:
        if st.session_state.predictions_log:
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
    st.markdown("### 📥 Nhập kết quả (Mỗi kỳ 1 dòng)")
    raw = st.text_area(
        "Nhập số 5 chữ số:",
        height=120,
        placeholder="71757\n81750\n92341\n...",
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
        try:
            with st.spinner("🧠 AI đang phân tích đa tầng..."):
                # Clean and add numbers
                added, stats = clean_and_add_numbers(raw, st.session_state.db)
                
                if stats['new'] > 0:
                    st.success(f"✅ Thêm {stats['new']} số mới (Tìm thấy: {stats['found']} | Trùng: {stats['duplicate']})")
                elif stats['found'] > 0:
                    st.info(f"ℹ️ Không có số mới ({stats['duplicate']} số đã có trong DB)")
                else:
                    st.error("❌ Không tìm thấy số 5 chữ số hợp lệ")
                
                # Generate prediction if enough data
                if len(st.session_state.db) >= 15:
                    pred = st.session_state.ai.predict(st.session_state.db)
                    st.session_state.pred = pred
                    st.session_state.last_risk = pred.get('risk_metrics', {})
                    st.rerun()
                else:
                    st.warning(f"⚠️ Cần ít nhất 15 kỳ (hiện có: {len(st.session_state.db)})")
                    
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")
    
    elif analyze_btn and not raw.strip():
        st.error("❌ Vui lòng nhập dữ liệu trước!")
    
    # Display Prediction
    if st.session_state.pred:
        p = st.session_state.pred
        risk = p.get('risk_metrics', {'score': 0, 'level': 'LOW', 'reasons': []})
        
        # Risk Banner (FIXED: proper color handling)
        if risk.get('level') == "LOW":
            color = "#238636"
            icon = "✅"
        else:
            color = "#da3633"
            icon = "⚠️"
        
        st.markdown(f'''
        <div class="risk-tag" style="background: {color}22; border-color: {color}; color: {color}">
            {icon} RISK: {risk.get("score", 0)}/100 | KHUYẾN NGHỊ: {risk.get("level", "N/A")}
        </div>
        ''', unsafe_allow_html=True)
        
        # 3 Main Numbers (FIXED: proper grid display)
        st.write("🔮 **3 SỐ CHÍNH (VÀO MẠNH)**")
        cols = st.columns(3)
        for i, num in enumerate(p.get('main_3', ['?', '?', '?'])):
            cols[i].markdown(f'''
            <div class="main-box">
                <div class="main-val">{num}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # 4 Support Numbers (FIXED: proper grid)
        st.write("🎲 **4 SỐ LÓT (GIỮ VỐN)**")
        support_nums = p.get('support_4', ['?', '?', '?', '?'])
        s_html = "".join([f'<div class="sup-box">{n}</div>' for n in support_nums])
        st.markdown(f'<div class="sup-container">{s_html}</div>', unsafe_allow_html=True)
        
        # Logic & Confidence (FIXED: safe access)
        logic = p.get('logic', 'N/A')
        confidence = p.get('confidence', 0)
        st.info(f"💡 Logic: {logic} | Tin cậy: {confidence}%")
        
        # Risk reasons if any
        if risk.get('reasons') and risk['reasons'] != ["Nhịp số tự nhiên"]:
            st.warning("⚠️ **Cảnh báo:**\n" + "\n".join([f"• {r}" for r in risk['reasons']]))
        
        # Copy code for easy betting
        st.code(','.join(p.get('main_3', []) + p.get('support_4', [])), language=None)
        st.caption("📋 Bấm vào code để copy dàn 7 số")
        
        # Result Confirmation & AI Learning (FIXED: correct win condition)
        st.markdown("---")
        st.markdown("### ✅ Xác nhận kết quả & Dạy AI")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            actual = st.text_input("Kết quả thực tế kỳ này (5 số):", key="actual_input", placeholder="12864")
        with col2:
            learn_btn = st.button("✅ GHI NHẬN", type="primary", use_container_width=True)
        
        if learn_btn and actual and len(actual) == 5 and actual.isdigit():
            try:
                # FIXED: Correct win condition for "3 số 5 tinh"
                is_win = check_win_3so5tinh(p.get('main_3', []), actual)
                
                # Determine which method to reward (simplified heuristic)
                layer_details = p.get('layer_details', {})
                if layer_details:
                    # Pick method whose top picks most match the actual result
                    best_method = 'frequency'  # default
                    best_match = 0
                    for method, picks in layer_details.items():
                        match = len(set(picks).intersection(set(actual)))
                        if match > best_match:
                            best_match = match
                            best_method = method
                else:
                    best_method = 'frequency'
                
                # Update AI weights (SELF-LEARNING)
                st.session_state.ai.update_weights(is_win, best_method)
                
                # Update bankroll
                bet = st.session_state.bankroll['bet_per_round']
                if is_win:
                    # Typical 5D bet payout: 1.9x
                    profit = bet * 1.9
                    st.session_state.bankroll['current'] += profit
                    st.success(f"🎉 TRÚNG! +₫{profit:,.0f} | AI đã ghi nhớ pattern")
                else:
                    st.session_state.bankroll['current'] -= bet
                    st.warning(f"❌ Trượt! -₫{bet:,.0f} | AI đang điều chỉnh")
                
                # Log result for analytics
                st.session_state.predictions_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'prediction': p.get('main_3', []),
                    'actual': actual,
                    'won': is_win,
                    'method_rewarded': best_method,
                    'confidence': p.get('confidence', 0)
                })
                
                # Keep log manageable
                if len(st.session_state.predictions_log) > 100:
                    st.session_state.predictions_log = st.session_state.predictions_log[-100:]
                
                time.sleep(2)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Lỗi ghi nhận: {str(e)}")
        elif learn_btn and (not actual or len(actual) != 5 or not actual.isdigit()):
            st.warning("⚠️ Vui lòng nhập đúng 5 chữ số!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8b949e; padding: 20px; font-size: 12px;">
        🎯 TITAN v37.5 PRO MAX | Multi-Layer AI<br>
        ⚠️ Không có AI nào chính xác 100% - Chơi có trách nhiệm
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 5. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()