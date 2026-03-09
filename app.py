# ==============================================================================
# TITAN AI v5.0 - Main Application
# ==============================================================================

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from config import Config
from database import DatabaseManager
from algorithms import TitanAI

# Page config
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS (same as before - include all CSS from v5.0)
st.markdown("""
<style>
    /* Include all CSS from v5.0 here */
    .stApp { background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%); color: #e6edf3; }
    #MainMenu, footer, header { visibility: hidden; }
    /* ... rest of CSS ... */
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    # Initialize session state
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'ai_engine' not in st.session_state:
        st.session_state.ai_engine = TitanAI()
    if 'result' not in st.session_state:
        st.session_state.result = None
    
    # Header
    st.markdown(f"""
    <div class="header-card">
        <div class="header-title">{Config.APP_TITLE}</div>
        <div class="header-subtitle">{Config.APP_SUBTITLE}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Độ Chính Xác")
        acc_stats = st.session_state.db_manager.get_accuracy_stats()
        st.metric("Tổng lần test", acc_stats['total'])
        st.metric("Trúng", acc_stats['wins'])
        st.metric("Win Rate", f"{acc_stats['win_rate']}%")
        
        st.markdown("---")
        if st.button("🗑️ Reset"):
            st.session_state.db_manager.clear()
            st.session_state.ai_engine.accuracy_history = []
            st.success("✅ Đã reset!")
            time.sleep(0.5)
            st.rerun()
    
    # Stats Overview
    acc_stats = st.session_state.db_manager.get_accuracy_stats()
    st.markdown("### 📊 Thống kê")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Tổng kỳ", len(st.session_state.db_manager.data))
    with col2:
        st.metric("🎯 Đã test", acc_stats['total'])
    with col3:
        color = "🟢" if acc_stats['win_rate'] >= 40 else "🟡" if acc_stats['win_rate'] >= 25 else "🔴"
        st.metric("Win Rate", f"{color} {acc_stats['win_rate']}%")
    with col4:
        if st.session_state.result:
            st.metric("🏠 House Control", f"{st.session_state.ai_engine.pattern_detector.risk_level}%")
    
    # Input Section
    st.markdown("### 📥 Nhập kết quả")
    raw_input = st.text_area(
        "Dán kết quả (mỗi kỳ 1 dòng, 5 chữ số):",
        height=150,
        placeholder="09215\n23823\n45976\n...",
        key="data_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        analyze_btn = st.button("🚀 PHÂN TÍCH", type="primary", use_container_width=True)
    with col2:
        if st.button("📋 Demo", use_container_width=True):
            demo = "\n".join(["87746", "56421", "69137", "00443", "04475"] * 10)
            st.session_state.data_input = demo
            st.rerun()
    with col3:
        if st.button("🔄 Mới", use_container_width=True):
            st.rerun()
    
    # Process Analysis
    if analyze_btn and raw_input.strip():
        with st.spinner("🧠 Đang phân tích pattern nhà cái..."):
            numbers = st.session_state.db_manager.clean_data(raw_input)
            
            if not numbers:
                st.error("❌ Không tìm thấy số 5 chữ số!")
            else:
                updated_data, count_added = st.session_state.db_manager.add_numbers(
                    numbers, 
                    st.session_state.db_manager.data
                )
                st.session_state.db_manager.data = updated_data
                
                if count_added > 0:
                    st.success(f"✅ Đã thêm {count_added} số mới")
                
                if len(st.session_state.db_manager.data) >= Config.MIN_HISTORY_LENGTH:
                    st.session_state.result = st.session_state.ai_engine.analyze(
                        st.session_state.db_manager.data
                    )
                    st.rerun()
                else:
                    st.warning(f"⚠️ Cần ít nhất {Config.MIN_HISTORY_LENGTH} kỳ")
    
    # Display Results (same as v5.0)
    if st.session_state.result:
        res = st.session_state.result
        risk = res['risk']
        
        # House Control Warning
        if st.session_state.ai_engine.pattern_detector.risk_level >= 50:
            st.markdown(f"""
            <div class="pattern-alert">
                🚨 CẢNH BÁO: Nhà cái đang điều khiển ({st.session_state.ai_engine.pattern_detector.risk_level}%)<br>
                <small>{res['house_warning']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Status
        if risk['level'] == 'OK':
            status_class, status_text = 'status-ok', '✅ CÓ THỂ TEST'
        elif risk['level'] == 'MEDIUM':
            status_class, status_text = 'status-warn', '⚠️ CẨN THẬN'
        else:
            status_class, status_text = 'status-stop', '🛑 RỦI RO CAO'
        
        st.markdown(f"""
        <div class="status-card {status_class}">
            {status_text} | Risk: {risk['score']}/100 | {risk['reason']}
        </div>
        """, unsafe_allow_html=True)
        
        # Display patterns, numbers, etc. (same as v5.0)
        # ... rest of display code ...
        
        # Test Verification
        st.markdown("---")
        st.markdown("### ✅ Test Độ Chính Xác")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            actual = st.text_input("Kết quả thực tế:", key="actual_result", placeholder="12864")
        with col2:
            if st.button("✅ GHI NHẬN", type="primary", use_container_width=True):
                if actual and len(actual) == 5 and actual.isdigit():
                    main_3 = res['main_3']
                    is_win = set(main_3).issubset(set(actual))
                    
                    st.session_state.db_manager.record_test(
                        prediction=main_3,
                        actual=actual,
                        won=is_win,
                        confidence=res['confidence'],
                        house_risk=st.session_state.ai_engine.pattern_detector.risk_level
                    )
                    
                    if is_win:
                        st.success(f"🎉 TRÚNG!")
                    else:
                        st.warning(f"❌ Trượt!")
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #94a3b8; padding: 20px; font-size: 12px;">
        {Config.APP_TITLE} | {Config.APP_SUBTITLE}<br>
        🔍 Phát hiện: Bệt cầu | Đảo cầu | Xoay cầu | Bẫy nhịp<br>
        ⚠️ Khi House Control >= 50%: Nên dừng
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()