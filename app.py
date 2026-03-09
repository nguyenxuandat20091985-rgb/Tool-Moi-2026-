# ==============================================================================
# TITAN AI v5.0 - Main Application
# Production-Ready Streamlit Application
# ==============================================================================

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from config import Config
from database import DatabaseManager
from algorithms import TitanAI

# Validate configuration on startup
try:
    Config.validate()
except Exception as e:
    st.error(f"⚠️ Lỗi cấu hình: {str(e)}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Styling
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%); color: #e6edf3; }
    #MainMenu, footer, header { visibility: hidden; }
    .header-card { background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%); border-radius: 20px; padding: 30px; text-align: center; margin-bottom: 25px; border: 2px solid rgba(124,58,237,0.3); }
    .header-title { font-size: 32px; font-weight: 900; color: white; margin: 0; }
    .header-subtitle { font-size: 14px; color: rgba(255,255,255,0.8); margin-top: 8px; }
    .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }
    .stat-box { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.1); }
    .stat-value { font-size: 32px; font-weight: 800; color: #60a5fa; }
    .stat-label { font-size: 12px; color: #94a3b8; margin-top: 8px; text-transform: uppercase; }
    .status-card { padding: 15px 25px; border-radius: 12px; text-align: center; font-weight: 700; font-size: 15px; margin: 20px 0; }
    .status-ok { background: linear-gradient(135deg, #059669, #10b981); color: white; border: 2px solid #34d399; }
    .status-warn { background: linear-gradient(135deg, #d97706, #f59e0b); color: white; border: 2px solid #fbbf24; }
    .status-stop { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; border: 2px solid #f87171; }
    .numbers-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 20px 0; }
    .num-card { background: linear-gradient(135deg, #1e293b, #334155); border: 3px solid #60a5fa; border-radius: 16px; padding: 25px 20px; text-align: center; }
    .num-value { font-size: 56px; font-weight: 900; color: #60a5fa; line-height: 1; }
    .num-label { font-size: 12px; color: #94a3b8; margin-top: 10px; text-transform: uppercase; }
    .support-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0; }
    .support-card { background: linear-gradient(135deg, #1e293b, #334155); border: 2px solid #34d399; border-radius: 12px; padding: 18px 12px; text-align: center; }
    .support-value { font-size: 36px; font-weight: 800; color: #34d399; }
    .info-box { background: rgba(96,165,250,0.1); border-left: 4px solid #60a5fa; border-radius: 10px; padding: 15px 20px; margin: 15px 0; }
    .pattern-alert { background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); color: white; padding: 15px 25px; border-radius: 12px; text-align: center; font-weight: 700; margin: 20px 0; border: 2px solid #fca5a5; }
    .stButton > button { background: linear-gradient(135deg, #1e3a8a, #7c3aed); color: white !important; border: none; border-radius: 12px; font-weight: 700; padding: 14px 32px; font-size: 15px; }
    .stTextArea textarea, .stTextInput input { background-color: #1e293b !important; color: #ffffff !important; border: 2px solid #475569 !important; border-radius: 12px; }
    @media (max-width: 600px) { .stats-grid { grid-template-columns: repeat(2, 1fr); } .num-value { font-size: 42px; } .support-value { font-size: 28px; } }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    
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
        if st.button("🗑️ Reset Dữ Liệu"):
            st.session_state.db_manager.clear()
            st.session_state.ai_engine.accuracy_history = []
            st.success("✅ Đã reset!")
            time.sleep(0.5)
            st.rerun()
    
    # Stats Overview
    acc_stats = st.session_state.db_manager.get_accuracy_stats()
    db_stats = st.session_state.db_manager.get_statistics()
    
    st.markdown("### 📊 Thống Kê")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Tổng kỳ", db_stats['total_numbers'])
    with col2:
        st.metric("🎯 Đã test", acc_stats['total'])
    with col3:
        color = "🟢" if acc_stats['win_rate'] >= 40 else "🟡" if acc_stats['win_rate'] >= 25 else "🔴"
        st.metric("Win Rate", f"{color} {acc_stats['win_rate']}%")
    with col4:
        if st.session_state.result and st.session_state.result.get('success'):
            st.metric("🏠 House Control", f"{st.session_state.ai_engine.pattern_detector.risk_level}%")
    
    # Input Section
    st.markdown("### 📥 Nhập Kết Quả")
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
            try:
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
                        st.warning(f"⚠️ Cần ít nhất {Config.MIN_HISTORY_LENGTH} kỳ (hiện có: {len(st.session_state.db_manager.data)})")
                        
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
    
    # Display Results
    if st.session_state.result and st.session_state.result.get('success'):
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
        
        # 3 Main Numbers
        st.markdown("### 🔮 3 SỐ DỰ ĐOÁN")
        main_3 = res['main_3']
        st.markdown(f"""
        <div class="numbers-grid">
            <div class="num-card"><div class="num-value">{main_3[0]}</div><div class="num-label">Số 1</div></div>
            <div class="num-card"><div class="num-value">{main_3[1]}</div><div class="num-label">Số 2</div></div>
            <div class="num-card"><div class="num-value">{main_3[2]}</div><div class="num-label">Số 3</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        # 4 Support Numbers
        st.markdown("### 🎲 4 SỐ THAM KHẢO")
        support_4 = res['support_4']
        st.markdown(f"""
        <div class="support-grid">
            <div class="support-card"><div class="support-value">{support_4[0]}</div></div>
            <div class="support-card"><div class="support-value">{support_4[1]}</div></div>
            <div class="support-card"><div class="support-value">{support_4[2]}</div></div>
            <div class="support-card"><div class="support-value">{support_4[3]}</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.code(','.join(main_3 + support_4), language=None)
        
        if res['logic']:
            st.markdown(f'<div class="info-box">💡 <strong>Logic:</strong> {res["logic"]}</div>', unsafe_allow_html=True)
        
        # Test Verification
        st.markdown("---")
        st.markdown("### ✅ Test Độ Chính Xác")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            actual = st.text_input("Kết quả thực tế:", key="actual_result", placeholder="12864")
        with col2:
            if st.button("✅ GHI NHẬN", type="primary", use_container_width=True):
                if actual and len(actual) == 5 and actual.isdigit():
                    is_win = set(main_3).issubset(set(actual))
                    
                    st.session_state.db_manager.record_test(
                        prediction=main_3,
                        actual=actual,
                        won=is_win,
                        confidence=res['confidence'],
                        house_risk=st.session_state.ai_engine.pattern_detector.risk_level
                    )
                    
                    if is_win:
                        st.success(f"🎉 TRÚNG! (Confidence: {res['confidence']}%)")
                    else:
                        missing = set(main_3) - set(actual)
                        st.warning(f"❌ Trượt! Thiếu: {', '.join(missing)}")
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #94a3b8; padding: 20px; font-size: 12px;">
        {Config.APP_TITLE} | {Config.APP_SUBTITLE}<br>
        ⚠️ Công cụ phân tích - Không đảm bảo 100% - Chơi có trách nhiệm<br>
        🛑 Khi House Control >= 50%: Nên dừng
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()