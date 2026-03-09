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

st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%); color: #e6edf3; }
    #MainMenu, footer, header { visibility: hidden; }
    .header-card { background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%); border-radius: 20px; padding: 30px; text-align: center; margin-bottom: 25px; }
    .header-title { font-size: 32px; font-weight: 900; color: white; margin: 0; }
    .header-subtitle { font-size: 14px; color: rgba(255,255,255,0.8); margin-top: 8px; }
    .status-card { padding: 15px 25px; border-radius: 12px; text-align: center; font-weight: 700; margin: 20px 0; }
    .status-ok { background: linear-gradient(135deg, #059669, #10b981); color: white; }
    .status-warn { background: linear-gradient(135deg, #d97706, #f59e0b); color: white; }
    .status-stop { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; }
    .numbers-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 20px 0; }
    .num-card { background: linear-gradient(135deg, #1e293b, #334155); border: 3px solid #60a5fa; border-radius: 16px; padding: 25px; text-align: center; }
    .num-value { font-size: 56px; font-weight: 900; color: #60a5fa; }
    .num-label { font-size: 12px; color: #94a3b8; margin-top: 10px; }
    .support-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0; }
    .support-card { background: #1e293b; border: 2px solid #34d399; border-radius: 12px; padding: 18px; text-align: center; }
    .support-value { font-size: 36px; font-weight: 800; color: #34d399; }
    .info-box { background: rgba(96,165,250,0.1); border-left: 4px solid #60a5fa; padding: 15px; margin: 15px 0; border-radius: 8px; }
    .pattern-alert { background: linear-gradient(135deg, #dc2626, #b91c1c); color: white; padding: 15px; border-radius: 12px; text-align: center; font-weight: 700; margin: 20px 0; }
    .stButton > button { background: linear-gradient(135deg, #1e3a8a, #7c3aed); color: white !important; border: none; border-radius: 12px; font-weight: 700; padding: 14px 32px; }
    .stTextArea textarea, .stTextInput input { background-color: #1e293b !important; color: #ffffff !important; border: 2px solid #475569 !important; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

def main():
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'ai_engine' not in st.session_state:
        st.session_state.ai_engine = TitanAI()
    if 'result' not in st.session_state:
        st.session_state.result = None
    
    st.markdown(f"""
    <div class="header-card">
        <div class="header-title">{Config.APP_TITLE}</div>
        <div class="header-subtitle">{Config.APP_SUBTITLE}</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 📊 Thống kê")
        acc_stats = st.session_state.db_manager.get_accuracy_stats()
        st.metric("Test", acc_stats['total'])
        st.metric("Win", f"{acc_stats['win_rate']}%")
        
        if st.button("🗑️ Reset"):
            st.session_state.db_manager.clear()
            st.success("✅ Reset!")
            time.sleep(0.5)
            st.rerun()
    
    acc_stats = st.session_state.db_manager.get_accuracy_stats()
    db_stats = st.session_state.db_manager.get_statistics()
    
    st.markdown("### 📊 Thống kê")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📦 Kỳ", db_stats['total_numbers'])
    with col2:
        st.metric("🎯 Test", acc_stats['total'])
    with col3:
        color = "🟢" if acc_stats['win_rate'] >= 40 else "🟡" if acc_stats['win_rate'] >= 25 else "🔴"
        st.metric("Win Rate", f"{color} {acc_stats['win_rate']}%")
    
    st.markdown("### 📥 Nhập kết quả")
    raw_input = st.text_area(
        "Dán kết quả (5 số/dòng):",
        height=150,
        placeholder="09215\n23823\n45976\n...",
        key="data_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🚀 PHÂN TÍCH", type="primary", use_container_width=True)
    with col2:
        if st.button("📋 Demo", use_container_width=True):
            st.session_state.data_input = "\n".join(["87746", "56421", "69137", "00443", "04475"] * 10)
            st.rerun()
    
    if analyze_btn and raw_input.strip():
        with st.spinner("🧠 Đang phân tích..."):
            try:
                numbers = st.session_state.db_manager.clean_data(raw_input)
                
                if not numbers:
                    st.error("❌ Không tìm thấy số!")
                else:
                    updated, count = st.session_state.db_manager.add_numbers(numbers)
                    
                    if count > 0:
                        st.success(f"✅ Thêm {count} số mới")
                    
                    if len(st.session_state.db_manager.data) >= Config.MIN_HISTORY_LENGTH:
                        st.session_state.result = st.session_state.ai_engine.analyze(st.session_state.db_manager.data)
                        st.rerun()
                    else:
                        st.warning(f"⚠️ Cần {Config.MIN_HISTORY_LENGTH}+ kỳ")
                        
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
    
    if st.session_state.result and st.session_state.result.get('success'):
        res = st.session_state.result
        risk = res['risk']
        
        if st.session_state.ai_engine.pattern_detector.risk_level >= 50:
            st.markdown(f"""
            <div class="pattern-alert">
                🚨 CẢNH BÁO: House Control {st.session_state.ai_engine.pattern_detector.risk_level}%<br>
                {res['house_warning']}
            </div>
            """, unsafe_allow_html=True)
        
        if risk['level'] == 'OK':
            status_class, status_text = 'status-ok', '✅ CÓ THỂ TEST'
        elif risk['level'] == 'MEDIUM':
            status_class, status_text = 'status-warn', '⚠️ CẨN THẬN'
        else:
            status_class, status_text = 'status-stop', '🛑 RỦI RO'
        
        st.markdown(f"""
        <div class="status-card {status_class}">
            {status_text} | Risk: {risk['score']}/100 | {risk['reason']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔮 3 SỐ CHÍNH")
        main_3 = res['main_3']
        st.markdown(f"""
        <div class="numbers-grid">
            <div class="num-card"><div class="num-value">{main_3[0]}</div><div class="num-label">Số 1</div></div>
            <div class="num-card"><div class="num-value">{main_3[1]}</div><div class="num-label">Số 2</div></div>
            <div class="num-card"><div class="num-value">{main_3[2]}</div><div class="num-label">Số 3</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎲 4 SỐ LÓT")
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
            st.markdown(f'<div class="info-box">💡 **Logic:** {res["logic"]}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ✅ Test")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            actual = st.text_input("Kết quả thực tế:", key="actual_result", placeholder="12864")
        with col2:
            if st.button("✅ GHI", type="primary", use_container_width=True):
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
                        st.success(f"🎉 TRÚNG! ({res['confidence']}%)")
                    else:
                        st.warning(f"❌ Trượt!")
                    st.rerun()
    
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#8b949e;padding:20px;'>TITAN AI v5.0 | ⚠️ Chơi có trách nhiệm</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()