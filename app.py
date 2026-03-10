import streamlit as st
import pandas as pd
import time
from datetime import datetime
from config import Config
from database import DatabaseManager
from algorithms import TitanAI

# ==============================================================================
# CONFIGURATION & UI STYLE (Tối ưu co giãn màn hình)
# ==============================================================================
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background: #010409; color: #e6edf3; }
    #MainMenu, footer, header { visibility: hidden; }
    
    .header-card {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        padding: 20px; border-radius: 15px; border-bottom: 4px solid #58a6ff;
        margin-bottom: 20px; text-align: center;
    }
    .header-title { font-size: calc(24px + 1.5vw); font-weight: 900; color: #58a6ff; }
    
    /* Box hiển thị số cực đại - Tự động co giãn */
    .num-box {
        font-size: calc(50px + 5vw); 
        font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 10px;
        text-shadow: 0 0 20px rgba(255, 88, 88, 0.4);
        line-height: 1.1; margin: 15px 0;
    }
    
    .lot-box {
        font-size: calc(30px + 2vw); 
        font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 5px;
    }

    .prediction-card {
        background: #0d1117; border: 2px solid #30363d;
        border-radius: 20px; padding: 25px; margin-top: 10px;
    }

    .status-bar {
        padding: 12px; border-radius: 10px; text-align: center;
        font-weight: 800; font-size: 1.2rem; margin-bottom: 10px;
    }
    
    .pattern-alert {
        background: rgba(255, 88, 88, 0.1); border-left: 5px solid #ff5858;
        color: #ffb8b8; padding: 15px; border-radius: 8px; margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'ai_engine' not in st.session_state:
        st.session_state.ai_engine = TitanAI()
    if 'result' not in st.session_state:
        st.session_state.result = None
    
    # Header
    st.markdown(f"""
    <div class="header-card">
        <div class="header-title">🛡️ {Config.APP_TITLE} v5.5</div>
        <div style="color: #8b949e;">{Config.APP_SUBTITLE} | OMNI PRO</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar & Stats Overview
    acc_stats = st.session_state.db_manager.get_accuracy_stats()
    st.markdown("### 📊 Chỉ số Hệ thống")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📦 Tổng kỳ", len(st.session_state.db_manager.data))
    with col2: st.metric("🎯 Win Rate", f"{acc_stats['win_rate']}%")
    with col3: 
        risk = st.session_state.result['risk']['score'] if st.session_state.result else 0
        st.metric("🏠 House Risk", f"{risk}%")
    with col4: st.metric("🔥 Gợi ý", "Dàn 7 số")
    
    # Input Section
    st.markdown("### 📥 Dán kết quả KU")
    raw_input = st.text_area("Hệ thống tự lọc rác:", height=120, key="data_input")
    
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: analyze_btn = st.button("🚀 GIẢI MÃ MATRIX AI", type="primary", use_container_width=True)
    with c2: 
        if st.button("🔄 Làm mới", use_container_width=True): st.rerun()
    with c3:
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.db_manager.clear()
            st.rerun()
    
    # Logic Xử lý
    if analyze_btn and raw_input.strip():
        with st.spinner("🧠 Đang quét nhịp cầu và bẫy nhà cái..."):
            numbers = st.session_state.db_manager.clean_data(raw_input)
            if numbers:
                st.session_state.db_manager.data, _ = st.session_state.db_manager.add_numbers(numbers, st.session_state.db_manager.data)
                # Chạy AI
                st.session_state.result = st.session_state.ai_engine.analyze(st.session_state.db_manager.data)
                st.rerun()
            else:
                st.error("❌ Không tìm thấy dãy 5 số hợp lệ!")

    # ==============================================================================
    # PHẦN HIỂN THỊ KẾ QUẢ (ĐÃ SỬA LỖI KHÔNG HIỆN SỐ)
    # ==============================================================================
    if st.session_state.result:
        res = st.session_state.result
        
        # 1. Cảnh báo Bẫy
        if res.get('house_warning'):
            st.markdown(f'<div class="pattern-alert">🚨 <b>PHÁT HIỆN BẪY:</b> {res["house_warning"]}</div>', unsafe_allow_html=True)
        
        # 2. Thanh trạng thái
        risk_score = res.get('risk', {}).get('score', 0)
        status_bg = "#238636" if risk_score < 40 else "#da3633"
        st.markdown(f'<div class="status-bar" style="background: {status_bg};">📢 QUYẾT ĐỊNH: {res.get("decision", "QUAN SÁT")}</div>', unsafe_allow_html=True)
        
        # 3. CARD HIỂN THỊ SỐ (KHỚP BIẾN m3 VÀ l4)
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        
        # Lấy số từ algorithms.py (Hỗ trợ cả tên biến cũ và mới để tránh lỗi)
        chinh = res.get('m3') or res.get('main_3') or '---'
        lot = res.get('l4') or res.get('support_4') or '----'
        
        st.markdown("<p style='text-align:center; color:#8b949e; margin:0;'>💎 3 SỐ 5 TINH CHỦ LỰC</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{chinh}</div>", unsafe_allow_html=True)
        
        st.markdown("<p style='text-align:center; color:#8b949e; margin:0;'>🛡️ DÀN LÓT GIỮ VỐN</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{lot}</div>", unsafe_allow_html=True)
        
        st.divider()
        st.write(f"🧠 **LOGIC AI:** {res.get('logic', 'Đang phân tích nhịp...')}")
        
        # Hộp copy dàn 7 số
        full_dan = "".join(sorted(set(str(chinh) + str(lot))))
        st.text_input("📋 DÀN 7 SỐ KUBET (Copy):", full_dan)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Test xác nhận
        st.markdown("---")
        st.markdown("### ✅ Test Độ Chính Xác")
        c_act, c_btn = st.columns([3, 1])
        with c_act:
            actual = st.text_input("Nhập kết quả vừa về:", placeholder="Ví dụ: 12864", key="act_val")
        with c_btn:
            if st.button("GHI NHẬN", type="primary", use_container_width=True):
                if len(actual) == 5:
                    is_win = set(chinh).issubset(set(actual))
                    st.session_state.db_manager.record_test(chinh, actual, is_win)
                    if is_win: st.success("🎉 TRÚNG!")
                    else: st.warning("❌ Trượt")
                    time.sleep(1)
                    st.rerun()

    # Footer
    st.markdown(f"""
    <div style="text-align: center; color: #94a3b8; padding: 20px; font-size: 12px;">
        {Config.APP_TITLE} v5.5 | AI Master Hub 2026<br>
        🔍 Pattern: Bệt | Đảo | Xoay | Bẫy nhịp<br>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
