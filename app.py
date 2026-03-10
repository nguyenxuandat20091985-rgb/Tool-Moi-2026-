import streamlit as st
import pandas as pd
import time
from datetime import datetime
# Lưu ý: Đảm bảo các file config.py, database.py, algorithms.py nằm cùng thư mục
from config import Config
from database import DatabaseManager
from algorithms import TitanAI

# ==============================================================================
# CONFIGURATION & UI STYLE (Tối ưu theo bản v24)
# ==============================================================================
st.set_page_config(
    page_title=Config.APP_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Nền tảng Dark Mode v24 */
    .stApp { background: #010409; color: #e6edf3; }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(90deg, #0d1117 0%, #161b22 100%);
        padding: 20px; border-radius: 15px; border-left: 5px solid #58a6ff;
        margin-bottom: 20px; text-align: center;
    }
    .header-title { font-size: 32px; font-weight: 900; color: #58a6ff; }
    
    /* Prediction Card - UI v24 */
    .prediction-card {
        background: #0d1117; border: 2px solid #58a6ff;
        border-radius: 20px; padding: 30px; margin-top: 20px;
        box-shadow: 0 4px 20px rgba(88, 166, 255, 0.15);
    }
    
    /* Box số cực đại (95px) */
    .num-box {
        font-size: 95px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 15px;
        text-shadow: 3px 3px 0px #000; line-height: 1; margin: 10px 0;
    }
    
    /* Box số lót */
    .lot-box {
        font-size: 60px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 5px;
    }
    
    /* Trạng thái & Cảnh báo */
    .status-bar {
        padding: 15px; border-radius: 10px; text-align: center;
        font-weight: 800; font-size: 1.4rem; margin-bottom: 15px;
    }
    .pattern-alert {
        background: rgba(255, 88, 88, 0.1); border: 1px solid #ff5858;
        color: #ff5858; padding: 15px; border-radius: 10px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MAIN APPLICATION LOGIC
# ==============================================================================

def main():
    # Khởi tạo Session State (Giữ đúng cấu trúc v5.0)
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'ai_engine' not in st.session_state:
        st.session_state.ai_engine = TitanAI()
    if 'result' not in st.session_state:
        st.session_state.result = None

    # --- Header ---
    st.markdown(f"""
    <div class="header-card">
        <div class="header-title">🚀 {Config.APP_TITLE} v5.1</div>
        <div style="color: #8b949e;">{Config.APP_SUBTITLE} | Modular Edition</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar Stats ---
    with st.sidebar:
        st.header("📊 Chỉ số Accuracy")
        stats = st.session_state.db_manager.get_accuracy_stats()
        st.metric("Win Rate", f"{stats['win_rate']}%")
        st.metric("Tổng lần ghi nhận", stats['total'])
        if st.button("🗑️ Reset Dữ liệu"):
            st.session_state.db_manager.clear()
            st.rerun()

    # --- Stats Overview Row ---
    st.markdown("### 📈 Tổng quan hệ thống")
    s1, s2, s3, s4 = st.columns(4)
    with s1: st.metric("📦 Data History", len(st.session_state.db_manager.data))
    with s2: st.metric("🎯 Lần thắng", stats['wins'])
    with s3:
        risk_val = st.session_state.ai_engine.pattern_detector.risk_level if hasattr(st.session_state.ai_engine, 'pattern_detector') else 0
        st.metric("🏠 House Control", f"{risk_val}%")
    with s4: st.metric("🔥 Khuyên dùng", "Dàn 7 số")

    # --- Input Section ---
    st.markdown("### 📥 Dán kết quả KU")
    raw_input = st.text_area("Hệ thống tự lọc số kỳ và rác:", height=150, placeholder="Dán dãy số tại đây...")
    
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        analyze_btn = st.button("🔥 GIẢI MÃ MATRIX AI", type="primary", use_container_width=True)
    with c2:
        if st.button("📋 Tải Demo", use_container_width=True):
            st.info("Chức năng đang tải dữ liệu mẫu...")
    with c3:
        if st.button("🔄 Làm mới", use_container_width=True):
            st.rerun()

    # --- Logic xử lý (Modular) ---
    if analyze_btn and raw_input.strip():
        with st.spinner("🧠 Đang quét nhịp cầu và bẫy nhà cái..."):
            # Sử dụng DatabaseManager để lọc sạch dữ liệu
            numbers = st.session_state.db_manager.clean_data(raw_input)
            
            if not numbers:
                st.error("❌ Không tìm thấy dãy 5 số hợp lệ!")
            else:
                # Cập nhật database nội bộ
                updated_data, added = st.session_state.db_manager.add_numbers(numbers, st.session_state.db_manager.data)
                st.session_state.db_manager.data = updated_data
                
                # Gọi bộ não AI giải mã
                st.session_state.result = st.session_state.ai_engine.analyze(st.session_state.db_manager.data)
                st.rerun()

    # --- Display Results (Tối ưu UI v24) ---
    if st.session_state.result:
        res = st.session_state.result
        
        # 1. Cảnh báo rủi ro
        risk_score = res.get('risk', {}).get('score', 0)
        status_bg = "#238636" if risk_score < 40 else "#da3633"
        st.markdown(f"<div class='status-bar' style='background: {status_color if 'status_color' in locals() else status_bg};'>📢 TRẠNG THÁI: {res.get('decision', 'SẴN SÀNG')}</div>", unsafe_allow_html=True)

        if risk_score >= 50:
            st.markdown(f"""<div class="pattern-alert">🚨 CẢNH BÁO: Phát hiện bẫy nhịp ({risk_score}%)<br><small>{res.get('house_warning', 'Nhà cái đang đảo cầu liên tục.')}</small></div>""", unsafe_allow_html=True)

        # 2. Card kết quả chính (UI v24)
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        main_col, supp_col = st.columns([1.6, 1])
        
        with main_col:
            st.markdown("<p style='text-align:center; color:#8b949e; margin:0;'>💎 3 SỐ 5 TINH CHỦ LỰC</p>", unsafe_allow_html=True)
            # Lấy kết quả từ AI (main_3)
            st.markdown(f"<div class='num-box'>{res.get('main_3', '---')}</div>", unsafe_allow_html=True)
        
        with supp_col:
            st.markdown("<p style='text-align:center; color:#8b949e; margin:0;'>🛡️ DÀN LÓT (GIỮ VỐN)</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='lot-box'>{res.get('support_4', '----')}</div>", unsafe_allow_html=True)
        
        st.divider()
        st.write(f"🧠 **LOGIC PHÂN TÍCH:** {res.get('logic', 'Phân tích dựa trên thuật toán xác suất.')}")
        
        # Ghép dàn copy cho nhanh
        full_dan = "".join(sorted(set(str(res.get('main_3', '')) + str(res.get('support_4', '')))))
        st.text_input("📋 DÀN 7 SỐ KUBET (Copy):", full_dan)
        st.markdown("</div>", unsafe_allow_html=True)

        # 3. Phần Test Ghi Nhận (Để nâng cấp độ chính xác)
        st.markdown("### ✅ Xác nhận kết quả thực tế")
        actual_col, record_col = st.columns([3, 1])
        with actual_col:
            actual = st.text_input("Nhập kết quả kỳ vừa về để AI học nhịp:", placeholder="Ví dụ: 80586")
        with record_col:
            if st.button("📝 GHI NHẬN", use_container_width=True):
                if len(actual) == 5:
                    # Ghi nhận vào DB để tính Win Rate
                    is_win = set(res.get('main_3', '')).issubset(set(actual))
                    st.session_state.db_manager.record_test(res.get('main_3'), actual, is_win)
                    st.success("Đã ghi nhận để tối ưu thuật toán!")
                    time.sleep(1)
                    st.rerun()

    # Footer
    st.markdown(f"""
    <div style="text-align: center; color: #505864; padding: 30px; font-size: 13px;">
        {Config.APP_TITLE} | Bản quyền thuộc về AI Master Hub<br>
        Hệ thống Modular giúp nâng cấp thuật toán độc lập mà không ảnh hưởng UI.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
