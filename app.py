import streamlit as st
import pandas as pd
import time
from datetime import datetime
from config import Config
from database import DatabaseManager
from algorithms import TitanAI

# ==============================================================================
# CONFIGURATION & UI ADAPTIVE STYLE
# ==============================================================================
st.set_page_config(
    page_title=f"{Config.APP_TITLE} OMNI PRO",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Tối ưu đa thiết bị (Responsive Design)
st.markdown("""
<style>
    /* Nền Dark Mode chiều sâu */
    .stApp { background: #010409; color: #e6edf3; }
    
    /* Header thích ứng màn hình */
    .header-card {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        padding: 20px; border-radius: 15px; border-bottom: 4px solid #58a6ff;
        margin-bottom: 20px; text-align: center;
    }
    .header-title { font-size: calc(24px + 1.5vw); font-weight: 900; color: #58a6ff; }
    
    /* Card kết quả Matrix v24 */
    .prediction-card {
        background: #0d1117; border: 2px solid #30363d;
        border-radius: 20px; padding: 25px; margin-top: 10px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
    
    /* Số chủ lực - Tự động co chữ khi màn hình hẹp */
    .num-box {
        font-size: calc(50px + 4vw); 
        font-weight: 900; 
        color: #ff5858;
        text-align: center; 
        letter-spacing: 10px;
        text-shadow: 0 0 20px rgba(255, 88, 88, 0.3);
        line-height: 1.1; 
        margin: 15px 0;
    }
    
    /* Số lót */
    .lot-box {
        font-size: calc(30px + 2vw); 
        font-weight: 700; 
        color: #58a6ff;
        text-align: center; 
        letter-spacing: 5px;
    }
    
    /* Thanh trạng thái Win Rate */
    .status-bar {
        padding: 12px; border-radius: 50px; text-align: center;
        font-weight: 800; font-size: 1.2rem; margin-bottom: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Cảnh báo bẫy nhà cái */
    .trap-alert {
        background: rgba(255, 88, 88, 0.1); border-left: 5px solid #ff5858;
        color: #ffb8b8; padding: 15px; border-radius: 8px; 
        font-size: 0.9rem; margin-top: 10px;
    }
    
    /* Tối ưu mobile: Ẩn bớt khoảng trống thừa */
    @media (max-width: 600px) {
        .num-box { letter-spacing: 5px; }
        .stTextArea textarea { font-size: 14px; }
    }
</style>
""", unsafe_allow_html=True)

def main():
    # 1. Khởi tạo lõi (Modular)
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'ai_engine' not in st.session_state:
        st.session_state.ai_engine = TitanAI()
    if 'result' not in st.session_state:
        st.session_state.result = None

    # --- Header Section ---
    st.markdown(f"""
    <div class="header-card">
        <div class="header-title">🛡️ TITAN v24.6 OMNI PRO</div>
        <div style="color: #8b949e; font-size: 0.9rem;">Hệ thống Giải mã Matrix & Chống bẫy Nhà cái</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Statistics Row ---
    stats = st.session_state.db_manager.get_accuracy_stats()
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("📦 Lịch sử", f"{len(st.session_state.db_manager.data)} kỳ")
    with c2: st.metric("🎯 Win Rate", f"{stats['win_rate']}%")
    with c3:
        # Lấy Risk từ kết quả mới nhất
        risk_val = st.session_state.result['risk']['score'] if st.session_state.result else 0
        st.metric("🏠 House Control", f"{risk_val}%")

    # --- Input Section ---
    st.markdown("### 📥 Dán dữ liệu (Copy từ KU/Thabet)")
    raw_input = st.text_area("Dán chuỗi số tại đây (hệ thống tự lọc rác):", height=120)
    
    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        if st.button("🚀 GIẢI MÃ MATRIX AI (GEMINI 1.5 PRO)", type="primary", use_container_width=True):
            if raw_input.strip():
                with st.spinner("🧠 AI đang quét nhịp cầu và bẫy nhà cái..."):
                    # 1. Làm sạch & Lưu trữ
                    numbers = st.session_state.db_manager.clean_data(raw_input)
                    if numbers:
                        st.session_state.db_manager.data, _ = st.session_state.db_manager.add_numbers(numbers, st.session_state.db_manager.data)
                        # 2. Phân tích lõi
                        st.session_state.result = st.session_state.ai_engine.analyze(st.session_state.db_manager.data)
                        st.rerun()
                    else:
                        st.error("❌ Dữ liệu không hợp lệ!")
    with btn_col2:
        if st.button("🔄 Làm mới", use_container_width=True):
            st.session_state.result = None
            st.rerun()

    # --- Results Display Section ---
    if st.session_state.result:
        res = st.session_state.result
        
        # Thanh trạng thái thông minh
        win_rate_calc = res.get('win_rate', 0)
        status_color = "#238636" if win_rate_calc > 60 else "#8e6a00" if win_rate_calc > 40 else "#da3633"
        
        st.markdown(f"""
        <div class="status-bar" style="background: {status_color};">
            {res.get('decision', 'QUAN SÁT')} | ĐỘ TIN CẬY: {win_rate_calc}%
        </div>
        """, unsafe_allow_html=True)

        # Cảnh báo bẫy (Nếu có)
        if res.get('house_warning'):
            st.markdown(f"""<div class="trap-alert">⚠️ <b>PHÁT HIỆN BẪY:</b> {res['house_warning']}</div>""", unsafe_allow_html=True)

        # Card kết quả v24.6
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        
        # Hiển thị số chủ lực (Dùng cho cả mobile & desktop)
        st.markdown("<p style='text-align:center; color:#8b949e; margin-bottom:5px;'>💎 3 SỐ 5 TINH CHỦ LỰC</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='num-box'>{res.get('m3', '---')}</div>", unsafe_allow_html=True)
        
        st.markdown("<p style='text-align:center; color:#8b949e; margin:10px 0 5px 0;'>🛡️ DÀN LÓT GIỮ VỐN</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='lot-box'>{res.get('l4', '----')}</div>", unsafe_allow_html=True)
        
        st.divider()
        # Logic từ AI Gemini
        st.markdown(f"**🧠 PHÂN TÍCH AI:** *{res.get('logic', 'Chưa có dữ liệu logic')}*")
        
        # Dàn copy nhanh
        dan_7 = "".join(sorted(set(str(res.get('m3','')) + str(res.get('l4','')))))
        st.text_input("📋 DÀN 7 SỐ KUBET (Copy):", dan_7)
        st.markdown("</div>", unsafe_allow_html=True)

        # Ghi nhận kết quả thực tế
        st.markdown("---")
        with st.expander("✅ Xác nhận kết quả để AI học tập"):
            c_act, c_btn = st.columns([3, 1])
            actual = c_act.text_input("Nhập kết quả kỳ vừa về (5 số):", key="actual_input")
            if c_btn.button("Ghi nhận", use_container_width=True):
                if len(actual) == 5:
                    is_win = set(res.get('m3','')).issubset(set(actual))
                    st.session_state.db_manager.record_test(res.get('m3'), actual, is_win)
                    st.success("Đã ghi nhận dữ liệu!")
                    time.sleep(1)
                    st.rerun()

    # --- Sidebar Option ---
    with st.sidebar:
        st.title("⚙️ Cấu hình OMNI")
        st.write("Phiên bản: 24.6.0 Pro")
        st.markdown("---")
        if st.button("🗑️ Xóa sạch dữ liệu cũ"):
            st.session_state.db_manager.clear()
            st.rerun()

    # Footer
    st.markdown(f"""
    <div style="text-align: center; color: #505864; padding: 20px; font-size: 0.8rem;">
        Hệ thống Titan AI Master Hub | 2026<br>
        <i>Cảnh báo: Luôn tuân thủ nguyên tắc quản lý vốn 1-2-4.</i>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
