import streamlit as st
import pandas as pd
from datetime import datetime
from algorithms import PredictionEngine # Import bộ não mới
import re

# Khởi tạo CSS cho giao diện Grid Mobile
st.set_page_config(page_title="TITAN v36.0 AI", layout="wide")
st.markdown("""
<style>
    .stApp { background: #010409; color: #e6edf3; }
    .number-container { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 10px; }
    .number-box { background: #0d1117; border: 2px solid #ff4b4b; border-radius: 12px; padding: 15px; text-align: center; }
    .number-val { font-size: 35px; font-weight: 800; color: #ff4b4b; }
    .support-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; }
    .support-box { background: #161b22; border: 1px solid #58a6ff; border-radius: 8px; padding: 10px; text-align: center; color: #58a6ff; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Khởi tạo Session
if "ai_engine" not in st.session_state: st.session_state.ai_engine = PredictionEngine()
if "lottery_db" not in st.session_state: st.session_state.lottery_db = []
if "last_prediction" not in st.session_state: st.session_state.last_prediction = None

def main():
    st.title("🎯 TITAN v36.0 - AI SELF-LEARNING")

    tab1, tab2 = st.tabs(["🚀 DỰ ĐOÁN", "💰 VỐN & AI LOG"])

    with tab1:
        raw_input = st.text_area("Nhập kết quả kỳ trước:", height=100, placeholder="87746")
        
        if st.button("🚀 PHÂN TÍCH AI", use_container_width=True, type="primary"):
            # Làm sạch dữ liệu
            new_nums = re.findall(r'\d{5}', raw_input)
            for n in new_nums:
                if n not in st.session_state.lottery_db:
                    st.session_state.lottery_db.append(n)
            
            # AI ra quyết định
            pred = st.session_state.ai_engine.predict(st.session_state.lottery_db)
            risk = st.session_state.ai_engine.calculate_risk(st.session_state.lottery_db)
            
            st.session_state.last_prediction = pred
            st.session_state.last_risk = risk
            st.rerun()

        if st.session_state.last_prediction:
            p = st.session_state.last_prediction
            r_score, r_level, _ = st.session_state.last_risk
            
            # Hiển thị Risk & Trọng số thuật toán hiện tại
            st.info(f"RISK: {r_score}/100 | KHUYẾN NGHỊ: {r_level}")
            
            st.write("🔮 **3 SỐ CHÍNH (VÀO MẠNH)**")
            cols_html = "".join([f'<div class="number-box"><div class="number-val">{n}</div></div>' for n in p['main_3']])
            st.markdown(f'<div class="number-container">{cols_html}</div>', unsafe_allow_html=True)
            
            st.write("🎲 **4 SỐ LÓT**")
            sup_html = "".join([f'<div class="support-box">{n}</div>' for n in p['support_4']])
            st.markdown(f'<div class="support-container">{sup_html}</div>', unsafe_allow_html=True)

            st.markdown("---")
            actual = st.text_input("Xác nhận kết quả kỳ này (5 số):")
            if st.button("✅ DẠY AI (XÁC NHẬN)"):
                if len(actual) == 5:
                    # Kiểm tra thắng thua
                    is_win = any(d in actual for d in p['main_3'])
                    # DẠY AI: Cập nhật trọng số dựa trên thực tế
                    method_to_reward = 'markov' if is_win else 'frequency' # Demo logic thưởng
                    st.session_state.ai_engine.update_weights(is_win, method_to_reward)
                    
                    if is_win: st.success("THẮNG! AI đã ghi nhớ quy luật này.")
                    else: st.warning("THUA! AI đang điều chỉnh lại thuật toán.")
                    time.sleep(1)
                    st.rerun()

    with tab2:
        st.write("### 🧠 Trạng thái trí tuệ AI")
        weights = st.session_state.ai_engine.weights
        st.write("Trọng số thuật toán hiện tại (AI đang ưu tiên):")
        st.json(weights)
        if st.button("🗑️ RESET AI"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    import time
    main()
