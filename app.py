import streamlit as st
import pandas as pd
from datetime import datetime
import re
import time
from algorithms import PredictionEngine

# CẤU HÌNH TRANG
st.set_page_config(page_title="TITAN v37.0 AI", layout="wide", initial_sidebar_state="collapsed")

# CSS GIAO DIỆN
st.markdown("""
<style>
    .stApp { background: #010409; color: #e6edf3; }
    .number-container { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 15px 0; }
    .number-box { background: #161b22; border: 2px solid #ff4b4b; border-radius: 12px; padding: 15px; text-align: center; box-shadow: 0 4px 15px rgba(255,75,75,0.3); }
    .number-val { font-size: 38px; font-weight: 900; color: #ff4b4b; }
    .support-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
    .support-box { background: #0d1117; border: 1px solid #58a6ff; border-radius: 8px; padding: 10px; text-align: center; color: #58a6ff; font-weight: bold; font-size: 24px; }
    .risk-banner { padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# KHỞI TẠO SESSION
if "ai_engine" not in st.session_state:
    st.session_state.ai_engine = PredictionEngine()
if "lottery_db" not in st.session_state:
    st.session_state.lottery_db = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "predictions_log" not in st.session_state:
    st.session_state.predictions_log = []

def main():
    st.title("🎯 TITAN v37.0 ULTRA AI")

    # SIDEBAR TRẠNG THÁI
    with st.sidebar:
        st.header("🧠 AI STATUS")
        try:
            status = st.session_state.ai_engine.get_ai_status()
            st.metric("Win Rate (10 kỳ)", f"{status['recent_win_rate']}%")
            st.write("Trọng số thuật toán:")
            st.json(status['weights'])
        except:
            st.error("AI Engine đang khởi động...")
        
        if st.button("🗑️ Reset Dữ Liệu"):
            st.session_state.clear()
            st.rerun()

    tab1, tab2 = st.tabs(["🚀 DỰ ĐOÁN", "📜 LỊCH SỬ"])

    with tab1:
        # Nhập liệu
        input_data = st.text_area("Nhập kết quả 5 số (ví dụ: 87746):", height=100)
        
        if st.button("🚀 PHÂN TÍCH AI", use_container_width=True, type="primary"):
            nums = re.findall(r'\d{5}', input_data)
            for n in nums:
                if n not in st.session_state.lottery_db:
                    st.session_state.lottery_db.append(n)
            
            if len(st.session_state.lottery_db) > 0:
                with st.spinner("AI đang giả lập Monte Carlo..."):
                    st.session_state.last_prediction = st.session_state.ai_engine.predict(st.session_state.lottery_db)
                st.rerun()
            else:
                st.error("Vui lòng nhập dữ liệu hợp lệ!")

        # Hiển thị kết quả
        if st.session_state.last_prediction:
            p = st.session_state.last_prediction
            risk = p.get('risk_metrics', {'score': 0, 'level': 'LOW', 'reasons': []})
            
            # Banner rủi ro
            color = "#238636" if risk['level'] == "LOW" else "#d29922" if risk['level'] == "MEDIUM" else "#da3633"
            st.markdown(f'<div class="risk-banner" style="border: 1px solid {color}; color: {color}; background: {color}22"> RISK: {risk["score"]}/100 - {risk["level"]} </div>', unsafe_allow_html=True)
            
            # Grid 3 số chính
            st.write("🔮 **3 SỐ CHÍNH (VÀO MẠNH)**")
            main_html = "".join([f'<div class="number-box"><div class="number-val">{n}</div></div>' for n in p['main_3']])
            st.markdown(f'<div class="number-container">{main_html}</div>', unsafe_allow_html=True)

            # Grid 4 số lót
            st.write("🎲 **4 SỐ LÓT (GIỮ VỐN)**")
            sup_html = "".join([f'<div class="support-box">{n}</div>' for n in p['support_4']])
            st.markdown(f'<div class="support-container">{sup_html}</div>', unsafe_allow_html=True)
            
            st.info(f"💡 Logic: {p['logic']} | Tin cậy: {p['confidence']}%")

            # Xác nhận kết quả để AI học
            st.markdown("---")
            actual = st.text_input("Nhập kết quả thực tế (5 số) để dạy AI:", max_chars=5)
            if st.button("✅ XÁC NHẬN & HỌC"):
                if len(actual) == 5:
                    is_win = any(d in actual for d in p['main_3'])
                    st.session_state.ai_engine.update_weights(is_win)
                    st.session_state.predictions_log.insert(0, {
                        "Giờ": datetime.now().strftime("%H:%M"),
                        "Dự đoán": ",".join(p['main_3']),
                        "Kết quả": actual,
                        "Status": "✅ THẮNG" if is_win else "❌ THUA"
                    })
                    st.success("AI đã ghi nhớ kết quả này!")
                    time.sleep(1)
                    st.rerun()

    with tab2:
        if st.session_state.predictions_log:
            st.table(pd.DataFrame(st.session_state.predictions_log))
        else:
            st.info("Chưa có lịch sử dự đoán.")

if __name__ == "__main__":
    main()
