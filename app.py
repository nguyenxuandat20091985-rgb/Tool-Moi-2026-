import streamlit as st
import pandas as pd
from datetime import datetime
import re
import time
from algorithms import PredictionEngine

# CONFIG
st.set_page_config(page_title="TITAN v37.0 AI", layout="wide")

# UI CSS
st.markdown("""
<style>
    .stApp { background: #010409; color: #e6edf3; }
    .number-container { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 15px 0; }
    .number-box { background: #161b22; border: 2px solid #ff4b4b; border-radius: 12px; padding: 15px; text-align: center; }
    .number-val { font-size: 38px; font-weight: 900; color: #ff4b4b; }
    .support-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
    .support-box { background: #0d1117; border: 1px solid #58a6ff; border-radius: 8px; padding: 10px; text-align: center; color: #58a6ff; font-weight: bold; font-size: 24px; }
</style>
""", unsafe_allow_html=True)

# SESSION
if "ai_engine" not in st.session_state: st.session_state.ai_engine = PredictionEngine()
if "lottery_db" not in st.session_state: st.session_state.lottery_db = []
if "last_prediction" not in st.session_state: st.session_state.last_prediction = None
if "predictions_log" not in st.session_state: st.session_state.predictions_log = []

def main():
    st.title("🎯 TITAN v37.0 ULTRA AI")
    
    with st.sidebar:
        status = st.session_state.ai_engine.get_ai_status()
        st.metric("Win Rate", f"{status['recent_win_rate']}%")
        st.json(status['weights'])
        if st.button("🗑️ Reset"):
            st.session_state.clear()
            st.rerun()

    tab1, tab2 = st.tabs(["🚀 DỰ ĐOÁN", "📜 LỊCH SỬ"])

    with tab1:
        raw_input = st.text_area("Nhập dữ liệu:", placeholder="87746...")
        if st.button("🚀 PHÂN TÍCH", use_container_width=True, type="primary"):
            nums = re.findall(r'\d{5}', raw_input)
            for n in nums:
                if n not in st.session_state.lottery_db: st.session_state.lottery_db.append(n)
            
            # CHẠY AI (Đồng bộ v37)
            pred = st.session_state.ai_engine.predict(st.session_state.lottery_db)
            st.session_state.last_prediction = pred
            st.rerun()

        if st.session_state.last_prediction:
            p = st.session_state.last_prediction
            # SỬA LỖI DÒNG 279: Sử dụng dict.get để an toàn
            risk = p.get('risk_metrics', {'score': 0, 'level': 'LOW', 'reasons': []})
            
            st.warning(f"RISK: {risk['score']}/100 - {risk['level']}")
            
            # Hiển thị 3 số chính
            st.write("🔮 **3 SỐ CHÍNH**")
            main_html = "".join([f'<div class="number-box"><div class="number-val">{n}</div></div>' for n in p['main_3']])
            st.markdown(f'<div class="number-container">{main_html}</div>', unsafe_allow_html=True)

            # Hiển thị 4 số lót
            st.write("🎲 **4 SỐ LÓT**")
            sup_html = "".join([f'<div class="support-box">{n}</div>' for n in p['support_4']])
            st.markdown(f'<div class="support-container">{sup_html}</div>', unsafe_allow_html=True)

            # Xác nhận kết quả
            st.markdown("---")
            actual = st.text_input("Nhập kết quả thực tế (5 số):", max_chars=5)
            if st.button("✅ XÁC NHẬN"):
                is_win = any(d in actual for d in p['main_3'])
                st.session_state.ai_engine.update_weights(is_win)
                st.session_state.predictions_log.insert(0, {"Time": datetime.now().strftime("%H:%M"), "Dự đoán": ",".join(p['main_3']), "Kết quả": actual, "Status": "✅" if is_win else "❌"})
                st.rerun()

    with tab2:
        if st.session_state.predictions_log:
            st.table(pd.DataFrame(st.session_state.predictions_log))

if __name__ == "__main__":
    main()
