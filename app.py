import streamlit as st
import pandas as pd
from datetime import datetime
import re
import time
from algorithms import PredictionEngine

# GIAO DIỆN MOBILE OPTIMIZED
st.set_page_config(page_title="TITAN v37.0 ULTRA", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background: #010409; color: #e6edf3; }
    .number-container { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 15px 0; }
    .number-box { background: #161b22; border: 2px solid #ff4b4b; border-radius: 12px; padding: 20px 5px; text-align: center; box-shadow: 0 4px 20px rgba(255,75,75,0.3); }
    .number-val { font-size: 45px; font-weight: 900; color: #ff4b4b; text-shadow: 0 0 10px rgba(255,75,75,0.5); }
    .support-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 20px;}
    .support-box { background: #0d1117; border: 1px solid #58a6ff; border-radius: 8px; padding: 15px 5px; text-align: center; color: #58a6ff; font-weight: bold; font-size: 28px; }
    .risk-banner { padding: 12px; border-radius: 10px; text-align: center; font-weight: bold; margin-bottom: 20px; border: 1px solid; }
</style>
""", unsafe_allow_html=True)

# KHỞI TẠO BỘ NHỚ
if "ai_engine" not in st.session_state: st.session_state.ai_engine = PredictionEngine()
if "lottery_db" not in st.session_state: st.session_state.lottery_db = []
if "last_prediction" not in st.session_state: st.session_state.last_prediction = None
if "history_log" not in st.session_state: st.session_state.history_log = []

def main():
    st.title("🎯 TITAN v37.0 ULTRA AI")

    # SIDEBAR
    with st.sidebar:
        st.header("🧠 AI MONITOR")
        try:
            status = st.session_state.ai_engine.get_ai_status()
            st.metric("Win Rate", f"{status['recent_win_rate']}%")
            st.metric("Memory", f"{status['pattern_memory_size']} patterns")
            st.write("Cấu trúc trọng số:")
            st.json(status['weights'])
        except: st.info("AI đang chờ dữ liệu đầu tiên...")
        
        if st.button("🗑️ RESET TOÀN BỘ"):
            st.session_state.clear()
            st.rerun()

    # TABS
    tab1, tab2 = st.tabs(["🚀 PHÂN TÍCH", "📜 NHẬT KÝ"])

    with tab1:
        raw_data = st.text_area("Nhập kết quả kỳ trước (5 số/dòng):", height=100, placeholder="87746\n23154...")
        
        if st.button("🚀 KÍCH HOẠT PHÂN TÍCH AI", use_container_width=True, type="primary"):
            new_nums = re.findall(r'\d{5}', raw_data)
            if new_nums:
                for n in new_nums:
                    if n not in st.session_state.lottery_db:
                        st.session_state.lottery_db.append(n)
                
                with st.spinner("Đang chạy giả lập Monte Carlo v37..."):
                    st.session_state.last_prediction = st.session_state.ai_engine.predict(st.session_state.lottery_db)
                st.rerun()
            else:
                st.error("Dữ liệu nhập vào không hợp lệ!")

        # HIỂN THỊ KẾT QUẢ DỰ ĐOÁN
        if st.session_state.last_prediction:
            p = st.session_state.last_prediction
            risk = p.get('risk_metrics', {'score': 0, 'level': 'LOW', 'reasons': []})
            
            # Risk Banner
            r_color = "#da3633" if risk['level'] == "HIGH" else "#d29922" if risk['level'] == "MEDIUM" else "#238636"
            st.markdown(f'<div class="risk-banner" style="color: {r_color}; border-color: {r_color}; background: {r_color}11">RISK: {risk["score"]}/100 | KHUYẾN NGHỊ: {risk["level"]}</div>', unsafe_allow_html=True)
            
            # Main 3 Numbers
            st.write("🔮 **3 SỐ CHÍNH (VÀO MẠNH)**")
            m_html = "".join([f'<div class="number-box"><div class="number-val">{n}</div><div style="font-size:10px; color:#8b949e">SỐ</div></div>' for n in p['main_3']])
            st.markdown(f'<div class="number-container">{m_html}</div>', unsafe_allow_html=True)

            # Support 4 Numbers
            st.write("🎲 **4 SỐ LÓT (GIỮ VỐN)**")
            s_html = "".join([f'<div class="support-box">{n}</div>' for n in p['support_4']])
            st.markdown(f'<div class="support-container">{s_html}</div>', unsafe_allow_html=True)
            
            # SỬA LỖI KEYERROR TẠI ĐÂY - Dùng .get để an toàn
            logic_text = p.get('logic', "Phân tích đa luồng AI")
            st.info(f"💡 Logic: {logic_text} | Tin cậy: {p.get('confidence', 0)}%")

            # Dạy AI
            st.markdown("---")
            col_in, col_btn = st.columns([2, 1])
            actual = col_in.text_input("Kết quả kỳ này (5 số):", max_chars=5)
            if col_btn.button("✅ DẠY AI", use_container_width=True):
                if len(actual) == 5:
                    is_win = any(d in actual for d in p['main_3'])
                    st.session_state.ai_engine.update_weights(is_win)
                    st.session_state.history_log.insert(0, {
                        "Time": datetime.now().strftime("%H:%M"),
                        "Dự đoán": ",".join(p['main_3']),
                        "Kết quả": actual,
                        "Trạng thái": "✅ THẮNG" if is_win else "❌ THUA"
                    })
                    st.success("AI đã học xong dữ liệu!")
                    time.sleep(1)
                    st.rerun()

    with tab2:
        if st.session_state.history_log:
            st.table(pd.DataFrame(st.session_state.history_log))
        else:
            st.info("Chưa có lịch sử.")

if __name__ == "__main__":
    main()
