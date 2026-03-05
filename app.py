import streamlit as st
import pandas as pd
import re
from algorithms import PredictionEngine

# CONFIG GIAO DIỆN LUXURY
st.set_page_config(page_title="TITAN v40.0 ULTIMATE", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&display=swap');
    .stApp { background: #010409; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; }
    .header-text { font-family: 'Orbitron', sans-serif; text-align: center; color: #ff3e3e; font-size: 2.5rem; font-weight: 900; text-shadow: 0 0 20px rgba(255, 62, 62, 0.5); margin-bottom: 10px; }
    
    /* Box Số Chính */
    .main-container { display: flex; justify-content: center; gap: 12px; margin: 25px 0; }
    .number-card { background: #0d1117; border: 2px solid #ff3e3e; border-radius: 20px; width: 30%; padding: 20px 0; text-align: center; box-shadow: 0 0 20px rgba(255, 62, 62, 0.2); }
    .number-text { font-family: 'Orbitron', sans-serif; font-size: 70px; font-weight: 900; color: #ffffff; line-height: 1; }
    .number-label { font-size: 10px; color: #8b949e; letter-spacing: 2px; margin-top: 5px; }
    
    /* Box Số Lót */
    .support-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 25px; }
    .support-item { background: #161b22; border: 1px solid #58a6ff; border-radius: 12px; padding: 12px; text-align: center; font-size: 30px; font-weight: 900; color: #58a6ff; }
    
    /* Banner Rủi Ro */
    .risk-box { padding: 15px; border-radius: 12px; margin-bottom: 20px; border-left: 6px solid; text-align: left; }
    .LEVEL-HIGH { background: rgba(248, 81, 73, 0.1); color: #f85149; border-color: #f85149; }
    .LEVEL-MEDIUM { background: rgba(210, 153, 34, 0.1); color: #d29922; border-color: #d29922; }
    .LEVEL-LOW { background: rgba(63, 185, 80, 0.1); color: #3fb950; border-color: #3fb950; }
</style>
""", unsafe_allow_html=True)

if "ai" not in st.session_state: st.session_state.ai = PredictionEngine()
if "db" not in st.session_state: st.session_state.db = []
if "res" not in st.session_state: st.session_state.res = None

def main():
    st.markdown("<h1 class='header-text'>TITAN v40.0</h1>", unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        st.header("🧠 AI MONITOR")
        stat = st.session_state.ai.get_ai_status()
        st.metric("Tỉ lệ Win dự đoán", f"{stat['win_rate']}%")
        st.info(f"Version: {stat['logic_version']}")
        if st.button("🗑️ Reset Dữ Liệu"):
            st.session_state.db = []
            st.rerun()

    # INPUT
    raw = st.text_area("📥 NHẬP KẾT QUẢ KỲ TRƯỚC (5 SỐ):", height=90, placeholder="71757\n81750...")
    if st.button("🚀 KÍCH HOẠT PHÂN TÍCH SUPREME", use_container_width=True):
        nums = re.findall(r'\d{5}', raw)
        if nums:
            st.session_state.db = nums
            st.session_state.res = st.session_state.ai.predict(st.session_state.db)
            st.rerun()
        else: st.error("Số không đúng định dạng!")

    # HIỂN THỊ KẾT QUẢ
    if st.session_state.res:
        r = st.session_state.res
        risk = r.get('risk')
        risk_lvl = risk['level']
        
        # Render Risk Banner (SỬA LỖI TYPERROR TẠI ĐÂY)
        st.markdown(f"""
        <div class="risk-box LEVEL-{risk_lvl}">
            <h3 style='margin:0'>RỦI RO: {risk['score']}/100 - {risk_lvl}</h3>
            <p style='margin:5px 0 0 0; font-size:13px;'>{" • ".join(risk['reasons'])}</p>
        </div>
        """, unsafe_allow_html=True)

        # Main 3 Numbers
        st.markdown("<div style='text-align:center; color:#8b949e; font-weight:bold;'>🔮 3 SỐ CHÍNH (VÀO MẠNH)</div>", unsafe_allow_html=True)
        main_html = "".join([f'<div class="number-card"><div class="number-text">{n}</div><div class="number-label">SỐ VIP</div></div>' for n in r['main_3']])
        st.markdown(f'<div class="main-container">{main_html}</div>', unsafe_allow_html=True)

        # Support 4 Numbers
        st.markdown("<div style='text-align:center; color:#8b949e; font-weight:bold; margin-top:15px;'>🎲 4 SỐ LÓT (GIỮ VỐN)</div>", unsafe_allow_html=True)
        sup_html = "".join([f'<div class="support-item">{n}</div>' for n in r['support_4']])
        st.markdown(f'<div class="support-grid">{sup_html}</div>', unsafe_allow_html=True)

        # Feedback
        st.markdown("---")
        c1, c2 = st.columns([3,1])
        actual = c1.text_input("Dạy AI: Kết quả kỳ này về gì?")
        if c2.button("✅ GHI NHẬN"):
            if len(actual) == 5:
                is_win = any(d in actual for d in r['main_3'])
                st.session_state.ai.update_learning(is_win)
                st.success("AI đã nạp nhịp cầu thành công!")
            else: st.warning("Nhập đủ 5 số")

if __name__ == "__main__":
    main()
