import streamlit as st
import pandas as pd
import re
from algorithms import PredictionEngine

# SUPREME UI CONFIG
st.set_page_config(page_title="TITAN v39.0 SUPREME", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&display=swap');
    .stApp { background: #020617; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    
    .supreme-header { font-family: 'Orbitron', sans-serif; text-align: center; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; font-weight: 900; margin-bottom: 20px; text-shadow: 0 0 30px rgba(56, 189, 248, 0.3); }
    
    .glass-card { background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 25px; backdrop-filter: blur(10px); margin-bottom: 20px; }
    
    .number-display { display: flex; justify-content: center; gap: 20px; margin: 30px 0; }
    .neon-box { background: #0f172a; border: 3px solid #f43f5e; border-radius: 25px; width: 120px; height: 160px; display: flex; flex-direction: column; align-items: center; justify-content: center; box-shadow: 0 0 25px rgba(244, 63, 94, 0.4); transition: 0.3s; }
    .neon-box:hover { transform: scale(1.05); box-shadow: 0 0 40px rgba(244, 63, 94, 0.6); }
    .num-large { font-family: 'Orbitron', sans-serif; font-size: 80px; font-weight: 900; color: #fff; line-height: 1; }
    .num-sub { font-size: 10px; color: #94a3b8; margin-top: 10px; letter-spacing: 2px; }
    
    .sup-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
    .sup-item { background: #1e293b; border: 1px solid #3b82f6; border-radius: 12px; padding: 15px; text-align: center; font-size: 30px; font-weight: 900; color: #3b82f6; box-shadow: inset 0 0 10px rgba(59, 130, 246, 0.2); }
    
    .risk-banner { padding: 15px; border-radius: 12px; font-weight: bold; text-align: center; border-left: 6px solid; margin-bottom: 20px; }
    .HIGH { background: rgba(244, 63, 94, 0.1); color: #f43f5e; border-color: #f43f5e; }
    .MEDIUM { background: rgba(245, 158, 11, 0.1); color: #f59e0b; border-color: #f59e0b; }
    .LOW { background: rgba(16, 185, 129, 0.1); color: #10b981; border-color: #10b981; }
</style>
""", unsafe_allow_html=True)

if "ai" not in st.session_state: st.session_state.ai = PredictionEngine()
if "db" not in st.session_state: st.session_state.db = []
if "res" not in st.session_state: st.session_state.res = None

def main():
    st.markdown("<h1 class='supreme-header'>TITAN v39.0 SUPREME</h1>", unsafe_allow_html=True)

    # SIDEBAR STATUS
    with st.sidebar:
        st.header("🛡️ AI SECURITY")
        try:
            status = st.session_state.ai.get_ai_status()
            st.metric("Tỉ lệ Chính xác", f"{status['win_rate']}%")
            st.success(f"Logic: {status['logic_version']}")
        except: st.error("AI đang khởi động...")
        if st.button("🗑️ Clear Cache"): 
            st.session_state.db = []
            st.rerun()

    # INPUT AREA
    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        raw = st.text_area("📊 NHẬP KẾT QUẢ KỲ TRƯỚC (5 SỐ):", height=100, placeholder="71757\n81750...")
        if st.button("🚀 KÍCH HOẠT PHÂN TÍCH HỆ THỐNG", use_container_width=True):
            nums = re.findall(r'\d{5}', raw)
            if nums:
                st.session_state.db = nums
                st.session_state.res = st.session_state.ai.predict(st.session_state.db)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # RESULTS AREA
    if st.session_state.res:
        r = st.session_state.res
        risk = r.get('risk')
        
        # Risk Header
        st.markdown(f"""
        <div class="risk-banner {risk['level']}">
            <h2 style='margin:0'>RỦI RO: {risk['score']}/100 - {risk['level']}</h2>
            <p style='margin:5px 0 0 0'>{" • ".join(risk['reasons'])}</p>
        </div>
        """, unsafe_allow_html=True)

        # Main 3 Numbers
        st.markdown("<div style='text-align:center; color:#94a3b8; font-weight:bold; letter-spacing:3px;'>🔮 3 SỐ CHÍNH (VÀO MẠNH)</div>", unsafe_allow_html=True)
        main_html = "".join([f'<div class="neon-box"><div class="num-large">{n}</div><div class="num-sub">PRIORITY</div></div>' for n in r['main_3']])
        st.markdown(f'<div class="number-display">{main_html}</div>', unsafe_allow_html=True)

        # Support 4 Numbers
        st.markdown("<div style='text-align:center; color:#94a3b8; font-weight:bold; letter-spacing:3px; margin-top:20px;'>🎲 4 SỐ LÓT (GIỮ VỐN)</div>", unsafe_allow_html=True)
        sup_html = "".join([f'<div class="sup-item">{n}</div>' for n in r['support_4']])
        st.markdown(f'<div class="sup-grid">{sup_html}</div>', unsafe_allow_html=True)

        # Info Footer
        st.markdown(f"""
        <div style='margin-top:30px; padding:15px; background:rgba(255,255,255,0.05); border-radius:10px; font-size:13px; text-align:center'>
            <b>Logic:</b> {r['logic']} | <b>Độ Tin Cậy:</b> {r['confidence']}% | <b>Data:</b> {len(st.session_state.db)} kỳ
        </div>
        """, unsafe_allow_html=True)

        # Feedback
        st.markdown("---")
        c1, c2 = st.columns([3,1])
        actual = c1.text_input("Xác nhận kết quả kỳ này (5 số):")
        if c2.button("✅ DẠY AI", use_container_width=True):
            if len(actual) == 5:
                is_win = any(d in actual for d in r['main_3'])
                st.session_state.ai.update_learning(is_win)
                st.success("AI đã cập nhật nhịp cầu mới!")
            else: st.warning("Nhập đủ 5 số")

if __name__ == "__main__":
    main()
