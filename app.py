import streamlit as st
import pandas as pd
import re
from algorithms import PredictionEngine

# CONFIG GIAO DIỆN ELITE
st.set_page_config(page_title="TITAN v38.5 PREMIER", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&display=swap');
    .stApp { background: #020617; color: #f8fafc; font-family: 'Segoe UI', sans-serif; }
    .main-title { font-family: 'Orbitron', sans-serif; text-align: center; color: #38bdf8; text-transform: uppercase; letter-spacing: 3px; }
    
    .status-container { background: rgba(30, 41, 59, 0.5); border-radius: 15px; padding: 20px; border: 1px solid #1e293b; margin-bottom: 25px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    
    .number-grid { display: flex; justify-content: space-around; gap: 10px; margin: 25px 0; }
    .num-card { background: #0f172a; border: 2px solid #ef4444; border-radius: 20px; width: 30%; padding: 25px 5px; text-align: center; position: relative; overflow: hidden; }
    .num-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: #ef4444; box-shadow: 0 0 15px #ef4444; }
    .num-val { font-family: 'Orbitron', sans-serif; font-size: 65px; font-weight: 900; color: #f8fafc; line-height: 1; }
    .num-label { font-size: 10px; color: #94a3b8; text-transform: uppercase; margin-top: 10px; }
    
    .sup-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; }
    .sup-card { background: #1e293b; border: 1px solid #3b82f6; border-radius: 10px; padding: 12px; text-align: center; color: #60a5fa; font-size: 28px; font-weight: 900; }
    
    .risk-high { color: #ef4444; border-left: 4px solid #ef4444; padding-left: 10px; }
    .risk-medium { color: #f59e0b; border-left: 4px solid #f59e0b; padding-left: 10px; }
    .risk-low { color: #10b981; border-left: 4px solid #10b981; padding-left: 10px; }
    
    .stButton > button { background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%); color: white; border: none; border-radius: 12px; padding: 15px; font-weight: bold; width: 100%; transition: 0.3s; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4); }
</style>
""", unsafe_allow_html=True)

if "ai" not in st.session_state: st.session_state.ai = PredictionEngine()
if "db" not in st.session_state: st.session_state.db = []
if "res" not in st.session_state: st.session_state.res = None

def main():
    st.markdown("<h1 class='main-title'>TITAN v38.5 ELITE</h1>", unsafe_allow_html=True)

    # 1. INPUT AREA
    with st.container():
        raw = st.text_area("📊 DÁN KẾT QUẢ KỲ TRƯỚC (5 SỐ):", height=80, placeholder="Ví dụ: 71757\n81750...")
        if st.button("🚀 KÍCH HOẠT HỆ THỐNG PHÂN TÍCH"):
            nums = re.findall(r'\d{5}', raw)
            if nums:
                st.session_state.db = nums
                st.session_state.res = st.session_state.ai.predict(st.session_state.db)
                st.rerun()
            else: st.error("Lỗi định dạng số!")

    # 2. ANALYSIS RESULTS
    if st.session_state.res:
        r = st.session_state.res
        risk = r.get('risk', {'score': 0, 'level': 'LOW', 'reasons': []})
        
        # Risk & Logic Status
        risk_class = f"risk-{risk['level'].lower()}"
        st.markdown(f"""
        <div class="status-container">
            <div class="{risk_class}">
                <h3 style='margin:0'>LEVEL: {risk['level']} | SCORE: {risk['score']}/100</h3>
                <p style='margin:5px 0 0 0; font-size:13px;'>{" • ".join(risk['reasons'])}</p>
            </div>
            <div style='margin-top:10px; border-top: 1px solid #334155; padding-top:10px; font-size:12px; color:#94a3b8'>
                <b>LOGIC:</b> {r.get('logic')} | <b>TIN CẬY:</b> {r.get('confidence')}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3 SỐ CHÍNH - NEON DESIGN
        st.markdown("<div style='text-align:center; color:#94a3b8; font-weight:bold;'>🔮 3 SỐ CHÍNH (XUỐNG VỐN)</div>", unsafe_allow_html=True)
        main_html = "".join([f'<div class="num-card"><div class="num-val">{n}</div><div class="num-label">ƯU TIÊN</div></div>' for n in r['main_3']])
        st.markdown(f'<div class="number-grid">{main_html}</div>', unsafe_allow_html=True)

        # 4 SỐ LÓT
        st.markdown("<div style='text-align:center; color:#94a3b8; font-weight:bold;'>🎲 4 SỐ LÓT (GIỮ VỐN)</div>", unsafe_allow_html=True)
        sup_html = "".join([f'<div class="sup-card">{n}</div>' for n in r['support_4']])
        st.markdown(f'<div class="sup-grid">{sup_html}</div>', unsafe_allow_html=True)

        # 3. FEEDBACK
        st.markdown("---")
        c1, c2 = st.columns([2,1])
        actual = c1.text_input("Kỳ này về gì?", max_chars=5)
        if c2.button("✅ DẠY AI"):
            if len(actual) == 5:
                is_win = any(d in actual for d in r['main_3'])
                st.session_state.ai.update_learning(is_win)
                st.success("Hệ thống đã nạp nhịp cầu!")
            else: st.warning("Nhập đủ 5 số")

    # SIDEBAR STATUS
    with st.sidebar:
        st.header("🧠 AI MONITOR")
        status = st.session_state.ai.get_ai_status()
        st.metric("Tỉ lệ Win dự đoán", f"{status['wr']}%")
        st.write("Cấu trúc trọng số hiện tại:")
        st.json(status['weights'])
        if st.button("🗑️ Reset Dữ Liệu"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
