import streamlit as st
import pandas as pd
import re
from algorithms import PredictionEngine

# CẤU HÌNH GIAO DIỆN SIÊU ĐẸP
st.set_page_config(page_title="TITAN v38.0 PREMIER", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0f172a 0%, #020617 100%); color: white; }
    .status-card { background: rgba(30, 41, 59, 0.7); border-radius: 15px; padding: 20px; border: 1px solid #334155; margin-bottom: 20px; }
    .number-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
    .num-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 2px solid #ef4444; border-radius: 20px; padding: 25px 10px; text-align: center; box-shadow: 0 0 20px rgba(239, 68, 68, 0.2); }
    .num-val { font-size: 60px; font-weight: 900; color: #ef4444; text-shadow: 0 0 15px rgba(239, 68, 68, 0.5); }
    .sup-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
    .sup-card { background: #1e293b; border: 1px solid #3b82f6; border-radius: 12px; padding: 15px; text-align: center; color: #3b82f6; font-size: 24px; font-weight: bold; }
    .risk-high { border-left: 5px solid #ef4444; color: #f87171; }
    .risk-low { border-left: 5px solid #22c55e; color: #4ade80; }
</style>
""", unsafe_allow_html=True)

if "engine" not in st.session_state: st.session_state.engine = PredictionEngine()
if "db" not in st.session_state: st.session_state.db = []
if "last_res" not in st.session_state: st.session_state.last_res = None

def main():
    st.title("🎯 TITAN v38.0 PREMIER AI")
    st.caption("Hệ thống phân tích cầu 5D Bet cao cấp - Tích hợp Gemini Logic")

    # Khu vực nhập liệu
    with st.expander("📥 NHẬP KẾT QUẢ KỲ TRƯỚC", expanded=True):
        raw_input = st.text_area("Dán danh sách số (5 số/dòng):", height=100, placeholder="71757\n81750...")
        if st.button("🚀 KÍCH HOẠT AI PHÂN TÍCH", use_container_width=True):
            nums = re.findall(r'\d{5}', raw_input)
            if nums:
                st.session_state.db = nums
                st.session_state.last_res = st.session_state.engine.predict(st.session_state.db)
                st.rerun()

    # Hiển thị kết quả
    if st.session_state.last_res:
        res = st.session_state.last_res
        risk = res['risk']
        
        # Hiển thị rủi ro & Cảnh báo lừa cầu
        risk_class = "risk-high" if risk['level'] == "HIGH" else "risk-low"
        st.markdown(f"""
        <div class="status-card {risk_class}">
            <h3 style='margin:0'>RISK: {risk['score']}/100 - {risk['level']}</h3>
            <p style='margin:5px 0 0 0'>{' | '.join(risk['reasons']) if risk['reasons'] else 'Cầu đang đi ổn định'}</p>
        </div>
        """, unsafe_allow_html=True)

        # 3 Số chính
        st.subheader("🔮 3 SỐ CHÍNH (VÀO MẠNH)")
        st.markdown(f"""
        <div class="number-grid">
            <div class="num-card"><div class="num-val">{res['main_3'][0]}</div><div>SỐ 1</div></div>
            <div class="num-card"><div class="num-val">{res['main_3'][1]}</div><div>SỐ 2</div></div>
            <div class="num-card"><div class="num-val">{res['main_3'][2]}</div><div>SỐ 3</div></div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Số lót
        st.subheader("🎲 4 SỐ LÓT (GIỮ VỐN)")
        sup_html = "".join([f'<div class="sup-card">{n}</div>' for n in res['support_4']])
        st.markdown(f'<div class="sup-grid">{sup_html}</div>', unsafe_allow_html=True)

        # Thông tin AI
        st.markdown(f"**💡 Logic:** {res['logic']} | **Độ tin cậy:** {res['confidence']}%")
        
        # Phản hồi thực tế để dạy AI
        st.markdown("---")
        col1, col2 = st.columns([2,1])
        actual = col1.text_input("Nhập kết quả kỳ này (5 số):")
        if col2.button("✅ DẠY AI"):
            if len(actual) == 5:
                is_win = any(d in actual for d in res['main_3'])
                st.session_state.engine.update_learning(is_win)
                st.success("AI đã học nhịp cầu này!")
            else: st.warning("Vui lòng nhập đủ 5 số")

if __name__ == "__main__":
    main()
