import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.express as px
import requests
import re
from io import StringIO
from datetime import datetime

# ================= 1. CẤU HÌNH GỐC =================
API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
genai.configure(api_key=API_KEY) 

SHEET_ID = "1McocCyb3PRI6S0bgodyZlJyE_kjET55io1Zv966dZpA"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet=data"

st.set_page_config(page_title="TITAN V12 - RECOVERY PRO", layout="wide") 

# --- Giao diện Cyberpunk Recovery ---
st.markdown("""
    <style>
    .stApp { background-color: #000500; color: #00ff00; }
    .status-card { background: #001a00; padding: 20px; border: 2px solid #00ff00; border-radius: 15px; box-shadow: 0 0 15px #00ff00; }
    .stButton>button { background: #00ff00 !important; color: black !important; font-weight: 900 !important; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ================= 2. THUẬT TOÁN TITAN V12 (FIX LỖI THUA) =================
class TitanRecoveryAI:
    def analyze(self, data_list):
        if len(data_list) < 10: return None
        
        # Chuyển dữ liệu thành ma trận số
        matrix = []
        for v in data_list:
            digits = [int(d) for d in str(v) if d.isdigit()]
            if len(digits) == 5: matrix.append(digits)
        matrix = np.array(matrix)

        analysis = {}
        last_row = matrix[-1] # Kỳ gần nhất

        for i in range(10):
            # Tính tần suất 10 kỳ và 5 kỳ
            f10 = sum(1 for row in matrix[-10:] if i in row)
            f5 = sum(1 for row in matrix[-5:] if i in row)
            
            # ƯU TIÊN SỐ ĐANG BỆT (Khác với code cũ)
            is_beting = 1 if i in last_row else 0
            
            score = (f10 * 10) + (f5 * 20) + (is_beting * 50)
            
            analysis[i] = {"score": score, "f10": f10, "is_beting": is_beting}
        
        # Chọn 2 số có Score cao nhất (Thực sự Nóng)
        sorted_best = sorted(analysis.items(), key=lambda x: x[1]['score'], reverse=True)
        return sorted_best[:2]

# ================= 3. GIAO DIỆN CHÍNH =================
def main():
    st.markdown("<h1 style='text-align: center;'>⚡ TITAN V12: RECOVERY PRO ⚡</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Thuật toán săn cầu bệt - Chống ngược dòng</p>", unsafe_allow_html=True)

    if 'history' not in st.session_state: st.session_state.history = []
    if 'current_pred' not in st.session_state: st.session_state.current_pred = "---"

    @st.cache_data(ttl=5)
    def load_data():
        try:
            res = requests.get(f"{SHEET_URL}&v={datetime.now().timestamp()}", timeout=10)
            df = pd.read_csv(StringIO(res.text), header=None).astype(str)
            all_nums = []
            for col in df.columns:
                matches = re.findall(r'\d{5}', "".join(df[col].tolist()))
                all_nums.extend(matches)
            return all_nums
        except: return []

    data = load_data()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 ĐỐI SOÁT KẾT QUẢ")
        raw_in = st.text_area("Dán số vừa ra vào đây:", height=100)
        if st.button("CHECK WIN/LOSS"):
            matches = re.findall(r'\d{5}', raw_in)
            if matches and st.session_state.current_pred != "---":
                for m in matches:
                    p = st.session_state.current_pred
                    # Thắng nếu trúng 1 trong 2 số của cặp
                    win = "🔥 WIN" if (p[0] in m or p[1] in m) else "❌ LOSS"
                    st.session_state.history.append({"Time": datetime.now().strftime("%H:%M"), "Pred": p, "Real": m, "Res": win})
                st.success("Đã cập nhật tỷ lệ thắng!")

    with col2:
        st.subheader("🤖 AI CHỐT SỐ (TỐI ƯU)")
        if st.button("🚀 PHÂN TÍCH NHỊP CẦU MỚI"):
            ai = TitanRecoveryAI()
            best_two = ai.analyze(data)
            if best_two:
                pair = f"{best_two[0][0]}{best_two[1][0]}"
                st.session_state.current_pred = pair
                
                # Gọi Gemini tư vấn tâm lý và vốn
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"Lịch sử {data[-5:]}. AI chốt {pair}. Cho 1 lời khuyên vào tiền cực ngắn."
                    advice = model.generate_content(prompt).text
                    st.warning(f"Lời khuyên: {advice}")
                except: pass

        st.markdown(f"""
            <div class="status-card">
                <h3 style='text-align: center;'>CẶP SỐ CẦN ĐÁNH</h3>
                <h1 style='text-align: center; font-size: 70px;'>{st.session_state.current_pred}</h1>
            </div>
        """, unsafe_allow_html=True)

    # Bảng lịch sử và Tỷ lệ thắng
    st.divider()
    if st.session_state.history:
        h_df = pd.DataFrame(st.session_state.history)
        wins = sum(1 for x in st.session_state.history if x['Res'] == "🔥 WIN")
        st.metric("TỶ LỆ THẮNG HIỆN TẠI", f"{(wins/len(st.session_state.history))*100:.1f}%")
        st.table(h_df.iloc[::-1])

if __name__ == "__main__":
    main()
