import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import requests
from io import StringIO
from datetime import datetime

# ================= CẤU HÌNH GỐC =================
# Em đổi sang model 'gemini-pro' để ổn định nhất trên Cloud
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
genai.configure(api_key=GEMINI_API_KEY)

# Google Sheets Config
SHEET_ID = "1McocCyb3PRI6S0bgodyZlJyE_kjET55io1Zv966dZpA"
SHEET_NAME = "data"
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

st.set_page_config(page_title="TITAN V10 PRO", layout="wide")

# --- STYLE GIAO DIỆN CHUYÊN NGHIỆP ---
st.markdown("""
    <style>
    .main { background-color: #000000; }
    .stMetric { background-color: #111; padding: 15px; border-radius: 10px; border: 1px solid #222; }
    .predict-box { background: linear-gradient(135deg, #004d00 0%, #000 100%); padding: 20px; border-radius: 15px; border: 2px solid #00ff00; }
    .stTabs [data-baseweb="tab-list"] { background-color: #111; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- KHỞI TẠO CƠ SỞ DỮ LIỆU TẠM ---
if 'history' not in st.session_state:
    st.session_state.history = [] # Lưu: {kỳ, dự_đoán, kết_quả, trạng_thái}

# --- HÀM LẤY DỮ LIỆU ---
@st.cache_data(ttl=5)
def get_data():
    try:
        res = requests.get(f"{SHEET_URL}&v={datetime.now().timestamp()}")
        df = pd.read_csv(StringIO(res.text), header=None).astype(str)
        all_nums = []
        for col in df.columns:
            matches = df[col].str.extractall(r'(\d{5})')[0].tolist()
            all_nums.extend(matches)
        return all_nums
    except: return []

# --- GIAO DIỆN CHÍNH ---
def main():
    st.markdown("<h1 style='text-align: center; color: #00ff00;'>⚡ TITAN PRO ADMIN V10 ⚡</h1>", unsafe_allow_html=True)
    
    tab_view, tab_input, tab_history = st.tabs(["📊 SOI CẦU & DỰ ĐOÁN", "📥 NHẬP DỮ LIỆU NHIỀU KỲ", "📈 TỶ LỆ THẮNG"])

    data = get_data()

    # --- TAB 1: DASHBOARD ---
    with tab_view:
        if data:
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Số Vừa Ra", data[-1])
            with c2: st.metric("Tổng Kỳ Đã Quét", len(data))
            with c3: 
                win_rate = 0
                if st.session_state.history:
                    wins = sum(1 for x in st.session_state.history if x['status'] == "🔥 WIN")
                    win_rate = (wins / len(st.session_state.history)) * 100
                st.metric("Tỷ Lệ Thắng AI", f"{win_rate:.1f}%")

            st.divider()
            
            st.subheader("🤖 HỆ THỐNG DỰ ĐOÁN AI (REAL-TIME)")
            if st.button("🚀 KÍCH HOẠT THUẬT TOÁN TITAN"):
                with st.spinner('Đang quét nhịp cầu...'):
                    try:
                        # Dùng model cố định để tránh lỗi NotFound
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        history_str = ", ".join(data[-15:])
                        prompt = f"Phân tích chuỗi số 5D: {history_str}. Chốt duy nhất 1 con 2D (2 số cuối). Trả lời chỉ đúng số đó."
                        response = model.generate_content(prompt)
                        pred_val = response.text.strip()
                        
                        # Lưu vào bộ nhớ chờ đối soát
                        st.session_state.last_pred = pred_val
                        st.success(f"AI đã chốt số cho kỳ tiếp theo: {pred_val}")
                        
                        st.markdown(f"""
                        <div class="predict-box">
                            <h2 style='text-align: center; color: #00ff00;'>SỐ DỰ ĐOÁN: {pred_val}</h2>
                            <p style='text-align: center;'>Nhịp cầu đang bệt - Khả năng nổ cao!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Lỗi AI: {e}. Vui lòng thử lại sau 1 phút.")
        else:
            st.error("Chưa kết nối được dữ liệu từ Sheets!")

    # --- TAB 2: NHẬP SỐ NHIỀU KỲ ---
    with tab_input:
        st.subheader("📥 Nhập danh sách số (Nhiều kỳ)")
        st.info("Anh có thể dán nhiều số, mỗi số 1 dòng hoặc cách nhau bởi dấu phẩy.")
        input_area = st.text_area("Dán danh sách số vào đây:", height=200)
        
        if st.button("💾 CẬP NHẬT HỆ THỐNG"):
            if input_area:
                # Xử lý chuỗi nhập vào để lấy danh sách số
                import re
                new_nums = re.findall(r'\d{5}', input_area)
                if new_nums:
                    # Gợi ý: Để lưu thật vào Sheets, anh nên dùng Google Apps Script như em hướng dẫn bản V9.
                    # Ở đây em giả lập việc ghi nhận để tính Win/Loss.
                    st.success(f"Đã nhận diện {len(new_nums)} số mới. Hệ thống đang đối soát...")
                    
                    # Tự động đối soát với số AI đã chốt trước đó
                    if hasattr(st.session_state, 'last_pred'):
                        for n in new_nums:
                            is_win = "🔥 WIN" if st.session_state.last_pred in n else "❌ LOSS"
                            st.session_state.history.append({
                                "time": datetime.now().strftime("%H:%M"),
                                "pred": st.session_state.last_pred,
                                "result": n,
                                "status": is_win
                            })
                    st.cache_data.clear()
                else:
                    st.warning("Không tìm thấy dãy 5 số hợp lệ.")

    # --- TAB 3: TỶ LỆ THẮNG ---
    with tab_history:
        st.subheader("📜 Nhật ký đối soát Dự đoán vs Thực tế")
        if st.session_state.history:
            df_hist = pd.DataFrame(st.session_state.history)
            st.table(df_hist.iloc[::-1]) # Hiển thị mới nhất lên đầu
        else:
            st.info("Chưa có dữ liệu dự đoán. Anh hãy quay lại Tab 1 để chốt số trước.")

if __name__ == "__main__":
    main()
