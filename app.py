import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import requests
from io import StringIO
from datetime import datetime

# ================= CẤU HÌNH HỆ THỐNG =================
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
genai.configure(api_key=GEMINI_API_KEY) 

# CẤU HÌNH GOOGLE SHEETS
SHEET_ID = "1McocCyb3PRI6S0bgodyZlJyE_kjET55io1Zv966dZpA"
SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet=data"
# URL Web App anh vừa tạo ở bước trên (Dùng để nhập số)
DEPLOY_URL = "https://script.google.com/macros/s/AKfycbz_XXXXXXXXX/exec" 

st.set_page_config(page_title="TITAN ULTIMATE V9", layout="wide")

# --- STYLE GIAO DIỆN VIP ---
st.markdown("""
    <style>
    .stApp { background-color: #050505; color: white; }
    .card { background: #111; padding: 20px; border-radius: 15px; border: 1px solid #333; margin-bottom: 20px; }
    .win-status { color: #00ff00; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background: #222; border-radius: 5px; padding: 10px 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- QUẢN LÝ DỮ LIỆU ---
@st.cache_data(ttl=10)
def fetch_data():
    try:
        res = requests.get(f"{SHEET_CSV_URL}&v={datetime.now().timestamp()}")
        df = pd.read_csv(StringIO(res.text), header=None).astype(str)
        nums = []
        for col in df.columns:
            matches = df[col].str.extractall(r'(\d{5})')[0].tolist()
            nums.extend(matches)
        return nums
    except: return []

# --- GIAO DIỆN CHÍNH ---
def main():
    st.markdown("<h1 style='text-align: center; color: #00ff00;'>⚡ TITAN ULTIMATE V9 ⚡</h1>", unsafe_allow_html=True)
    
    # Khởi tạo bộ nhớ tạm cho lịch sử dự đoán
    if 'predict_history' not in st.session_state:
        st.session_state.predict_history = []

    tab1, tab2, tab3 = st.tabs(["📊 DASHBOARD", "✍️ NHẬP SỐ MỚI", "📜 LỊCH SỬ & TỶ LỆ"])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        data = fetch_data()
        if data:
            last_num = data[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Kỳ gần nhất", last_num)
            c2.metric("Tổng dữ liệu", len(data))
            
            # PHẦN DỰ ĐOÁN AI
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🤖 DỰ ĐOÁN AI KỲ TIẾP THEO")
            if st.button("KÍCH HOẠT SOI CẦU"):
                with st.spinner('AI đang tính toán...'):
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"Dữ liệu 5D: {data[-20:]}. Chốt 1 cặp 2D, 1 số 3D. Trả lời: 2D: XX, 3D: XXX"
                    res = model.generate_content(prompt).text
                    # Lưu vào lịch sử
                    st.session_state.predict_history.append({"time": datetime.now().strftime("%H:%M"), "pred": res, "status": "Waiting..."})
                    st.write(f"### {res}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Đang chờ dữ liệu từ Sheets...")

    # --- TAB 2: NHẬP SỐ ---
    with tab2:
        st.subheader("📝 Cập nhật kết quả kỳ mới")
        new_val = st.text_input("Nhập dãy 5 số vừa ra:", placeholder="Ví dụ: 10245")
        if st.button("XÁC NHẬN LƯU"):
            if len(new_val) == 5 and new_val.isdigit():
                try:
                    # Gửi dữ liệu lên Google Sheets thông qua Web App
                    requests.get(f"{DEPLOY_URL}?number={new_val}")
                    st.success(f"Đã lưu thành công số {new_val} vào hệ thống!")
                    st.cache_data.clear()
                except:
                    st.error("Lỗi kết nối Web App. Vui lòng kiểm tra lại Deploy URL.")
            else:
                st.error("Vui lòng nhập đúng 5 chữ số!")

    # --- TAB 3: TỶ LỆ THẮNG & LỊCH SỬ ---
    with tab3:
        st.subheader("📈 Thống kê hiệu quả dự đoán")
        if st.session_state.predict_history:
            # Thuật toán tính tỷ lệ thắng đơn giản
            history_df = pd.DataFrame(st.session_state.predict_history)
            win_count = sum(1 for x in st.session_state.predict_history if x['status'] == "WIN")
            total = len(st.session_state.predict_history)
            
            c1, c2 = st.columns(2)
            c1.metric("Tổng dự đoán", total)
            c2.metric("Tỷ lệ thắng %", f"{(win_count/total)*100:.1f}%" if total > 0 else "0%")
            
            st.table(history_df)
        else:
            st.info("Chưa có dữ liệu dự đoán để thống kê.")

if __name__ == "__main__":
    main()
