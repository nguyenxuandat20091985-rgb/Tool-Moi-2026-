import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import requests
import re
from io import StringIO
from datetime import datetime

# --- 1. CẤU HÌNH GỐC (FIXED) ---
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
genai.configure(api_key=GEMINI_API_KEY)

SHEET_ID = "1McocCyb3PRI6S0bgodyZlJyE_kjET55io1Zv966dZpA"
# Link truy xuất thẳng vào sheet 'data' đã đổi tên
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet=data"

st.set_page_config(page_title="TITAN V10 FINAL", layout="wide")

# --- 2. GIAO DIỆN VIP ONE-PAGE ---
st.markdown("""
    <style>
    .main { background-color: #000; color: #fff; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; background: linear-gradient(45deg, #00ff00, #008000); color: black; font-weight: bold; border: none; }
    .status-card { padding: 20px; border-radius: 15px; background: #111; border: 1px solid #333; margin-bottom: 15px; }
    .predict-text { color: #00ff00; font-size: 45px; font-weight: bold; text-align: center; text-shadow: 0 0 10px #00ff00; }
    [data-testid="stMetricValue"] { color: #00ff00 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. KHỞI TẠO BỘ NHỚ ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_pred' not in st.session_state:
    st.session_state.last_pred = "ĐANG CHỜ..."

def get_data():
    try:
        # Thêm timestamp để tránh lấy dữ liệu cũ trong bộ nhớ đệm
        res = requests.get(f"{SHEET_URL}&v={datetime.now().timestamp()}", timeout=10)
        df = pd.read_csv(StringIO(res.text), header=None).astype(str)
        all_nums = []
        for col in df.columns:
            matches = df[col].str.extractall(r'(\d{5})')[0].tolist()
            all_nums.extend(matches)
        return all_nums
    except: return []

# --- 4. GIAO DIỆN CHÍNH ---
st.markdown("<h1 style='text-align: center; color: #00ff00;'>⚡ TITAN FINAL SUPER ADMIN ⚡</h1>", unsafe_allow_html=True)

data = get_data()

# CHIA 2 CỘT TỐI ƯU
col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.markdown("### 📥 NHẬP KẾT QUẢ MỚI")
    input_data = st.text_area("Dán danh sách số (nhiều kỳ cũng được):", height=120, placeholder="Ví dụ: 88231, 10024...")
    
    if st.button("💾 LƯU & ĐỐI SOÁT WIN/LOSS"):
        new_nums = re.findall(r'\d{5}', input_data)
        if new_nums:
            for n in new_nums:
                # So sánh 2 số cuối của dự đoán với 2 số cuối của kết quả thực tế
                pred = str(st.session_state.last_pred)
                real = str(n)[-2:]
                status = "🔥 WIN" if pred == real else "❌ LOSS"
                
                st.session_state.history.append({
                    "Kỳ": datetime.now().strftime("%H:%M"),
                    "Dự đoán 2D": pred,
                    "Thực tế": n,
                    "Kết quả": status
                })
            st.success(f"Đã đối soát xong {len(new_nums)} kỳ!")
        else:
            st.error("Không tìm thấy dãy 5 số hợp lệ!")

    st.divider()
    st.markdown("### 📊 CHỈ SỐ HỆ THỐNG")
    if data:
        wins = sum(1 for x in st.session_state.history if x['Kết quả'] == "🔥 WIN")
        total = len(st.session_state.history)
        win_rate = (wins / total * 100) if total > 0 else 0
        
        c1, c2 = st.columns(2)
        c1.metric("Kỳ gần nhất", data[-1])
        c2.metric("Tỷ lệ thắng %", f"{win_rate:.1f}%")
        st.metric("Tổng dữ liệu quét", len(data))
    else:
        st.warning("Đang kết nối Sheets...")

with col_right:
    st.markdown("### 🤖 TRÍ TUỆ AI CHỐT SỐ (2D)")
    st.markdown('<div class="status-card">', unsafe_allow_html=True)
    
    if st.button("🚀 KÍCH HOẠT SOI CẦU TITAN"):
        if not data:
            st.error("Chưa có dữ liệu để soi cầu!")
        else:
            with st.spinner('AI đang quét nhịp cầu...'):
                try:
                    # Fix model gemini-1.5-flash cực mạnh
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    history_context = ", ".join(data[-25:])
                    prompt = f"Phân tích nhịp số 5D: {history_context}. Dự đoán duy nhất 2 số cuối kỳ tiếp theo. Chỉ trả về 2 chữ số, không giải thích."
                    response = model.generate_content(prompt)
                    st.session_state.last_pred = response.text.strip()[:2] # Chỉ lấy 2 ký tự đầu
                except Exception as e:
                    st.error("Lỗi AI. Vui lòng thử lại sau 30 giây.")
    
    st.markdown(f'<p class="predict-text">{st.session_state.last_pred}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📜 LỊCH SỬ ĐỐI SOÁT")
    if st.session_state.history:
        # Hiển thị bảng lịch sử, số mới nhất lên đầu
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist.iloc[::-1], use_container_width=True, height=350)
    else:
        st.info("Chưa có lịch sử dự đoán hôm nay.")

# NÚT LÀM MỚI TOÀN BỘ
if st.sidebar.button("🗑 Xóa lịch sử phiên"):
    st.session_state.history = []
    st.rerun()
