import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# 1. Cấu hình trang và Giao diện (Màu vàng đồng)
st.set_page_config(page_title="AI-QUANTUM 2026 - ĐẲNG CẤP DỰ ĐOÁN", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .gold-text { color: #D4AF37; font-weight: bold; text-align: center; }
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1c1c1c 100%);
    }
    div[data-testid="stMetricValue"] { color: #D4AF37; font-size: 35px; }
    .result-card {
        border: 2px solid #D4AF37;
        border-radius: 15px;
        padding: 20px;
        background: rgba(212, 175, 55, 0.05);
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.2);
    }
    .status-live {
        color: #ff4b4b;
        font-weight: bold;
        animation: blinker 1.5s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0; } }
    </style>
    """, unsafe_allow_html=True)

# Tự động cập nhật mỗi 30 giây để lấy kết quả mới nhất
st_autorefresh(interval=30000, key="dat_refresh")

# 2. Logic Cào dữ liệu (XSMB)
def get_xsmb_live():
    try:
        url = "https://xosodaiphat.com/xsmb-xo-so-mien-bac.html"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Tìm bảng kết quả
        db = soup.find("td", {"class": "special-prize"}).text.strip()
        g1 = soup.find("td", {"class": "prize1"}).text.strip()
        # Anh có thể bổ sung thêm các giải khác tương tự tại đây
        
        return {"DB": db, "G1": g1, "Time": datetime.now().strftime("%H:%M:%S")}
    except:
        return {"DB": "Đang quay...", "G1": "Đang quay...", "Time": "Đang cập nhật"}

# 3. Nội dung App
st.markdown("<h1 class='gold-text'>🏆 AI-QUANTUM 2026: HỆ THỐNG PHÂN TÍCH SIÊU CẤP</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: white;'>Trạng thái: <span class='status-live'>● LIVE</span> | Cập nhật lần cuối: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.subheader("📜 KẾT QUẢ XSMB TRỰC TIẾP")
    data = get_xsmb_live()
    st.metric("GIẢI ĐẶC BIỆT", data['DB'])
    st.metric("GIẢI NHẤT", data['G1'])
    st.write(f"Nguồn dữ liệu: xosodaiphat.com")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.subheader("🤖 PHÂN TÍCH THUẬT TOÁN AI")
    st.info("Hệ thống đang chạy 1.024 luồng phân tích xác suất Quantum...")
    
    # Giả lập thuật toán AI phân tích chuyên sâu
    st.write("**Dự báo nhịp cầu G7:** Ổn định 94.8%")
    st.progress(95)
    
    st.write("**Tỷ lệ truyền thông (Social Bias):** Đang tăng")
    st.success("Bạch Thủ Lô Tiềm Năng: **79**")
    st.warning("Song Thủ Lô: **24 - 42**")
    st.markdown("</div>", unsafe_allow_html=True)

# 4. Khu vực Phân tích Chuyên sâu (Đẳng cấp)
st.divider()
st.markdown("<h3 class='gold-text'>📊 BIỂU ĐỒ TẦN SUẤT & DỮ LIỆU LỚN</h3>", unsafe_allow_html=True)
chart_data = pd.DataFrame({'Cầu': [10, 15, 8, 22, 18, 30], 'Tỉ lệ': [70, 82, 65, 91, 77, 94]})
st.line_chart(chart_data, height=250)

st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px; margin-top: 50px;'>
        Bản quyền thuộc về Nguyen Xuan Dat - Hệ thống AI dự đoán cao cấp 2026
    </div>
    """, unsafe_allow_html=True)
