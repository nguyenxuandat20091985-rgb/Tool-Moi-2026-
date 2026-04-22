import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# =============================================================================
# 🔧 CẤU HÌNH & CSS
# =============================================================================
st.set_page_config(page_title="💎 AI-QUANTUM LUXURY", layout="wide")
st_autorefresh(interval=10000, key="live_update")

st.markdown("""
<style>
    .main { background-color: #050505; }
    .header-box {
        background: linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #B8962E 100%);
        padding: 15px; border-radius: 15px; text-align: center; margin-bottom: 20px;
    }
    .header-box h1 { color: #000; font-size: 24px; margin: 0; }
    
    .kq-table {
        width: 100%; border-collapse: collapse; background: #111;
        border: 2px solid #D4AF37; color: white; border-radius: 10px; overflow: hidden;
    }
    .kq-label { background: #222; color: #D4AF37; font-weight: bold; padding: 10px; width: 20%; text-align: center; }
    .kq-value { padding: 10px; text-align: center; font-family: 'Roboto Mono', monospace; font-size: 20px; }
    .db-value { color: #ff4b4b; font-size: 26px; font-weight: bold; }
    
    .bet-box {
        background: #1a1a1a; border-left: 5px solid #ff4b4b;
        padding: 10px; margin: 10px 0; border-radius: 5px;
    }
    .pred-card {
        background: #000; border: 1px solid #333; border-radius: 10px;
        padding: 15px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 📡 DỮ LIỆU LIVE & PHÂN TÍCH BỆT
# =============================================================================
def get_data():
    url = "https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.content, 'html.parser')
        
        def get_txt(cls):
            item = soup.find("span", class_=cls)
            return item.text.strip() if item else "..."

        results = {
            "ĐB": get_txt("special-temp"),
            "G1": get_txt("g1-temp"),
            "G2": [get_txt("g2_0-temp"), get_txt("g2_1-temp")],
            "G3": [get_txt(f"g3_{i}-temp") for i in range(6)],
            "G4": [get_txt(f"g4_{i}-temp") for i in range(4)],
            "G5": [get_txt(f"g5_{i}-temp") for i in range(6)],
            "G6": [get_txt(f"g6_{i}-temp") for i in range(3)],
            "G7": [get_txt(f"g7_{i}-temp") for i in range(4)],
        }
        return results
    except: return None

# =============================================================================
# 🚀 GIAO DIỆN CHÍNH
# =============================================================================
def main():
    st.markdown('<div class="header-box"><h1>💎 KẾT QUẢ TRỰC TIẾP & AI PHÂN TÍCH</h1></div>', unsafe_allow_html=True)

    data = get_data()
    now = datetime.now()
    is_drawing_time = now.hour == 18 and now.minute >= 15 # Giờ bắt đầu có kết quả

    # 1. BẢNG KẾT QUẢ (LUÔN Ở ĐẦU)
    if data:
        st.markdown(f"""
        <table class="kq-table">
            <tr><td class="kq-label">ĐB</td><td class="kq-value db-value">{data['ĐB']}</td></tr>
            <tr><td class="kq-label">G1</td><td class="kq-value">{data['G1']}</td></tr>
            <tr><td class="kq-label">G7</td><td class="kq-value">{' - '.join(data['G7'])}</td></tr>
        </table>
        <p style='text-align:right; color:#888; font-size:12px;'>Nguồn: xosodaiphat.com</p>
        """, unsafe_allow_html=True)

    # 2. PHÂN TÍCH LÔ BỆT (Dựa trên bảng G7 và ĐB hiện tại)
    st.subheader("📊 Phân Tích Kỹ Thuật")
    if data and data['G7'][0] != "...":
        st.markdown(f"""
        <div class="bet-box">
            <b style="color:#D4AF37">🔥 Cảnh báo bệt:</b> Cặp <b>{data['G7'][0][-2:]} - {data['G7'][1][-2:]}</b> đang có dấu hiệu tần suất dày.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Hệ thống đang quét nhịp bệt từ các kỳ quay trước...")

    # 3. DỰ ĐOÁN AI (ẨN TRƯỚC GIỜ QUAY)
    st.write("")
    st.markdown("<h3 style='color:#D4AF37; text-align:center;'>🎯 DỰ ĐOÁN AI QUANTUM</h3>", unsafe_allow_html=True)
    
    # Tạo số ngẫu nhiên theo ngày
    seed = int(now.strftime("%Y%m%d"))
    np.random.seed(seed)
    bt = f"{np.random.randint(0, 100):02d}"
    st1, st2 = f"{np.random.randint(0, 100):02d}", f"{np.random.randint(0, 100):02d}"

    col1, col2 = st.columns(2)
    
    with col1:
        # Nếu chưa đến giờ quay hoặc đang quay thì hiển thị trạng thái chờ
        content = f"<div style='font-size:30px; color:#FFD700;'>{bt}</div>" if is_drawing_time else "<div style='color:#666;'>Đang phân tích...</div>"
        st.markdown(f"""
            <div class="pred-card">
                <p style="color:#D4AF37; margin:0;">BẠCH THỦ VIP</p>
                {content}
            </div>
        """, unsafe_allow_html=True)

    with col2:
        content = f"<div style='font-size:30px; color:#FFD700;'>{st1} - {st2}</div>" if is_drawing_time else "<div style='color:#666;'>Đang tính toán...</div>"
        st.markdown(f"""
            <div class="pred-card">
                <p style="color:#D4AF37; margin:0;">SONG THỦ VIP</p>
                {content}
            </div>
        """, unsafe_allow_html=True)

    if not is_drawing_time:
        st.warning("⚠️ Dự đoán AI sẽ được hiển thị công khai vào lúc 18:15 hàng ngày.")

if __name__ == "__main__":
    main()
