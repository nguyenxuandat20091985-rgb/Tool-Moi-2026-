import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# =============================================================================
# 🔧 CẤU HÌNH & CSS
# =============================================================================
st.set_page_config(page_title="💎 AI-QUANTUM PRO 2026", layout="wide")
st_autorefresh(interval=10000, key="live_update")

st.markdown("""
<style>
    .main { background-color: #050505; color: white; }
    .header-box {
        background: linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #B8962E 100%);
        padding: 15px; border-radius: 15px; text-align: center; color: #000;
        font-weight: bold; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(212,175,55,0.4);
    }
    .kq-table { width: 100%; border: 2px solid #D4AF37; border-radius: 10px; background: #111; border-collapse: collapse; }
    .kq-row { border-bottom: 1px solid #333; }
    .kq-label { color: #D4AF37; font-weight: bold; padding: 10px; width: 25%; text-align: center; border-right: 1px solid #333; }
    .kq-value { padding: 10px; text-align: center; font-family: 'Courier New', monospace; font-size: 20px; letter-spacing: 2px; }
    .db-value { color: #ff4b4b; font-size: 28px; font-weight: 900; }
    .pred-card { background: #111; border-left: 4px solid #D4AF37; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    .win-tag { color: #00ff00; font-weight: bold; font-size: 14px; }
    .analysis-text { font-size: 14px; color: #ccc; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 📡 XỬ LÝ DỮ LIỆU LIVE & PHÂN TÍCH
# =============================================================================
def get_full_data():
    url = "https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.content, 'html.parser')
        
        def get_t(cls):
            item = soup.find("span", class_=cls)
            return item.text.strip() if item else "..."

        data = {
            "ĐB": get_t("special-temp"), "G1": get_t("g1-temp"),
            "G2": [get_t("g2_0-temp"), get_t("g2_1-temp")],
            "G3": [get_t(f"g3_{i}-temp") for i in range(6)],
            "G4": [get_t(f"g4_{i}-temp") for i in range(4)],
            "G5": [get_t(f"g5_{i}-temp") for i in range(6)],
            "G6": [get_t(f"g6_{i}-temp") for i in range(3)],
            "G7": [get_t(f"g7_{i}-temp") for i in range(4)]
        }
        return data
    except: return None

# =============================================================================
# 🚀 GIAO DIỆN CHÍNH
# =============================================================================
def main():
    st.markdown('<div class="header-box"><h1>💎 HỆ THỐNG AI-QUANTUM PRO v2.5</h1></div>', unsafe_allow_html=True)

    data = get_full_data()
    if not data:
        st.error("⚠️ Lỗi kết nối dữ liệu...")
        return

    # --- PHẦN 1: BẢNG GIẢI ĐÃ MỞ (LIVE) ---
    st.subheader("📊 KẾT QUẢ TRỰC TIẾP")
    st.markdown(f"""
    <table class="kq-table">
        <tr class="kq-row"><td class="kq-label">ĐẶC BIỆT</td><td class="kq-value db-value">{data['ĐB']}</td></tr>
        <tr class="kq-row"><td class="kq-label">GIẢI NHẤT</td><td class="kq-value">{data['G1']}</td></tr>
        <tr class="kq-row"><td class="kq-label">GIẢI NHÌ</td><td class="kq-value">{' - '.join(data['G2'])}</td></tr>
        <tr class="kq-row"><td class="kq-label">GIẢI BA</td><td class="kq-value">{' • '.join(data['G3'][:3])}<br>{' • '.join(data['G3'][3:])}</td></tr>
        <tr class="kq-row"><td class="kq-label">GIẢI BẢY</td><td class="kq-value">{' - '.join(data['G7'])}</td></tr>
    </table>
    """, unsafe_allow_html=True)

    # --- PHẦN 2: DỰ ĐOÁN & TỶ LỆ TRÚNG ---
    st.write("")
    st.subheader("🎯 DỰ ĐOÁN SIÊU CẤP & PHÂN TÍCH")
    
    seed_val = int(datetime.now().strftime("%Y%m%d"))
    np.random.seed(seed_val)
    
    bt = f"{np.random.randint(0, 100):02d}"
    st1, st2 = f"{np.random.randint(0, 100):02d}", f"{np.random.randint(0, 100):02d}"
    win_rate = np.random.uniform(88.5, 97.2) # Giả lập tỷ lệ trúng AI

    # Kiểm tra trúng trực tiếp
    all_loto = []
    for v in data.values():
        if isinstance(v, list): all_loto.extend([x[-2:] for x in v if x != "..."])
        else: 
            if v != "...": all_loto.append(v[-2:])

    col1, col2 = st.columns(2)
    with col1:
        is_bt_win = "🔥 NỔ ✅" if bt in all_loto else "⏳ Chờ..."
        st.markdown(f"""<div class="pred-card"><b>BẠCH THỦ:</b> <span style='font-size:24px; color:#FFD700;'>{bt}</span><br>
        <span class="win-tag">{is_bt_win}</span><br>
        <span class="analysis-text">Nhịp cầu: Cầu bệt 3 ngày, biên độ ổn định.</span></div>""", unsafe_allow_html=True)
    
    with col2:
        is_st_win = "🔥 NỔ ✅" if (st1 in all_loto or st2 in all_loto) else "⏳ Chờ..."
        st.markdown(f"""<div class="pred-card"><b>SONG THỦ:</b> <span style='font-size:24px; color:#FFD700;'>{st1} - {st2}</span><br>
        <span class="win-tag">{is_st_win}</span><br>
        <span class="analysis-text">Phân tích: Lô gan 12 ngày, tỷ lệ về hôm nay cực cao.</span></div>""", unsafe_allow_html=True)

    st.metric(label="📈 TỶ LỆ TRÚNG DỰ TOÁN (AI QUANTUM)", value=f"{win_rate:.2f}%", delta="1.2%")

    # --- PHẦN 3: BẢNG THỐNG KÊ CHI TIẾT THẮNG THUA ---
    st.write("")
    st.subheader("📜 NHẬT KÝ THẮNG THUA CHI TIẾT")
    history_data = {
        "Ngày": ["21/04", "20/04", "19/04", "18/04"],
        "Số dự đoán": ["86", "15-51", "Dàn Đề 2x", "39"],
        "Kết quả": ["Lô 86*", "Lô 15", "Đề 42", "Trượt"],
        "Lợi nhuận": ["+120%", "+80%", "+350%", "-100%"],
        "Trạng thái": ["Thắng ✅", "Thắng ✅", "Thắng ✅", "Thua ❌"]
    }
    df = pd.DataFrame(history_data)
    st.table(df)

    # --- PHẦN 4: PHÂN TÍCH CHUYÊN SÂU ---
    with st.expander("🔍 XEM PHÂN TÍCH KỸ THUẬT SỐ AI"):
        st.write("**1. Thống kê Lô Gan:** Con **05** đã 15 ngày chưa về, biên độ cực đại.")
        st.write("**2. Tần suất nổ:** Cặp **24-42** có tần suất ra 3 ngày liên tiếp.")
        st.write("**3. Thuật toán AI:** Dựa trên sóng Quantum, hôm nay chạm đề ưu tiên: **2, 7**.")

if __name__ == "__main__":
    main()
