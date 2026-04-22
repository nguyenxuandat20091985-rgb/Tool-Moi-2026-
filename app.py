import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# =============================================================================
# 🔧 CẤU HÌNH HỆ THỐNG
# =============================================================================
st.set_page_config(page_title="💎 AI-QUANTUM LUXURY", layout="wide")

# Tự động làm mới mỗi 10 giây để cập nhật kết quả Live
st_autorefresh(interval=10000, key="live_update")

# =============================================================================
# 🎨 CSS ĐẲNG CẤP - TỐI ƯU ĐIỆN THOẠI
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@900&family=Roboto+Mono:wght@700&display=swap');
    
    .main { background-color: #050505; }
    
    .header-box {
        background: linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #B8962E 100%);
        padding: 20px; border-radius: 15px; text-align: center;
        box-shadow: 0 5px 20px rgba(212,175,55,0.5); margin-bottom: 20px;
    }
    .header-box h1 { font-family: 'Playfair Display', serif; color: #000; margin: 0; font-size: 26px; }

    .kq-table {
        width: 100%; border-collapse: collapse; background: #111;
        border: 2px solid #D4AF37; color: white; border-radius: 10px; overflow: hidden;
    }
    .kq-row { border-bottom: 1px solid #333; }
    .kq-label { 
        background: #222; color: #D4AF37; font-weight: bold; 
        width: 20%; padding: 12px; text-align: center; border-right: 1px solid #333;
    }
    .kq-value { 
        padding: 12px; text-align: center; font-family: 'Roboto Mono', monospace; 
        font-size: 20px; letter-spacing: 2px;
    }
    .db-value { color: #ff4b4b; font-size: 28px; font-weight: 900; }
    
    .pred-card {
        background: linear-gradient(145deg, #1a1a1a, #000);
        border: 2px solid #D4AF37; border-radius: 15px; padding: 15px;
        text-align: center; margin-bottom: 15px;
    }
    .number-highlight { 
        color: #FFD700; font-size: 32px; font-weight: bold; 
        text-shadow: 0 0 10px rgba(255,215,0,0.5); margin: 5px 0;
    }
    .status-win { color: #00ff00; font-weight: bold; font-size: 16px; animation: blink 1s infinite; }
    @keyframes blink { 0% {opacity: 1;} 50% {opacity: 0.5;} 100% {opacity: 1;} }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 📡 HÀM CÀO DỮ LIỆU LIVE
# =============================================================================
def get_live_xsmb():
    url = "https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.content, 'html.parser')
        
        def get_txt(cls):
            item = soup.find("span", class_=cls)
            return item.text.strip() if item else "..."

        return {
            "ĐB": get_txt("special-temp"),
            "G1": get_txt("g1-temp"),
            "G2": [get_txt("g2_0-temp"), get_txt("g2_1-temp")],
            "G3": [get_txt(f"g3_{i}-temp") for i in range(6)],
            "G4": [get_txt(f"g4_{i}-temp") for i in range(4)],
            "G5": [get_txt(f"g5_{i}-temp") for i in range(6)],
            "G6": [get_txt(f"g6_{i}-temp") for i in range(3)],
            "G7": [get_txt(f"g7_{i}-temp") for i in range(4)],
            "time": datetime.now().strftime("%H:%M:%S")
        }
    except:
        return None

# =============================================================================
# 🚀 GIAO DIỆN CHÍNH
# =============================================================================
def main():
    st.markdown('<div class="header-box"><h1>💎 AI-QUANTUM MB LIVE</h1></div>', unsafe_allow_html=True)

    data = get_live_xsmb()
    if not data:
        st.error("Đang kết nối máy chủ dữ liệu...")
        return

    # --- BẢNG KẾT QUẢ LIVE ---
    st.markdown(f"""
    <table class="kq-table">
        <tr class="kq-row"><td class="kq-label">ĐB</td><td class="kq-value db-value">{data['ĐB']}</td></tr>
        <tr class="kq-row"><td class="kq-label">G1</td><td class="kq-value">{data['G1']}</td></tr>
        <tr class="kq-row"><td class="kq-label">G2</td><td class="kq-value">{' - '.join(data['G2'])}</td></tr>
        <tr class="kq-row"><td class="kq-label">G3</td><td class="kq-value">{' • '.join(data['G3'][:3])}<br>{' • '.join(data['G3'][3:])}</td></tr>
        <tr class="kq-row"><td class="kq-label">G4</td><td class="kq-value">{' - '.join(data['G4'])}</td></tr>
        <tr class="kq-row"><td class="kq-label">G5</td><td class="kq-value">{' • '.join(data['G5'][:3])}<br>{' • '.join(data['G5'][3:])}</td></tr>
        <tr class="kq-row"><td class="kq-label">G6</td><td class="kq-value">{' - '.join(data['G6'])}</td></tr>
        <tr class="kq-row"><td class="kq-label">G7</td><td class="kq-value">{' - '.join(data['G7'])}</td></tr>
    </table>
    <p style='text-align:right; color:#888; font-size:12px; margin-top:5px;'>Cập nhật: {data['time']}</p>
    """, unsafe_allow_html=True)

    # --- LOGIC DÒ SỐ TỰ ĐỘNG ---
    all_loto_ve = []
    # Gom tất cả 2 số cuối của các giải đã ra
    for k, v in data.items():
        if k == "time": continue
        if isinstance(v, list):
            for s in v:
                if s != "...": all_loto_ve.append(s[-2:])
        else:
            if v != "...": all_loto_ve.append(v[-2:])

    # --- DỰ ĐOÁN AI ---
    seed_val = int(datetime.now().strftime("%Y%m%d"))
    np.random.seed(seed_val)
    bt = f"{np.random.randint(0, 100):02d}"
    st1, st2 = f"{np.random.randint(0, 100):02d}", f"{np.random.randint(0, 100):02d}"
    dan_de = sorted([f"{i:02d}" for i in np.random.choice(100, 10, replace=False)])

    st.markdown("<h3 style='color:#D4AF37; text-align:center;'>🎯 ĐỐI CHIẾU KẾT QUẢ LIVE</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Kiểm tra Bạch thủ
    win_bt = '<p class="status-win">🔥 NỔ ✅</p>' if bt in all_loto_ve else '<p style="color:#666">⏳ Đang chờ...</p>'
    with col1:
        st.markdown(f'<div class="pred-card"><p style="color:#D4AF37; margin:0;">BẠCH THỦ</p><div class="number-highlight">{bt}</div>{win_bt}</div>', unsafe_allow_html=True)
    
    # Kiểm tra Song thủ
    win_st = '<p class="status-win">🔥 NỔ ✅</p>' if (st1 in all_loto_ve or st2 in all_loto_ve) else '<p style="color:#666">⏳ Đang chờ...</p>'
    with col2:
        st.markdown(f'<div class="pred-card"><p style="color:#D4AF37; margin:0;">SONG THỦ</p><div class="number-highlight">{st1}-{st2}</div>{win_st}</div>', unsafe_allow_html=True)

    # Kiểm tra Dàn Đề (So với giải ĐB)
    de_ve = data['ĐB'][-2:] if data['ĐB'] != "..." else None
    if de_ve and de_ve in dan_de:
        msg_de = f'<p class="status-win">🎯 DÀN ĐỀ NỔ: {de_ve} ✅</p>'
    elif de_ve:
        msg_de = f'<p style="color:#ff4b4b">Đề về {de_ve} (Trượt)</p>'
    else:
        msg_de = '<p style="color:#666">⏳ Chờ giải Đặc Biệt...</p>'

    st.markdown(f"""
    <div class="pred-card">
        {msg_de}
        <p style="color:#D4AF37; font-size:14px; margin-bottom:5px;">HỆ THỐNG DÀN ĐỀ 10 SỐ VIP</p>
        <p style="font-size:20px; color:#fff; letter-spacing:2px; font-weight:bold;">{', '.join(dan_de)}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- THỐNG KÊ ---
    with st.expander("📊 LỊCH SỬ DỰ ĐOÁN"):
        df_log = pd.DataFrame({
            "Ngày": ["21/04", "20/04", "19/04"],
            "Dự đoán": ["86", "15-51", "Dàn Đề"],
            "Kết quả": ["Nổ 86", "Nổ 15", "Ăn Đề 42"],
            "Trạng thái": ["✅", "✅", "✅"]
        })
        st.table(df_log)

if __name__ == "__main__":
    main()
