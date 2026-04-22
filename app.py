import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
from streamlit_autorefresh import st_autorefresh

# =============================================================================
# 🔧 CẤU HÌNH & GIAO DIỆN
# =============================================================================
st.set_page_config(page_title="💎 AI-QUANTUM MB LIVE", layout="wide")
st_autorefresh(interval=10000, key="live_update")

st.markdown("""
<style>
    .main { background-color: #050505; }
    .header-box {
        background: linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #B8962E 100%);
        padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;
    }
    .live-status { color: #00ff00; font-weight: bold; text-align: center; margin-bottom: 10px; }
    
    /* Khu vực Lô đã về */
    .result-container {
        background: #111; border: 1px solid #D4AF37; border-radius: 10px;
        padding: 15px; margin-bottom: 20px; text-align: center;
    }
    .win-number {
        display: inline-block; background: #222; color: #FFD700;
        padding: 5px 12px; margin: 4px; border-radius: 5px;
        font-weight: bold; font-size: 18px; border: 1px solid #333;
    }
    
    /* Card dự đoán */
    .pred-card {
        background: #111; border: 2px solid #D4AF37; border-radius: 15px;
        padding: 20px; text-align: center; margin-bottom: 15px;
    }
    .number-highlight { color: #FFD700; font-size: 35px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 📡 DATA FETCHING
# =============================================================================
def get_live_data():
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
        }
    except: return None

# =============================================================================
# 🚀 MAIN APP
# =============================================================================
def main():
    st.markdown('<div class="header-box"><h1 style="color:black; margin:0;">💎 AI-QUANTUM: MIỀN BẮC</h1></div>', unsafe_allow_html=True)
    
    data = get_live_data()
    if not data:
        st.warning("Đang kết nối dữ liệu...")
        return

    # --- 1. XỬ LÝ SỐ ĐÃ VỀ (ĐƯA LÊN ĐẦU) ---
    all_numbers = []
    for k, v in data.items():
        if isinstance(v, list):
            all_numbers.extend([n for n in v if n != "..."])
        else:
            if v != "...": all_numbers.append(v)
    
    # Lấy 2 số cuối của các giải đã về
    loto_ve = [n[-2:] for n in all_numbers]

    if loto_ve:
        st.markdown("<p style='color:#D4AF37; font-weight:bold;'>🔥 KẾT QUẢ ĐÃ VỀ TỪ NHÀ ĐÀI:</p>", unsafe_allow_html=True)
        num_html = "".join([f'<div class="win-number">{n}</div>' for n in loto_ve])
        st.markdown(f'<div class="result-container">{num_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="live-status">🔴 ĐANG CHỜ GIỜ QUAY (18:15)...</p>', unsafe_allow_html=True)

    # --- 2. DỰ ĐOÁN AI (KHU VỰC CHÍNH) ---
    seed = int(datetime.now().strftime("%Y%m%d"))
    np.random.seed(seed)
    bt = f"{np.random.randint(0, 100):02d}"
    st1, st2 = f"{np.random.randint(0, 100):02d}", f"{np.random.randint(0, 100):02d}"

    st.markdown("<h3 style='color:#D4AF37; text-align:center;'>🎯 DỰ ĐOÁN AI QUANTUM</h3>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    # Tự động kiểm tra trúng/trượt
    status_bt = "✅ NỔ" if bt in loto_ve else "⏳ Chờ"
    status_st = "✅ NỔ" if (st1 in loto_ve or st2 in loto_ve) else "⏳ Chờ"

    with c1:
        st.markdown(f'<div class="pred-card"><p style="color:#D4AF37">BẠCH THỦ</p><div class="number-highlight">{bt}</div><p>{status_bt}</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="pred-card"><p style="color:#D4AF37">SONG THỦ</p><div class="number-highlight">{st1} - {st2}</div><p>{status_st}</p></div>', unsafe_allow_html=True)

    # --- 3. BẢNG CHI TIẾT (CHỈ HIỆN GIẢI ĐÃ CÓ) ---
    with st.expander("📝 XEM CHI TIẾT BẢNG GIẢI"):
        for giai, gia_tri in data.items():
            if isinstance(gia_tri, list):
                # Chỉ hiện những số đã có, ẩn dấu "..."
                hien_thi = " - ".join([n for n in gia_tri if n != "..."])
                if hien_thi:
                    st.write(f"**{giai}:** {hien_thi}")
            else:
                if gia_tri != "...":
                    st.write(f"**{giai}:** {gia_tri}")

if __name__ == "__main__":
    main()
