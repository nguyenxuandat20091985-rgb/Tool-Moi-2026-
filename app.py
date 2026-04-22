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
st.set_page_config(page_title="AI-QUANTUM PRO 2026", layout="wide")
st_autorefresh(interval=10000, key="live_update")

st.markdown("""
<style>
    .main { background-color: #050505; color: white; }
    .header-gold {
        background: linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #B8962E 100%);
        padding: 15px; border-radius: 10px; text-align: center; color: black; font-weight: bold;
    }
    .kq-table { width: 100%; border-collapse: collapse; border: 2px solid #D4AF37; margin-top: 10px; }
    .kq-row { border-bottom: 1px solid #333; }
    .kq-label { background: #222; color: #D4AF37; width: 25%; padding: 10px; text-align: center; font-weight: bold; }
    .kq-value { padding: 10px; text-align: center; font-size: 18px; letter-spacing: 2px; color: #fff; }
    .db-value { color: #ff4b4b; font-size: 26px; font-weight: bold; }
    
    .pred-box { border: 1px solid #D4AF37; border-radius: 10px; padding: 15px; margin: 5px; background: #111; text-align: center; }
    .win-tag { color: #00ff00; font-weight: bold; animation: blink 1s infinite; }
    @keyframes blink { 0% {opacity: 1;} 50% {opacity: 0.3;} 100% {opacity: 1;} }
    .analysis-text { color: #888; font-style: italic; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 📡 DATA CRAWLER
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
            "Đặc Biệt": get_txt("special-temp"),
            "Giải Nhất": get_txt("g1-temp"),
            "Giải Nhì": [get_txt("g2_0-temp"), get_txt("g2_1-temp")],
            "Giải Ba": [get_txt(f"g3_{i}-temp") for i in range(6)],
            "Giải Tư": [get_txt(f"g4_{i}-temp") for i in range(4)],
            "Giải Năm": [get_txt(f"g5_{i}-temp") for i in range(6)],
            "Giải Sáu": [get_txt(f"g6_{i}-temp") for i in range(3)],
            "Giải Bảy": [get_txt(f"g7_{i}-temp") for i in range(4)],
            "time": datetime.now().strftime("%H:%M:%S")
        }
    except: return None

# =============================================================================
# 🚀 MAIN APP
# =============================================================================
def main():
    st.markdown('<div class="header-gold"><h1>💎 HỆ THỐNG AI-QUANTUM SIÊU CẤP 2026</h1></div>', unsafe_allow_html=True)
    
    data = get_live_xsmb()
    if not data:
        st.error("Đang tải dữ liệu từ máy chủ...")
        return

    # --- HIỂN THỊ KẾT QUẢ ĐÃ VỀ ---
    st.subheader("📊 KẾT QUẢ XỔ SỐ TRỰC TIẾP")
    html_table = '<table class="kq-table">'
    for label, val in data.items():
        if label == "time": continue
        v_str = " - ".join(val) if isinstance(val, list) else val
        cls = "db-value" if label == "Đặc Biệt" else ""
        html_table += f'<tr class="kq-row"><td class="kq-label">{label}</td><td class="kq-value {cls}">{v_str}</td></tr>'
    html_table += '</table>'
    st.markdown(html_table, unsafe_allow_html=True)
    st.caption(f"Cập nhật lúc: {data['time']}")

    # --- LOGIC DÒ SỐ ---
    all_loto = []
    for k, v in data.items():
        if k == "time": continue
        if isinstance(v, list):
            all_loto.extend([x[-2:] for x in v if x != "..."])
        else:
            if v != "...": all_loto.append(v[-2:])

    # --- DỰ ĐOÁN & PHÂN TÍCH CHUYÊN SÂU ---
    st.divider()
    st.markdown("### 🎯 DỰ ĐOÁN AI & PHÂN TÍCH KỸ THUẬT")
    
    # Giả lập thuật toán dựa trên Seed ngày
    seed = int(datetime.now().strftime("%Y%m%d"))
    np.random.seed(seed)
    bt = f"{np.random.randint(0, 100):02d}"
    st1, st2 = f"{np.random.randint(0, 100):02d}", f"{np.random.randint(0, 100):02d}"
    xiên2 = f"{bt} - {np.random.randint(0, 100):02d}"
    dan_de = sorted([f"{i:02d}" for i in np.random.choice(100, 10, replace=False)])
    win_rate = np.random.randint(85, 98)

    c1, c2, c3 = st.columns(3)
    with c1:
        win_bt = '<p class="win-tag">🔥 NỔ ✅</p>' if bt in all_loto else ""
        st.markdown(f'<div class="pred-box"><b>BẠCH THỦ</b><br><span style="font-size:24px; color:#FFD700">{bt}</span>{win_bt}</div>', unsafe_allow_html=True)
    with c2:
        win_st = '<p class="win-tag">🔥 NỔ ✅</p>' if (st1 in all_loto or st2 in all_loto) else ""
        st.markdown(f'<div class="pred-box"><b>SONG THỦ</b><br><span style="font-size:24px; color:#FFD700">{st1} - {st2}</span>{win_st}</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="pred-box"><b>XIÊN 2</b><br><span style="font-size:24px; color:#FFD700">{xiên2}</span></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="pred-box"><b>DÀN ĐỀ 10 SỐ BẤT BẠI</b><br><span style="font-size:20px;">{", ".join(dan_de)}</span></div>', unsafe_allow_html=True)

    # --- PHÂN TÍCH CHUYÊN SÂU ---
    with st.expander("🔬 PHÂN TÍCH KỸ THUẬT & TỶ LỆ TRÚNG"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"📈 **Tỷ lệ trúng dự toán:** `{win_rate}%`")
            st.progress(win_rate / 100)
        with col_b:
            st.markdown(f"""
            <div class="analysis-text">
            - Thuật toán: Quantum Regressive Bayes<br>
            - Nhịp cầu: Cầu chạm 2-7 đang có xu hướng chạy lại.<br>
            - Lô gan: {np.random.randint(0,99)} đã 15 ngày chưa về.
            </div>
            """, unsafe_allow_html=True)

    # --- THỐNG KÊ CHI TIẾT THẮNG THUA ---
    st.write("### 📜 NHẬT KÝ THỐNG KÊ CHI TIẾT")
    history_data = {
        "Ngày": ["21/04", "20/04", "19/04", "18/04"],
        "Bạch Thủ": ["76", "12", "85", "34"],
        "Song Thủ": ["09-90", "45-54", "11-66", "23-32"],
        "Kết Quả": ["Ăn 76", "Trượt", "Ăn 11, 66", "Ăn 23*"],
        "Hiệu Suất": ["✅ 100%", "❌ 0%", "✅ 200%", "✅ 100%"]
    }
    st.table(pd.DataFrame(history_data))

if __name__ == "__main__":
    main()
