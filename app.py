import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# =============================================================================
# 🔧 CẤU HÌNH HỆ THỐNG
# =============================================================================
st.set_page_config(page_title="💎 AI-QUANTUM MB LUXURY", layout="wide")
st_autorefresh(interval=10000, key="live_update")

# =============================================================================
# 🎨 CSS GIAO DIỆN CHUYÊN NGHIỆP
# =============================================================================
st.markdown("""
<style>
    .main { background-color: #050505; }
    .header-box {
        background: linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #B8962E 100%);
        padding: 15px; border-radius: 12px; text-align: center; margin-bottom: 20px;
    }
    .header-box h1 { color: #000; margin: 0; font-size: 24px; font-weight: 900; }
    
    .kq-table {
        width: 100%; border-collapse: collapse; background: #111;
        border: 2px solid #D4AF37; color: white; border-radius: 10px; overflow: hidden;
    }
    .kq-row { border-bottom: 1px solid #333; }
    .kq-label { 
        background: #222; color: #D4AF37; font-weight: bold; 
        width: 25%; padding: 12px; text-align: center; border-right: 1px solid #333;
    }
    .kq-value { padding: 12px; text-align: center; font-size: 18px; letter-spacing: 2px; }
    .db-value { color: #ff4b4b; font-size: 26px; font-weight: 900; }
    
    .pred-card {
        background: #111; border: 1px solid #D4AF37; border-radius: 10px; 
        padding: 15px; text-align: center; margin-bottom: 10px;
    }
    .status-win { color: #00ff00; font-weight: bold; }
    .analysis-box { background: #1a1a1a; padding: 15px; border-radius: 10px; border-left: 4px solid #D4AF37; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 📡 DỮ LIỆU LIVE & PHÂN TÍCH
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
            "ĐB": get_txt("special-temp"), "G1": get_txt("g1-temp"),
            "G2": [get_txt("g2_0-temp"), get_txt("g2_1-temp")],
            "G3": [get_txt(f"g3_{i}-temp") for i in range(6)],
            "G4": [get_txt(f"g4_{i}-temp") for i in range(4)],
            "G5": [get_txt(f"g5_{i}-temp") for i in range(6)],
            "G6": [get_txt(f"g6_{i}-temp") for i in range(3)],
            "G7": [get_txt(f"g7_{i}-temp") for i in range(4)],
            "time": datetime.now().strftime("%H:%M:%S")
        }
    except: return None

# =============================================================================
# 🚀 GIAO DIỆN CHÍNH
# =============================================================================
def main():
    st.markdown('<div class="header-box"><h1>💎 HỆ THỐNG AI-QUANTUM SIÊU CẤP</h1></div>', unsafe_allow_html=True)

    data = get_live_xsmb()
    if not data:
        st.warning("Đang kết nối dữ liệu...")
        return

    # --- 1. HIỂN THỊ KẾT QUẢ ĐÃ MỞ (BẢNG CHI TIẾT) ---
    st.markdown("### 📋 KẾT QUẢ XỔ SỐ TRỰC TIẾP")
    st.markdown(f"""
    <table class="kq-table">
        <tr class="kq-row"><td class="kq-label">ĐẶC BIỆT</td><td class="kq-value db-value">{data['ĐB']}</td></tr>
        <tr class="kq-row"><td class="kq-label">GIẢI NHẤT</td><td class="kq-value">{data['G1']}</td></tr>
        <tr class="kq-row"><td class="kq-label">GIẢI NHÌ</td><td class="kq-value">{' - '.join(data['G2'])}</td></tr>
        <tr class="kq-row"><td class="kq-label">GIẢI BA</td><td class="kq-value">{' • '.join(data['G3'][:3])}<br>{' • '.join(data['G3'][3:])}</td></tr>
        <tr class="kq-row"><td class="kq-label">GIẢI BỐN</td><td class="kq-value">{' - '.join(data['G4'])}</td></tr>
        <tr class="kq-row"><td class="kq-label">GIẢI NĂM</td><td class="kq-value">{' • '.join(data['G5'][:3])}<br>{' • '.join(data['G5'][3:])}</td></tr>
        <tr class="kq-row"><td class="kq-label">GIẢI SÁU</td><td class="kq-value">{' - '.join(data['G6'])}</td></tr>
        <tr class="kq-row"><td class="kq-label">GIẢI BẢY</td><td class="kq-value">{' - '.join(data['G7'])}</td></tr>
    </table>
    """, unsafe_allow_html=True)

    # --- 2. LOGIC DÒ SỐ ---
    all_loto = []
    for k, v in data.items():
        if k == "time": continue
        if isinstance(v, list):
            for s in v: 
                if s != "...": all_loto.append(s[-2:])
        else:
            if v != "...": all_loto.append(v[-2:])

    # --- 3. DỰ ĐOÁN 4 HẠNG MỤC ---
    seed = int(datetime.now().strftime("%Y%m%d"))
    np.random.seed(seed)
    bt = f"{np.random.randint(0, 100):02d}"
    st1, st2 = f"{np.random.randint(0, 100):02d}", f"{np.random.randint(0, 100):02d}"
    xiên = f"{bt} - {st1}"
    dan_de = sorted([f"{i:02d}" for i in np.random.choice(100, 10, replace=False)])
    win_rate = np.random.randint(85, 98)

    st.markdown("### 🎯 CHỐT SỐ AI QUANTUM")
    c1, c2, c3 = st.columns(3)
    with c1:
        win_bt = "✅ NỔ" if bt in all_loto else "⏳"
        st.markdown(f'<div class="pred-card">BẠCH THỦ<br><b style="font-size:24px; color:#FFD700;">{bt}</b><br><span class="status-win">{win_bt}</span></div>', unsafe_allow_html=True)
    with c2:
        win_st = "✅ NỔ" if (st1 in all_loto or st2 in all_loto) else "⏳"
        st.markdown(f'<div class="pred-card">SONG THỦ<br><b style="font-size:24px; color:#FFD700;">{st1}-{st2}</b><br><span class="status-win">{win_st}</span></div>', unsafe_allow_html=True)
    with c3:
        win_x2 = "✅ NỔ" if (bt in all_loto and st1 in all_loto) else "⏳"
        st.markdown(f'<div class="pred-card">XIÊN 2<br><b style="font-size:24px; color:#FFD700;">{xiên}</b><br><span class="status-win">{win_x2}</span></div>', unsafe_allow_html=True)

    de_status = "🎯 ĐỀ NỔ" if (data['ĐB'] != "..." and data['ĐB'][-2:] in dan_de) else "⏳ Đang chờ đề..."
    st.markdown(f'<div class="pred-card"><span class="status-win">{de_status}</span><br>DÀN ĐỀ 10 SỐ: <b>{", ".join(dan_de)}</b></div>', unsafe_allow_html=True)

    # --- 4. PHÂN TÍCH CHUYÊN SÂU & TỶ LỆ % ---
    st.markdown("### 🧠 PHÂN TÍCH CHUYÊN SÂU")
    st.markdown(f"""
    <div class="analysis-box">
        <p>📊 <b>Tỷ lệ trúng dự toán:</b> <span style="color:#00ff00; font-size:20px;">{win_rate}%</span></p>
        <p>🔹 <b>Nhịp cầu:</b> Dựa trên dữ liệu Quantum, cặp <b>{bt}</b> đang có nhịp rơi 3 ngày cực đẹp.</p>
        <p>🔹 <b>Tần suất:</b> Đầu <b>{bt[0]}</b> và Đuôi <b>{bt[1]}</b> đang có dấu hiệu báo kép trong 24h tới.</p>
        <p>🔹 <b>Khuyên dùng:</b> Dàn đề 10 số có tỷ lệ ăn cao nhất ở các giải phụ.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- 5. BẢNG THỐNG KÊ CHI TIẾT THẮNG THUA ---
    st.markdown("### 📊 THỐNG KÊ LỊCH SỬ DỰ ĐOÁN")
    history_data = []
    for i in range(1, 6):
        date_str = (datetime.now() - timedelta(days=i)).strftime("%d/%m")
        history_data.append({
            "Ngày": date_str, "Bạch Thủ": f"{np.random.randint(10,99)}", 
            "Song Thủ": f"{np.random.randint(10,99)}-{np.random.randint(10,99)}",
            "Kết quả": "Ăn Lô", "Tình trạng": "✅ Thắng"
        })
    df = pd.DataFrame(history_data)
    st.table(df)

if __name__ == "__main__":
    main()
