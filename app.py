# =============================================================================
# 💎 AI-QUANTUM PRO 2026 - XSMB REAL-TIME DASHBOARD
# Phiên bản: 3.0 - Fixed & Improved
# =============================================================================
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import time
import re

# =============================================================================
# 🔧 CẤU HÌNH HỆ THỐNG
# =============================================================================
st.set_page_config(
    page_title="💎 AI-QUANTUM PRO 2026",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Auto-refresh 60 giây
st_autorefresh(interval=60000, key="live_update", limit=None)

# =============================================================================
# 🎨 CSS STYLING
# =============================================================================
st.markdown("""
<style>
    .main { background-color: #0a0a0f; color: #ffffff; }
    .stApp { background-color: #0a0a0f; }
    
    .header-gold {
        background: linear-gradient(135deg, #D4AF37 0%, #FFD700 30%, #B8962E 70%, #D4AF37 100%);
        padding: 20px; border-radius: 15px; text-align: center; 
        color: #000; font-weight: bold; box-shadow: 0 4px 20px rgba(212, 175, 55, 0.4);
        margin-bottom: 20px;
    }
    
    .kq-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid #D4AF37; border-radius: 15px; 
        padding: 20px; margin: 10px 0;
        box-shadow: 0 8px 32px rgba(212, 175, 55, 0.2);
    }
    .kq-table { width: 100%; border-collapse: collapse; }
    .kq-row { border-bottom: 1px solid #2a2a4a; }
    .kq-row:hover { background: rgba(212, 175, 55, 0.1); }
    .kq-label { 
        background: transparent; color: #D4AF37; width: 25%; 
        padding: 12px 15px; text-align: left; font-weight: 600; font-size: 14px;
    }
    .kq-value { 
        padding: 12px 15px; text-align: right; font-size: 16px; 
        letter-spacing: 1px; color: #fff; font-family: 'Courier New', monospace;
    }
    .db-label { color: #FFD700; font-size: 18px; font-weight: bold; }
    .db-value { 
        color: #ff4b4b; font-size: 36px; font-weight: 800; 
        letter-spacing: 6px; text-shadow: 0 0 20px rgba(255, 75, 75, 0.5);
    }
    
    .pred-box { 
        border: 2px solid #D4AF37; border-radius: 12px; padding: 20px; 
        margin: 8px 0; background: linear-gradient(135deg, #111 0%, #1a1a2e 100%); 
        text-align: center;
    }
    .win-tag { 
        color: #00ff88; font-weight: bold; font-size: 14px; 
        animation: pulse 1.5s infinite; margin-top: 8px;
    }
    @keyframes pulse { 
        0%, 100% { opacity: 1; } 
        50% { opacity: 0.7; } 
    }
    
    .disclaimer {
        background: rgba(255, 107, 107, 0.15); border-left: 4px solid #ff6b6b;
        padding: 12px 15px; border-radius: 0 8px 8px 0; margin: 15px 0;
        font-size: 13px; color: #ffaaaa;
    }
    
    .success-box {
        background: rgba(0, 255, 136, 0.15); border-left: 4px solid #00ff88;
        padding: 12px 15px; border-radius: 0 8px 8px 0; margin: 15px 0;
        font-size: 13px; color: #88ffbb;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #D4AF37, #FFD700);
        color: #000; font-weight: bold; border: none; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 📡 MOCK DATA - DỮ LIỆU MẪU (FALLBACK)
# =============================================================================
def get_mock_data():
    """Trả về dữ liệu mẫu khi scraping fail"""
    today = datetime.now()
    # Tạo số ngẫu nhiên nhưng ổn định trong ngày
    seed = int(today.strftime("%Y%m%d"))
    np.random.seed(seed)
    
    return {
        "Đặc Biệt": f"{np.random.randint(10000, 99999):05d}",
        "Giải Nhất": f"{np.random.randint(10000, 99999):05d}",
        "Giải Nhì": [f"{np.random.randint(10000, 99999):05d}" for _ in range(2)],
        "Giải Ba": [f"{np.random.randint(10000, 99999):05d}" for _ in range(6)],
        "Giải Tư": [f"{np.random.randint(1000, 9999):04d}" for _ in range(4)],
        "Giải Năm": [f"{np.random.randint(1000, 9999):04d}" for _ in range(6)],
        "Giải Sáu": [f"{np.random.randint(100, 999):03d}" for _ in range(3)],
        "Giải Bảy": [f"{np.random.randint(0, 99):02d}" for _ in range(4)],
        "time": datetime.now().strftime("%H:%M:%S %d/%m"),
        "source": "Dữ liệu tham khảo",
        "is_mock": True
    }

# =============================================================================
# 📡 DATA CRAWLER - CẢI TIẾN VỚI RETRY & FALLBACK
# =============================================================================
@st.cache_data(ttl=300, show_spinner="🔄 Đang tải kết quả...")
def get_live_xsmb():
    """
    Crawl kết quả XSMB với multiple strategies và retry
    """
    urls = [
        "https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html",
        "https://xoso.com.vn/xsmb.html",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'vi-VN,vi;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
    
    for url_index, url in enumerate(urls):
        for attempt in range(3):  # Retry 3 lần
            try:
                res = requests.get(url, headers=headers, timeout=15)
                res.raise_for_status()
                soup = BeautifulSoup(res.content, 'html.parser')
                
                # Strategy 1: Tìm theo class
                result = extract_by_class(soup)
                if result and result.get("Đặc Biệt") and result["Đặc Biệt"] != "...":
                    result["source"] = f"Lấy từ {url.split('//')[1].split('/')[0]}"
                    result["is_mock"] = False
                    return result
                
                # Strategy 2: Tìm theo text patterns
                result = extract_by_pattern(soup)
                if result and result.get("Đặc Biệt") and result["Đặc Biệt"] != "...":
                    result["source"] = f"Lấy từ {url.split('//')[1].split('/')[0]}"
                    result["is_mock"] = False
                    return result
                    
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
    
    # Nếu tất cả fail, trả về mock data
    st.warning("⚠️ Không thể kết nối máy chủ. Hiển thị dữ liệu tham khảo.")
    mock_data = get_mock_data()
    return mock_data

def extract_by_class(soup):
    """Extract data by CSS classes"""
    def get_txt(classes):
        if isinstance(classes, str):
            classes = [classes]
        for cls in classes:
            item = soup.find("span", class_=cls)
            if item and item.text.strip():
                return item.text.strip()
        return "..."
    
    def get_list(classes_list):
        result = []
        for cls in classes_list:
            val = get_txt(cls)
            if val != "...":
                result.append(val)
        return result if result else ["..."]
    
    return {
        "Đặc Biệt": get_txt(["special-temp", "special", "db-value"]),
        "Giải Nhất": get_txt(["g1-temp", "g1", "g1-value"]),
        "Giải Nhì": [get_txt(f"g2_{i}-temp") or get_txt(f"g2-{i}") or "..." for i in range(2)],
        "Giải Ba": [get_txt(f"g3_{i}-temp") or get_txt(f"g3-{i}") or "..." for i in range(6)],
        "Giải Tư": [get_txt(f"g4_{i}-temp") or get_txt(f"g4-{i}") or "..." for i in range(4)],
        "Giải Năm": [get_txt(f"g5_{i}-temp") or get_txt(f"g5-{i}") or "..." for i in range(6)],
        "Giải Sáu": [get_txt(f"g6_{i}-temp") or get_txt(f"g6-{i}") or "..." for i in range(3)],
        "Giải Bảy": [get_txt(f"g7_{i}-temp") or get_txt(f"g7-{i}") or "..." for i in range(4)],
        "time": datetime.now().strftime("%H:%M:%S %d/%m"),
    }

def extract_by_pattern(soup):
    """Extract data by text patterns"""
    result = {}
    
    # Tìm tất cả các số
    all_numbers = soup.find_all(string=re.compile(r'^\d{2,5}$'))
    
    if all_numbers:
        # Giả sử số đầu tiên là đặc biệt
        result["Đặc Biệt"] = all_numbers[0].strip() if all_numbers else "..."
        result["Giải Nhất"] = all_numbers[1].strip() if len(all_numbers) > 1 else "..."
        result["time"] = datetime.now().strftime("%H:%M:%S %d/%m")
        return result
    
    return {}

# =============================================================================
# 🎲 LOGIC GỢI Ý SỐ
# =============================================================================
def generate_suggestions(all_loto: list) -> dict:
    """Tạo gợi ý số tham khảo"""
    seed = int(datetime.now().strftime("%Y%m%d%H%M")) % 10000
    np.random.seed(seed)
    
    bt = f"{np.random.randint(0, 100):02d}"
    st1 = f"{np.random.randint(0, 100):02d}"
    st2 = f"{np.random.randint(0, 100):02d}"
    xiên2 = f"{bt} - {np.random.randint(0, 100):02d}"
    dan_de = sorted([f"{i:02d}" for i in np.random.choice(100, 10, replace=False)])
    
    return {
        "bach_thu": bt,
        "song_thu": (st1, st2),
        "xien_2": xiên2,
        "dan_de": dan_de,
        "matched_bt": bt in all_loto,
        "matched_st": st1 in all_loto or st2 in all_loto
    }

# =============================================================================
# 🚀 MAIN APPLICATION
# =============================================================================
def main():
    # --- HEADER ---
    st.markdown('''
    <div class="header-gold">
        <h1 style="margin:0; font-size:24px;">💎 AI-QUANTUM PRO 2026</h1>
        <p style="margin:5px 0 0; font-size:14px;">🎯 XSMB Real-Time • Thống Kê • Tham Khảo</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # --- KẾT QUẢ TRỰC TIẾP ---
    st.markdown("### 📡 KẾT QUẢ XSMB TRỰC TIẾP")
    
    try:
        data = get_live_xsmb()
    except Exception as e:
        st.error(f"❌ Lỗi hệ thống: {str(e)}")
        data = None
    
    if not data:
        st.error("❌ Không thể tải kết quả. Vui lòng:")
        st.write("1. Kiểm tra kết nối internet")
        st.write("2. Thử bấm 'Làm mới' bên dưới")
        st.write("3. Đợi 1-2 phút và tải lại trang")
        
        if st.button("🔄 Thử tải lại ngay", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        return

    # Kiểm tra có phải mock data không
    if data.get("is_mock", False):
        st.info("ℹ️ Đang hiển thị dữ liệu tham khảo (demo mode)")
    else:
        st.success(f"✅ Kết quả tải thành công từ {data.get('source', 'nguồn')}")
    
    # Controls
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        st.caption(f"🕐 Cập nhật: **{data.get('time', 'N/A')}**")
    with col_ctrl2:
        if st.button("🔄 Làm mới", use_container_width=True):
            st.cache_data.clear()
            time.sleep(0.5)
            st.rerun()
    
    # Build result table
    html_table = '<div class="kq-container"><table class="kq-table">'
    
    # ĐẶC BIỆT
    db_val = data.get("Đặc Biệt", "...")
    if db_val and db_val != "...":
        html_table += f'''
        <tr class="kq-row" style="background: rgba(212, 175, 55, 0.2);">
            <td class="kq-label db-label">🏆 ĐẶC BIỆT</td>
            <td class="kq-value db-value">{db_val}</td>
        </tr>
        '''
    
    # Các giải còn lại
    priority_order = ["Giải Nhất", "Giải Nhì", "Giải Ba", "Giải Tư", "Giải Năm", "Giải Sáu", "Giải Bảy"]
    
    for label in priority_order:
        val = data.get(label)
        if not val:
            continue
            
        if isinstance(val, list):
            clean_vals = [x for x in val if x and x != "..."]
            if not clean_vals:
                continue
            v_str = " • ".join(clean_vals)
        else:
            if val == "...":
                continue
            v_str = val
            
        html_table += f'<tr class="kq-row"><td class="kq-label">{label}</td><td class="kq-value">{v_str}</td></tr>'
    
    html_table += '</table></div>'
    st.markdown(html_table, unsafe_allow_html=True)
    
    st.markdown("<div style='border-top: 2px dashed #D4AF37; margin: 25px 0;'></div>", unsafe_allow_html=True)
    
    # --- GỢI Ý THAM KHẢO ---
    st.markdown("### 🎯 GỢI Ý SỐ THAM KHẢO")
    
    # Extract all 2-digit numbers
    all_loto = []
    for k, v in data.items():
        if k in ["time", "source", "is_mock"]:
            continue
        if isinstance(v, list):
            all_loto.extend([x[-2:] for x in v if x and len(x) >= 2 and x != "..."])
        elif v and v != "...":
            all_loto.append(v[-2:])
    
    suggestions = generate_suggestions(all_loto)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        match_indicator = '<p class="win-tag">✅ KHỚP KQ</p>' if suggestions["matched_bt"] else ""
        st.markdown(f'''
        <div class="pred-box">
            <div style="font-size:13px; color:#aaa; margin-bottom:8px;">🎯 BẠCH THỦ</div>
            <div style="font-size:32px; color:#FFD700; font-weight:bold;">{suggestions["bach_thu"]}</div>
            {match_indicator}
        </div>
        ''', unsafe_allow_html=True)
    
    with c2:
        match_indicator = '<p class="win-tag">✅ KHỚP KQ</p>' if suggestions["matched_st"] else ""
        st.markdown(f'''
        <div class="pred-box">
            <div style="font-size:13px; color:#aaa; margin-bottom:8px;">🎯 SONG THỦ</div>
            <div style="font-size:26px; color:#FFD700; font-weight:600;">
                {suggestions["song_thu"][0]} • {suggestions["song_thu"][1]}
            </div>
            {match_indicator}
        </div>
        ''', unsafe_allow_html=True)
    
    with c3:
        st.markdown(f'''
        <div class="pred-box">
            <div style="font-size:13px; color:#aaa; margin-bottom:8px;">🎯 XIÊN 2</div>
            <div style="font-size:24px; color:#FFD700; font-weight:600;">{suggestions["xien_2"]}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Dàn đề
    st.markdown(f'''
    <div class="pred-box" style="margin-top: 5px;">
        <div style="font-size:14px; color:#aaa; margin-bottom:10px;">📋 DÀN ĐỀ 10 SỐ</div>
        <div style="font-size:20px; color:#fff; letter-spacing:2px;">{", ".join(suggestions["dan_de"])}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # DISCLAIMER
    st.markdown('''
    <div class="disclaimer">
        ⚠️ <b>LƯU Ý</b>: Kết quả chỉ mang tính chất tham khảo giải trí. 
        Xổ số là trò chơi may rủi ngẫu nhiên. Vui lòng chơi có trách nhiệm.
    </div>
    ''', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown('''
    <div style="text-align: center; color: #666; font-size: 12px; padding: 20px;">
        💎 AI-QUANTUM PRO 2026 • Built with Streamlit<br>
        <b>Chơi xổ số có trách nhiệm - 18+ only</b>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()