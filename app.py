# =============================================================================
# 💎 AI-QUANTUM PRO 2026 - XSMB REAL-TIME & STATISTICS
# Phiên bản: 4.0 - Full Features
# =============================================================================
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import time
import json
import os

# =============================================================================
# 🔧 CẤU HÌNH
# =============================================================================
st.set_page_config(
    page_title="💎 AI-QUANTUM PRO 2026",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh 120 giây
st_autorefresh(interval=120000, key="live_update", limit=None)

# =============================================================================
# 🎨 CSS
# =============================================================================
st.markdown("""
<style>
    .main { background-color: #0a0a0f; color: #ffffff; }
    .header-gold {
        background: linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #B8962E 100%);
        padding: 20px; border-radius: 15px; text-align: center; 
        color: #000; font-weight: bold; margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(212, 175, 55, 0.5);
    }
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #D4AF37; border-radius: 12px;
        padding: 15px; text-align: center; margin: 5px;
    }
    .stat-value { font-size: 28px; font-weight: bold; color: #FFD700; }
    .stat-label { font-size: 12px; color: #888; margin-top: 5px; }
    .win { color: #00ff88; }
    .loss { color: #ff4b4b; }
    .pred-box {
        border: 2px solid #D4AF37; border-radius: 12px;
        padding: 20px; background: linear-gradient(135deg, #111, #1a1a2e);
        text-align: center; margin: 8px 0;
    }
    .iframe-container {
        border: 2px solid #D4AF37; border-radius: 15px;
        overflow: hidden; box-shadow: 0 8px 32px rgba(212, 175, 55, 0.3);
    }
    .disclaimer {
        background: rgba(255, 107, 107, 0.15);
        border-left: 4px solid #ff6b6b;
        padding: 12px; border-radius: 0 8px 8px 0;
        margin: 15px 0; font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 💾 QUẢN LÝ STATISTICS
# =============================================================================
def init_statistics():
    """Khởi tạo statistics trong session state"""
    if 'statistics' not in st.session_state:
        st.session_state.statistics = {
            'predictions': [],
            'total_predictions': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0
        }

def save_prediction(date, prediction_type, numbers, result_numbers, is_win):
    """Lưu kết quả dự đoán"""
    pred = {
        'date': date,
        'type': prediction_type,
        'numbers': numbers,
        'result_numbers': result_numbers,
        'is_win': is_win,
        'timestamp': datetime.now().isoformat()
    }
    st.session_state.statistics['predictions'].append(pred)
    st.session_state.statistics['total_predictions'] += 1
    if is_win:
        st.session_state.statistics['wins'] += 1
    else:
        st.session_state.statistics['losses'] += 1
    
    total = st.session_state.statistics['total_predictions']
    st.session_state.statistics['win_rate'] = (
        st.session_state.statistics['wins'] / total * 100 if total > 0 else 0
    )

def get_statistics_dataframe():
    """Trả về DataFrame thống kê"""
    preds = st.session_state.statistics['predictions']
    if not preds:
        return pd.DataFrame()
    return pd.DataFrame(preds)

# =============================================================================
# 📡 SCRAPING
# =============================================================================
@st.cache_data(ttl=300)
def get_live_xsmb():
    """Crawl kết quả XSMB"""
    url = "https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.content, 'html.parser')
        
        def get_txt(cls):
            item = soup.find("span", class_=cls)
            return item.text.strip() if item else "..."
        
        return {
            "Đặc Biệt": get_txt("special-temp"),
            "Giải Nhất": get_txt("g1-temp"),
            "Giải Nhì": [get_txt(f"g2_{i}-temp") for i in range(2)],
            "Giải Ba": [get_txt(f"g3_{i}-temp") for i in range(6)],
            "Giải Tư": [get_txt(f"g4_{i}-temp") for i in range(4)],
            "Giải Năm": [get_txt(f"g5_{i}-temp") for i in range(6)],
            "Giải Sáu": [get_txt(f"g6_{i}-temp") for i in range(3)],
            "Giải Bảy": [get_txt(f"g7_{i}-temp") for i in range(4)],
            "time": datetime.now().strftime("%H:%M:%S %d/%m")
        }
    except:
        return None

# =============================================================================
# 🎲 LOGIC DỰ ĐOÁN THẬT (PHÂN TÍCH TẦN SUẤT)
# =============================================================================
def analyze_frequency(all_loto):
    """Phân tích tần suất xuất hiện"""
    from collections import Counter
    counter = Counter(all_loto)
    
    # Số xuất hiện nhiều (nóng)
    hot_numbers = [num for num, count in counter.most_common(5)]
    
    # Số ít xuất hiện (lạnh - có thể về)
    all_possible = [f"{i:02d}" for i in range(100)]
    cold_numbers = [num for num in all_possible if num not in counter]
    
    return hot_numbers, cold_numbers[:5]

def generate_real_predictions(all_loto, data):
    """Tạo dự đoán thật dựa trên phân tích"""
    hot, cold = analyze_frequency(all_loto)
    
    # Bạch thủ: chọn từ số nóng hoặc lạnh
    bt = hot[0] if hot and np.random.random() > 0.5 else (cold[0] if cold else f"{np.random.randint(0,100):02d}")
    
    # Song thủ: 1 nóng + 1 lạnh
    st1 = hot[1] if len(hot) > 1 else f"{np.random.randint(0,100):02d}"
    st2 = cold[1] if len(cold) > 1 else f"{np.random.randint(0,100):02d}"
    
    # Xiên 2
    xien2 = f"{bt} - {st1}"
    
    # Dàn đề 10 số: kết hợp nóng + lạnh + ngẫu nhiên
    dan_de = list(set(hot[:3] + cold[:3] + [f"{np.random.randint(0,100):02d}" for _ in range(4)]))
    dan_de = sorted(dan_de)[:10]
    
    return {
        "bach_thu": bt,
        "song_thu": (st1, st2),
        "xien_2": xien2,
        "dan_de": dan_de,
        "hot_numbers": hot,
        "cold_numbers": cold
    }

# =============================================================================
# 🚀 MAIN APP
# =============================================================================
def main():
    init_statistics()
    
    # HEADER
    st.markdown('''
    <div class="header-gold">
        <h1 style="margin:0;">💎 AI-QUANTUM PRO 2026</h1>
        <p style="margin:5px 0 0;">🎯 XSMB Real-Time • Dự Đoán Thật • Thống Kê Thắng Thua</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # SIDEBAR - THỐNG KÊ
    with st.sidebar:
        st.markdown("### 📊 THỐNG KÊ DỰ ĐOÁN")
        
        stats = st.session_state.statistics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'''
            <div class="stat-card">
                <div class="stat-value win">{stats['wins']}</div>
                <div class="stat-label">Thắng</div>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
            <div class="stat-card">
                <div class="stat-value loss">{stats['losses']}</div>
                <div class="stat-label">Thua</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="stat-card" style="margin-top: 10px;">
            <div class="stat-value" style="color: {'#00ff88' if stats['win_rate'] >= 50 else '#ff4b4b'}">
                {stats['win_rate']:.1f}%
            </div>
            <div class="stat-label">Tỷ lệ thắng</div>
            <div class="stat-label">Tổng: {stats['total_predictions']} lần</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.divider()
        
        # Nút reset
        if st.button("🔄 Xóa lịch sử", use_container_width=True):
            st.session_state.statistics = {
                'predictions': [],
                'total_predictions': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0
            }
            st.rerun()
        
        # Export statistics
        if st.session_state.statistics['predictions']:
            df_stats = get_statistics_dataframe()
            csv = df_stats.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Tải lịch sử CSV",
                data=csv,
                file_name=f"thong_ke_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # TABS
    tab1, tab2, tab3 = st.tabs(["📡 Kết Quả & Dự Đoán", "🌐 Website Trực Tiếp", "📈 Lịch Sử Thống Kê"])
    
    with tab1:
        st.markdown("### 📡 KẾT QUẢ XSMB TRỰC TIẾP")
        
        data = get_live_xsmb()
        if not 
            st.error("❌ Không thể tải kết quả. Đang hiển thị dữ liệu tham khảo.")
            data = {
                "Đặc Biệt": "48076",
                "Giải Nhất": "66442",
                "Giải Nhì": ["97779", "94665"],
                "time": datetime.now().strftime("%H:%M:%S %d/%m")
            }
        
        # Hiển thị kết quả
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"🕐 Cập nhật: **{data.get('time', 'N/A')}**")
        with col2:
            if st.button("🔄 Làm mới", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        # Bảng kết quả
        st.markdown(f'''
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); 
                    border: 2px solid #D4AF37; border-radius: 15px; padding: 20px;">
            <div style="text-align: center; font-size: 36px; color: #ff4b4b; 
                        font-weight: bold; letter-spacing: 4px; margin: 10px 0;">
                🏆 {data.get('Đặc Biệt', '....')}
            </div>
            <div style="text-align: center; font-size: 18px; color: #fff; margin: 10px 0;">
                Giải Nhất: {data.get('Giải Nhất', '....')}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # DỰ ĐOÁN
        st.markdown("### 🎯 DỰ ĐOÁN AI-QUANTUM")
        
        # Extract all 2-digit numbers
        all_loto = []
        for k, v in data.items():
            if k == "time": continue
            if isinstance(v, list):
                all_loto.extend([x[-2:] for x in v if x and x != "..."])
            elif v and v != "...":
                all_loto.append(v[-2:])
        
        # Generate predictions
        predictions = generate_real_predictions(all_loto, data)
        
        # Display predictions
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">🎯 BẠCH THỦ</div>
                <div style="font-size:36px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {predictions['bach_thu']}
                </div>
                <div style="font-size:11px; color:#666;">
                    Số nóng: {', '.join(predictions['hot_numbers'][:3]) if predictions['hot_numbers'] else 'N/A'}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with c2:
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">🎯 SONG THỦ</div>
                <div style="font-size:28px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {predictions['song_thu'][0]} - {predictions['song_thu'][1]}
                </div>
                <div style="font-size:11px; color:#666;">
                    1 nóng + 1 lạnh
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with c3:
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">🎯 XIÊN 2</div>
                <div style="font-size:24px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {predictions['xien_2']}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Dàn đề
        st.markdown(f'''
        <div class="pred-box" style="margin-top: 10px;">
            <div style="color:#aaa; font-size:14px; margin-bottom:10px;">
                📋 DÀN ĐỀ 10 SỐ (Kết hợp nóng + lạnh + phân tích)
            </div>
            <div style="font-size:20px; color:#fff; letter-spacing:2px;">
                {', '.join(predictions['dan_de'])}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Nút kiểm tra kết quả
        st.markdown("---")
        st.markdown("### ✅ KIỂM TRA KẾT QUẢ DỰ ĐOÁN")
        
        col_check1, col_check2 = st.columns(2)
        with col_check1:
            check_bt = st.text_input("Nhập Bạch Thủ dự đoán", max_chars=2, placeholder="VD: 76")
        with col_check2:
            check_st = st.text_input("Nhập Song Thủ dự đoán", max_chars=7, placeholder="VD: 09-90")
        
        if st.button("🎯 Kiểm Tra Kết Quả", use_container_width=True, type="primary"):
            if check_bt or check_st:
                result_db = data.get("Đặc Biệt", "")[-2:] if data.get("Đặc Biệt") else ""
                result_loto = all_loto
                
                is_win_bt = check_bt in result_loto if check_bt else False
                is_win_st = any(s.strip() in result_loto for s in check_st.split('-')) if check_st else False
                
                if is_win_bt:
                    st.success(f"🎉 Bạch thủ {check_bt} TRÚNG! (Có trong kết quả)")
                    save_prediction(
                        datetime.now().strftime("%d/%m %H:%M"),
                        "Bạch Thủ",
                        check_bt,
                        f"Có trong {len(result_loto)} số",
                        True
                    )
                elif check_bt:
                    st.error(f"❌ Bạch thủ {check_bt} không trúng")
                    save_prediction(
                        datetime.now().strftime("%d/%m %H:%M"),
                        "Bạch Thủ",
                        check_bt,
                        f"ĐB: {result_db}",
                        False
                    )
                
                if is_win_st:
                    st.success(f"🎉 Song thủ {check_st} TRÚNG!")
                    save_prediction(
                        datetime.now().strftime("%d/%m %H:%M"),
                        "Song Thủ",
                        check_st,
                        "Trúng 1 hoặc 2 số",
                        True
                    )
                elif check_st:
                    st.error(f"❌ Song thủ {check_st} không trúng")
                    save_prediction(
                        datetime.now().strftime("%d/%m %H:%M"),
                        "Song Thủ",
                        check_st,
                        "Không trùng",
                        False
                    )
                
                st.rerun()
            else:
                st.warning("⚠️ Vui lòng nhập ít nhất 1 số để kiểm tra")
        
        st.markdown("---")
        st.markdown('''
        <div class="disclaimer">
            ⚠️ <b>LƯU Ý</b>: Ứng dụng này sử dụng phân tích thống kê và tần suất để đưa ra dự đoán. 
            Tuy nhiên, xổ số là trò chơi may rủi. Hãy chơi có trách nhiệm và trong khả năng tài chính.
        </div>
        ''', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 🌐 WEBSITE XOSODAIPHAT.COM")
        st.markdown("Xem kết quả trực tiếp từ website chính thức:")
        
        # Embed website
        st.markdown('''
        <div class="iframe-container" style="height: 800px;">
            <iframe src="https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html" 
                    style="width:100%; height:100%; border:none;"
                    sandbox="allow-same-origin allow-scripts allow-popups allow-forms">
            </iframe>
        </div>
        ''', unsafe_allow_html=True)
        
        st.info("💡 Mẹo: Kéo xuống để xem đầy đủ kết quả và thống kê từ website")
    
    with tab3:
        st.markdown("### 📈 LỊCH SỬ DỰ ĐOÁN & THỐNG KÊ")
        
        if st.session_state.statistics['predictions']:
            df = get_statistics_dataframe()
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Tổng lượt", st.session_state.statistics['total_predictions'])
            col2.metric("Thắng", st.session_state.statistics['wins'], 
                       delta=f"{st.session_state.statistics['win_rate']:.1f}%")
            col3.metric("Thua", st.session_state.statistics['losses'])
            col4.metric("Tỷ lệ thắng", f"{st.session_state.statistics['win_rate']:.1f}%")
            
            st.divider()
            
            # Filter
            filter_type = st.selectbox("Lọc theo loại", ["Tất cả", "Bạch Thủ", "Song Thủ"])
            
            if filter_type != "Tất cả":
                df = df[df['type'] == filter_type]
            
            # Display table
            st.dataframe(
                df[['date', 'type', 'numbers', 'result_numbers', 'is_win']].sort_values('date', ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": "Ngày/Giờ",
                    "type": "Loại",
                    "numbers": "Số dự đoán",
                    "result_numbers": "Kết quả",
                    "is_win": st.column_config.CheckboxColumn("Trúng?", help="Kết quả")
                }
            )
            
            # Chart
            if not df.empty:
                st.markdown("#### 📊 Biểu đồ thắng/thua theo thời gian")
                df_chart = df.copy()
                df_chart['is_win_num'] = df_chart['is_win'].map({True: 1, False: 0})
                st.line_chart(df_chart.set_index('date')['is_win_num'])
        else:
            st.info("📭 Chưa có lịch sử dự đoán. Hãy bắt đầu dự đoán và kiểm tra kết quả!")
    
    # FOOTER
    st.markdown("---")
    st.markdown('''
    <div style="text-align: center; color: #666; font-size: 12px; padding: 20px;">
        💎 AI-QUANTUM PRO 2026 • Dự Đoán XSMB Thông Minh<br>
        <b>Chơi xổ số có trách nhiệm - 18+ only</b>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()