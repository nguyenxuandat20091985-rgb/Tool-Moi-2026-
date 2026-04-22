# =============================================================================
# 💎 AI-QUANTUM PRO 2026 - XSMB REAL-TIME & STATISTICS
# =============================================================================
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import time
from collections import Counter

# =============================================================================
# 🔧 CẤU HÌNH
# =============================================================================
st.set_page_config(
    page_title="💎 AI-QUANTUM PRO 2026",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .disclaimer {
        background: rgba(255, 107, 107, 0.15);
        border-left: 4px solid #ff6b6b;
        padding: 12px; border-radius: 0 8px 8px 0;
        margin: 15px 0; font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 💾 STATISTICS MANAGEMENT
# =============================================================================
def init_statistics():
    if 'statistics' not in st.session_state:
        st.session_state.statistics = {
            'predictions': [],
            'total_predictions': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0
        }

def save_prediction(date, pred_type, numbers, result_numbers, is_win):
    pred = {
        'date': date,
        'type': pred_type,
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
    if total > 0:
        st.session_state.statistics['win_rate'] = (
            st.session_state.statistics['wins'] / total * 100
        )

# =============================================================================
# 📡 SCRAPING
# =============================================================================
@st.cache_data(ttl=300)
def get_live_xsmb():
    url = "https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
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
    except Exception as e:
        return None

# =============================================================================
# 🎲 PREDICTION LOGIC
# =============================================================================
def generate_predictions(all_loto):
    counter = Counter(all_loto)
    hot_numbers = [num for num, count in counter.most_common(5)]
    
    all_possible = [f"{i:02d}" for i in range(100)]
    cold_numbers = [num for num in all_possible if num not in counter][:5]
    
    bt = hot_numbers[0] if hot_numbers else f"{np.random.randint(0,100):02d}"
    st1 = hot_numbers[1] if len(hot_numbers) > 1 else f"{np.random.randint(0,100):02d}"
    st2 = cold_numbers[0] if cold_numbers else f"{np.random.randint(0,100):02d}"
    
    dan_de = list(set(hot_numbers[:3] + cold_numbers[:3]))
    while len(dan_de) < 10:
        dan_de.append(f"{np.random.randint(0,100):02d}")
    dan_de = sorted(dan_de)[:10]
    
    return {
        "bach_thu": bt,
        "song_thu": (st1, st2),
        "xien_2": f"{bt} - {st1}",
        "dan_de": dan_de,
        "hot_numbers": hot_numbers,
        "cold_numbers": cold_numbers
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
        <p style="margin:5px 0 0;">🎯 XSMB Real-Time • Dự Đoán • Thống Kê</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("### 📊 THỐNG KÊ")
        
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
        
        wr = stats['win_rate']
        st.markdown(f'''
        <div class="stat-card" style="margin-top: 10px;">
            <div class="stat-value" style="color: {'#00ff88' if wr >= 50 else '#ff4b4b'}">
                {wr:.1f}%
            </div>
            <div class="stat-label">Tỷ lệ thắng</div>
            <div class="stat-label">Tổng: {stats['total_predictions']}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.divider()
        
        if st.button("🔄 Xóa lịch sử", use_container_width=True):
            st.session_state.statistics = {
                'predictions': [],
                'total_predictions': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0
            }
            st.rerun()
        
        if stats['predictions']:
            df_stats = pd.DataFrame(stats['predictions'])
            csv = df_stats.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Tải CSV",
                data=csv,
                file_name=f"thong_ke_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # TABS
    tab1, tab2, tab3 = st.tabs(["📡 Kết Quả & Dự Đoán", "🌐 Website", "📈 Lịch Sử"])
    
    with tab1:
        st.markdown("### 📡 KẾT QUẢ XSMB")
        
        data = get_live_xsmb()
        
        if data is None:
            st.error("❌ Không tải được kết quả")
            data = {
                "Đặc Biệt": "48076",
                "Giải Nhất": "66442",
                "Giải Nhì": ["97779", "94665"],
                "time": datetime.now().strftime("%H:%M:%S %d/%m")
            }
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"🕐 Cập nhật: **{data.get('time', 'N/A')}**")
        with col2:
            if st.button("🔄 Làm mới", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        # Display result
        db = data.get("Đặc Biệt", "....")
        st.markdown(f'''
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); 
                    border: 2px solid #D4AF37; border-radius: 15px; 
                    padding: 20px; text-align: center; margin: 10px 0;">
            <div style="font-size: 18px; color: #aaa; margin-bottom: 10px;">🏆 ĐẶC BIỆT</div>
            <div style="font-size: 42px; color: #ff4b4b; font-weight: bold; 
                        letter-spacing: 6px;">{db}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 DỰ ĐOÁN")
        
        # Extract loto numbers
        all_loto = []
        for k, v in data.items():
            if k == "time":
                continue
            if isinstance(v, list):
                for x in v:
                    if x and x != "...":
                        all_loto.append(x[-2:])
            elif v and v != "...":
                all_loto.append(v[-2:])
        
        preds = generate_predictions(all_loto)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">🎯 BẠCH THỦ</div>
                <div style="font-size:36px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {preds['bach_thu']}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with c2:
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">🎯 SONG THỦ</div>
                <div style="font-size:28px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {preds['song_thu'][0]} - {preds['song_thu'][1]}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with c3:
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">🎯 XIÊN 2</div>
                <div style="font-size:24px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {preds['xien_2']}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="pred-box">
            <div style="color:#aaa; font-size:14px; margin-bottom:10px;">📋 DÀN ĐỀ 10 SỐ</div>
            <div style="font-size:18px; color:#fff;">{', '.join(preds['dan_de'])}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ✅ KIỂM TRA KẾT QUẢ")
        
        col_check1, col_check2 = st.columns(2)
        with col_check1:
            check_bt = st.text_input("Bạch Thủ", max_chars=2, placeholder="VD: 76")
        with col_check2:
            check_st = st.text_input("Song Thủ", max_chars=7, placeholder="VD: 09-90")
        
        if st.button("🎯 Kiểm Tra", use_container_width=True, type="primary"):
            if check_bt or check_st:
                result_loto = all_loto
                
                if check_bt:
                    is_win_bt = check_bt in result_loto
                    if is_win_bt:
                        st.success(f"🎉 Bạch thủ {check_bt} TRÚNG!")
                        save_prediction(
                            datetime.now().strftime("%d/%m %H:%M"),
                            "Bạch Thủ",
                            check_bt,
                            "Trúng",
                            True
                        )
                    else:
                        st.error(f"❌ Bạch thủ {check_bt} trượt")
                        save_prediction(
                            datetime.now().strftime("%d/%m %H:%M"),
                            "Bạch Thủ",
                            check_bt,
                            "Trượt",
                            False
                        )
                
                if check_st:
                    st_parts = [s.strip() for s in check_st.split('-')]
                    is_win_st = any(s in result_loto for s in st_parts)
                    if is_win_st:
                        st.success(f"🎉 Song thủ {check_st} TRÚNG!")
                        save_prediction(
                            datetime.now().strftime("%d/%m %H:%M"),
                            "Song Thủ",
                            check_st,
                            "Trúng",
                            True
                        )
                    else:
                        st.error(f"❌ Song thủ {check_st} trượt")
                        save_prediction(
                            datetime.now().strftime("%d/%m %H:%M"),
                            "Song Thủ",
                            check_st,
                            "Trượt",
                            False
                        )
                
                st.rerun()
            else:
                st.warning("⚠️ Nhập ít nhất 1 số")
        
        st.markdown("---")
        st.markdown('''
        <div class="disclaimer">
            ⚠️ <b>LƯU Ý</b>: Ứng dụng mang tính chất tham khảo giải trí. 
            Xổ số là may rủi. Chơi có trách nhiệm!
        </div>
        ''', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 🌐 WEBSITE XOSODAIPHAT")
        st.markdown('''
        <div style="border: 2px solid #D4AF37; border-radius: 15px; 
                    overflow: hidden; height: 800px;">
            <iframe src="https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html" 
                    style="width:100%; height:100%; border:none;"
                    sandbox="allow-same-origin allow-scripts">
            </iframe>
        </div>
        ''', unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 📈 LỊCH SỬ")
        
        if st.session_state.statistics['predictions']:
            df = pd.DataFrame(st.session_state.statistics['predictions'])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Tổng", st.session_state.statistics['total_predictions'])
            col2.metric("Thắng", st.session_state.statistics['wins'])
            col3.metric("Tỷ lệ", f"{st.session_state.statistics['win_rate']:.1f}%")
            
            st.divider()
            
            display_df = df[['date', 'type', 'numbers', 'result_numbers', 'is_win']].copy()
            display_df = display_df.sort_values('date', ascending=False)
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": "Ngày",
                    "type": "Loại",
                    "numbers": "Số",
                    "result_numbers": "Kết quả",
                    "is_win": st.column_config.CheckboxColumn("Trúng?")
                }
            )
        else:
            st.info("📭 Chưa có lịch sử")
    
    # FOOTER
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #666; padding: 20px;">'
                '💎 AI-QUANTUM PRO 2026<br>Chơi xổ số có trách nhiệm - 18+</div>',
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()