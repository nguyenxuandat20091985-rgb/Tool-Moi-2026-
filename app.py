# =============================================================================
# 💎 AI-QUANTUM PRO 2026 - GEMINI AI POWERED
# Phiên bản: 5.0 FINAL - Full Features
# =============================================================================
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import Counter
import json
import os
import time

# =============================================================================
# 🔧 CẤU HÌNH
# =============================================================================
st.set_page_config(
    page_title="💎 AI-QUANTUM PRO 2026 | Gemini AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gemini API Key
GEMINI_API_KEY = "AIzaSyARQk3lpoHnK51LQ62OR4vciH0XMFFIZjg"

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
        transition: transform 0.2s;
    }
    .pred-box:hover { transform: translateY(-3px); }
    .disclaimer {
        background: rgba(255, 107, 107, 0.15);
        border-left: 4px solid #ff6b6b;
        padding: 12px; border-radius: 0 8px 8px 0;
        margin: 15px 0; font-size: 13px;
    }
    .analysis-box {
        background: rgba(212, 175, 55, 0.1);
        border: 1px solid #D4AF37;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        font-size: 13px;
    }
    .gan-number {
        background: rgba(255, 75, 75, 0.2);
        color: #ff4b4b;
        padding: 3px 8px;
        border-radius: 5px;
        margin: 2px;
        display: inline-block;
    }
    .hot-number {
        background: rgba(0, 255, 136, 0.2);
        color: #00ff88;
        padding: 3px 8px;
        border-radius: 5px;
        margin: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 💾 STATISTICS MANAGEMENT
# =============================================================================
def init_statistics():
    """Khởi tạo statistics"""
    if 'statistics' not in st.session_state:
        st.session_state.statistics = {
            'predictions': [],
            'daily_stats': {},
            'total_predictions': 0,
            'wins': 0,
            'losses': 0
        }
    
    # Load từ file nếu có
    if os.path.exists('statistics.json'):
        try:
            with open('statistics.json', 'r', encoding='utf-8') as f:
                st.session_state.statistics = json.load(f)
        except:
            pass

def save_statistics():
    """Lưu statistics ra file"""
    try:
        with open('statistics.json', 'w', encoding='utf-8') as f:
            json.dump(st.session_state.statistics, f, ensure_ascii=False, indent=2)
    except:
        pass

def save_prediction(date_str, pred_type, numbers, result_numbers, is_win, confidence=0):
    """Lưu kết quả dự đoán"""
    today = date_str
    
    if today not in st.session_state.statistics['daily_stats']:
        st.session_state.statistics['daily_stats'][today] = {
            'date': today,
            'predictions': 0,
            'wins': 0,
            'losses': 0
        }
    
    pred = {
        'date': today,
        'time': datetime.now().strftime("%H:%M"),
        'type': pred_type,
        'numbers': numbers,
        'result_numbers': result_numbers,
        'is_win': is_win,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    }
    
    st.session_state.statistics['predictions'].append(pred)
    st.session_state.statistics['total_predictions'] += 1
    st.session_state.statistics['daily_stats'][today]['predictions'] += 1
    
    if is_win:
        st.session_state.statistics['wins'] += 1
        st.session_state.statistics['daily_stats'][today]['wins'] += 1
    else:
        st.session_state.statistics['losses'] += 1
        st.session_state.statistics['daily_stats'][today]['losses'] += 1
    
    save_statistics()

def get_daily_stats():
    """Lấy thống kê theo ngày"""
    return st.session_state.statistics.get('daily_stats', {})

# =============================================================================
# 📡 DATA SCRAPING
# =============================================================================
@st.cache_data(ttl=300)
def get_live_xsmb():
    """Crawl kết quả XSMB"""
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
    except:
        return None

def get_historical_data(days=30):
    """Giả lập dữ liệu lịch sử"""
    historical = {}
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%d/%m")
        np.random.seed(i * 1000)
        historical[date] = {
            "Đặc Biệt": f"{np.random.randint(10000, 99999):05d}",
            "Giải Nhất": f"{np.random.randint(10000, 99999):05d}",
            "Giải Nhì": [f"{np.random.randint(10000, 99999):05d}" for _ in range(2)],
            "Giải Ba": [f"{np.random.randint(10000, 99999):05d}" for _ in range(6)],
            "Giải Tư": [f"{np.random.randint(1000, 9999):04d}" for _ in range(4)],
            "Giải Năm": [f"{np.random.randint(1000, 9999):04d}" for _ in range(6)],
            "Giải Sáu": [f"{np.random.randint(100, 999):03d}" for _ in range(3)],
            "Giải Bảy": [f"{np.random.randint(0, 99):02d}" for _ in range(4)],
        }
    return historical

# =============================================================================
# 🧠 ADVANCED ANALYSIS
# =============================================================================
class AdvancedAnalyzer:
    def __init__(self, historical_data):
        self.historical = historical_data
        self.all_loto = self._extract_all_loto()
    
    def _extract_all_loto(self):
        """Tách tất cả số 2 chữ số"""
        loto = []
        for date, data in self.historical.items():
            for key, val in data.items():
                if isinstance(val, list):
                    for v in val:
                        if v and len(v) >= 2:
                            loto.append(v[-2:])
                elif val and len(val) >= 2:
                    loto.append(val[-2:])
        return loto
    
    def analyze_frequency(self):
        """Phân tích tần suất"""
        counter = Counter(self.all_loto)
        total = len(self.all_loto)
        
        frequency = {}
        for num, count in counter.items():
            frequency[num] = {
                'count': count,
                'percentage': (count / total * 100) if total > 0 else 0
            }
        
        return frequency
    
    def analyze_gap(self):
        """Phân tích số gan"""
        gap = {}
        all_nums = [f"{i:02d}" for i in range(100)]
        
        for num in all_nums:
            days_since = 0
            for date, data in self.historical.items():
                found = False
                for key, val in data.items():
                    if isinstance(val, list):
                        if any(v.endswith(num) for v in val if v):
                            found = True
                            break
                    elif val and val.endswith(num):
                        found = True
                        break
                
                if found:
                    break
                days_since += 1
            
            gap[num] = days_since
        
        return gap
    
    def get_hot_numbers(self, top_n=10):
        """Lấy số nóng"""
        freq = self.analyze_frequency()
        sorted_freq = sorted(freq.items(), key=lambda x: x[1]['count'], reverse=True)
        return [num for num, _ in sorted_freq[:top_n]]
    
    def get_cold_numbers(self, top_n=10):
        """Lấy số lạnh"""
        freq = self.analyze_frequency()
        sorted_freq = sorted(freq.items(), key=lambda x: x[1]['count'])
        return [num for num, _ in sorted_freq[:top_n]]
    
    def get_gan_numbers(self, top_n=10):
        """Lấy số gan"""
        gap = self.analyze_gap()
        sorted_gap = sorted(gap.items(), key=lambda x: x[1], reverse=True)
        return [(num, days) for num, days in sorted_gap[:top_n]]

# =============================================================================
# 🤖 GEMINI AI
# =============================================================================
def query_gemini_for_prediction(hot_nums, cold_nums, gan_nums):
    """Sử dụng Gemini AI để dự đoán"""
    
    prompt = f"""
Bạn là chuyên gia xổ số. Hãy phân tích và dự đoán XSMB.

DỮ LIỆU:
- Số NÓNG: {', '.join(hot_nums[:5])}
- Số LẠNH: {', '.join(cold_nums[:5])}
- Số GAN: {', '.join([f"{num}({days} ngày)" for num, days in gan_nums[:5]])}

TRẢ LỜI THEO JSON:
{{
    "analysis": "Phân tích ngắn",
    "bach_thu": "XX",
    "song_thu": ["XX", "YY"],
    "dan_de": ["01", "23", ...],
    "confidence": 85
}}
"""
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 500
            }
        }
        
        import re
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']
            
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        
        return None
        
    except:
        return None

# =============================================================================
# 🎲 PREDICTION ENGINE
# =============================================================================
def generate_predictions(all_loto, historical_data):
    """Tạo dự đoán"""
    
    analyzer = AdvancedAnalyzer(historical_data)
    
    hot_nums = analyzer.get_hot_numbers(10)
    cold_nums = analyzer.get_cold_numbers(10)
    gan_nums = analyzer.get_gan_numbers(10)
    
    # Query Gemini AI
    gemini_result = query_gemini_for_prediction(hot_nums, cold_nums, gan_nums)
    
    if gemini_result is None:
        # Fallback về thuật toán thường
        bt = hot_nums[0] if hot_nums else f"{np.random.randint(0,100):02d}"
        st1 = hot_nums[1] if len(hot_nums) > 1 else gan_nums[0][0] if gan_nums else f"{np.random.randint(0,100):02d}"
        st2 = cold_nums[0] if cold_nums else f"{np.random.randint(0,100):02d}"
        
        dan_de = list(set(hot_nums[:4] + cold_nums[:3] + [num for num, _ in gan_nums[:3]]))
        while len(dan_de) < 10:
            dan_de.append(f"{np.random.randint(0,100):02d}")
        dan_de = sorted(dan_de)[:10]
        
        confidence = 65
        analysis = "Phân tích dựa trên tần suất và số gan"
        using_ai = False
    else:
        bt = gemini_result.get('bach_thu', hot_nums[0] if hot_nums else "00")
        st1, st2 = gemini_result.get('song_thu', ["00", "00"])
        dan_de = gemini_result.get('dan_de', [f"{i:02d}" for i in range(10)])
        confidence = gemini_result.get('confidence', 65)
        analysis = gemini_result.get('analysis', "AI phân tích")
        using_ai = True
    
    return {
        "bach_thu": bt,
        "song_thu": (st1, st2),
        "xien_2": f"{bt} - {st1}",
        "dan_de": dan_de,
        "hot_numbers": hot_nums,
        "cold_numbers": cold_nums,
        "gan_numbers": gan_nums,
        "confidence": confidence,
        "ai_analysis": analysis,
        "using_ai": using_ai
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
        <p style="margin:5px 0 0;">🤖 Gemini AI • Phân Tích Chuyên Sâu • Thống Kê Tự Động</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # SIDEBAR - DAILY STATS
    with st.sidebar:
        st.markdown("### 📊 THỐNG KÊ THEO NGÀY")
        
        daily_stats = get_daily_stats()
        
        if daily_stats:
            sorted_dates = sorted(daily_stats.keys(), reverse=True)[:7]
            
            for date in sorted_dates:
                stats = daily_stats[date]
                total = stats['predictions']
                wins = stats['wins']
                rate = (wins / total * 100) if total > 0 else 0
                
                st.markdown(f'''
                <div class="stat-card" style="margin: 5px 0;">
                    <div style="font-size: 14px; font-weight: bold; color: #fff;">📅 {date}</div>
                    <div style="display: flex; justify-content: space-around; margin-top: 8px;">
                        <span class="win">✅ {wins}</span>
                        <span class="loss">❌ {total - wins}</span>
                        <span style="color: {'#00ff88' if rate >= 50 else '#ff4b4b'}">{rate:.0f}%</span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("📭 Chưa có dữ liệu")
        
        st.divider()
        
        # Overall stats
        st.markdown("### 📈 TỔNG QUAN")
        total = st.session_state.statistics['total_predictions']
        wins = st.session_state.statistics['wins']
        losses = st.session_state.statistics['losses']
        win_rate = (wins / total * 100) if total > 0 else 0
        
        col1, col2 = st.columns(2)
        col1.markdown(f'''
        <div class="stat-card">
            <div class="stat-value win">{wins}</div>
            <div class="stat-label">Thắng</div>
        </div>
        ''', unsafe_allow_html=True)
        col2.markdown(f'''
        <div class="stat-card">
            <div class="stat-value loss">{losses}</div>
            <div class="stat-label">Thua</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="stat-card" style="margin-top: 10px;">
            <div class="stat-value" style="color: {'#00ff88' if win_rate >= 50 else '#ff4b4b'}">
                {win_rate:.1f}%
            </div>
            <div class="stat-label">Tỷ lệ thắng</div>
            <div class="stat-label">Tổng: {total}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.divider()
        
        if st.button("🔄 Xóa lịch sử", use_container_width=True):
            st.session_state.statistics = {
                'predictions': [],
                'daily_stats': {},
                'total_predictions': 0,
                'wins': 0,
                'losses': 0
            }
            save_statistics()
            st.rerun()
        
        if st.session_state.statistics['predictions']:
            df_stats = pd.DataFrame(st.session_state.statistics['predictions'])
            csv = df_stats.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Tải CSV",
                data=csv,
                file_name=f"xsmb_stats_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Dự Đoán", "📡 Kết Quả", "📊 Phân Tích", "📜 Lịch Sử"])
    
    with tab1:
        st.markdown("### 🎯 DỰ ĐOÁN VỚI AI")
        
        data = get_live_xsmb()
        historical = get_historical_data(30)
        
        if data is None:
            st.error("❌ Không tải được kết quả")
            data = {
                "Đặc Biệt": "48076",
                "Giải Nhất": "66442",
                "time": datetime.now().strftime("%H:%M:%S %d/%m")
            }
        
        # Generate predictions
        with st.spinner("🤖 AI đang phân tích..."):
            predictions = generate_predictions([], historical)
        
        # Display AI status
        if predictions['using_ai']:
            st.success("✅ Sử dụng Gemini AI")
        else:
            st.warning("⚠️ AI không khả dụng, dùng thuật toán thường")
        
        st.markdown(f'''
        <div class="analysis-box">
            <b>🧠 PHÂN TÍCH:</b><br>
            {predictions['ai_analysis']}<br>
            <b>ĐỘ TIN CẬY:</b> <span style="color: {'#00ff88' if predictions['confidence'] >= 70 else '#ffa500'}">
                {predictions['confidence']}%
            </span>
        </div>
        ''', unsafe_allow_html=True)
        
        # Predictions
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">🎯 BẠCH THỦ</div>
                <div style="font-size:42px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {predictions['bach_thu']}
                </div>
                <div style="font-size:11px; color:#888;">Confidence: {predictions['confidence']}%</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with c2:
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">🎯 SONG THỦ</div>
                <div style="font-size:30px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {predictions['song_thu'][0]} - {predictions['song_thu'][1]}
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
        
        st.markdown(f'''
        <div class="pred-box">
            <div style="color:#aaa; font-size:14px; margin-bottom:10px;">📋 DÀN ĐỀ 10 SỐ</div>
            <div style="font-size:20px; color:#fff;">{', '.join(predictions['dan_de'])}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Check result
        st.markdown("---")
        st.markdown("### ✅ KIỂM TRA KẾT QUẢ")
        
        col_check1, col_check2 = st.columns(2)
        with col_check1:
            check_bt = st.text_input("Bạch Thủ", max_chars=2, placeholder="VD: 76")
        with col_check2:
            check_st = st.text_input("Song Thủ", max_chars=7, placeholder="VD: 09-90")
        
        if st.button("🎯 Kiểm Tra", use_container_width=True, type="primary"):
            if check_bt or check_st:
                # Extract current loto
                current_loto = []
                for k, v in data.items():
                    if k == "time":
                        continue
                    if isinstance(v, list):
                        for x in v:
                            if x and x != "...":
                                current_loto.append(x[-2:])
                    elif v and v != "...":
                        current_loto.append(v[-2:])
                
                today = datetime.now().strftime("%d/%m")
                
                if check_bt:
                    is_win = check_bt in current_loto
                    save_prediction(today, "Bạch Thủ", check_bt, 
                                   "Trúng" if is_win else "Trượt", is_win,
                                   predictions['confidence'])
                    
                    if is_win:
                        st.success(f"🎉 Bạch thủ {check_bt} TRÚNG!")
                        st.balloons()
                    else:
                        st.error(f"❌ Bạch thủ {check_bt} trượt")
                
                if check_st:
                    parts = [s.strip() for s in check_st.split('-')]
                    is_win = any(p in current_loto for p in parts)
                    save_prediction(today, "Song Thủ", check_st,
                                   "Trúng" if is_win else "Trượt", is_win,
                                   predictions['confidence'])
                    
                    if is_win:
                        st.success(f"🎉 Song thủ {check_st} TRÚNG!")
                    else:
                        st.error(f"❌ Song thủ {check_st} trượt")
                
                st.rerun()
            else:
                st.warning("⚠️ Nhập ít nhất 1 số")
        
        st.markdown("---")
        st.markdown('''
        <div class="disclaimer">
            ⚠️ <b>LƯU Ý</b>: Gemini AI phân tích dựa trên thống kê, nhưng xổ số vẫn là may rủi. 
            Chơi có trách nhiệm!
        </div>
        ''', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📡 KẾT QUẢ TRỰC TIẾP")
        
        data = get_live_xsmb()
        if data:
            st.caption(f"🕐 Cập nhật: {data.get('time', 'N/A')}")
            
            db = data.get("Đặc Biệt", "....")
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); 
                        border: 2px solid #D4AF37; border-radius: 15px; 
                        padding: 30px; text-align: center; margin: 10px 0;">
                <div style="font-size: 18px; color: #aaa; margin-bottom: 15px;">🏆 ĐẶC BIỆT</div>
                <div style="font-size: 48px; color: #ff4b4b; font-weight: bold; 
                            letter-spacing: 8px;">{db}</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.error("❌ Không tải được kết quả")
    
    with tab3:
        st.markdown("### 📊 PHÂN TÍCH CHUYÊN SÂU")
        
        historical = get_historical_data(30)
        analyzer = AdvancedAnalyzer(historical)
        
        sub1, sub2, sub3 = st.tabs(["🔥 Số Nóng/Lạnh", "📉 Số Gan", "📈 Pattern"])
        
        with sub1:
            hot = analyzer.get_hot_numbers(15)
            cold = analyzer.get_cold_numbers(15)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔥 SỐ NÓNG**")
                hot_html = " ".join([f'<span class="hot-number">{n}</span>' for n in hot])
                st.markdown(f'<div style="margin: 10px 0;">{hot_html}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("**❄️ SỐ LẠNH**")
                cold_html = " ".join([f'<span class="gan-number">{n}</span>' for n in cold])
                st.markdown(f'<div style="margin: 10px 0;">{cold_html}</div>', unsafe_allow_html=True)
        
        with sub2:
            gan = analyzer.get_gan_numbers(20)
            st.markdown("**📉 SỐ GAN (Số ngày chưa về)**")
            
            gan_data = {
                "Số": [g[0] for g in gan],
                "Ngày": [g[1] for g in gan]
            }
            st.dataframe(pd.DataFrame(gan_data), use_container_width=True, hide_index=True)
        
        with sub3:
            st.info("📊 Pattern analysis - Coming soon")
    
    with tab4:
        st.markdown("### 📜 LỊCH SỬ DỰ ĐOÁN")
        
        if st.session_state.statistics['predictions']:
            df = pd.DataFrame(st.session_state.statistics['predictions'])
            
            dates = sorted(df['date'].unique(), reverse=True)
            selected_date = st.selectbox("Chọn ngày", dates)
            
            df_filtered = df[df['date'] == selected_date]
            
            st.dataframe(
                df_filtered[['time', 'type', 'numbers', 'result_numbers', 'is_win', 'confidence']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "time": "Giờ",
                    "type": "Loại",
                    "numbers": "Số",
                    "result_numbers": "Kết quả",
                    "is_win": st.column_config.CheckboxColumn("Trúng?"),
                    "confidence": "Độ tin cậy"
                }
            )
        else:
            st.info("📭 Chưa có lịch sử")
    
    # FOOTER
    st.markdown("---")
    st.markdown('''
    <div style="text-align: center; color: #666; padding: 20px;">
        💎 AI-QUANTUM PRO 2026 • Gemini AI Powered<br>
        Chơi xổ số có trách nhiệm - 18+ only
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()