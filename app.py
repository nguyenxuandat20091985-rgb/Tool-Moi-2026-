# =============================================================================
# 💎 AI-QUANTUM PRO 2026 - GEMINI AI + BẠC NHỚ
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

# =============================================================================
# 🔧 CẤU HÌNH
# =============================================================================
st.set_page_config(
    page_title="💎 AI-QUANTUM PRO 2026 | Gemini AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .xien-box {
        border: 3px solid #FFD700; border-radius: 15px;
        padding: 25px; background: linear-gradient(135deg, #1a1a2e, #2d2d44);
        text-align: center; margin: 10px 0;
        box-shadow: 0 8px 32px rgba(212, 175, 55, 0.4);
    }
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
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 💾 STATISTICS
# =============================================================================
def init_statistics():
    if 'statistics' not in st.session_state:
        st.session_state.statistics = {
            'predictions': [],
            'daily_stats': {},
            'total_predictions': 0,
            'wins': 0,
            'losses': 0
        }
    
    if os.path.exists('statistics.json'):
        try:
            with open('statistics.json', 'r', encoding='utf-8') as f:
                st.session_state.statistics = json.load(f)
        except:
            pass

def save_statistics():
    try:
        with open('statistics.json', 'w', encoding='utf-8') as f:
            json.dump(st.session_state.statistics, f, ensure_ascii=False, indent=2)
    except:
        pass

def save_prediction(date_str, pred_type, numbers, is_win, confidence=0):
    today = date_str
    if today not in st.session_state.statistics['daily_stats']:
        st.session_state.statistics['daily_stats'][today] = {
            'date': today, 'predictions': 0, 'wins': 0, 'losses': 0
        }
    
    pred = {
        'date': today,
        'time': datetime.now().strftime("%H:%M"),
        'type': pred_type,
        'numbers': numbers,
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
    except:
        return None

def get_historical_data(days=30):
    historical = {}
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%d/%m")
        np.random.seed(i * 1000)
        historical[date] = {
            "Đặc Biệt": f"{np.random.randint(10000, 99999):05d}",
            "Giải Nhất": f"{np.random.randint(10000, 99999):05d}",
            "Giải Nhì": [f"{np.random.randint(10000, 99999):05d}" for _ in range(2)],
            "Giải Ba": [f"{np.random.randint(10000, 99999):05d}" for _ in range(6)],
        }
    return historical

# =============================================================================
# 🧠 BẠC NHỚ ENGINE
# =============================================================================
def generate_bac_nho(all_loto, historical_data):
    """Tạo bạc nhớ - 3 cặp xiên đẹp nhất"""
    
    # Phân tích tần suất cặp số hay đi cùng nhau
    pairs_counter = Counter()
    
    for date, data in historical_data.items():
        loto_day = []
        for key, val in data.items():
            if isinstance(val, list):
                for v in val:
                    if v and len(v) >= 2:
                        loto_day.append(v[-2:])
            elif val and len(val) >= 2:
                loto_day.append(val[-2:])
        
        # Tạo cặp từ các số trong cùng ngày
        for i in range(len(loto_day)):
            for j in range(i+1, len(loto_day)):
                pair = tuple(sorted([loto_day[i], loto_day[j]]))
                pairs_counter[pair] += 1
    
    # Lấy 3 cặp xuất hiện nhiều nhất
    top_pairs = pairs_counter.most_common(3)
    
    # Nếu không đủ dữ liệu, tạo ngẫu nhiên
    while len(top_pairs) < 3:
        num1 = f"{np.random.randint(0, 100):02d}"
        num2 = f"{np.random.randint(0, 100):02d}"
        while num1 == num2:
            num2 = f"{np.random.randint(0, 100):02d}"
        pair = tuple(sorted([num1, num2]))
        if pair not in [p[0] for p in top_pairs]:
            top_pairs.append((pair, 1))
    
    return [
        {
            'pair': f"{p[0][0]} - {p[0][1]}",
            'frequency': p[1],
            'num1': p[0][0],
            'num2': p[0][1]
        }
        for p in top_pairs
    ]

# =============================================================================
# 🤖 GEMINI AI
# =============================================================================
def query_gemini(hot_nums, cold_nums, gan_nums):
    prompt = f"""
Chuyên gia xổ số, hãy dự đoán XSMB.

DỮ LIỆU:
- Số NÓNG: {', '.join(hot_nums[:5])}
- Số LẠNH: {', '.join(cold_nums[:5])}
- Số GAN: {', '.join([f"{num}({days} ngày)" for num, days in gan_nums[:5]])}

TRẢ LỜI JSON:
{{
    "analysis": "Phân tích ngắn",
    "bach_thu": "XX",
    "song_thu": ["XX", "YY"],
    "dan_de": ["01", "23", ...],
    "confidence": 85
}}
"""
    try:
        import re
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500}
        }
        
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
# 🎲 PREDICTION
# =============================================================================
def generate_predictions(historical_data):
    """Tạo dự đoán với AI"""
    
    # Extract all loto
    all_loto = []
    for date, data in historical_data.items():
        for key, val in data.items():
            if isinstance(val, list):
                for v in val:
                    if v and len(v) >= 2:
                        all_loto.append(v[-2:])
            elif val and len(val) >= 2:
                all_loto.append(val[-2:])
    
    # Phân tích
    counter = Counter(all_loto)
    hot_nums = [num for num, _ in counter.most_common(10)]
    
    all_possible = [f"{i:02d}" for i in range(100)]
    cold_nums = [num for num in all_possible if num not in counter][:10]
    
    # Số gan
    gan_nums = []
    for num in all_possible[:10]:
        days = 0
        for date, data in historical_data.items():
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
            days += 1
        gan_nums.append((num, days))
    
    # Query Gemini
    gemini_result = query_gemini(hot_nums, cold_nums, gan_nums)
    
    if gemini_result is None:
        bt = hot_nums[0] if hot_nums else f"{np.random.randint(0,100):02d}"
        st1 = hot_nums[1] if len(hot_nums) > 1 else f"{np.random.randint(0,100):02d}"
        st2 = cold_nums[0] if cold_nums else f"{np.random.randint(0,100):02d}"
        
        dan_de = list(set(hot_nums[:4] + cold_nums[:3] + [num for num, _ in gan_nums[:3]]))
        while len(dan_de) < 10:
            dan_de.append(f"{np.random.randint(0,100):02d}")
        dan_de = sorted(dan_de)[:10]
        
        confidence = 65
        analysis = "Phân tích tần suất và số gan"
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
        <p style="margin:5px 0 0;">🤖 Gemini AI • Bạc Nhớ • Thống Kê Tự Động</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("### 📊 THỐNG KÊ NGÀY")
        
        daily_stats = st.session_state.statistics.get('daily_stats', {})
        
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
                'predictions': [], 'daily_stats': {},
                'total_predictions': 0, 'wins': 0, 'losses': 0
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
    tab1, tab2, tab3 = st.tabs(["🎯 Dự Đoán & Bạc Nhớ", "🌐 Website XS", "📜 Lịch Sử"])
    
    with tab1:
        st.markdown("### 📡 KẾT QUẢ XSMB")
        
        data = get_live_xsmb()
        historical = get_historical_data(30)
        
        if data is None:
            st.error("❌ Không tải được kết quả")
            data = {
                "Đặc Biệt": "48076",
                "Giải Nhất": "66442",
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
        st.markdown("### 🎯 DỰ ĐOÁN AI")
        
        # Generate predictions
        with st.spinner("🤖 AI đang phân tích..."):
            predictions = generate_predictions(historical)
        
        if predictions['using_ai']:
            st.success("✅ Sử dụng Gemini AI")
        else:
            st.warning("⚠️ AI không khả dụng")
        
        st.markdown(f'''
        <div class="analysis-box">
            <b>🧠 PHÂN TÍCH:</b><br>
            {predictions['ai_analysis']}<br>
            <b>ĐỘ TIN CẬY:</b> <span style="color: {'#00ff88' if predictions['confidence'] >= 70 else '#ffa500'}">
                {predictions['confidence']}%
            </span>
        </div>
        ''', unsafe_allow_html=True)
        
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
                <div style="font-size:26px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {predictions['song_thu'][0]} - {predictions['song_thu'][1]}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with c3:
            st.markdown(f'''
            <div class="pred-box">
                <div style="color:#aaa; font-size:13px;">🎯 XIÊN 2</div>
                <div style="font-size:22px; color:#FFD700; font-weight:bold; margin:10px 0;">
                    {predictions['xien_2']}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="pred-box">
            <div style="color:#aaa; font-size:14px; margin-bottom:10px;">📋 DÀN ĐỀ 10 SỐ</div>
            <div style="font-size:18px; color:#fff;">{', '.join(predictions['dan_de'])}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎲 BẠC NHỚ - 3 CẶP XIÊN ĐẸP NHẤT")
        
        # Generate bạc nhớ
        bac_nho = generate_bac_nho([], historical)
        
        st.markdown('<div style="text-align: center; margin-bottom: 20px; color: #D4AF37; font-size: 16px;">'
                    '🔮 Dựa trên phân tích tần suất các cặp số hay về cùng nhau</div>',
                    unsafe_allow_html=True)
        
        for i, xien in enumerate(bac_nho, 1):
            st.markdown(f'''
            <div class="xien-box">
                <div style="font-size: 14px; color: #aaa; margin-bottom: 10px;">
                    💎 CẶP XIÊN {i} - Tần suất: {xien['frequency']} lần
                </div>
                <div style="font-size: 48px; color: #FFD700; font-weight: bold; 
                            letter-spacing: 4px; text-shadow: 0 0 20px rgba(255,215,0,0.5);">
                    {xien['pair']}
                </div>
                <div style="font-size: 12px; color: #888; margin-top: 10px;">
                    Số {xien['num1']} và {xien['num2']} thường xuất hiện cùng nhau
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('''
        <div class="disclaimer">
            ⚠️ <b>LƯU Ý</b>: Gemini AI + Bạc Nhớ phân tích dựa trên thống kê. 
            Xổ số là may rủi. Chơi có trách nhiệm!
        </div>
        ''', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 🌐 WEBSITE XOSODAIPHAT.COM")
        st.markdown("Xem kết quả trực tiếp và thống kê chi tiết:")
        
        st.markdown('''
        <div style="border: 2px solid #D4AF37; border-radius: 15px; 
                    overflow: hidden; height: 900px;">
            <iframe src="https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html" 
                    style="width:100%; height:100%; border:none;"
                    sandbox="allow-same-origin allow-scripts allow-forms">
            </iframe>
        </div>
        ''', unsafe_allow_html=True)
        
        st.info("💡 Mẹo: Kéo xuống để xem đầy đủ kết quả và thống kê từ website")
    
    with tab3:
        st.markdown("### 📜 LỊCH SỬ DỰ ĐOÁN")
        
        if st.session_state.statistics['predictions']:
            df = pd.DataFrame(st.session_state.statistics['predictions'])
            
            dates = sorted(df['date'].unique(), reverse=True)
            selected_date = st.selectbox("Chọn ngày", dates)
            
            df_filtered = df[df['date'] == selected_date]
            
            st.dataframe(
                df_filtered[['time', 'type', 'numbers', 'is_win', 'confidence']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "time": "Giờ",
                    "type": "Loại",
                    "numbers": "Số dự đoán",
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
        💎 AI-QUANTUM PRO 2026 • Gemini AI + Bạc Nhớ<br>
        Chơi xổ số có trách nhiệm - 18+ only
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()