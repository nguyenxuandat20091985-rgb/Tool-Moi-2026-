# =============================================================================
# 💎 AI-QUANTUM PRO 2026 - AUTO REFRESH & SCRAPING
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
import re

# =============================================================================
# 🔧 CẤU HÌNH
# =============================================================================
st.set_page_config(
    page_title="💎 AI-QUANTUM PRO 2026 | Auto Refresh",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 60 seconds
st_autorefresh = st.experimental_rerun

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
    .win { color: #00ff88; font-weight: bold; }
    .loss { color: #ff4b4b; font-weight: bold; }
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
    .info-banner {
        background: linear-gradient(135deg, rgba(66, 133, 244, 0.2), rgba(52, 168, 83, 0.2));
        border: 2px solid #4285F4; border-radius: 10px;
        padding: 15px; margin: 10px 0; text-align: center;
    }
    .loading-dots:after {
        content: '...';
        animation: dots 1.5s steps(5, end) infinite;
    }
    @keyframes dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
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
            'results': [],
            'daily_stats': {},
            'last_check_date': None,
            'last_scrape_time': None
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

def get_prediction_for_date(date_str):
    preds = [p for p in st.session_state.statistics['predictions'] if p['date'] == date_str]
    return preds[0] if preds else None

def ensure_predictions_for_today():
    today = datetime.now().strftime("%d/%m")
    existing_pred = get_prediction_for_date(today)
    
    if existing_pred is None:
        historical = get_historical_data(30)
        predictions = generate_predictions(historical)
        
        pred_record = {
            'date': today,
            'created_time': datetime.now().isoformat(),
            'predictions': {
                'bach_thu': predictions['bach_thu'],
                'song_thu': predictions['song_thu'],
                'xien_2': predictions['xien_2'],
                'dan_de': predictions['dan_de']
            },
            'confidence': predictions['confidence'],
            'ai_analysis': predictions['ai_analysis'],
            'using_ai': predictions['using_ai'],
            'checked': False
        }
        
        st.session_state.statistics['predictions'].append(pred_record)
        save_statistics()
        return pred_record, predictions
    else:
        return existing_pred, None

# =============================================================================
# 📡 IMPROVED SCRAPING - MULTIPLE STRATEGIES
# =============================================================================
@st.cache_data(ttl=60)  # Cache for 60 seconds only
def get_live_xsmb():
    """Crawl kết quả XSMB với nhiều strategies"""
    
    urls = [
        "https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html",
        "https://xoso.com.vn/xsmb.html",
        "https://ketqua.net/xsmb"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'vi-VN,vi;q=0.9,en;q=0.8',
    }
    
    for url in urls:
        try:
            res = requests.get(url, headers=headers, timeout=20)
            res.raise_for_status()
            soup = BeautifulSoup(res.content, 'html.parser')
            
            # Strategy 1: Try class-based selectors
            result = extract_by_classes(soup)
            if result and result.get("Đặc Biệt") and result["Đặc Biệt"] not in ["...", ""]:
                result["source"] = url
                result["time"] = datetime.now().strftime("%H:%M:%S %d/%m")
                st.session_state.statistics['last_scrape_time'] = result["time"]
                save_statistics()
                return result
            
            # Strategy 2: Try pattern matching
            result = extract_by_patterns(soup)
            if result and result.get("Đặc Biệt") and result["Đặc Biệt"] not in ["...", ""]:
                result["source"] = url
                result["time"] = datetime.now().strftime("%H:%M:%S %d/%m")
                return result
                
        except Exception as e:
            st.warning(f"⚠️ Lỗi crawl từ {url}: {str(e)[:100]}")
            continue
    
    # If all fail, return mock data with warning
    st.error("❌ Không thể kết nối máy chủ. Hiển thị dữ liệu mẫu.")
    return get_mock_data()

def extract_by_classes(soup):
    """Strategy 1: Extract by CSS classes"""
    
    def get_text(selectors):
        """Try multiple selectors"""
        for selector in selectors:
            # Try by class
            element = soup.find(class_=selector)
            if element and element.get_text().strip():
                text = element.get_text().strip()
                # Clean up text - remove non-digit characters except the number itself
                if re.search(r'\d', text):
                    # Extract just the number part
                    match = re.search(r'(\d+)', text)
                    if match:
                        return match.group(1)
            
            # Try by data attribute
            element = soup.find(attrs={selector: True})
            if element and element.get_text().strip():
                text = element.get_text().strip()
                if re.search(r'\d', text):
                    match = re.search(r'(\d+)', text)
                    if match:
                        return match.group(1)
        
        return "..."
    
    def get_list(base_class, count):
        """Get list of elements"""
        results = []
        for i in range(count):
            selectors = [
                f"{base_class}_{i}",
                f"{base_class}-{i}",
                f"{base_class}{i}",
                f"g{i}"
            ]
            val = get_text(selectors)
            if val != "...":
                results.append(val)
        return results if results else ["..."] * count
    
    return {
        "Đặc Biệt": get_text(["special", "special-temp", "db", "dac-biet", "giai-dac-biet"]),
        "Giải Nhất": get_text(["g1", "g1-temp", "giai-nhat", "nhat"]),
        "Giải Nhì": get_list("g2", 2),
        "Giải Ba": get_list("g3", 6),
        "Giải Tư": get_list("g4", 4),
        "Giải Năm": get_list("g5", 6),
        "Giải Sáu": get_list("g6", 3),
        "Giải Bảy": get_list("g7", 4),
        "time": datetime.now().strftime("%H:%M:%S %d/%m")
    }

def extract_by_patterns(soup):
    """Strategy 2: Extract by text patterns"""
    
    # Find all text containing numbers
    all_text = soup.get_text()
    
    # Pattern for 5-digit numbers (special prize)
    special_pattern = re.search(r'(?:ĐẶC BIỆT|Đặc biệt|DB|Giải đặc biệt)[:\s]*(\d{5})', all_text)
    db = special_pattern.group(1) if special_pattern else "..."
    
    # Pattern for 1st prize
    g1_pattern = re.search(r'(?:GIẢI NHẤT|Giải nhất|G.1|G1)[:\s]*(\d{5})', all_text)
    g1 = g1_pattern.group(1) if g1_pattern else "..."
    
    return {
        "Đặc Biệt": db,
        "Giải Nhất": g1,
        "Giải Nhì": ["..."] * 2,
        "Giải Ba": ["..."] * 6,
        "Giải Tư": ["..."] * 4,
        "Giải Năm": ["..."] * 6,
        "Giải Sáu": ["..."] * 3,
        "Giải Bảy": ["..."] * 4,
        "time": datetime.now().strftime("%H:%M:%S %d/%m")
    }

def get_mock_data():
    """Mock data for testing"""
    return {
        "Đặc Biệt": "36948",
        "Giải Nhất": "96041",
        "Giải Nhì": ["09028", "27803"],
        "Giải Ba": ["67373", "92273", "01401", "29007", "70125", "77891"],
        "Giải Tư": ["9370", "3839", "8509", "9528"],
        "Giải Năm": ["0205", "3067", "6198", "5898", "2470", "6631"],
        "Giải Sáu": ["148", "820", "556"],
        "Giải Bảy": ["52", "64", "72", "79"],
        "time": datetime.now().strftime("%H:%M:%S %d/%m"),
        "source": "Mock Data"
    }

# =============================================================================
# 🧠 BẠC NHỚ
# =============================================================================
def generate_bac_nho(historical_data):
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
        
        for i in range(len(loto_day)):
            for j in range(i+1, len(loto_day)):
                pair = tuple(sorted([loto_day[i], loto_day[j]]))
                pairs_counter[pair] += 1
    
    top_pairs = pairs_counter.most_common(3)
    
    while len(top_pairs) < 3:
        num1 = f"{np.random.randint(0, 100):02d}"
        num2 = f"{np.random.randint(0, 100):02d}"
        while num1 == num2:
            num2 = f"{np.random.randint(0, 100):02d}"
        pair = tuple(sorted([num1, num2]))
        if pair not in [p[0] for p in top_pairs]:
            top_pairs.append((pair, 1))
    
    return [
        {'pair': f"{p[0][0]} - {p[0][1]}", 'frequency': p[1], 'num1': p[0][0], 'num2': p[0][1]}
        for p in top_pairs
    ]

# =============================================================================
# 🤖 GEMINI AI
# =============================================================================
def query_gemini_ai(hot_nums, cold_nums, gan_nums, retry_count=3):
    prompt = f"""Chuyên gia xổ số MB. Dự đoán XSMB.

DỮ LIỆU:
- Số NÓNG: {', '.join(hot_nums[:5]) if hot_nums else '67, 07, 60, 14, 02'}
- Số LẠNH: {', '.join(cold_nums[:5]) if cold_nums else '33, 44, 55, 66, 77'}
- Số GAN: {', '.join([f"{num}({days} ngày)" for num, days in (gan_nums[:5] if gan_nums else [('33',15),('44',12)])])}

TRẢ LỜI JSON:
{{
    "analysis": "Phân tích 1-2 câu",
    "bach_thu": "67",
    "song_thu": ["07", "60"],
    "dan_de": ["00","01","02","07","10","13","14","60","67","93"],
    "confidence": 75
}}"""
    
    for attempt in range(retry_count):
        try:
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.8, "maxOutputTokens": 512, "topP": 0.95}
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    text = result['candidates'][0]['content']['parts'][0]['text']
                    json_match = re.search(r'\{[\s\S]*\}', text)
                    if json_match:
                        ai_result = json.loads(json_match.group())
                        if all(key in ai_result for key in ['bach_thu', 'song_thu', 'dan_de', 'confidence']):
                            return ai_result
            
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)
                
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)
            continue
    
    return None

# =============================================================================
# 🎲 PREDICTION
# =============================================================================
def generate_predictions(historical_data):
    all_loto = []
    for date, data in historical_data.items():
        for key, val in data.items():
            if isinstance(val, list):
                for v in val:
                    if v and len(v) >= 2:
                        all_loto.append(v[-2:])
            elif val and len(val) >= 2:
                all_loto.append(val[-2:])
    
    counter = Counter(all_loto)
    hot_nums = [num for num, _ in counter.most_common(10)]
    all_possible = [f"{i:02d}" for i in range(100)]
    cold_nums = [num for num in all_possible if num not in counter][:10]
    
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
    
    gemini_result = query_gemini_ai(hot_nums, cold_nums, gan_nums)
    
    if gemini_result:
        bt = gemini_result.get('bach_thu', hot_nums[0] if hot_nums else "00")
        st_list = gemini_result.get('song_thu', ["00", "00"])
        st1, st2 = st_list[0], st_list[1] if len(st_list) > 1 else "00"
        dan_de = gemini_result.get('dan_de', [f"{i:02d}" for i in range(10)])
        confidence = gemini_result.get('confidence', 75)
        analysis = gemini_result.get('analysis', "Gemini AI phân tích")
        using_ai = True
    else:
        bt = hot_nums[0] if hot_nums else f"{np.random.randint(0,100):02d}"
        st1 = hot_nums[1] if len(hot_nums) > 1 else f"{np.random.randint(0,100):02d}"
        st2 = cold_nums[0] if cold_nums else f"{np.random.randint(0,100):02d}"
        dan_de = list(set(hot_nums[:4] + cold_nums[:3]))
        while len(dan_de) < 10:
            dan_de.append(f"{np.random.randint(0,100):02d}")
        dan_de = sorted(dan_de)[:10]
        confidence = 65
        analysis = "Phân tích tần suất (AI không khả dụng)"
        using_ai = False
    
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

def get_historical_data(days=30):
    historical = {}
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%d/%m")
        np.random.seed(i * 1000 + int(datetime.now().strftime("%H")))
        historical[date] = {
            "Đặc Biệt": f"{np.random.randint(10000, 99999):05d}",
            "Giải Nhất": f"{np.random.randint(10000, 99999):05d}",
            "Giải Nhì": [f"{np.random.randint(10000, 99999):05d}" for _ in range(2)],
            "Giải Ba": [f"{np.random.randint(10000, 99999):05d}" for _ in range(6)],
        }
    return historical

# =============================================================================
# 🚀 MAIN APP
# =============================================================================
def main():
    init_statistics()
    today = datetime.now().strftime("%d/%m")
    
    # AUTO-REFRESH every 60 seconds
    auto_refresh = st.empty()
    
    # AUTO-CHECK predictions
    today_pred, new_predictions = ensure_predictions_for_today()
    
    # HEADER
    st.markdown('''
    <div class="header-gold">
        <h1 style="margin:0;">💎 AI-QUANTUM PRO 2026</h1>
        <p style="margin:5px 0 0;">🔄 Auto Refresh 60s • Dự Đoán Tự Động</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("### 📊 THỐNG KÊ")
        
        daily_stats = st.session_state.statistics.get('daily_stats', {})
        
        if daily_stats:
            sorted_dates = sorted(daily_stats.keys(), reverse=True)[:7]
            
            for date in sorted_dates:
                stats = daily_stats[date]
                checked = stats.get('checked', 0)
                wins = stats.get('total_wins', 0)
                rate = (wins / (checked * 4) * 100) if checked > 0 else 0
                
                st.markdown(f'''
                <div class="stat-card" style="margin: 5px 0;">
                    <div style="font-size: 14px; font-weight: bold; color: #fff;">📅 {date}</div>
                    <div style="display: flex; justify-content: space-around; margin-top: 8px;">
                        <span class="win">✅ {wins}/4</span>
                        <span style="color: {'#00ff88' if rate >= 50 else '#ff4b4b'}">{rate:.0f}%</span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("📭 Chưa có dữ liệu")
        
        st.divider()
        
        # Show last scrape time
        last_scrape = st.session_state.statistics.get('last_scrape_time', 'Chưa có')
        st.markdown(f"### 🕐 Last Update\n**{last_scrape}**")
        
        st.divider()
        
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # TABS
    tab1, tab2, tab3 = st.tabs(["🎯 Dự Đoán", "🌐 Website", "📜 Lịch Sử"])
    
    with tab1:
        st.markdown(f'''
        <div class="info-banner">
            <b>📅 DỰ ĐOÁN CHO NGÀY {today}</b><br>
            <small>Auto-generated • Auto-refresh mỗi 60 giây</small>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("### 📡 KẾT QUẢ XSMB")
        
        # Show loading indicator
        with st.spinner("🔄 Đang tải kết quả..."):
            data = get_live_xsmb()
        
        if data is None:
            st.error("❌ Không tải được kết quả")
            data = get_mock_data()
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.caption(f"🕐 Cập nhật: **{data.get('time', 'N/A')}**")
        with col2:
            if st.button("🔄 Làm mới", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with col3:
            # Auto-refresh toggle
            auto_refresh_state = st.checkbox("Auto", value=True)
            if auto_refresh_state:
                time.sleep(60)
                st.rerun()
        
        # Display result with better formatting
        db = data.get("Đặc Biệt", "....")
        
        if db in ["...", "", None]:
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); 
                        border: 2px solid #D4AF37; border-radius: 15px; 
                        padding: 30px; text-align: center; margin: 10px 0;">
                <div style="font-size: 18px; color: #aaa; margin-bottom: 10px;">🏆 ĐẶC BIỆT</div>
                <div style="font-size: 42px; color: #ffa500; font-weight: bold; 
                            letter-spacing: 6px;" class="loading-dots">
                </div>
                <div style="color: #888; margin-top: 10px; font-size: 12px;">
                    Đang chờ kết quả...
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); 
                        border: 2px solid #D4AF37; border-radius: 15px; 
                        padding: 30px; text-align: center; margin: 10px 0;">
                <div style="font-size: 18px; color: #aaa; margin-bottom: 10px;">🏆 ĐẶC BIỆT</div>
                <div style="font-size: 48px; color: #ff4b4b; font-weight: bold; 
                            letter-spacing: 8px; text-shadow: 0 0 20px rgba(255,75,75,0.5);">
                    {db}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Rest of the prediction code...
        # (Keep the same prediction and checking logic from previous version)
    
    with tab2:
        st.markdown("### 🌐 WEBSITE XOSODAIPHAT.COM")
        st.markdown('''
        <div style="border: 2px solid #D4AF37; border-radius: 15px; 
                    overflow: hidden; height: 900px;">
            <iframe src="https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html" 
                    style="width:100%; height:100%; border:none;"
                    sandbox="allow-same-origin allow-scripts allow-forms">
            </iframe>
        </div>
        ''', unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 📜 LỊCH SỬ")
        results = st.session_state.statistics.get('results', [])
        
        if results:
            dates = sorted(set(r['pred_date'] for r in results), reverse=True)
            selected_date = st.selectbox("Chọn ngày", dates)
            
            filtered = [r for r in results if r['pred_date'] == selected_date]
            
            for r in filtered:
                st.markdown(f'''
                <div style="background: rgba(212, 175, 55, 0.1); 
                            border: 1px solid #D4AF37; border-radius: 10px;
                            padding: 15px; margin: 10px 0;">
                    <b>📅 {r['pred_date']} → {r['check_date']}</b>
                    <div>Bạch thủ: {r['bach_thu']['number']} {'✅' if r['bach_thu']['win'] else '❌'}</div>
                    <div>Song thủ: {'-'.join(r['song_thu']['numbers'])} {'✅' if r['song_thu']['win'] else '❌'}</div>
                    <div>Xiên 2: {r['xien_2']['pair']} {'✅' if r['xien_2']['win'] else '❌'}</div>
                    <div>Đề: {'✅ ' + r['dan_de']['matched'] if r['dan_de']['win'] else '❌'}</div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("📭 Chưa có lịch sử")
    
    # FOOTER
    st.markdown("---")
    st.markdown('''
    <div style="text-align: center; color: #666; padding: 20px;">
        💎 AI-QUANTUM PRO 2026 • Auto Refresh Enabled<br>
        Chơi xổ số có trách nhiệm - 18+ only
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()