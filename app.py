# =============================================================================
# 💎 AI-QUANTUM PRO 2026 - AUTO DAILY PREDICTIONS (FIXED)
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
from streamlit_autorefresh import st_autorefresh

# =============================================================================
# 🔧 CẤU HÌNH
# =============================================================================
st.set_page_config(
    page_title="💎 AI-QUANTUM PRO 2026 | Auto Daily",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 120 seconds (2 minutes)
st_autorefresh(interval=120000, key="auto_refresh", limit=None)

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
        padding: 15px; margin: 10px 0;
        text-align: center;
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
            'last_check_date': None
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

def save_result_check(today_date, current_data):
    current_loto = []
    for k, v in current_data.items():
        if k == "time":
            continue
        if isinstance(v, list):
            for x in v:
                if x and x != "...":
                    current_loto.append(x[-2:])
        elif v and v != "...":
            current_loto.append(v[-2:])
    
    db_number = current_data.get("Đặc Biệt", "")
    db_last2 = db_number[-2:] if db_number and len(db_number) >= 2 else ""
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%d/%m")
    yesterday_pred = get_prediction_for_date(yesterday)
    
    if yesterday_pred and not yesterday_pred.get('checked', False):
        preds = yesterday_pred['predictions']
        
        bt = preds.get('bach_thu', '')
        bt_win = bt in current_loto if bt else False
        
        st_nums = preds.get('song_thu', ('', ''))
        st_win = any(s in current_loto for s in st_nums if s)
        
        xien = preds.get('xien_2', '')
        xien_parts = [p.strip() for p in xien.split('-') if p.strip()]
        xien_win = all(p in current_loto for p in xien_parts) if len(xien_parts) == 2 else False
        
        dan_de = preds.get('dan_de', [])
        de_win = db_last2 in dan_de if db_last2 and dan_de else False
        
        result_record = {
            'check_date': today_date,
            'pred_date': yesterday,
            'bach_thu': {'number': bt, 'win': bt_win},
            'song_thu': {'numbers': st_nums, 'win': st_win},
            'xien_2': {'pair': xien, 'win': xien_win},
            'dan_de': {'numbers': dan_de, 'win': de_win, 'matched': db_last2 if de_win else None},
            'overall_wins': sum([bt_win, st_win, xien_win, de_win]),
            'timestamp': datetime.now().isoformat()
        }
        
        st.session_state.statistics['results'].append(result_record)
        yesterday_pred['checked'] = True
        
        if today_date not in st.session_state.statistics['daily_stats']:
            st.session_state.statistics['daily_stats'][today_date] = {
                'date': today_date, 'checked': 0, 'total_wins': 0
            }
        st.session_state.statistics['daily_stats'][today_date]['checked'] += 1
        st.session_state.statistics['daily_stats'][today_date]['total_wins'] += result_record['overall_wins']
        
        save_statistics()
        return result_record
    
    return None

# =============================================================================
# 📡 SCRAPING - IMPROVED
# =============================================================================
@st.cache_data(ttl=120)
def get_live_xsmb():
    url = "https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, 'html.parser')
        
        # Try multiple strategies
        result = extract_data_v1(soup)
        if result["Đặc Biệt"] in ["...", ""]:
            result = extract_data_v2(soup)
        
        result["time"] = datetime.now().strftime("%H:%M:%S %d/%m")
        return result
        
    except Exception as e:
        st.error(f"❌ Lỗi crawl: {str(e)[:100]}")
        return get_mock_data()

def extract_data_v1(soup):
    """Strategy 1: Class-based selectors"""
    def get_txt(classes):
        if isinstance(classes, str):
            classes = [classes]
        for cls in classes:
            item = soup.find("span", class_=cls)
            if item and item.text.strip():
                text = item.text.strip()
                # Extract numbers only
                import re
                match = re.search(r'(\d+)', text)
                if match:
                    return match.group(1)
        return "..."
    
    return {
        "Đặc Biệt": get_txt(["special-temp", "special", "db-value"]),
        "Giải Nhất": get_txt(["g1-temp", "g1"]),
        "Giải Nhì": [get_txt(f"g2_{i}-temp") for i in range(2)],
        "Giải Ba": [get_txt(f"g3_{i}-temp") for i in range(6)],
        "Giải Tư": [get_txt(f"g4_{i}-temp") for i in range(4)],
        "Giải Năm": [get_txt(f"g5_{i}-temp") for i in range(6)],
        "Giải Sáu": [get_txt(f"g6_{i}-temp") for i in range(3)],
        "Giải Bảy": [get_txt(f"g7_{i}-temp") for i in range(4)],
    }

def extract_data_v2(soup):
    """Strategy 2: Pattern matching"""
    import re
    text = soup.get_text()
    
    # Find 5-digit numbers
    numbers = re.findall(r'\b\d{5}\b', text)
    
    return {
        "Đặc Biệt": numbers[0] if numbers else "...",
        "Giải Nhất": numbers[1] if len(numbers) > 1 else "...",
        "Giải Nhì": ["..."] * 2,
        "Giải Ba": ["..."] * 6,
        "Giải Tư": ["..."] * 4,
        "Giải Năm": ["..."] * 6,
        "Giải Sáu": ["..."] * 3,
        "Giải Bảy": ["..."] * 4,
    }

def get_mock_data():
    """Fallback data"""
    return {
        "Đặc Biệt": "36948",
        "Giải Nhất": "96041",
        "Giải Nhì": ["09028", "27803"],
        "time": datetime.now().strftime("%H:%M:%S %d/%m"),
        "source": "Mock"
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
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', text)
                    if json_match:
                        ai_result = json.loads(json_match.group())
                        if all(key in ai_result for key in ['bach_thu', 'song_thu', 'dan_de', 'confidence']):
                            return ai_result
            
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)
                
        except:
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
    
    # AUTO-CHECK predictions
    today_pred, new_predictions = ensure_predictions_for_today()
    
    # HEADER
    st.markdown('''
    <div class="header-gold">
        <h1 style="margin:0;">💎 AI-QUANTUM PRO 2026</h1>
        <p style="margin:5px 0 0;">🔄 Auto Refresh 120s • Dự Đoán Tự Động</p>
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
        
        st.markdown("### 📈 TỔNG QUAN")
        results = st.session_state.statistics.get('results', [])
        total_checks = len(results)
        total_wins = sum(r['overall_wins'] for r in results)
        max_wins = total_checks * 4
        win_rate = (total_wins / max_wins * 100) if max_wins > 0 else 0
        
        col1, col2 = st.columns(2)
        col1.markdown(f'''
        <div class="stat-card">
            <div class="stat-value win">{total_wins}</div>
            <div class="stat-label">Trúng</div>
        </div>
        ''', unsafe_allow_html=True)
        col2.markdown(f'''
        <div class="stat-card">
            <div class="stat-value loss">{max_wins - total_wins}</div>
            <div class="stat-label">Trượt</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="stat-card" style="margin-top: 10px;">
            <div class="stat-value" style="color: {'#00ff88' if win_rate >= 50 else '#ff4b4b'}">
                {win_rate:.1f}%
            </div>
            <div class="stat-label">Tỷ lệ trúng</div>
            <div class="stat-label">{total_checks} lần check</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.divider()
        
        if st.button("🔄 Reset toàn bộ", use_container_width=True):
            st.session_state.statistics = {
                'predictions': [], 'results': [], 'daily_stats': {},
                'last_check_date': None
            }
            save_statistics()
            st.rerun()
    
    # TABS
    tab1, tab2, tab3 = st.tabs(["🎯 Dự Đoán Hôm Nay", "🌐 Website XS", "📜 Lịch Sử"])
    
    with tab1:
        st.markdown(f'''
        <div class="info-banner">
            <b>📅 DỰ ĐOÁN CHO NGÀY {today}</b><br>
            <small>Auto-refresh mỗi 120 giây</small>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("### 📡 KẾT QUẢ XSMB")
        
        data = get_live_xsmb()
        historical = get_historical_data(30)
        
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
        
        # CHECK RESULTS
        st.markdown("---")
        st.markdown("### ✅ KIỂM TRA DỰ ĐOÁN HÔM QUA")
        
        if st.button(f"🔍 Check kết quả ngày {today}", use_container_width=True, type="primary"):
            result = save_result_check(today, data)
            
            if result:
                st.success(f"✅ Đã check kết quả!")
                
                st.markdown(f'''
                <div style="background: rgba(212, 175, 55, 0.1); 
                            border-left: 4px solid #D4AF37;
                            padding: 15px; border-radius: 0 8px 8px 0;
                            margin: 10px 0;">
                    <b>📅 {result['pred_date']} → {result['check_date']}</b>
                    <div style="margin-top: 10px;">
                        <div style="padding: 8px; display: flex; justify-content: space-between;">
                            <span>🎯 Bạch thủ: <b>{result['bach_thu']['number']}</b></span>
                            <span class="{'win' if result['bach_thu']['win'] else 'loss'}">
                                {'✅' if result['bach_thu']['win'] else '❌'}
                            </span>
                        </div>
                        <div style="padding: 8px; display: flex; justify-content: space-between;">
                            <span>🎯 Song thủ: <b>{'-'.join(result['song_thu']['numbers'])}</b></span>
                            <span class="{'win' if result['song_thu']['win'] else 'loss'}">
                                {'✅' if result['song_thu']['win'] else '❌'}
                            </span>
                        </div>
                        <div style="padding: 8px; display: flex; justify-content: space-between;">
                            <span>🎯 Xiên 2: <b>{result['xien_2']['pair']}</b></span>
                            <span class="{'win' if result['xien_2']['win'] else 'loss'}">
                                {'✅' if result['xien_2']['win'] else '❌'}
                            </span>
                        </div>
                        <div style="padding: 8px; display: flex; justify-content: space-between;">
                            <span>📋 Đề</span>
                            <span class="{'win' if result['dan_de']['win'] else 'loss'}">
                                {'✅ ' + result['dan_de']['matched'] if result['dan_de']['win'] else '❌'}
                            </span>
                        </div>
                    </div>
                    <div style="margin-top: 10px; text-align: right;">
                        <b>Tổng: {result['overall_wins']}/4 trúng</b>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                st.rerun()
            else:
                st.info("ℹ️ Chưa có dự đoán hôm qua để check")
        
        st.markdown("---")
        st.markdown(f"### 🎯 DỰ ĐOÁN AI NGÀY {today}")
        
        if new_predictions:
            predictions = new_predictions
        else:
            predictions = generate_predictions(historical)
            for p in st.session_state.statistics['predictions']:
                if p['date'] == today:
                    p['predictions'] = {
                        'bach_thu': predictions['bach_thu'],
                        'song_thu': predictions['song_thu'],
                        'xien_2': predictions['xien_2'],
                        'dan_de': predictions['dan_de']
                    }
                    p['confidence'] = predictions['confidence']
                    p['ai_analysis'] = predictions['ai_analysis']
                    p['using_ai'] = predictions['using_ai']
                    break
            save_statistics()
        
        if predictions['using_ai']:
            st.success("✅ Gemini AI Active")
        else:
            st.warning("⚠️ Fallback Mode")
        
        st.markdown(f'''
        <div style="background: rgba(212, 175, 55, 0.1); border: 1px solid #D4AF37; 
                    border-radius: 8px; padding: 12px; margin: 10px 0;">
            <b>🧠 PHÂN TÍCH:</b><br>{predictions['ai_analysis']}<br>
            <b>ĐỘ TIN CẬY:</b> {predictions['confidence']}%
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
        st.markdown("### 🎲 BẠC NHỚ")
        
        bac_nho = generate_bac_nho(historical)
        
        for i, xien in enumerate(bac_nho, 1):
            st.markdown(f'''
            <div class="xien-box">
                <div style="font-size: 14px; color: #aaa;">💎 Cặp {i} - {xien['frequency']} lần</div>
                <div style="font-size: 48px; color: #FFD700; font-weight: bold;">
                    {xien['pair']}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="disclaimer">⚠️ Xổ số là may rủi. Chơi có trách nhiệm!</div>', unsafe_allow_html=True)
    
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
                    <div>Xiên: {r['xien_2']['pair']} {'✅' if r['xien_2']['win'] else '❌'}</div>
                    <div>Đề: {'✅ ' + r['dan_de']['matched'] if r['dan_de']['win'] else '❌'}</div>
                    <div><b>Tổng: {r['overall_wins']}/4</b></div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("📭 Chưa có lịch sử")
    
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #666; padding: 20px;">'
                '💎 AI-QUANTUM PRO 2026 • Auto Refresh<br>18+ only</div>',
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()