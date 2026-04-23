# =============================================================================
# 💎 AI-QUANTUM PRO 2026 - ULTIMATE VERSION 2.0
# Features: Multi-Source Scraping • Auto-Update • Advanced AI • Real-Time Stats
# =============================================================================
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import json
import os
import time
import re
import hashlib
import urllib.parse

# =============================================================================
# 🔧 CẤU HÌNH
# =============================================================================
st.set_page_config(
    page_title="💎 AI-QUANTUM PRO 2026 | Ultimate",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

GEMINI_API_KEY = "AIzaSyARQk3lpoHnK51LQ62OR4vciH0XMFFIZjg"

# =============================================================================
# 🎨 CSS - PROFESSIONAL UI
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #0a0e1a; color: #e6edf3; font-size: 16px; }
    .stApp { background-color: #0a0e1a; }
    
    .header-gold {
        background: linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #B8962E 100%);
        padding: 30px; border-radius: 20px; text-align: center; 
        color: #000; font-weight: 800; margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(212, 175, 55, 0.4);
    }
    
    .stat-card {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 2px solid #D4AF37; border-radius: 15px;
        padding: 20px; text-align: center; margin: 10px;
    }
    .stat-value { font-size: 36px; font-weight: 800; color: #FFD700; }
    .stat-label { font-size: 14px; color: #8b949e; margin-top: 8px; }
    .win { color: #00ff88; font-weight: bold; }
    .loss { color: #ff4b4b; font-weight: bold; }
    
    .result-box {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 3px solid #D4AF37; border-radius: 20px;
        padding: 30px; text-align: center; margin: 20px 0;
        box-shadow: 0 8px 32px rgba(212, 175, 55, 0.3);
    }
    .result-number {
        font-size: 56px; font-weight: 800; color: #ff4b4b;
        letter-spacing: 10px; text-shadow: 0 0 30px rgba(255, 75, 75, 0.6);
    }
    
    .pred-box {
        border: 2px solid #D4AF37; border-radius: 15px;
        padding: 25px; background: linear-gradient(135deg, #1a1f2e, #16213e);
        text-align: center; margin: 15px 0;
    }
    
    .live-indicator {
        display: inline-block; width: 10px; height: 10px;
        background: #00ff88; border-radius: 50%;
        animation: pulse 2s infinite; margin-right: 8px;
    }
    @keyframes pulse {
        0% { opacity: 1; box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
        70% { opacity: 0.7; box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); }
        100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
    }
    
    .vip-badge {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #000; padding: 6px 16px; border-radius: 20px;
        font-size: 12px; font-weight: 700; display: inline-block;
    }
    .free-badge {
        background: #30363d; color: #8b949e;
        padding: 6px 16px; border-radius: 20px;
        font-size: 12px; font-weight: 700; display: inline-block;
    }
    
    h1, h2, h3, h4 { font-weight: 700; color: #e6edf3; }
    
    .stButton > button {
        font-size: 16px; font-weight: 600; padding: 12px 28px;
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white; border: none; border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 🔐 AUTHENTICATION SYSTEM - IMPROVED
# =============================================================================
def init_auth():
    if 'users' not in st.session_state:
        st.session_state.users = {
            'admin': {
                'password_hash': hashlib.sha256('admin123'.encode()).hexdigest(),
                'role': 'admin',
                'vip_tier': None,
                'vip_until': None,
                'is_active': True,
                'created_at': datetime.now().isoformat()
            },
            'guest': {
                'password_hash': '',
                'role': 'user',
                'vip_tier': None,
                'vip_until': None,
                'is_active': True,
                'created_at': datetime.now().isoformat()
            }
        }
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    if 'login_time' not in st.session_state:
        st.session_state.login_time = None
    
    if os.path.exists('users.json'):
        try:
            with open('users.json', 'r', encoding='utf-8') as f:
                st.session_state.users = json.load(f)
        except:
            pass

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login(username, password):
    users = st.session_state.get('users', {})
    user = users.get(username)
    
    if user and user.get('is_active', True):
        if user['password_hash'] == hash_password(password):
            st.session_state.current_user = username
            st.session_state.login_time = datetime.now()
            return True
    return False

def logout():
    st.session_state.current_user = None
    st.session_state.login_time = None

def check_session_valid():
    if not st.session_state.current_user or not st.session_state.login_time:
        return False
    if datetime.now() - st.session_state.login_time > timedelta(hours=24):
        logout()
        return False
    return True

def get_user_info(username=None):
    if username is None:
        username = st.session_state.current_user
    if not username:
        return None
    return st.session_state.users.get(username)

def is_vip(username=None):
    user = get_user_info(username)
    if not user or not user.get('vip_tier'):
        return False
    if user.get('vip_until'):
        return datetime.fromisoformat(user['vip_until']) > datetime.now()
    return True

def get_vip_tier(username=None):
    user = get_user_info(username)
    return user.get('vip_tier') if user and is_vip(username) else 'free'

def save_users():
    try:
        with open('users.json', 'w', encoding='utf-8') as f:
            json.dump(st.session_state.users, f, ensure_ascii=False, indent=2)
    except:
        pass

def register_user(username, password):
    if username in st.session_state.users:
        return False, "Username đã tồn tại"
    st.session_state.users[username] = {
        'password_hash': hash_password(password),
        'role': 'user',
        'vip_tier': None,
        'vip_until': None,
        'is_active': True,
        'created_at': datetime.now().isoformat()
    }
    save_users()
    return True, "Đăng ký thành công"

# =============================================================================
# 💎 VIP TIERS
# =============================================================================
VIP_TIERS = {
    "free": {
        "name": "Miễn phí",
        "price": 0,
        "duration_days": 0,
        "features": ["Dự đoán cơ bản", "Kết quả real-time", "Thống kê 7 ngày"],
        "ai_priority": "normal",
        "data_depth": 7
    },
    "dong_hanh": {
        "name": "Gói Đồng hành",
        "price": 50000,
        "duration_days": 30,
        "features": [
            '<span class="check">✅</span> Tất cả tính năng Free',
            '<span class="check">✅</span> Thống kê tần suất 10 năm',
            '<span class="check">✅</span> AI Quantum phân tích nâng cao',
            '<span class="check">✅</span> Cập nhật real-time không delay',
            '<span class="check">✅</span> Không quảng cáo',
            '<span class="check">✅</span> Support 24/7'
        ],
        "ai_priority": "high",
        "data_depth": 3650
    },
    "nhiet_huyet": {
        "name": "Gói Nhiệt huyết",
        "price": 150000,
        "duration_days": 90,
        "features": [
            '<span class="check">✅</span> Tất cả tính năng Gói Đồng hành',
            '<span class="check">✅</span> Phân tích pattern chuyên sâu',
            '<span class="check">✅</span> Notification khi có số đẹp',
            '<span class="check">✅</span> Export dữ liệu CSV/Excel',
            '<span class="check">✅</span> Bonus: 1 tháng miễn phí'
        ],
        "ai_priority": "priority",
        "data_depth": 3650
    },
    "cong_hien": {
        "name": "Gói Cống hiến",
        "price": 500000,
        "duration_days": 365,
        "features": [
            '<span class="check">✅</span> Tất cả tính năng Gói Nhiệt huyết',
            '<span class="check">✅</span> Custom AI model training',
            '<span class="check">✅</span> Access community VIP',
            '<span class="check">✅</span> Priority support 1:1',
            '<span class="check">✅</span> Bonus: 2 tháng + Badge đặc biệt'
        ],
        "ai_priority": "ultra",
        "data_depth": 3650
    }
}

# =============================================================================
# 💳 PAYMENT - VIETQR
# =============================================================================
def generate_vietqr_url(account_number, amount, message, bank_id="BIDV"):
    encoded_message = urllib.parse.quote(message, safe='')
    vietqr_url = f"https://img.vietqr.io/image/{bank_id}-{account_number}-compact.png?amount={amount}&addInfo={encoded_message}"
    return vietqr_url

def display_pricing_table():
    BANK_INFO = {
        'account_name': 'NGUYEN XUAN DAT',
        'account_number': '4430269669',
        'bank_id': 'BIDV',
        'bank': 'BIDV - PGD Quảng Yên'
    }
    
    st.markdown('''
    <div style="text-align: center; margin: 30px 0;">
        <h3>🤝 ỦNG HỘ DUY TRÌ DỰ ÁN AI QUANTUM</h3>
        <p style="color: #8b949e; font-size: 15px;">
            Mọi khoản đóng góp sẽ được sử dụng để duy trì máy chủ và nâng cấp thuật toán AI.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    cols = st.columns(3)
    tier_keys = ['dong_hanh', 'nhiet_huyet', 'cong_hien']
    
    for idx, tier_key in enumerate(tier_keys):
        tier = VIP_TIERS[tier_key]
        
        with cols[idx]:
            st.markdown(f'''
            <div class="pricing-card">
                <h4 style="color: #FFD700; margin: 0;">{tier['name']}</h4>
                <h2 style="color: #fff; margin: 15px 0;">{tier['price']:,} VNĐ</h2>
            ''', unsafe_allow_html=True)
            
            for feature in tier['features']:
                st.markdown(f'<div style="margin: 8px 0; color: #e6edf3;">{feature}</div>', unsafe_allow_html=True)
            
            if st.button(f"🎁 Ủng hộ ngay", key=f"pay_{tier_key}", use_container_width=True):
                user_id = st.session_state.current_user or 'guest'
                transfer_message = f"{user_id} ung ho AI Quantum {tier_key}"
                qr_url = generate_vietqr_url(BANK_INFO['account_number'], tier['price'], transfer_message, BANK_INFO['bank_id'])
                
                st.success(f"✅ Gói **{tier['name']}**")
                st.image(qr_url, caption="📷 Quét QR để thanh toán", use_column_width=True)
                st.markdown(f"**💰 Số tiền:** {tier['price']:,} VNĐ")
                st.markdown(f"**📝 Nội dung:** {transfer_message}")
            
            st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 💾 STATISTICS - ENHANCED
# =============================================================================
def init_statistics():
    if 'statistics' not in st.session_state:
        st.session_state.statistics = {
            'predictions': [],
            'results': [],
            'daily_stats': {},
            'last_check_date': None,
            'last_update': None
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

# =============================================================================
# 📡 MULTI-SOURCE SCRAPING - IMPROVED
# =============================================================================
@st.cache_data(ttl=60)
def get_live_xsmb_multi_source():
    """Lấy kết quả từ nhiều nguồn để đảm bảo có dữ liệu"""
    
    sources = [
        "https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html",
        "https://xoso.com.vn/xsmb.htm",
        "https://minhngoc.net.vn/ket-qua-xo-so/mien-bac.html"
    ]
    
    for source_url in sources:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            res = requests.get(source_url, headers=headers, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.content, 'html.parser')
            
            result = extract_xsmb_data(soup, source_url)
            if result and result.get("Đặc Biệt") not in ["...", "", None]:
                result["time"] = datetime.now().strftime("%H:%M:%S %d/%m")
                result["source"] = source_url
                return result
        except:
            continue
    
    return get_mock_data()

def extract_xsmb_data(soup, source_url):
    """Extract data based on website structure"""
    
    if "xosodaiphat" in source_url:
        return extract_from_xosodaiphat(soup)
    elif "xoso.com.vn" in source_url:
        return extract_from_xoso(soup)
    elif "minhngoc" in source_url:
        return extract_from_minhngoc(soup)
    
    return None

def extract_from_xosodaiphat(soup):
    """Extract from xosodaiphat.com"""
    def get_number(element):
        if element:
            text = element.get_text().strip()
            match = re.search(r'\d{5}', text)
            return match.group() if match else "..."
        return "..."
    
    # Try different selectors
    db = get_number(soup.find(class_="special-temp")) or \
         get_number(soup.find(class_="special")) or \
         get_number(soup.find(class_="db-value"))
    
    return {
        "Đặc Biệt": db if db else "...",
        "Giải Nhất": get_number(soup.find(class_="g1-temp")) or "...",
        "Giải Nhì": ["...", "..."],
        "Giải Ba": ["...", "...", "...", "...", "...", "..."],
        "Giải Tư": ["...", "...", "...", "..."],
        "Giải Năm": ["...", "...", "...", "...", "...", "..."],
        "Giải Sáu": ["...", "...", "..."],
        "Giải Bảy": ["...", "...", "...", "..."],
    }

def extract_from_xoso(soup):
    """Extract from xoso.com.vn"""
    def get_number(selector):
        el = soup.select_one(selector)
        if el:
            text = el.get_text().strip()
            match = re.search(r'\d{5}', text)
            return match.group() if match else "..."
        return "..."
    
    return {
        "Đặc Biệt": get_number(".special-number") or "...",
        "Giải Nhất": get_number(".prize-1") or "...",
    }

def extract_from_minhngoc(soup):
    """Extract from minhngoc.net.vn"""
    def get_number(selector):
        el = soup.select_one(selector)
        if el:
            text = el.get_text().strip()
            match = re.search(r'\d{5}', text)
            return match.group() if match else "..."
        return "..."
    
    return {
        "Đặc Biệt": get_number(".special") or "...",
        "Giải Nhất": get_number(".prize1") or "...",
    }

def get_mock_data():
    return {
        "Đặc Biệt": "36948",
        "Giải Nhất": "96041",
        "Giải Nhì": ["09028", "27803"],
        "time": datetime.now().strftime("%H:%M:%S %d/%m"),
        "source": "Mock"
    }

# =============================================================================
# 🤖 ADVANCED AI PREDICTION
# =============================================================================
def query_gemini_ai_advanced(hot_nums, cold_nums, gan_nums, historical_patterns):
    """Advanced AI prediction with pattern analysis"""
    
    prompt = f"""Chuyên gia xổ số MB cao cấp. Phân tích nâng cao với AI.

DỮ LIỆU PHÂN TÍCH:
- Số NÓNG (xuất hiện nhiều): {', '.join(hot_nums[:5]) if hot_nums else '67, 07, 60, 14, 02'}
- Số LẠNH (ít xuất hiện): {', '.join(cold_nums[:5]) if cold_nums else '33, 44, 55, 66, 77'}
- Pattern lịch sử: {historical_patterns}

YÊU CẦU PHÂN TÍCH:
1. Phân tích xu hướng 3-5 ngày gần nhất
2. Dự đoán dựa trên chu kỳ và pattern
3. Đưa ra độ tin cậy cho từng con số

TRẢ LỜI JSON:
{{
    "analysis": "Phân tích chi tiết 2-3 câu về xu hướng và pattern",
    "bach_thu": "67",
    "song_thu": ["07", "60"],
    "xien_2": "67-07",
    "dan_de": ["00","01","02","07","10","13","14","60","67","93"],
    "confidence": 75,
    "reasoning": "Lý do chọn các con số này"
}}"""
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 512, "topP": 0.9}
        }
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                text = result['candidates'][0]['content']['parts'][0]['text']
                json_match = re.search(r'\{[\s\S]*\}', text)
                if json_match:
                    ai_result = json.loads(json_match.group())
                    required_keys = ['bach_thu', 'song_thu', 'dan_de', 'confidence']
                    if all(key in ai_result for key in required_keys):
                        return ai_result
    except:
        pass
    
    return None

def analyze_patterns(historical_data):
    """Phân tích pattern từ dữ liệu lịch sử"""
    patterns = {
        'hot_numbers': [],
        'cold_numbers': [],
        'frequent_pairs': [],
        'recent_trend': []
    }
    
    all_numbers = []
    for date, data in historical_data.items():
        for key, val in data.items():
            if isinstance(val, list):
                for v in val:
                    if v and len(v) >= 2:
                        all_numbers.append(v[-2:])
            elif val and len(val) >= 2:
                all_numbers.append(val[-2:])
    
    counter = Counter(all_numbers)
    patterns['hot_numbers'] = [num for num, _ in counter.most_common(10)]
    patterns['cold_numbers'] = [num for num in [f"{i:02d}" for i in range(100)] if num not in counter][:10]
    
    return patterns

def generate_predictions_advanced(historical_data):
    """Generate predictions with advanced AI"""
    
    patterns = analyze_patterns(historical_data)
    
    gemini_result = query_gemini_ai_advanced(
        patterns['hot_numbers'],
        patterns['cold_numbers'],
        [],
        f"Hot: {patterns['hot_numbers'][:3]}, Cold: {patterns['cold_numbers'][:3]}"
    )
    
    if gemini_result:
        return {
            "bach_thu": gemini_result.get('bach_thu', "00"),
            "song_thu": (gemini_result.get('song_thu', ["00", "00"])[0], 
                        gemini_result.get('song_thu', ["00", "00"])[1] if len(gemini_result.get('song_thu', [])) > 1 else "00"),
            "xien_2": gemini_result.get('xien_2', "00-00"),
            "dan_de": gemini_result.get('dan_de', [f"{i:02d}" for i in range(10)]),
            "confidence": gemini_result.get('confidence', 75),
            "ai_analysis": gemini_result.get('analysis', "AI phân tích nâng cao"),
            "using_ai": True
        }
    
    # Fallback
    return {
        "bach_thu": patterns['hot_numbers'][0] if patterns['hot_numbers'] else "00",
        "song_thu": (patterns['hot_numbers'][1] if len(patterns['hot_numbers']) > 1 else "00",
                    patterns['cold_numbers'][0] if patterns['cold_numbers'] else "00"),
        "xien_2": f"{patterns['hot_numbers'][0] if patterns['hot_numbers'] else '00'}-{patterns['hot_numbers'][1] if len(patterns['hot_numbers']) > 1 else '00'}",
        "dan_de": patterns['hot_numbers'][:10] if len(patterns['hot_numbers']) >= 10 else [f"{i:02d}" for i in range(10)],
        "confidence": 65,
        "ai_analysis": "Phân tích tần suất thống kê",
        "using_ai": False
    }

# =============================================================================
# 📊 REAL-TIME STATISTICS
# =============================================================================
def calculate_win_loss_stats(predictions, results):
    """Tính toán thống kê trúng/trượt"""
    stats = {
        'total_predictions': len(predictions),
        'total_checked': len(results),
        'bach_thu_wins': 0,
        'song_thu_wins': 0,
        'xien_wins': 0,
        'de_wins': 0,
        'total_wins': 0
    }
    
    for result in results:
        if result['bach_thu']['win']:
            stats['bach_thu_wins'] += 1
        if result['song_thu']['win']:
            stats['song_thu_wins'] += 1
        if result['xien_2']['win']:
            stats['xien_wins'] += 1
        if result['dan_de']['win']:
            stats['de_wins'] += 1
        stats['total_wins'] += result['overall_wins']
    
    return stats

# =============================================================================
# 🚀 MAIN APP
# =============================================================================
def main():
    init_auth()
    init_statistics()
    
    if not check_session_valid():
        st.session_state.current_user = None
    
    today = datetime.now().strftime("%d/%m")
    current_user = st.session_state.current_user
    user_tier = get_vip_tier()
    
    # LOGIN PAGE
    if not current_user:
        st.markdown(f'''
        <div class="header-gold">
            <h1 style="margin:0; font-size: 40px;">💎 AI-QUANTUM PRO 2026</h1>
            <p style="margin:10px 0 0; font-size: 18px;">📊 Ultimate Data Analytics Platform</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["🔑 Đăng nhập", "📝 Đăng ký"])
        
        with tab_login:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Đăng nhập", use_container_width=True, type="primary"):
                if login(username, password):
                    st.success("✅ Đăng nhập thành công!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Sai username hoặc password")
        
        with tab_register:
            new_username = st.text_input("Username mới", key="reg_user")
            new_password = st.text_input("Password", type="password", key="reg_pass")
            if st.button("Đăng ký", use_container_width=True):
                success, msg = register_user(new_username, new_password)
                if success:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("➡️ Dùng thử miễn phí", use_container_width=True):
            st.session_state.current_user = "guest"
            st.session_state.login_time = datetime.now()
            st.rerun()
        return
    
    # AUTO-UPDATE CHECK
    try:
        data = get_live_xsmb_multi_source()
        
        # Check if today's results are available
        if data.get("Đặc Biệt") and data.get("Đặc Biệt") != "...":
            # Auto-save if not already saved
            if not any(r['check_date'] == today for r in st.session_state.statistics.get('results', [])):
                save_result_check(today, data)
    except:
        data = get_mock_data()
    
    # HEADER
    st.markdown(f'''
    <div class="header-gold">
        <h1 style="margin:0; display:flex; justify-content:space-between; align-items:center;">
            <span>💎 AI-QUANTUM PRO 2026</span>
            <span style="font-size:16px;">
                <span class="live-indicator"></span>
                👤 {current_user} 
                <span class="{'vip-badge' if is_vip() else 'free-badge'}">
                    {VIP_TIERS[user_tier]['name'] if is_vip() else 'Free'}
                </span>
            </span>
        </h1>
        <p style="margin:10px 0 0; font-size: 16px;">🔄 Auto-Update • Real-Time Analytics</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown(f"### 👤 {current_user}")
        if is_vip():
            user_info = get_user_info()
            until = user_info.get('vip_until', '') if user_info else ''
            st.markdown(f'<span class="vip-badge">VIP đến: {until[:10] if until else "N/A"}</span>', unsafe_allow_html=True)
        
        if st.button("🚪 Đăng xuất", use_container_width=True):
            logout()
            st.rerun()
        
        st.divider()
        
        menu_options = ["🎯 Dự đoán", "💎 Gói VIP", "📊 Thống kê Real-Time", "📜 Lịch sử", "🌐 Website XS"]
        page = st.radio("🧭 Menu", menu_options)
    
    # MAIN CONTENT
    try:
        if page == "🎯 Dự đoán":
            st.markdown(f'''
            <div class="info-banner">
                <b>📅 DỰ LIỆU NGÀY {today}</b><br>
                <small><span class="live-indicator"></span>Auto-update • Cập nhật từ nhiều nguồn</small>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown("### 📡 KẾT QUẢ XSMB")
            
            historical = get_historical_data(30 if user_tier == 'free' else VIP_TIERS[user_tier]['data_depth'])
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"🕐 Cập nhật: **{data.get('time', 'N/A')}** từ {data.get('source', 'N/A')}")
            with col2:
                if st.button("🔄 Làm mới", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()
            
            db = data.get("Đặc Biệt", "....")
            st.markdown(f'''
            <div class="result-box">
                <div style="font-size: 22px; color: #8b949e; margin-bottom: 20px;">🏆 ĐẶC BIỆT</div>
                <div class="result-number">{db}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # PREDICTIONS
            st.markdown("---")
            st.markdown(f"### 🎯 DỰ ĐOÁN AI NGÀY {today}")
            
            today_pred, new_predictions = ensure_predictions_for_today()
            
            if new_predictions:
                predictions = new_predictions
            else:
                predictions = generate_predictions_advanced(historical)
                for p in st.session_state.statistics['predictions']:
                    if p['date'] == today:
                        p['predictions'] = predictions
                        p['confidence'] = predictions.get('confidence', 75)
                        p['ai_analysis'] = predictions.get('ai_analysis', '')
                        p['using_ai'] = predictions.get('using_ai', False)
                        break
                save_statistics()
            
            if predictions.get('using_ai', False):
                st.success("✅ Gemini AI Active - Phân tích nâng cao")
            else:
                st.info("ℹ️ Statistical Analysis Mode")
            
            st.markdown(f'''
            <div style="background: rgba(212, 175, 55, 0.15); border: 2px solid #D4AF37; border-radius: 12px; padding: 20px; margin: 20px 0;">
                <b style="font-size: 16px;">🧠 PHÂN TÍCH AI:</b><br>{predictions.get('ai_analysis', 'N/A')}<br>
                <b style="font-size: 16px;">📊 ĐỘ TIN CẬY:</b> {predictions.get('confidence', 0)}%
            </div>
            ''', unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'''<div class="pred-box"><div style="color:#8b949e;">🎯 BẠCH THỦ</div><div style="font-size:48px; color:#FFD700; font-weight:800; margin:15px 0;">{predictions.get('bach_thu', '00')}</div></div>''', unsafe_allow_html=True)
            with c2:
                st.markdown(f'''<div class="pred-box"><div style="color:#8b949e;">🎯 SONG THỦ</div><div style="font-size:28px; color:#FFD700; font-weight:700; margin:15px 0;">{predictions.get('song_thu', ('00','00'))[0]} - {predictions.get('song_thu', ('00','00'))[1]}</div></div>''', unsafe_allow_html=True)
            with c3:
                st.markdown(f'''<div class="pred-box"><div style="color:#8b949e;">🎯 XIÊN 2</div><div style="font-size:24px; color:#FFD700; font-weight:700; margin:15px 0;">{predictions.get('xien_2', '00-00')}</div></div>''', unsafe_allow_html=True)
            
            st.markdown(f'''<div class="pred-box"><div style="color:#8b949e; margin-bottom:15px;">📋 DÀN ĐỀ 10 SỐ</div><div style="font-size:20px; color:#e6edf3;">{', '.join(predictions.get('dan_de', []))}</div></div>''', unsafe_allow_html=True)
            
            # WIN/LOSS STATS
            st.markdown("---")
            st.markdown("### 📊 THỐNG KÊ TRÚNG/TRƯỢT")
            
            results = st.session_state.statistics.get('results', [])
            predictions_list = st.session_state.statistics.get('predictions', [])
            stats = calculate_win_loss_stats(predictions_list, results)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Bạch Thủ", f"{stats['bach_thu_wins']}/{stats['total_checked']}", f"{(stats['bach_thu_wins']/max(stats['total_checked'],1)*100):.0f}%")
            col2.metric("Song Thủ", f"{stats['song_thu_wins']}/{stats['total_checked']}", f"{(stats['song_thu_wins']/max(stats['total_checked'],1)*100):.0f}%")
            col3.metric("Xiên 2", f"{stats['xien_wins']}/{stats['total_checked']}", f"{(stats['xien_wins']/max(stats['total_checked'],1)*100):.0f}%")
            col4.metric("Đề", f"{stats['de_wins']}/{stats['total_checked']}", f"{(stats['de_wins']/max(stats['total_checked'],1)*100):.0f}%")
            
            st.markdown("---")
            st.markdown('<div style="background: rgba(248, 81, 73, 0.15); border-left: 4px solid #f85149; padding: 15px; border-radius: 0 8px 8px 0; margin: 20px 0;">⚠️ Dữ liệu phân tích chỉ mang tính chất tham khảo. Người dùng tự chịu trách nhiệm với hành vi sử dụng.</div>', unsafe_allow_html=True)
        
        elif page == "💎 Gói VIP":
            display_pricing_table()
        
        elif page == "📊 Thống kê Real-Time":
            st.markdown("### 📈 THỐNG KÊ CHI TIẾT")
            
            results = st.session_state.statistics.get('results', [])
            predictions_list = st.session_state.statistics.get('predictions', [])
            stats = calculate_win_loss_stats(predictions_list, results)
            
            col1, col2 = st.columns(2)
            col1.markdown(f'''<div class="stat-card"><div class="stat-value" style="color: #00ff88;">{stats['total_wins']}</div><div class="stat-label">Tổng số trúng</div></div>''', unsafe_allow_html=True)
            col2.markdown(f'''<div class="stat-card"><div class="stat-value" style="color: #ff4b4b;">{stats['total_checked'] * 4 - stats['total_wins']}</div><div class="stat-label">Tổng số trượt</div></div>''', unsafe_allow_html=True)
            
            st.markdown("### 📅 Thống kê theo ngày")
            daily_stats = st.session_state.statistics.get('daily_stats', {})
            if daily_stats:
                for date in sorted(daily_stats.keys(), reverse=True)[:7]:
                    stats_day = daily_stats[date]
                    checked = stats_day.get('checked', 0)
                    wins = stats_day.get('total_wins', 0)
                    rate = (wins / (checked * 4) * 100) if checked > 0 else 0
                    
                    st.markdown(f'''
                    <div class="stat-card" style="margin: 10px 0;">
                        <div style="font-size: 16px; font-weight: 700; color: #fff;">📅 {date}</div>
                        <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                            <span class="win">✅ {wins}/4 trúng</span>
                            <span style="color: {'#00ff88' if rate >= 50 else '#ff4b4b'}; font-size: 16px; font-weight: 700;">{rate:.0f}%</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("📭 Chưa có dữ liệu thống kê")
        
        elif page == "📜 Lịch sử":
            st.markdown("### 📜 LỊCH SỬ DỰ ĐOÁN")
            results = st.session_state.statistics.get('results', [])
            if results:
                dates = sorted(set(r['pred_date'] for r in results), reverse=True)
                selected_date = st.selectbox("Chọn ngày", dates)
                filtered = [r for r in results if r['pred_date'] == selected_date]
                for r in filtered:
                    st.markdown(f'''
                    <div style="background: rgba(212, 175, 55, 0.15); border: 2px solid #D4AF37; border-radius: 12px; padding: 20px; margin: 15px 0;">
                        <b style="font-size: 16px;">📅 {r['pred_date']} → {r['check_date']}</b>
                        <div style="margin-top: 15px;">
                            <div style="padding: 12px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between;">
                                <span>🎯 Bạch thủ: <b>{r['bach_thu']['number']}</b></span>
                                <span class="{'win' if r['bach_thu']['win'] else 'loss'}">{'✅' if r['bach_thu']['win'] else '❌'}</span>
                            </div>
                            <div style="padding: 12px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between;">
                                <span>🎯 Song thủ: <b>{'-'.join(r['song_thu']['numbers'])}</b></span>
                                <span class="{'win' if r['song_thu']['win'] else 'loss'}">{'✅' if r['song_thu']['win'] else '❌'}</span>
                            </div>
                            <div style="padding: 12px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between;">
                                <span>🎯 Xiên 2: <b>{r['xien_2']['pair']}</b></span>
                                <span class="{'win' if r['xien_2']['win'] else 'loss'}">{'✅' if r['xien_2']['win'] else '❌'}</span>
                            </div>
                            <div style="padding: 12px; display: flex; justify-content: space-between;">
                                <span>📋 Đề</span>
                                <span class="{'win' if r['dan_de']['win'] else 'loss'}">{'✅' if r['dan_de']['win'] else '❌'}</span>
                            </div>
                        </div>
                        <div style="margin-top: 15px; text-align: right; font-size: 18px; font-weight: 700;">
                            Tổng: {r['overall_wins']}/4 trúng
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("📭 Chưa có lịch sử")
        
        elif page == "🌐 Website XS":
            st.markdown("### 🌐 XOSODAIPHAT.COM")
            st.markdown('''
            <div class="iframe-container">
                <iframe src="https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html" 
                        style="width:100%; height:100%; border:none;"
                        sandbox="allow-same-origin allow-scripts allow-forms">
                </iframe>
            </div>
            ''', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Lỗi hệ thống: {str(e)}")

if __name__ == "__main__":
    main()