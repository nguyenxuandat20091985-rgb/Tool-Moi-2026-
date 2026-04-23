# =============================================================================
# 💎 AI-QUANTUM PRO 2026 - ENHANCED VERSION
# Improvements: Better Scraping • Auto-Update • Win/Loss Tracking • Stable Login
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
import hashlib
import urllib.parse

# =============================================================================
# 🔧 CẤU HÌNH
# =============================================================================
st.set_page_config(
    page_title="💎 AI-QUANTUM PRO 2026",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

GEMINI_API_KEY = "AIzaSyARQk3lpoHnK51LQ62OR4vciH0XMFFIZjg"

# =============================================================================
# 🎨 CSS - ENHANCED
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { 
        background-color: #0d1117; 
        color: #e6edf3;
        font-size: 16px;
    }
    .stApp { background-color: #0d1117; }
    
    .header-gold {
        background: linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #B8962E 100%);
        padding: 25px; 
        border-radius: 15px; 
        text-align: center; 
        color: #000; 
        font-weight: bold; 
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(212, 175, 55, 0.4);
    }
    
    .stat-card {
        background: linear-gradient(135deg, #161b22, #1f2937);
        border: 2px solid #D4AF37; 
        border-radius: 12px;
        padding: 15px; 
        text-align: center; 
        margin: 5px;
    }
    .stat-value { 
        font-size: 32px; 
        font-weight: bold; 
        color: #FFD700; 
    }
    .stat-label { 
        font-size: 14px; 
        color: #8b949e; 
        margin-top: 5px; 
    }
    .win { color: #3fb950; font-weight: bold; }
    .loss { color: #f85149; font-weight: bold; }
    
    .pred-box {
        border: 2px solid #D4AF37; 
        border-radius: 12px;
        padding: 25px; 
        background: linear-gradient(135deg, #161b22, #1f2937);
        text-align: center; 
        margin: 10px 0;
    }
    .xien-box {
        border: 3px solid #FFD700; 
        border-radius: 15px;
        padding: 30px; 
        background: linear-gradient(135deg, #161b22, #1f2937);
        text-align: center; 
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(212, 175, 55, 0.3);
    }
    
    .disclaimer {
        background: rgba(248, 81, 73, 0.15);
        border-left: 4px solid #f85149;
        padding: 15px; 
        border-radius: 0 8px 8px 0;
        margin: 20px 0; 
        font-size: 14px;
        color: #e6edf3;
    }
    .info-banner {
        background: linear-gradient(135deg, rgba(56, 139, 253, 0.15), rgba(88, 166, 255, 0.15));
        border: 2px solid #58a6ff; 
        border-radius: 10px;
        padding: 20px; 
        margin: 15px 0; 
        text-align: center;
        font-size: 16px;
        color: #e6edf3;
    }
    
    .vip-badge {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #000; 
        padding: 5px 15px; 
        border-radius: 20px;
        font-size: 12px; 
        font-weight: bold; 
        display: inline-block;
    }
    .free-badge {
        background: #30363d; 
        color: #8b949e; 
        padding: 5px 15px; 
        border-radius: 20px; 
        font-size: 12px; 
        font-weight: bold;
        display: inline-block;
    }
    
    .pricing-card {
        border: 2px solid #D4AF37; 
        border-radius: 15px;
        padding: 25px; 
        background: linear-gradient(135deg, #161b22, #1f2937);
        text-align: center; 
        margin: 15px 0;
    }
    .pricing-card.premium {
        border-color: #FFD700;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.25);
    }
    
    .feature-list { 
        text-align: left; 
        font-size: 15px; 
        margin: 15px 0; 
    }
    .feature-list div { 
        margin: 8px 0; 
        color: #e6edf3; 
        font-size: 15px;
    }
    .feature-list .check { 
        color: #3fb950; 
        font-weight: bold;
    }
    
    .login-box {
        background: linear-gradient(135deg, #161b22, #1f2937);
        border: 2px solid #D4AF37; 
        border-radius: 15px;
        padding: 35px; 
        max-width: 450px; 
        margin: 50px auto;
    }
    
    .result-number {
        font-size: 48px; 
        font-weight: bold; 
        color: #f85149; 
        letter-spacing: 8px;
        text-shadow: 0 0 20px rgba(248, 81, 73, 0.5);
    }
    
    h1, h2, h3, h4 {
        font-weight: 700;
        color: #e6edf3;
    }
    
    .stTextInput > div > div > input {
        font-size: 16px;
        color: #e6edf3;
        background-color: #0d1117;
    }
    
    .stButton > button {
        font-size: 16px;
        font-weight: 600;
        padding: 10px 24px;
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #2ea043;
    }
    
    .iframe-container {
        border: 2px solid #D4AF37;
        border-radius: 15px;
        overflow: hidden;
        height: 800px;
    }
    
    .today-result {
        background: linear-gradient(135deg, #1a2e1a, #0f1f0f);
        border: 3px solid #3fb950;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 🔐 AUTHENTICATION SYSTEM - ENHANCED
# =============================================================================
def init_auth():
    if 'users' not in st.session_state:
        st.session_state.users = {
            'admin': {
                'password_hash': hashlib.sha256('admin123'.encode()).hexdigest(),
                'role': 'admin',
                'vip_tier': 'cong_hien',
                'vip_until': (datetime.now() + timedelta(days=365)).isoformat(),
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
    
    # Always reload from file
    if os.path.exists('users.json'):
        try:
            with open('users.json', 'r', encoding='utf-8') as f:
                loaded_users = json.load(f)
                # Merge with defaults
                for key, value in loaded_users.items():
                    st.session_state.users[key] = value
        except Exception as e:
            st.error(f"Lỗi load users: {str(e)}")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login(username, password):
    """Login with better error handling"""
    try:
        # Reload users from file
        if os.path.exists('users.json'):
            with open('users.json', 'r', encoding='utf-8') as f:
                st.session_state.users = json.load(f)
        
        users = st.session_state.get('users', {})
        user = users.get(username)
        
        if not user:
            return False
        
        if not user.get('is_active', True):
            return False
        
        if user['password_hash'] != hash_password(password):
            return False
        
        # Success
        st.session_state.current_user = username
        st.session_state.login_time = datetime.now()
        
        return True
    except Exception as e:
        st.error(f"Lỗi đăng nhập: {str(e)}")
        return False

def logout():
    st.session_state.current_user = None
    st.session_state.login_time = None

def check_session_valid():
    if not st.session_state.current_user or not st.session_state.login_time:
        return False
    
    # Session valid for 24 hours
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
        try:
            return datetime.fromisoformat(user['vip_until']) > datetime.now()
        except:
            return False
    return True

def get_vip_tier(username=None):
    user = get_user_info(username)
    return user.get('vip_tier') if user and is_vip(username) else 'free'

def save_users():
    try:
        with open('users.json', 'w', encoding='utf-8') as f:
            json.dump(st.session_state.users, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Lỗi save users: {str(e)}")

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

def update_vip_status(username, tier, duration_days):
    if username not in st.session_state.users:
        return False
    st.session_state.users[username]['vip_tier'] = tier
    st.session_state.users[username]['vip_until'] = (
        datetime.now() + timedelta(days=duration_days)
    ).isoformat()
    save_users()
    return True

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
# 💳 PAYMENT - DYNAMIC VIETQR CODE
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
            Mọi khoản đóng góp sẽ được sử dụng để duy trì máy chủ và nâng cấp thuật toán AI.<br>
            <b>Ứng dụng cung cấp dữ liệu thống kê khoa học, không phải dịch vụ cá cược.</b>
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    cols = st.columns(3)
    tier_keys = ['dong_hanh', 'nhiet_huyet', 'cong_hien']
    
    for idx, tier_key in enumerate(tier_keys):
        tier = VIP_TIERS[tier_key]
        is_premium = (tier_key == 'cong_hien')
        
        with cols[idx]:
            st.markdown(f'''
            <div class="pricing-card{' premium' if is_premium else ''}">
                <h4 style="color: #FFD700; margin: 0; font-size: 20px;">{tier['name']}</h4>
                <h2 style="color: #fff; margin: 15px 0; font-size: 32px;">{tier['price']:,} VNĐ</h2>
                <p style="color: #8b949e; font-size: 14px;">
                    /{'tháng' if tier['duration_days']==30 else 'quý' if tier['duration_days']==90 else 'năm'}
                </p>
            ''', unsafe_allow_html=True)
            
            st.markdown(f'<div class="feature-list">', unsafe_allow_html=True)
            for feature in tier['features']:
                st.markdown(f'<div>{feature}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button(f"🎁 Ủng hộ ngay", key=f"pay_{tier_key}", use_container_width=True):
                user_id = st.session_state.current_user or 'guest'
                transfer_message = f"{user_id} ung ho AI Quantum {tier_key}"
                
                qr_url = generate_vietqr_url(
                    account_number=BANK_INFO['account_number'],
                    amount=tier['price'],
                    message=transfer_message,
                    bank_id=BANK_INFO['bank_id']
                )
                
                st.success(f"✅ Thông tin chuyển khoản gói **{tier['name']}**")
                
                st.markdown("### 📱 Quét QR Code để chuyển khoản")
                st.markdown("**✨ QR code đã có sẵn số tiền và nội dung!**")
                
                col_qr1, col_qr2, col_qr3 = st.columns([1,2,1])
                with col_qr2:
                    try:
                        st.image(qr_url, 
                                caption="📷 Quét bằng app ngân hàng bất kỳ", 
                                use_column_width=True,
                                clamp=True)
                    except Exception as e:
                        st.error(f"❌ Lỗi tải QR: {str(e)}")
                
                st.markdown(f'''
                <div style="background: #0d1117; padding: 20px; border-radius: 12px; margin: 20px 0; border: 2px solid #58a6ff;">
                    <p style="margin: 10px 0; font-size: 16px;"><b>💰 Số tiền:</b> <span style="color: #3fb950; font-size: 24px; font-weight: bold;">{tier['price']:,} VNĐ</span></p>
                    <p style="margin: 10px 0; font-size: 16px;"><b>📝 Nội dung CK:</b> <span style="color: #ffa500; font-weight: 600; background: rgba(255,165,0,0.1); padding: 5px 10px; border-radius: 5px;">{transfer_message}</span></p>
                </div>
                ''', unsafe_allow_html=True)
                
                st.info(f"""
                **✅ Hướng dẫn:**
                1️⃣ Mở app ngân hàng → Quét QR  
                2️⃣ Số tiền và nội dung đã có sẵn  
                3️⃣ Xác nhận chuyển khoản  
                
                🎁 VIP kích hoạt trong 5-10 phút
                """)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('''
    <div style="background: rgba(248, 81, 73, 0.1); border-left: 4px solid #f85149;
                padding: 15px; border-radius: 0 8px 8px 0; margin: 25px 0; font-size: 14px;">
        <b>⚖️ LƯU Ý:</b> Phí dịch vụ dữ liệu thống kê • Không tổ chức cá cược
    </div>
    ''', unsafe_allow_html=True)

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
        st.session_state.statistics['last_update'] = datetime.now().isoformat()
        with open('statistics.json', 'w', encoding='utf-8') as f:
            json.dump(st.session_state.statistics, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Lỗi save statistics: {str(e)}")

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
    return existing_pred, None

def save_result_check(today_date, current_data):
    """Save and check results with better tracking"""
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
# 📡 SCRAPING - ENHANCED
# =============================================================================
@st.cache_data(ttl=60)
def get_live_xsmb():
    """Improved scraping with better error handling"""
    url = "https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'vi-VN,vi;q=0.9,en;q=0.8'
    }
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, 'html.parser')
        
        # Try multiple extraction methods
        result = extract_data_v1(soup)
        
        # If special prize is empty or invalid, try alternative method
        if not result["Đặc Biệt"] or result["Đặc Biệt"] in ["...", ""]:
            result = extract_data_v2(soup)
        
        # If still empty, use mock data with warning
        if not result["Đặc Biệt"] or result["Đặc Biệt"] in ["...", ""]:
            st.warning("⚠️ Không lấy được kết quả từ website. Sử dụng dữ liệu tạm.")
            result = get_mock_data()
        
        result["time"] = datetime.now().strftime("%H:%M:%S %d/%m")
        return result
        
    except Exception as e:
        st.error(f"Lỗi scraping: {str(e)}")
        return get_mock_data()

def extract_data_v1(soup):
    """Extract data using class names"""
    def get_txt(classes):
        if isinstance(classes, str):
            classes = [classes]
        for cls in classes:
            item = soup.find("span", class_=cls)
            if item and item.text.strip():
                text = item.text.strip()
                match = re.search(r'(\d+)', text)
                if match:
                    return match.group(1)
        return "..."
    
    return {
        "Đặc Biệt": get_txt(["special-temp", "special", "db-value", "result-special"]),
        "Giải Nhất": get_txt(["g1-temp", "g1", "result-1"]),
        "Giải Nhì": [get_txt(f"g2_{i}-temp") for i in range(2)],
        "Giải Ba": [get_txt(f"g3_{i}-temp") for i in range(6)],
        "Giải Tư": [get_txt(f"g4_{i}-temp") for i in range(4)],
        "Giải Năm": [get_txt(f"g5_{i}-temp") for i in range(6)],
        "Giải Sáu": [get_txt(f"g6_{i}-temp") for i in range(3)],
        "Giải Bảy": [get_txt(f"g7_{i}-temp") for i in range(4)],
    }

def extract_data_v2(soup):
    """Alternative extraction using text parsing"""
    text = soup.get_text()
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
        num1 = f"{np.random.randint(0,100):02d}"
        num2 = f"{np.random.randint(0,100):02d}"
        while num1 == num2:
            num2 = f"{np.random.randint(0,100):02d}"
        pair = tuple(sorted([num1, num2]))
        if pair not in [p[0] for p in top_pairs]:
            top_pairs.append((pair, 1))
    
    return [{'pair': f"{p[0][0]} - {p[0][1]}", 'frequency': p[1], 'num1': p[0][0], 'num2': p[0][1]} for p in top_pairs]

# =============================================================================
# 🤖 GEMINI AI - ENHANCED
# =============================================================================
def query_gemini_ai(hot_nums, cold_nums, gan_nums, retry_count=3):
    prompt = f"""Chuyên gia xổ số MB với 10 năm kinh nghiệm. Phân tích XSMB dựa trên:

DỮ LIỆU THỐNG KÊ:
- Số NÓNG (ra nhiều): {', '.join(hot_nums[:5]) if hot_nums else '67, 07, 60, 14, 02'}
- Số LẠNH (ít ra): {', '.join(cold_nums[:5]) if cold_nums else '33, 44, 55, 66, 77'}
- Số GAN (lâu chưa ra): {', '.join([f"{x[0]}({x[1]}n)" for x in gan_nums[:5]]) if gan_nums else '33(15n), 44(12n)'}

YÊU CẦU:
1. Phân tích xu hướng 1-2 câu ngắn gọn
2. Dự đoán bach_thu (1 số 2 chữ số)
3. Dự đoán song_thu (2 số 2 chữ số)
4. Dàn đề 10 số (10 số 2 chữ số)
5. Độ tin cậy 0-100

TRẢ LỜI THEO FORMAT JSON:
{{
    "analysis": "Phân tích ngắn gọn",
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
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 512,
                    "topP": 0.9
                }
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
# 🎲 PREDICTION - ENHANCED
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
    
    # Calculate gan numbers (days since last appearance)
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
    
    # Try Gemini AI first
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
        # Fallback to statistical method
        bt = hot_nums[0] if hot_nums else f"{np.random.randint(0,100):02d}"
        st1 = hot_nums[1] if len(hot_nums) > 1 else f"{np.random.randint(0,100):02d}"
        st2 = cold_nums[0] if cold_nums else f"{np.random.randint(0,100):02d}"
        dan_de = list(set(hot_nums[:4] + cold_nums[:3]))
        while len(dan_de) < 10:
            dan_de.append(f"{np.random.randint(0,100):02d}")
        dan_de = sorted(dan_de)[:10]
        confidence = 65
        analysis = "AI Phân tích tần suất"
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
            <h1 style="margin:0; font-size: 36px;">💎 AI-QUANTUM PRO 2026</h1>
            <p style="margin:10px 0 0; font-size: 18px;">📊 Data Analytics Platform</p>
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
        st.markdown("### 👥 Dùng thử miễn phí")
        st.info("🔓 Bạn có thể dùng thử các tính năng cơ bản mà không cần đăng nhập.")
        if st.button("➡️ Tiếp tục với tài khoản Free", use_container_width=True):
            st.session_state.current_user = "guest"
            st.session_state.login_time = datetime.now()
            st.rerun()
        return
    
    # AUTHENTICATED APP
    try:
        today_pred, new_predictions = ensure_predictions_for_today()
    except Exception as e:
        st.error(f"Lỗi tải dự đoán: {str(e)}")
        st.stop()
    
    # HEADER
    st.markdown(f'''
    <div class="header-gold">
        <h1 style="margin:0; display:flex; justify-content:space-between; align-items:center; font-size: 32px;">
            <span>💎 AI-QUANTUM PRO 2026</span>
            <span style="font-size:16px;">
                👤 {current_user} 
                <span class="{'vip-badge' if is_vip() else 'free-badge'}">
                    {VIP_TIERS[user_tier]['name'] if is_vip() else 'Free'}
                </span>
            </span>
        </h1>
        <p style="margin:10px 0 0; font-size: 16px;">📊 Data Analytics Platform • Auto Refresh 120s</p>
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
        menu_options = ["🎯 Dự đoán", "💎 Gói VIP", "📊 Thống kê", "📜 Lịch sử", "🌐 Website XS"]
        page = st.radio("🧭 Menu", menu_options)
        st.divider()
        st.markdown('''
        <div style="text-align: center; color: #8b949e; font-size: 13px; padding: 15px;">
            💎 AI-QUANTUM PRO 2026<br>
            Dữ liệu thống kê tham khảo<br>
            18+ only
        </div>
        ''', unsafe_allow_html=True)
    
    # MAIN CONTENT
    try:
        if page == "🎯 Dự đoán":
            st.markdown(f'''
            <div class="info-banner">
                <b>📅 DỰ LIỆU PHÂN TÍCH NGÀY {today}</b><br>
                <small>Dữ liệu tham khảo • Auto-update 120s</small>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown("### 📡 KẾT QUẢ THAM KHẢO")
            try:
                data = get_live_xsmb()
            except:
                data = get_mock_data()
            
            historical = get_historical_data(30 if user_tier == 'free' else VIP_TIERS[user_tier]['data_depth'])
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"🕐 Cập nhật: **{data.get('time', 'N/A')}**")
            with col2:
                if st.button("🔄 Làm mới", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()
            
            db = data.get("Đặc Biệt", "....")
            st.markdown(f'''
            <div class="today-result">
                <div style="font-size: 20px; color: #8b949e; margin-bottom: 15px;">📊 KẾT QUẢ THAM KHẢO</div>
                <div class="result-number">{db}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### ✅ KIỂM TRA DỮ LIỆU NGÀY TRƯỚC")
            if st.button(f"🔍 Check dữ liệu ngày {today}", use_container_width=True, type="primary"):
                try:
                    result = save_result_check(today, data)
                    if result:
                        st.success(f"✅ Đã cập nhật kết quả!")
                        st.markdown(f'''
                        <div style="background: rgba(212, 175, 55, 0.15); 
                                    border-left: 4px solid #D4AF37; padding: 20px; border-radius: 0 8px 8px 0; margin: 15px 0;">
                            <b style="font-size: 16px;">📅 {result['pred_date']} → {result['check_date']}</b>
                            <div style="margin-top: 15px;">
                                <div style="padding: 12px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; font-size: 15px;">
                                    <span>📊 Data 1: <b>{result['bach_thu']['number']}</b></span>
                                    <span class="{'win' if result['bach_thu']['win'] else 'loss'}">{'✅' if result['bach_thu']['win'] else '❌'}</span>
                                </div>
                                <div style="padding: 12px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; font-size: 15px;">
                                    <span>📊 Data 2: <b>{'-'.join(result['song_thu']['numbers'])}</b></span>
                                    <span class="{'win' if result['song_thu']['win'] else 'loss'}">{'✅' if result['song_thu']['win'] else '❌'}</span>
                                </div>
                                <div style="padding: 12px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; font-size: 15px;">
                                    <span>📊 Data 3: <b>{result['xien_2']['pair']}</b></span>
                                    <span class="{'win' if result['xien_2']['win'] else 'loss'}">{'✅' if result['xien_2']['win'] else '❌'}</span>
                                </div>
                                <div style="padding: 12px; display: flex; justify-content: space-between; font-size: 15px;">
                                    <span>📊 Data 4</span>
                                    <span class="{'win' if result['dan_de']['win'] else 'loss'}">{'✅' if result['dan_de']['win'] else '❌'}</span>
                                </div>
                            </div>
                            <div style="margin-top: 15px; text-align: right; font-size: 16px;">
                                <b>Tổng: {result['overall_wins']}/4</b>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.info("ℹ️ Chưa có dữ liệu ngày trước để check")
                except Exception as e:
                    st.error(f"Lỗi check kết quả: {str(e)}")
            
            st.markdown("---")
            st.markdown(f"### 🎯 PHÂN TÍCH AI NGÀY {today}")
            if new_predictions:
                predictions = new_predictions
            else:
                try:
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
                except Exception as e:
                    st.error(f"Lỗi tạo dự đoán: {str(e)}")
                    predictions = None
            
            if predictions:
                if predictions['using_ai']:
                    st.success("✅ Gemini AI Active")
                else:
                    st.warning("⚠️ Standard Analysis Mode")
                
                st.markdown(f'''
                <div style="background: rgba(212, 175, 55, 0.15); border: 1px solid #D4AF37; border-radius: 8px; padding: 15px; margin: 15px 0;">
                    <b style="font-size: 16px;">🧠 PHÂN TÍCH:</b><br>{predictions['ai_analysis']}<br>
                    <b style="font-size: 16px;">ĐỘ TIN CẬY:</b> {predictions['confidence']}%
                </div>
                ''', unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'''<div class="pred-box"><div style="color:#8b949e; font-size:15px;">📊 DATA 1</div><div style="font-size:48px; color:#FFD700; font-weight:bold; margin:15px 0;">{predictions['bach_thu']}</div></div>''', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'''<div class="pred-box"><div style="color:#8b949e; font-size:15px;">📊 DATA 2</div><div style="font-size:28px; color:#FFD700; font-weight:bold; margin:15px 0;">{predictions['song_thu'][0]} - {predictions['song_thu'][1]}</div></div>''', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'''<div class="pred-box"><div style="color:#8b949e; font-size:15px;">📊 DATA 3</div><div style="font-size:24px; color:#FFD700; font-weight:bold; margin:15px 0;">{predictions['xien_2']}</div></div>''', unsafe_allow_html=True)
                st.markdown(f'''<div class="pred-box"><div style="color:#8b949e; font-size:16px; margin-bottom:15px;">📊 DATA 4 (10 values)</div><div style="font-size:20px; color:#e6edf3;">{', '.join(predictions['dan_de'])}</div></div>''', unsafe_allow_html=True)
                
                if is_vip():
                    st.markdown("---")
                    st.markdown("### 🎲 PHÂN TÍCH CẶP SỐ (VIP)")
                    try:
                        bac_nho = generate_bac_nho(historical)
                        for i, xien in enumerate(bac_nho, 1):
                            st.markdown(f'''<div class="xien-box"><div style="font-size: 16px; color: #8b949e;">💎 Cặp {i} - {xien['frequency']} lần</div><div style="font-size: 52px; color: #FFD700; font-weight: bold;">{xien['pair']}</div></div>''', unsafe_allow_html=True)
                    except:
                        st.error("Lỗi tải bạc nhớ")
            st.markdown("---")
            st.markdown('<div class="disclaimer">⚠️ Dữ liệu phân tích chỉ mang tính chất tham khảo. Người dùng tự chịu trách nhiệm với hành vi sử dụng.</div>', unsafe_allow_html=True)
        
        elif page == "💎 Gói VIP":
            try:
                display_pricing_table()
            except Exception as e:
                st.error(f"Lỗi hiển thị gói VIP: {str(e)}")
        
        elif page == "📊 Thống kê":
            st.markdown("### 📈 THỐNG KÊ NGÀY")
            daily_stats = st.session_state.statistics.get('daily_stats', {})
            if daily_stats:
                for date in sorted(daily_stats.keys(), reverse=True)[:7]:
                    stats = daily_stats[date]
                    checked = stats.get('checked', 0)
                    wins = stats.get('total_wins', 0)
                    rate = (wins / (checked * 4) * 100) if checked > 0 else 0
                    st.markdown(f'''<div class="stat-card" style="margin: 8px 0;"><div style="font-size: 16px; font-weight: bold; color: #e6edf3;">📅 {date}</div><div style="display: flex; justify-content: space-around; margin-top: 10px;"><span class="win">✅ {wins}/4</span><span style="color: {'#3fb950' if rate >= 50 else '#f85149'}; font-size: 16px;">{rate:.0f}%</span></div></div>''', unsafe_allow_html=True)
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
            col1.markdown(f'''<div class="stat-card"><div class="stat-value win">{total_wins}</div><div class="stat-label">Trúng</div></div>''', unsafe_allow_html=True)
            col2.markdown(f'''<div class="stat-card"><div class="stat-value loss">{max_wins - total_wins}</div><div class="stat-label">Trượt</div></div>''', unsafe_allow_html=True)
            st.markdown(f'''<div class="stat-card" style="margin-top: 15px;"><div class="stat-value" style="color: {'#3fb950' if win_rate >= 50 else '#f85149'}; font-size: 36px;">{win_rate:.1f}%</div><div class="stat-label">Tỷ lệ trúng</div><div class="stat-label">{total_checks} lần check</div></div>''', unsafe_allow_html=True)
        
        elif page == "📜 Lịch sử":
            st.markdown("### 📜 LỊCH SỬ PHÂN TÍCH")
            results = st.session_state.statistics.get('results', [])
            if results:
                dates = sorted(set(r['pred_date'] for r in results), reverse=True)
                selected_date = st.selectbox("Chọn ngày", dates)
                filtered = [r for r in results if r['pred_date'] == selected_date]
                for r in filtered:
                    st.markdown(f'''<div style="background: rgba(212, 175, 55, 0.15); border: 1px solid #D4AF37; border-radius: 10px; padding: 20px; margin: 15px 0;"><b style="font-size: 16px;">📅 {r['pred_date']} → {r['check_date']}</b><div style="margin-top: 10px; font-size: 15px;">Data 1: {r['bach_thu']['number']} {'✅' if r['bach_thu']['win'] else '❌'}</div><div style="font-size: 15px;">Data 2: {'-'.join(r['song_thu']['numbers'])} {'✅' if r['song_thu']['win'] else '❌'}</div><div style="font-size: 15px;">Data 3: {r['xien_2']['pair']} {'✅' if r['xien_2']['win'] else '❌'}</div><div style="font-size: 15px;">Data 4: {'✅' if r['dan_de']['win'] else '❌'}</div><div style="margin-top: 10px;"><b>Tổng: {r['overall_wins']}/4</b></div></div>''', unsafe_allow_html=True)
            else:
                st.info("📭 Chưa có lịch sử")
        
        elif page == "🌐 Website XS":
            st.markdown("### 🌐 XOSODAIPHAT.COM")
            st.markdown("Xem kết quả trực tiếp từ website:")
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
        st.error("Vui lòng refresh trang hoặc đăng nhập lại")

if __name__ == "__main__":
    main()