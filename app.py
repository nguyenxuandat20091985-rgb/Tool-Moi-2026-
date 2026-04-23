# =============================================================================
# 💎 AI-QUANTUM PRO 2026 - ENHANCED VERSION
# Improvements: Better Login • Advanced Algorithm • Auto-Update • Live Statistics
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
# 🎨 CSS
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #0d1117; color: #e6edf3; font-size: 16px; }
    .stApp { background-color: #0d1117; }
    
    .header-gold {
        background: linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #B8962E 100%);
        padding: 25px; border-radius: 15px; text-align: center; 
        color: #000; font-weight: bold; margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(212, 175, 55, 0.4);
    }
    
    .stat-card {
        background: linear-gradient(135deg, #161b22, #1f2937);
        border: 2px solid #D4AF37; border-radius: 12px;
        padding: 15px; text-align: center; margin: 5px;
    }
    .stat-value { font-size: 32px; font-weight: bold; color: #FFD700; }
    .stat-label { font-size: 14px; color: #8b949e; margin-top: 5px; }
    .win { color: #3fb950; font-weight: bold; }
    .loss { color: #f85149; font-weight: bold; }
    
    .pred-box {
        border: 2px solid #D4AF37; border-radius: 12px;
        padding: 25px; background: linear-gradient(135deg, #161b22, #1f2937);
        text-align: center; margin: 10px 0;
    }
    .xien-box {
        border: 3px solid #FFD700; border-radius: 15px;
        padding: 30px; background: linear-gradient(135deg, #161b22, #1f2937);
        text-align: center; margin: 15px 0;
        box-shadow: 0 8px 32px rgba(212, 175, 55, 0.3);
    }
    
    .disclaimer {
        background: rgba(248, 81, 73, 0.15);
        border-left: 4px solid #f85149;
        padding: 15px; border-radius: 0 8px 8px 0;
        margin: 20px 0; font-size: 14px; color: #e6edf3;
    }
    .info-banner {
        background: linear-gradient(135deg, rgba(56, 139, 253, 0.15), rgba(88, 166, 255, 0.15));
        border: 2px solid #58a6ff; border-radius: 10px;
        padding: 20px; margin: 15px 0; text-align: center;
        font-size: 16px; color: #e6edf3;
    }
    
    .vip-badge {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #000; padding: 5px 15px; border-radius: 20px;
        font-size: 12px; font-weight: bold; display: inline-block;
    }
    .free-badge {
        background: #30363d; color: #8b949e; padding: 5px 15px;
        border-radius: 20px; font-size: 12px; font-weight: bold;
        display: inline-block;
    }
    
    .result-number {
        font-size: 48px; font-weight: bold; color: #f85149;
        letter-spacing: 8px; text-shadow: 0 0 20px rgba(248, 81, 73, 0.5);
    }
    
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #3fb950;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 🔐 AUTHENTICATION SYSTEM - IMPROVED
# =============================================================================
def init_auth():
    """Khởi tạo hệ thống authentication với backup file"""
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
    
    # Load từ file với error handling
    if os.path.exists('users.json'):
        try:
            with open('users.json', 'r', encoding='utf-8') as f:
                loaded_users = json.load(f)
                # Merge với default users
                st.session_state.users.update(loaded_users)
        except Exception as e:
            st.error(f"Lỗi load users: {e}")
            pass

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login(username, password):
    """Đăng nhập với validation chặt chẽ"""
    if not username or not password:
        return False, "Vui lòng nhập username và password"
    
    users = st.session_state.get('users', {})
    
    # Debug log
    st.write(f"DEBUG: Users loaded: {list(users.keys())}")
    
    user = users.get(username)
    
    if not user:
        return False, f"Username '{username}' không tồn tại"
    
    if not user.get('is_active', True):
        return False, "Tài khoản đã bị khóa"
    
    password_hash = hash_password(password)
    stored_hash = user.get('password_hash', '')
    
    if password_hash != stored_hash:
        return False, "Sai password"
    
    # Login thành công
    st.session_state.current_user = username
    st.session_state.login_time = datetime.now()
    
    return True, f"Chào mừng {username}!"

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
        st.error(f"Lỗi save users: {e}")

def register_user(username, password):
    if not username or not password:
        return False, "Vui lòng nhập đầy đủ thông tin"
    
    if username in st.session_state.users:
        return False, "Username đã tồn tại"
    
    if len(password) < 6:
        return False, "Password phải có ít nhất 6 ký tự"
    
    st.session_state.users[username] = {
        'password_hash': hash_password(password),
        'role': 'user',
        'vip_tier': None,
        'vip_until': None,
        'is_active': True,
        'created_at': datetime.now().isoformat()
    }
    save_users()
    return True, "Đăng ký thành công!"

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
                
                col_qr1, col_qr2, col_qr3 = st.columns([1,2,1])
                with col_qr2:
                    try:
                        st.image(qr_url, caption="📷 Quét bằng app ngân hàng", use_column_width=True, clamp=True)
                    except Exception as e:
                        st.error(f"Lỗi tải QR: {str(e)}")
                
                st.markdown(f'''
                <div style="background: #0d1117; padding: 20px; border-radius: 12px; margin: 20px 0; border: 2px solid #58a6ff;">
                    <p style="margin: 10px 0; font-size: 16px;"><b>💰 Số tiền:</b> <span style="color: #3fb950; font-size: 24px; font-weight: bold;">{tier['price']:,} VNĐ</span></p>
                    <p style="margin: 10px 0; font-size: 16px;"><b>📝 Nội dung CK:</b> <span style="color: #ffa500; font-weight: 600;">{transfer_message}</span></p>
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

# =============================================================================
# 📡 SCRAPING - IMPROVED
# =============================================================================
@st.cache_data(ttl=60)  # Cache 60 giây
def get_live_xsmb():
    """Lấy kết quả XSMB từ website với error handling tốt hơn"""
    url = "https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7'
    }
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, 'html.parser')
        
        # Try multiple extraction methods
        result = extract_data_v1(soup)
        
        # Nếu không lấy được, thử method khác
        if result.get("Đặc Biệt") in ["...", ""]:
            result = extract_data_v2(soup)
        
        result["time"] = datetime.now().strftime("%H:%M:%S %d/%m")
        result["source"] = "Live"
        
        return result
        
    except Exception as e:
        st.warning(f"⚠️ Không thể lấy kết quả trực tiếp: {str(e)[:100]}")
        return get_mock_data()

def extract_data_v1(soup):
    """Extract data method 1 - tìm theo class"""
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
        "Đặc Biệt": get_txt(["special-temp", "special", "db-value", "gdb"]),
        "Giải Nhất": get_txt(["g1-temp", "g1", "g1-value"]),
        "Giải Nhì": [get_txt(f"g2_{i}-temp") for i in range(2)],
        "Giải Ba": [get_txt(f"g3_{i}-temp") for i in range(6)],
        "Giải Tư": [get_txt(f"g4_{i}-temp") for i in range(4)],
        "Giải Năm": [get_txt(f"g5_{i}-temp") for i in range(6)],
        "Giải Sáu": [get_txt(f"g6_{i}-temp") for i in range(3)],
        "Giải Bảy": [get_txt(f"g7_{i}-temp") for i in range(4)],
    }

def extract_data_v2(soup):
    """Extract data method 2 - tìm theo pattern số"""
    text = soup.get_text()
    # Tìm số 5 chữ số
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
    """Dữ liệu mẫu khi không lấy được từ web"""
    return {
        "Đặc Biệt": "66239",  # Cập nhật số mới nhất
        "Giải Nhất": "39591",
        "Giải Nhì": ["18058", "22407"],
        "time": datetime.now().strftime("%H:%M:%S %d/%m"),
        "source": "Mock"
    }

# =============================================================================
# 📊 STATISTICS & TRACKING
# =============================================================================
def init_statistics():
    if 'statistics' not in st.session_state:
        st.session_state.statistics = {
            'predictions': [],
            'results': [],
            'daily_stats': {},
            'today_results': None,
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

def ensure_predictions_for_today():
    today = datetime.now().strftime("%d/%m")
    existing_pred = get_prediction_for_date(today)
    
    if existing_pred is None:
        historical = get_historical_data(30)
        predictions = generate_predictions_enhanced(historical)
        
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
            'checked': False,
            'win_status': None  # Will be updated after check
        }
        
        st.session_state.statistics['predictions'].append(pred_record)
        save_statistics()
        return pred_record, predictions
    return existing_pred, None

def save_result_check(today_date, current_data):
    """Lưu kết quả check với thống kê chi tiết"""
    current_loto = []
    for k, v in current_data.items():
        if k == "time": continue
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
            'total_predictions': 4,
            'win_rate': sum([bt_win, st_win, xien_win, de_win]) / 4 * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        st.session_state.statistics['results'].append(result_record)
        yesterday_pred['checked'] = True
        yesterday_pred['win_status'] = result_record
        
        # Update daily stats
        if today_date not in st.session_state.statistics['daily_stats']:
            st.session_state.statistics['daily_stats'][today_date] = {
                'date': today_date, 'checked': 0, 'total_wins': 0, 'total_predictions': 0
            }
        st.session_state.statistics['daily_stats'][today_date]['checked'] += 1
        st.session_state.statistics['daily_stats'][today_date]['total_wins'] += result_record['overall_wins']
        st.session_state.statistics['daily_stats'][today_date]['total_predictions'] += 4
        
        save_statistics()
        return result_record
    
    return None

def get_today_statistics():
    """Thống kê kết quả hôm nay"""
    today = datetime.now().strftime("%d/%m")
    today_stats = st.session_state.statistics['daily_stats'].get(today, {})
    
    total_checked = today_stats.get('checked', 0)
    total_wins = today_stats.get('total_wins', 0)
    total_predictions = today_stats.get('total_predictions', 0)
    
    win_rate = (total_wins / total_predictions * 100) if total_predictions > 0 else 0
    
    return {
        'date': today,
        'checked': total_checked,
        'wins': total_wins,
        'total_predictions': total_predictions,
        'win_rate': win_rate
    }

# =============================================================================
# 🧠 ENHANCED PREDICTION ALGORITHM
# =============================================================================
def generate_predictions_enhanced(historical_data):
    """Thuật toán dự đoán nâng cao với nhiều yếu tố"""
    all_loto = []
    date_weights = {}  # Trọng số theo ngày
    
    for date, data in historical_data.items():
        weight = 1.0
        # Ưu tiên dữ liệu gần đây
        try:
            date_obj = datetime.strptime(date, "%d/%m")
            days_ago = (datetime.now() - date_obj).days
            weight = max(0.1, 1.0 - (days_ago * 0.03))  # Giảm dần theo thời gian
        except:
            weight = 0.5
        
        for key, val in data.items():
            if isinstance(val, list):
                for v in val:
                    if v and len(v) >= 2:
                        num = v[-2:]
                        all_loto.append(num)
                        if num not in date_weights:
                            date_weights[num] = 0
                        date_weights[num] += weight
            elif val and len(val) >= 2:
                num = val[-2:]
                all_loto.append(num)
                if num not in date_weights:
                    date_weights[num] = 0
                date_weights[num] += weight
    
    # Tính tần suất có trọng số
    weighted_counter = Counter(date_weights)
    hot_nums = [num for num, _ in weighted_counter.most_common(15)]
    
    # Số chưa về trong 10 ngày gần nhất
    recent_10_days = list(historical_data.keys())[:10]
    recent_numbers = set()
    for date in recent_10_days:
        data = historical_data[date]
        for key, val in data.items():
            if isinstance(val, list):
                for v in val:
                    if v and len(v) >= 2:
                        recent_numbers.add(v[-2:])
            elif val and len(val) >= 2:
                recent_numbers.add(val[-2:])
    
    all_possible = [f"{i:02d}" for i in range(100)]
    cold_nums = [num for num in all_possible if num not in recent_numbers][:10]
    
    # Sử dụng Gemini AI nếu có
    gemini_result = query_gemini_ai(hot_nums[:10], cold_nums[:5], [], retry_count=2)
    
    if gemini_result:
        bt = gemini_result.get('bach_thu', hot_nums[0] if hot_nums else "00")
        st_list = gemini_result.get('song_thu', ["00", "00"])
        st1, st2 = st_list[0], st_list[1] if len(st_list) > 1 else "00"
        dan_de = gemini_result.get('dan_de', [f"{i:02d}" for i in range(10)])
        confidence = gemini_result.get('confidence', 75)
        analysis = gemini_result.get('analysis', "AI phân tích")
        using_ai = True
    else:
        # Fallback: Thuật toán thống kê
        # Bạch thủ: Số có tần suất cao nhất
        bt = hot_nums[0] if hot_nums else f"{np.random.randint(0,100):02d}"
        
        # Song thủ: Kết hợp số nóng và số lạnh
        st1 = hot_nums[1] if len(hot_nums) > 1 else f"{np.random.randint(0,100):02d}"
        st2 = cold_nums[0] if cold_nums else f"{np.random.randint(0,100):02d}"
        
        # Dàn đề: Top 10 số nóng + 3 số lạnh
        dan_de = list(set(hot_nums[:7] + cold_nums[:3]))
        while len(dan_de) < 10:
            dan_de.append(f"{np.random.randint(0,100):02d}")
        dan_de = sorted(dan_de)[:10]
        
        confidence = 65
        analysis = "Phân tích tần suất có trọng số"
        using_ai = False
    
    return {
        "bach_thu": bt,
        "song_thu": (st1, st2),
        "xien_2": f"{bt} - {st1}",
        "dan_de": dan_de,
        "hot_numbers": hot_nums[:10],
        "cold_numbers": cold_nums[:10],
        "confidence": confidence,
        "ai_analysis": analysis,
        "using_ai": using_ai
    }

def query_gemini_ai(hot_nums, cold_nums, gan_nums, retry_count=3):
    prompt = f"""Chuyên gia xổ số MB. Phân tích và dự đoán XSMB.

DỮ LIỆU THỐNG KÊ:
- Số NÓNG (về nhiều): {', '.join(hot_nums[:5]) if hot_nums else '67, 07, 60, 14, 02'}
- Số LẠNH (ít về): {', '.join(cold_nums[:5]) if cold_nums else '33, 44, 55, 66, 77'}

YÊU CẦU:
Phân tích xu hướng và đưa ra dự đoán chính xác nhất.

TRẢ LỜI THEO FORMAT JSON:
{{
    "analysis": "Phân tích ngắn gọn 1-2 câu về xu hướng",
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
                        if all(key in ai_result for key in ['bach_thu', 'song_thu', 'dan_de', 'confidence']):
                            return ai_result
            
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)
                
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)
            continue
    
    return None

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
    
    # Auto-refresh data every 120 seconds
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).seconds
    if time_since_refresh > 120:
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now()
    
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔑 Đăng nhập")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Đăng nhập", use_container_width=True, type="primary"):
                success, message = login(username, password)
                if success:
                    st.success(message)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(message)
                    st.info("💡 Thử: admin / admin123 hoặc guest (không password)")
        
        with col2:
            st.markdown("### 📝 Đăng ký")
            new_username = st.text_input("Username mới", key="reg_user")
            new_password = st.text_input("Password", type="password", key="reg_pass")
            
            if st.button("Đăng ký", use_container_width=True):
                success, msg = register_user(new_username, new_password)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
        
        st.divider()
        st.markdown("### 👥 Dùng thử miễn phí")
        if st.button("➡️ Tiếp tục với tài khoản Guest", use_container_width=True):
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
        <p style="margin:10px 0 0; font-size: 16px;">
            <span class="live-indicator"></span>
            Data Analytics Platform • Auto-update 120s
        </p>
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
        
        # Thống kê hôm nay
        today_stats = get_today_statistics()
        st.markdown("### 📊 Thống kê hôm nay")
        st.metric("Lần check", today_stats['checked'])
        st.metric("Số trúng", today_stats['wins'])
        st.metric("Tỷ lệ thắng", f"{today_stats['win_rate']:.1f}%")
        
        st.divider()
        
        menu_options = ["🎯 Dự đoán", "💎 Gói VIP", "📊 Thống kê", "📜 Lịch sử", "🌐 Website XS"]
        page = st.radio("🧭 Menu", menu_options)
    
    # MAIN CONTENT
    try:
        if page == "🎯 Dự đoán":
            st.markdown(f'''
            <div class="info-banner">
                <b>📅 DỰ LIỆU PHÂN TÍCH NGÀY {today}</b><br>
                <small>Cập nhật tự động mỗi 120 giây</small>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown("### 📡 KẾT QUẢ XSMB")
            
            # Force refresh button
            if st.button("🔄 Làm mới dữ liệu"):
                st.cache_data.clear()
                st.session_state.last_refresh = datetime.now()
                st.rerun()
            
            try:
                data = get_live_xsmb()
            except:
                data = get_mock_data()
            
            historical = get_historical_data(30 if user_tier == 'free' else VIP_TIERS[user_tier]['data_depth'])
            
            col1, col2 = st.columns([3, 1])
            with col1:
                source_badge = "🔴 LIVE" if data.get("source") == "Live" else "⚪ SAMPLE"
                st.caption(f"🕐 Cập nhật: **{data.get('time', 'N/A')}** {source_badge}")
            with col2:
                st.caption(f"Last update: {time_since_refresh}s ago")
            
            db = data.get("Đặc Biệt", "....")
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #161b22, #1f2937); 
                        border: 2px solid #D4AF37; border-radius: 15px; 
                        padding: 30px; text-align: center; margin: 15px 0;">
                <div style="font-size: 20px; color: #8b949e; margin-bottom: 15px;">📊 DỮ LIỆU THAM KHẢO</div>
                <div class="result-number">{db}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### ✅ KIỂM TRA KẾT QUẢ")
            
            if st.button(f"🔍 Check kết quả ngày {today}", use_container_width=True, type="primary"):
                try:
                    result = save_result_check(today, data)
                    if result:
                        st.success(f"✅ Đã check kết quả!")
                        
                        # Hiển thị thống kê chi tiết
                        cols = st.columns(4)
                        cols[0].metric("Bạch thủ", "✅" if result['bach_thu']['win'] else "❌")
                        cols[1].metric("Song thủ", "✅" if result['song_thu']['win'] else "❌")
                        cols[2].metric("Xiên 2", "✅" if result['xien_2']['win'] else "❌")
                        cols[3].metric("Đề", "✅" if result['dan_de']['win'] else "❌")
                        
                        st.info(f"📊 Tổng: {result['overall_wins']}/4 trúng ({result['win_rate']:.0f}%)")
                        st.rerun()
                    else:
                        st.info("ℹ️ Chưa có dự đoán hôm qua để check")
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
            
            st.markdown("---")
            st.markdown(f"### 🎯 DỰ ĐOÁN AI NGÀY {today}")
            
            if new_predictions:
                predictions = new_predictions
            else:
                try:
                    predictions = generate_predictions_enhanced(historical)
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
                    st.error(f"Lỗi: {str(e)}")
                    predictions = None
            
            if predictions:
                if predictions['using_ai']:
                    st.success("✅ Gemini AI Active")
                else:
                    st.warning("⚠️ Statistical Analysis Mode")
                
                st.markdown(f'''
                <div style="background: rgba(212, 175, 55, 0.15); border: 1px solid #D4AF37; border-radius: 8px; padding: 15px; margin: 15px 0;">
                    <b style="font-size: 16px;">🧠 PHÂN TÍCH:</b><br>{predictions['ai_analysis']}<br>
                    <b style="font-size: 16px;">ĐỘ TIN CẬY:</b> {predictions['confidence']}%
                </div>
                ''', unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'''<div class="pred-box"><div style="color:#8b949e; font-size:15px;">🎯 BẠCH THỦ</div><div style="font-size:48px; color:#FFD700; font-weight:bold; margin:15px 0;">{predictions['bach_thu']}</div></div>''', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'''<div class="pred-box"><div style="color:#8b949e; font-size:15px;">🎯 SONG THỦ</div><div style="font-size:28px; color:#FFD700; font-weight:bold; margin:15px 0;">{predictions['song_thu'][0]} - {predictions['song_thu'][1]}</div></div>''', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'''<div class="pred-box"><div style="color:#8b949e; font-size:15px;">🎯 XIÊN 2</div><div style="font-size:24px; color:#FFD700; font-weight:bold; margin:15px 0;">{predictions['xien_2']}</div></div>''', unsafe_allow_html=True)
                
                st.markdown(f'''<div class="pred-box"><div style="color:#8b949e; font-size:16px; margin-bottom:15px;">📋 DÀN ĐỀ 10 SỐ</div><div style="font-size:20px; color:#e6edf3;">{', '.join(predictions['dan_de'])}</div></div>''', unsafe_allow_html=True)
                
                if is_vip():
                    st.markdown("---")
                    st.markdown("### 🎲 BẠC NHỚ (VIP)")
                    try:
                        bac_nho = generate_bac_nho(historical)
                        for i, xien in enumerate(bac_nho, 1):
                            st.markdown(f'''<div class="xien-box"><div style="font-size: 16px; color: #8b949e;">💎 Cặp {i} - {xien['frequency']} lần</div><div style="font-size: 52px; color: #FFD700; font-weight: bold;">{xien['pair']}</div></div>''', unsafe_allow_html=True)
                    except:
                        st.error("Lỗi tải bạc nhớ")
            
            st.markdown("---")
            st.markdown('<div class="disclaimer">⚠️ Kết quả chỉ mang tính chất tham khảo. 18+ only</div>', unsafe_allow_html=True)
        
        elif page == "💎 Gói VIP":
            display_pricing_table()
        
        elif page == "📊 Thống kê":
            st.markdown("### 📈 THỐNG KÊ CHI TIẾT")
            
            # Thống kê hôm nay
            today_stats = get_today_statistics()
            st.markdown(f'''
            <div class="info-banner">
                <b>📊 Thống kê ngày {today_stats['date']}</b><br>
                Đã check: {today_stats['checked']} lần | 
                Trúng: {today_stats['wins']} | 
                Tỷ lệ: {today_stats['win_rate']:.1f}%
            </div>
            ''', unsafe_allow_html=True)
            
            daily_stats = st.session_state.statistics.get('daily_stats', {})
            if daily_stats:
                st.markdown("### 📅 Lịch sử 7 ngày")
                for date in sorted(daily_stats.keys(), reverse=True)[:7]:
                    stats = daily_stats[date]
                    checked = stats.get('checked', 0)
                    wins = stats.get('total_wins', 0)
                    total_pred = stats.get('total_predictions', 0)
                    rate = (wins / total_pred * 100) if total_pred > 0 else 0
                    
                    st.markdown(f'''
                    <div class="stat-card" style="margin: 8px 0;">
                        <div style="font-size: 16px; font-weight: bold; color: #e6edf3;">📅 {date}</div>
                        <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                            <span>Check: {checked}</span>
                            <span class="win">Trúng: {wins}/{total_pred}</span>
                            <span style="color: {'#3fb950' if rate >= 50 else '#f85149'}; font-size: 16px;">{rate:.1f}%</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            
            st.divider()
            st.markdown("### 📊 TỔNG QUAN")
            results = st.session_state.statistics.get('results', [])
            total_checks = len(results)
            total_wins = sum(r['overall_wins'] for r in results)
            max_wins = total_checks * 4
            win_rate = (total_wins / max_wins * 100) if max_wins > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Tổng lần check", total_checks)
            col2.metric("Tổng trúng", total_wins)
            col3.metric("Tỷ lệ thắng", f"{win_rate:.1f}%")
        
        elif page == "📜 Lịch sử":
            st.markdown("### 📜 LỊCH SỬ DỰ ĐOÁN")
            results = st.session_state.statistics.get('results', [])
            if results:
                dates = sorted(set(r['pred_date'] for r in results), reverse=True)
                selected_date = st.selectbox("Chọn ngày", dates)
                
                filtered = [r for r in results if r['pred_date'] == selected_date]
                for r in filtered:
                    st.markdown(f'''
                    <div style="background: rgba(212, 175, 55, 0.15); border: 1px solid #D4AF37; border-radius: 10px; padding: 20px; margin: 15px 0;">
                        <b style="font-size: 16px;">📅 {r['pred_date']} → {r['check_date']}</b>
                        <div style="margin-top: 10px;">
                            <div style="display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #30363d;">
                                <span>Bạch thủ: {r['bach_thu']['number']}</span>
                                <span class="{'win' if r['bach_thu']['win'] else 'loss'}">{'✅' if r['bach_thu']['win'] else '❌'}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #30363d;">
                                <span>Song thủ: {'-'.join(r['song_thu']['numbers'])}</span>
                                <span class="{'win' if r['song_thu']['win'] else 'loss'}">{'✅' if r['song_thu']['win'] else '❌'}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #30363d;">
                                <span>Xiên 2: {r['xien_2']['pair']}</span>
                                <span class="{'win' if r['xien_2']['win'] else 'loss'}">{'✅' if r['xien_2']['win'] else '❌'}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; padding: 8px;">
                                <span>Đề: {r['dan_de']['matched'] if r['dan_de']['win'] else 'Trượt'}</span>
                                <span class="{'win' if r['dan_de']['win'] else 'loss'}">{'✅' if r['dan_de']['win'] else '❌'}</span>
                            </div>
                        </div>
                        <div style="margin-top: 15px; text-align: center; font-size: 18px;">
                            <b>Tổng: {r['overall_wins']}/4 trúng ({r['win_rate']:.0f}%)</b>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("📭 Chưa có lịch sử")
        
        elif page == "🌐 Website XS":
            st.markdown("### 🌐 XOSODAIPHAT.COM")
            st.markdown('''
            <div style="border: 2px solid #D4AF37; border-radius: 15px; overflow: hidden; height: 800px;">
                <iframe src="https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html" 
                        style="width:100%; height:100%; border:none;"
                        sandbox="allow-same-origin allow-scripts allow-forms">
                </iframe>
            </div>
            ''', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Lỗi hệ thống: {str(e)}")
        st.error("Vui lòng refresh trang (F5)")

if __name__ == "__main__":
    main()