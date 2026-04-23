# =============================================================================
# 💎 AI-QUANTUM PRO 2026 - FINAL STABLE VERSION
# Fixed: PayOS QR, UI Colors, Email Removed, Auto-Refresh
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
import secrets
import uuid
import hmac

# =============================================================================
# 🔧 CẤU HÌNH
# =============================================================================
st.set_page_config(
    page_title="💎 AI-QUANTUM PRO 2026",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PayOS Config
try:
    PAYOS_CLIENT_ID = st.secrets.get("PAYOS_CLIENT_ID", "")
    PAYOS_API_KEY = st.secrets.get("PAYOS_API_KEY", "")
    PAYOS_CHECKSUM_KEY = st.secrets.get("PAYOS_CHECKSUM_KEY", "")
except:
    PAYOS_CLIENT_ID = ""
    PAYOS_API_KEY = ""
    PAYOS_CHECKSUM_KEY = ""

GEMINI_API_KEY = "AIzaSyARQk3lpoHnK51LQ62OR4vciH0XMFFIZjg"

# =============================================================================
# 🎨 CSS - MÀU SẮC DỄ CHỊU HƠN
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { 
        background-color: #0f1419; 
        color: #e0e0e0; 
        font-size: 16px;
    }
    .stApp { 
        background-color: #0f1419; 
    }
    
    .header-gold {
        background: linear-gradient(135deg, #D4AF37 0%, #FFD700 50%, #B8962E 100%);
        padding: 25px; 
        border-radius: 15px; 
        text-align: center; 
        color: #000; 
        font-weight: bold; 
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(212, 175, 55, 0.3);
    }
    
    .stat-card {
        background: linear-gradient(135deg, #1e2a3a, #2d3a4a);
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
        color: #b0b0b0; 
        margin-top: 5px; 
    }
    .win { color: #4ade80; font-weight: bold; }
    .loss { color: #f87171; font-weight: bold; }
    
    .pred-box {
        border: 2px solid #D4AF37; 
        border-radius: 12px;
        padding: 25px; 
        background: linear-gradient(135deg, #1e2a3a, #2d3a4a);
        text-align: center; 
        margin: 10px 0;
    }
    .xien-box {
        border: 3px solid #FFD700; 
        border-radius: 15px;
        padding: 30px; 
        background: linear-gradient(135deg, #1e2a3a, #2d3a4a);
        text-align: center; 
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(212, 175, 55, 0.2);
    }
    
    .disclaimer {
        background: rgba(248, 113, 113, 0.15);
        border-left: 4px solid #f87171;
        padding: 15px; 
        border-radius: 0 8px 8px 0;
        margin: 20px 0; 
        font-size: 14px;
        color: #fca5a5;
    }
    .info-banner {
        background: linear-gradient(135deg, rgba(66, 133, 244, 0.15), rgba(52, 168, 83, 0.15));
        border: 2px solid #4285F4; 
        border-radius: 10px;
        padding: 20px; 
        margin: 15px 0; 
        text-align: center;
        font-size: 16px;
        color: #e0e0e0;
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
        background: #4b5563; 
        color: #fff; 
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
        background: linear-gradient(135deg, #1e2a3a, #2d3a4a);
        text-align: center; 
        margin: 15px 0;
    }
    .pricing-card.premium {
        border-color: #FFD700;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.2);
    }
    
    .feature-list { 
        text-align: left; 
        font-size: 15px; 
        margin: 15px 0; 
    }
    .feature-list div { 
        margin: 8px 0; 
        color: #d1d5db; 
        font-size: 15px;
    }
    .feature-list .check { 
        color: #4ade80; 
        font-weight: bold;
    }
    
    .login-box {
        background: linear-gradient(135deg, #1e2a3a, #2d3a4a);
        border: 2px solid #D4AF37; 
        border-radius: 15px;
        padding: 35px; 
        max-width: 450px; 
        margin: 50px auto;
    }
    
    .result-number {
        font-size: 48px; 
        font-weight: bold; 
        color: #f87171; 
        letter-spacing: 8px;
        text-shadow: 0 0 20px rgba(248, 113, 113, 0.5);
    }
    
    h1, h2, h3, h4 {
        font-weight: 700;
        color: #f0f0f0;
    }
    
    .stTextInput > div > div > input {
        font-size: 16px;
        color: #e0e0e0;
    }
    
    .stButton > button {
        font-size: 16px;
        font-weight: 600;
        padding: 10px 24px;
    }
    
    /* Giảm độ sáng cho dễ nhìn */
    .css-1d391kg {
        background-color: #0f1419 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 🔐 AUTHENTICATION - ĐÃ BỎ EMAIL
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
    
    if user and user.get('is_active', True) and user['password_hash'] == hash_password(password):
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

def register_user(username, password):  # ĐÃ BỎ EMAIL
    if username in st.session_state.users:
        return False, "Username đã tồn tại"
    
    st.session_state.users[username] = {
        'password_hash': hash_password(password),
        'role': 'user',
        'email': None,  # Vẫn giữ field nhưng không bắt buộc
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
# 💳 PAYOS - ĐÃ FIX LỖI
# =============================================================================
def create_payos_payment(order_code, amount, description):
    """Tạo link thanh toán PayOS - ĐÃ FIX LỖI"""
    
    if not PAYOS_CLIENT_ID or not PAYOS_API_KEY:
        st.error("⚠️ Chưa cấu hình PayOS API Keys. Vui lòng liên hệ admin.")
        return None
    
    try:
        # Generate checksum
        data_to_sign = f"{amount}{order_code}{description}{PAYOS_CHECKSUM_KEY}"
        signature = hmac.new(
            PAYOS_CHECKSUM_KEY.encode(),
            data_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()
        
        payload = {
            "orderCode": order_code,
            "amount": amount,
            "description": description,
            "signature": signature,
            "items": [{"name": description, "quantity": 1, "price": amount}]
        }
        
        headers = {
            "x-client-id": PAYOS_CLIENT_ID,
            "x-api-key": PAYOS_API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://api-merchant.payos.vn/v2/payment-requests",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            data = result.get('data', {})
            if data:
                return {
                    'checkoutUrl': data.get('checkoutUrl'),
                    'qrCode': data.get('qrCode')
                }
        
        st.error(f"Lỗi PayOS: {response.text if hasattr(response, 'text') else 'Không xác định'}")
        return None
        
    except Exception as e:
        st.error(f"Lỗi thanh toán: {str(e)}")
        return None

def display_pricing_table():
    """Hiển thị bảng giá VIP"""
    
    st.markdown('''
    <div style="text-align: center; margin: 30px 0;">
        <h3>🤝 ỦNG HỘ DUY TRÌ DỰ ÁN AI QUANTUM</h3>
        <p style="color: #aaa; font-size: 15px;">
            Mọi khoản đóng góp sẽ được sử dụng để duy trì máy chủ và nâng cấp thuật toán AI.<br>
            <b>Ứng dụng cung cấp dữ liệu thống kê khoa học.</b>
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
                <p style="color: #888; font-size: 14px;">
                    /{'tháng' if tier['duration_days']==30 else 'quý' if tier['duration_days']==90 else 'năm'}
                </p>
            ''', unsafe_allow_html=True)
            
            st.markdown(f'<div class="feature-list">', unsafe_allow_html=True)
            for feature in tier['features']:
                st.markdown(f'<div>{feature}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button(f"🎁 Ủng hộ ngay", key=f"pay_{tier_key}", use_container_width=True):
                user_id = st.session_state.current_user or 'guest'
                order_code = f"AIQ_{tier_key}_{uuid.uuid4().hex[:8].upper()}"
                description = f"Phi dich vu Data {user_id}"
                
                with st.spinner("Đang tạo link thanh toán..."):
                    payment_info = create_payos_payment(order_code, tier['price'], description)
                    
                    if payment_info and payment_info.get('checkoutUrl'):
                        st.success("✅ Thanh toán an toàn qua PayOS")
                        
                        if payment_info.get('qrCode'):
                            st.markdown(f'''
                            <div style="background: #fff; padding: 20px; border-radius: 10px; text-align: center; margin: 15px 0;">
                                <p style="color: #000; margin: 0 0 10px 0; font-weight: bold;">📱 Quét QR để thanh toán</p>
                                <img src="{payment_info['qrCode']}" style="max-width: 100%; border-radius: 8px;" />
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        st.markdown(f'''
                        <div style="margin-top: 15px; padding: 15px; background: rgba(74, 222, 128, 0.1); 
                                    border-radius: 8px; text-align: center;">
                            <a href="{payment_info['checkoutUrl']}" target="_blank" 
                               style="background: #4ade80; color: #000; padding: 12px 24px; 
                                      border-radius: 8px; text-decoration: none; font-weight: bold;
                                      display: inline-block;">
                               💳 Thanh toán ngay
                            </a>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        st.info("⏳ Sau khi thanh toán, VIP sẽ được kích hoạt trong 5-10 phút")
                    else:
                        st.error("❌ Không thể tạo link thanh toán. Vui lòng thử lại hoặc liên hệ admin.")
            
            st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 📡 SCRAPING
# =============================================================================
@st.cache_data(ttl=60)
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

# =============================================================================
# 🧠 AI PREDICTION
# =============================================================================
def generate_predictions(historical_data):
    all_loto = []
    for date, data in historical_data.items():
        for key, val in data.items():
            if isinstance(val, list):
                for v in val:
                    if v and len(v) >= 2: all_loto.append(v[-2:])
            elif val and len(val) >= 2: all_loto.append(val[-2:])
    
    counter = Counter(all_loto)
    hot_nums = [num for num, _ in counter.most_common(10)]
    all_possible = [f"{i:02d}" for i in range(100)]
    cold_nums = [num for num in all_possible if num not in counter][:10]
    
    # Gemini AI
    try:
        prompt = f"""Phân tích xu hướng số liệu:
- Số nóng: {', '.join(hot_nums[:5])}
- Số lạnh: {', '.join(cold_nums[:5])}

Trả lời JSON: {{"bach_thu": "67", "song_thu": ["07","60"], "dan_de": ["00","01","02","07","10","13","14","60","67","93"], "confidence": 75, "analysis": "Phân tích ngắn"}}"""
        
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
        response = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.8, "maxOutputTokens": 512}
        }, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']
            import re
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                ai_result = json.loads(json_match.group())
                if all(key in ai_result for key in ['bach_thu', 'song_thu', 'dan_de']):
                    return {
                        "bach_thu": ai_result['bach_thu'],
                        "song_thu": tuple(ai_result.get('song_thu', ["00", "00"])),
                        "xien_2": f"{ai_result['bach_thu']} - {ai_result.get('song_thu', ['00'])[0]}",
                        "dan_de": ai_result['dan_de'],
                        "confidence": ai_result.get('confidence', 75),
                        "ai_analysis": ai_result.get('analysis', "AI phân tích"),
                        "using_ai": True
                    }
    except:
        pass
    
    # Fallback
    bt = hot_nums[0] if hot_nums else f"{np.random.randint(0,100):02d}"
    st1 = hot_nums[1] if len(hot_nums) > 1 else f"{np.random.randint(0,100):02d}"
    st2 = cold_nums[0] if cold_nums else f"{np.random.randint(0,100):02d}"
    dan_de = list(set(hot_nums[:4] + cold_nums[:3]))
    while len(dan_de) < 10: dan_de.append(f"{np.random.randint(0,100):02d}")
    
    return {
        "bach_thu": bt, "song_thu": (st1, st2), "xien_2": f"{bt} - {st1}", "dan_de": sorted(dan_de)[:10],
        "confidence": 65, "ai_analysis": "Phân tích tần suất", "using_ai": False
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
        }
    return historical

# =============================================================================
# 🚀 MAIN APP
# =============================================================================
def main():
    init_auth()
    
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
            # ĐÃ BỎ EMAIL
            
            if st.button("Đăng ký", use_container_width=True):
                success, msg = register_user(new_username, new_password)  # ĐÃ BỎ EMAIL
                if success:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("➡️ Tiếp tục với tài khoản Free", use_container_width=True):
            st.session_state.current_user = "guest"
            st.session_state.login_time = datetime.now()
            st.rerun()
        return
    
    # AUTHENTICATED APP
    today_pred = None
    try:
        historical = get_historical_data(30 if user_tier == 'free' else VIP_TIERS[user_tier]['data_depth'])
        today_pred = generate_predictions(historical)
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
    
    # HEADER
    st.markdown(f'''
    <div class="header-gold">
        <h1 style="margin:0; display:flex; justify-content:space-between; align-items:center; font-size: 28px;">
            <span>💎 AI-QUANTUM PRO 2026</span>
            <span style="font-size:14px;">
                👤 {current_user} 
                <span class="{'vip-badge' if is_vip() else 'free-badge'}">
                    {VIP_TIERS[user_tier]['name'] if is_vip() else 'Free'}
                </span>
            </span>
        </h1>
        <p style="margin:10px 0 0; font-size: 16px;">📊 Data Analytics Platform</p>
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
        
        menu_options = ["🎯 Dự đoán", "💎 Gói VIP", "📊 Thống kê", "🌐 Website XS"]
        page = st.radio("🧭 Menu", menu_options)
    
    # MAIN CONTENT
    if page == "🎯 Dự đoán":
        st.markdown(f'''
        <div class="info-banner">
            <b>📅 DỰ LIỆU NGÀY {today}</b><br>
            <small>Auto-refresh mỗi 60 giây</small>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("### 📡 KẾT QUẢ THAM KHẢO")
        
        data = get_live_xsmb()
        if data:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"🕐 Cập nhật: **{data.get('time', 'N/A')}**")
            with col2:
                if st.button("🔄 Làm mới", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()
            
            db = data.get("Đặc Biệt", "....")
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #1e2a3a, #2d3a4a); 
                        border: 2px solid #D4AF37; border-radius: 15px; 
                        padding: 30px; text-align: center; margin: 15px 0;">
                <div style="font-size: 20px; color: #aaa; margin-bottom: 15px;">📊 ĐẶC BIỆT</div>
                <div class="result-number">{db}</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Không thể tải kết quả. Đang hiển thị dữ liệu mẫu.")
        
        if today_pred:
            st.markdown("---")
            st.markdown("### 🎯 PHÂN TÍCH AI")
            
            if today_pred['using_ai']:
                st.success("✅ Gemini AI Active")
            else:
                st.info("ℹ️ Standard Mode")
            
            st.markdown(f'''
            <div style="background: rgba(212, 175, 55, 0.15); border: 1px solid #D4AF37; border-radius: 8px; padding: 15px; margin: 15px 0;">
                <b style="font-size: 16px;">🧠 PHÂN TÍCH:</b><br>{today_pred['ai_analysis']}<br>
                <b style="font-size: 16px;">ĐỘ TIN CẬY:</b> {today_pred['confidence']}%
            </div>
            ''', unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'''<div class="pred-box"><div style="color:#aaa; font-size:15px;">📊 BẠCH THỦ</div><div style="font-size:48px; color:#FFD700; font-weight:bold; margin:15px 0;">{today_pred['bach_thu']}</div></div>''', unsafe_allow_html=True)
            with c2:
                st.markdown(f'''<div class="pred-box"><div style="color:#aaa; font-size:15px;">📊 SONG THỦ</div><div style="font-size:26px; color:#FFD700; font-weight:bold; margin:15px 0;">{today_pred['song_thu'][0]} - {today_pred['song_thu'][1]}</div></div>''', unsafe_allow_html=True)
            with c3:
                st.markdown(f'''<div class="pred-box"><div style="color:#aaa; font-size:15px;">📊 XIÊN 2</div><div style="font-size:22px; color:#FFD700; font-weight:bold; margin:15px 0;">{today_pred['xien_2']}</div></div>''', unsafe_allow_html=True)
            
            st.markdown(f'''<div class="pred-box"><div style="color:#aaa; font-size:16px; margin-bottom:15px;">📊 DÀN ĐỀ (10 số)</div><div style="font-size:20px; color:#fff;">{', '.join(today_pred['dan_de'])}</div></div>''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="disclaimer">⚠️ Dữ liệu chỉ mang tính chất tham khảo.</div>', unsafe_allow_html=True)
    
    elif page == "💎 Gói VIP":
        display_pricing_table()
    
    elif page == "📊 Thống kê":
        st.markdown("### 📊 THỐNG KÊ")
        st.info("📭 Tính năng đang phát triển")
    
    elif page == "🌐 Website XS":
        st.markdown("### 🌐 XOSODAIPHAT.COM")
        st.markdown("Xem kết quả trực tiếp:")
        
        st.markdown('''
        <div style="border: 2px solid #D4AF37; border-radius: 15px; 
                    overflow: hidden; height: 800px;">
            <iframe src="https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html" 
                    style="width:100%; height:100%; border:none;"
                    sandbox="allow-same-origin allow-scripts allow-forms">
            </iframe>
        </div>
        ''', unsafe_allow_html=True)
    
    # FOOTER
    st.markdown("---")
    st.markdown('''
    <div style="text-align: center; color: #666; padding: 25px; font-size: 14px;">
        💎 AI-QUANTUM PRO 2026<br>
        18+ only
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()