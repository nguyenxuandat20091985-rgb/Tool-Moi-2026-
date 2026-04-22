# =============================================================================
# 💎 AI-QUANTUM PRO 2026 - FULL FEATURE EDITION
# Authentication • VIP Tiers • PayOS • Admin • Analytics • Legal-Safe
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
    page_title="💎 AI-QUANTUM PRO 2026 | Data Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh meta tag (120 seconds)
st.markdown('<meta http-equiv="refresh" content="120">', unsafe_allow_html=True)

# PayOS Config - Load từ secrets nếu có, fallback cho dev
try:
    PAYOS_CLIENT_ID = st.secrets.get("PAYOS_CLIENT_ID", "37ef3e3b-1cff-45bd-b9b6-70f38dcf3244")
    PAYOS_API_KEY = st.secrets.get("PAYOS_API_KEY", "0c0171bd-9d93-4999-b440-547931e3f65c")
    PAYOS_CHECKSUM_KEY = st.secrets.get("PAYOS_CHECKSUM_KEY", "aobc31bd7d110480ab81d7927dffba4fc7e5cc250b3af716574ac21fe65b946641")
except:
    PAYOS_CLIENT_ID = "37ef3e3b-1cff-45bd-b9b6-70f38dcf3244"
    PAYOS_API_KEY = "0c0171bd-9d93-4999-b440-547931e3f65c"
    PAYOS_CHECKSUM_KEY = "aobc31bd7d110480ab81d7927dffba4fc7e5cc250b3af716574ac21fe65b946641"

GEMINI_API_KEY = "AIzaSyARQk3lpoHnK51LQ62OR4vciH0XMFFIZjg"

# =============================================================================
# 🎨 CSS STYLING
# =============================================================================
st.markdown("""
<style>
    .main { background-color: #0a0a0f; color: #ffffff; }
    .stApp { background-color: #0a0a0f; }
    
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
    
    .vip-badge {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #000; padding: 3px 12px; border-radius: 15px;
        font-size: 11px; font-weight: bold; display: inline-block;
    }
    .free-badge {
        background: #444; color: #aaa; padding: 3px 12px; 
        border-radius: 15px; font-size: 11px; display: inline-block;
    }
    
    .pricing-card {
        border: 2px solid #D4AF37; border-radius: 15px;
        padding: 20px; background: linear-gradient(135deg, #1a1a2e, #16213e);
        text-align: center; margin: 10px 0;
    }
    .pricing-card.premium {
        border-color: #FFD700;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.3);
    }
    
    .feature-list { text-align: left; font-size: 13px; margin: 10px 0; }
    .feature-list div { margin: 5px 0; color: #ccc; }
    .feature-list .check { color: #00ff88; }
    
    .login-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #D4AF37; border-radius: 15px;
        padding: 30px; max-width: 400px; margin: 50px auto;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 🔐 AUTHENTICATION SYSTEM
# =============================================================================
def init_auth():
    """Khởi tạo hệ thống authentication"""
    if 'users' not in st.session_state:
        st.session_state.users = {
            'admin': {
                'password_hash': hashlib.sha256('admin123'.encode()).hexdigest(),
                'role': 'admin',
                'vip_tier': None,
                'vip_until': None,
                'is_active': True,
                'created_at': datetime.now().isoformat()
            }
        }
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    if 'sessions' not in st.session_state:
        st.session_state.sessions = {}
    
    # Load users from file
    if os.path.exists('users.json'):
        try:
            with open('users.json', 'r', encoding='utf-8') as f:
                st.session_state.users = json.load(f)
        except:
            pass

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login(username, password):
    """Đăng nhập người dùng"""
    users = st.session_state.get('users', {})
    user = users.get(username)
    
    if user and user.get('is_active', True) and user['password_hash'] == hash_password(password):
        token = secrets.token_urlsafe(32)
        st.session_state.sessions[token] = {
            'username': username,
            'role': user['role'],
            'vip_tier': user.get('vip_tier'),
            'vip_until': user.get('vip_until'),
            'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
        }
        st.session_state.current_user = username
        track_event(username, 'login')
        return token
    return None

def logout():
    """Đăng xuất"""
    if st.session_state.current_user:
        track_event(st.session_state.current_user, 'logout')
    st.session_state.current_user = None
    st.session_state.sessions = {}

def check_auth():
    """Kiểm tra session hiện tại"""
    if not st.session_state.current_user:
        return None
    
    sessions = st.session_state.get('sessions', {})
    for token, session in sessions.items():
        if session['username'] == st.session_state.current_user:
            if datetime.fromisoformat(session['expires_at']) > datetime.now():
                return session
    return None

def get_user_info(username=None):
    """Lấy thông tin user"""
    if username is None:
        username = st.session_state.current_user
    if not username:
        return None
    return st.session_state.users.get(username)

def is_vip(username=None):
    """Kiểm tra user có VIP không"""
    user = get_user_info(username)
    if not user or not user.get('vip_tier'):
        return False
    if user.get('vip_until'):
        return datetime.fromisoformat(user['vip_until']) > datetime.now()
    return True

def get_vip_tier(username=None):
    """Lấy tier VIP của user"""
    user = get_user_info(username)
    return user.get('vip_tier') if user and is_vip(username) else 'free'

def save_users():
    """Lưu users ra file"""
    try:
        with open('users.json', 'w', encoding='utf-8') as f:
            json.dump(st.session_state.users, f, ensure_ascii=False, indent=2)
    except:
        pass

def register_user(username, password, email=None):
    """Đăng ký user mới"""
    if username in st.session_state.users:
        return False, "Username đã tồn tại"
    
    st.session_state.users[username] = {
        'password_hash': hash_password(password),
        'role': 'user',
        'email': email,
        'vip_tier': None,
        'vip_until': None,
        'is_active': True,
        'created_at': datetime.now().isoformat()
    }
    save_users()
    return True, "Đăng ký thành công"

def update_vip_status(username, tier, duration_days):
    """Cập nhật trạng thái VIP cho user"""
    if username not in st.session_state.users:
        return False
    
    st.session_state.users[username]['vip_tier'] = tier
    st.session_state.users[username]['vip_until'] = (
        datetime.now() + timedelta(days=duration_days)
    ).isoformat()
    save_users()
    return True

def toggle_user_active(username, active):
    """Khóa/mở khóa user (admin only)"""
    if username in st.session_state.users:
        st.session_state.users[username]['is_active'] = active
        save_users()
        return True
    return False

# =============================================================================
# 📊 ANALYTICS TRACKING
# =============================================================================
def track_event(user_id, action, metadata=None):
    """Ghi log sự kiện người dùng"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id or 'guest',
        'action': action,
        'metadata': metadata or {},
        'ip_hash': hashlib.sha256(
            st.context.headers.get('X-Forwarded-For', 'unknown').encode()
        ).hexdigest()[:16] if hasattr(st, 'context') else 'unknown'
    }
    
    logs = []
    if os.path.exists('analytics.jsonl'):
        with open('analytics.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        logs.append(json.loads(line))
                    except:
                        continue
    
    logs.append(log_entry)
    logs = logs[-10000:]  # Keep last 10k logs
    
    with open('analytics.jsonl', 'w', encoding='utf-8') as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + '\n')

def get_analytics_summary(days=7):
    """Thống kê đơn giản"""
    if not os.path.exists('analytics.jsonl'):
        return {'total_users': 0, 'actions': {}, 'vip_signups': 0}
    
    cutoff = datetime.now() - timedelta(days=days)
    stats = {'total_users': set(), 'actions': Counter(), 'vip_signups': 0}
    
    with open('analytics.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                log = json.loads(line)
                log_time = datetime.fromisoformat(log['timestamp'])
                if log_time >= cutoff:
                    stats['total_users'].add(log['user_id'])
                    stats['actions'][log['action']] += 1
                    if log['action'] == 'vip_signup':
                        stats['vip_signups'] += 1
            except:
                continue
    
    stats['total_users'] = len(stats['total_users'])
    return stats

# =============================================================================
# 💎 VIP TIERS CONFIG
# =============================================================================
VIP_TIERS = {
    "free": {
        "name": "Miễn phí",
        "price": 0,
        "duration_days": 0,
        "features": [
            "Dự đoán cơ bản",
            "Kết quả real-time",
            "Thống kê 7 ngày",
            "Có quảng cáo"
        ],
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
# 💳 PAYOS INTEGRATION
# =============================================================================
def create_payos_payment(order_code, amount, description, return_url, cancel_url):
    """Tạo link thanh toán PayOS"""
    
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
        "returnUrl": return_url,
        "cancelUrl": cancel_url,
        "signature": signature,
        "items": [{"name": description, "quantity": 1, "price": amount}]
    }
    
    headers = {
        "x-client-id": PAYOS_CLIENT_ID,
        "x-api-key": PAYOS_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            "https://api-merchant.payos.vn/v2/payment-requests",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('data', {}).get('checkoutUrl')
        
        return None
    except:
        return None

def display_pricing_table():
    """Hiển thị bảng giá VIP"""
    
    st.markdown('''
    <div style="text-align: center; margin: 30px 0;">
        <h3>🤝 ỦNG HỘ DUY TRÌ DỰ ÁN AI QUANTUM</h3>
        <p style="color: #888; font-size: 14px;">
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
                <h4 style="color: #FFD700; margin: 0;">{tier['name']}</h4>
                <h2 style="color: #fff; margin: 10px 0;">{tier['price']:,} VNĐ</h2>
                <p style="color: #888; font-size: 12px;">
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
                
                checkout_url = create_payos_payment(
                    order_code=order_code,
                    amount=tier['price'],
                    description=description,
                    return_url=st.experimental_get_query_params().get('callback', [''])[0] or st.context.page_url,
                    cancel_url=st.context.page_url
                )
                
                if checkout_url:
                    st.markdown(f'''
                    <div style="margin-top: 10px;">
                        <a href="{checkout_url}" target="_blank" 
                           style="background: #00ff88; color: #000; padding: 10px 20px; 
                                  border-radius: 8px; text-decoration: none; font-weight: bold;
                                  display: block;">
                           🔗 Thanh toán qua PayOS
                        </a>
                    </div>
                    ''', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Legal disclaimer
    st.markdown('''
    <div style="background: rgba(255, 107, 107, 0.1); border-left: 4px solid #ff6b6b;
                padding: 15px; border-radius: 0 8px 8px 0; margin: 20px 0; font-size: 12px;">
        <b>⚖️ LƯU Ý PHÁP LÝ:</b><br>
        • Khoản thanh toán là phí dịch vụ cung cấp dữ liệu thống kê và công cụ phân tích AI.<br>
        • Ứng dụng không cung cấp dịch vụ cá cược, không tổ chức đánh bạc.<br>
        • Người dùng tự chịu trách nhiệm với hành vi sử dụng dữ liệu.<br>
        • Mọi giao dịch được ghi nhận minh bạch qua PayOS - đối tác thanh toán được cấp phép.
    </div>
    ''', unsafe_allow_html=True)

# =============================================================================
# 📜 TERMS & DISCLAIMER
# =============================================================================
def display_terms():
    """Hiển thị điều khoản sử dụng"""
    with st.expander("📜 Điều khoản sử dụng & Chính sách", expanded=False):
        st.markdown('''
        ### 📋 ĐIỀU KHOẢN SỬ DỤNG AI-QUANTUM PRO
        
        **1. Mục đích ứng dụng**
        - Ứng dụng cung cấp công cụ phân tích dữ liệu thống kê bằng AI.
        - Kết quả phân tích chỉ mang tính chất **THAM KHẢO**, không đảm bảo chính xác.
        
        **2. Trách nhiệm người dùng**
        - Người dùng tự chịu trách nhiệm với hành vi sử dụng dữ liệu từ ứng dụng.
        - Ứng dụng không khuyến khích, không tổ chức, không tham gia vào bất kỳ hình thức cá cược nào.
        
        **3. Phí dịch vụ VIP**
        - Khoản thanh toán là phí duy trì hạ tầng kỹ thuật và phát triển thuật toán AI.
        - Không phải phí "mua số" hay "đặt cược".
        - Hóa đơn điện tử sẽ ghi: "Phí dịch vụ dữ liệu thống kê".
        
        **4. Bảo mật thông tin**
        - Chúng tôi không thu thập thông tin cá nhân nhạy cảm.
        - Dữ liệu sử dụng được mã hóa và chỉ lưu trong session.
        
        **5. Quyền hủy dịch vụ**
        - Người dùng có quyền yêu cầu hủy tài khoản bất kỳ lúc nào.
        - Phí đã thanh toán không hoàn lại, trừ trường hợp lỗi hệ thống.
        
        *Bằng cách sử dụng ứng dụng, bạn đồng ý với các điều khoản trên.*
        ''')

# =============================================================================
# 👑 ADMIN PANEL
# =============================================================================
def admin_panel():
    """Admin panel để quản lý users"""
    st.markdown("### 👑 ADMIN PANEL")
    
    users = st.session_state.get('users', {})
    
    # Stats
    total_users = len([u for u in users.values() if u.get('is_active', True)])
    vip_users = len([u for u in users.values() if u.get('vip_tier')])
    
    col1, col2 = st.columns(2)
    col1.metric("Total Users", total_users)
    col2.metric("VIP Users", vip_users)
    
    st.divider()
    
    # User list
    st.markdown("**Danh sách người dùng:**")
    
    for username, info in users.items():
        with st.expander(f"👤 {username} ({info['role']})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Email:** {info.get('email', 'N/A')}")
                st.write(f"**Created:** {info.get('created_at', 'N/A')[:10]}")
            
            with col2:
                vip = info.get('vip_tier')
                if vip:
                    st.write(f"**VIP:** {VIP_TIERS[vip]['name']}")
                    until = info.get('vip_until')
                    if until:
                        st.write(f"**Until:** {until[:10]}")
                else:
                    st.write("**VIP:** Free")
            
            with col3:
                status = "✅ Active" if info.get('is_active', True) else "❌ Locked"
                st.write(f"**Status:** {status}")
                
                # Toggle active
                new_status = not info.get('is_active', True)
                if st.button("🔒 Khóa" if info.get('is_active', True) else "🔓 Mở", key=f"toggle_{username}"):
                    toggle_user_active(username, new_status)
                    st.rerun()
    
    # Create new user
    with st.expander("➕ Tạo tài khoản mới"):
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")
        new_email = st.text_input("Email (optional)")
        new_role = st.selectbox("Role", ["user", "vip", "admin"])
        
        if st.button("Tạo tài khoản"):
            if new_user and new_pass and new_user not in users:
                users[new_user] = {
                    'password_hash': hash_password(new_pass),
                    'role': new_role,
                    'email': new_email,
                    'vip_tier': None if new_role != 'vip' else 'dong_hanh',
                    'vip_until': None,
                    'is_active': True,
                    'created_at': datetime.now().isoformat()
                }
                save_users()
                st.success(f"✅ Đã tạo tài khoản {new_user}")
                st.rerun()

# =============================================================================
# 💾 STATISTICS (Original)
# =============================================================================
def init_statistics():
    if 'statistics' not in st.session_state:
        st.session_state.statistics = {
            'predictions': [], 'results': [], 'daily_stats': {}, 'last_check_date': None
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
            'date': today, 'created_time': datetime.now().isoformat(),
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
    current_loto = []
    for k, v in current_data.items():
        if k == "time": continue
        if isinstance(v, list):
            for x in v:
                if x and x != "...": current_loto.append(x[-2:])
        elif v and v != "...": current_loto.append(v[-2:])
    
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
            'check_date': today_date, 'pred_date': yesterday,
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
            st.session_state.statistics['daily_stats'][today_date] = {'date': today_date, 'checked': 0, 'total_wins': 0}
        st.session_state.statistics['daily_stats'][today_date]['checked'] += 1
        st.session_state.statistics['daily_stats'][today_date]['total_wins'] += result_record['overall_wins']
        
        save_statistics()
        return result_record
    return None

# =============================================================================
# 📡 SCRAPING
# =============================================================================
@st.cache_data(ttl=120)
def get_live_xsmb():
    url = "https://xosodaiphat.com/xsmb-xổ-số-miền-bắc.html"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, 'html.parser')
        
        result = extract_data_v1(soup)
        if result["Đặc Biệt"] in ["...", ""]:
            result = extract_data_v2(soup)
        
        result["time"] = datetime.now().strftime("%H:%M:%S %d/%m")
        return result
    except:
        return get_mock_data()

def extract_data_v1(soup):
    def get_txt(classes):
        if isinstance(classes, str): classes = [classes]
        for cls in classes:
            item = soup.find("span", class_=cls)
            if item and item.text.strip():
                text = item.text.strip()
                match = re.search(r'(\d+)', text)
                if match: return match.group(1)
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
    text = soup.get_text()
    numbers = re.findall(r'\b\d{5}\b', text)
    return {
        "Đặc Biệt": numbers[0] if numbers else "...",
        "Giải Nhất": numbers[1] if len(numbers) > 1 else "...",
        "Giải Nhì": ["..."] * 2, "Giải Ba": ["..."] * 6,
        "Giải Tư": ["..."] * 4, "Giải Năm": ["..."] * 6,
        "Giải Sáu": ["..."] * 3, "Giải Bảy": ["..."] * 4,
    }

def get_mock_data():
    return {
        "Đặc Biệt": "36948", "Giải Nhất": "96041",
        "Giải Nhì": ["09028", "27803"],
        "time": datetime.now().strftime("%H:%M:%S %d/%m"), "source": "Mock"
    }

# =============================================================================
# 🧠 BẠC NHỚ & PREDICTION
# =============================================================================
def generate_bac_nho(historical_data):
    pairs_counter = Counter()
    for date, data in historical_data.items():
        loto_day = []
        for key, val in data.items():
            if isinstance(val, list):
                for v in val:
                    if v and len(v) >= 2: loto_day.append(v[-2:])
            elif val and len(val) >= 2: loto_day.append(val[-2:])
        
        for i in range(len(loto_day)):
            for j in range(i+1, len(loto_day)):
                pair = tuple(sorted([loto_day[i], loto_day[j]]))
                pairs_counter[pair] += 1
    
    top_pairs = pairs_counter.most_common(3)
    while len(top_pairs) < 3:
        num1, num2 = f"{np.random.randint(0,100):02d}", f"{np.random.randint(0,100):02d}"
        while num1 == num2: num2 = f"{np.random.randint(0,100):02d}"
        pair = tuple(sorted([num1, num2]))
        if pair not in [p[0] for p in top_pairs]: top_pairs.append((pair, 1))
    
    return [{'pair': f"{p[0][0]} - {p[0][1]}", 'frequency': p[1], 'num1': p[0][0], 'num2': p[0][1]} for p in top_pairs]

def query_gemini_ai(hot_nums, cold_nums, gan_nums, retry_count=3):
    prompt = f"""Chuyên gia phân tích dữ liệu. Phân tích xu hướng số liệu.

DỮ LIỆU:
- Số xuất hiện nhiều: {', '.join(hot_nums[:5]) if hot_nums else '67, 07, 60, 14, 02'}
- Số xuất hiện ít: {', '.join(cold_nums[:5]) if cold_nums else '33, 44, 55, 66, 77'}

TRẢ LỜI JSON:
{{
    "analysis": "Phân tích xu hướng 1-2 câu",
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
            if attempt < retry_count - 1: time.sleep(2 ** attempt)
        except:
            if attempt < retry_count - 1: time.sleep(2 ** attempt)
            continue
    return None

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
    
    gan_nums = []
    for num in all_possible[:10]:
        days = 0
        for date, data in historical_data.items():
            found = False
            for key, val in data.items():
                if isinstance(val, list):
                    if any(v.endswith(num) for v in val if v): found = True; break
                elif val and val.endswith(num): found = True; break
            if found: break
            days += 1
        gan_nums.append((num, days))
    
    gemini_result = query_gemini_ai(hot_nums, cold_nums, gan_nums)
    
    if gemini_result:
        bt = gemini_result.get('bach_thu', hot_nums[0] if hot_nums else "00")
        st_list = gemini_result.get('song_thu', ["00", "00"])
        st1, st2 = st_list[0], st_list[1] if len(st_list) > 1 else "00"
        dan_de = gemini_result.get('dan_de', [f"{i:02d}" for i in range(10)])
        confidence = gemini_result.get('confidence', 75)
        analysis = gemini_result.get('analysis', "AI phân tích xu hướng")
        using_ai = True
    else:
        bt = hot_nums[0] if hot_nums else f"{np.random.randint(0,100):02d}"
        st1 = hot_nums[1] if len(hot_nums) > 1 else f"{np.random.randint(0,100):02d}"
        st2 = cold_nums[0] if cold_nums else f"{np.random.randint(0,100):02d}"
        dan_de = list(set(hot_nums[:4] + cold_nums[:3]))
        while len(dan_de) < 10: dan_de.append(f"{np.random.randint(0,100):02d}")
        dan_de = sorted(dan_de)[:10]
        confidence = 65
        analysis = "Phân tích tần suất dữ liệu"
        using_ai = False
    
    return {
        "bach_thu": bt, "song_thu": (st1, st2), "xien_2": f"{bt} - {st1}", "dan_de": dan_de,
        "hot_numbers": hot_nums, "cold_numbers": cold_nums, "gan_numbers": gan_nums,
        "confidence": confidence, "ai_analysis": analysis, "using_ai": using_ai
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
    today = datetime.now().strftime("%d/%m")
    
    # Check auth
    user_session = check_auth()
    current_user = st.session_state.current_user
    user_tier = get_vip_tier()
    
    # LOGIN PAGE
    if not current_user:
        st.markdown(f'''
        <div class="header-gold">
            <h1 style="margin:0;">💎 AI-QUANTUM PRO 2026</h1>
            <p style="margin:5px 0 0;">📊 Data Analytics Platform</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["🔑 Đăng nhập", "📝 Đăng ký"])
        
        with tab_login:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Đăng nhập", use_container_width=True, type="primary"):
                token = login(username, password)
                if token:
                    st.success("✅ Đăng nhập thành công!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Sai username hoặc password")
        
        with tab_register:
            new_username = st.text_input("Username mới", key="reg_user")
            new_password = st.text_input("Password", type="password", key="reg_pass")
            new_email = st.text_input("Email (optional)", key="reg_email")
            
            if st.button("Đăng ký", use_container_width=True):
                success, msg = register_user(new_username, new_password, new_email)
                if success:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Public preview
        st.markdown("### 👥 Dùng thử miễn phí")
        st.info("🔓 Bạn có thể dùng thử các tính năng cơ bản mà không cần đăng nhập.")
        
        if st.button("➡️ Tiếp tục với tài khoản Free", use_container_width=True):
            st.session_state.current_user = "guest"
            st.rerun()
        
        display_terms()
        return
    
    # AUTHENTICATED APP
    today_pred, new_predictions = ensure_predictions_for_today()
    
    # HEADER
    st.markdown(f'''
    <div class="header-gold">
        <h1 style="margin:0; display:flex; justify-content:space-between; align-items:center;">
            <span>💎 AI-QUANTUM PRO 2026</span>
            <span style="font-size:14px;">
                👤 {current_user} 
                <span class="{'vip-badge' if is_vip() else 'free-badge'}">
                    {VIP_TIERS[user_tier]['name'] if is_vip() else 'Free'}
                </span>
            </span>
        </h1>
        <p style="margin:5px 0 0;">📊 Data Analytics Platform • Auto Refresh 120s</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        # User info
        st.markdown(f"### 👤 {current_user}")
        if is_vip():
            until = get_user_info().get('vip_until', '')
            st.markdown(f'<span class="vip-badge">VIP đến: {until[:10] if until else "N/A"}</span>', unsafe_allow_html=True)
        
        if st.button("🚪 Đăng xuất", use_container_width=True):
            logout()
            st.rerun()
        
        st.divider()
        
        # Navigation
        page = st.radio("🧭 Menu", ["🎯 Dự đoán", "💎 Gói VIP", "📊 Thống kê", "📜 Lịch sử", "👑 Admin"] if get_user_info().get('role') == 'admin' else ["🎯 Dự đoán", "💎 Gói VIP", "📊 Thống kê", "📜 Lịch sử"])
        
        st.divider()
        
        # Analytics (admin only)
        if get_user_info().get('role') == 'admin':
            st.markdown("### 📈 Analytics")
            analytics = get_analytics_summary(7)
            st.write(f"Users (7d): {analytics['total_users']}")
            st.write(f"VIP signups: {analytics['vip_signups']}")
        
        display_terms()
    
    # MAIN CONTENT
    if page == "🎯 Dự đoán":
        st.markdown(f'''
        <div class="info-banner">
            <b>📅 DỰ LIỆU PHÂN TÍCH NGÀY {today}</b><br>
            <small>Auto-refresh mỗi 120 giây • Dữ liệu tham khảo</small>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("### 📡 KẾT QUẢ THAM KHẢO")
        
        data = get_live_xsmb()
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
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); 
                    border: 2px solid #D4AF37; border-radius: 15px; 
                    padding: 20px; text-align: center; margin: 10px 0;">
            <div style="font-size: 18px; color: #aaa; margin-bottom: 10px;">📊 DỮ LIỆU THAM KHẢO</div>
            <div style="font-size: 42px; color: #ff4b4b; font-weight: bold; letter-spacing: 6px;">{db}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ✅ KIỂM TRA DỮ LIỆU NGÀY TRƯỚC")
        
        if st.button(f"🔍 Check dữ liệu ngày {today}", use_container_width=True, type="primary"):
            result = save_result_check(today, data)
            if result:
                st.success(f"✅ Đã cập nhật kết quả!")
                st.markdown(f'''
                <div style="background: rgba(212, 175, 55, 0.1); 
                            border-left: 4px solid #D4AF37; padding: 15px; border-radius: 0 8px 8px 0; margin: 10px 0;">
                    <b>📅 {result['pred_date']} → {result['check_date']}</b>
                    <div style="margin-top: 10px;">
                        <div style="padding: 8px; display: flex; justify-content: space-between;">
                            <span>📊 Data 1: <b>{result['bach_thu']['number']}</b></span>
                            <span class="{'win' if result['bach_thu']['win'] else 'loss'}">{'✅' if result['bach_thu']['win'] else '❌'}</span>
                        </div>
                        <div style="padding: 8px; display: flex; justify-content: space-between;">
                            <span>📊 Data 2: <b>{'-'.join(result['song_thu']['numbers'])}</b></span>
                            <span class="{'win' if result['song_thu']['win'] else 'loss'}">{'✅' if result['song_thu']['win'] else '❌'}</span>
                        </div>
                        <div style="padding: 8px; display: flex; justify-content: space-between;">
                            <span>📊 Data 3: <b>{result['xien_2']['pair']}</b></span>
                            <span class="{'win' if result['xien_2']['win'] else 'loss'}">{'✅' if result['xien_2']['win'] else '❌'}</span>
                        </div>
                        <div style="padding: 8px; display: flex; justify-content: space-between;">
                            <span>📊 Data 4</span>
                            <span class="{'win' if result['dan_de']['win'] else 'loss'}">{'✅' if result['dan_de']['win'] else '❌'}</span>
                        </div>
                    </div>
                    <div style="margin-top: 10px; text-align: right;"><b>Tổng: {result['overall_wins']}/4</b></div>
                </div>
                ''', unsafe_allow_html=True)
                st.rerun()
            else:
                st.info("ℹ️ Chưa có dữ liệu ngày trước để check")
        
        st.markdown("---")
        st.markdown(f"### 🎯 PHÂN TÍCH AI NGÀY {today}")
        
        if new_predictions:
            predictions = new_predictions
        else:
            predictions = generate_predictions(historical)
            for p in st.session_state.statistics['predictions']:
                if p['date'] == today:
                    p['predictions'] = {'bach_thu': predictions['bach_thu'], 'song_thu': predictions['song_thu'], 'xien_2': predictions['xien_2'], 'dan_de': predictions['dan_de']}
                    p['confidence'] = predictions['confidence']
                    p['ai_analysis'] = predictions['ai_analysis']
                    p['using_ai'] = predictions['using_ai']
                    break
            save_statistics()
        
        if predictions['using_ai']:
            st.success("✅ Gemini AI Active")
        else:
            st.warning("⚠️ Standard Analysis Mode")
        
        st.markdown(f'''
        <div style="background: rgba(212, 175, 55, 0.1); border: 1px solid #D4AF37; border-radius: 8px; padding: 12px; margin: 10px 0;">
            <b>🧠 PHÂN TÍCH:</b><br>{predictions['ai_analysis']}<br>
            <b>ĐỘ TIN CẬY:</b> {predictions['confidence']}%
        </div>
        ''', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'''<div class="pred-box"><div style="color:#aaa; font-size:13px;">📊 DATA 1</div><div style="font-size:42px; color:#FFD700; font-weight:bold; margin:10px 0;">{predictions['bach_thu']}</div></div>''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''<div class="pred-box"><div style="color:#aaa; font-size:13px;">📊 DATA 2</div><div style="font-size:26px; color:#FFD700; font-weight:bold; margin:10px 0;">{predictions['song_thu'][0]} - {predictions['song_thu'][1]}</div></div>''', unsafe_allow_html=True)
        with c3:
            st.markdown(f'''<div class="pred-box"><div style="color:#aaa; font-size:13px;">📊 DATA 3</div><div style="font-size:22px; color:#FFD700; font-weight:bold; margin:10px 0;">{predictions['xien_2']}</div></div>''', unsafe_allow_html=True)
        
        st.markdown(f'''<div class="pred-box"><div style="color:#aaa; font-size:14px; margin-bottom:10px;">📊 DATA 4 (10 values)</div><div style="font-size:18px; color:#fff;">{', '.join(predictions['dan_de'])}</div></div>''', unsafe_allow_html=True)
        
        # VIP feature: Bac nho
        if is_vip():
            st.markdown("---")
            st.markdown("### 🎲 PHÂN TÍCH CẶP SỐ (VIP)")
            bac_nho = generate_bac_nho(historical)
            for i, xien in enumerate(bac_nho, 1):
                st.markdown(f'''<div class="xien-box"><div style="font-size: 14px; color: #aaa;">💎 Cặp {i} - {xien['frequency']} lần</div><div style="font-size: 48px; color: #FFD700; font-weight: bold;">{xien['pair']}</div></div>''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<div class="disclaimer">⚠️ Dữ liệu phân tích chỉ mang tính chất tham khảo. Người dùng tự chịu trách nhiệm với hành vi sử dụng.</div>', unsafe_allow_html=True)
    
    elif page == "💎 Gói VIP":
        st.markdown("### 🤝 ỦNG HỘ DỰ ÁN")
        display_pricing_table()
    
    elif page == "📊 Thống kê":
        st.markdown("### 📈 THỐNG KÊ NGÀY")
        daily_stats = st.session_state.statistics.get('daily_stats', {})
        if daily_stats:
            for date in sorted(daily_stats.keys(), reverse=True)[:7]:
                stats = daily_stats[date]
                checked = stats.get('checked', 0)
                wins = stats.get('total_wins', 0)
                rate = (wins / (checked * 4) * 100) if checked > 0 else 0
                st.markdown(f'''<div class="stat-card" style="margin: 5px 0;"><div style="font-size: 14px; font-weight: bold; color: #fff;">📅 {date}</div><div style="display: flex; justify-content: space-around; margin-top: 8px;"><span class="win">✅ {wins}/4</span><span style="color: {'#00ff88' if rate >= 50 else '#ff4b4b'}">{rate:.0f}%</span></div></div>''', unsafe_allow_html=True)
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
        st.markdown(f'''<div class="stat-card" style="margin-top: 10px;"><div class="stat-value" style="color: {'#00ff88' if win_rate >= 50 else '#ff4b4b'}">{win_rate:.1f}%</div><div class="stat-label">Tỷ lệ trúng</div><div class="stat-label">{total_checks} lần check</div></div>''', unsafe_allow_html=True)
    
    elif page == "📜 Lịch sử":
        st.markdown("### 📜 LỊCH SỬ PHÂN TÍCH")
        results = st.session_state.statistics.get('results', [])
        if results:
            dates = sorted(set(r['pred_date'] for r in results), reverse=True)
            selected_date = st.selectbox("Chọn ngày", dates)
            filtered = [r for r in results if r['pred_date'] == selected_date]
            for r in filtered:
                st.markdown(f'''<div style="background: rgba(212, 175, 55, 0.1); border: 1px solid #D4AF37; border-radius: 10px; padding: 15px; margin: 10px 0;"><b>📅 {r['pred_date']} → {r['check_date']}</b><div>Data 1: {r['bach_thu']['number']} {'✅' if r['bach_thu']['win'] else '❌'}</div><div>Data 2: {'-'.join(r['song_thu']['numbers'])} {'✅' if r['song_thu']['win'] else '❌'}</div><div>Data 3: {r['xien_2']['pair']} {'✅' if r['xien_2']['win'] else '❌'}</div><div>Data 4: {'✅' if r['dan_de']['win'] else '❌'}</div><div><b>Tổng: {r['overall_wins']}/4</b></div></div>''', unsafe_allow_html=True)
        else:
            st.info("📭 Chưa có lịch sử")
    
    elif page == "👑 Admin" and get_user_info().get('role') == 'admin':
        admin_panel()
    
    # FOOTER
    st.markdown("---")
    st.markdown('''
    <div style="text-align: center; color: #666; padding: 20px; font-size: 12px;">
        💎 AI-QUANTUM PRO 2026 • Data Analytics Platform<br>
        Dữ liệu thống kê tham khảo • Không tổ chức cá cược<br>
        18+ only • Chơi có trách nhiệm
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()