# ==============================================================================
# FILE: app.py - TITAN v34.0 ULTRA (SMART AI VERSION)
# ==============================================================================

import streamlit as st
import google.generativeai as genai
import re
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import time
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="TITAN v34.0 ULTRA | Smart AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS - Enhanced with animations
st.markdown("""
<style>
    .stApp {
        background-color: #010409;
        color: #e6edf3;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Status bars with glow effect */
    .status-green {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(35,134,54,0.5);
        animation: pulse 2s infinite;
    }
    
    .status-red {
        background: linear-gradient(135deg, #da3633, #f85149);
        color: white;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(218,54,51,0.5);
    }
    
    .status-yellow {
        background: linear-gradient(135deg, #d29922, #f0b429);
        color: #0d1117;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 20px rgba(35,134,54,0.5); }
        50% { box-shadow: 0 0 30px rgba(35,134,54,0.8); }
        100% { box-shadow: 0 0 20px rgba(35,134,54,0.5); }
    }
    
    /* Number boxes with 3D effect */
    .numbers-container {
        display: flex;
        flex-direction: row;
        justify-content: center;
        align-items: center;
        gap: 12px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    
    .num-box {
        font-size: 55px;
        font-weight: 900;
        color: #ff5858;
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 3px solid #ff5858;
        border-radius: 15px;
        padding: 20px 30px;
        min-width: 80px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(255,88,88,0.4), inset 0 0 20px rgba(255,88,88,0.1);
        text-shadow: 0 0 15px rgba(255,88,88,0.8);
    }
    
    .lot-box {
        font-size: 40px;
        font-weight: 800;
        color: #58a6ff;
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 2px solid #58a6ff;
        border-radius: 12px;
        padding: 15px 22px;
        min-width: 65px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(88,166,255,0.3);
    }
    
    /* Cards */
    .prediction-card {
        background: linear-gradient(135deg, #0d1117, #161b22);
        border: 1px solid #30363d;
        border-radius: 15px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        gap: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #58a6ff, #1f6feb);
        color: white;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        padding: 15px 35px;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(35,134,54,0.5);
    }
    
    /* Mobile responsive */
    @media (max-width: 600px) {
        .num-box {
            font-size: 40px;
            padding: 15px 20px;
            min-width: 60px;
        }
        .lot-box {
            font-size: 32px;
            padding: 12px 18px;
            min-width: 50px;
        }
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #58a6ff, #1f6feb);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SMART CACHE & API
# ==============================================================================

@st.cache_resource
def init_gemini():
    """Initialize Gemini API with error handling."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    except Exception as e:
        st.error(f"❌ Lỗi API: {str(e)}")
        return None

@st.cache_data
def get_quota_status():
    """Smart quota management with auto-reset."""
    if "quota" not in st.session_state:
        st.session_state.quota = {
            "used": 0,
            "limit": 15,
            "reset_date": datetime.now().date().isoformat(),
            "last_call": None
        }
    
    quota = st.session_state.quota
    today = datetime.now().date().isoformat()
    
    if quota["reset_date"] != today:
        quota["used"] = 0
        quota["reset_date"] = today
    
    return quota

def use_quota():
    """Consume quota with tracking."""
    quota = get_quota_status()
    if quota["used"] < quota["limit"]:
        quota["used"] += 1
        quota["last_call"] = datetime.now().isoformat()
        return True
    return False

# ==============================================================================
# 3. ADVANCED DATA CLEANING (FIXED!)
# ==============================================================================

def clean_lottery_data_advanced(raw_text, existing_db):
    """
    SMART DATA CLEANING - Xử lý mọi định dạng
    - Loại bỏ khoảng trắng thừa
    - Chuẩn hóa số 5 chữ số
    - Phát hiện số trùng thông minh
    """
    if not raw_text.strip():
        return [], {"found": 0, "new": 0, "duplicate_input": 0, "already_in_db": 0, "invalid": 0}
    
    # Bước 1: Chuẩn hóa text (loại bỏ khoảng trắng thừa)
    normalized = re.sub(r'\s+', ' ', raw_text.strip())
    
    # Bước 2: Tách dòng và làm sạch từng dòng
    lines = normalized.split('\n')
    cleaned_numbers = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Loại bỏ tất cả khoảng trắng trong dòng (xử lý "41 720" → "41720")
        line_no_spaces = re.sub(r'\s', '', line)
        
        # Tìm tất cả số 5 chữ số trong dòng
        matches = re.findall(r'\d{5}', line_no_spaces)
        cleaned_numbers.extend(matches)
    
    # Bước 3: Loại duplicate trong input mới
    unique_input = []
    seen = set()
    stats = {
        "found": len(cleaned_numbers),
        "new": 0,
        "duplicate_input": 0,
        "already_in_db": 0,
        "invalid": 0
    }
    
    db_set = set(existing_db)
    
    for num in cleaned_numbers:
        if len(num) != 5:
            stats["invalid"] += 1
            continue
        
        if num in seen:
            stats["duplicate_input"] += 1
            continue
        seen.add(num)
        
        if num in db_set:
            stats["already_in_db"] += 1
            continue
        
        unique_input.append(num)
        stats["new"] += 1
    
    return unique_input, stats

def add_to_database(new_numbers):
    """Add numbers with deduplication and sorting."""
    if "lottery_db" not in st.session_state:
        st.session_state.lottery_db = []
    
    # Thêm số mới vào đầu (mới nhất trước)
    st.session_state.lottery_db = new_numbers + st.session_state.lottery_db
    
    # Loại duplicate toàn bộ database
    seen = set()
    unique_db = []
    for num in st.session_state.lottery_db:
        if num not in seen:
            seen.add(num)
            unique_db.append(num)
    
    st.session_state.lottery_db = unique_db[:3000]  # Giới hạn 3000 số

def check_win(prediction_3, result_5):
    """Check win condition for 3 số 5 tinh."""
    if len(prediction_3) != 3 or len(result_5) != 5:
        return False
    result_digits = set(result_5)
    return all(digit in result_digits for digit in prediction_3)

# ==============================================================================
# 4. SMART PREDICTION ALGORITHMS (KUBET-OPTIMIZED)
# ==============================================================================

def calculate_risk_advanced(history, window=50):
    """
    ADVANCED RISK DETECTION - Phát hiện bẫy nhà cái Kubet
    """
    if len(history) < 20:
        return 0, []
    
    recent = history[-window:] if len(history) >= window else history
    all_digits = ''.join(recent)
    freq = Counter(all_digits)
    reasons = []
    risk = 0
    
    # 1. Số xuất hiện quá nhiều (nhà cái đang điều khiển)
    total_slots = len(all_digits)
    if total_slots > 0:
        most_common = freq.most_common(3)
        for num, count in most_common:
            rate = count / total_slots
            if rate > 0.25:  # >25% tần suất
                risk += 20
                reasons.append(f"⚠️ Số '{num}' xuất hiện {rate*100:.0f}% (bất thường)")
    
    # 2. Cầu bệt bất thường (nhà cái đang "bệt" để dụ)
    for pos in range(5):
        seq = [n[pos] if len(n) > pos else '0' for n in recent]
        max_streak = 1
        current = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 1
        if max_streak >= 5:
            risk += 30
            reasons.append(f"🚫 Cầu bệt {max_streak} kỳ vị trí {pos} (NHÀ CÁI ĐIỀU KHIỂN)")
    
    # 3. Entropy quá thấp (kết quả giả)
    if len(all_digits) > 0:
        entropy = -sum((c/len(all_digits)) * np.log2(c/len(all_digits)) 
                      for c in freq.values() if c > 0)
        if entropy < 2.8:
            risk += 25
            reasons.append(f"⚠️ Entropy thấp ({entropy:.2f}) - Kết quả không tự nhiên")
    
    # 4. Tổng các số quá ổn định
    totals = [sum(int(d) for d in n) for n in recent if len(n) == 5]
    if len(totals) > 10:
        std_dev = np.std(totals)
        if std_dev < 2.5:
            risk += 15
            reasons.append(f"⚠️ Tổng quá ổn định (σ={std_dev:.2f})")
    
    # 5. Phát hiện pattern "bẫy" (số về rồi ngắt đột ngột)
    for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        positions = [i for i, num in enumerate(recent) if digit in num]
        if len(positions) >= 3:
            gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            if gaps and max(gaps) > 10:
                risk += 10
                reasons.append(f"⚠️ Số {digit} có khoảng cách bất thường")
    
    return min(100, risk), reasons

def analyze_frequency_smart(history, window=100):
    """
    SMART FREQUENCY ANALYSIS - Trọng số thông minh
    """
    recent = history[-window:] if len(history) >= window else history
    
    weighted_freq = defaultdict(float)
    
    # Trọng số exponential (kỳ gần nặng hơn nhiều)
    for idx, num in enumerate(recent):
        # Exponential decay: recent = 3.0, older = 1.0
        weight = 1.0 + 2.0 * np.exp(-idx / 20)
        for digit in num:
            if digit.isdigit():
                weighted_freq[digit] += weight
    
    sorted_items = sorted(weighted_freq.items(), key=lambda x: x[1], reverse=True)
    top_3 = [str(x[0]) for x in sorted_items[:3]]
    
    while len(top_3) < 3:
        for i in range(10):
            if str(i) not in top_3:
                top_3.append(str(i))
                break
    
    return {
        'top_3': top_3,
        'scores': {k: round(v, 2) for k, v in sorted_items[:10]},
        'method': 'Exponential Weighted Frequency'
    }

def analyze_positions_smart(history, window=50):
    """
    SMART POSITION ANALYSIS - Phân tích từng vị trí với correlation
    """
    recent = history[-window:] if len(history) >= window else history
    
    pos_freq = [Counter() for _ in range(5)]
    pos_pairs = defaultdict(int)  # Track digit pairs across positions
    
    for num in recent:
        for i, digit in enumerate(num[:5]):
            pos_freq[i][digit] += 1
        
        # Track pairs (useful for pattern detection)
        for i in range(4):
            pair = (num[i], num[i+1])
            pos_pairs[pair] += 1
    
    pos_top = []
    for i in range(5):
        if pos_freq[i]:
            pos_top.append(pos_freq[i].most_common(1)[0][0])
        else:
            pos_top.append('0')
    
    # Vote based on position strength
    all_candidates = []
    for i in range(5):
        weight = 5 - i  # Earlier positions weighted higher
        for digit, count in pos_freq[i].most_common(3):
            all_candidates.extend([digit] * weight)
    
    vote_count = Counter(all_candidates)
    top_3 = [str(x[0]) for x in vote_count.most_common(3)]
    
    while len(top_3) < 3:
        for i in range(10):
            if str(i) not in top_3:
                top_3.append(str(i))
                break
    
    return {
        'top_3': top_3,
        'pos_top': pos_top,
        'pos_freq': [dict(cf.most_common(5)) for cf in pos_freq],
        'votes': dict(vote_count.most_common(10)),
        'method': 'Position-Weighted Analysis'
    }

def analyze_hot_cold_smart(history, recent_window=10, cold_window=20):
    """
    SMART HOT/COLD - Phát hiện số "đến kỳ" của nhà cái
    """
    recent = history[-recent_window:] if len(history) >= recent_window else history
    older = history[-cold_window:-recent_window] if len(history) >= cold_window else []
    
    recent_digits = Counter(''.join(recent))
    older_digits = Counter(''.join(older)) if older else Counter()
    
    # Hot: về nhiều trong 10 kỳ gần
    hot = [str(x[0]) for x in recent_digits.most_common(5)]
    
    # Cold: không về trong 20 kỳ
    all_recent = ''.join(recent)
    cold = [str(i) for i in range(10) if str(i) not in all_recent]
    
    # DUE: Số lạnh nhưng từng hot (nhà cái sắp "nhả")
    due = []
    for num in cold:
        old_count = older_digits.get(num, 0)
        if old_count >= 4:  # Từng về nhiều, giờ không về → sắp về
            due.append(num)
    
    # VERY HOT: Về liên tục 3+ kỳ (cẩn thận gãy cầu)
    very_hot = []
    for digit in hot:
        consecutive = 0
        for num in recent:
            if digit in num:
                consecutive += 1
            else:
                break
        if consecutive >= 3:
            very_hot.append(digit)
    
    return {
        'hot': hot,
        'cold': cold,
        'due': due,
        'very_hot': very_hot,
        'method': 'Smart Hot/Cold Cycle Detection'
    }

def detect_patterns_advanced(history, window=30):
    """
    ADVANCED PATTERN DETECTION - Phát hiện pattern nhà cái Kubet
    """
    recent = history[-window:] if len(history) >= window else history
    
    patterns = {
        'bet': [],      # Cầu bệt
        'nhip2': [],    # Cầu nhịp 2
        'nhip3': [],    # Cầu nhịp 3
        'dao': [],      # Cầu đảo
        'tam_giac': [], # Cầu tam giác
        'detected': [],
        'likely': [],
        'avoid': []     # Số nên tránh
    }
    
    # 1. Cầu bệt (streak) - Nhà cái thường bệt 4-5 kỳ rồi gãy
    for pos in range(5):
        seq = [n[pos] if len(n) > pos else '0' for n in recent]
        for i in range(len(seq) - 2):
            if seq[i] == seq[i+1] == seq[i+2]:
                digit = seq[i]
                streak_len = 3
                for j in range(i+3, len(seq)):
                    if seq[j] == digit:
                        streak_len += 1
                    else:
                        break
                
                if digit not in patterns['bet']:
                    patterns['bet'].append(digit)
                    patterns['detected'].append(f'Bệt {streak_len} kỳ vị {pos}: {digit}')
                    
                    # Nếu bệt 4+ kỳ → sắp gãy → TRÁNH
                    if streak_len >= 4:
                        if digit not in patterns['avoid']:
                            patterns['avoid'].append(digit)
                            patterns['detected'].append(f'⚠️ Sắp gãy cầu: {digit}')
                    else:
                        if digit not in patterns['likely']:
                            patterns['likely'].append(digit)
    
    # 2. Cầu nhịp 2 (X _ X _ X)
    for pos in range(5):
        seq = [n[pos] if len(n) > pos else '0' for n in recent]
        for i in range(len(seq) - 4):
            if seq[i] == seq[i+2] == seq[i+4] and seq[i] != seq[i+1]:
                digit = seq[i]
                if digit not in patterns['nhip2']:
                    patterns['nhip2'].append(digit)
                    patterns['detected'].append(f'Nhịp-2 vị {pos}: {digit}')
                    if digit not in patterns['likely']:
                        patterns['likely'].append(digit)
    
    # 3. Cầu nhịp 3 (X _ _ X _ _ X)
    for pos in range(5):
        seq = [n[pos] if len(n) > pos else '0' for n in recent]
        for i in range(len(seq) - 6):
            if seq[i] == seq[i+3] == seq[i+6]:
                digit = seq[i]
                if digit not in patterns['nhip3']:
                    patterns['nhip3'].append(digit)
                    patterns['detected'].append(f'Nhịp-3 vị {pos}: {digit}')
                    if digit not in patterns['likely']:
                        patterns['likely'].append(digit)
    
    # 4. Cầu đảo (AB → BA)
    for i in range(len(recent) - 1):
        curr, next_draw = recent[i], recent[i+1]
        if len(curr) >= 2 and len(next_draw) >= 2:
            if curr[0:2] == next_draw[1::-1]:
                patterns['dao'].extend([curr[0], curr[1]])
                patterns['detected'].append(f'Đảo: {curr[0:2]} → {next_draw[0:2]}')
    
    # 5. Cầu tam giác (A → B → C → A)
    for pos in range(5):
        seq = [n[pos] if len(n) > pos else '0' for n in recent]
        for i in range(len(seq) - 3):
            if seq[i] == seq[i+3] and seq[i] != seq[i+1] and seq[i+1] != seq[i+2]:
                digit = seq[i]
                if digit not in patterns['tam_giac']:
                    patterns['tam_giac'].append(digit)
                    patterns['detected'].append(f'Tam-giác vị {pos}: {digit}')
    
    return patterns

def ai_kubet_analysis(history, model):
    """
    AI ANALYSIS SPECIFIC FOR KUBET - Tìm điểm yếu nhà cái
    """
    if not model:
        return {"error": "AI not available", "fallback": True}
    
    quota = get_quota_status()
    if quota["used"] >= quota["limit"]:
        return {"error": "Quota exceeded", "fallback": True}
    
    try:
        # Lấy 50 kỳ gần nhất
        context = history[:50] if len(history) >= 50 else history
        
        prompt = f"""
Bạn là chuyên gia phân tích xổ số KUBET. Nhiệm vụ: Tìm ĐIỂM YẾU của nhà cái để dự đoán.

Dữ liệu 50 kỳ gần nhất (mới nhất ở đầu):
{', '.join(context)}

PHÂN TÍCH:
1. Tìm 3 số có xác suất về cao NHẤT trong kỳ tiếp theo
2. Phát hiện pattern nhà cái đang dùng (bệt, nhịp, đảo...)
3. Xác định số nào nhà cái đang "bẫy" (nên tránh)
4. Đánh giá mức độ điều khiển của nhà cái (0-100%)

Trả về JSON STRICT format:
{{
    "main_3": ["4", "7", "9"],
    "support_4": ["2", "1", "8", "5"],
    "avoid": ["3"],
    "decision": "ĐÁNH" hoặc "THEO DÕI" hoặc "DỪNG",
    "confidence": 75,
    "house_control": 60,
    "logic": "Giải thích ngắn gọn bằng tiếng Việt",
    "patterns_found": ["bệt 3 kỳ số 7", "nhịp 2 số 4"],
    "method": "Kubet AI Pattern Analysis"
}}

Chỉ trả về JSON, không markdown."""
        
        response = model.generate_content(prompt, timeout=30)
        text = response.text.strip()
        
        # Parse JSON với nhiều fallback
        try:
            result = json.loads(text)
        except:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                return {"error": "JSON parse failed", "fallback": True}
        
        # Validate
        if not isinstance(result, dict) or 'main_3' not in result:
            return {"error": "Invalid response", "fallback": True}
        
        use_quota()
        return result
        
    except Exception as e:
        return {"error": f"AI Error: {str(e)}", "fallback": True}

def consensus_engine_v2(stat_result, pos_result, hotcold_result, pattern_result, ai_result=None):
    """
    V2 CONSENSUS - Thông minh hơn với weighted voting
    """
    all_votes = []
    avoid_votes = []
    
    # Frequency 40% (tăng lên vì quan trọng nhất)
    for num in stat_result.get('top_3', []):
        all_votes.extend([num] * 5)
    
    # Position 30%
    for num in pos_result.get('top_3', []):
        all_votes.extend([num] * 4)
    
    # Hot/Cold 20%
    for num in hotcold_result.get('hot', [])[:3]:
        all_votes.extend([num] * 3)
    
    # Due numbers (quan trọng cho Kubet)
    for num in hotcold_result.get('due', []):
        all_votes.extend([num] * 4)
    
    # Pattern 10%
    for num in pattern_result.get('likely', []):
        all_votes.append(num)
    
    # Numbers to avoid
    for num in pattern_result.get('avoid', []):
        avoid_votes.append(num)
    
    # AI bonus (nếu có)
    if ai_result and 'main_3' in ai_result and not ai_result.get('error'):
        for num in ai_result['main_3']:
            all_votes.extend([num] * 6)  # AI có trọng số cao
        for num in ai_result.get('avoid', []):
            avoid_votes.append(num)
    
    if not all_votes:
        return None
    
    vote_count = Counter(all_votes)
    avoid_set = set(avoid_votes)
    
    # Get top 3, excluding avoid numbers
    final_3 = []
    for num, count in vote_count.most_common():
        if num not in final_3 and num not in avoid_set:
            final_3.append(num)
        if len(final_3) == 3:
            break
    
    # Fill if needed
    while len(final_3) < 3:
        for i in range(10):
            if str(i) not in final_3 and str(i) not in avoid_set:
                final_3.append(str(i))
                break
    
    # Support 4
    remaining = [n for n, c in vote_count.most_common(10) if n not in final_3 and n not in avoid_set]
    support_4 = remaining[:4]
    while len(support_4) < 4:
        for i in range(10):
            if str(i) not in final_3 and str(i) not in support_4 and str(i) not in avoid_set:
                support_4.append(str(i))
                break
    
    # Confidence calculation
    if vote_count:
        top_vote = vote_count.most_common(1)[0][1]
        confidence = min(95, 55 + top_vote * 3)
    else:
        confidence = 50
    
    # Reduce confidence if avoid numbers are in top votes
    if avoid_set:
        confidence = max(40, confidence - 15)
    
    return {
        'main_3': final_3,
        'support_4': support_4,
        'confidence': confidence,
        'avoid': list(avoid_set),
        'vote_breakdown': dict(vote_count.most_common(10)),
        'method': 'Smart Consensus V2'
    }

def predict_3_numbers_smart(history, model=None):
    """
    MAIN PREDICTION - Smart multi-layer analysis
    """
    if len(history) < 20:
        return {
            'error': f'Cần ít nhất 20 kỳ dữ liệu (hiện có: {len(history)})',
            'main_3': ['?', '?', '?'],
            'support_4': ['0', '0', '0', '0'],
            'decision': 'CHỜ DỮ LIỆU',
            'confidence': 0,
            'logic': 'Vui lòng nhập thêm kết quả lịch sử',
            'risk_score': 0
        }
    
    # STEP 1: Risk Detection FIRST
    risk_score, risk_reasons = calculate_risk_advanced(history)
    
    if risk_score >= 70:
        return {
            'main_3': ['0', '0', '0'],
            'support_4': ['0', '0', '0', '0'],
            'decision': 'DỪNG',
            'confidence': 95,
            'logic': f'🚫 Rủi ro rất cao ({risk_score}/100) - Nhà cái đang điều khiển mạnh',
            'risk_score': risk_score,
            'risk_reasons': risk_reasons
        }
    
    # STEP 2: Run all analysis methods
    stat_result = analyze_frequency_smart(history, window=100)
    pos_result = analyze_positions_smart(history, window=50)
    hotcold_result = analyze_hot_cold_smart(history)
    pattern_result = detect_patterns_advanced(history, window=30)
    
    # STEP 3: AI Analysis (if available)
    ai_result = None
    if model and get_quota_status()["used"] < get_quota_status()["limit"]:
        ai_result = ai_kubet_analysis(history, model)
    
    # STEP 4: Consensus Engine V2
    consensus = consensus_engine_v2(stat_result, pos_result, hotcold_result, pattern_result, ai_result)
    
    if not consensus:
        return {'error': 'Không thể tạo dự đoán', 'risk_score': risk_score}
    
    # STEP 5: Final Decision
    if risk_score < 30 and consensus['confidence'] >= 75:
        decision = 'ĐÁNH'
    elif risk_score < 50:
        decision = 'THEO DÕI'
    else:
        decision = 'DỪNG'
    
    # STEP 6: Build Logic
    logic_parts = []
    if stat_result['top_3']:
        logic_parts.append(f"Tần suất: {','.join(stat_result['top_3'])}")
    if hotcold_result.get('due'):
        logic_parts.append(f"Đến kỳ: {','.join(hotcold_result['due'][:2])}")
    if pattern_result['detected']:
        logic_parts.append(f"{len(pattern_result['detected'])} pattern")
    if consensus.get('avoid'):
        logic_parts.append(f"Tránh: {','.join(consensus['avoid'])}")
    
    logic = ' | '.join(logic_parts) if logic_parts else 'Phân tích đa lớp thông minh'
    
    # Add AI info if available
    if ai_result and not ai_result.get('error'):
        logic += f" | AI: {ai_result.get('logic', '')}"
    
    return {
        'main_3': consensus['main_3'],
        'support_4': consensus['support_4'],
        'decision': decision,
        'confidence': consensus['confidence'],
        'logic': logic,
        'risk_score': risk_score,
        'risk_reasons': risk_reasons,
        'details': {
            'frequency': stat_result,
            'positions': pos_result,
            'hot_cold': hotcold_result,
            'patterns': pattern_result,
            'ai': ai_result if ai_result and not ai_result.get('error') else None,
            'votes': consensus.get('vote_breakdown', {}),
            'avoid': consensus.get('avoid', [])
        }
    }

# ==============================================================================
# 5. WIN RATE TRACKING
# ==============================================================================

def log_prediction(prediction, actual_result=None):
    """Log prediction with full details."""
    if "predictions_log" not in st.session_state:
        st.session_state.predictions_log = []
    
    entry = {
        'timestamp': datetime.now().isoformat(),
        'predicted': prediction.get('main_3', []),
        'support': prediction.get('support_4', []),
        'confidence': prediction.get('confidence', 0),
        'decision': prediction.get('decision', ''),
        'risk_score': prediction.get('risk_score', 0),
        'actual_result': actual_result,
        'won': None
    }
    
    if actual_result and len(actual_result) == 5 and actual_result.isdigit():
        if len(prediction.get('main_3', [])) == 3:
            entry['won'] = check_win(''.join(prediction['main_3']), actual_result)
    
    st.session_state.predictions_log.insert(0, entry)
    
    if len(st.session_state.predictions_log) > 200:
        st.session_state.predictions_log = st.session_state.predictions_log[:200]

# ==============================================================================
# 6. UI COMPONENTS
# ==============================================================================

def render_prediction_display(result, risk_info):
    """Render prediction with enhanced UI."""
    risk_score, risk_reasons = risk_info
    
    if not result or 'main_3' not in result:
        st.warning("⚠️ Chưa có kết quả dự đoán.")
        return
    
    main_3 = result['main_3']
    support_4 = result.get('support_4', ['?']*4)
    avoid = result.get('details', {}).get('avoid', [])
    
    main_3 = (main_3 + ['?']*3)[:3]
    support_4 = (support_4 + ['?']*4)[:4]
    
    # Status bar
    if result['decision'] == 'ĐÁNH':
        status_class = "status-green"
        status_icon = "✅"
    elif result['decision'] == 'DỪNG':
        status_class = "status-red"
        status_icon = "🛑"
    else:
        status_class = "status-yellow"
        status_icon = "⚠️"
    
    st.markdown(f"""
    <div class="{status_class}">
        {status_icon} RISK: {risk_score}/100 | KHUYẾN NGHỊ: {result['decision']}
    </div>
    """, unsafe_allow_html=True)
    
    # 3 SỐ CHÍNH
    st.markdown(f"""
    <div style="text-align: center; margin: 25px 0;">
        <div style="color: #8b949e; font-size: 15px; margin-bottom: 15px; font-weight: bold;">
            🔮 3 SỐ CHÍNH (Độ tin cậy: {result['confidence']}%)
        </div>
        <div class="numbers-container">
            <div class="num-box">{main_3[0]}</div>
            <div class="num-box">{main_3[1]}</div>
            <div class="num-box">{main_3[2]}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Avoid numbers warning
    if avoid:
        st.markdown(f"""
        <div style="text-align: center; margin: 15px 0; padding: 10px; background: rgba(218,54,51,0.2); border: 1px solid #da3633; border-radius: 8px;">
            <span style="color: #f85149; font-weight: bold;">🚫 TRÁNH: {', '.join(avoid)}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Divider
    st.markdown("<div style='border-top: 2px solid #30363d; margin: 20px 0;'></div>", unsafe_allow_html=True)
    
    # 4 SỐ LÓT
    st.markdown(f"""
    <div style="text-align: center; margin: 25px 0;">
        <div style="color: #8b949e; font-size: 14px; margin-bottom: 15px; font-weight: bold;">
            🎲 4 SỐ LÓT
        </div>
        <div class="numbers-container">
            <div class="lot-box">{support_4[0]}</div>
            <div class="lot-box">{support_4[1]}</div>
            <div class="lot-box">{support_4[2]}</div>
            <div class="lot-box">{support_4[3]}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Logic
    st.info(f"💡 **Logic:** {result['logic']}")
    
    # Risk warnings
    if risk_reasons:
        warning_text = "⚠️ **Cảnh báo:**\n"
        for reason in risk_reasons:
            warning_text += f"• {reason}\n"
        st.warning(warning_text)
    
    # Copy code
    st.markdown("---")
    numbers_to_copy = ','.join(main_3 + support_4)
    st.code(numbers_to_copy, language=None)
    st.caption("📋 Bấm vào code để copy dàn 7 số")

def render_analysis_details(result):
    """Render detailed analysis."""
    if not result or 'details' not in result:
        return
    
    details = result['details']
    
    with st.expander("🧠 Chi tiết phân tích thông minh", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Tần suất (40%)")
            if 'frequency' in details and 'scores' in details['frequency']:
                freq_df = pd.DataFrame(
                    list(details['frequency']['scores'].items()), 
                    columns=['Số', 'Điểm']
                ).sort_values('Điểm', ascending=False)
                st.bar_chart(freq_df.set_index('Số'))
            
            st.markdown("### 📍 Vị trí (30%)")
            if 'positions' in details and 'pos_top' in details['positions']:
                pos_data = {f'Vị {i}': details['positions']['pos_top'][i] for i in range(5)}
                st.json(pos_data)
        
        with col2:
            st.markdown("### 🔥 Nóng/Lạnh/Đến kỳ (20%)")
            if 'hot_cold' in details:
                hc = details['hot_cold']
                st.markdown(f"**🔥 Nóng:** {' '.join(hc.get('hot', [])[:5])}")
                if hc.get('due'):
                    st.markdown(f"**⏰ Đến kỳ:** {' '.join(hc['due'])} (SẮP VỀ)")
                if hc.get('very_hot'):
                    st.markdown(f"**⚠️ Rất nóng:** {' '.join(hc['very_hot'])} (Cẩn thận gãy)")
            
            st.markdown("### 🔄 Pattern (10%)")
            if 'patterns' in details and details['patterns'].get('detected'):
                for p in details['patterns']['detected'][:7]:
                    st.markdown(f"• {p}")
        
        # AI Analysis
        if details.get('ai'):
            st.markdown("---")
            st.markdown("### 🤖 AI Kubet Analysis")
            ai = details['ai']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{ai.get('confidence', 0)}%")
            with col2:
                st.metric("House Control", f"{ai.get('house_control', 0)}%")
            with col3:
                st.metric("Decision", ai.get('decision', 'N/A'))
            st.markdown(f"*{ai.get('logic', '')}*")
        
        # Vote breakdown
        if 'votes' in details and details['votes']:
            st.markdown("---")
            st.markdown("### 🗳️ Consensus Voting")
            votes_df = pd.DataFrame(
                list(details['votes'].items()), 
                columns=['Số', 'Phiếu']
            ).sort_values('Phiếu', ascending=False)
            st.dataframe(votes_df, hide_index=True, use_container_width=True)

def render_stats_tab():
    """Render statistics tab."""
    st.header("📊 Thống kê & Database")
    
    if "lottery_db" not in st.session_state or not st.session_state.lottery_db:
        st.info("📭 Chưa có dữ liệu.")
        return
    
    db = st.session_state.lottery_db
    predictions_log = st.session_state.get('predictions_log', [])
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Tổng kỳ", len(db))
    with col2:
        recent_digits = ''.join(db[:100])
        hottest = Counter(recent_digits).most_common(1)[0][0] if recent_digits else "-"
        st.metric("🔥 Số nóng nhất", hottest)
    with col3:
        if predictions_log:
            decided = [e for e in predictions_log if e['won'] is not None]
            if decided:
                wins = sum(1 for e in decided if e['won'])
                win_rate = round(wins/len(decided)*100, 1)
                st.metric("🎯 Win Rate", f"{win_rate}%")
            else:
                st.metric("🎯 Win Rate", "Chưa có")
        else:
            st.metric("🎯 Win Rate", "Chưa có")
    with col4:
        quota = get_quota_status()
        st.metric("🤖 Gemini", f"{quota['limit'] - quota['used']}/{quota['limit']}")
    
    # Charts
    tab1, tab2 = st.tabs(["📈 Biểu đồ", "📋 Dữ liệu"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Tần suất 50 kỳ gần")
            recent_50 = db[:50]
            all_digits = ''.join(recent_50)
            freq = Counter(all_digits)
            df_freq = pd.DataFrame(
                [(str(d), c) for d, c in sorted(freq.items())], 
                columns=['Số', 'Tần suất']
            )
            st.bar_chart(df_freq.set_index('Số'))
        
        with col2:
            st.markdown("### Top 5 Số Nóng")
            all_digits_full = ''.join(db)
            freq_full = Counter(all_digits_full)
            top_5 = freq_full.most_common(5)
            
            for num, count in top_5:
                st.metric(f"Số {num}", f"{count} lần")
    
    with tab2:
        st.markdown("### 💾 Quản lý Database")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(db, indent=2, ensure_ascii=False),
                file_name=f"titan_db_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        with c2:
            uploaded = st.file_uploader("📤 Upload JSON", type="json")
            if uploaded:
                try:
                    new_db = json.load(uploaded)
                    if isinstance(new_db, list) and all(len(str(x))==5 for x in new_db):
                        st.session_state.lottery_db = new_db
                        st.success("✅ Đã tải database thành công!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ File không hợp lệ")
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
        
        with c3:
            if st.button("🗑️ Xóa toàn bộ", type="primary"):
                st.session_state.lottery_db = []
                st.success("✅ Đã xóa dữ liệu!")
                time.sleep(1)
                st.rerun()
        
        st.markdown("### 📜 Lịch sử 20 kỳ gần")
        df_hist = pd.DataFrame(db[:20], columns=["Kết Quả"])
        df_hist.index = [f"#{i+1}" for i in range(len(df_hist))]
        st.dataframe(df_hist, use_container_width=True, hide_index=False)

# ==============================================================================
# 7. MAIN APP
# ==============================================================================

def main():
    # Initialize session state
    if "lottery_db" not in st.session_state:
        st.session_state.lottery_db = []
    if "predictions_log" not in st.session_state:
        st.session_state.predictions_log = []
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "last_risk" not in st.session_state:
        st.session_state.last_risk = None
    
    # Header
    st.title("🧠 TITAN v34.0 ULTRA")
    st.caption("Smart AI Prediction | Tìm điểm yếu nhà cái Kubet")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Trạng thái Hệ thống")
        quota = get_quota_status()
        
        if quota["used"] >= quota["limit"]:
            st.error(f"🚫 Gemini: HẾT ({quota['limit']}/{quota['limit']})")
        elif quota["used"] <= 3:
            st.success(f"✅ Gemini: {quota['limit'] - quota['used']}/{quota['limit']}")
        else:
            st.warning(f"⚠️ Gemini: {quota['limit'] - quota['used']}/{quota['limit']}")
        
        st.markdown("---")
        st.markdown("### 📋 Hướng dẫn nhanh")
        st.markdown("""
        1️⃣ Dán kết quả (5 số/dòng)  
        2️⃣ Nhấn "🚀 PHÂN TÍCH THÔNG MINH"  
        3️⃣ Xem kết quả + pattern nhà cái  
        4️⃣ Theo dõi win rate ở Tab 3
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Cảnh báo quan trọng")
        st.warning("""
        • Risk >= 70: DỪNG ngay  
        • Số trong mục TRÁNH: Không đánh  
        • Luôn quản lý vốn
        """)
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Nhập & Dự đoán", "🧠 Phân tích Chi tiết", "📊 Thống kê"])
    
    # ==================== TAB 1 ====================
    with tab1:
        st.header("📥 Nhập Kết Quả & Phân Tích Thông Minh")
        
        st.markdown("""
        **💡 Mẹo:** Dán càng nhiều kỳ lịch sử, AI càng thông minh!
        - Tối thiểu: 20 kỳ
        - Tốt nhất: 100+ kỳ
        - Hệ thống tự động làm sạch dữ liệu
        """)
        
        input_text = st.text_area(
            "📋 Dữ liệu thô (5 chữ số mỗi dòng)",
            height=200,
            placeholder="Ví dụ:\n87746\n56421\n69137\n...",
            key="input_area"
        )
        
        if st.button("🚀 PHÂN TÍCH THÔNG MINH", type="primary"):
            if input_text.strip():
                with st.spinner("🧠 Đang phân tích thông minh..."):
                    start_time = time.time()
                    
                    # 1. Clean data (FIXED - xử lý mọi định dạng)
                    new_nums, stats = clean_lottery_data_advanced(input_text, st.session_state.lottery_db)
                    
                    if stats['new'] > 0:
                        add_to_database(new_nums)
                        elapsed = time.time() - start_time
                        
                        st.success(f"✅ Xử lý xong trong {elapsed:.2f}s | Thêm {stats['new']} số mới (Tìm thấy: {stats['found']} | Trùng: {stats['duplicate_input']} | Có trong DB: {stats['already_in_db']})")
                        
                        # 2. Smart prediction
                        model = init_gemini()
                        result = predict_3_numbers_smart(st.session_state.lottery_db, model)
                        risk_info = (result.get('risk_score', 0), result.get('risk_reasons', []))
                        
                        # 3. Log prediction
                        log_prediction(result)
                        
                        # 4. Save state
                        st.session_state.last_prediction = result
                        st.session_state.last_risk = risk_info
                        
                        # 5. Display result
                        st.markdown("### 🎯 Kết quả Dự đoán Thông minh")
                        render_prediction_display(result, risk_info)
                        
                    elif stats['found'] > 0:
                        st.warning(f"⚠️ Không có số mới (tất cả đã có trong DB)")
                        if len(st.session_state.lottery_db) >= 20:
                            model = init_gemini()
                            result = predict_3_numbers_smart(st.session_state.lottery_db, model)
                            risk_info = (result.get('risk_score', 0), result.get('risk_reasons', []))
                            st.session_state.last_prediction = result
                            st.session_state.last_risk = risk_info
                            st.markdown("### 🎯 Kết quả Dự đoán")
                            render_prediction_display(result, risk_info)
                    else:
                        st.error("❌ Không tìm thấy số 5 chữ số hợp lệ. Kiểm tra định dạng!")
            else:
                st.error("❌ Vui lòng nhập dữ liệu trước!")
        
        elif st.session_state.last_prediction and st.session_state.last_risk:
            st.markdown("### 🎯 Kết quả Dự đoán Gần nhất")
            render_prediction_display(st.session_state.last_prediction, st.session_state.last_risk)
            
            # Win tracking
            st.markdown("---")
            st.markdown("##### 🎲 Nhập kết quả thực tế để theo dõi Win Rate:")
            col1, col2 = st.columns([3, 1])
            with col1:
                actual_input = st.text_input("Kết quả thực tế (5 số)", key="actual_input", placeholder="Ví dụ: 12864")
            with col2:
                if st.button("✅ Ghi nhận", key="record_win"):
                    if actual_input and len(actual_input) == 5 and actual_input.isdigit():
                        if st.session_state.predictions_log:
                            st.session_state.predictions_log[0]['actual_result'] = actual_input
                            st.session_state.predictions_log[0]['won'] = check_win(
                                ''.join(st.session_state.last_prediction['main_3']), 
                                actual_input
                            )
                            status = '🎉 TRÚNG' if st.session_state.predictions_log[0]['won'] else '❌ Trượt'
                            st.success(f"✅ {status}")
                    else:
                        st.error("Nhập đúng 5 chữ số!")
    
    # ==================== TAB 2 ====================
    with tab2:
        st.header("🧠 Phân Tích Chi Tiết")
        
        if st.session_state.last_prediction:
            render_prediction_display(st.session_state.last_prediction, st.session_state.last_risk)
            render_analysis_details(st.session_state.last_prediction)
            
            if st.session_state.last_risk[1]:
                st.markdown("### ⚠️ Phân tích Rủi ro Chi tiết")
                for reason in st.session_state.last_risk[1]:
                    st.markdown(f"• {reason}")
        else:
            st.info("👈 Nhập dữ liệu ở Tab 1 để xem phân tích chi tiết")
    
    # ==================== TAB 3 ====================
    with tab3:
        render_stats_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#8b949e; font-size:12px; padding:20px;'>
        🧠 TITAN v34.0 ULTRA | Smart AI Prediction System<br>
        ⚠️ Công cụ hỗ trợ - Không đảm bảo 100% - Chơi có trách nhiệm
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()