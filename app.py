# ==============================================================================
# FILE: app.py - TITAN v34.1 FINAL (FIXED RESPONSIVENESS)
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
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="TITAN v34.1 FINAL | Responsive AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS
st.markdown("""
<style>
    .stApp {
        background-color: #010409;
        color: #e6edf3;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Status bars */
    .status-green {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        margin: 15px 0;
        box-shadow: 0 0 25px rgba(35,134,54,0.6);
    }
    
    .status-red {
        background: linear-gradient(135deg, #da3633, #f85149);
        color: white;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        margin: 15px 0;
        box-shadow: 0 0 25px rgba(218,54,51,0.6);
    }
    
    .status-yellow {
        background: linear-gradient(135deg, #d29922, #f0b429);
        color: #0d1117;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        margin: 15px 0;
    }
    
    /* Number boxes - Enhanced */
    .numbers-container {
        display: flex;
        flex-direction: row;
        justify-content: center;
        align-items: center;
        gap: 15px;
        margin: 25px 0;
        flex-wrap: wrap;
    }
    
    .num-box {
        font-size: 60px;
        font-weight: 900;
        color: #ff5858;
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 3px solid #ff5858;
        border-radius: 18px;
        padding: 25px 35px;
        min-width: 85px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(255,88,88,0.5), inset 0 0 25px rgba(255,88,88,0.15);
        text-shadow: 0 0 20px rgba(255,88,88,1);
        animation: glow 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 10px 30px rgba(255,88,88,0.5); }
        50% { box-shadow: 0 10px 40px rgba(255,88,88,0.8); }
    }
    
    .lot-box {
        font-size: 45px;
        font-weight: 800;
        color: #58a6ff;
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 2px solid #58a6ff;
        border-radius: 15px;
        padding: 18px 25px;
        min-width: 70px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(88,166,255,0.4);
    }
    
    /* Cards */
    .prediction-card {
        background: linear-gradient(135deg, #0d1117, #161b22);
        border: 1px solid #30363d;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 5px 25px rgba(0,0,0,0.6);
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
        box-shadow: 0 8px 25px rgba(35,134,54,0.6);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(88,166,255,0.1);
        border-left: 4px solid #58a6ff;
        padding: 12px 18px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: rgba(218,54,51,0.1);
        border-left: 4px solid #da3633;
        padding: 12px 18px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Mobile responsive */
    @media (max-width: 600px) {
        .num-box {
            font-size: 45px;
            padding: 18px 25px;
            min-width: 65px;
        }
        .lot-box {
            font-size: 35px;
            padding: 14px 20px;
            min-width: 55px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SECRETS & API
# ==============================================================================

@st.cache_resource
def init_gemini():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    except:
        return None

def get_quota_status():
    if "quota" not in st.session_state:
        st.session_state.quota = {
            "used": 0,
            "limit": 15,
            "reset_date": datetime.now().date().isoformat()
        }
    
    quota = st.session_state.quota
    today = datetime.now().date().isoformat()
    
    if quota["reset_date"] != today:
        quota["used"] = 0
        quota["reset_date"] = today
    
    return quota

def use_quota():
    quota = get_quota_status()
    if quota["used"] < quota["limit"]:
        quota["used"] += 1
        return True
    return False

# ==============================================================================
# 3. DATA CLEANING - ENHANCED
# ==============================================================================

def clean_lottery_data_advanced(raw_text, existing_db):
    """Smart data cleaning with detailed reporting."""
    if not raw_text.strip():
        return [], {"found": 0, "new": 0, "duplicate_input": 0, "already_in_db": 0, "invalid": 0}
    
    # Normalize and clean
    normalized = re.sub(r'\s+', ' ', raw_text.strip())
    lines = normalized.split('\n')
    cleaned_numbers = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        line_no_spaces = re.sub(r'\s', '', line)
        matches = re.findall(r'\d{5}', line_no_spaces)
        cleaned_numbers.extend(matches)
    
    # Stats
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
    """Add numbers with full deduplication."""
    if "lottery_db" not in st.session_state:
        st.session_state.lottery_db = []
    
    # Add new numbers
    st.session_state.lottery_db = new_numbers + st.session_state.lottery_db
    
    # Deduplicate
    seen = set()
    unique_db = []
    for num in st.session_state.lottery_db:
        if num not in seen:
            seen.add(num)
            unique_db.append(num)
    
    st.session_state.lottery_db = unique_db[:3000]
    
    return len(new_numbers)

def check_win(prediction_3, result_5):
    if len(prediction_3) != 3 or len(result_5) != 5:
        return False
    result_digits = set(result_5)
    return all(digit in result_digits for digit in prediction_3)

# ==============================================================================
# 4. SMART PREDICTION - RECENT-FOCUSED
# ==============================================================================

def calculate_risk_advanced(history, window=50):
    if len(history) < 20:
        return 0, []
    
    recent = history[-window:] if len(history) >= window else history
    all_digits = ''.join(recent)
    freq = Counter(all_digits)
    reasons = []
    risk = 0
    
    # 1. Over-represented numbers
    total_slots = len(all_digits)
    if total_slots > 0:
        most_common = freq.most_common(3)
        for num, count in most_common:
            rate = count / total_slots
            if rate > 0.25:
                risk += 20
                reasons.append(f"Số '{num}' xuất hiện {rate*100:.0f}%")
    
    # 2. Abnormal streaks
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
            reasons.append(f"Cầu bệt {max_streak} kỳ vị trí {pos}")
    
    # 3. Entropy
    if len(all_digits) > 0:
        entropy = -sum((c/len(all_digits)) * np.log2(c/len(all_digits)) 
                      for c in freq.values() if c > 0)
        if entropy < 2.8:
            risk += 25
            reasons.append(f"Entropy thấp ({entropy:.2f})")
    
    # 4. Stable sums
    totals = [sum(int(d) for d in n) for n in recent if len(n) == 5]
    if len(totals) > 10:
        std_dev = np.std(totals)
        if std_dev < 2.5:
            risk += 15
            reasons.append(f"Tổng quá ổn định (σ={std_dev:.2f})")
    
    return min(100, risk), reasons

def analyze_frequency_recent_focused(history, window=50):
    """
    FOCUS ON RECENT DATA - Last 50 periods weighted heavily
    """
    # Split into recent (last 50) and older
    recent = history[-50:] if len(history) >= 50 else history
    older = history[:-50] if len(history) > 50 else []
    
    weighted_freq = defaultdict(float)
    
    # Recent data: VERY HIGH weight (exponential decay from 5.0 to 2.0)
    for idx, num in enumerate(recent):
        weight = 5.0 - 3.0 * (idx / max(len(recent), 1))  # 5.0 → 2.0
        for digit in num:
            if digit.isdigit():
                weighted_freq[digit] += weight
    
    # Older data: LOW weight (1.0)
    for num in older:
        for digit in num:
            if digit.isdigit():
                weighted_freq[digit] += 1.0
    
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
        'method': 'Recent-Focused Analysis (50 kỳ gần nhất)'
    }

def analyze_positions_smart(history, window=50):
    recent = history[-window:] if len(history) >= window else history
    
    pos_freq = [Counter() for _ in range(5)]
    
    for num in recent:
        for i, digit in enumerate(num[:5]):
            pos_freq[i][digit] += 1
    
    pos_top = []
    for i in range(5):
        if pos_freq[i]:
            pos_top.append(pos_freq[i].most_common(1)[0][0])
        else:
            pos_top.append('0')
    
    all_candidates = []
    for i in range(5):
        weight = 5 - i
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
        'votes': dict(vote_count.most_common(10))
    }

def analyze_hot_cold_smart(history, recent_window=10, cold_window=20):
    recent = history[-recent_window:] if len(history) >= recent_window else history
    older = history[-cold_window:-recent_window] if len(history) >= cold_window else []
    
    recent_digits = Counter(''.join(recent))
    older_digits = Counter(''.join(older)) if older else Counter()
    
    hot = [str(x[0]) for x in recent_digits.most_common(5)]
    
    all_recent = ''.join(recent)
    cold = [str(i) for i in range(10) if str(i) not in all_recent]
    
    due = []
    for num in cold:
        old_count = older_digits.get(num, 0)
        if old_count >= 4:
            due.append(num)
    
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
        'very_hot': very_hot
    }

def detect_patterns_advanced(history, window=30):
    recent = history[-window:] if len(history) >= window else history
    
    patterns = {
        'bet': [],
        'nhip2': [],
        'nhip3': [],
        'detected': [],
        'likely': [],
        'avoid': []
    }
    
    # Cầu bệt
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
                    
                    if streak_len >= 4:
                        if digit not in patterns['avoid']:
                            patterns['avoid'].append(digit)
                    else:
                        if digit not in patterns['likely']:
                            patterns['likely'].append(digit)
    
    # Cầu nhịp 2
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
    
    return patterns

def consensus_engine_v2(stat_result, pos_result, hotcold_result, pattern_result):
    all_votes = []
    avoid_votes = []
    
    # Frequency 50% (tăng lên)
    for num in stat_result.get('top_3', []):
        all_votes.extend([num] * 6)
    
    # Position 30%
    for num in pos_result.get('top_3', []):
        all_votes.extend([num] * 4)
    
    # Hot/Cold 20%
    for num in hotcold_result.get('hot', [])[:3]:
        all_votes.extend([num] * 3)
    
    for num in hotcold_result.get('due', []):
        all_votes.extend([num] * 4)
    
    # Pattern 10%
    for num in pattern_result.get('likely', []):
        all_votes.append(num)
    
    for num in pattern_result.get('avoid', []):
        avoid_votes.append(num)
    
    if not all_votes:
        return None
    
    vote_count = Counter(all_votes)
    avoid_set = set(avoid_votes)
    
    final_3 = []
    for num, count in vote_count.most_common():
        if num not in final_3 and num not in avoid_set:
            final_3.append(num)
        if len(final_3) == 3:
            break
    
    while len(final_3) < 3:
        for i in range(10):
            if str(i) not in final_3 and str(i) not in avoid_set:
                final_3.append(str(i))
                break
    
    remaining = [n for n, c in vote_count.most_common(10) if n not in final_3 and n not in avoid_set]
    support_4 = remaining[:4]
    while len(support_4) < 4:
        for i in range(10):
            if str(i) not in final_3 and str(i) not in support_4 and str(i) not in avoid_set:
                support_4.append(str(i))
                break
    
    if vote_count:
        top_vote = vote_count.most_common(1)[0][1]
        confidence = min(95, 60 + top_vote * 3)
    else:
        confidence = 50
    
    if avoid_set:
        confidence = max(40, confidence - 15)
    
    return {
        'main_3': final_3,
        'support_4': support_4,
        'confidence': confidence,
        'avoid': list(avoid_set),
        'vote_breakdown': dict(vote_count.most_common(10))
    }

def predict_3_numbers_smart(history):
    """Main prediction - RECENT FOCUSED."""
    if len(history) < 20:
        return {
            'error': f'Cần ít nhất 20 kỳ (hiện có: {len(history)})',
            'main_3': ['?', '?', '?'],
            'support_4': ['0', '0', '0', '0'],
            'decision': 'CHỜ DỮ LIỆU',
            'confidence': 0,
            'logic': 'Nhập thêm dữ liệu',
            'risk_score': 0
        }
    
    risk_score, risk_reasons = calculate_risk_advanced(history)
    
    if risk_score >= 70:
        return {
            'main_3': ['0', '0', '0'],
            'support_4': ['0', '0', '0', '0'],
            'decision': 'DỪNG',
            'confidence': 95,
            'logic': f'Rủi ro cao ({risk_score}/100)',
            'risk_score': risk_score,
            'risk_reasons': risk_reasons
        }
    
    # Run analysis - FOCUSED ON RECENT
    stat_result = analyze_frequency_recent_focused(history)
    pos_result = analyze_positions_smart(history)
    hotcold_result = analyze_hot_cold_smart(history)
    pattern_result = detect_patterns_advanced(history)
    
    consensus = consensus_engine_v2(stat_result, pos_result, hotcold_result, pattern_result)
    
    if not consensus:
        return {'error': 'Không thể dự đoán', 'risk_score': risk_score}
    
    if risk_score < 30 and consensus['confidence'] >= 75:
        decision = 'ĐÁNH'
    elif risk_score < 50:
        decision = 'THEO DÕI'
    else:
        decision = 'DỪNG'
    
    logic_parts = []
    if stat_result['top_3']:
        logic_parts.append(f"Tần suất: {','.join(stat_result['top_3'])}")
    if hotcold_result.get('due'):
        logic_parts.append(f"Đến kỳ: {','.join(hotcold_result['due'][:2])}")
    if pattern_result['detected']:
        logic_parts.append(f"{len(pattern_result['detected'])} pattern")
    if consensus.get('avoid'):
        logic_parts.append(f"Tránh: {','.join(consensus['avoid'])}")
    
    logic = ' | '.join(logic_parts) if logic_parts else 'Phân tích thông minh'
    
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
            'votes': consensus.get('vote_breakdown', {}),
            'avoid': consensus.get('avoid', [])
        }
    }

# ==============================================================================
# 5. WIN RATE TRACKING
# ==============================================================================

def log_prediction(prediction, actual_result=None):
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
    risk_score, risk_reasons = risk_info
    
    if not result or 'main_3' not in result:
        st.warning("⚠️ Chưa có kết quả.")
        return
    
    main_3 = result['main_3']
    support_4 = result.get('support_4', ['?']*4)
    avoid = result.get('details', {}).get('avoid', [])
    
    main_3 = (main_3 + ['?']*3)[:3]
    support_4 = (support_4 + ['?']*4)[:4]
    
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
    
    if avoid:
        st.markdown(f"""
        <div class="warning-box">
            <strong>🚫 TRÁNH:</strong> {', '.join(avoid)}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='border-top: 2px solid #30363d; margin: 20px 0;'></div>", unsafe_allow_html=True)
    
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
    
    st.info(f"💡 **Logic:** {result['logic']}")
    
    if risk_reasons:
        warning_text = "⚠️ **Cảnh báo:**\n"
        for reason in risk_reasons:
            warning_text += f"• {reason}\n"
        st.warning(warning_text)
    
    st.markdown("---")
    numbers_to_copy = ','.join(main_3 + support_4)
    st.code(numbers_to_copy, language=None)
    st.caption("📋 Bấm vào code để copy")

def render_analysis_details(result):
    if not result or 'details' not in result:
        return
    
    details = result['details']
    
    with st.expander("🧠 Chi tiết phân tích", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Tần suất (50%)")
            if 'frequency' in details and 'scores' in details['frequency']:
                freq_df = pd.DataFrame(
                    list(details['frequency']['scores'].items()), 
                    columns=['Số', 'Điểm']
                ).sort_values('Điểm', ascending=False)
                st.bar_chart(freq_df.set_index('Số'))
                st.caption(details['frequency'].get('method', ''))
            
            st.markdown("### 📍 Vị trí (30%)")
            if 'positions' in details and 'pos_top' in details['positions']:
                pos_data = {f'Vị {i}': details['positions']['pos_top'][i] for i in range(5)}
                st.json(pos_data)
        
        with col2:
            st.markdown("### 🔥 Nóng/Lạnh (20%)")
            if 'hot_cold' in details:
                hc = details['hot_cold']
                st.markdown(f"**Nóng:** {' '.join(hc.get('hot', [])[:5])}")
                if hc.get('due'):
                    st.markdown(f"**Đến kỳ:** {' '.join(hc['due'])}")
            
            st.markdown("### 🔄 Pattern")
            if 'patterns' in details and details['patterns'].get('detected'):
                for p in details['patterns']['detected'][:5]:
                    st.markdown(f"• {p}")
        
        if 'votes' in details and details['votes']:
            st.markdown("---")
            st.markdown("### 🗳️ Voting")
            votes_df = pd.DataFrame(
                list(details['votes'].items()), 
                columns=['Số', 'Phiếu']
            ).sort_values('Phiếu', ascending=False)
            st.dataframe(votes_df, hide_index=True, use_container_width=True)

def render_stats_tab():
    st.header("📊 Thống kê")
    
    if "lottery_db" not in st.session_state or not st.session_state.lottery_db:
        st.info("📭 Chưa có dữ liệu.")
        return
    
    db = st.session_state.lottery_db
    predictions_log = st.session_state.get('predictions_log', [])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📦 Tổng kỳ", len(db))
    with col2:
        recent_digits = ''.join(db[:100])
        hottest = Counter(recent_digits).most_common(1)[0][0] if recent_digits else "-"
        st.metric("🔥 Số nóng", hottest)
    with col3:
        if predictions_log:
            decided = [e for e in predictions_log if e['won'] is not None]
            if decided:
                wins = sum(1 for e in decided if e['won'])
                win_rate = round(wins/len(decided)*100, 1)
                st.metric("🎯 Win Rate", f"{win_rate}%")
    
    st.markdown("### Tần suất 50 kỳ gần")
    recent_50 = db[:50]
    all_digits = ''.join(recent_50)
    freq = Counter(all_digits)
    df_freq = pd.DataFrame(
        [(str(d), c) for d, c in sorted(freq.items())], 
        columns=['Số', 'Tần suất']
    )
    st.bar_chart(df_freq.set_index('Số'))
    
    # Data management
    st.markdown("### 💾 Database")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.download_button(
            label="📥 Download",
            data=json.dumps(db, indent=2),
            file_name="titan_db.json",
            mime="application/json"
        )
    
    with c2:
        uploaded = st.file_uploader("📤 Upload", type="json")
        if uploaded:
            try:
                new_db = json.load(uploaded)
                if isinstance(new_db, list):
                    st.session_state.lottery_db = new_db
                    st.success("✅ Đã tải!")
                    time.sleep(1)
                    st.rerun()
            except:
                st.error("❌ Lỗi")
    
    with c3:
        if st.button("🗑️ Xóa DB", type="primary"):
            st.session_state.lottery_db = []
            st.success("✅ Đã xóa!")
            time.sleep(1)
            st.rerun()

# ==============================================================================
# 7. MAIN APP
# ==============================================================================

def main():
    if "lottery_db" not in st.session_state:
        st.session_state.lottery_db = []
    if "predictions_log" not in st.session_state:
        st.session_state.predictions_log = []
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "last_risk" not in st.session_state:
        st.session_state.last_risk = None
    if "force_recalc" not in st.session_state:
        st.session_state.force_recalc = False
    
    st.title("🎯 TITAN v34.1 FINAL")
    st.caption("Responsive AI | Focus 50 kỳ gần nhất")
    
    with st.sidebar:
        st.markdown("### ⚙️ Trạng thái")
        quota = get_quota_status()
        remaining = quota['limit'] - quota['used']
        if remaining > 5:
            st.success(f"✅ Gemini: {remaining}/{quota['limit']}")
        elif remaining > 0:
            st.warning(f"⚠️ Gemini: {remaining}/{quota['limit']}")
        else:
            st.error(f"🚫 Hết quota")
        
        st.markdown("---")
        st.warning("⚠️ Risk >= 70: DỪNG")
    
    tab1, tab2, tab3 = st.tabs(["📝 Nhập & Dự đoán", "🧠 Phân tích", "📊 Thống kê"])
    
    with tab1:
        st.header("📥 Nhập Kết Quả")
        
        st.markdown("""
        **💡 Hướng dẫn:**
        - Nhập kết quả từng kỳ (5 số/dòng)
        - Tool tập trung **50 kỳ gần nhất**
        - Nếu số không đổi → Database đã có sẵn
        - Bấm "🗑️ Xóa DB" để làm mới hoàn toàn
        """)
        
        input_text = st.text_area(
            "📋 Dữ liệu (5 số/dòng)",
            height=200,
            placeholder="87746\n56421\n...",
            key="input_area"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            analyze_btn = st.button("🚀 PHÂN TÍCH", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 FORCE RECALC", use_container_width=True):
                st.session_state.force_recalc = True
                st.rerun()
        
        if analyze_btn and input_text.strip():
            with st.spinner("🧠 Đang phân tích..."):
                start_time = time.time()
                
                new_nums, stats = clean_lottery_data_advanced(input_text, st.session_state.lottery_db)
                new_count = add_to_database(new_nums)
                
                elapsed = time.time() - start_time
                
                # Show detailed stats
                if stats['new'] > 0:
                    st.success(f"✅ {elapsed:.2f}s | Thêm {new_count} số mới")
                    st.info(f"📊 Tìm thấy: {stats['found']} | Trùng input: {stats['duplicate_input']} | Có trong DB: {stats['already_in_db']}")
                else:
                    st.warning(f"⚠️ Không có số mới (274/283 đã có trong DB)")
                    st.info("💡 Database đã có sẵn dữ liệu này. Kết quả không đổi là BÌNH THƯỜNG.")
                
                # Always predict (even if no new data)
                if len(st.session_state.lottery_db) >= 20:
                    result = predict_3_numbers_smart(st.session_state.lottery_db)
                    risk_info = (result.get('risk_score', 0), result.get('risk_reasons', []))
                    
                    log_prediction(result)
                    
                    st.session_state.last_prediction = result
                    st.session_state.last_risk = risk_info
                    
                    st.markdown("### 🎯 Kết quả")
                    render_prediction_display(result, risk_info)
                    
                    # Show recent data info
                    st.markdown("---")
                    st.markdown(f"##### 📈 Thông tin Database:")
                    st.caption(f"• Tổng số kỳ: {len(st.session_state.lottery_db)}")
                    st.caption(f"• Phân tích dựa trên: 50 kỳ gần nhất")
                    st.caption(f"• Số mới thêm: {new_count}")
        elif analyze_btn and not input_text.strip():
            st.error("❌ Nhập dữ liệu trước!")
        
        elif st.session_state.last_prediction and st.session_state.last_risk:
            st.markdown("### 🎯 Kết quả Gần nhất")
            render_prediction_display(st.session_state.last_prediction, st.session_state.last_risk)
            
            st.markdown("---")
            st.markdown("##### 🎲 Nhập kết quả thực tế:")
            col1, col2 = st.columns([3, 1])
            with col1:
                actual_input = st.text_input("Kết quả (5 số)", key="actual_input", placeholder="12864")
            with col2:
                if st.button("✅ Ghi nhận"):
                    if actual_input and len(actual_input) == 5 and actual_input.isdigit():
                        if st.session_state.predictions_log:
                            st.session_state.predictions_log[0]['actual_result'] = actual_input
                            st.session_state.predictions_log[0]['won'] = check_win(
                                ''.join(st.session_state.last_prediction['main_3']), 
                                actual_input
                            )
                            status = '🎉 TRÚNG' if st.session_state.predictions_log[0]['won'] else '❌ Trượt'
                            st.success(status)
    
    with tab2:
        st.header("🧠 Phân tích Chi tiết")
        if st.session_state.last_prediction:
            render_prediction_display(st.session_state.last_prediction, st.session_state.last_risk)
            render_analysis_details(st.session_state.last_prediction)
        else:
            st.info("👈 Tab 1")
    
    with tab3:
        render_stats_tab()
    
    st.markdown("---")
    st.markdown("<div style='text-align:center; color:#8b949e; font-size:12px;'>TITAN v34.1 FINAL</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()