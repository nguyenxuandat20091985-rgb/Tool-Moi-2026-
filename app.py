# ==============================================================================
# FILE: app.py
# TITAN v33.0 PRO - Multi-AI Lottery Prediction System
# Chiến lược: Thống kê thực tế > AI Hallucination
# ==============================================================================

import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import time

# ==============================================================================
# 1. CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="TITAN v33.0 PRO | 3 Số 5 Tinh",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Dark Theme v22 Style
st.markdown("""
<style>
    :root {
        --bg-primary: #010409;
        --bg-secondary: #0d1117;
        --border-color: #30363d;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --accent-red: #ff5858;
        --accent-blue: #58a6ff;
        --accent-green: #238636;
        --accent-yellow: #d29922;
        --accent-red-dark: #da3633;
    }
    
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Prediction Card */
    .prediction-card {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    
    /* Number Display Boxes */
    .num-box {
        font-size: 70px;
        font-weight: 800;
        color: var(--accent-red);
        letter-spacing: 10px;
        text-align: center;
        display: inline-block;
        margin: 0 8px;
        text-shadow: 0 0 20px rgba(255,88,88,0.3);
    }
    
    .lot-box {
        font-size: 50px;
        font-weight: 700;
        color: var(--accent-blue);
        letter-spacing: 5px;
        text-align: center;
        display: inline-block;
        margin: 0 5px;
    }
    
    /* Status Bar */
    .status-bar {
        padding: 12px 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: 700;
        font-size: 16px;
        margin: 10px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .status-green { 
        background: linear-gradient(135deg, var(--accent-green), #2ea043); 
        color: white; 
    }
    .status-red { 
        background: linear-gradient(135deg, var(--accent-red-dark), #f85149); 
        color: white; 
    }
    .status-yellow { 
        background: linear-gradient(135deg, var(--accent-yellow), #f0b429); 
        color: #0d1117; 
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-secondary);
        padding: 4px;
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: var(--text-secondary);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-blue);
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-green), #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 12px 28px;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(35,134,54,0.4);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Metric Cards */
    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--accent-blue);
    }
    .metric-label {
        font-size: 12px;
        color: var(--text-secondary);
        text-transform: uppercase;
    }
    
    /* Analysis Section */
    .analysis-section {
        background: var(--bg-secondary);
        border-left: 4px solid var(--accent-blue);
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Win Rate Badge */
    .winrate-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }
    .winrate-high { background: rgba(35,134,54,0.3); color: #3fb950; }
    .winrate-med { background: rgba(210,153,34,0.3); color: #d29922; }
    .winrate-low { background: rgba(218,54,51,0.3); color: #f85149; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SECRETS & API MANAGEMENT
# ==============================================================================

@st.cache_resource
def init_gemini():
    """Initialize Gemini API safely from secrets."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    except KeyError:
        return None
    except Exception:
        return None

def get_quota_status():
    """Manage daily API quota (15 requests/day for free tier)."""
    if "quota" not in st.session_state:
        st.session_state.quota = {
            "used": 0,
            "limit": 15,
            "reset_date": datetime.now().date().isoformat()
        }
    
    quota = st.session_state.quota
    today = datetime.now().date().isoformat()
    
    # Reset quota on new day
    if quota["reset_date"] != today:
        quota["used"] = 0
        quota["reset_date"] = today
    
    return quota

def use_quota():
    """Consume one API quota unit."""
    quota = get_quota_status()
    if quota["used"] < quota["limit"]:
        quota["used"] += 1
        return True
    return False

# ==============================================================================
# 3. DATA CLEANING & MANAGEMENT
# ==============================================================================

def clean_lottery_data(raw_text, existing_db):
    """
    Clean raw input: extract 5-digit numbers, remove duplicates, filter existing.
    
    Returns:
        tuple: (new_numbers_list, stats_dict)
    """
    if not raw_text.strip():
        return [], {"found": 0, "new": 0, "duplicate_input": 0, "already_in_db": 0, "invalid": 0}
    
    # Extract all 5-digit sequences using regex
    matches = re.findall(r'\b\d{5}\b', raw_text)
    
    new_entries = []
    seen_in_input = set()
    db_set = set(existing_db)
    
    stats = {
        "found": len(matches),
        "new": 0,
        "duplicate_input": 0,
        "already_in_db": 0,
        "invalid": 0
    }
    
    for match in matches:
        # Skip if duplicate in same input
        if match in seen_in_input:
            stats["duplicate_input"] += 1
            continue
        seen_in_input.add(match)
        
        # Skip if already in database
        if match in db_set:
            stats["already_in_db"] += 1
            continue
        
        # Valid new entry
        new_entries.append(match)
        stats["new"] += 1
    
    return new_entries, stats

def add_to_database(new_numbers):
    """Add new numbers to session state database (newest first, max 3000)."""
    if "lottery_db" not in st.session_state:
        st.session_state.lottery_db = []
    
    # Prepend new numbers (most recent first)
    st.session_state.lottery_db = new_numbers + st.session_state.lottery_db
    
    # Limit to 3000 most recent draws
    if len(st.session_state.lottery_db) > 3000:
        st.session_state.lottery_db = st.session_state.lottery_db[:3000]

def check_win(prediction_3, result_5):
    """
    Check if 3-number prediction wins against 5-digit result.
    Win condition: result contains ALL 3 predicted numbers (any position).
    """
    if len(prediction_3) != 3 or len(result_5) != 5:
        return False
    result_digits = set(result_5)
    return all(digit in result_digits for digit in prediction_3)

# ==============================================================================
# 4. CORE PREDICTION ALGORITHMS
# ==============================================================================

def calculate_risk(history, window=50):
    """
    Calculate risk score 0-100 based on pattern anomalies.
    Higher score = higher risk = should stop betting.
    """
    if len(history) < window:
        return 0, []
    
    recent = history[-window:]
    all_digits = ''.join(recent)
    freq = Counter(all_digits)
    reasons = []
    risk = 0
    
    # 1. Over-represented numbers (>40% appearance rate)
    total_slots = len(all_digits)
    if total_slots > 0:
        most_common = freq.most_common(1)[0]
        if most_common[1] > total_slots * 0.4:
            risk += 30
            reasons.append(f"⚠️ Số '{most_common[0]}' xuất hiện quá nhiều ({most_common[1]} lần)")
    
    # 2. Abnormal streaks (same number, same position, 5+ consecutive)
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
            risk += 25
            reasons.append(f"⚠️ Cầu bệt {max_streak} kỳ ở vị trí {pos}")
    
    # 3. Low entropy (too uniform = potentially manipulated)
    if len(all_digits) > 0:
        entropy = -sum((c/len(all_digits)) * np.log2(c/len(all_digits)) 
                      for c in freq.values() if c > 0)
        if entropy < 2.5:
            risk += 25
            reasons.append("⚠️ Phân phối quá đều (entropy thấp) - Có thể bị điều khiển")
    
    # 4. Overly stable sum totals
    totals = [sum(int(d) for d in n) for n in recent if len(n) == 5]
    if len(totals) > 10:
        std_dev = np.std(totals)
        if std_dev < 2.0:
            risk += 20
            reasons.append(f"⚠️ Tổng số quá ổn định (σ={std_dev:.2f})")
    
    return min(100, risk), reasons

def analyze_frequency(history, window=100):
    """
    Frequency analysis with recency weighting.
    Recent draws have 2x weight of older draws.
    """
    recent = history[-window:] if len(history) >= window else history
    
    weighted_freq = defaultdict(float)
    
    for idx, num in enumerate(recent):
        # Weight: recent = ~2.0, older = ~1.0
        weight = 1.0 + (idx / max(len(recent), 1))
        for digit in num:
            if digit.isdigit():
                weighted_freq[digit] += weight
    
    # Get top 3, exclude numbers that appeared 3+ consecutive recent draws
    sorted_items = sorted(weighted_freq.items(), key=lambda x: x[1], reverse=True)
    top_3 = []
    
    for digit, score in sorted_items:
        if len(top_3) >= 3:
            break
        # Check if this digit appeared in last 3 draws (avoid breaking streaks)
        last_3 = ''.join(history[:3]) if len(history) >= 3 else ''
        if last_3.count(digit) < 3:  # Only exclude if appeared 3x in last 3 draws
            top_3.append(digit)
    
    return {
        'top_3': top_3,
        'freq_dict': dict(weighted_freq),
        'scores': {k: round(v, 2) for k, v in sorted_items[:10]}
    }

def analyze_positions(history, window=50):
    """
    Position-based analysis: analyze each of 5 positions separately.
    """
    recent = history[-window:] if len(history) >= window else history
    
    pos_freq = [Counter() for _ in range(5)]
    
    for num in recent:
        for i, digit in enumerate(num[:5]):
            pos_freq[i][digit] += 1
    
    # Most common digit at each position
    pos_top = []
    for i in range(5):
        if pos_freq[i]:
            pos_top.append(pos_freq[i].most_common(1)[0][0])
        else:
            pos_top.append('0')
    
    # Collect candidates: top 3 from each position
    all_candidates = []
    for i in range(5):
        all_candidates.extend([x[0] for x in pos_freq[i].most_common(3)])
    
    # Vote to get top 3 overall
    vote_count = Counter(all_candidates)
    top_3 = [str(x[0]) for x in vote_count.most_common(3)]
    
    return {
        'top_3': top_3,
        'pos_top': pos_top,
        'pos_freq': [dict(cf.most_common(5)) for cf in pos_freq],
        'votes': dict(vote_count.most_common(10))
    }

def analyze_hot_cold(history, recent_window=10, cold_window=15):
    """
    Hot/Cold/Due number analysis.
    """
    recent = history[-recent_window:] if len(history) >= recent_window else history
    older = history[-cold_window:-recent_window] if len(history) >= cold_window else []
    
    recent_digits = Counter(''.join(recent))
    older_digits = Counter(''.join(older)) if older else Counter()
    
    # Hot: appeared frequently in recent 10 draws
    hot = [str(x[0]) for x in recent_digits.most_common(5)]
    
    # Cold: didn't appear in recent 15 draws
    all_recent = ''.join(recent)
    cold = [str(i) for i in range(10) if str(i) not in all_recent]
    
    # Due: cold numbers that were hot in older period (due to return)
    due = []
    for num in cold:
        if older_digits.get(num, 0) >= 3:
            due.append(num)
    
    return {
        'hot': hot,
        'cold': cold,
        'due': due,
        'recent_counts': dict(recent_digits)
    }

def detect_patterns(history, window=30):
    """
    Detect real patterns: streaks, rhythms, reversals.
    """
    recent = history[-window:] if len(history) >= window else history
    
    patterns = {
        'bet': [],      # Streak patterns
        'nhip2': [],    # Rhythm-2: X _ X _ X
        'nhip3': [],    # Rhythm-3: X _ _ X _ _ X
        'dao': [],      # Reversal: AB -> BA
        'detected': [],
        'likely': []    # Numbers likely to appear next
    }
    
    # 1. Streak detection (same number, same position, 3+ consecutive)
    for pos in range(5):
        seq = [n[pos] if len(n) > pos else '0' for n in recent]
        for i in range(len(seq) - 2):
            if seq[i] == seq[i+1] == seq[i+2]:
                digit = seq[i]
                if digit not in patterns['bet']:
                    patterns['bet'].append(digit)
                    patterns['detected'].append(f'📊 Bệt vị {pos}: {digit} (3+ kỳ)')
                    # Streaks often continue to 4-5, so include as candidate
                    if digit not in patterns['likely']:
                        patterns['likely'].append(digit)
    
    # 2. Rhythm-2: X _ X pattern
    for pos in range(5):
        seq = [n[pos] if len(n) > pos else '0' for n in recent]
        for i in range(len(seq) - 2):
            if seq[i] == seq[i+2] and seq[i] != seq[i+1]:
                digit = seq[i]
                if digit not in patterns['nhip2']:
                    patterns['nhip2'].append(digit)
                    patterns['detected'].append(f'🔄 Nhịp-2 vị {pos}: {digit}')
                    if digit not in patterns['likely']:
                        patterns['likely'].append(digit)
    
    # 3. Rhythm-3: X _ _ X pattern
    for pos in range(5):
        seq = [n[pos] if len(n) > pos else '0' for n in recent]
        for i in range(len(seq) - 3):
            if seq[i] == seq[i+3]:
                digit = seq[i]
                if digit not in patterns['nhip3']:
                    patterns['nhip3'].append(digit)
                    patterns['detected'].append(f'🔁 Nhịp-3 vị {pos}: {digit}')
                    if digit not in patterns['likely']:
                        patterns['likely'].append(digit)
    
    # 4. Reversal pattern: AB -> BA in adjacent positions
    for i in range(len(recent) - 1):
        curr, next_draw = recent[i], recent[i+1]
        if len(curr) >= 2 and len(next_draw) >= 2:
            if curr[0:2] == next_draw[1::-1]:  # AB -> BA
                patterns['dao'].extend([curr[0], curr[1]])
                patterns['detected'].append(f'↔️ Đảo cặp: {curr[0:2]} → {next_draw[0:2]}')
    
    return patterns

def ai_gemini_deep_analysis(history, model):
    """
    Gemini AI deep analysis with structured JSON output.
    """
    if not model:
        return {"error": "AI not initialized", "fallback": True}
    
    quota = get_quota_status()
    if quota["used"] >= quota["limit"]:
        return {"error": "Quota exceeded", "fallback": True}
    
    try:
        # Prepare context: last 50 draws
        context = history[:50] if len(history) >= 50 else history
        
        prompt = f"""
Bạn là chuyên gia phân tích xổ số. Phân tích dãy số 5 chữ số sau (mới nhất ở đầu):
{', '.join(context)}

Nhiệm vụ: Dự đoán 3 số (0-9) có khả năng xuất hiện cao nhất trong kết quả tiếp theo.
Yêu cầu trả về JSON STRICT format sau (tiếng Việt cho logic):

{{
    "main_3": ["1", "2", "6"],
    "support_4": ["3", "4", "7", "9"],
    "decision": "ĐÁNH" hoặc "THEO DÕI" hoặc "DỪNG",
    "confidence": 75,
    "logic": "Giải thích ngắn gọn lý do chọn 3 số này",
    "method": "Gemini Pattern Analysis"
}}

Chỉ trả về JSON thuần, không markdown, không giải thích thêm.
"""
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Robust JSON parsing
        try:
            result = json.loads(text)
        except:
            # Try extract JSON from markdown code block
            match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
            if match:
                result = json.loads(match.group(1))
            else:
                # Try find first { and last }
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    result = json.loads(text[start:end])
                else:
                    raise ValueError("No valid JSON found")
        
        use_quota()
        return result
        
    except Exception as e:
        return {"error": f"AI Error: {str(e)}", "fallback": True}

def consensus_engine(stat_result, pos_result, hotcold_result, pattern_result, gemini_result=None):
    """
    Combine all analysis methods with weighted voting.
    Weights: Frequency 40%, Position 30%, Hot/Cold 20%, Pattern 10%
    """
    all_votes = []
    
    # Frequency analysis (40% weight = 4 votes per number)
    for num in stat_result.get('top_3', []):
        all_votes.extend([num] * 4)
    
    # Position analysis (30% weight = 3 votes)
    for num in pos_result.get('top_3', []):
        all_votes.extend([num] * 3)
    
    # Hot numbers (20% weight = 2 votes)
    for num in hotcold_result.get('hot', [])[:3]:
        all_votes.extend([num] * 2)
    
    # Pattern detected numbers (10% weight = 1 vote)
    for num in pattern_result.get('likely', []):
        all_votes.append(num)
    
    # Gemini AI suggestion (if available and valid, add 5 bonus votes)
    if gemini_result and 'main_3' in gemini_result and not gemini_result.get('error'):
        for num in gemini_result['main_3']:
            all_votes.extend([num] * 5)
    
    if not all_votes:
        return None
    
    # Vote counting
    vote_count = Counter(all_votes)
    
    # Get top 3 unique numbers
    final_3 = []
    for num, count in vote_count.most_common():
        if num not in final_3:
            final_3.append(num)
        if len(final_3) == 3:
            break
    
    # Fill remaining if needed
    while len(final_3) < 3:
        for i in range(10):
            if str(i) not in final_3:
                final_3.append(str(i))
                break
    
    # Support 4: next most voted excluding main 3
    remaining = [n for n, c in vote_count.most_common(10) if n not in final_3]
    support_4 = remaining[:4]
    while len(support_4) < 4:
        for i in range(10):
            if str(i) not in final_3 and str(i) not in support_4:
                support_4.append(str(i))
                break
    
    # Calculate confidence based on vote consensus
    if vote_count:
        top_vote = vote_count.most_common(1)[0][1]
        confidence = min(95, 50 + top_vote * 3)
    else:
        confidence = 50
    
    return {
        'main_3': final_3,
        'support_4': support_4,
        'confidence': confidence,
        'method': 'Consensus Multi-Method',
        'vote_breakdown': dict(vote_count.most_common(10))
    }

def predict_3_numbers(history, model=None):
    """
    Main prediction function: combines all methods.
    
    Returns dict with prediction, decision, confidence, risk, and details.
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
    
    # 1. Risk Detection FIRST
    risk_score, risk_reasons = calculate_risk(history)
    
    if risk_score >= 60:
        return {
            'main_3': ['0', '0', '0'],
            'support_4': ['0', '0', '0', '0'],
            'decision': 'DỪNG',
            'confidence': 95,
            'logic': f'⚠️ Rủi ro cao ({risk_score}/100): ' + '; '.join(risk_reasons[:2]),
            'risk_score': risk_score,
            'risk_reasons': risk_reasons
        }
    
    # 2. Run all analysis methods
    stat_result = analyze_frequency(history, window=100)
    pos_result = analyze_positions(history, window=50)
    hotcold_result = analyze_hot_cold(history)
    pattern_result = detect_patterns(history, window=30)
    
    # 3. Gemini AI (if available and quota permits)
    gemini_result = None
    if model and get_quota_status()["used"] < get_quota_status()["limit"]:
        gemini_result = ai_gemini_deep_analysis(history, model)
    
    # 4. Consensus Engine
    consensus = consensus_engine(stat_result, pos_result, hotcold_result, pattern_result, gemini_result)
    
    if not consensus:
        return {'error': 'Không thể tạo dự đoán', 'risk_score': risk_score}
    
    # 5. Final Decision Logic
    if risk_score < 30 and consensus['confidence'] >= 70:
        decision = 'ĐÁNH'
    elif risk_score < 50:
        decision = 'THEO DÕI'
    else:
        decision = 'DỪNG'
    
    # 6. Build Logic Explanation
    logic_parts = []
    if stat_result['top_3']:
        logic_parts.append(f"📊 Tần suất: {','.join(stat_result['top_3'])}")
    if pattern_result['detected']:
        logic_parts.append(f"🔄 Pattern: {len(pattern_result['detected'])} phát hiện")
    if hotcold_result['due']:
        logic_parts.append(f"⏰ Số đến kỳ: {','.join(hotcold_result['due'][:2])}")
    
    logic = ' | '.join(logic_parts) if logic_parts else 'Phân tích đa phương pháp'
    
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
            'gemini': gemini_result if gemini_result and not gemini_result.get('error') else None,
            'votes': consensus.get('vote_breakdown', {})
        }
    }

# ==============================================================================
# 5. WIN RATE TRACKING
# ==============================================================================

def log_prediction(prediction, actual_result=None):
    """Log a prediction for win rate tracking."""
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
        'won': None  # Will be set when actual_result is provided
    }
    
    if actual_result and len(actual_result) == 5 and len(prediction.get('main_3', [])) == 3:
        entry['won'] = check_win(''.join(prediction['main_3']), actual_result)
    
    st.session_state.predictions_log.insert(0, entry)
    
    # Keep only last 200 predictions
    if len(st.session_state.predictions_log) > 200:
        st.session_state.predictions_log = st.session_state.predictions_log[:200]

def calculate_win_rate(log_entries):
    """Calculate overall and segmented win rates."""
    if not log_entries:
        return {'overall': 0, 'by_confidence': {}, 'total': 0, 'wins': 0}
    
    # Filter entries with known outcomes
    decided = [e for e in log_entries if e['won'] is not None]
    
    if not decided:
        return {'overall': 0, 'by_confidence': {}, 'total': 0, 'wins': 0}
    
    total = len(decided)
    wins = sum(1 for e in decided if e['won'])
    
    # By confidence bracket
    by_conf = {}
    for bracket in ['<50', '50-69', '70-84', '85+']:
        if bracket == '<50':
            subset = [e for e in decided if e['confidence'] < 50]
        elif bracket == '50-69':
            subset = [e for e in decided if 50 <= e['confidence'] < 70]
        elif bracket == '70-84':
            subset = [e for e in decided if 70 <= e['confidence'] < 85]
        else:
            subset = [e for e in decided if e['confidence'] >= 85]
        
        if subset:
            w = sum(1 for e in subset if e['won'])
            by_conf[bracket] = {'rate': round(w/len(subset)*100), 'count': len(subset)}
    
    return {
        'overall': round(wins/total*100, 1),
        'by_confidence': by_conf,
        'total': total,
        'wins': wins
    }

# ==============================================================================
# 6. UI COMPONENTS
# ==============================================================================

def render_prediction_display(result, risk_info):
    """Render the main prediction card with styling."""
    risk_score, risk_reasons = risk_info
    
    if not result or 'main_3' not in result:
        st.warning("⚠️ Chưa có kết quả dự đoán. Vui lòng nhập dữ liệu và phân tích.")
        return
    
    main_3 = result['main_3']
    support_4 = result.get('support_4', ['?']*4)
    
    # Ensure proper length
    main_3 = (main_3 + ['?']*3)[:3]
    support_4 = (support_4 + ['?']*4)[:4]
    
    # Status color
    if result['decision'] == 'ĐÁNH':
        status_class = 'status-green'
        status_icon = '✅'
    elif result['decision'] == 'DỪNG':
        status_class = 'status-red'
        status_icon = '🛑'
    else:
        status_class = 'status-yellow'
        status_icon = '⚠️'
    
    # Render card
    st.markdown(f"""
    <div class="prediction-card">
        <div class="status-bar {status_class}">
            {status_icon} RISK: {risk_score}/100 | KHUYẾN NGHỊ: {result['decision']}
        </div>
        
        <div style="text-align:center; margin:15px 0;">
            <div style="color:{st.theme.secondaryTextColor if hasattr(st, 'theme') else '#8b949e'}; font-size:14px; margin-bottom:8px;">
                🔮 3 SỐ CHÍNH (Độ tin cậy: {result['confidence']}%)
            </div>
            <span class="num-box">{main_3[0]}</span>
            <span class="num-box">{main_3[1]}</span>
            <span class="num-box">{main_3[2]}</span>
        </div>
        
        <div style="text-align:center; padding-top:15px; border-top:1px solid var(--border-color);">
            <div style="color:{st.theme.secondaryTextColor if hasattr(st, 'theme') else '#8b949e'}; font-size:13px; margin-bottom:8px;">
                🎲 4 SỐ LÓT
            </div>
            <span class="lot-box">{support_4[0]}</span>
            <span class="lot-box">{support_4[1]}</span>
            <span class="lot-box">{support_4[2]}</span>
            <span class="lot-box">{support_4[3]}</span>
        </div>
        
        <div style="margin-top:15px; padding:12px; background:rgba(13,17,23,0.8); border-radius:8px; font-size:13px;">
            <strong>💡 Logic:</strong> {result['logic']}
            {f'<br><strong>⚠️ Cảnh báo:</strong> ' + '<br>• '.join(risk_reasons) if risk_reasons else ''}
        </div>
        
        <div style="text-align:right; margin-top:12px;">
            <button style="background:var(--border-color); border:none; color:var(--text-primary); 
                          padding:8px 16px; border-radius:6px; cursor:pointer; font-size:13px;"
                    onclick="navigator.clipboard.writeText('{','.join(main_3 + support_4)}')">
                📋 Copy Dàn 7 Số
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_analysis_details(result):
    """Render detailed breakdown of each analysis method."""
    if not result or 'details' not in result:
        return
    
    details = result['details']
    
    with st.expander("🔍 Chi tiết phân tích từng phương pháp", expanded=False):
        cols = st.columns(2)
        
        with cols[0]:
            # Frequency Analysis
            st.subheader("📊 Phân tích Tần suất (40%)")
            if 'frequency' in details and 'scores' in details['frequency']:
                freq_df = pd.DataFrame(
                    list(details['frequency']['scores'].items()), 
                    columns=['Số', 'Điểm']
                ).sort_values('Điểm', ascending=False)
                st.bar_chart(freq_df.set_index('Số'))
                st.caption(f"Top 3: {', '.join(details['frequency'].get('top_3', []))}")
            
            # Position Analysis
            st.subheader("📍 Phân tích Vị trí (30%)")
            if 'positions' in details and 'pos_top' in details['positions']:
                pos_data = {f'Vị {i}': details['positions']['pos_top'][i] for i in range(5)}
                st.json(pos_data)
        
        with cols[1]:
            # Hot/Cold Analysis
            st.subheader("🔥 Số Nóng/Lạnh (20%)")
            if 'hot_cold' in details:
                hc = details['hot_cold']
                st.markdown(f"**Nóng:** {' '.join(hc.get('hot', [])[:5])}")
                st.markdown(f"**Lạnh:** {' '.join(hc.get('cold', [])[:5]) if hc.get('cold') else 'Không có'}")
                if hc.get('due'):
                    st.markdown(f"**Đến kỳ:** {' '.join(hc['due'])} ⏰")
            
            # Pattern Detection
            st.subheader("🔄 Pattern Phát hiện (10%)")
            if 'patterns' in details and details['patterns'].get('detected'):
                for p in details['patterns']['detected'][:5]:
                    st.markdown(f"• {p}")
            else:
                st.caption("Không phát hiện pattern rõ ràng")
        
        # Gemini AI Result
        if details.get('gemini'):
            st.divider()
            st.subheader("🤖 Gemini AI Analysis")
            gem = details['gemini']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{gem.get('confidence', 0)}%")
                st.metric("Decision", gem.get('decision', 'N/A'))
            with col2:
                st.markdown(f"*{gem.get('logic', 'No explanation')}*")
        
        # Vote Breakdown
        if 'votes' in details and details['votes']:
            st.divider()
            st.subheader("🗳️ Consensus Voting")
            votes_df = pd.DataFrame(
                list(details['votes'].items()), 
                columns=['Số', 'Phiếu']
            ).sort_values('Phiếu', ascending=False)
            st.dataframe(votes_df, hide_index=True, use_container_width=True)

def render_stats_tab():
    """Render statistics and database management tab."""
    st.header("📊 Thống kê & Quản lý Database")
    
    if "lottery_db" not in st.session_state or not st.session_state.lottery_db:
        st.info("📭 Chưa có dữ liệu. Vui lòng nhập kết quả ở Tab 1.")
        return
    
    db = st.session_state.lottery_db
    predictions_log = st.session_state.get('predictions_log', [])
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Tổng kỳ", len(db))
    with col2:
        recent_digits = ''.join(db[:100])
        st.metric("🔥 Số nóng nhất", Counter(recent_digits).most_common(1)[0][0] if recent_digits else "-")
    with col3:
        win_stats = calculate_win_rate(predictions_log)
        st.metric("🎯 Win Rate", f"{win_stats['overall']}%")
    with col4:
        quota = get_quota_status()
        st.metric("🤖 Gemini Quota", f"{quota['limit'] - quota['used']}/{quota['limit']}")
    
    # Charts
    tab_chart1, tab_chart2 = st.tabs(["📈 Biểu đồ", "📋 Dữ liệu"])
    
    with tab_chart1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tần suất 50 kỳ gần")
            recent_50 = db[:50]
            all_digits = ''.join(recent_50)
            freq = Counter(all_digits)
            df_freq = pd.DataFrame(
                [(str(d), c) for d, c in sorted(freq.items())], 
                columns=['Số', 'Tần suất']
            )
            st.bar_chart(df_freq.set_index('Số'), color="#58a6ff")
        
        with col2:
            st.subheader("Top 5 Số Nóng (Toàn bộ)")
            all_digits_full = ''.join(db)
            freq_full = Counter(all_digits_full)
            top_5 = freq_full.most_common(5)
            
            hot_html = "<div style='display:flex; gap:8px; flex-wrap:wrap;'>"
            for num, count in top_5:
                hot_html += f"""
                <div style='background:var(--bg-secondary); border:1px solid var(--border-color); 
                           padding:12px 20px; border-radius:8px; text-align:center; min-width:60px;'>
                    <div style='font-size:24px; font-weight:700; color:var(--accent-red);'>{num}</div>
                    <div style='font-size:11px; color:var(--text-secondary);'>{count} lần</div>
                </div>"""
            hot_html += "</div>"
            st.markdown(hot_html, unsafe_allow_html=True)
    
    with tab_chart2:
        # Data Management
        st.subheader("💾 Quản lý Database")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            # Download
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(db, indent=2, ensure_ascii=False),
                file_name=f"titan_db_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with c2:
            # Upload
            uploaded = st.file_uploader("📤 Upload JSON", type="json", key="uploader")
            if uploaded:
                try:
                    new_db = json.load(uploaded)
                    if isinstance(new_db, list) and all(len(str(x))==5 for x in new_db):
                        st.session_state.lottery_db = new_db
                        st.success("✅ Đã tải database thành công!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ File không hợp lệ: cần danh sách số 5 chữ số")
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
        
        with c3:
            # Clear
            if st.button("🗑️ Xóa toàn bộ", type="primary", use_container_width=True):
                st.session_state.lottery_db = []
                st.success("✅ Đã xóa dữ liệu!")
                time.sleep(1)
                st.rerun()
        
        # Recent History Table
        st.subheader("📜 Lịch sử 20 kỳ gần nhất")
        df_hist = pd.DataFrame(db[:20], columns=["Kết Quả"])
        df_hist.index = [f"#{i+1}" for i in range(len(df_hist))]
        st.dataframe(df_hist, use_container_width=True, hide_index=False)
    
    # Win Rate Analysis
    if predictions_log:
        st.divider()
        st.subheader("🎯 Phân tích Hiệu quả Dự đoán")
        
        win_stats = calculate_win_rate(predictions_log)
        
        # Overall
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{'#3fb950' if win_stats['overall']>=50 else '#f85149'}">
                    {win_stats['overall']}%
                </div>
                <div class="metric-label">Win Rate Tổng</div>
                <div style="font-size:11px; color:var(--text-secondary); margin-top:5px;">
                    {win_stats['wins']}/{win_stats['total']} lần đánh
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # By confidence
            if win_stats['by_confidence']:
                st.markdown("**Win Rate theo Confidence:**")
                for bracket, data in win_stats['by_confidence'].items():
                    color = '#3fb950' if data['rate'] >= 50 else '#f85149'
                    st.markdown(f"""
                    <span style="display:inline-block; margin:2px 5px 2px 0; padding:3px 10px; 
                                background:rgba(13,17,23,0.8); border-radius:4px; font-size:12px;">
                        <strong style="color:{color}">{bracket}%</strong>: {data['rate']}% ({data['count']})
                    </span>
                    """, unsafe_allow_html=True)
        
        # Recent predictions table
        st.subheader("📋 Lịch sử Dự đoán Gần đây")
        if predictions_log:
            log_df = pd.DataFrame([
                {
                    'Thời gian': e['timestamp'][:16].replace('T', ' '),
                    'Dự đoán': ','.join(e['predicted']),
                    'Kết quả': e['actual_result'] or '-',
                    '✅ Trúng': '✓' if e['won'] else '✗' if e['won'] is False else '?',
                    'Confidence': f"{e['confidence']}%",
                    'Quyết định': e['decision']
                }
                for e in predictions_log[:15]
            ])
            st.dataframe(log_df, use_container_width=True, hide_index=True)

# ==============================================================================
# 7. MAIN APPLICATION
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
    st.title("🎰 TITAN v33.0 PRO")
    st.caption("Multi-AI Lottery Prediction | 3 Số 5 Tinh | Thống kê thực tế > AI Hallucination")
    
    # Sidebar: Quick Info
    with st.sidebar:
        st.markdown("### ⚙️ Trạng thái Hệ thống")
        
        quota = get_quota_status()
        st.progress(quota["used"] / quota["limit"])
        st.caption(f"🤖 Gemini: {quota['used']}/{quota['limit']} requests hôm nay")
        
        st.markdown("---")
        st.markdown("### 📋 Hướng dẫn nhanh")
        st.markdown("""
        1. Dán kết quả xổ số (5 số/dòng)
        2. Nhấn "LƯU & PHÂN TÍCH"
        3. Xem kết quả dự đoán ngay
        4. Theo dõi win rate ở Tab 3
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Lưu ý quan trọng")
        st.warning("""
        • Tool hỗ trợ phân tích, không đảm bảo trúng 100%
        • Luôn quản lý vốn và biết điểm dừng
        • Risk Score >= 60: Nên DỪNG đánh
        """)
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Nhập & Dự đoán", "🔍 Phân tích Chi tiết", "📊 Thống kê"])
    
    # ==================== TAB 1: INPUT & QUICK RESULT ====================
    with tab1:
        st.header("📥 Nhập Kết Quả & Dự đoán Nhanh")
        
        # Input area
        st.markdown("Dán kết quả xổ số (5 chữ số mỗi dòng). Hệ thống tự động làm sạch và phân tích.")
        
        input_text = st.text_area(
            "📋 Dữ liệu thô",
            height=180,
            placeholder="Ví dụ:\n12345\n67890\n54321\n83959\n...",
            key="input_area"
        )
        
        # Action button
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            analyze_btn = st.button("🚀 LƯU & PHÂN TÍCH", type="primary", use_container_width=True)
        
        # Processing
        if analyze_btn and input_text.strip():
            with st.spinner("🔄 Đang xử lý: Làm sạch dữ liệu → Phân tích → Dự đoán..."):
                # 1. Clean and add data
                new_nums, stats = clean_lottery_data(input_text, st.session_state.lottery_db)
                
                if stats['new'] > 0:
                    add_to_database(new_nums)
                    st.success(f"✅ Thêm {stats['new']} số mới (Tìm thấy: {stats['found']} | Trùng input: {stats['duplicate_input']} | Có trong DB: {stats['already_in_db']})")
                    
                    # 2. Run prediction
                    model = init_gemini()
                    result = predict_3_numbers(st.session_state.lottery_db, model)
                    risk_info = (result.get('risk_score', 0), result.get('risk_reasons', []))
                    
                    # 3. Log prediction
                    log_prediction(result)
                    
                    # 4. Save to state
                    st.session_state.last_prediction = result
                    st.session_state.last_risk = risk_info
                    
                    # 5. Show result immediately
                    st.markdown("### 🎯 Kết quả Dự đoán")
                    render_prediction_display(result, risk_info)
                    
                elif stats['found'] > 0:
                    st.warning(f"⚠️ Không có số mới nào được thêm (tất cả đã có trong database)")
                    # Still show prediction if we have enough data
                    if len(st.session_state.lottery_db) >= 20:
                        model = init_gemini()
                        result = predict_3_numbers(st.session_state.lottery_db, model)
                        risk_info = (result.get('risk_score', 0), result.get('risk_reasons', []))
                        st.session_state.last_prediction = result
                        st.session_state.last_risk = risk_info
                        st.markdown("### 🎯 Kết quả Dự đoán (từ dữ liệu hiện có)")
                        render_prediction_display(result, risk_info)
                else:
                    st.error("❌ Không tìm thấy số 5 chữ số hợp lệ. Kiểm tra định dạng đầu vào.")
        
        elif analyze_btn and not input_text.strip():
            st.error("❌ Vui lòng nhập dữ liệu trước khi phân tích!")
        
        # Show last result if exists (for persistent view)
        elif st.session_state.last_prediction and st.session_state.last_risk:
            st.markdown("### 🎯 Kết quả Dự đoán Gần nhất")
            render_prediction_display(st.session_state.last_prediction, st.session_state.last_risk)
            
            # Quick win tracking
            st.markdown("---")
            st.markdown("##### 🎲 Nhập kết quả thực tế để theo dõi Win Rate:")
            col_real1, col_real2 = st.columns([3, 1])
            with col_real1:
                actual_input = st.text_input("Kết quả thực tế (5 số)", key="actual_input", placeholder="Ví dụ: 12864")
            with col_real2:
                if st.button("✅ Ghi nhận", key="record_win"):
                    if actual_input and len(actual_input) == 5 and actual_input.isdigit():
                        # Update last prediction with actual result
                        if st.session_state.predictions_log:
                            st.session_state.predictions_log[0]['actual_result'] = actual_input
                            st.session_state.predictions_log[0]['won'] = check_win(
                                ''.join(st.session_state.last_prediction['main_3']), 
                                actual_input
                            )
                            st.success(f"✅ Đã ghi nhận: {'🎉 TRÚNG' if st.session_state.predictions_log[0]['won'] else '❌ Trượt'}")
                    else:
                        st.error("Nhập đúng 5 chữ số!")
    
    # ==================== TAB 2: DETAILED ANALYSIS ====================
    with tab2:
        st.header("🔍 Phân tích Chi tiết")
        
        if st.session_state.last_prediction:
            # Show main prediction summary
            render_prediction_display(st.session_state.last_prediction, st.session_state.last_risk)
            
            # Detailed breakdown
            render_analysis_details(st.session_state.last_prediction)
            
            # Risk explanation
            if st.session_state.last_risk[1]:
                st.markdown("### ⚠️ Phân tích Rủi ro")
                for reason in st.session_state.last_risk[1]:
                    st.markdown(f"• {reason}")
        else:
            st.info("👈 Vui lòng nhập dữ liệu và phân tích ở Tab 1 để xem chi tiết.")
    
    # ==================== TAB 3: STATISTICS ====================
    with tab3:
        render_stats_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:var(--text-secondary); font-size:12px; padding:20px;">
        🔮 TITAN v33.0 PRO | Multi-AI Lottery Prediction System<br>
        ⚠️ Công cụ hỗ trợ phân tích - Không đảm bảo kết quả - Hãy chơi có trách nhiệm
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 8. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()