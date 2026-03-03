# ==============================================================================
# FILE: app.py - VERSION 3.0 (NO CUSTOM HTML)
# TITAN v33.0 PRO - Pure Streamlit Components
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
    page_title="TITAN v33.0 PRO | 3 Số 5 Tinh",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS chỉ cho màu sắc cơ bản (không dùng HTML custom)
st.markdown("""
<style>
    .stApp {
        background-color: #010409;
        color: #e6edf3;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Number boxes */
    .num-display {
        font-size: 60px;
        font-weight: bold;
        color: #ff5858;
        text-align: center;
    }
    
    .lot-display {
        font-size: 45px;
        font-weight: bold;
        color: #58a6ff;
        text-align: center;
    }
    
    /* Status colors */
    .status-green {
        background-color: #238636;
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    
    .status-red {
        background-color: #da3633;
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    
    .status-yellow {
        background-color: #d29922;
        color: #0d1117;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    
    /* Cards */
    .prediction-card {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #58a6ff;
        color: white;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        padding: 12px 28px;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #0d1117;
        color: #e6edf3;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SECRETS & API MANAGEMENT
# ==============================================================================

@st.cache_resource
def init_gemini():
    """Initialize Gemini API safely."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    except:
        return None

def get_quota_status():
    """Manage daily API quota."""
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
    """Clean raw input and extract 5-digit numbers."""
    if not raw_text.strip():
        return [], {"found": 0, "new": 0, "duplicate_input": 0, "already_in_db": 0}
    
    # Regex để lọc số 5 chữ số
    matches = re.findall(r'\b\d{5}\b', raw_text)
    
    new_entries = []
    seen_in_input = set()
    db_set = set(existing_db)
    
    stats = {
        "found": len(matches),
        "new": 0,
        "duplicate_input": 0,
        "already_in_db": 0,
    }
    
    for match in matches:
        if match in seen_in_input:
            stats["duplicate_input"] += 1
            continue
        seen_in_input.add(match)
        
        if match in db_set:
            stats["already_in_db"] += 1
            continue
        
        new_entries.append(match)
        stats["new"] += 1
    
    return new_entries, stats

def add_to_database(new_numbers):
    """Add new numbers to database."""
    if "lottery_db" not in st.session_state:
        st.session_state.lottery_db = []
    
    st.session_state.lottery_db = new_numbers + st.session_state.lottery_db
    
    if len(st.session_state.lottery_db) > 3000:
        st.session_state.lottery_db = st.session_state.lottery_db[:3000]

def check_win(prediction_3, result_5):
    """Check if prediction wins (3 số 5 tinh)."""
    if len(prediction_3) != 3 or len(result_5) != 5:
        return False
    result_digits = set(result_5)
    return all(digit in result_digits for digit in prediction_3)

# ==============================================================================
# 4. CORE PREDICTION ALGORITHMS
# ==============================================================================

def calculate_risk(history, window=50):
    """Calculate risk score 0-100."""
    if len(history) < window:
        return 0, []
    
    recent = history[-window:]
    all_digits = ''.join(recent)
    freq = Counter(all_digits)
    reasons = []
    risk = 0
    
    # 1. Over-represented numbers
    total_slots = len(all_digits)
    if total_slots > 0:
        most_common = freq.most_common(1)[0]
        if most_common[1] > total_slots * 0.30:
            risk += 30
            reasons.append(f"Số '{most_common[0]}' xuất hiện quá nhiều ({most_common[1]} lần)")
    
    # 2. Abnormal streaks (cầu bệt)
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
        if max_streak >= 4:
            risk += 25
            reasons.append(f"Cầu bệt {max_streak} kỳ ở vị trí {pos}")
    
    # 3. Low entropy
    if len(all_digits) > 0:
        entropy = -sum((c/len(all_digits)) * np.log2(c/len(all_digits)) 
                      for c in freq.values() if c > 0)
        if entropy < 2.5:
            risk += 25
            reasons.append("Phân phối quá đều (entropy thấp)")
    
    # 4. Overly stable sums
    totals = [sum(int(d) for d in n) for n in recent if len(n) == 5]
    if len(totals) > 10:
        std_dev = np.std(totals)
        if std_dev < 2.0:
            risk += 20
            reasons.append(f"Tổng số quá ổn định (σ={std_dev:.2f})")
    
    return min(100, risk), reasons

def analyze_frequency(history, window=100):
    """Frequency analysis with weighting."""
    recent = history[-window:] if len(history) >= window else history
    
    weighted_freq = defaultdict(float)
    
    for idx, num in enumerate(recent):
        weight = 1.0 + (idx / max(len(recent), 1))
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
        'scores': {k: round(v, 2) for k, v in sorted_items[:10]}
    }

def analyze_positions(history, window=50):
    """Position-based analysis."""
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
        all_candidates.extend([x[0] for x in pos_freq[i].most_common(3)])
    
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

def analyze_hot_cold(history, recent_window=10, cold_window=15):
    """Hot/Cold analysis."""
    recent = history[-recent_window:] if len(history) >= recent_window else history
    older = history[-cold_window:-recent_window] if len(history) >= cold_window else []
    
    recent_digits = Counter(''.join(recent))
    older_digits = Counter(''.join(older)) if older else Counter()
    
    hot = [str(x[0]) for x in recent_digits.most_common(5)]
    
    all_recent = ''.join(recent)
    cold = [str(i) for i in range(10) if str(i) not in all_recent]
    
    due = []
    for num in cold:
        if older_digits.get(num, 0) >= 3:
            due.append(num)
    
    return {
        'hot': hot,
        'cold': cold,
        'due': due
    }

def detect_patterns(history, window=30):
    """Pattern detection."""
    recent = history[-window:] if len(history) >= window else history
    
    patterns = {
        'bet': [],
        'nhip2': [],
        'nhip3': [],
        'detected': [],
        'likely': []
    }
    
    # Cầu bệt
    for pos in range(5):
        seq = [n[pos] if len(n) > pos else '0' for n in recent]
        for i in range(len(seq) - 2):
            if seq[i] == seq[i+1] == seq[i+2]:
                digit = seq[i]
                if digit not in patterns['bet']:
                    patterns['bet'].append(digit)
                    patterns['detected'].append(f'Bệt vị {pos}: {digit}')
                    if digit not in patterns['likely']:
                        patterns['likely'].append(digit)
    
    # Cầu nhịp 2
    for pos in range(5):
        seq = [n[pos] if len(n) > pos else '0' for n in recent]
        for i in range(len(seq) - 2):
            if seq[i] == seq[i+2] and seq[i] != seq[i+1]:
                digit = seq[i]
                if digit not in patterns['nhip2']:
                    patterns['nhip2'].append(digit)
                    patterns['detected'].append(f'Nhịp-2 vị {pos}: {digit}')
                    if digit not in patterns['likely']:
                        patterns['likely'].append(digit)
    
    return patterns

def consensus_engine(stat_result, pos_result, hotcold_result, pattern_result):
    """Combine all methods with weighted voting."""
    all_votes = []
    
    # Frequency 40%
    for num in stat_result.get('top_3', []):
        all_votes.extend([num] * 4)
    
    # Position 30%
    for num in pos_result.get('top_3', []):
        all_votes.extend([num] * 3)
    
    # Hot/Cold 20%
    for num in hotcold_result.get('hot', [])[:3]:
        all_votes.extend([num] * 2)
    
    # Pattern 10%
    for num in pattern_result.get('likely', []):
        all_votes.append(num)
    
    if not all_votes:
        return None
    
    vote_count = Counter(all_votes)
    
    final_3 = []
    for num, count in vote_count.most_common():
        if num not in final_3:
            final_3.append(num)
        if len(final_3) == 3:
            break
    
    while len(final_3) < 3:
        for i in range(10):
            if str(i) not in final_3:
                final_3.append(str(i))
                break
    
    remaining = [n for n, c in vote_count.most_common(10) if n not in final_3]
    support_4 = remaining[:4]
    while len(support_4) < 4:
        for i in range(10):
            if str(i) not in final_3 and str(i) not in support_4:
                support_4.append(str(i))
                break
    
    if vote_count:
        top_vote = vote_count.most_common(1)[0][1]
        confidence = min(95, 50 + top_vote * 3)
    else:
        confidence = 50
    
    return {
        'main_3': final_3,
        'support_4': support_4,
        'confidence': confidence,
        'vote_breakdown': dict(vote_count.most_common(10))
    }

def predict_3_numbers(history):
    """Main prediction function."""
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
    
    risk_score, risk_reasons = calculate_risk(history)
    
    if risk_score >= 60:
        return {
            'main_3': ['0', '0', '0'],
            'support_4': ['0', '0', '0', '0'],
            'decision': 'DỪNG',
            'confidence': 95,
            'logic': f'Rủi ro cao ({risk_score}/100)',
            'risk_score': risk_score,
            'risk_reasons': risk_reasons
        }
    
    stat_result = analyze_frequency(history, window=100)
    pos_result = analyze_positions(history, window=50)
    hotcold_result = analyze_hot_cold(history)
    pattern_result = detect_patterns(history, window=30)
    
    consensus = consensus_engine(stat_result, pos_result, hotcold_result, pattern_result)
    
    if not consensus:
        return {'error': 'Không thể tạo dự đoán', 'risk_score': risk_score}
    
    if risk_score < 30 and consensus['confidence'] >= 70:
        decision = 'ĐÁNH'
    elif risk_score < 50:
        decision = 'THEO DÕI'
    else:
        decision = 'DỪNG'
    
    logic_parts = []
    if stat_result['top_3']:
        logic_parts.append(f"Tần suất: {','.join(stat_result['top_3'])}")
    if pattern_result['detected']:
        logic_parts.append(f"Pattern: {len(pattern_result['detected'])} phát hiện")
    
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
            'votes': consensus.get('vote_breakdown', {})
        }
    }

# ==============================================================================
# 5. WIN RATE TRACKING
# ==============================================================================

def log_prediction(prediction, actual_result=None):
    """Log prediction for tracking."""
    if "predictions_log" not in st.session_state:
        st.session_state.predictions_log = []
    
    entry = {
        'timestamp': datetime.now().isoformat(),
        'predicted': prediction.get('main_3', []),
        'confidence': prediction.get('confidence', 0),
        'decision': prediction.get('decision', ''),
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
# 6. UI COMPONENTS - PURE STREAMLIT (NO CUSTOM HTML)
# ==============================================================================

def render_prediction_display(result, risk_info):
    """Render prediction card using PURE STREAMLIT COMPONENTS."""
    risk_score, risk_reasons = risk_info
    
    if not result or 'main_3' not in result:
        st.warning("⚠️ Chưa có kết quả dự đoán.")
        return
    
    main_3 = result['main_3']
    support_4 = result.get('support_4', ['?']*4)
    
    main_3 = (main_3 + ['?']*3)[:3]
    support_4 = (support_4 + ['?']*4)[:4]
    
    # Status bar using Streamlit columns
    if result['decision'] == 'ĐÁNH':
        status_color = "🟢"
        status_bg = "status-green"
    elif result['decision'] == 'DỪNG':
        status_color = "🔴"
        status_bg = "status-red"
    else:
        status_color = "🟡"
        status_bg = "status-yellow"
    
    # Status bar
    st.markdown(f"""
    <div class="{status_bg}">
        {status_color} RISK: {risk_score}/100 | KHUYẾN NGHỊ: {result['decision']}
    </div>
    """, unsafe_allow_html=True)
    
    # 3 SỐ CHÍNH - Using columns for proper display
    st.markdown("<div style='text-align: center; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown(f"🔮 **3 SỐ CHÍNH** (Độ tin cậy: {result['confidence']}%)", unsafe_allow_html=True)
    
    # Display 3 main numbers in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='num-display'>{main_3[0]}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='num-display'>{main_3[1]}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='num-display'>{main_3[2]}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Divider
    st.markdown("---")
    
    # 4 SỐ LÓT
    st.markdown("<div style='text-align: center; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("🎲 **4 SỐ LÓT**", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='lot-display'>{support_4[0]}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='lot-display'>{support_4[1]}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='lot-display'>{support_4[2]}</div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='lot-display'>{support_4[3]}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Logic explanation in info box
    st.info(f"💡 **Logic:** {result['logic']}")
    
    # Risk warnings
    if risk_reasons:
        warning_text = "⚠️ **Cảnh báo:**\n"
        for reason in risk_reasons:
            warning_text += f"• {reason}\n"
        st.warning(warning_text)
    
    # Copy button with code block
    st.markdown("---")
    numbers_to_copy = ','.join(main_3 + support_4)
    st.code(numbers_to_copy, language=None)
    st.caption("📋 Bấm vào code trên để copy dàn 7 số")

def render_analysis_details(result):
    """Render detailed analysis."""
    if not result or 'details' not in result:
        return
    
    details = result['details']
    
    with st.expander("🔍 Chi tiết phân tích", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Phân tích Tần suất (40%)")
            if 'frequency' in details and 'scores' in details['frequency']:
                freq_df = pd.DataFrame(
                    list(details['frequency']['scores'].items()), 
                    columns=['Số', 'Điểm']
                ).sort_values('Điểm', ascending=False)
                st.bar_chart(freq_df.set_index('Số'))
                st.caption(f"Top 3: {', '.join(details['frequency'].get('top_3', []))}")
            
            st.markdown("### 📍 Phân tích Vị trí (30%)")
            if 'positions' in details and 'pos_top' in details['positions']:
                pos_data = {f'Vị {i}': details['positions']['pos_top'][i] for i in range(5)}
                st.json(pos_data)
        
        with col2:
            st.markdown("### 🔥 Số Nóng/Lạnh (20%)")
            if 'hot_cold' in details:
                hc = details['hot_cold']
                st.markdown(f"**🔥 Nóng:** {' '.join(hc.get('hot', [])[:5])}")
                st.markdown(f"**❄️ Lạnh:** {' '.join(hc.get('cold', [])[:5]) if hc.get('cold') else 'Không có'}")
                if hc.get('due'):
                    st.markdown(f"**⏰ Đến kỳ:** {' '.join(hc['due'])}")
            
            st.markdown("### 🔄 Pattern Phát hiện (10%)")
            if 'patterns' in details and details['patterns'].get('detected'):
                for p in details['patterns']['detected'][:5]:
                    st.markdown(f"• {p}")
            else:
                st.caption("Không phát hiện pattern rõ ràng")
        
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
        st.info("📭 Chưa có dữ liệu. Vui lòng nhập kết quả ở Tab 1.")
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
        st.dataframe(df_hist, use_container_width=True)

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
    st.caption("3 Số 5 Tinh | Thống kê thực tế > AI Hallucination")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Trạng thái Hệ thống")
        display_quota_warning()
        
        st.markdown("---")
        st.markdown("### 📋 Hướng dẫn nhanh")
        st.markdown("""
        1️⃣ Dán kết quả xổ số (5 số/dòng)  
        2️⃣ Nhấn "LƯU & PHÂN TÍCH"  
        3️⃣ Xem kết quả dự đoán ngay  
        4️⃣ Theo dõi win rate ở Tab 3
        """)
        
        st.markdown("---")
        st.warning("""
        ⚠️ Tool hỗ trợ phân tích  
        ⚠️ Risk >= 60: Nên DỪNG  
        ⚠️ Quản lý vốn cẩn thận
        """)
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Nhập & Dự đoán", "🔍 Phân tích Chi tiết", "📊 Thống kê"])
    
    # ==================== TAB 1 ====================
    with tab1:
        st.header("📥 Nhập Kết Quả & Dự đoán Nhanh")
        
        st.markdown("Dán kết quả xổ số (5 chữ số mỗi dòng). Hệ thống tự động làm sạch và phân tích.")
        
        input_text = st.text_area(
            "📋 Dữ liệu thô",
            height=180,
            placeholder="Ví dụ:\n12345\n67890\n54321\n...",
            key="input_area"
        )
        
        if st.button("🚀 LƯU & PHÂN TÍCH", type="primary"):
            if input_text.strip():
                with st.spinner("🔄 Đang xử lý: Làm sạch → Lưu → Phân tích → Dự đoán..."):
                    new_nums, stats = clean_lottery_data(input_text, st.session_state.lottery_db)
                    
                    if stats['new'] > 0:
                        add_to_database(new_nums)
                        st.success(f"✅ Thêm {stats['new']} số mới (Tìm thấy: {stats['found']} | Trùng: {stats['duplicate_input']} | Có trong DB: {stats['already_in_db']})")
                        
                        result = predict_3_numbers(st.session_state.lottery_db)
                        risk_info = (result.get('risk_score', 0), result.get('risk_reasons', []))
                        
                        log_prediction(result)
                        
                        st.session_state.last_prediction = result
                        st.session_state.last_risk = risk_info
                        
                        st.markdown("### 🎯 Kết quả Dự đoán")
                        render_prediction_display(result, risk_info)
                        
                    elif stats['found'] > 0:
                        st.warning(f"⚠️ Không có số mới (tất cả đã có trong DB)")
                        if len(st.session_state.lottery_db) >= 20:
                            result = predict_3_numbers(st.session_state.lottery_db)
                            risk_info = (result.get('risk_score', 0), result.get('risk_reasons', []))
                            st.session_state.last_prediction = result
                            st.session_state.last_risk = risk_info
                            st.markdown("### 🎯 Kết quả Dự đoán")
                            render_prediction_display(result, risk_info)
                    else:
                        st.error("❌ Không tìm thấy số 5 chữ số hợp lệ")
            else:
                st.error("❌ Vui lòng nhập dữ liệu!")
        
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
        st.header("🔍 Phân tích Chi tiết")
        
        if st.session_state.last_prediction:
            render_prediction_display(st.session_state.last_prediction, st.session_state.last_risk)
            render_analysis_details(st.session_state.last_prediction)
            
            if st.session_state.last_risk[1]:
                st.markdown("### ⚠️ Phân tích Rủi ro")
                for reason in st.session_state.last_risk[1]:
                    st.markdown(f"• {reason}")
        else:
            st.info("👈 Nhập dữ liệu ở Tab 1 để xem chi tiết")
    
    # ==================== TAB 3 ====================
    with tab3:
        render_stats_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align:center; color:#8b949e; font-size:12px; padding:20px;'>TITAN v33.0 PRO | Chơi có trách nhiệm</div>", unsafe_allow_html=True)

def display_quota_warning():
    """Display quota status in sidebar."""
    quota = get_quota_status()
    remaining = quota["limit"] - quota["used"]
    
    if remaining == 0:
        st.error(f"🚫 Gemini: HẾT ({quota['limit']}/{quota['limit']})")
    elif remaining <= 3:
        st.warning(f"⚠️ Gemini: {remaining}/{quota['limit']}")
    else:
        st.success(f"✅ Gemini: {remaining}/{quota['limit']}")

# ==============================================================================
# 8. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()