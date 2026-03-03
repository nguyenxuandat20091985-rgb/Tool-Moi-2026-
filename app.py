# ==============================================================================
# FILE: app.py - FINAL VERSION (FIXED MOBILE LAYOUT)
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
    page_title="TITAN v33.0 PRO",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS với flexbox để hiển thị ngang
st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background-color: #010409;
        color: #e6edf3;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Status bars */
    .status-green {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .status-red {
        background: linear-gradient(135deg, #da3633, #f85149);
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .status-yellow {
        background: linear-gradient(135deg, #d29922, #f0b429);
        color: #0d1117;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    
    /* Number display - HORIZONTAL LAYOUT with flexbox */
    .numbers-container {
        display: flex;
        flex-direction: row;
        justify-content: center;
        align-items: center;
        gap: 15px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    
    .num-box {
        font-size: 60px;
        font-weight: bold;
        color: #ff5858;
        background: linear-gradient(135deg, #0d1117, #161b22);
        border: 2px solid #30363d;
        border-radius: 12px;
        padding: 15px 25px;
        min-width: 80px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(255,88,88,0.2);
    }
    
    .lot-box {
        font-size: 45px;
        font-weight: bold;
        color: #58a6ff;
        background: linear-gradient(135deg, #0d1117, #161b22);
        border: 2px solid #30363d;
        border-radius: 10px;
        padding: 12px 20px;
        min-width: 65px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(88,166,255,0.2);
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
        gap: 8px;
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
        width: 100%;
    }
    
    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #0d1117;
        color: #e6edf3;
        border: 1px solid #30363d;
    }
    
    /* Mobile responsive */
    @media (max-width: 600px) {
        .num-box {
            font-size: 45px;
            padding: 10px 15px;
            min-width: 60px;
        }
        .lot-box {
            font-size: 35px;
            padding: 8px 12px;
            min-width: 50px;
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
# 3. DATA FUNCTIONS
# ==============================================================================

def clean_lottery_data(raw_text, existing_db):
    if not raw_text.strip():
        return [], {"found": 0, "new": 0, "duplicate_input": 0, "already_in_db": 0}
    
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
    if "lottery_db" not in st.session_state:
        st.session_state.lottery_db = []
    
    st.session_state.lottery_db = new_numbers + st.session_state.lottery_db
    
    if len(st.session_state.lottery_db) > 3000:
        st.session_state.lottery_db = st.session_state.lottery_db[:3000]

def check_win(prediction_3, result_5):
    if len(prediction_3) != 3 or len(result_5) != 5:
        return False
    result_digits = set(result_5)
    return all(digit in result_digits for digit in prediction_3)

# ==============================================================================
# 4. PREDICTION ALGORITHMS
# ==============================================================================

def calculate_risk(history, window=50):
    if len(history) < window:
        return 0, []
    
    recent = history[-window:]
    all_digits = ''.join(recent)
    freq = Counter(all_digits)
    reasons = []
    risk = 0
    
    total_slots = len(all_digits)
    if total_slots > 0:
        most_common = freq.most_common(1)[0]
        if most_common[1] > total_slots * 0.30:
            risk += 30
            reasons.append(f"Số '{most_common[0]}' xuất hiện quá nhiều ({most_common[1]} lần)")
    
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
    
    if len(all_digits) > 0:
        entropy = -sum((c/len(all_digits)) * np.log2(c/len(all_digits)) 
                      for c in freq.values() if c > 0)
        if entropy < 2.5:
            risk += 25
            reasons.append("Phân phối quá đều (entropy thấp)")
    
    totals = [sum(int(d) for d in n) for n in recent if len(n) == 5]
    if len(totals) > 10:
        std_dev = np.std(totals)
        if std_dev < 2.0:
            risk += 20
            reasons.append(f"Tổng số quá ổn định (σ={std_dev:.2f})")
    
    return min(100, risk), reasons

def analyze_frequency(history, window=100):
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
    recent = history[-window:] if len(history) >= window else history
    
    patterns = {
        'bet': [],
        'nhip2': [],
        'detected': [],
        'likely': []
    }
    
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
    all_votes = []
    
    for num in stat_result.get('top_3', []):
        all_votes.extend([num] * 4)
    
    for num in pos_result.get('top_3', []):
        all_votes.extend([num] * 3)
    
    for num in hotcold_result.get('hot', [])[:3]:
        all_votes.extend([num] * 2)
    
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
# 6. UI COMPONENTS - FLEXBOX HORIZONTAL LAYOUT
# ==============================================================================

def render_prediction_display(result, risk_info):
    """Render prediction with HORIZONTAL number display using flexbox."""
    risk_score, risk_reasons = risk_info
    
    if not result or 'main_3' not in result:
        st.warning("⚠️ Chưa có kết quả dự đoán.")
        return
    
    main_3 = result['main_3']
    support_4 = result.get('support_4', ['?']*4)
    
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
    
    # 3 SỐ CHÍNH - HORIZONTAL with flexbox
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <div style="color: #8b949e; font-size: 14px; margin-bottom: 15px;">
            🔮 3 SỐ CHÍNH (Độ tin cậy: {}%)
        </div>
        <div class="numbers-container">
            <div class="num-box">{}</div>
            <div class="num-box">{}</div>
            <div class="num-box">{}</div>
        </div>
    </div>
    """.format(result['confidence'], main_3[0], main_3[1], main_3[2]), unsafe_allow_html=True)
    
    # Divider
    st.markdown("<div style='border-top: 1px solid #30363d; margin: 20px 0;'></div>", unsafe_allow_html=True)
    
    # 4 SỐ LÓT - HORIZONTAL with flexbox
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <div style="color: #8b949e; font-size: 13px; margin-bottom: 15px;">
            🎲 4 SỐ LÓT
        </div>
        <div class="numbers-container">
            <div class="lot-box">{}</div>
            <div class="lot-box">{}</div>
            <div class="lot-box">{}</div>
            <div class="lot-box">{}</div>
        </div>
    </div>
    """.format(support_4[0], support_4[1], support_4[2], support_4[3]), unsafe_allow_html=True)
    
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
    st.caption("📋 Bấm vào code trên để copy")

def render_analysis_details(result):
    if not result or 'details' not in result:
        return
    
    details = result['details']
    
    with st.expander("🔍 Chi tiết phân tích", expanded=False):
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
            st.markdown("### 🔥 Nóng/Lạnh (20%)")
            if 'hot_cold' in details:
                hc = details['hot_cold']
                st.markdown(f"**Nóng:** {' '.join(hc.get('hot', [])[:5])}")
                if hc.get('due'):
                    st.markdown(f"**Đến kỳ:** {' '.join(hc['due'])}")
            
            st.markdown("### 🔄 Pattern (10%)")
            if 'patterns' in details and details['patterns'].get('detected'):
                for p in details['patterns']['detected'][:5]:
                    st.markdown(f"• {p}")

def render_stats_tab():
    st.header("📊 Thống kê")
    
    if "lottery_db" not in st.session_state or not st.session_state.lottery_db:
        st.info("📭 Chưa có dữ liệu.")
        return
    
    db = st.session_state.lottery_db
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📦 Tổng kỳ", len(db))
    with col2:
        quota = get_quota_status()
        st.metric("🤖 Gemini", f"{quota['limit'] - quota['used']}/{quota['limit']}")
    
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
            file_name=f"titan_db.json",
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
                st.error("❌ Lỗi file")
    
    with c3:
        if st.button("🗑️ Xóa", type="primary"):
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
    
    st.title("🎰 TITAN v33.0 PRO")
    st.caption("3 Số 5 Tinh")
    
    with st.sidebar:
        st.markdown("### ⚙️ Trạng thái")
        quota = get_quota_status()
        st.write(f"🤖 Gemini: {quota['limit'] - quota['used']}/{quota['limit']}")
        st.markdown("---")
        st.warning("⚠️ Risk >= 60: DỪNG")
    
    tab1, tab2, tab3 = st.tabs(["📝 Nhập & Dự đoán", "🔍 Phân tích", "📊 Thống kê"])
    
    with tab1:
        st.header("📥 Nhập Kết Quả")
        
        input_text = st.text_area(
            "📋 Dữ liệu (5 số/dòng)",
            height=180,
            placeholder="12345\n67890\n..."
        )
        
        if st.button("🚀 LƯU & PHÂN TÍCH", type="primary"):
            if input_text.strip():
                with st.spinner("🔄 Đang xử lý..."):
                    new_nums, stats = clean_lottery_data(input_text, st.session_state.lottery_db)
                    
                    if stats['new'] > 0:
                        add_to_database(new_nums)
                        st.success(f"✅ Thêm {stats['new']} số mới")
                        
                        result = predict_3_numbers(st.session_state.lottery_db)
                        risk_info = (result.get('risk_score', 0), result.get('risk_reasons', []))
                        
                        log_prediction(result)
                        
                        st.session_state.last_prediction = result
                        st.session_state.last_risk = risk_info
                        
                        st.markdown("### 🎯 Kết quả")
                        render_prediction_display(result, risk_info)
                    else:
                        st.warning("⚠️ Không có số mới")
            else:
                st.error("❌ Nhập dữ liệu!")
        
        elif st.session_state.last_prediction:
            st.markdown("### 🎯 Kết quả Gần nhất")
            render_prediction_display(st.session_state.last_prediction, st.session_state.last_risk)
    
    with tab2:
        st.header("🔍 Phân tích")
        if st.session_state.last_prediction:
            render_prediction_display(st.session_state.last_prediction, st.session_state.last_risk)
            render_analysis_details(st.session_state.last_prediction)
        else:
            st.info("👈 Tab 1")
    
    with tab3:
        render_stats_tab()
    
    st.markdown("---")
    st.markdown("<div style='text-align:center; color:#8b949e; font-size:12px;'>TITAN v33.0 PRO</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()