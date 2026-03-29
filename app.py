import streamlit as st
import re, pandas as pd, numpy as np, math
from collections import Counter
from itertools import combinations

# === CẤU HÌNH ===
LUCKY_OX = [0, 2, 5, 6, 7, 8]
st.set_page_config(page_title="TITAN V50", page_icon="⚡", layout="centered")

# === CSS MOBILE OPTIMIZED ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
    
    .stApp {
        background: #000000;
        color: #FFD700;
        font-family: 'Orbitron', monospace;
    }
    
    .main-header {
        font-size: 32px;
        font-weight: 900;
        text-align: center;
        color: #FFD700;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
        margin-bottom: 10px;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-around;
        margin: 10px 0;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border: 2px solid #00fff5;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        flex: 1;
        margin: 0 5px;
        min-width: 80px;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 3px solid #FFD700;
        text-align: center;
    }
    
    .big-number {
        font-size: 48px;
        font-weight: 900;
        letter-spacing: 10px;
        color: #00FFCC;
        text-shadow: 0 0 15px #00FFCC;
    }
    
    .medium-number {
        font-size: 36px;
        font-weight: 900;
        letter-spacing: 6px;
        color: #FFD700;
    }
    
    .score {
        font-size: 24px;
        font-weight: bold;
        color: #00ff40;
    }
    
    .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 10px;
        font-weight: bold;
        margin: 3px;
    }
    
    .tag-hot { background: #ff0040; color: white; }
    .tag-quantum { background: #9900ff; color: white; }
    .tag-safe { background: #00ff40; color: black; }
    
    .confidence-bar {
        height: 25px;
        background: #1a1a1a;
        border-radius: 12px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #000;
        font-size: 14px;
    }
    
    .history-table {
        font-size: 12px;
    }
    
    .win { color: #00ff40; font-weight: bold; }
    .lose { color: #ff0040; font-weight: bold; }
    
    button {
        font-weight: bold;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# === THUẬT TOÁN TINH GỌN ===

def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    nums = [n for n in re.findall(r"\d{5}", clean_text) if n]
    seen = set()
    return [n for n in nums if not (n in seen or seen.add(n))]

def calculate_all_scores(db):
    """Tính điểm cho tất cả cặp số - Tối ưu hóa"""
    if len(db) < 15:
        return []
    
    recent_str = "".join(db[-50:])
    single_pool = Counter(recent_str)
    pair_pool = Counter()
    
    for num in db[-50:]:
        for p in combinations(sorted(set(num)), 2):
            pair_pool[p] += 1
    
    # Tìm anchors
    anchors = [d for d, _ in Counter("".join(db[-20:])).most_common(3)]
    
    results = []
    for p in combinations("0123456789", 2):
        pair = "".join(p)
        score = 0
        
        # Frequency
        freq = pair_pool.get(p, 0)
        score += freq * 5
        
        # Gan calculation
        gan = 0
        for num in reversed(db):
            if not set(p).issubset(set(num)):
                gan += 1
            else:
                break
        
        # Streak calculation
        streak = 0
        for num in reversed(db):
            if set(p).issubset(set(num)):
                streak += 1
            else:
                break
        
        # Scoring logic (tối ưu)
        if 4 <= gan <= 10:
            score += 50  # Golden zone
        elif 1 <= gan <= 3:
            score += 25
        elif gan > 15:
            score -= 30
        
        if streak >= 3:
            score -= 60  # Anti-bet
        elif streak == 1:
            score += 30
        
        # Anchor bonus
        if p[0] in anchors or p[1] in anchors:
            score += 20
        
        # Lucky ox
        if any(int(d) in LUCKY_OX for d in p):
            score += 10
        
        # Digital root
        dr = (int(p[0]) + int(p[1])) % 9
        recent_dr = Counter(sum(int(d) for d in n) % 9 for n in db[-20:])
        if recent_dr and dr == recent_dr.most_common(1)[0][0]:
            score += 25
        
        results.append({
            'pair': pair,
            'score': score,
            'gan': gan,
            'streak': streak,
            'freq': freq
        })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)

def predict_v50(db):
    """Dự đoán tinh gọn"""
    if len(db) < 15:
        return None
    
    pairs = calculate_all_scores(db)
    top_pairs = pairs[:5]
    
    # Top 3 triples
    triples = []
    for t in combinations("0123456789", 3):
        score = 0
        for p in combinations(t, 2):
            for pair_data in pairs:
                if pair_data['pair'] == "".join(p):
                    score += pair_data['score']
                    break
        triples.append((''.join(t), score))
    
    triples.sort(key=lambda x: x[1], reverse=True)
    
    # Top 8
    single_pool = Counter("".join(db[-50:]))
    top8 = "".join([d for d, _ in single_pool.most_common(8)])
    
    # Confidence
    if top_pairs:
        top_score = top_pairs[0]['score']
        confidence = min(90, max(40, 40 + top_score / 5))
    else:
        confidence = 50
    
    # Crowd numbers
    crowd = [d for d, _ in Counter("".join(db[-10:])).most_common(3)]
    
    return {
        'pairs': top_pairs,
        'triples': triples[:3],
        'top8': top8,
        'confidence': confidence,
        'crowd': crowd
    }

# === GIAO DIỆN COMPACT ===

st.markdown('<h1 class="main-header">⚡ TITAN V50</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888; font-size:12px;">Compact Pro | Mobile Optimized</p>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_area("📥 Kết quả (kỳ mới nhất ở dưới):", height=100, 
                          placeholder="84890\n07119\n33627")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 SOI", type="primary", use_container_width=True):
        nums = get_nums(user_input)
        if len(nums) >= 15:
            if "last_pred" in st.session_state and nums:
                lp = st.session_state.last_pred
                last = nums[-1]
                if lp and lp['pairs']:
                    best = lp['pairs'][0]['pair']
                    win = all(d in last for d in best)
                    st.session_state.history.insert(0, {
                        'Kỳ': last, 'Dự đoán': best,
                        'KQ': '🔥' if win else '❌'
                    })
            
            st.session_state.last_pred = predict_v50(nums)
            st.rerun()
        else:
            st.warning(f"Cần 15+ kỳ (có {len(nums)})")

with col2:
    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# === HIỂN THỊ KẾT QUẢ ===

if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    # Metrics row
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div style="font-size:10px; color:#888;">TIN CẬY</div>
            <div style="font-size:24px; font-weight:900; color:#00fff5;">{res['confidence']:.0f}%</div>
        </div>
        <div class="metric-box">
            <div style="font-size:10px; color:#888;">TOP PAIR</div>
            <div style="font-size:24px; font-weight:900; color:#FFD700;">{res['pairs'][0]['pair']}</div>
        </div>
        <div class="metric-box">
            <div style="font-size:10px; color:#888;">HOT</div>
            <div style="font-size:20px; font-weight:900; color:#ff0040;">{','.join(res['crowd'])}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence bar
    st.markdown(f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width:{res['confidence']}%;">{res['confidence']:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top pair
    st.markdown("""<div style="text-align:center; margin:15px 0;">🎯 CẶP VIP</div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="prediction-box">
        <div class="big-number">{res['pairs'][0]['pair'][0]} - {res['pairs'][0]['pair'][1]}</div>
        <div class="score">Score: {res['pairs'][0]['score']:.0f}</div>
        <div style="margin-top:10px;">
            <span class="tag tag-safe">Gan: {res['pairs'][0]['gan']}</span>
            <span class="tag tag-hot">Bệt: {res['pairs'][0]['streak']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top 5 pairs (compact)
    st.markdown("""<div style="text-align:center; margin:15px 0;">🎯 TOP 5 PAIRS</div>""", unsafe_allow_html=True)
    
    for i, p in enumerate(res['pairs'][:5]):
        tags = ""
        if p['gan'] >= 4 and p['gan'] <= 10:
            tags += '<span class="tag tag-quantum">GAN VÀNG</span>'
        if p['streak'] >= 1:
            tags += '<span class="tag tag-hot">BỆT</span>'
        
        st.markdown(f"""
        <div style="background:#1a1a2e; border-radius:10px; padding:10px; margin:5px 0; 
                    border-left: 4px solid {'#00ff40' if i == 0 else '#888'};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:24px; font-weight:900; color:#FFD700;">{p['pair'][0]}-{p['pair'][1]}</span>
                <span style="font-size:18px; color:#00ff40;">{p['score']:.0f}</span>
            </div>
            <div style="font-size:10px; color:#888; margin-top:5px;">{tags}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Top 8
    st.markdown(f"""
    <div class="prediction-box" style="margin-top:15px;">
        <div style="font-size:12px; color:#888;">ĐỘ PHỦ SẢNH</div>
        <div class="medium-number">{res['top8']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # History
    if st.session_state.history:
        st.markdown("""<div style="text-align:center; margin:15px 0;">📋 LỊCH SỬ</div>""", unsafe_allow_html=True)
        
        df = pd.DataFrame(st.session_state.history[:10])
        
        def color_kq(val):
            return 'color: #00ff40' if '🔥' in val else 'color: #ff0040'
        
        st.dataframe(df.style.applymap(color_kq, subset=['KQ']), 
                     use_container_width=True, hide_index=True)
        
        # Win rate
        wins = sum(1 for h in st.session_state.history if '🔥' in h['KQ'])
        total = len(st.session_state.history)
        rate = (wins/total*100) if total > 0 else 0
        
        st.markdown(f"""
        <div class="metric-box" style="margin-top:10px;">
            <div style="font-size:12px; color:#888;">TỶ LỆ THẮNG</div>
            <div style="font-size:28px; font-weight:900; color:{'#00ff40' if rate >= 40 else '#ff0040'};">
                {rate:.0f}% ({wins}/{total})
            </div>
        </div>
        """, unsafe_allow_html=True)

# === FOOTER ===
st.markdown("""
<div style="text-align:center; color:#444; font-size:10px; margin-top:20px; padding-top:10px; border-top:1px solid #333;">
    TITAN V50 COMPACT PRO | Mobile Optimized
</div>
""", unsafe_allow_html=True)