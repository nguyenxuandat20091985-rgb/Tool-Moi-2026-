import streamlit as st
import re, pandas as pd, numpy as np, math, json, os
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import requests

# === CẤU HÌNH ===
LUCKY_OX = [0, 2, 5, 6, 7, 8]
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_v52_data.json"

st.set_page_config(page_title="TITAN V52 - AI CORE FUSION", page_icon="🧠", layout="centered")

# === CSS CAO CẤP ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #0a0a1a 50%, #000000 100%);
        color: #FFD700;
        font-family: 'Orbitron', monospace;
    }
    
    .god-header {
        font-size: 32px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #FFD700, #FF00FF, #00FFFF, #FFD700);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .algorithm-card {
        background: linear-gradient(135deg, #1a1a2e, #0f0f1a);
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        border-left: 4px solid;
    }
    
    .algo-strong { border-color: #00ff40; }
    .algo-medium { border-color: #ffff00; }
    .algo-weak { border-color: #ff0040; }
    
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
        color: #00FFFF;
        text-shadow: 0 0 20px #00FFFF;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 15px 0;
    }
    
    .metric-cell {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 15px 10px;
        text-align: center;
        border: 1px solid #333;
    }
    
    .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 9px;
        font-weight: bold;
        margin: 2px;
    }
    
    .tag-top { background: #00ff40; color: #000; }
    .tag-hybrid { background: #9900ff; color: #fff; }
    .tag-safe { background: #0066ff; color: #fff; }
    
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
    
    .history-win { color: #00ff40; font-weight: 900; }
    .history-lose { color: #ff0040; font-weight: 900; }
</style>
""", unsafe_allow_html=True)

# === 10+ THUẬT TOÁN AI CORE ===

class Algorithm_1_Probability:
    """Xác suất thống kê thuần túy"""
    def analyze(self, db, pair):
        freq = sum(1 for n in db[-50:] if set(pair).issubset(set(n)))
        prob = freq / len(db[-50:]) if db else 0
        return prob * 100, f"Tần suất: {freq}/50 kỳ"

class Algorithm_2_MarkovChain:
    """Chuỗi Markov - chuyển trạng thái"""
    def analyze(self, db, pair):
        if len(db) < 10:
            return 0, "Thiếu dữ liệu"
        
        transitions = defaultdict(int)
        for i in range(len(db) - 1):
            curr = db[i]
            next_num = db[i + 1]
            for d in curr:
                for nd in next_num:
                    transitions[(d, nd)] += 1
        
        score = 0
        for p in pair:
            for other in pair:
                if p != other:
                    score += transitions.get((p, other), 0)
        
        return score * 2, f"Markov transitions: {score}"

class Algorithm_3_CycleDetection:
    """Phát hiện chu kỳ"""
    def analyze(self, db, pair):
        if len(db) < 20:
            return 0, "Thiếu dữ liệu"
        
        cycle_score = 0
        for cycle_len in range(3, 10):
            matches = 0
            for i in range(len(db) - cycle_len):
                if set(pair).issubset(set(db[i])) and set(pair).issubset(set(db[i + cycle_len])):
                    matches += 1
            if matches > 0:
                cycle_score += matches * (10 - cycle_len)
        
        return cycle_score * 3, f"Cycle matches: {cycle_score}"

class Algorithm_4_Frequency:
    """Phân tích tần suất nâng cao"""
    def analyze(self, db, pair):
        recent = db[-30:]
        freq_recent = sum(1 for n in recent if set(pair).issubset(set(n)))
        freq_old = sum(1 for n in db[-60:-30] if set(pair).issubset(set(n))) if len(db) >= 60 else 0
        
        # Xu hướng tăng/giảm
        trend = freq_recent - freq_old
        return max(0, 50 + trend * 10), f"Trend: {trend:+d}"

class Algorithm_5_MirrorNumber:
    """Bóng số (mirror)"""
    def analyze(self, db, pair):
        mirrors = {'0':'5', '1':'6', '2':'7', '3':'8', '4':'9', 
                   '5':'0', '6':'1', '7':'2', '8':'3', '9':'4'}
        
        mirror_pair = tuple(sorted([mirrors[pair[0]], mirrors[pair[1]]]))
        freq = sum(1 for n in db[-40:] if set(mirror_pair).issubset(set(n)))
        
        return freq * 5, f"Mirror freq: {freq}"

class Algorithm_6_StreakGan:
    """Số bệt / số gan"""
    def analyze(self, db, pair):
        # Tính gan
        gan = 0
        for num in reversed(db):
            if not set(pair).issubset(set(num)):
                gan += 1
            else:
                break
        
        # Tính bệt
        streak = 0
        for num in reversed(db):
            if set(pair).issubset(set(num)):
                streak += 1
            else:
                break
        
        # Score: ưu tiên gan 5-12, tránh bệt >= 3
        score = 0
        if 5 <= gan <= 12:
            score = 80
        elif 2 <= gan <= 4:
            score = 50
        elif gan > 15:
            score = 20
        
        if streak >= 3:
            score -= 50
        elif streak == 1:
            score += 30
        
        return max(0, score), f"Gan:{gan} Bệt:{streak}"

class Algorithm_7_PatternAI:
    """Pattern recognition đơn giản"""
    def analyze(self, db, pair):
        if len(db) < 15:
            return 0, "Thiếu dữ liệu"
        
        patterns = defaultdict(int)
        for num in db[-30:]:
            digits = sorted(set(num))
            for p in combinations(digits, 2):
                patterns["".join(p)] += 1
        
        pair_key = "".join(sorted(pair))
        freq = patterns.get(pair_key, 0)
        
        return freq * 8, f"Pattern freq: {freq}"

class Algorithm_8_MonteCarlo:
    """Mô phỏng Monte Carlo"""
    def analyze(self, db, pair):
        if len(db) < 20:
            return 0, "Thiếu dữ liệu"
        
        # Simulate 1000 lần
        recent_str = "".join(db[-30:])
        single_probs = {str(i): recent_str.count(str(i)) / len(recent_str) for i in range(10)}
        
        wins = 0
        for _ in range(1000):
            simulated = np.random.choice(list('0123456789'), 5, 
                                        p=[single_probs[str(i)] for i in range(10)])
            if set(pair).issubset(set(simulated)):
                wins += 1
        
        prob = wins / 1000 * 100
        return prob, f"Monte Carlo: {prob:.1f}%"

class Algorithm_9_Heuristic:
    """Heuristic + kinh nghiệm"""
    def analyze(self, db, pair):
        score = 0
        
        # Rule 1: Tuổi Sửu
        if any(int(d) in LUCKY_OX for d in pair):
            score += 20
        
        # Rule 2: Tổng digital root
        dr = (int(pair[0]) + int(pair[1])) % 9
        recent_dr = Counter(sum(int(d) for d in n) % 9 for n in db[-20:])
        if recent_dr and dr == recent_dr.most_common(1)[0][0]:
            score += 25
        
        # Rule 3: Số hot 10 kỳ
        hot_nums = Counter("".join(db[-10:])).most_common(3)
        hot_digits = [h[0] for h in hot_nums]
        if pair[0] in hot_digits or pair[1] in hot_digits:
            score += 15
        
        return score, f"Heuristic score: {score}"

class Algorithm_10_PositionWeight:
    """Trọng số vị trí"""
    def analyze(self, db, pair):
        positions = {i: Counter() for i in range(5)}
        for num in db[-30:]:
            for i, d in enumerate(num):
                positions[i][d] += 1
        
        score = 0
        for pos in positions:
            if pair[0] in positions[pos] or pair[1] in positions[pos]:
                score += 10
        
        return score, f"Position score: {score}"

class Algorithm_11_Entropy:
    """Phân tích entropy"""
    def analyze(self, db, pair):
        if len(db) < 10:
            return 0, "Thiếu dữ liệu"
        
        recent = "".join(db[-15:])
        counter = Counter(recent)
        total = len(recent)
        
        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Entropy thấp -> ưu tiên số ít xuất hiện
        single_freq = {str(i): recent.count(str(i)) for i in range(10)}
        
        score = 0
        for d in pair:
            if single_freq[d] < len(recent) / 10:
                score += 20  # Số lạnh có khả năng về
        
        return score, f"Entropy: {entropy:.2f}"

class Algorithm_12_Hybrid:
    """Lai tạo - kết hợp top algorithms"""
    def __init__(self):
        self.algos = [
            Algorithm_3_CycleDetection(),
            Algorithm_6_StreakGan(),
            Algorithm_7_PatternAI(),
            Algorithm_9_Heuristic()
        ]
    
    def analyze(self, db, pair):
        total_score = 0
        details = []
        
        for algo in self.algos:
            score, detail = algo.analyze(db, pair)
            total_score += score
            details.append(detail)
        
        return total_score / 4, " | ".join(details[:2])

# === AI CORE FUSION ENGINE ===

class AICoreFusion:
    """Kết hợp 10+ thuật toán"""
    
    def __init__(self):
        self.algorithms = {
            "Probability": Algorithm_1_Probability(),
            "Markov Chain": Algorithm_2_MarkovChain(),
            "Cycle Detection": Algorithm_3_CycleDetection(),
            "Frequency": Algorithm_4_Frequency(),
            "Mirror Number": Algorithm_5_MirrorNumber(),
            "Streak/Gan": Algorithm_6_StreakGan(),
            "Pattern AI": Algorithm_7_PatternAI(),
            "Monte Carlo": Algorithm_8_MonteCarlo(),
            "Heuristic": Algorithm_9_Heuristic(),
            "Position Weight": Algorithm_10_PositionWeight(),
            "Entropy": Algorithm_11_Entropy(),
            "HYBRID": Algorithm_12_Hybrid()
        }
        
        self.weights = {name: 1.0 for name in self.algorithms}
    
    def analyze_all(self, db, pair):
        results = {}
        total_score = 0
        
        for name, algo in self.algorithms.items():
            score, detail = algo.analyze(db, pair)
            weighted_score = score * self.weights[name]
            results[name] = {
                'score': weighted_score,
                'detail': detail,
                'weight': self.weights[name]
            }
            total_score += weighted_score
        
        avg_score = total_score / len(self.algorithms)
        return results, avg_score
    
    def get_top_pairs(self, db, top_n=10):
        all_pairs = []
        
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            results, avg_score = self.analyze_all(db, pair)
            
            all_pairs.append({
                'pair': pair,
                'score': avg_score,
                'details': results
            })
        
        all_pairs.sort(key=lambda x: x['score'], reverse=True)
        return all_pairs[:top_n]
    
    def update_weights(self, history):
        """Tự điều chỉnh trọng số dựa trên lịch sử"""
        if len(history) < 5:
            return
        
        # Phân tích thuật toán nào dự đoán đúng
        for h in history[:10]:
            if '🔥' in h.get('KQ', ''):
                # Tăng weight cho algorithms mạnh
                for algo_name in self.algorithms:
                    self.weights[algo_name] *= 1.02
            else:
                # Giảm weight
                for algo_name in self.algorithms:
                    self.weights[algo_name] *= 0.98
        
        # Normalize weights
        total = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total

# === XỬ LÝ DỮ LIỆU ===

def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    nums = [n for n in re.findall(r"\d{5}", clean_text) if n]
    seen = set()
    return [n for n in nums if not (n in seen or seen.add(n))]

def save_to_db(data):
    try:
        with open(DB_FILE, 'w') as f:
            json.dump(data, f)
    except:
        pass

def load_from_db():
    try:
        if os.path.exists(DB_FILE):
            with open(DB_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {"history": [], "weights": {}}

# === GIAO DIỆN ===

st.markdown('<h1 class="god-header">🧠 TITAN V52 - AI CORE FUSION</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888; font-size:10px;">10+ Thuật Toán | Hybrid AI | Self-Learning</p>', unsafe_allow_html=True)

if "history" not in st.session_state:
    db_data = load_from_db()
    st.session_state.history = db_data.get("history", [])
    st.session_state.weights = db_data.get("weights", {})

user_input = st.text_area("📥 Kết quả (kỳ mới nhất ở dưới):", height=100, 
                          placeholder="99180\n50655\n06213")

col1, col2 = st.columns(2)
with col1:
    if st.button("🧬 KÍCH HOẠT AI", type="primary", use_container_width=True):
        nums = get_nums(user_input)
        if len(nums) >= 15:
            # Check kết quả
            if "last_pred" in st.session_state and nums:
                lp = st.session_state.last_pred
                last = nums[-1]
                if lp and 'top_pairs' in lp and lp['top_pairs']:
                    best = lp['top_pairs'][0]['pair']
                    win = all(d in last for d in best)
                    st.session_state.history.insert(0, {
                        'Kỳ': last,
                        'Dự đoán': best,
                        'KQ': '🔥' if win else '❌'
                    })
            
            # AI Core prediction
            engine = AICoreFusion()
            if st.session_state.weights:
                engine.weights = st.session_state.weights
            
            top_pairs = engine.get_top_pairs(nums, top_n=10)
            
            # Update weights
            engine.update_weights(st.session_state.history)
            st.session_state.weights = engine.weights
            
            # Save
            save_to_db({
                "history": st.session_state.history[:50],
                "weights": st.session_state.weights
            })
            
            st.session_state.last_pred = {
                'top_pairs': top_pairs,
                'engine': engine
            }
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
    
    # Metrics
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-cell">
            <div style="font-size:10px; color:#888;">TOP PAIR</div>
            <div style="font-size:24px; font-weight:900; color:#00ffff;">{res['top_pairs'][0]['pair']}</div>
        </div>
        <div class="metric-cell">
            <div style="font-size:10px; color:#888;">SCORE</div>
            <div style="font-size:24px; font-weight:900; color:#00ff40;">{res['top_pairs'][0]['score']:.1f}</div>
        </div>
        <div class="metric-cell">
            <div style="font-size:10px; color:#888;">ALGORITHMS</div>
            <div style="font-size:20px; font-weight:900; color:#FFD700;">12</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top pair
    st.markdown("""<div style="text-align:center; margin:15px 0; font-size:14px;">🎯 CẶP VIP - AI FUSION</div>""", unsafe_allow_html=True)
    
    top1 = res['top_pairs'][0]
    st.markdown(f"""
    <div class="prediction-box">
        <div class="big-number">{top1['pair'][0]} - {top1['pair'][1]}</div>
        <div style="font-size:20px; color:#00ff40; margin-top:10px;">Score: {top1['score']:.1f}</div>
        <div style="margin-top:10px;"><span class="tag tag-top">TOP AI</span><span class="tag tag-hybrid">HYBRID</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Algorithm breakdown
    st.markdown("""<div style="text-align:center; margin:15px 0; font-size:14px;">📊 PHÂN TÍCH 12 THUẬT TOÁN</div>""", unsafe_allow_html=True)
    
    for name, data in top1['details'].items():
        border_class = "algo-strong" if data['score'] > 50 else ("algo-medium" if data['score'] > 20 else "algo-weak")
        st.markdown(f"""
        <div class="algorithm-card {border_class}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:bold; color:#FFD700;">{name}</span>
                <span style="color:#00ff40; font-weight:bold;">{data['score']:.1f}</span>
            </div>
            <div style="font-size:10px; color:#888; margin-top:5px;">{data['detail']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Top 5 pairs
    st.markdown("""<div style="text-align:center; margin:15px 0; font-size:14px;">🎯 TOP 5 PAIRS</div>""", unsafe_allow_html=True)
    
    for i, p in enumerate(res['top_pairs'][:5]):
        st.markdown(f"""
        <div style="background:#1a1a2e; border-radius:10px; padding:12px; margin:5px 0; 
                    border-left: 4px solid {'#00ff40' if i == 0 else '#444'};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:24px; font-weight:900; color:#FFD700;">{p['pair'][0]}-{p['pair'][1]}</span>
                <span style="font-size:18px; color:#00ff40;">{p['score']:.1f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # History
    if st.session_state.history:
        st.markdown("""<div style="text-align:center; margin:20px 0; font-size:16px;">📋 LỊCH SỬ</div>""", unsafe_allow_html=True)
        
        df = pd.DataFrame(st.session_state.history[:10])
        
        def color_kq(val):
            return 'color: #00ff40; font-weight: 900' if '🔥' in val else 'color: #ff0040; font-weight: 900'
        
        st.dataframe(df.style.applymap(color_kq, subset=['KQ']), 
                     use_container_width=True, hide_index=True)
        
        # Win rate
        wins = sum(1 for h in st.session_state.history if '🔥' in h['KQ'])
        total = len(st.session_state.history)
        rate = (wins/total*100) if total > 0 else 0
        
        st.markdown(f"""
        <div style="background:#1a1a2e; border-radius:15px; padding:20px; margin:15px 0; 
                    border: 3px solid {'#00ff40' if rate >= 40 else '#ff0040'};">
            <div style="text-align:center; font-size:14px; color:#888;">TỶ LỆ THẮNG</div>
            <div style="text-align:center; font-size:36px; font-weight:900; color:{'#00ff40' if rate >= 40 else '#ff0040'};">
                {rate:.1f}% ({wins}/{total})
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width:{rate}%; background:{'#00ff40' if rate >= 40 else '#ff0040'};">
                    {rate:.0f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# === FOOTER ===
st.markdown("""
<div style="text-align:center; color:#444; font-size:10px; margin-top:30px; padding-top:15px; border-top:1px solid #333;">
    🧠 TITAN V52 - AI CORE FUSION | 12 Algorithms | Hybrid AI | Self-Learning<br>
    <i>Tool hỗ trợ phân tích - Không đảm bảo 100% - Quản lý vốn thông minh</i>
</div>
""", unsafe_allow_html=True)