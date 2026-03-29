import streamlit as st
import re, pandas as pd, numpy as np, math
from collections import Counter, defaultdict
from itertools import combinations
import random

# === CẤU HÌNH ===
LUCKY_OX = [0, 2, 5, 6, 7, 8]
st.set_page_config(page_title="TITAN V52 - MULTI-STRATEGY AI", page_icon="🎯", layout="wide")

# === CSS ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #0a0a1a 50%, #000000 100%);
        color: #FFD700;
        font-family: 'Orbitron', monospace;
    }
    
    .main-header {
        font-size: 36px;
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
    
    .strategy-box {
        background: linear-gradient(135deg, #1a1a2e, #0f0f1a);
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        border-left: 4px solid #FFD700;
    }
    
    .strategy-name {
        font-size: 14px;
        font-weight: 900;
        color: #00FFFF;
        margin-bottom: 8px;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #FFD700;
        text-align: center;
    }
    
    .big-number {
        font-size: 48px;
        font-weight: 900;
        letter-spacing: 10px;
        color: #00FFCC;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
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
        font-size: 10px;
        font-weight: bold;
        margin: 2px;
    }
    
    .tag-green { background: #00ff40; color: #000; }
    .tag-red { background: #ff0040; color: #fff; }
    .tag-yellow { background: #ffff00; color: #000; }
    .tag-blue { background: #0099ff; color: #fff; }
    
    .progress-bar {
        height: 20px;
        background: #1a1a1a;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
        color: #000;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #3a1a1a, #2a0a0a);
        border: 2px solid #ff0040;
        color: #ff6680;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    
    .success-box {
        background: linear-gradient(135deg, #1a3a1a, #0a2a0a);
        border: 2px solid #00ff40;
        color: #66ff80;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# === 10+ THUẬT TOÁN DỰ ĐOÁN ===

class MultiStrategyAI:
    """Hệ thống đa chiến thuật với 10+ thuật toán"""
    
    def __init__(self, db, history=None):
        self.db = db
        self.history = history or []
        self.strategy_weights = self._init_weights()
        
    def _init_weights(self):
        """Khởi tạo trọng số cho từng chiến thuật"""
        return {
            'frequency': 1.0,
            'markov': 1.0,
            'cycle': 1.0,
            'mirror': 1.0,
            'gan': 1.0,
            'streak': 1.0,
            'position': 1.0,
            'monte_carlo': 1.0,
            'pattern': 1.0,
            'heuristic': 1.0
        }
    
    def strategy_frequency(self, pair):
        """1. Frequency Analysis - Phân tích tần suất"""
        freq = sum(1 for n in self.db[-50:] if set(pair).issubset(set(n)))
        expected = 50 * 0.2  # Xác suất lý thuyết ~20%
        
        if freq == 0:
            return 30  # Chưa về -> sắp về
        elif 3 <= freq <= 12:
            return 70  # Bình thường
        elif freq > 15:
            return 20  # Quá nhiều -> tránh
        return 50
    
    def strategy_markov(self, pair):
        """2. Markov Chain - Chuỗi chuyển trạng thái"""
        if len(self.db) < 10:
            return 50
        
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(self.db) - 1):
            curr = self.db[i]
            next_num = self.db[i + 1]
            for d in curr:
                for nd in next_num:
                    transitions[d][nd] += 1
        
        score = 0
        for d in pair:
            if d in transitions:
                total = sum(transitions[d].values())
                for other in pair:
                    if other != d and other in transitions[d]:
                        prob = transitions[d][other] / total
                        score += prob * 100
        
        return min(100, score)
    
    def strategy_cycle(self, pair):
        """3. Cycle Detection - Phát hiện chu kỳ"""
        if len(self.db) < 20:
            return 50
        
        # Tìm chu kỳ 5-15
        for cycle_len in range(5, 15):
            matches = 0
            for i in range(len(self.db) - cycle_len - 1):
                if set(pair).issubset(set(self.db[i])) and set(pair).issubset(set(self.db[i + cycle_len])):
                    matches += 1
            
            if matches >= 2:
                return 80  # Phát hiện chu kỳ
        
        return 40
    
    def strategy_mirror(self, pair):
        """4. Mirror/Bóng số - Số đối xứng"""
        mirrors = {'0': '5', '1': '6', '2': '7', '3': '8', '4': '9',
                   '5': '0', '6': '1', '7': '2', '8': '3', '9': '4'}
        
        score = 0
        for d in pair:
            mirror_d = mirrors.get(d, d)
            # Check nếu mirror xuất hiện nhiều
            mirror_count = sum(1 for n in self.db[-30:] if mirror_d in n)
            if mirror_count > 8:
                score += 30
        
        return min(100, score)
    
    def strategy_gan(self, pair):
        """5. Số Gan - Số lâu chưa về"""
        gan = 0
        for num in reversed(self.db):
            if not set(pair).issubset(set(num)):
                gan += 1
            else:
                break
        
        # Vùng vàng: 5-12 kỳ
        if 5 <= gan <= 12:
            return 90
        elif 2 <= gan <= 4:
            return 60
        elif gan > 18:
            return 20  # Gan quá sâu
        return 50
    
    def strategy_streak(self, pair):
        """6. Số Bệt - Tránh bệt quá dài"""
        streak = 0
        for num in reversed(self.db):
            if set(pair).issubset(set(num)):
                streak += 1
            else:
                break
        
        if streak == 0:
            return 50
        elif streak == 1:
            return 70  # Rơi lại
        elif streak >= 3:
            return 10  # Bệt quá dài -> tránh
        return 40
    
    def strategy_position(self, pair):
        """7. Position Pattern - Pattern từng vị trí"""
        if len(self.db) < 20:
            return 50
        
        positions = {i: defaultdict(int) for i in range(5)}
        for num in self.db[-30:]:
            for i, d in enumerate(num):
                positions[i][d] += 1
        
        score = 0
        for d in pair:
            for pos in positions:
                if positions[pos][d] > 5:  # Hot ở vị trí này
                    score += 20
        
        return min(100, score)
    
    def strategy_monte_carlo(self, pair):
        """8. Monte Carlo Simulation - Mô phỏng ngẫu nhiên"""
        if len(self.db) < 30:
            return 50
        
        # Simulate 1000 lần
        simulations = 1000
        hits = 0
        
        recent_str = "".join(self.db[-30:])
        digit_probs = Counter(recent_str)
        total = sum(digit_probs.values())
        
        for _ in range(simulations):
            # Generate số ngẫu nhiên theo xác suất
            sim_num = ""
            for _ in range(5):
                r = random.randint(0, total - 1)
                cumsum = 0
                for d, count in digit_probs.items():
                    cumsum += count
                    if r < cumsum:
                        sim_num += d
                        break
            
            if set(pair).issubset(set(sim_num)):
                hits += 1
        
        prob = hits / simulations * 100
        return prob * 1.5  # Scale up
    
    def strategy_pattern(self, pair):
        """9. Pattern Recognition - Nhận diện pattern AI"""
        if len(self.db) < 20:
            return 50
        
        # Tìm pattern phức tạp
        patterns = defaultdict(int)
        for i in range(len(self.db) - 2):
            key = (self.db[i], self.db[i+1])
            patterns[key] += 1
        
        # Check nếu pair này xuất hiện sau pattern tương tự
        score = 0
        for (prev, curr), count in patterns.items():
            if count >= 2 and set(pair).issubset(set(curr)):
                score += 25
        
        return min(100, score)
    
    def strategy_heuristic(self, pair):
        """10. Heuristic + Kinh nghiệm dân gian"""
        score = 0
        
        # Rule 1: Số tuổi Sửu
        if any(int(d) in LUCKY_OX for d in pair):
            score += 20
        
        # Rule 2: Tổng digital root
        dr = (int(pair[0]) + int(pair[1])) % 9
        recent_dr = Counter(sum(int(d) for d in n) % 9 for n in self.db[-20:])
        if recent_dr and dr == recent_dr.most_common(1)[0][0]:
            score += 25
        
        # Rule 3: Số kề nhau
        if abs(int(pair[0]) - int(pair[1])) == 1:
            score += 15
        
        # Rule 4: Cùng chẵn/lẻ
        if (int(pair[0]) % 2) == (int(pair[1]) % 2):
            score += 10
        
        return min(100, score)
    
    def calculate_hybrid_score(self, pair):
        """Kết hợp tất cả chiến thuật"""
        strategies = {
            'frequency': self.strategy_frequency(pair),
            'markov': self.strategy_markov(pair),
            'cycle': self.strategy_cycle(pair),
            'mirror': self.strategy_mirror(pair),
            'gan': self.strategy_gan(pair),
            'streak': self.strategy_streak(pair),
            'position': self.strategy_position(pair),
            'monte_carlo': self.strategy_monte_carlo(pair),
            'pattern': self.strategy_pattern(pair),
            'heuristic': self.strategy_heuristic(pair)
        }
        
        # Tính điểm weighted
        total_score = 0
        total_weight = 0
        
        for strategy, score in strategies.items():
            weight = self.strategy_weights.get(strategy, 1.0)
            total_score += score * weight
            total_weight += weight
        
        final_score = total_score / total_weight if total_weight > 0 else 50
        
        # Adjust theo history
        if self.history:
            pair_losses = sum(1 for h in self.history if h.get('Dự đoán') == pair and '❌' in h.get('KQ', ''))
            if pair_losses >= 2:
                final_score *= 0.6  # Giảm mạnh nếu thua nhiều
        
        return final_score, strategies
    
    def predict(self):
        """Dự đoán cuối cùng"""
        if len(self.db) < 15:
            return None
        
        # Generate tất cả cặp
        all_pairs = []
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            score, strategies = self.calculate_hybrid_score(pair)
            
            all_pairs.append({
                'pair': pair,
                'score': score,
                'strategies': strategies,
                'top_strategy': max(strategies.items(), key=lambda x: x[1])[0]
            })
        
        all_pairs.sort(key=lambda x: x['score'], reverse=True)
        top_pairs = all_pairs[:10]
        
        # Tính confidence
        if top_pairs:
            confidence = min(95, max(30, top_pairs[0]['score']))
        else:
            confidence = 50
        
        # Top 3 triples
        triples = []
        for t in combinations("0123456789", 3):
            score = sum(
                next((p['score'] for p in top_pairs if p['pair'] == ''.join(c)), 0)
                for c in combinations(t, 2)
            )
            triples.append((''.join(t), score))
        triples.sort(key=lambda x: x[1], reverse=True)
        
        # Top 8
        single_pool = Counter("".join(self.db[-50:]))
        top8 = "".join([d for d, _ in single_pool.most_common(8)])
        
        return {
            'pairs': top_pairs,
            'triples': triples[:5],
            'top8': top8,
            'confidence': confidence,
            'strategy_breakdown': top_pairs[0]['strategies'] if top_pairs else {}
        }

# === XỬ LÝ DỮ LIỆU ===

def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    nums = [n for n in re.findall(r"\d{5}", clean_text) if n]
    seen = set()
    return [n for n in nums if not (n in seen or seen.add(n))]

# === GIAO DIỆN ===

st.markdown('<h1 class="main-header">🎯 TITAN V52 - MULTI-STRATEGY AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888; font-size:11px;">10+ Thuật Toán | Hybrid AI | Self-Learning</p>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_area("📥 Kết quả (kỳ mới nhất ở dưới):", height=100, 
                          placeholder="07988\n35782\n01053")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 PHÂN TÍCH", type="primary", use_container_width=True):
        nums = get_nums(user_input)
        if len(nums) >= 15:
            if "last_pred" in st.session_state and nums:
                lp = st.session_state.last_pred
                last = nums[-1]
                if lp and lp['pairs']:
                    best = lp['pairs'][0]['pair']
                    win = all(d in last for d in best)
                    st.session_state.history.insert(0, {
                        'Kỳ': last,
                        'Dự đoán': best,
                        'KQ': '🔥' if win else '❌'
                    })
            
            ai = MultiStrategyAI(nums, st.session_state.history)
            st.session_state.last_pred = ai.predict()
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
            <div style="font-size:10px; color:#888;">TIN CẬY</div>
            <div style="font-size:24px; font-weight:900; color:#00ffff;">{res['confidence']:.0f}%</div>
        </div>
        <div class="metric-cell">
            <div style="font-size:10px; color:#888;">TOP PAIR</div>
            <div style="font-size:24px; font-weight:900; color:#FFD700;">{res['pairs'][0]['pair']}</div>
        </div>
        <div class="metric-cell">
            <div style="font-size:10px; color:#888;">SCORE</div>
            <div style="font-size:24px; font-weight:900; color:#00ff40;">{res['pairs'][0]['score']:.0f}</div>
        </div>
        <div class="metric-cell">
            <div style="font-size:10px; color:#888;">STRATEGY</div>
            <div style="font-size:14px; font-weight:900; color:#ff00ff;">{res['pairs'][0]['top_strategy'].upper()}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy breakdown
    if res['strategy_breakdown']:
        st.markdown("""<div style="text-align:center; margin:15px 0; font-size:14px;">📊 PHÂN TÍCH 10 CHIẾN THUẬT</div>""", unsafe_allow_html=True)
        
        cols = st.columns(2)
        for i, (strategy, score) in enumerate(res['strategy_breakdown'].items()):
            with cols[i % 2]:
                bar_color = "#00ff40" if score >= 70 else ("#ffff00" if score >= 40 else "#ff0040")
                st.markdown(f"""
                <div class="strategy-box">
                    <div class="strategy-name">{strategy.upper()}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width:{score}%; background:{bar_color};">{score:.0f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Top pair
    st.markdown("""<div style="text-align:center; margin:15px 0; font-size:14px;">🎯 CẶP VIP</div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="prediction-card">
        <div class="big-number">{res['pairs'][0]['pair'][0]} - {res['pairs'][0]['pair'][1]}</div>
        <div style="margin-top:10px; font-size:18px; color:#00ff40;">Score: {res['pairs'][0]['score']:.0f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top 5 pairs
    st.markdown("""<div style="text-align:center; margin:15px 0; font-size:14px;">🎯 TOP 5 PAIRS</div>""", unsafe_allow_html=True)
    
    for i, p in enumerate(res['pairs'][:5]):
        st.markdown(f"""
        <div style="background:#1a1a2e; border-radius:10px; padding:12px; margin:5px 0; 
                    border-left: 4px solid {'#00ff40' if i == 0 else '#444'};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:26px; font-weight:900; color:#FFD700;">{p['pair'][0]}-{p['pair'][1]}</span>
                <span style="font-size:18px; color:#00ff40;">{p['score']:.0f}</span>
            </div>
            <div style="font-size:10px; color:#888; margin-top:5px;">Best: {p['top_strategy']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Top 8
    st.markdown(f"""
    <div class="prediction-card" style="margin-top:15px;">
        <div style="font-size:12px; color:#888;">ĐỘ PHỦ SẢNH</div>
        <div style="font-size:32px; font-weight:900; letter-spacing:8px; color:#00ffff;">{res['top8']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # History
    if st.session_state.history:
        st.markdown("""<div style="text-align:center; margin:20px 0; font-size:16px;">📋 LỊCH SỬ</div>""", unsafe_allow_html=True)
        
        df = pd.DataFrame(st.session_state.history[:15])
        
        def color_kq(val):
            return 'color: #00ff40; font-weight: 900' if '🔥' in val else 'color: #ff0040; font-weight: 900'
        
        st.dataframe(df.style.applymap(color_kq, subset=['KQ']), 
                     use_container_width=True, hide_index=True)
        
        wins = sum(1 for h in st.session_state.history if '🔥' in h['KQ'])
        total = len(st.session_state.history)
        rate = (wins/total*100) if total > 0 else 0
        
        color_rate = "#00ff40" if rate >= 40 else ("#ffff00" if rate >= 30 else "#ff0040")
        
        st.markdown(f"""
        <div style="border: 3px solid {color_rate}; border-radius: 15px; padding: 20px; margin-top:15px; text-align:center;">
            <div style="font-size:14px; color:#888;">TỶ LỆ THẮNG</div>
            <div style="font-size:36px; font-weight:900; color:{color_rate};">{rate:.1f}% ({wins}/{total})</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width:{rate}%; background:{color_rate};">{rate:.0f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# === FOOTER ===
st.markdown("""
<div style="text-align:center; color:#444; font-size:10px; margin-top:30px; padding-top:15px; border-top:1px solid #333;">
    🎯 TITAN V52 - MULTI-STRATEGY AI | 10+ Algorithms | Hybrid Intelligence<br>
    <i>Tool hỗ trợ phân tích - Không đảm bảo 100% - Quản lý vốn thông minh</i>
</div>
""", unsafe_allow_html=True)