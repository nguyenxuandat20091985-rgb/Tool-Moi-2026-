import streamlit as st
import re, pandas as pd, numpy as np, math
from collections import Counter, defaultdict
from itertools import combinations
import random

# === CẤU HÌNH ===
LUCKY_OX = [0, 2, 5, 6, 7, 8]
st.set_page_config(page_title="TITAN V49 - OMNISCIENT", page_icon="🧠", layout="wide")

# === CSS CAO CẤP ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
        color: #FFD700;
        font-family: 'Orbitron', monospace;
    }
    
    .main-header {
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #FFD700, #FFA500, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
        animation: glow 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.3); }
    }
    
    .metric-box {
        background: linear-gradient(135deg, #16213e, #0f3460);
        border: 2px solid #00fff5;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 255, 245, 0.3);
        margin: 10px 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border: 3px solid;
        border-image-slice: 1;
        border-image-source: linear-gradient(90deg, #FFD700, #FF00FF, #00fff5);
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    }
    
    .number-display {
        font-size: 42px;
        font-weight: 900;
        letter-spacing: 8px;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #000, #1a1a1a);
        border-radius: 15px;
        margin: 10px 0;
        border: 2px solid #FFD700;
    }
    
    .confidence-meter {
        height: 30px;
        background: #1a1a1a;
        border-radius: 15px;
        overflow: hidden;
        margin: 10px 0;
        position: relative;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00);
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #000;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #2d1b1b, #1a0a0a);
        border: 2px solid #ff0040;
        color: #ff6680;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        animation: pulse-warning 1.5s infinite;
    }
    
    @keyframes pulse-warning {
        0%, 100% { box-shadow: 0 0 10px rgba(255, 0, 64, 0.5); }
        50% { box-shadow: 0 0 30px rgba(255, 0, 64, 0.8); }
    }
    
    .success-box {
        background: linear-gradient(135deg, #1b2d1b, #0a1a0a);
        border: 2px solid #00ff40;
        color: #66ff80;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .tag {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: bold;
        margin: 3px;
        text-transform: uppercase;
    }
    
    .tag-hot { background: #ff0040; color: white; }
    .tag-cold { background: #0066ff; color: white; }
    .tag-shadow { background: #9900ff; color: white; }
    .tag-trap { background: #ff0000; color: white; animation: blink 0.5s infinite; }
    .tag-gold { background: #FFD700; color: black; }
    
    @keyframes blink { 50% { opacity: 0.5; } }
    
    .history-table {
        background: #0a0a0a;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .stDataFrame {
        background: #0a0a0a !important;
    }
</style>
""", unsafe_allow_html=True)

# === THUẬT TOÁN ĐA LỚP ===

class QuantumAnalyzer:
    """Lớp 1: Phân tích lượng tử đa chiều"""
    
    @staticmethod
    def digital_root_sequence(nums):
        """Phân tích chuỗi digital root"""
        roots = [sum(int(d) for d in n) % 9 for n in nums]
        return Counter(roots).most_common(3)
    
    @staticmethod
    def fibonacci_weight(nums):
        """Áp dụng dãy Fibonacci để tính trọng số"""
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        weights = {}
        for idx, num in enumerate(reversed(nums[-10:])):
            for d in num:
                if d not in weights:
                    weights[d] = 0
                weights[d] += fib[idx] if idx < len(fib) else fib[-1]
        return weights
    
    @staticmethod
    def prime_factor_analysis(nums):
        """Phân tích số nguyên tố"""
        primes = [2, 3, 5, 7]
        prime_counts = {p: 0 for p in primes}
        for num in nums:
            for d in num:
                digit = int(d)
                if digit in primes:
                    prime_counts[digit] += 1
        return prime_counts

class AntiPatternDetector:
    """Lớp 2: Phát hiện bẫy và anti-pattern"""
    
    def __init__(self, db):
        self.db = db
        self.trap_pairs = set()
        self.detect_traps()
    
    def detect_traps(self):
        """Phát hiện cặp số bẫy"""
        if len(self.db) < 20:
            return
        
        # Pattern 1: Over-appearance (xuất hiện quá nhiều)
        pair_counts = Counter()
        for num in self.db[-30:]:
            for p in combinations(sorted(set(num)), 2):
                pair_counts[p] += 1
        
        for pair, count in pair_counts.items():
            if count >= 8:  # Về >= 8 lần trong 30 kỳ
                self.trap_pairs.add(pair)
        
        # Pattern 2: Repetition trap (lặp lại liên tiếp)
        for i in range(len(self.db) - 2):
            curr_set = set(self.db[i])
            next_set = set(self.db[i+1])
            if len(curr_set & next_set) >= 3:  # Trùng >= 3 số
                for p in combinations(sorted(curr_set & next_set), 2):
                    self.trap_pairs.add(p)
    
    def is_trap(self, pair):
        return tuple(sorted(pair)) in self.trap_pairs

class BehavioralAnalyzer:
    """Lớp 3: Phân tích tâm lý nhà cái"""
    
    @staticmethod
    def reverse_psychology_score(db, pair):
        """
        Điểm tâm lý ngược: 
        Nếu đám đông nghĩ cặp này sẽ về -> nhà cái sẽ không cho về
        """
        if len(db) < 15:
            return 0
        
        # Đếm tần suất trong 15 kỳ gần
        recent_count = sum(1 for n in db[-15:] if set(pair).issubset(set(n)))
        
        # Nếu về > 4 lần trong 15 kỳ -> khả năng cao là bẫy tâm lý
        if recent_count >= 4:
            return -50  # Điểm âm cao
        elif recent_count >= 2:
            return -20
        return 10  # Bình thường
    
    @staticmethod
    def crowd_behavior(db):
        """Phân tích hành vi đám đông"""
        # Số nào đang được đám đông chú ý (về nhiều gần đây)
        recent_str = "".join(db[-10:])
        hot_numbers = Counter(recent_str).most_common(3)
        return [n[0] for n in hot_numbers]

class RiskManager:
    """Lớp 4: Quản lý rủi ro theo Kelly Criterion"""
    
    @staticmethod
    def calculate_kelly_fraction(win_rate, odds=1.85):
        """
        Kelly Criterion: f = (bp - q) / b
        b = odds - 1
        p = win rate
        q = 1 - p
        """
        b = odds - 1
        p = win_rate / 100
        q = 1 - p
        
        kelly = (b * p - q) / b
        return max(0, min(kelly, 0.25))  # Giới hạn 0-25%
    
    @staticmethod
    def diversification_score(predictions):
        """Điểm đa dạng hóa - tránh put all eggs in one basket"""
        if len(predictions) < 3:
            return 0
        
        all_numbers = set()
        for pred in predictions:
            all_numbers.update(pred)
        
        # Tỷ lệ phủ sóng
        coverage = len(all_numbers) / 10
        return coverage

class NeuralPatternMatcher:
    """Lớp 5: Match pattern kiểu neural network"""
    
    def __init__(self, db):
        self.db = db
        self.pattern_memory = defaultdict(list)
        self.build_pattern_memory()
    
    def build_pattern_memory(self):
        """Xây dựng bộ nhớ pattern"""
        for idx in range(len(self.db) - 1):
            current = self.db[idx]
            next_num = self.db[idx + 1]
            
            # Lưu nhớ: Sau khi current xuất hiện, next_num xuất hiện
            for d in current:
                for nd in next_num:
                    self.pattern_memory[d].append(nd)
    
    def get_next_probability(self, last_num):
        """Dự đoán số có khả năng xuất hiện tiếp theo"""
        if not last_num:
            return {}
        
        prob = Counter()
        for d in last_num:
            if d in self.pattern_memory:
                for next_d in self.pattern_memory[d]:
                    prob[next_d] += 1
        
        total = sum(prob.values())
        if total == 0:
            return {}
        
        return {k: v/total * 100 for k, v in prob.items()}

class TITANV49Engine:
    """Engine chính kết hợp 5 lớp"""
    
    def __init__(self, db):
        self.db = db
        self.quantum = QuantumAnalyzer()
        self.anti_pattern = AntiPatternDetector(db)
        self.behavioral = BehavioralAnalyzer()
        self.risk_manager = RiskManager()
        self.neural = NeuralPatternMatcher(db)
    
    def calculate_composite_score(self, pair):
        """Tính điểm tổng hợp từ 5 lớp"""
        score = 0
        details = {}
        
        # === LỚP 1: QUANTUM ===
        dr_score = self._quantum_score(pair)
        score += dr_score
        details['quantum'] = dr_score
        
        # === LỚP 2: ANTI-PATTERN ===
        if self.anti_pattern.is_trap(pair):
            score -= 100
            details['trap'] = -100
        else:
            score += 20
            details['trap'] = 20
        
        # === LỚP 3: BEHAVIORAL ===
        behavior_score = self.behavioral.reverse_psychology_score(self.db, pair)
        score += behavior_score
        details['behavior'] = behavior_score
        
        # === LỚP 4: FREQUENCY & GAN ===
        freq_score = self._frequency_score(pair)
        score += freq_score
        details['frequency'] = freq_score
        
        # === LỚP 5: NEURAL PATTERN ===
        neural_score = self._neural_score(pair)
        score += neural_score
        details['neural'] = neural_score
        
        # === BONUS: TUỔI SỬU ===
        if any(int(d) in LUCKY_OX for d in pair):
            score += 15
            details['lucky'] = 15
        
        return score, details
    
    def _quantum_score(self, pair):
        """Điểm lượng tử"""
        score = 0
        
        # Digital root match
        dr_mode = self.quantum.digital_root_sequence(self.db)
        if dr_mode:
            pair_dr = (int(pair[0]) + int(pair[1])) % 9
            if pair_dr == dr_mode[0][0]:
                score += 30
        
        # Fibonacci weight
        fib_weights = self.quantum.fibonacci_weight(self.db)
        for d in pair:
            if d in fib_weights:
                score += min(fib_weights[d] / 10, 25)
        
        return score
    
    def _frequency_score(self, pair):
        """Điểm tần suất & nhịp"""
        score = 0
        
        # Tính gan
        gan = 0
        for num in reversed(self.db):
            if not set(pair).issubset(set(num)):
                gan += 1
            else:
                break
        
        # Tính bệt
        streak = 0
        for num in reversed(self.db):
            if set(pair).issubset(set(num)):
                streak += 1
            else:
                break
        
        # Golden zone: gan 5-12
        if 5 <= gan <= 12:
            score += 60
        elif 2 <= gan <= 4:
            score += 30
        elif gan > 18:
            score -= 40
        
        # Anti-bet: tránh bệt >= 3
        if streak >= 3:
            score -= 80
        elif streak == 1:
            score += 40
        
        return score
    
    def _neural_score(self, pair):
        """Điểm neural pattern"""
        if not self.db:
            return 0
        
        last_num = self.db[-1]
        probs = self.neural.get_next_probability(last_num)
        
        score = 0
        for d in pair:
            if d in probs:
                score += probs[d] * 0.5
        
        return score
    
    def predict(self):
        """Dự đoán cuối cùng"""
        if len(self.db) < 15:
            return None
        
        # Generate tất cả cặp có thể
        all_pairs = []
        recent_str = "".join(self.db[-50:])
        single_pool = Counter(recent_str)
        
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            score, details = self.calculate_composite_score(p)
            
            # Check nếu pair này có trong dữ liệu gần đây
            freq = sum(1 for n in self.db[-30:] if set(p).issubset(set(n)))
            
            all_pairs.append({
                'pair': pair,
                'score': score,
                'details': details,
                'frequency': freq
            })
        
        # Sort và lấy top 5
        all_pairs.sort(key=lambda x: x['score'], reverse=True)
        top_pairs = all_pairs[:5]
        
        # Tính độ tin cậy
        if top_pairs:
            top_score = top_pairs[0]['score']
            confidence = min(95, max(30, (top_score + 100) / 3))
        else:
            confidence = 50
        
        # 3 số top
        all_triples = []
        for t in combinations("0123456789", 3):
            score, _ = self.calculate_composite_score(t)
            all_triples.append((''.join(t), score))
        
        all_triples.sort(key=lambda x: x[1], reverse=True)
        top_triples = all_triples[:3]
        
        # Top 8 numbers
        top_8 = "".join([d for d, _ in single_pool.most_common(8)])
        
        return {
            'pairs': top_pairs,
            'triples': top_triples,
            'top8': top_8,
            'confidence': confidence,
            'crowd_numbers': self.behavioral.crowd_behavior(self.db)
        }

# === XỬ LÝ DỮ LIỆU ===

def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    nums = [n for n in re.findall(r"\d{5}", clean_text) if n]
    
    # Lọc trùng nhưng giữ thứ tự
    seen = set()
    unique = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    
    return unique

# === GIAO DIỆN ===

st.markdown('<h1 class="main-header">🧠 TITAN V49 - OMNISCIENT</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#00fff5; font-size:14px;">5-Lớp Thuật Toán | Anti-Pattern | Behavioral Analysis | Neural Network | Kelly Criterion</p>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_area(
    "📥 **Dán kết quả thực tế** (kỳ mới nhất ở dưới cùng):",
    height=200,
    placeholder="Ví dụ:\n22814\n71553\n75641\n..."
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    if st.button("🚀 KÍCH HOẠT OMNISCIENT", type="primary"):
        nums = get_nums(user_input)
        
        if len(nums) >= 15:
            # Check kết quả kỳ trước
            if "last_pred" in st.session_state and nums:
                lp = st.session_state.last_pred
                last_actual = nums[-1]
                
                if lp and 'pairs' in lp and lp['pairs']:
                    best_pair = lp['pairs'][0]['pair']
                    win = all(d in last_actual for d in best_pair)
                    
                    st.session_state.history.insert(0, {
                        'Kỳ': last_actual,
                        'Dự đoán': best_pair,
                        'Điểm': lp['pairs'][0]['score'],
                        'KQ': '🔥 WIN' if win else '❌'
                    })
            
            # Dự đoán mới
            engine = TITANV49Engine(nums)
            st.session_state.last_pred = engine.predict()
            st.session_state.last_nums = nums
            st.rerun()
        else:
            st.error(f"⚠️ Cần ít nhất 15 kỳ (hiện có: {len(nums)})")
    
with col2:
    if st.button(" Thống kê"):
        st.session_state.show_stats = True
    
with col3:
    if st.button("🗑️ Reset"):
        st.session_state.clear()
        st.rerun()

# === HIỂN THỊ KẾT QUẢ ===

if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    # === CẢNH BÁO ===
    if res['confidence'] < 60:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <b>CẢNH BÁO RỦI RO CAO</b><br>
            Độ tin cậy thấp. Nhà cái có thể đang đảo cầu. Khuyến nghị: <b>NGHỈ</b> hoặc đánh thăm dò.
        </div>
        """, unsafe_allow_html=True)
    elif res['confidence'] >= 80:
        st.markdown("""
        <div class="success-box">
            ✅ <b>TÍN HIỆU TỐT</b><br>
            Độ tin cậy cao. Có thể vào tiền theo kế hoạch quản lý vốn.
        </div>
        """, unsafe_allow_html=True)
    
    # === METRICS ===
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size:12px; color:#888;">ĐỘ TIN CẬY</div>
            <div style="font-size:32px; font-weight:900; color:#00fff5;">{res['confidence']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        crowd_str = ", ".join(res['crowd_numbers'])
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size:12px; color:#888;">SỐ ĐÁM ĐÔNG</div>
            <div style="font-size:24px; font-weight:900; color:#ff0040;">{crowd_str}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size:12px; color:#888;">TOP PAIR</div>
            <div style="font-size:28px; font-weight:900; color:#FFD700;">{res['pairs'][0]['pair']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        kelly_frac = RiskManager.calculate_kelly_fraction(res['confidence'])
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size:12px; color:#888;">KELLY BET</div>
            <div style="font-size:28px; font-weight:900; color:#00ff40;">{kelly_frac*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # === DỰ ĐOÁN 2 TINH ===
    st.markdown('<h2 style="color:#FFD700; margin-top:30px;">🎯 2 TINH - TOP 5</h2>', unsafe_allow_html=True)
    
    cols = st.columns(5)
    for i, pair_data in enumerate(res['pairs']):
        with cols[i]:
            tags = ""
            
            # Check trap
            if pair_data['score'] < 0:
                tags += '<span class="tag tag-trap">⚠️ BẪY</span>'
            
            # Check crowd
            if any(d in res['crowd_numbers'] for d in pair_data['pair']):
                tags += '<span class="tag tag-hot">HOT</span>'
            
            # Check quantum
            if pair_data['details'].get('quantum', 0) > 30:
                tags += '<span class="tag tag-gold">QUANTUM</span>'
            
            st.markdown(f"""
            <div class="prediction-card" style="border-color: {'#ff0040' if pair_data['score'] < 0 else '#00ff40'};">
                <div class="number-display" style="font-size:32px; letter-spacing:4px;">
                    {pair_data['pair'][0]} - {pair_data['pair'][1]}
                </div>
                <div style="text-align:center;">
                    <div style="color:#888; font-size:11px;">SCORE</div>
                    <div style="font-size:24px; font-weight:900; color:{'#ff0040' if pair_data['score'] < 0 else '#00ff40'};">
                        {pair_data['score']:.0f}
                    </div>
                    <div style="margin-top:10px;">{tags}</div>
                </div>
                <div style="margin-top:10px; font-size:10px; color:#666;">
                    {pair_data['details']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # === DỰ ĐOÁN 3 TINH ===
    st.markdown('<h2 style="color:#FFD700; margin-top:30px;">💎 3 TINH - TOP 3</h2>', unsafe_allow_html=True)
    
    cols3 = st.columns(3)
    for i, (triple, score) in enumerate(res['triples']):
        with cols3[i]:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="number-display" style="font-size:28px;">
                    {triple[0]} - {triple[1]} - {triple[2]}
                </div>
                <div style="text-align:center; font-size:20px; font-weight:900; color:#FFD700;">
                    Score: {score:.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # === ĐỘ PHỦ SẢNH ===
    st.markdown('<h2 style="color:#00fff5; margin-top:30px;">🎯 ĐỘ PHỦ SẢNH (8 SỐ)</h2>', unsafe_allow_html=True)
    st.markdown(f'<div class="number-display">{res["top8"]}</div>', unsafe_allow_html=True)
    
    # === PHÂN TÍCH CHI TIẾT ===
    if st.checkbox("📊 Hiển thị phân tích chi tiết"):
        st.markdown('<h3 style="color:#888;">Chi tiết 5 lớp thuật toán cho TOP 1:</h3>', unsafe_allow_html=True)
        
        top1 = res['pairs'][0]
        details = top1['details']
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <h4>Điểm từng lớp:</h4>
                <ul style="text-align:left; list-style:none; padding:0;">
                    <li>🔮 Quantum: <b style="color:#00fff5;">{details.get('quantum', 0):.0f}</b></li>
                    <li>🚫 Anti-Pattern: <b style="color:{'#ff0040' if details.get('trap', 0) < 0 else '#00ff40'};">{details.get('trap', 0)}</b></li>
                    <li>🧠 Behavioral: <b style="color:#ff9900;">{details.get('behavior', 0)}</b></li>
                    <li>📈 Frequency: <b style="color:#9900ff;">{details.get('frequency', 0)}</b></li>
                    <li>🔗 Neural: <b style="color:#00ff99;">{details.get('neural', 0):.0f}</b></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <h4>Khuyến nghị:</h4>
                <p style="text-align:left; font-size:13px;">
                • Kelly Bet: Tính toán % vốn tối ưu<br>
                • Tránh crowd numbers<br>
                • Ưu tiên pairs có score > 100<br>
                • Tránh pairs có tag ⚠️ BẪY
                </p>
            </div>
            """, unsafe_allow_html=True)

# === LỊCH SỬ ===
if st.session_state.history:
    st.markdown('<h2 style="color:#FFD700; margin-top:30px;">📋 Lịch sử đối soát</h2>', unsafe_allow_html=True)
    
    df = pd.DataFrame(st.session_state.history[:20])
    
    # Color coding
    def color_kq(val):
        if 'WIN' in val:
            return 'color: #00ff40; font-weight: bold'
        return 'color: #ff0040; font-weight: bold'
    
    st.dataframe(
        df.style.applymap(color_kq, subset=['KQ']),
        use_container_width=True,
        hide_index=True
    )
    
    # Win rate
    wins = sum(1 for h in st.session_state.history if 'WIN' in h['KQ'])
    total = len(st.session_state.history)
    win_rate = (wins / total * 100) if total > 0 else 0
    
    st.markdown(f"""
    <div class="metric-box" style="margin-top:20px;">
        <div style="font-size:16px;">TỶ LỆ THẮNG</div>
        <div style="font-size:36px; font-weight:900; color:{'#00ff40' if win_rate >= 50 else '#ff0040'};">
            {win_rate:.1f}% ({wins}/{total})
        </div>
        <div class="confidence-meter" style="margin-top:10px;">
            <div class="confidence-fill" style="width:{win_rate}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# === FOOTER ===
st.markdown("""
<div style="margin-top:50px; padding:20px; text-align:center; color:#444; font-size:11px; border-top:1px solid #333;">
    TITAN V49 - OMNISCIENT | 5-Layer Quantum Analysis | Anti-Pattern Detection | Behavioral Psychology<br>
    <i>Lưu ý: Tool hỗ trợ phân tích, không đảm bảo 100%. Quản lý vốn thông minh.</i>
</div>
""", unsafe_allow_html=True)