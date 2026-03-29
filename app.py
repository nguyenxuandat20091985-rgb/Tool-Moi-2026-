import streamlit as st
import re, json, os, pandas as pd, numpy as np, math
from collections import Counter, defaultdict
from itertools import combinations
import requests
from datetime import datetime
import time

# === CẤU HÌNH API ===
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

# === CẤU HÌNH HỆ THỐNG ===
LUCKY_OX = [0, 2, 5, 6, 7, 8]
DATA_FILE = "titan_v52_data.json"

st.set_page_config(page_title="TITAN V52 - AI QUANTUM", page_icon="🧠", layout="wide")

# === CSS NEON UI ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0a2e 50%, #0a0a0a 100%);
        color: #FFD700;
        font-family: 'Rajdhani', sans-serif;
    }
    
    .main-header {
        font-size: 42px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
        text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
        font-family: 'Orbitron', monospace;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .neon-box {
        background: linear-gradient(135deg, #1a1a2e, #0f0f1a);
        border: 2px solid;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
    }
    
    .neon-cyan {
        border-color: #00ffff;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.4);
    }
    
    .neon-pink {
        border-color: #ff00ff;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.4);
    }
    
    .neon-gold {
        border-color: #FFD700;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.4);
    }
    
    .prediction-number {
        font-size: 64px;
        font-weight: 900;
        letter-spacing: 15px;
        text-align: center;
        color: #00ffff;
        text-shadow: 0 0 30px #00ffff, 0 0 60px #00ffff;
        font-family: 'Orbitron', monospace;
        margin: 20px 0;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin: 20px 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #0f0f1a);
        border: 2px solid #00ffff;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 900;
        color: #00ffff;
        text-shadow: 0 0 10px #00ffff;
    }
    
    .metric-label {
        font-size: 12px;
        color: #888;
        margin-top: 5px;
        text-transform: uppercase;
    }
    
    .algorithm-card {
        background: #1a1a2e;
        border-left: 4px solid #ff00ff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
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
    
    .tag-hot { background: #ff0040; color: white; box-shadow: 0 0 10px #ff0040; }
    .tag-cold { background: #0066ff; color: white; box-shadow: 0 0 10px #0066ff; }
    .tag-gan { background: #ff9900; color: black; box-shadow: 0 0 10px #ff9900; }
    .tag-bet { background: #00ff40; color: black; box-shadow: 0 0 10px #00ff40; }
    .tag-ai { background: #9900ff; color: white; box-shadow: 0 0 10px #9900ff; }
    
    .confidence-bar {
        height: 30px;
        background: #0a0a0a;
        border-radius: 15px;
        overflow: hidden;
        margin: 10px 0;
        border: 2px solid #333;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        color: #000;
        font-size: 16px;
        transition: width 1s ease;
    }
    
    .history-win { color: #00ff40; font-weight: 900; text-shadow: 0 0 10px #00ff40; }
    .history-lose { color: #ff0040; font-weight: 900; text-shadow: 0 0 10px #ff0040; }
    
    button {
        background: linear-gradient(135deg, #00ffff, #0080ff);
        color: #000;
        font-weight: 900;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.4);
    }
    
    button:hover {
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.6);
    }
    
    .tab-content {
        background: #0a0a0a;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #333;
    }
    
    .thinking-process {
        background: linear-gradient(135deg, #1a0a2e, #0a0a1a);
        border: 2px solid #9900ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# === HỆ THỐNG AI ĐA THUẬT TOÁN ===

class MultiAlgorithmEngine:
    """Engine kết hợp 10+ thuật toán phân tích"""
    
    def __init__(self, db):
        self.db = db
        self.recent = db[-30:] if len(db) >= 30 else db
        self.medium = db[-60:] if len(db) >= 60 else db
        self.long = db
        
    def frequency_analysis(self):
        """1. Phân tích tần suất"""
        all_digits = "".join(self.recent)
        counter = Counter(all_digits)
        total = len(all_digits)
        return {d: count/total for d, count in counter.items()}
    
    def cycle_detection(self):
        """2. Phát hiện chu kỳ lặp"""
        cycles = {}
        for digit in "0123456789":
            positions = [i for i, num in enumerate(self.db) if digit in num]
            if len(positions) >= 3:
                gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    cycles[digit] = avg_gap
        return cycles
    
    def gap_analysis(self):
        """3. Phân tích số gan (khoảng cách)"""
        gaps = {}
        for digit in "0123456789":
            gap = 0
            for num in reversed(self.db):
                if digit in num:
                    break
                gap += 1
            gaps[digit] = gap
        return gaps
    
    def hot_cold_analysis(self):
        """4. Phân tích số nóng/lạnh"""
        freq = self.frequency_analysis()
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        hot = [d for d, _ in sorted_freq[:3]]
        cold = [d for d, _ in sorted_freq[-3:]]
        return hot, cold
    
    def pattern_detection(self):
        """5. Phát hiện pattern đặc biệt"""
        patterns = {
            'repeating': [],
            'mirror': [],
            'complement': []
        }
        
        # Repeating pattern
        for i in range(len(self.db) - 1):
            common = set(self.db[i]) & set(self.db[i+1])
            if len(common) >= 2:
                patterns['repeating'].extend(list(common))
        
        # Complement (9-x)
        for num in self.recent:
            for d in num:
                comp = str(9 - int(d))
                if comp in num:
                    patterns['complement'].append((d, comp))
        
        return patterns
    
    def bayesian_probability(self, pair):
        """6. Xác suất Bayes"""
        # Prior probability
        total_pairs = len(self.recent)
        pair_count = sum(1 for num in self.recent if set(pair).issubset(set(num)))
        prior = pair_count / total_pairs if total_pairs > 0 else 0
        
        # Likelihood based on recent trends
        recent_count = sum(1 for num in self.recent[-10:] if set(pair).issubset(set(num)))
        likelihood = recent_count / 10 if recent_count > 0 else 0.01
        
        # Posterior
        posterior = (likelihood * prior) / (likelihood * prior + 0.01 * (1 - prior))
        return min(posterior * 100, 95)
    
    def markov_chain(self):
        """7. Markov Chain - Transition probability"""
        transitions = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(self.db) - 1):
            current = set(self.db[i])
            next_num = set(self.db[i + 1])
            
            for d in current:
                for nd in next_num:
                    transitions[d][nd] += 1
        
        # Convert to probabilities
        prob_matrix = {}
        for d in transitions:
            total = sum(transitions[d].values())
            prob_matrix[d] = {nd: count/total for nd, count in transitions[d].items()}
        
        return prob_matrix
    
    def monte_carlo_simulation(self, pair, iterations=1000):
        """8. Monte Carlo Simulation"""
        if len(self.db) < 20:
            return 0
        
        # Calculate base probability
        freq = self.frequency_analysis()
        p1 = freq.get(pair[0], 0.1)
        p2 = freq.get(pair[1], 0.1)
        
        # Simulate
        wins = 0
        for _ in range(iterations):
            # Simulate 5 positions
            positions = []
            for _ in range(5):
                r = np.random.random()
                cumulative = 0
                for d, prob in freq.items():
                    cumulative += prob
                    if r <= cumulative:
                        positions.append(d)
                        break
                else:
                    positions.append(np.random.choice(list(freq.keys())))
            
            # Check if pair wins
            if pair[0] in positions and pair[1] in positions:
                wins += 1
        
        return (wins / iterations) * 100
    
    def time_series_analysis(self):
        """9. Time Series - Trend analysis"""
        trends = {}
        for digit in "0123456789":
            # Count in first half vs second half of recent
            mid = len(self.recent) // 2
            first_half = sum(1 for num in self.recent[:mid] if digit in num)
            second_half = sum(1 for num in self.recent[mid:] if digit in num)
            
            if second_half > first_half * 1.2:
                trends[digit] = "increasing"
            elif second_half < first_half * 0.8:
                trends[digit] = "decreasing"
            else:
                trends[digit] = "stable"
        
        return trends
    
    def heuristic_analysis(self):
        """10. Heuristic rules (kinh nghiệm)"""
        heuristics = {
            'lucky_ox': LUCKY_OX,
            'avoid_consecutive': [],
            'prefer_gap_5_12': []
        }
        
        # Check consecutive numbers
        gaps = self.gap_analysis()
        for d, gap in gaps.items():
            if gap > 15:
                heuristics['avoid_consecutive'].append(d)
            elif 5 <= gap <= 12:
                heuristics['prefer_gap_5_12'].append(d)
        
        return heuristics
    
    def position_weight_analysis(self):
        """11. Phân tích trọng số vị trí"""
        positions = {i: Counter() for i in range(5)}
        
        for num in self.recent:
            for i, d in enumerate(num):
                positions[i][d] += 1
        
        # Find hot numbers by position
        hot_by_pos = {}
        for pos in positions:
            if positions[pos]:
                hot_by_pos[pos] = [d for d, _ in positions[pos].most_common(2)]
        
        return hot_by_pos
    
    def entropy_analysis(self):
        """12. Entropy - Độ hỗn loạn"""
        all_digits = "".join(self.recent)
        counter = Counter(all_digits)
        total = len(all_digits)
        
        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy

# === AI INTEGRATION ===

class AIQuantumAnalyzer:
    """Sử dụng AI thật để phân tích pattern"""
    
    def __init__(self, db, algorithms):
        self.db = db
        self.algorithms = algorithms
        
    def analyze_with_gemini(self, pair_data):
        """Gửi dữ liệu đến Gemini AI để phân tích"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            
            model = genai.GenerativeModel('gemini-pro')
            
            # Prepare context
            recent_results = self.db[-15:] if len(self.db) >= 15 else self.db
            
            prompt = f"""
Phân tích xổ số 5D Bet (chọn 2 số từ 0-9, thắng nếu cả 2 xuất hiện trong 5 số kết quả).

Dữ liệu 15 kỳ gần nhất:
{', '.join(recent_results)}

Top pairs đang được đề xuất:
{json.dumps(pair_data, indent=2)}

Hãy phân tích:
1. Pattern nào đang xuất hiện?
2. Cặp số nào có xác suất cao nhất?
3. Có dấu hiệu bẫy hay không?
4. Đề xuất 1 cặp tối ưu nhất.

Trả lời dưới dạng JSON:
{{
    "analysis": "phân tích ngắn gọn",
    "recommended_pair": "XY",
    "confidence": 0-100,
    "reasoning": ["lý do 1", "lý do 2"]
}}
"""
            
            response = model.generate_content(prompt)
            result = response.text
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            st.warning(f"Gemini AI error: {e}")
        
        return None
    
    def analyze_with_nvidia(self, features):
        """Gửi đến NVIDIA AI cho phân tích nâng cao"""
        try:
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta/llama-3.1-70b-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": f"""
Phân tích thống kê nâng cao cho xổ số 5D:

Features:
{json.dumps(features, indent=2)}

Đề xuất cặp số tối ưu dựa trên phân tích đa chiều.
Trả lời JSON: {{"pair": "XY", "score": 0-100}}
"""
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 200
            }
            
            response = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content']
                import re
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                    
        except Exception as e:
            st.warning(f"NVIDIA AI error: {e}")
        
        return None

# === HỆ THỐNG TỰ HỌC ===

class SelfLearningSystem:
    """Học từ kết quả để cải thiện"""
    
    def __init__(self):
        self.history = []
        self.weights = {
            'frequency': 1.0,
            'gap': 1.0,
            'pattern': 1.0,
            'bayesian': 1.0,
            'monte_carlo': 1.0
        }
    
    def add_result(self, prediction, actual, won):
        self.history.append({
            'prediction': prediction,
            'actual': actual,
            'won': won,
            'timestamp': datetime.now().isoformat()
        })
        
        # Adjust weights based on result
        if won:
            # Increase weights for successful predictions
            for key in self.weights:
                self.weights[key] = min(self.weights[key] * 1.05, 2.0)
        else:
            # Decrease weights for failed predictions
            for key in self.weights:
                self.weights[key] = max(self.weights[key] * 0.95, 0.5)
    
    def get_adjusted_score(self, base_score, algorithm_type):
        return base_score * self.weights.get(algorithm_type, 1.0)

# === XỬ LÝ DỮ LIỆU ===

def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    nums = [n for n in re.findall(r"\d{5}", clean_text) if n]
    seen = set()
    return [n for n in nums if not (n in seen or seen.add(n))]

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"history": [], "learning_weights": {}}
    return {"history": [], "learning_weights": {}}

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# === PREDICTION ENGINE ===

def calculate_comprehensive_prediction(db, learning_system):
    """Tính toán dự đoán toàn diện từ 12 thuật toán"""
    
    engine = MultiAlgorithmEngine(db)
    
    # Run all algorithms
    freq = engine.frequency_analysis()
    cycles = engine.cycle_detection()
    gaps = engine.gap_analysis()
    hot, cold = engine.hot_cold_analysis()
    patterns = engine.pattern_detection()
    trends = engine.time_series_analysis()
    heuristics = engine.heuristic_analysis()
    pos_weights = engine.position_weight_analysis()
    entropy = engine.entropy_analysis()
    markov = engine.markov_chain()
    
    # Generate all pairs and score them
    all_pairs = []
    
    for p in combinations("0123456789", 2):
        pair = "".join(p)
        scores = {}
        
        # 1. Frequency score
        freq_score = (freq.get(p[0], 0.1) + freq.get(p[1], 0.1)) * 100
        scores['frequency'] = freq_score
        
        # 2. Gap score (golden zone 5-12)
        gap1, gap2 = gaps.get(p[0], 10), gaps.get(p[1], 10)
        gap_score = 0
        if 5 <= gap1 <= 12: gap_score += 50
        elif 2 <= gap1 <= 4: gap_score += 25
        if 5 <= gap2 <= 12: gap_score += 50
        elif 2 <= gap2 <= 4: gap_score += 25
        scores['gap'] = gap_score
        
        # 3. Pattern score
        pattern_score = 0
        if p[0] in patterns['repeating'] or p[1] in patterns['repeating']:
            pattern_score += 30
        scores['pattern'] = pattern_score
        
        # 4. Bayesian probability
        bayes_prob = engine.bayesian_probability(p)
        scores['bayesian'] = bayes_prob
        
        # 5. Monte Carlo simulation
        mc_prob = engine.monte_carlo_simulation(p, iterations=500)
        scores['monte_carlo'] = mc_prob
        
        # 6. Trend score
        trend_score = 0
        if trends.get(p[0]) == "increasing": trend_score += 25
        if trends.get(p[1]) == "increasing": trend_score += 25
        scores['trend'] = trend_score
        
        # 7. Heuristic score
        heuristic_score = 0
        if p[0] in heuristics['prefer_gap_5_12']: heuristic_score += 20
        if p[1] in heuristics['prefer_gap_5_12']: heuristic_score += 20
        if any(int(d) in LUCKY_OX for d in p): heuristic_score += 15
        scores['heuristic'] = heuristic_score
        
        # 8. Position weight score
        pos_score = 0
        for pos, hot_nums in pos_weights.items():
            if p[0] in hot_nums: pos_score += 10
            if p[1] in hot_nums: pos_score += 10
        scores['position'] = pos_score
        
        # 9. Hot/Cold bonus
        hotcold_score = 0
        if p[0] in hot: hotcold_score += 20
        if p[1] in hot: hotcold_score += 20
        scores['hotcold'] = hotcold_score
        
        # 10. Entropy adjustment
        entropy_adj = 1.0
        if entropy < 2.8:  # Low entropy - predictable
            entropy_adj = 1.2
        elif entropy > 3.3:  # High entropy - chaotic
            entropy_adj = 0.8
        scores['entropy_adj'] = entropy_adj
        
        # Calculate weighted total score
        total_score = 0
        for algo, score in scores.items():
            if algo != 'entropy_adj':
                weight = learning_system.weights.get(algo, 1.0)
                total_score += score * weight
        
        total_score *= scores.get('entropy_adj', 1.0)
        
        all_pairs.append({
            'pair': pair,
            'total_score': total_score,
            'scores': scores,
            'gap': (gap1, gap2),
            'freq': (freq.get(p[0], 0), freq.get(p[1], 0))
        })
    
    # Sort by total score
    all_pairs.sort(key=lambda x: x['total_score'], reverse=True)
    
    return {
        'pairs': all_pairs[:10],
        'hot_numbers': hot,
        'cold_numbers': cold,
        'gaps': gaps,
        'entropy': entropy,
        'trends': trends,
        'algorithms': engine
    }

# === GIAO DIỆN CHÍNH ===

st.markdown('<h1 class="main-header">🧠 TITAN V52 - AI QUANTUM PREDICTION</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888; font-size:12px;">12 Thuật Toán | AI Thực Tế | Tự Học | Multi-Layer Analysis</p>', unsafe_allow_html=True)

# Initialize
if "data" not in st.session_state:
    st.session_state.data = load_data()
if "learning_system" not in st.session_state:
    st.session_state.learning_system = SelfLearningSystem()
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Nhập Kết Quả",
    "🎯 Dự Đoán AI",
    "🔥 Số Bệt",
    "⏳ Số Gan",
    "📊 Lịch Sử"
])

with tab1:
    st.markdown('<div class="neon-box neon-cyan"><h3>📥 Nhập kết quả (kỳ mới nhất ở dưới)</h3></div>', unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Dán kết quả tại đây:",
        height=200,
        placeholder="84890\n07119\n33627\n..."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Lưu Dữ Liệu", use_container_width=True):
            nums = get_nums(user_input)
            if nums:
                st.session_state.data['last_input'] = nums
                save_data(st.session_state.data)
                st.success(f"✅ Đã lưu {len(nums)} kỳ")
            else:
                st.error("❌ Không tìm thấy số hợp lệ")
    
    with col2:
        if st.button(" Phân Tích AI", type="primary", use_container_width=True):
            nums = get_nums(user_input)
            if len(nums) >= 15:
                # Check previous prediction
                if st.session_state.last_prediction and len(nums) > 0:
                    last_actual = nums[-1]
                    pred_pair = st.session_state.last_prediction['pairs'][0]['pair']
                    won = all(d in last_actual for d in pred_pair)
                    
                    st.session_state.learning_system.add_result(pred_pair, last_actual, won)
                    
                    # Add to history
                    st.session_state.data['history'].insert(0, {
                        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'result': last_actual,
                        'prediction': pred_pair,
                        'won': won
                    })
                    save_data(st.session_state.data)
                
                # New prediction
                st.session_state.prediction_result = calculate_comprehensive_prediction(
                    nums, 
                    st.session_state.learning_system
                )
                st.session_state.last_prediction = st.session_state.prediction_result
                st.session_state.current_db = nums
                st.rerun()
            else:
                st.error(f"⚠️ Cần ít nhất 15 kỳ (hiện có: {len(nums)})")

with tab2:
    if "prediction_result" in st.session_state:
        res = st.session_state.prediction_result
        
        # Metrics
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(res['hot_numbers'])}</div>
                <div class="metric-label">SỐ NÓNG</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{res['entropy']:.2f}</div>
                <div class="metric-label">ENTROPY</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.current_db)}</div>
                <div class="metric-label">SỐ KỲ</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_weight = sum(st.session_state.learning_system.weights.values()) / len(st.session_state.learning_system.weights)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_weight:.2f}x</div>
                <div class="metric-label">AI WEIGHT</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Top prediction
        top_pair = res['pairs'][0]
        st.markdown(f"""
        <div class="neon-box neon-gold">
            <div style="text-align:center; font-size:14px; color:#888;">🎯 BẠCH THỦ 2 SỐ</div>
            <div class="prediction-number">{top_pair['pair'][0]} - {top_pair['pair'][1]}</div>
            <div style="text-align:center;">
                <span class="tag tag-ai">SCORE: {top_pair['total_score']:.1f}</span>
                <span class="tag tag-bet">GAN: {top_pair['gap'][0]}/{top_pair['gap'][1]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # AI Analysis
        if st.button("🤖 Yêu Cầu AI Phân Tích Sâu"):
            with st.spinner("Đang kết nối AI..."):
                ai_analyzer = AIQuantumAnalyzer(
                    st.session_state.current_db,
                    res['algorithms']
                )
                
                pair_data = [{'pair': p['pair'], 'score': p['total_score']} for p in res['pairs'][:5]]
                
                gemini_result = ai_analyzer.analyze_with_gemini(pair_data)
                
                if gemini_result:
                    st.markdown(f"""
                    <div class="neon-box neon-pink">
                        <h4>🧠 PHÂN TÍCH TỪ GEMINI AI:</h4>
                        <p>{gemini_result.get('analysis', '')}</p>
                        <div style="margin-top:15px;">
                            <strong>Cặp đề xuất:</strong> {gemini_result.get('recommended_pair', 'N/A')}<br>
                            <strong>Độ tin cậy:</strong> {gemini_result.get('confidence', 0):.0f}%
                        </div>
                        <div style="margin-top:10px;">
                            <strong>Lý do:</strong>
                            <ul>
                                {''.join([f'<li>{r}</li>' for r in gemini_result.get('reasoning', [])])}
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Top 5 pairs
        st.markdown('<h3 style="color:#00ffff;">📊 TOP 5 CẶP SỐ</h3>', unsafe_allow_html=True)
        
        for i, p in enumerate(res['pairs'][:5]):
            tags = []
            if p['gap'][0] >= 5 and p['gap'][0] <= 12:
                tags.append('<span class="tag tag-gan">GAN VÀNG</span>')
            if p['pair'][0] in res['hot_numbers'] or p['pair'][1] in res['hot_numbers']:
                tags.append('<span class="tag tag-hot">HOT</span>')
            if p['scores']['bayesian'] > 60:
                tags.append('<span class="tag tag-ai">BAYES</span>')
            
            st.markdown(f"""
            <div class="neon-box neon-cyan" style="padding:15px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:28px; font-weight:900; color:#FFD700;">#{i+1}: {p['pair'][0]} - {p['pair'][1]}</span>
                    <span style="font-size:24px; color:#00ff40;">{p['total_score']:.1f}</span>
                </div>
                <div style="margin-top:10px;">{''.join(tags)}</div>
                <div style="font-size:11px; color:#888; margin-top:5px;">
                    Freq: {p['freq'][0]:.1%}/{p['freq'][1]:.1%} | 
                    Gap: {p['gap'][0]}/{p['gap'][1]} | 
                    Monte Carlo: {p['scores']['monte_carlo']:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed algorithm breakdown
        if st.checkbox("📈 Hiển thị chi tiết 12 thuật toán"):
            st.markdown('<h3>🔬 PHÂN TÍCH TỪNG THUẬT TOÁN</h3>', unsafe_allow_html=True)
            
            algo_names = {
                'frequency': 'Tần Suất',
                'gap': 'Số Gan',
                'pattern': 'Pattern',
                'bayesian': 'Bayes',
                'monte_carlo': 'Monte Carlo',
                'trend': 'Xu Hướng',
                'heuristic': 'Kinh Nghiệm',
                'position': 'Vị Trí',
                'hotcold': 'Nóng/Lạnh'
            }
            
            for algo, name in algo_names.items():
                score = top_pair['scores'].get(algo, 0)
                st.markdown(f"""
                <div class="algorithm-card">
                    <strong>{name}:</strong> {score:.1f} điểm
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{min(score, 100)}%;">{score:.1f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
    else:
        st.info("👈 Vui lòng nhập dữ liệu và nhấn 'Phân Tích AI' ở tab Nhập Kết Quả")

with tab3:
    st.markdown('<h2>🔥 SỐ BỆT (Lặp Nhiều Lần)</h2>', unsafe_allow_html=True)
    
    if "prediction_result" in st.session_state:
        res = st.session_state.prediction_result
        
        # Find repeating patterns
        db = st.session_state.current_db
        repeating = []
        
        for i in range(len(db) - 1):
            common = set(db[i]) & set(db[i+1])
            if len(common) >= 2:
                repeating.append({
                    'period': f"{i+1} → {i+2}",
                    'numbers': list(common),
                    'result1': db[i],
                    'result2': db[i+1]
                })
        
        if repeating:
            for r in repeating[-10:]:
                st.markdown(f"""
                <div class="neon-box neon-pink">
                    <strong>Kỳ {r['period']}</strong>: {' '.join(r['numbers'])}<br>
                    <small>{r['result1']} → {r['result2']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Không tìm thấy pattern bệt đặc biệt")

with tab4:
    st.markdown('<h2>⏳ SỐ GAN (Lâu Chưa Ra)</h2>', unsafe_allow_html=True)
    
    if "prediction_result" in st.session_state:
        res = st.session_state.prediction_result
        
        gaps = res['gaps']
        sorted_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
        
        for digit, gap in sorted_gaps:
            if gap >= 5:
                tag_class = "tag-gan" if gap >= 10 else "tag"
                st.markdown(f"""
                <div class="neon-box" style="border-color: {'#ff0000' if gap >= 15 else '#ff9900'};">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="font-size:24px; font-weight:900;">Số {digit}</span>
                        <span class="tag {tag_class}">{gap} kỳ</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{min(gap*5, 100)}%; background:{'#ff0000' if gap >= 15 else '#ff9900'};">
                            {gap} kỳ
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

with tab5:
    st.markdown('<h2>📊 LỊCH SỬ DỰ ĐOÁN</h2>', unsafe_allow_html=True)
    
    if st.session_state.data.get('history'):
        df = pd.DataFrame(st.session_state.data['history'][:50])
        
        def color_result(val):
            return 'color: #00ff40; font-weight: 900; text-shadow: 0 0 10px #00ff40' if val else 'color: #ff0040; font-weight: 900; text-shadow: 0 0 10px #ff0040'
        
        st.dataframe(
            df.style.applymap(color_result, subset=['won']),
            use_container_width=True,
            hide_index=True
        )
        
        # Statistics
        total = len(st.session_state.data['history'])
        wins = sum(1 for h in st.session_state.data['history'] if h.get('won'))
        win_rate = (wins/total*100) if total > 0 else 0
        
        st.markdown(f"""
        <div class="neon-box neon-gold" style="text-align:center; margin-top:20px;">
            <div style="font-size:18px;">TỶ LỆ THẮNG</div>
            <div style="font-size:48px; font-weight:900; color:{'#00ff40' if win_rate >= 40 else '#ff0040'};">
                {win_rate:.1f}%
            </div>
            <div style="font-size:16px;">({wins}/{total} kỳ)</div>
            <div class="confidence-bar" style="margin-top:15px;">
                <div class="confidence-fill" style="width:{win_rate}%; background:{'#00ff40' if win_rate >= 40 else '#ff0040'};">
                    {win_rate:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Learning weights
        st.markdown('<h3> TRỌNG SỐ HỌC TỪ AI</h3>', unsafe_allow_html=True)
        
        weights = st.session_state.learning_system.weights
        for algo, weight in weights.items():
            st.markdown(f"""
            <div class="algorithm-card">
                <strong>{algo}:</strong> {weight:.2f}x
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{(weight/2)*100}%; background:#9900ff;">
                        {weight:.2f}x
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Chưa có lịch sử dự đoán")

# === FOOTER ===
st.markdown("""
<div style="text-align:center; color:#444; font-size:10px; margin-top:50px; padding-top:20px; border-top:1px solid #333;">
    🧠 TITAN V52 - AI QUANTUM PREDICTION ENGINE<br>
    12 Algorithms | Real AI Integration | Self-Learning System<br>
    <i>Tool hỗ trợ phân tích - Không đảm bảo 100% - Quản lý vốn thông minh</i>
</div>
""", unsafe_allow_html=True)