import streamlit as st
import re, pandas as pd, numpy as np, math
from collections import Counter, defaultdict
from itertools import combinations
import hashlib, time

# === CẤU HÌNH ===
LUCKY_OX = [0, 2, 5, 6, 7, 8]
st.set_page_config(page_title="TITAN V52 - QUANTUM META", page_icon="⚛️", layout="centered")

# === CSS QUANTUM ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
    
    .stApp {
        background: radial-gradient(circle at center, #0a0a1a 0%, #000000 100%);
        color: #FFD700;
        font-family: 'Orbitron', monospace;
    }
    
    .quantum-header {
        font-size: 32px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00ffff, #ff00ff, #ffff00, #00ffff);
        background-size: 400% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: quantum-flow 4s linear infinite;
    }
    
    @keyframes quantum-flow {
        0% { background-position: 0% 50%; }
        100% { background-position: 400% 50%; }
    }
    
    .approach-tabs {
        display: flex;
        gap: 10px;
        margin: 15px 0;
        flex-wrap: wrap;
    }
    
    .tab {
        flex: 1;
        min-width: 100px;
        padding: 12px;
        background: #1a1a2e;
        border: 2px solid #333;
        border-radius: 10px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        font-size: 11px;
    }
    
    .tab.active {
        border-color: #00ffff;
        background: #0a1a2e;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
    }
    
    .tab.win { border-color: #00ff40; }
    .tab.lose { border-color: #ff0040; }
    .tab.chaos { border-color: #ffff00; }
    
    .quantum-box {
        background: linear-gradient(135deg, #1a0a2e, #0a1a2e);
        border: 2px solid;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        position: relative;
        overflow: hidden;
    }
    
    .quantum-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.05), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .number-display {
        font-size: 48px;
        font-weight: 900;
        letter-spacing: 10px;
        text-align: center;
        margin: 15px 0;
        text-shadow: 0 0 20px currentColor;
    }
    
    .confidence-quantum {
        height: 30px;
        background: #0a0a1a;
        border-radius: 15px;
        overflow: hidden;
        position: relative;
        margin: 10px 0;
    }
    
    .confidence-quantum::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00, #ffff00, #ff0000);
        background-size: 200% 100%;
        animation: confidence-wave 2s linear infinite;
        opacity: 0.3;
    }
    
    @keyframes confidence-wave {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    
    .metric-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 10px;
        margin: 15px 0;
    }
    
    .metric-item {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        border: 1px solid #333;
    }
    
    .tag-quantum {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 9px;
        font-weight: bold;
        margin: 2px;
        background: linear-gradient(135deg, #ff00ff, #00ffff);
        color: #fff;
    }
    
    .history-table {
        font-size: 11px;
    }
    
    .win-text { color: #00ff40; font-weight: 900; }
    .lose-text { color: #ff0040; font-weight: 900; }
    
    .superposition {
        display: flex;
        justify-content: space-around;
        align-items: center;
        padding: 15px;
        background: #0a0a1a;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .state {
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        flex: 1;
        margin: 0 5px;
        opacity: 0.6;
        transition: all 0.3s;
    }
    
    .state.active {
        opacity: 1;
        box-shadow: 0 0 20px currentColor;
    }
</style>
""", unsafe_allow_html=True)

# === 3 APPROACHES PARALLEL ===

class BayesianApproach:
    """
    Approach 1: Bayesian Inference
    - Update prior beliefs with new evidence
    - Calculate posterior probability for each pair
    - Use Beta distribution for win/loss modeling
    """
    def __init__(self):
        self.prior_alpha = 1  # Success prior
        self.prior_beta = 1   # Failure prior
        
    def update(self, win):
        if win:
            self.prior_alpha += 1
        else:
            self.prior_beta += 1
    
    def get_posterior_mean(self):
        return self.prior_alpha / (self.prior_alpha + self.prior_beta)
    
    def predict(self, db, history):
        if len(db) < 15:
            return None, 0
        
        # Calculate posterior for each pair
        pairs_score = []
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            
            # Likelihood from data
            freq = sum(1 for n in db[-30:] if set(pair).issubset(set(n)))
            gan = self._calculate_gan(db, pair)
            
            # Bayesian score
            likelihood = freq / 30 if freq > 0 else 0.01
            posterior = (self.prior_alpha + freq) / (self.prior_alpha + self.prior_beta + 30)
            
            # Adjust by gan
            if 4 <= gan <= 10:
                posterior *= 1.5
            
            pairs_score.append((pair, posterior * 100, gan, freq))
        
        pairs_score.sort(key=lambda x: x[1], reverse=True)
        confidence = self.get_posterior_mean() * 100
        
        return pairs_score[:5], confidence
    
    def _calculate_gan(self, db, pair):
        gan = 0
        for num in reversed(db):
            if not set(pair).issubset(set(num)):
                gan += 1
            else:
                break
        return gan

class FractalApproach:
    """
    Approach 2: Fractal Time Series Analysis
    - Detect self-similar patterns
    - Calculate Hurst exponent
    - Find fractal dimension
    - Exploit long-term memory
    """
    def __init__(self):
        self.window_sizes = [5, 10, 20]
        
    def calculate_hurst(self, series):
        """Calculate Hurst exponent"""
        if len(series) < 10:
            return 0.5
        
        lags = range(2, min(20, len(series)//2))
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        
        try:
            hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
            return hurst
        except:
            return 0.5
    
    def detect_fractal_patterns(self, db):
        """Find repeating patterns at different scales"""
        patterns = defaultdict(list)
        
        for window in self.window_sizes:
            if len(db) < window * 2:
                continue
            
            for i in range(len(db) - window):
                chunk = db[i:i+window]
                chunk_hash = hashlib.md5("".join(chunk).encode()).hexdigest()[:8]
                patterns[chunk_hash].append(i)
        
        # Find patterns that repeat
        repeating = {k: v for k, v in patterns.items() if len(v) >= 2}
        return repeating
    
    def predict(self, db, history):
        if len(db) < 20:
            return None, 0
        
        hurst = self.calculate_hurst([sum(int(d) for d in n) for n in db])
        patterns = self.detect_fractal_patterns(db)
        
        # If hurst > 0.5, trend following
        # If hurst < 0.5, mean reversion
        # If hurst ≈ 0.5, random walk
        
        pairs_score = []
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            score = 50  # Base score
            
            if hurst > 0.6:
                # Trend following - bet on recent
                recent_freq = sum(1 for n in db[-10:] if set(pair).issubset(set(n)))
                score += recent_freq * 10
            elif hurst < 0.4:
                # Mean reversion - bet on absent
                gan = self._calculate_gan(db, pair)
                if 5 <= gan <= 15:
                    score += 30
            else:
                # Random - use frequency
                freq = sum(1 for n in db[-30:] if set(pair).issubset(set(n)))
                if 3 <= freq <= 7:
                    score += 20
            
            pairs_score.append((pair, score, self._calculate_gan(db, pair), 0))
        
        pairs_score.sort(key=lambda x: x[1], reverse=True)
        confidence = 40 + abs(hurst - 0.5) * 100
        
        return pairs_score[:5], confidence
    
    def _calculate_gan(self, db, pair):
        gan = 0
        for num in reversed(db):
            if not set(pair).issubset(set(num)):
                gan += 1
            else:
                break
        return gan

class MetaLearningApproach:
    """
    Approach 3: Ensemble Meta-Learning
    - Combine multiple weak learners
    - Adaptive weighting based on recent performance
    - Stacking with meta-classifier
    """
    def __init__(self):
        self.weights = {'bayesian': 0.33, 'fractal': 0.33, 'heuristic': 0.34}
        self.recent_performance = {'bayesian': [], 'fractal': [], 'heuristic': []}
        
    def update_weights(self, approach_name, correct):
        self.recent_performance[approach_name].append(1 if correct else 0)
        
        # Keep only last 10
        for key in self.recent_performance:
            self.recent_performance[key] = self.recent_performance[key][-10:]
        
        # Calculate accuracy for each
        accuracies = {}
        for key, perf in self.recent_performance.items():
            if perf:
                accuracies[key] = sum(perf) / len(perf)
            else:
                accuracies[key] = 0.33
        
        # Normalize weights
        total = sum(accuracies.values())
        if total > 0:
            self.weights = {k: v/total for k, v in accuracies.items()}
    
    def heuristic_score(self, db, pair):
        """Simple heuristic rules"""
        score = 0
        
        # Frequency
        freq = sum(1 for n in db[-30:] if set(pair).issubset(set(n)))
        if 2 <= freq <= 8:
            score += 30
        
        # Gan
        gan = 0
        for num in reversed(db):
            if not set(pair).issubset(set(num)):
                gan += 1
            else:
                break
        
        if 5 <= gan <= 12:
            score += 40
        
        # Streak
        streak = 0
        for num in reversed(db):
            if set(pair).issubset(set(num)):
                streak += 1
            else:
                break
        
        if streak == 0:
            score += 20
        elif streak >= 3:
            score -= 40
        
        return score, gan
    
    def predict(self, db, history):
        if len(db) < 15:
            return None, 0
        
        # Get predictions from each approach
        bayesian = BayesianApproach()
        fractal = FractalApproach()
        
        bayes_pairs, bayes_conf = bayesian.predict(db, history)
        fractal_pairs, fractal_conf = fractal.predict(db, history)
        
        # Heuristic
        heuristic_pairs = []
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            score, gan = self.heuristic_score(db, pair)
            heuristic_pairs.append((pair, score, gan, 0))
        heuristic_pairs.sort(key=lambda x: x[1], reverse=True)
        heuristic_conf = 50
        
        # Ensemble voting
        all_pairs = set()
        for pairs in [bayes_pairs, fractal_pairs, heuristic_pairs]:
            if pairs:
                for p in pairs:
                    all_pairs.add(p[0])
        
        ensemble_scores = []
        for pair in all_pairs:
            total_score = 0
            
            # Bayesian vote
            if bayes_pairs:
                for i, (p, score, _, _) in enumerate(bayes_pairs):
                    if p == pair:
                        total_score += score * self.weights['bayesian'] * (5-i)
                        break
            
            # Fractal vote
            if fractal_pairs:
                for i, (p, score, _, _) in enumerate(fractal_pairs):
                    if p == pair:
                        total_score += score * self.weights['fractal'] * (5-i)
                        break
            
            # Heuristic vote
            if heuristic_pairs:
                for i, (p, score, _, _) in enumerate(heuristic_pairs):
                    if p == pair:
                        total_score += score * self.weights['heuristic'] * (5-i)
                        break
            
            ensemble_scores.append((pair, total_score, 0, 0))
        
        ensemble_scores.sort(key=lambda x: x[1], reverse=True)
        confidence = (bayes_conf + fractal_conf + heuristic_conf) / 3
        
        return ensemble_scores[:5], confidence

# === QUANTUM META-LEARNING ENGINE ===

class QuantumMetaEngine:
    """
    Superposition of all approaches
    Collapses to best one based on measurement (results)
    """
    def __init__(self, db, history):
        self.db = db
        self.history = history
        self.bayesian = BayesianApproach()
        self.fractal = FractalApproach()
        self.meta = MetaLearningApproach()
        
        # Initialize from history
        self._learn_from_history()
    
    def _learn_from_history(self):
        """Update all approaches from history"""
        for h in self.history[:20]:  # Last 20
            if 'KQ' in h and 'Dự đoán' in h:
                win = '🔥' in h['KQ']
                self.bayesian.update(win)
    
    def get_approach_accuracies(self):
        """Calculate accuracy for each approach"""
        acc = {}
        for name, perf in self.meta.recent_performance.items():
            if perf:
                acc[name] = sum(perf) / len(perf) * 100
            else:
                acc[name] = 50
        return acc
    
    def should_skip(self, confidence, approach_accuracies):
        """Decide whether to skip this round"""
        # Skip if all approaches have low accuracy
        avg_acc = sum(approach_accuracies.values()) / len(approach_accuracies)
        if avg_acc < 35:
            return True, "TẤT CẢ APPROACH ĐANG THUA"
        
        # Skip if confidence too low
        if confidence < 45:
            return True, "ĐỘ TIN CẬY THẤP"
        
        # Skip if entropy too high
        entropy = self._calculate_entropy()
        if entropy > 3.4:
            return True, "QUÁ HỖN LOẠN"
        
        return False, "OK"
    
    def _calculate_entropy(self):
        if len(self.db) < 10:
            return 3.5
        
        recent = "".join(self.db[-10:])
        counter = Counter(recent)
        total = len(recent)
        
        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def predict(self):
        """Quantum prediction - superposition of approaches"""
        if len(self.db) < 15:
            return None
        
        # Get predictions from all approaches
        bayes_pairs, bayes_conf = self.bayesian.predict(self.db, self.history)
        fractal_pairs, fractal_conf = self.fractal.predict(self.db, self.history)
        meta_pairs, meta_conf = self.meta.predict(self.db, self.history)
        
        # Get approach accuracies
        approach_acc = self.get_approach_accuracies()
        
        # Choose best approach based on recent performance
        best_approach = max(approach_acc.keys(), key=lambda k: approach_acc[k])
        
        if best_approach == 'bayesian':
            final_pairs = bayes_pairs
            confidence = bayes_conf
        elif best_approach == 'fractal':
            final_pairs = fractal_pairs
            confidence = fractal_conf
        else:
            final_pairs = meta_pairs
            confidence = meta_conf
        
        # Check if should skip
        skip, skip_reason = self.should_skip(confidence, approach_acc)
        
        # Top 8
        single_pool = Counter("".join(self.db[-50:]))
        top8 = "".join([d for d, _ in single_pool.most_common(8)])
        
        return {
            'pairs': final_pairs,
            'confidence': confidence,
            'skip': skip,
            'skip_reason': skip_reason,
            'approach_accuracies': approach_acc,
            'best_approach': best_approach,
            'top8': top8,
            'bayesian_conf': bayes_conf,
            'fractal_conf': fractal_conf,
            'meta_conf': meta_conf
        }

# === XỬ LÝ DỮ LIỆU ===

def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    nums = [n for n in re.findall(r"\d{5}", clean_text) if n]
    seen = set()
    return [n for n in nums if not (n in seen or seen.add(n))]

# === GIAO DIỆN ===

st.markdown('<h1 class="quantum-header">⚛️ TITAN V52 - QUANTUM META-LEARNING</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888; font-size:10px;">3 Approaches | Adaptive Weighting | Quantum Selection</p>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_area("📥 Kết quả (kỳ mới nhất ở dưới):", height=100, 
                          placeholder="07988\n35782")

col1, col2 = st.columns(2)
with col1:
    if st.button("⚛️ KÍCH HOẠT", type="primary", use_container_width=True):
        nums = get_nums(user_input)
        if len(nums) >= 15:
            # Check result
            if "last_pred" in st.session_state and nums:
                lp = st.session_state.last_pred
                last = nums[-1]
                if lp and lp['pairs'] and not lp.get('skip', False):
                    best = lp['pairs'][0][0]
                    win = all(d in last for d in best)
                    
                    # Update meta-learning
                    if 'best_approach' in lp:
                        lp['meta'].update_weights(lp['best_approach'], win)
                    
                    st.session_state.history.insert(0, {
                        'Kỳ': last,
                        'Dự đoán': best,
                        'KQ': '🔥' if win else '❌'
                    })
            
            engine = QuantumMetaEngine(nums, st.session_state.history)
            st.session_state.last_pred = engine.predict()
            st.session_state.last_pred['meta'] = engine.meta  # Save for learning
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
    
    # === APPROACH TABS ===
    st.markdown("""<div style="text-align:center; margin:15px 0; font-size:14px;">📊 SO SÁNH 3 APPROACHES</div>""", unsafe_allow_html=True)
    
    acc = res['approach_accuracies']
    
    st.markdown(f"""
    <div class="approach-tabs">
        <div class="tab {'active' if res['best_approach'] == 'bayesian' else ''} {'win' if acc['bayesian'] >= 40 else 'lose' if acc['bayesian'] < 35 else ''}">
            <div style="font-weight:900; font-size:12px;">BAYESIAN</div>
            <div style="color:{'#00ff40' if acc['bayesian'] >= 40 else '#ff0040' if acc['bayesian'] < 35 else '#ffff00'}; font-size:16px;">{acc['bayesian']:.0f}%</div>
        </div>
        <div class="tab {'active' if res['best_approach'] == 'fractal' else ''} {'win' if acc['fractal'] >= 40 else 'lose' if acc['fractal'] < 35 else ''}">
            <div style="font-weight:900; font-size:12px;">FRACTAL</div>
            <div style="color:{'#00ff40' if acc['fractal'] >= 40 else '#ff0040' if acc['fractal'] < 35 else '#ffff00'}; font-size:16px;">{acc['fractal']:.0f}%</div>
        </div>
        <div class="tab {'active' if res['best_approach'] == 'heuristic' else ''} {'win' if acc['heuristic'] >= 40 else 'lose' if acc['heuristic'] < 35 else ''}">
            <div style="font-weight:900; font-size:12px;">ENSEMBLE</div>
            <div style="color:{'#00ff40' if acc['heuristic'] >= 40 else '#ff0040' if acc['heuristic'] < 35 else '#ffff00'}; font-size:16px;">{acc['heuristic']:.0f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === SKIP WARNING ===
    if res['skip']:
        st.markdown(f"""
        <div class="quantum-box" style="border-color:#ff0040;">
            <div style="font-size:24px; color:#ff0040; text-align:center; font-weight:900;">
                ⚠️ KHÔNG NÊN ĐÁNH
            </div>
            <div style="text-align:center; margin-top:10px; color:#ff6680;">{res['skip_reason']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # === METRICS ===
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-item">
                <div style="font-size:9px; color:#888;">CONFIDENCE</div>
                <div style="font-size:20px; font-weight:900; color:#00ffff;">{res['confidence']:.0f}%</div>
            </div>
            <div class="metric-item">
                <div style="font-size:9px; color:#888;">BEST APPROACH</div>
                <div style="font-size:16px; font-weight:900; color:#FFD700;">{res['best_approach'].upper()}</div>
            </div>
            <div class="metric-item">
                <div style="font-size:9px; color:#888;">TOP PAIR</div>
                <div style="font-size:20px; font-weight:900; color:#00ff40;">{res['pairs'][0][0]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # === TOP PAIR ===
        st.markdown("""<div style="text-align:center; margin:15px 0; font-size:14px;">🎯 CẶP VIP</div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="quantum-box" style="border-color:#00ffff;">
            <div class="number-display" style="color:#00ffff;">{res['pairs'][0][0][0]} - {res['pairs'][0][0][1]}</div>
            <div style="text-align:center;">
                <span class="tag-quantum">SCORE: {res['pairs'][0][1]:.0f}</span>
                <span class="tag-quantum">GAN: {res['pairs'][0][2]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # === TOP 5 ===
        st.markdown("""<div style="text-align:center; margin:15px 0; font-size:14px;">🎯 TOP 5</div>""", unsafe_allow_html=True)
        
        for i, (pair, score, gan, _) in enumerate(res['pairs'][:5]):
            tags = ""
            if 4 <= gan <= 10:
                tags += '<span class="tag-quantum">GAN VÀNG</span>'
            
            st.markdown(f"""
            <div style="background:#1a1a2e; border-radius:8px; padding:10px; margin:5px 0; 
                        border-left: 4px solid {'#00ff40' if i == 0 else '#444'};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:22px; font-weight:900; color:#FFD700;">{pair[0]}-{pair[1]}</span>
                    <span style="font-size:16px; color:#00ff40;">{score:.0f}</span>
                </div>
                <div style="margin-top:5px;">{tags}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # === TOP 8 ===
        st.markdown(f"""
        <div class="quantum-box" style="border-color:#ffff00; margin-top:15px;">
            <div style="font-size:11px; color:#888; text-align:center;">ĐỘ PHỦ SẢNH</div>
            <div style="font-size:28px; font-weight:900; letter-spacing:6px; text-align:center; color:#ffff00;">{res['top8']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # === HISTORY ===
    if st.session_state.history:
        st.markdown("""<div style="text-align:center; margin:20px 0; font-size:14px;">📋 LỊCH SỬ</div>""", unsafe_allow_html=True)
        
        df = pd.DataFrame(st.session_state.history[:15])
        
        def color_kq(val):
            return 'color: #00ff40; font-weight: 900' if '🔥' in val else 'color: #ff0040; font-weight: 900'
        
        st.dataframe(df.style.applymap(color_kq, subset=['KQ']), 
                     use_container_width=True, hide_index=True)
        
        # Win rate
        wins = sum(1 for h in st.session_state.history if '🔥' in h['KQ'])
        total = len(st.session_state.history)
        rate = (wins/total*100) if total > 0 else 0
        
        color_rate = "#00ff40" if rate >= 40 else ("#ffff00" if rate >= 30 else "#ff0040")
        
        st.markdown(f"""
        <div class="quantum-box" style="border-color:{color_rate}; margin-top:15px;">
            <div style="font-size:12px; text-align:center;">TỶ LỆ THẮNG</div>
            <div style="font-size:32px; font-weight:900; text-align:center; color:{color_rate};">{rate:.1f}% ({wins}/{total})</div>
            <div class="confidence-quantum">
                <div style="width:{rate}%; height:100%; background:{color_rate}; opacity:0.7;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#444; font-size:9px; margin-top:20px; padding-top:10px; border-top:1px solid #333;">
    ⚛️ TITAN V52 - QUANTUM META-LEARNING | 3 Parallel Approaches | Adaptive Selection<br>
    <i>Bayesian Inference | Fractal Analysis | Ensemble Learning</i>
</div>
""", unsafe_allow_html=True)