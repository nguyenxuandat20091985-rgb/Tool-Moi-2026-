# ==============================================================================
# TITAN AI ENGINE - Single File Lottery Prediction System
# 3 Numbers from 5 Digits Prediction | Multi-Algorithm AI
# ==============================================================================
# Requirements: streamlit, pandas, numpy (only)
# Usage: streamlit run titan_ai.py
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import random
import re
import math
import time
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. PAGE CONFIG & CSS - Mobile Optimized
# ==============================================================================

st.set_page_config(
    page_title="🎯 TITAN AI ENGINE",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean CSS - No HTML errors, mobile responsive
st.markdown("""
<style>
    /* Global */
    .stApp { background: #010409; color: #e6edf3; font-family: system-ui, -apple-system, sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Main Number Cards */
    .main-card {
        background: linear-gradient(135deg, #161b22, #0d1117);
        border: 3px solid #ff5858;
        border-radius: 16px;
        padding: 25px 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(255,88,88,0.25);
    }
    .main-num { font-size: 52px; font-weight: 900; color: #ff5858; line-height: 1; }
    .main-label { font-size: 12px; color: #8b949e; margin-top: 10px; text-transform: uppercase; }
    
    /* Support Number Cards */
    .sup-card {
        background: #161b22;
        border: 2px solid #58a6ff;
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
        color: #58a6ff;
        font-weight: 800;
        font-size: 32px;
    }
    
    /* Status Banner */
    .status-banner {
        padding: 12px 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        font-size: 15px;
        margin: 15px 0;
    }
    .status-ok { background: rgba(35,134,54,0.2); border: 1px solid #238636; color: #3fb950; }
    .status-warn { background: rgba(210,153,34,0.2); border: 1px solid #d29922; color: #f0b429; }
    .status-stop { background: rgba(218,54,51,0.2); border: 1px solid #da3633; color: #f85149; }
    
    /* Stats Table */
    .dataframe { background: #0d1117 !important; color: #e6edf3 !important; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white; border: none; border-radius: 10px;
        font-weight: 700; padding: 14px 32px; font-size: 15px;
    }
    
    /* Mobile Responsive */
    @media (max-width: 600px) {
        .main-num { font-size: 42px; }
        .sup-card { font-size: 26px; padding: 15px 10px; }
        .main-card { padding: 20px 15px; }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. AI ENGINE CLASS - All Algorithms in One
# ==============================================================================

class TitanAI:
    """
    Multi-Algorithm AI Engine for 3-of-5 Lottery Prediction
    Algorithms: Frequency | Gap | Markov | Monte Carlo | Pattern | Hot/Cold | Ensemble
    """
    
    def __init__(self):
        # Algorithm weights for ensemble
        self.weights = {
            'frequency': 25,
            'gap': 20,
            'markov': 20,
            'monte_carlo': 15,
            'pattern': 12,
            'hot_cold': 8
        }
    
    def analyze(self, history, max_simulations=2000):
        """
        Main analysis: Run all algorithms and return ensemble prediction.
        Args:
            history: List of 5-digit strings (newest first)
            max_simulations: Monte Carlo iterations (optimized for speed)
        Returns:
            dict with main_3, support_4, stats_df, risk_info
        """
        if not history or len(history) < 15:
            return self._fallback()
        
        # Clean and validate data
        clean_data = self._clean_history(history)
        if len(clean_data) < 15:
            return self._fallback("Cần ít nhất 15 kỳ dữ liệu hợp lệ")
        
        # Run all analysis layers
        results = {}
        results['frequency'] = self._analyze_frequency(clean_data)
        results['gap'] = self._analyze_gap(clean_data)
        results['markov'] = self._analyze_markov(clean_data)
        results['monte_carlo'] = self._analyze_monte_carlo(clean_data, max_simulations)
        results['pattern'] = self._analyze_pattern(clean_data)
        results['hot_cold'] = self._analyze_hot_cold(clean_data)
        
        # Ensemble voting
        ensemble = self._ensemble_vote(results)
        
        # Build statistics DataFrame
        stats_df = self._build_stats_df(clean_data, results)
        
        # Risk assessment
        risk = self._calculate_risk(clean_data)
        
        return {
            'main_3': ensemble['main_3'],
            'support_4': ensemble['support_4'],
            'stats_df': stats_df,
            'risk': risk,
            'confidence': ensemble['confidence'],
            'logic': self._build_logic(results, ensemble)
        }
    
    def _clean_history(self, history):
        """Clean and validate history data."""
        cleaned = []
        for item in history:
            s = str(item).strip()
            # Extract exactly 5 digits
            match = re.search(r'\d{5}', s)
            if match:
                cleaned.append(match.group())
        return cleaned
    
    def _analyze_frequency(self, data):
        """1️⃣ Frequency Analysis: Count digit occurrences with recency weight."""
        weighted = Counter()
        n = len(data)
        
        for idx, num in enumerate(data):
            # Exponential weight: recent = 3.0, older = 1.0
            weight = 3.0 - 2.0 * (idx / max(n, 1))
            for d in num:
                weighted[d] += weight
        
        scores = {d: weighted.get(d, 0) for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        
        return {'scores': scores, 'top_3': top_3, 'method': 'Frequency'}
    
    def _analyze_gap(self, data):
        """2️⃣ Gap/Cycle Analysis: Numbers not seen recently are 'due'."""
        last_seen = {d: -1 for d in '0123456789'}
        
        for idx, num in enumerate(data):
            for d in num:
                if last_seen[d] == -1:  # First time seeing this digit
                    last_seen[d] = idx
        
        # Calculate gap scores: larger gap = higher score (up to limit)
        scores = {}
        for d in '0123456789':
            gap = last_seen[d] if last_seen[d] >= 0 else len(data)
            # Score peaks at gap ~15, then decreases (avoid extremely cold)
            if gap <= 15:
                scores[d] = gap * 2.5
            else:
                scores[d] = max(0, 37.5 - (gap - 15) * 0.5)
        
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        return {'scores': scores, 'top_3': top_3, 'method': 'Gap Analysis'}
    
    def _analyze_markov(self, data):
        """3️⃣ Markov Chain: Analyze digit transition probabilities."""
        if len(data) < 20:
            return {'scores': {d: 10 for d in '0123456789'}, 'top_3': ['1','5','9'], 'method': 'Markov'}
        
        # Build transition matrix: last_digit -> next_digits count
        transitions = defaultdict(Counter)
        
        for i in range(len(data) - 1):
            curr, next_num = data[i], data[i + 1]
            for pos in range(5):
                if pos < len(curr) and pos < len(next_num):
                    transitions[curr[pos]][next_num[pos]] += 1
        
        # Predict based on most recent number
        last_num = data[0] if data else '00000'
        next_prob = Counter()
        
        for pos, last_d in enumerate(last_num[:5]):
            if last_d in transitions and transitions[last_d]:
                total = sum(transitions[last_d].values())
                for next_d, count in transitions[last_d].items():
                    # Weight middle positions higher
                    pos_weight = 1.0 + 0.25 * (2 - abs(pos - 2))
                    next_prob[next_d] += (count / total) * pos_weight
        
        scores = {d: next_prob.get(d, 0) * 10 for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        
        # Fill if needed
        while len(top_3) < 3:
            for d in '0123456789':
                if d not in top_3:
                    top_3.append(d)
                    break
        
        return {'scores': scores, 'top_3': top_3, 'method': 'Markov Chain'}
    
    def _analyze_monte_carlo(self, data, n_simulations):
        """4️⃣ Monte Carlo: Weighted random simulation for probability estimation."""
        if len(data) < 20:
            return {'scores': {d: 10 for d in '0123456789'}, 'top_3': ['2','4','6'], 'method': 'Monte Carlo'}
        
        # Build weighted pool from recent data
        recent = data[:80] if len(data) >= 80 else data
        pool = []
        
        for idx, num in enumerate(recent):
            # Weight: recent = 4, older = 1
            weight = max(1, 4 - idx // 20)
            for d in num:
                pool.extend([d] * weight)
        
        if not pool:
            return {'scores': {d: 10 for d in '0123456789'}, 'top_3': ['0','1','2'], 'method': 'Monte Carlo'}
        
        # Run simulations (optimized count)
        sim_count = Counter()
        for _ in range(n_simulations):
            # Sample 3 digits with replacement
            sample = random.choices(pool, k=3)
            for d in sample:
                sim_count[d] += 1
        
        # Normalize to 0-100 scale
        total = sum(sim_count.values()) or 1
        scores = {d: sim_count.get(d, 0) / total * 100 for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        
        return {'scores': scores, 'top_3': top_3, 'method': 'Monte Carlo'}
    
    def _analyze_pattern(self, data):
        """5️⃣ Pattern Detection: Find repeating sequences and rhythms."""
        if len(data) < 25:
            return {'scores': {d: 10 for d in '0123456789'}, 'top_3': ['3','5','7'], 'method': 'Pattern'}
        
        recent = data[:40] if len(data) >= 40 else data
        candidates = Counter()
        
        # Pattern 1: Consecutive streaks (cầu bệt)
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            for i in range(len(seq) - 2):
                if seq[i] == seq[i+1] == seq[i+2]:
                    d = seq[i]
                    # Streak of 3: likely to continue (but not 4+)
                    streak = 3
                    while i + streak < len(seq) and seq[i + streak] == d:
                        streak += 1
                    if streak == 3:
                        candidates[d] += 4
                    elif streak >= 4:
                        candidates[d] -= 2  # Likely to break
        
        # Pattern 2: Rhythm-2 (X _ X _ X)
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            for i in range(len(seq) - 4):
                if seq[i] == seq[i+2] == seq[i+4] and seq[i] != seq[i+1]:
                    candidates[seq[i]] += 3
        
        # Pattern 3: Pair reversals (AB -> BA)
        for i in range(len(recent) - 1):
            a, b = recent[i], recent[i+1]
            if len(a) >= 2 and len(b) >= 2:
                if a[0:2] == b[1::-1]:  # AB -> BA
                    candidates[a[0]] += 2
                    candidates[a[1]] += 2
        
        # Fallback to frequency if no patterns found
        if not candidates:
            all_digits = ''.join(recent)
            freq = Counter(all_digits)
            for d, c in freq.most_common(3):
                candidates[d] += 3
        
        scores = {d: candidates.get(d, 0) * 2 for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        
        while len(top_3) < 3:
            for d in '0123456789':
                if d not in top_3:
                    top_3.append(d)
                    break
        
        return {'scores': scores, 'top_3': top_3, 'method': 'Pattern Detection'}
    
    def _analyze_hot_cold(self, data):
        """6️⃣ Hot/Cold Detection: Identify trending vs dormant numbers."""
        recent = data[:15] if len(data) >= 15 else data
        older = data[15:45] if len(data) >= 45 else data
        
        recent_count = Counter(''.join(recent))
        older_count = Counter(''.join(older)) if older else Counter()
        
        scores = {}
        for d in '0123456789':
            r = recent_count.get(d, 0)
            o = older_count.get(d, 0)
            
            if r >= 4:  # Hot: appearing frequently recently
                scores[d] = 25 + r * 3
            elif r == 0 and o >= 3:  # Cold but was hot: due to return
                scores[d] = 20 + o * 2
            elif r >= 2:  # Warm
                scores[d] = 15 + r * 2
            else:  # Neutral/Cold
                scores[d] = 10 + r
        
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        return {'scores': scores, 'top_3': top_3, 'method': 'Hot/Cold'}
    
    def _ensemble_vote(self, results):
        """7️⃣ Ensemble: Combine all algorithms with weighted voting."""
        votes = Counter()
        
        for algo_name, result in results.items():
            weight = self.weights.get(algo_name, 10)
            for d in result.get('top_3', []):
                votes[d] += weight
        
        # Get top 3
        main_3 = [d for d, _ in votes.most_common(3)]
        while len(main_3) < 3:
            for d in '0123456789':
                if d not in main_3:
                    main_3.append(d)
                    break
        
        # Support 4: next best
        remaining = [d for d, _ in votes.most_common(10) if d not in main_3]
        support_4 = remaining[:4]
        while len(support_4) < 4:
            for d in '0123456789':
                if d not in main_3 and d not in support_4:
                    support_4.append(d)
                    break
        
        # Confidence: based on vote agreement
        if votes:
            top_votes = [c for _, c in votes.most_common(3)]
            avg_vote = sum(top_votes) / 3
            confidence = min(95, 55 + avg_vote)
        else:
            confidence = 50
        
        return {'main_3': main_3, 'support_4': support_4, 'confidence': int(confidence)}
    
    def _build_stats_df(self, data, results):
        """Build comprehensive statistics DataFrame."""
        rows = []
        
        for d in '0123456789':
            row = {'Digit': d}
            
            # Frequency
            row['Frequency'] = results['frequency']['scores'].get(d, 0)
            
            # Gap
            row['Gap'] = results['gap']['scores'].get(d, 0)
            
            # Markov
            row['Markov'] = results['markov']['scores'].get(d, 0)
            
            # Monte Carlo
            row['Monte_Carlo'] = results['monte_carlo']['scores'].get(d, 0)
            
            # Pattern
            row['Pattern'] = results['pattern']['scores'].get(d, 0)
            
            # Hot/Cold
            row['Hot_Cold'] = results['hot_cold']['scores'].get(d, 0)
            
            # AI Score: weighted ensemble
            ai_score = (
                row['Frequency'] * 0.25 +
                row['Gap'] * 0.20 +
                row['Markov'] * 0.20 +
                row['Monte_Carlo'] * 0.15 +
                row['Pattern'] * 0.12 +
                row['Hot_Cold'] * 0.08
            )
            row['AI_Score'] = round(ai_score, 1)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df.sort_values('AI_Score', ascending=False).reset_index(drop=True)
    
    def _calculate_risk(self, data):
        """Calculate risk level based on data patterns."""
        if len(data) < 20:
            return {'score': 30, 'level': 'MEDIUM', 'reason': 'Dữ liệu ít'}
        
        all_digits = ''.join(data[:50])
        counts = Counter(all_digits)
        total = len(all_digits)
        
        score = 0
        reasons = []
        
        # Check entropy (randomness)
        entropy = 0
        for c in counts.values():
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
        
        if entropy < 2.8:
            score += 25
            reasons.append('Kết quả quá đều (có thể bị điều khiển)')
        elif entropy > 3.4:
            score += 15
            reasons.append('Biến động quá mạnh')
        
        # Check for abnormal streaks
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in data[:30]]
            max_streak = 1
            curr = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    curr += 1
                    max_streak = max(max_streak, curr)
                else:
                    curr = 1
            if max_streak >= 5:
                score += 30
                reasons.append(f'Cầu bệt {max_streak} kỳ ở vị trí {pos}')
                break
        
        score = min(100, score)
        level = 'HIGH' if score >= 50 else 'MEDIUM' if score >= 25 else 'OK'
        
        return {
            'score': score,
            'level': level,
            'reason': '; '.join(reasons) if reasons else 'Nhịp số tự nhiên'
        }
    
    def _build_logic(self, results, ensemble):
        """Build human-readable explanation."""
        parts = []
        
        # Top frequency digits
        freq_top = [d for d, _ in sorted(results['frequency']['scores'].items(), key=lambda x: -x[1])[:2]]
        if freq_top:
            parts.append(f"Tần suất: {','.join(freq_top)}")
        
        # Gap due numbers
        gap_top = [d for d, _ in sorted(results['gap']['scores'].items(), key=lambda x: -x[1])[:1]]
        if gap_top and results['gap']['scores'].get(gap_top[0], 0) > 25:
            parts.append(f"Đến kỳ: {gap_top[0]}")
        
        # Pattern detected
        if results['pattern']['method'] != 'Pattern':
            parts.append('Pattern phát hiện')
        
        # Agreement indicator
        if ensemble['confidence'] >= 75:
            parts.append('Đa số thuật toán đồng thuận')
        
        return ' | '.join(parts) if parts else 'Phân tích đa thuật toán'
    
    def _fallback(self, msg="Chưa đủ dữ liệu"):
        """Return safe fallback prediction."""
        return {
            'main_3': ['?', '?', '?'],
            'support_4': ['0', '0', '0', '0'],
            'stats_df': pd.DataFrame({
                'Digit': list('0123456789'),
                'Frequency': [0]*10,
                'Gap': [0]*10,
                'Markov': [0]*10,
                'Monte_Carlo': [0]*10,
                'AI_Score': [0]*10
            }),
            'risk': {'score': 0, 'level': 'LOW', 'reason': msg},
            'confidence': 0,
            'logic': msg
        }

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def check_win(main_3, result):
    """Check win condition: all 3 predicted digits must appear in 5-digit result."""
    if not main_3 or not result or len(result) != 5:
        return False
    pred_set = set(str(d) for d in main_3 if str(d).isdigit() and d != '?')
    result_set = set(result)
    return pred_set.issubset(result_set)

def format_num(n):
    """Format number for display."""
    try:
        return f"{float(n):.1f}" if isinstance(n, (int, float)) else str(n)
    except:
        return str(n)

# ==============================================================================
# 4. MAIN APPLICATION
# ==============================================================================

def main():
    # Initialize session state
    if 'db' not in st.session_state:
        st.session_state.db = []
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'ai' not in st.session_state:
        st.session_state.ai = TitanAI()
    
    # Header
    st.title("🎯 TITAN AI ENGINE")
    st.caption("3 Numbers from 5 Digits | Multi-Algorithm Prediction")
    
    # Sidebar: Quick Stats
    with st.sidebar:
        st.markdown("### 📊 Thống kê")
        st.metric("📦 Tổng kỳ", len(st.session_state.db))
        
        if st.session_state.result:
            st.metric("🎯 Độ tin cậy", f"{st.session_state.result['confidence']}%")
            risk = st.session_state.result['risk']
            st.metric("⚠️ Risk Score", f"{risk['score']}/100")
        
        st.markdown("---")
        st.markdown("### 💡 Hướng dẫn")
        st.markdown("""
        1. Nhập kết quả (5 số/dòng)
        2. Bấm "PHÂN TÍCH"
        3. Xem 3 số chính + 4 số lót
        4. Nhập kết quả thực tế để kiểm tra
        """)
        
        if st.button("🗑️ Xóa dữ liệu"):
            st.session_state.db = []
            st.session_state.result = None
            st.success("✅ Đã xóa!")
            time.sleep(0.5)
            st.rerun()
    
    # Main: Data Input
    st.markdown("### 📥 Nhập kết quả lịch sử")
    st.markdown("*Mỗi kỳ 1 dòng, 5 chữ số. Khuyến nghị: 50-500 kỳ*")
    
    raw_input = st.text_area(
        "📋 Dữ liệu:",
        height=150,
        placeholder="12345\n67890\n54321\n...",
        key="data_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        analyze_btn = st.button("🚀 PHÂN TÍCH AI", type="primary", use_container_width=True)
    with col2:
        if st.button("📋 Demo", use_container_width=True):
            demo = "\n".join([
                "87746", "56421", "69137", "00443", "04475",
                "64472", "16755", "58569", "62640", "99723",
                "33769", "14671", "92002", "65449", "26073",
                "93388", "31215", "51206", "41291", "24993"
            ] * 5)  # 100 periods demo
            st.session_state.data_input = demo
            st.rerun()
    with col3:
        if st.button("🔄 Làm mới", use_container_width=True):
            st.rerun()
    
    # Process analysis
    if analyze_btn and raw_input.strip():
        with st.spinner("🧠 Đang phân tích..."):
            # Extract 5-digit numbers
            numbers = re.findall(r'\d{5}', raw_input)
            
            if not numbers:
                st.error("❌ Không tìm thấy số 5 chữ số hợp lệ!")
            else:
                # Update database (deduplicate, newest first)
                existing = set(st.session_state.db)
                added = 0
                for n in numbers:
                    if n not in existing:
                        st.session_state.db.insert(0, n)
                        existing.add(n)
                        added += 1
                
                # Limit to 500 for performance
                if len(st.session_state.db) > 500:
                    st.session_state.db = st.session_state.db[:500]
                
                if added > 0:
                    st.success(f"✅ Đã thêm {added} số mới")
                else:
                    st.info("ℹ️ Dữ liệu đã có trong hệ thống")
                
                # Run AI analysis
                if len(st.session_state.db) >= 15:
                    st.session_state.result = st.session_state.ai.analyze(st.session_state.db)
                    st.rerun()
                else:
                    st.warning(f"⚠️ Cần ít nhất 15 kỳ (hiện có: {len(st.session_state.db)})")
    
    # Display Results
    if st.session_state.result:
        res = st.session_state.result
        
        # Risk Banner
        risk = res['risk']
        if risk['level'] == 'OK':
            status_class = 'status-ok'
            status_icon = '✅'
            action = 'CÓ THỂ ĐÁNH'
        elif risk['level'] == 'MEDIUM':
            status_class = 'status-warn'
            status_icon = '⚠️'
            action = 'THEO DÕI'
        else:
            status_class = 'status-stop'
            status_icon = '🛑'
            action = 'NÊN DỪNG'
        
        st.markdown(f"""
        <div class="status-banner {status_class}">
            {status_icon} {action} | Risk: {risk['score']}/100 | {risk['reason']}
        </div>
        """, unsafe_allow_html=True)
        
        # 3 Main Numbers - Large Cards
        st.markdown("### 🔮 3 SỐ CHÍNH")
        main_3 = res['main_3']
        
        col1, col2, col3 = st.columns(3)
        for i, (col, num) in enumerate(zip([col1, col2, col3], main_3)):
            col.markdown(f"""
            <div class="main-card">
                <div class="main-num">{num}</div>
                <div class="main-label">Số {i+1}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 4 Support Numbers
        st.markdown("### 🎲 4 SỐ LÓT")
        support_4 = res['support_4']
        
        sup_cols = st.columns(4)
        for col, num in zip(sup_cols, support_4):
            col.markdown(f'<div class="sup-card">{num}</div>', unsafe_allow_html=True)
        
        # Copy Code
        st.markdown("---")
        st.code(','.join(main_3 + support_4), language=None)
        st.caption("📋 Bấm vào code để copy dàn 7 số")
        
        # Logic Explanation
        if res['logic']:
            st.info(f"💡 **Logic:** {res['logic']}")
        
        # Statistics Table
        st.markdown("### 📊 Bảng Thống kê Chi tiết")
        
        if 'stats_df' in res and res['stats_df'] is not None:
            df = res['stats_df'].copy()
            
            # Format for display
            display_df = df[['Digit', 'Frequency', 'Gap', 'Markov', 'Monte_Carlo', 'AI_Score']].copy()
            display_df['Frequency'] = display_df['Frequency'].apply(lambda x: f"{x:.1f}")
            display_df['Gap'] = display_df['Gap'].apply(lambda x: f"{x:.1f}")
            display_df['Markov'] = display_df['Markov'].apply(lambda x: f"{x:.1f}")
            display_df['Monte_Carlo'] = display_df['Monte_Carlo'].apply(lambda x: f"{x:.1f}")
            display_df['AI_Score'] = display_df['AI_Score'].apply(lambda x: f"{x:.1f}")
            
            # Highlight top 3
            def highlight_top(s):
                return ['background: rgba(255,88,88,0.15); font-weight: bold' if i < 3 else '' for i in range(len(s))]
            
            st.dataframe(
                display_df.style.apply(highlight_top, subset=['AI_Score']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Chưa có dữ liệu thống kê")
        
        # Result Verification
        st.markdown("---")
        st.markdown("### ✅ Xác nhận kết quả thực tế")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            actual = st.text_input("Kết quả kỳ này (5 số):", key="actual_result", placeholder="12864")
        with col2:
            verify_btn = st.button("✅ Kiểm tra", type="primary", use_container_width=True)
        
        if verify_btn and actual and len(actual) == 5 and actual.isdigit():
            is_win = check_win(main_3, actual)
            
            if is_win:
                st.success(f"🎉 **TRÚNG!** 3 số {main_3} có trong {actual}")
            else:
                missing = set(main_3) - set(actual)
                st.warning(f"❌ **Trượt!** Thiếu: {', '.join(missing)}")
            
            st.markdown("*Lưu ý: Tool hỗ trợ phân tích, không đảm bảo 100% chính xác*")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8b949e; padding: 20px; font-size: 12px;">
        🎯 TITAN AI ENGINE | Multi-Algorithm Prediction System<br>
        ⚠️ Hỗ trợ phân tích - Không đảm bảo trúng - Chơi có trách nhiệm
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 5. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()